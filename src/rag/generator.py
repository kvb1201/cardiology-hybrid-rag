"""
generator.py
------------
Deterministic, fixed downstream RAG generator for the Cardiology-Chat research study.

Research Question:
  "Do conventional retrieval metrics reliably predict automated downstream
   answer quality in cardiovascular RAG?"

Methodological Controls:
  - The experiment varies ONLY retrieval.
  - Across all 5 retrievers (BM25, Dense, LSH, BD-RRF, BDL-RRF) for the same query and K,
    generation uses the exact same:
      * generator model
      * system prompt
      * user prompt template
      * temperature (fixed at 0)
      * max output tokens
      * context formatting
      * decoding configuration
  - The generator prompt NEVER contains:
      * qrels or relevance grades
      * retriever identity
      * K cutoff value
      * retrieval metrics
      * reference evidence
      * evaluation instructions
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class DailyQuotaExceededError(RuntimeError):
    """Raised when the LLM provider daily request or token quota has been exhausted."""
    pass


# ── Default Configuration Paths ──────────────────────────────────────────────
DEFAULT_CONFIG_PATH = ROOT / "experiments" / "configs" / "generation.yaml"
DEFAULT_RUNS_PATH   = ROOT / "experiments" / "results" / "stage3_retrieval_runs.json"
DEFAULT_OUTPUT_PATH = ROOT / "experiments" / "results" / "stage3_generated_answers.jsonl"

# ── Versioned Prompt Templates ───────────────────────────────────────────────
SYSTEM_PROMPTS: Dict[str, str] = {
    "v1.0": (
        "You are a clinical information question-answering assistant specializing in cardiovascular medicine.\n"
        "Your task is to provide an accurate, concise, and evidence-grounded answer to the user's clinical question based strictly on the provided evidence passages.\n\n"
        "Guidelines:\n"
        "1. Base your answer solely on the provided evidence passages.\n"
        "2. Do not fabricate, extrapolate, or invent information not directly supported by the evidence.\n"
        "3. If the evidence passages do not contain sufficient information to answer the question completely, clearly state the limitations and uncertainty based on the available evidence.\n"
        "4. Do not cite external sources or mention retrieval systems, search ranks, scores, or evaluation processes.\n"
        "5. Maintain an objective, professional, and clear clinical communication tone."
    ),
}

GENERATION_PROMPTS: Dict[str, str] = {
    "v1.0": (
        "Evidence Passages:\n"
        "{context}\n\n"
        "Question: {query}\n\n"
        "Please provide a concise, evidence-grounded answer:"
    ),
}


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

def load_generation_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and validate the centralized generation YAML configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Generation configuration file not found at: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "generation" not in data:
        raise ValueError("Invalid generation.yaml: root key 'generation' required.")

    gen_cfg = data["generation"]
    required_fields = [
        "provider",
        "model",
        "temperature",
        "system_prompt_version",
        "generation_prompt_version",
    ]
    for field in required_fields:
        if field not in gen_cfg:
            raise ValueError(f"Missing required generation configuration field: '{field}'")

    if "max_tokens" not in gen_cfg and "max_output_tokens" not in gen_cfg:
        raise ValueError("Missing required field: 'max_output_tokens' or 'max_tokens'")

    # Normalize max_tokens
    if "max_tokens" not in gen_cfg:
        gen_cfg["max_tokens"] = gen_cfg["max_output_tokens"]

    if gen_cfg["temperature"] != 0:
        raise ValueError(
            f"Experimental protocol requires temperature=0 for determinism; got {gen_cfg['temperature']}"
        )

    return gen_cfg


# ---------------------------------------------------------------------------
# Context Formatter
# ---------------------------------------------------------------------------

def format_evidence_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved evidence passages into a neutral, uniform context block.

    Rules:
      - Passage ordering matches retrieval ranking strictly (1..K).
      - Source document title is shown as metadata.
      - Chunk IDs, scores, ranks, and retriever identities are NEVER exposed.
      - Identical format across all retrieval systems and all K values.

    Parameters
    ----------
    results:
        List of result dictionaries from the reconstructed retrieval run.

    Returns
    -------
    str
        Formatted context block.
    """
    if not results:
        return "[No evidence passages retrieved for this query.]"

    blocks: List[str] = []
    for idx, item in enumerate(results, start=1):
        meta = item.get("metadata", {})
        title = meta.get("title") or item.get("title") or item.get("doc_id", "Cardiovascular Evidence Document")
        text = item.get("text", "").strip()

        block = f"[Evidence {idx}]\nSource: {title}\n{text}"
        blocks.append(block)

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_generator_prompt(
    query_text: str,
    context: str,
    system_version: str = "v1.0",
    prompt_version: str = "v1.0",
) -> Dict[str, str]:
    """Construct versioned system and user prompts for the generator.

    Guarantees:
      - Contains only the query and formatted context.
      - Zero retrieval metadata, qrels, or ground-truth evidence.

    Returns
    -------
    Dict[str, str]
        {"system_prompt": str, "user_prompt": str}
    """
    if system_version not in SYSTEM_PROMPTS:
        raise KeyError(f"Unknown system prompt version: '{system_version}'")
    if prompt_version not in GENERATION_PROMPTS:
        raise KeyError(f"Unknown generation prompt version: '{prompt_version}'")

    system_prompt = SYSTEM_PROMPTS[system_version]
    user_prompt = GENERATION_PROMPTS[prompt_version].format(
        context=context,
        query=query_text.strip(),
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


# ---------------------------------------------------------------------------
# Resumable Output Management
# ---------------------------------------------------------------------------

def load_completed_runs(output_file: str | Path = DEFAULT_OUTPUT_PATH) -> Set[Tuple[str, str, int]]:
    """Read existing generated answers JSONL file and return completed (query_id, retriever, k) keys.

    Only records with status == 'success' are treated as completed.
    """
    path = Path(output_file)
    if not path.exists():
        return set()

    completed: Set[Tuple[str, str, int]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "success":
                    key = (rec["query_id"], rec["retriever"], int(rec["k"]))
                    completed.add(key)
            except Exception:  # noqa: BLE001
                continue
    return completed


def append_generation_record(
    record: Dict[str, Any],
    output_file: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Atomically append a single generation record to the output JSONL file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# LLM Caller Interface
# ---------------------------------------------------------------------------

class LLMCaller:
    """Interface for dispatching generation calls to configured provider."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.provider = config.get("provider", "unspecified").lower()
        self.model = config.get("model", "unspecified")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 512)
        self.top_p = float(config.get("top_p", 1.0))
        self.timeout_seconds = config.get("timeout_seconds", 30)

        retry_cfg = config.get("retry", {})
        self.max_retries = retry_cfg.get("max_retries", 3)
        self.initial_delay = retry_cfg.get("initial_delay_seconds", 2.0)
        self.backoff_factor = retry_cfg.get("backoff_factor", 2.0)
        self.max_delay = retry_cfg.get("max_delay_seconds", 10.0)
        self.last_usage: Dict[str, Any] = {}
        self.total_retries: int = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the provider LLM with retry and rate-limit handling.

        Raises
        ------
        RuntimeError
            If provider is unspecified or required API credentials are missing.
        """
        if self.provider == "unspecified" or not self.provider:
            raise RuntimeError(
                "Generator model/provider is unspecified and requires explicit project configuration."
            )

        if self.provider == "mock":
            # Deterministic offline mock for testing without API keys
            return (
                f"[Mock Answer from {self.model}]: Evidence-grounded response "
                f"synthesizing the provided clinical material."
            )

        # Dispatch real providers
        last_exception: Optional[Exception] = None
        delay = self.initial_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "openai":
                    return self._call_openai(system_prompt, user_prompt)
                elif self.provider == "gemini":
                    return self._call_gemini(system_prompt, user_prompt)
                elif self.provider == "anthropic":
                    return self._call_anthropic(system_prompt, user_prompt)
                else:
                    raise ValueError(f"Unsupported provider: '{self.provider}'")
            except DailyQuotaExceededError:
                # Quota safety: Never retry daily quota exhaustion
                raise
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if attempt < self.max_retries:
                    self.total_retries += 1
                logger.warning(
                    f"LLM call failed (attempt {attempt}/{self.max_retries}): {exc}. Retrying in {delay:.1f}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_exception}") from last_exception

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY environment variable not set.")
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": self.top_p,
            },
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        if resp.status_code == 429:
            err_data: Dict[str, Any] = {}
            try:
                err_data = resp.json().get("error", {})
            except Exception:  # noqa: BLE001
                pass
            err_str = str(err_data).lower()
            err_msg = str(err_data.get("message", "")).lower()
            if (
                "generaterequestsperday" in err_str
                or "free_tier_requests" in err_str
                or "please retry in" in err_msg
                or "quota exceeded" in err_msg
                or "daily request limit" in err_msg
            ):
                raise DailyQuotaExceededError(
                    f"Gemini daily quota exhausted: {err_data.get('message', 'Daily quota limit reached')}"
                )
        resp.raise_for_status()
        data = resp.json()
        self.last_usage = data.get("usageMetadata", {})
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p]
        if not text_parts:
            finish_reason = candidates[0].get("finishReason", "UNKNOWN")
            raise RuntimeError(f"Gemini API returned no text parts (finishReason={finish_reason})")
        return "".join(text_parts).strip()

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
        import requests
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# RAG Generator Engine
# ---------------------------------------------------------------------------

class RAGGenerator:
    """Manages prompt creation, context formatting, LLM dispatch, and resumable execution."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config = config if config is not None else load_generation_config(config_path)
        self.caller = LLMCaller(self.config)
        self.sys_version = self.config.get("system_prompt_version", "v1.0")
        self.prompt_version = self.config.get("generation_prompt_version", "v1.0")

    def execute_condition(
        self,
        run_record: Dict[str, Any],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute generation for a single retrieval condition (query x retriever x K).

        Parameters
        ----------
        run_record:
            Run dictionary from stage3_retrieval_runs.json.
        dry_run:
            If True, builds prompt and validates schema without calling LLM.

        Returns
        -------
        Dict[str, Any]
            Structured generation output record conforming to schema.
        """
        qid = run_record["query_id"]
        qtext = run_record["query_text"]
        topic = run_record.get("topic", "")
        qtype = run_record.get("query_type", "")
        ret_id = run_record["retriever"]
        k = run_record["k"]
        results = run_record.get("results", [])
        retrieved_chunk_ids = [r["chunk_id"] for r in results]

        # Step 1: Format context
        context_str = format_evidence_context(results)

        # Step 2: Build prompts
        prompts = build_generator_prompt(
            query_text=qtext,
            context=context_str,
            system_version=self.sys_version,
            prompt_version=self.prompt_version,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        if dry_run:
            return {
                "query_id": qid,
                "query_text": qtext,
                "topic": topic,
                "query_type": qtype,
                "retriever": ret_id,
                "k": k,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "generated_answer": "[Dry Run: Context and prompts verified; no API call made.]",
                "model": self.config.get("model", "unspecified"),
                "temperature": self.config.get("temperature", 0.0),
                "prompt_version": self.prompt_version,
                "generation_timestamp": timestamp,
                "status": "dry_run",
                "prompts_preview": prompts,
            }

        # Step 3: Call LLM
        try:
            answer = self.caller.generate(
                system_prompt=prompts["system_prompt"],
                user_prompt=prompts["user_prompt"],
            )
            rec = {
                "query_id": qid,
                "query_text": qtext,
                "topic": topic,
                "query_type": qtype,
                "retriever": ret_id,
                "k": k,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "generated_answer": answer,
                "model": self.config.get("model", "unspecified"),
                "temperature": self.config.get("temperature", 0.0),
                "prompt_version": self.prompt_version,
                "generation_timestamp": timestamp,
                "status": "success",
            }
            if hasattr(self.caller, "last_usage") and self.caller.last_usage:
                rec["usage_metadata"] = self.caller.last_usage
            return rec
        except DailyQuotaExceededError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Generation failed for {qid} / {ret_id} / K={k}: {exc}")
            return {
                "query_id": qid,
                "query_text": qtext,
                "topic": topic,
                "query_type": qtype,
                "retriever": ret_id,
                "k": k,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "model": self.config.get("model", "unspecified"),
                "temperature": self.config.get("temperature", 0.0),
                "prompt_version": self.prompt_version,
                "generation_timestamp": timestamp,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }


# ---------------------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------------------

def run_generation_experiment(
    generator: RAGGenerator,
    runs_path: str | Path = DEFAULT_RUNS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    dry_run: bool = False,
    limit: Optional[int] = None,
    filter_keys: Optional[List[Tuple[str, str, int]]] = None,
) -> Dict[str, Any]:
    """Execute generation across experiment conditions with resumability.

    Parameters
    ----------
    generator:
        Configured RAGGenerator instance.
    runs_path:
        Path to stage3_retrieval_runs.json.
    output_path:
        Path to output stage3_generated_answers.jsonl.
    dry_run:
        If True, validates without calling LLM.
    limit:
        Optional limit on the number of conditions (e.g. 3 for pilot, 485 for Batch 1).
    filter_keys:
        Optional list of (query_id, retriever, k) to run specifically.

    Returns
    -------
    Dict[str, Any]
        Summary statistics of the generation run.
    """
    with open(runs_path, "r", encoding="utf-8") as f:
        runs_data = json.load(f)

    all_runs: List[Dict[str, Any]] = runs_data.get("runs", [])
    if len(all_runs) != 975:
        raise ValueError(f"Expected 975 runs in {runs_path}, found {len(all_runs)}")

    # Check already completed runs for resumability
    completed_keys = load_completed_runs(output_path) if not dry_run else set()
    previously_completed_count = len(completed_keys)

    # Filter runs
    selected_runs: List[Dict[str, Any]] = []
    for r in all_runs:
        key = (r["query_id"], r["retriever"], int(r["k"]))
        if filter_keys is not None and key not in filter_keys:
            continue
        if key in completed_keys:
            continue
        selected_runs.append(r)

    if limit is not None:
        selected_runs = selected_runs[:limit]

    results: List[Dict[str, Any]] = []
    success_count = 0
    failed_count = 0
    quota_stopped = False
    quota_error_msg = ""
    skipped_count = len(all_runs) - len(selected_runs)

    for idx, run_rec in enumerate(selected_runs, 1):
        try:
            t_cond = time.perf_counter()
            rec = generator.execute_condition(run_rec, dry_run=dry_run)
            dur_cond = time.perf_counter() - t_cond
            results.append(rec)
            if not dry_run:
                append_generation_record(rec, output_path)
                if rec.get("status") == "success":
                    success_count += 1
                    status_lbl = "SUCCESS"
                else:
                    failed_count += 1
                    status_lbl = f"FAILED ({rec.get('error_type')})"

                qid = run_rec["query_id"]
                ret = run_rec["retriever"]
                k_val = run_rec["k"]
                if idx % 10 == 0 or idx == 1 or idx == len(selected_runs):
                    print(
                        f"[{idx:3d}/{len(selected_runs):3d}] {qid} / {ret:18s} / K={k_val:2d} -> "
                        f"{status_lbl} ({dur_cond:.1f}s) | Progress: {success_count} new successes",
                        flush=True,
                    )
        except DailyQuotaExceededError as q_err:
            logger.error(f"Daily quota reached. Stopping batch safely: {q_err}")
            print(f"\n[QUOTA STOP] Daily quota reached: {q_err}. Stopping batch safely.", flush=True)
            quota_stopped = True
            quota_error_msg = str(q_err)
            break

    total_completed = len(load_completed_runs(output_path)) if not dry_run else success_count

    return {
        "total_conditions": len(all_runs),
        "selected_conditions": len(selected_runs),
        "previously_completed": previously_completed_count,
        "skipped_already_completed": skipped_count,
        "new_success_count": success_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "total_completed": total_completed,
        "remaining_conditions": len(all_runs) - total_completed,
        "quota_stopped": quota_stopped,
        "quota_error_msg": quota_error_msg,
        "dry_run": dry_run,
        "results": results,
    }
