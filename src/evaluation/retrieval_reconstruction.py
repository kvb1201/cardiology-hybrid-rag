"""
retrieval_reconstruction.py
---------------------------
Deterministic retrieval run reconstruction and evaluation for Stage 3.

This module reconstructs the ranked retrieval outputs for all 65 evaluation queries across
5 frozen retrieval systems at cutoffs K = 5, 10, 20:
  1. BM25 (Stage 2A baseline)
  2. Dense (Stage 2D FAISS baseline)
  3. LSH (Stage 2C MinHash-LSH baseline)
  4. BM25 + Dense RRF (SysA_BD, rrf_k=60, candidate_depth=20)
  5. BM25 + Dense + LSH RRF (SysB_BDL, rrf_k=60, candidate_depth=20)

Total reconstructed runs: 65 queries * 5 retrievers * 3 K values = 975 runs.

CRITICAL CONSTRAINTS:
- Uses ONLY frozen retrievers, frozen indexes, frozen corpus, and frozen qrels.
- Ranking logic and parameters are NEVER modified.
- Evaluation metrics are computed strictly using the frozen `retrieval_metrics.py`.
- Reconstructed aggregate metrics are validated against frozen Stage 2 baseline results.
- No LLM generation or API calls are performed.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lsh import LSHRetriever
from src.evaluation.retrieval_metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_QUERIES_PATH = Path("data/evaluation/queries.csv")
DEFAULT_QRELS_PATH   = Path("data/evaluation/qrels.csv")
DEFAULT_CHUNKS_PATH  = Path("data/chunks/chunks.jsonl")

INDEX_BM25_DIR = Path("indexes/bm25")
INDEX_DENSE_DIR = Path("indexes/faiss")
INDEX_LSH_DIR   = Path("indexes/lsh")

DEFAULT_RUNS_OUTPUT_PATH    = Path("experiments/results/stage3_retrieval_runs.json")
DEFAULT_METRICS_OUTPUT_PATH = Path("experiments/results/stage3_retrieval_metrics.json")

# ── System identifiers and display names ──────────────────────────────────────
RETRIEVER_ORDER = [
    "bm25",
    "dense",
    "lsh",
    "bm25_dense_rrf",
    "bm25_dense_lsh_rrf",
]

RETRIEVER_DISPLAY_NAMES = {
    "bm25": "BM25",
    "dense": "Dense",
    "lsh": "LSH",
    "bm25_dense_rrf": "SysA_BD (BM25 + Dense RRF)",
    "bm25_dense_lsh_rrf": "SysB_BDL (BM25 + Dense + LSH RRF)",
}

# Mapping to Stage 2 result keys in stage2e_hybrid_rrf.json
STAGE2_SYSTEM_KEY_MAP = {
    "bm25": "BM25",
    "dense": "Dense",
    "lsh": "LSH",
    "bm25_dense_rrf": "SysA_BD",
    "bm25_dense_lsh_rrf": "SysB_BDL",
}

# ── Stage 2 frozen baseline metrics for verification ─────────────────────────
STAGE2_BASELINE_METRICS = {
    "bm25": {
        "NDCG@5": 0.5389,
        "NDCG@10": 0.5188,
        "NDCG@20": 0.5514,
        "MRR": 0.7452,
        "MAP": 0.3783,
    },
    "dense": {
        "NDCG@5": 0.4095,
        "NDCG@10": 0.3781,
        "NDCG@20": 0.4175,
        "MRR": 0.5928,
        "MAP": 0.2478,
    },
    "lsh": {
        "NDCG@5": 0.0814,
        "NDCG@10": 0.0739,
        "NDCG@20": 0.0738,
        "MRR": 0.1750,
        "MAP": 0.0386,
    },
    "bm25_dense_rrf": {
        "NDCG@5": 0.5309,
        "NDCG@10": 0.5097,
        "NDCG@20": 0.5395,
        "MRR": 0.7560,
        "MAP": 0.3610,
    },
    "bm25_dense_lsh_rrf": {
        "NDCG@5": 0.5301,
        "NDCG@10": 0.5018,
        "NDCG@20": 0.5349,
        "MRR": 0.7459,
        "MAP": 0.3544,
    },
}

K_VALUES = [5, 10, 20]
MAX_EVAL_K = 20
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_DEPTH = 20


# ---------------------------------------------------------------------------
# Retriever Loader
# ---------------------------------------------------------------------------

def load_all_retrievers(
    bm25_dir: str | Path = INDEX_BM25_DIR,
    dense_dir: str | Path = INDEX_DENSE_DIR,
    lsh_dir: str | Path = INDEX_LSH_DIR,
    rrf_k: int = DEFAULT_RRF_K,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> Dict[str, BaseRetriever]:
    """Load and initialize all 5 frozen retrievers from their saved indexes.

    Parameters
    ----------
    bm25_dir:
        Directory containing bm25_index.pkl.
    dense_dir:
        Directory containing dense_index.faiss.
    lsh_dir:
        Directory containing lsh_index.pkl.
    rrf_k:
        RRF smoothing constant (fixed at 60).
    candidate_depth:
        Candidate pool depth for RRF fusion (fixed at 20).

    Returns
    -------
    Dict[str, BaseRetriever]
        Mapping from retriever ID to loaded retriever instance.
    """
    bm25 = BM25Retriever(index_path=Path(bm25_dir))
    bm25.load()

    dense = DenseRetriever(index_path=Path(dense_dir))
    dense.load()

    lsh = LSHRetriever(index_path=Path(lsh_dir))
    lsh.load()

    sys_a = HybridRetriever(
        retrievers=[bm25, dense],
        rrf_k=rrf_k,
        candidate_depth=candidate_depth,
    )

    sys_b = HybridRetriever(
        retrievers=[bm25, dense, lsh],
        rrf_k=rrf_k,
        candidate_depth=candidate_depth,
    )

    return {
        "bm25": bm25,
        "dense": dense,
        "lsh": lsh,
        "bm25_dense_rrf": sys_a,
        "bm25_dense_lsh_rrf": sys_b,
    }


# ---------------------------------------------------------------------------
# Qrels Loader
# ---------------------------------------------------------------------------

def load_qrels_data(
    qrels_path: str | Path = DEFAULT_QRELS_PATH,
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    """Load qrels into binary relevant sets and graded relevance dictionaries.

    Relevance scale:
      0 = not relevant
      1 = partially relevant (considered relevant for binary metrics threshold >= 1)
      2 = highly relevant

    Returns
    -------
    Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]]]
        (qrels_binary, qrels_graded)
    """
    path = Path(qrels_path)
    if not path.exists():
        raise FileNotFoundError(f"Qrels file not found at: {path}")

    qrels_binary: Dict[str, Set[str]] = defaultdict(set)
    qrels_graded: Dict[str, Dict[str, int]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row["query_id"]
            cid = row["chunk_id"]
            rel = int(row["relevance"])
            qrels_graded[qid][cid] = rel
            if rel >= 1:
                qrels_binary[qid].add(cid)

    return dict(qrels_binary), dict(qrels_graded)


# ---------------------------------------------------------------------------
# Retrieval Run Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_retrieval_runs(
    retrievers: Optional[Dict[str, BaseRetriever]] = None,
    queries_path: str | Path = DEFAULT_QUERIES_PATH,
    k_values: List[int] = K_VALUES,
) -> Dict[str, Any]:
    """Reconstruct all 975 retrieval runs deterministically.

    For each of the 65 queries, retrieves top-20 results from each of the 5
    retrievers, then slices to K in [5, 10, 20].

    Parameters
    ----------
    retrievers:
        Optional pre-loaded dictionary of retrievers. If None, loads them from disk.
    queries_path:
        Path to queries.csv.
    k_values:
        List of cutoff values K. Default: [5, 10, 20].

    Returns
    -------
    Dict[str, Any]
        Dictionary containing metadata and all 975 runs.
    """
    queries_file = Path(queries_path)
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found at: {queries_file}")

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    if len(queries) != 65:
        raise ValueError(f"Expected 65 queries, got {len(queries)}")

    if retrievers is None:
        retrievers = load_all_retrievers()

    max_k = max(k_values)
    runs: List[Dict[str, Any]] = []

    for ret_id in RETRIEVER_ORDER:
        retriever = retrievers[ret_id]
        display_name = RETRIEVER_DISPLAY_NAMES[ret_id]

        for q in queries:
            qid = q["query_id"]
            qtext = q["query_text"]
            topic = q.get("topic", "")
            qtype = q.get("query_type", "")

            # Execute search with max_k
            results_pool = retriever.search(qtext, top_k=max_k)

            # Build runs for each requested K
            for k in k_values:
                sliced_results = results_pool[:k]

                result_dicts: List[Dict[str, Any]] = []
                for rank_idx, r in enumerate(sliced_results, start=1):
                    # Ensure deterministic ranks 1..K
                    result_dicts.append({
                        "rank": rank_idx,
                        "chunk_id": r.chunk_id,
                        "score": float(r.score),
                        "text": r.text,
                        "doc_id": r.doc_id,
                        "metadata": dict(r.metadata),
                    })

                run_entry = {
                    "query_id": qid,
                    "query_text": qtext,
                    "topic": topic,
                    "query_type": qtype,
                    "retriever": ret_id,
                    "retriever_name": display_name,
                    "k": k,
                    "n_results": len(result_dicts),
                    "results": result_dicts,
                }
                runs.append(run_entry)

    expected_total_runs = len(queries) * len(RETRIEVER_ORDER) * len(k_values)
    if len(runs) != expected_total_runs:
        raise ValueError(f"Expected {expected_total_runs} runs, got {len(runs)}")

    return {
        "metadata": {
            "schema_version": "1.0",
            "description": "Deterministic retrieval runs for Stage 3 downstream RAG generation and evaluation.",
            "total_runs": len(runs),
            "n_queries": len(queries),
            "retrievers": RETRIEVER_ORDER,
            "k_values": k_values,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "queries_path": str(queries_file),
                "chunks_path": str(DEFAULT_CHUNKS_PATH),
                "bm25_index": str(INDEX_BM25_DIR / "bm25_index.pkl"),
                "dense_index": str(INDEX_DENSE_DIR / "dense_index.faiss"),
                "lsh_index": str(INDEX_LSH_DIR / "lsh_index.pkl"),
            },
        },
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Retrieval Metrics Computation & Validation
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    runs_data: Dict[str, Any],
    qrels_path: str | Path = DEFAULT_QRELS_PATH,
) -> Dict[str, Any]:
    """Compute retrieval metrics for every (query, retriever, K) and aggregate across queries.

    Computes:
      - Precision@K
      - Recall@K
      - NDCG@K
      - Reciprocal Rank (RR) / MRR
      - Average Precision (AP) / MAP

    Compares reconstructed aggregates to the frozen Stage 2 baseline results.

    Parameters
    ----------
    runs_data:
        Output of :func:`reconstruct_retrieval_runs`.
    qrels_path:
        Path to qrels.csv.

    Returns
    -------
    Dict[str, Any]
        Dictionary with per-query metrics, aggregate metrics, and Stage 2 validation results.
    """
    qrels_binary, qrels_graded = load_qrels_data(qrels_path)
    runs = runs_data.get("runs", [])

    per_query_metrics: List[Dict[str, Any]] = []

    # Organizers for computing aggregates
    # Key: (retriever, k) -> list of metric values
    agg_p: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    agg_r: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    agg_n: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    agg_rr: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    agg_ap: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    # For full top-20 MRR and MAP matching Stage 2 protocol:
    # Key: retriever -> (list of retrieved ID lists, list of relevant ID sets)
    ret_pool_20: Dict[str, List[List[str]]] = defaultdict(list)
    rel_pool_20: Dict[str, List[Set[str]]] = defaultdict(list)

    for run in runs:
        qid = run["query_id"]
        ret_id = run["retriever"]
        k = run["k"]
        cids = [r["chunk_id"] for r in run["results"]]

        rel_set = qrels_binary.get(qid, set())
        rel_grades = qrels_graded.get(qid, {})

        p_k = precision_at_k(cids, rel_set, k)
        r_k = recall_at_k(cids, rel_set, k)
        n_k = ndcg_at_k(cids, rel_set, k, graded_relevance=rel_grades)
        rr_k = reciprocal_rank(cids, rel_set)
        ap_k = average_precision(cids, rel_set)

        per_query_metrics.append({
            "query_id": qid,
            "retriever": ret_id,
            "k": k,
            "precision_at_k": p_k,
            "recall_at_k": r_k,
            "ndcg_at_k": n_k,
            "reciprocal_rank": rr_k,
            "average_precision": ap_k,
            "n_retrieved": len(cids),
            "n_relevant_total": len(rel_set),
            "hits_at_k": sum(1 for cid in cids if cid in rel_set),
        })

        agg_p[(ret_id, k)].append(p_k)
        agg_r[(ret_id, k)].append(r_k)
        agg_n[(ret_id, k)].append(n_k)
        agg_rr[(ret_id, k)].append(rr_k)
        agg_ap[(ret_id, k)].append(ap_k)

        if k == MAX_EVAL_K:
            ret_pool_20[ret_id].append(cids)
            rel_pool_20[ret_id].append(rel_set)

    # Compute aggregates
    aggregates_by_system: Dict[str, Dict[str, Any]] = {}
    validation_results: Dict[str, Dict[str, Any]] = {}

    for ret_id in RETRIEVER_ORDER:
        sys_agg: Dict[str, Any] = {
            "retriever": ret_id,
            "display_name": RETRIEVER_DISPLAY_NAMES[ret_id],
        }

        for k in K_VALUES:
            sys_agg[f"P@{k}"] = round(sum(agg_p[(ret_id, k)]) / len(agg_p[(ret_id, k)]), 4)
            sys_agg[f"R@{k}"] = round(sum(agg_r[(ret_id, k)]) / len(agg_r[(ret_id, k)]), 4)
            sys_agg[f"NDCG@{k}"] = round(sum(agg_n[(ret_id, k)]) / len(agg_n[(ret_id, k)]), 4)
            sys_agg[f"RR@{k}"] = round(sum(agg_rr[(ret_id, k)]) / len(agg_rr[(ret_id, k)]), 4)
            sys_agg[f"AP@{k}"] = round(sum(agg_ap[(ret_id, k)]) / len(agg_ap[(ret_id, k)]), 4)

        # Standard Stage 2 IR protocol: MRR and MAP evaluated on top-20 pool
        mrr_20 = round(mean_reciprocal_rank(ret_pool_20[ret_id], rel_pool_20[ret_id]), 4)
        map_20 = round(mean_average_precision(ret_pool_20[ret_id], rel_pool_20[ret_id]), 4)
        sys_agg["MRR"] = mrr_20
        sys_agg["MAP"] = map_20

        aggregates_by_system[ret_id] = sys_agg

        # Validate against Stage 2 baseline
        baseline = STAGE2_BASELINE_METRICS[ret_id]
        val_entry: Dict[str, Any] = {}
        all_passed = True

        for metric_name, expected_val in baseline.items():
            reconstructed_val = sys_agg.get(metric_name)
            delta = abs(reconstructed_val - expected_val) if reconstructed_val is not None else 999.0
            # Allow tolerance for float rounding (0.001)
            passed = delta <= 0.0015
            if not passed:
                all_passed = False
            val_entry[metric_name] = {
                "expected": expected_val,
                "reconstructed": reconstructed_val,
                "delta": round(delta, 5),
                "passed": passed,
            }

        val_entry["status"] = "PASSED" if all_passed else "FAILED"
        validation_results[ret_id] = val_entry

    return {
        "metadata": {
            "schema_version": "1.0",
            "description": "Evaluation metrics for reconstructed retrieval runs and Stage 2 reproduction check.",
            "n_queries": 65,
            "primary_metric": "NDCG@10",
            "k_values": K_VALUES,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "stage2_validation": {
            "all_systems_passed": all(
                v["status"] == "PASSED" for v in validation_results.values()
            ),
            "systems": validation_results,
        },
        "aggregate_metrics": aggregates_by_system,
        "per_query_metrics": per_query_metrics,
    }


# ---------------------------------------------------------------------------
# Save and Load Helpers
# ---------------------------------------------------------------------------

def save_retrieval_runs(
    runs_data: Dict[str, Any],
    output_path: str | Path = DEFAULT_RUNS_OUTPUT_PATH,
) -> Path:
    """Save reconstructed retrieval runs to JSON deterministically."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(runs_data, f, indent=2, ensure_ascii=False)
    return out


def save_retrieval_metrics(
    metrics_data: Dict[str, Any],
    output_path: str | Path = DEFAULT_METRICS_OUTPUT_PATH,
) -> Path:
    """Save retrieval metrics to JSON deterministically."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    return out


class RetrievalRunsCollection:
    """Reader and lookup wrapper for stage3_retrieval_runs.json."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.metadata = data.get("metadata", {})
        self.runs: List[Dict[str, Any]] = data.get("runs", [])
        self._lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {
            (r["query_id"], r["retriever"], r["k"]): r
            for r in self.runs
        }

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.runs)

    def get_run(self, query_id: str, retriever: str, k: int) -> Optional[Dict[str, Any]]:
        """Look up a specific run by (query_id, retriever, k)."""
        return self._lookup.get((query_id, retriever, k))

    def get_runs_for_query(self, query_id: str) -> List[Dict[str, Any]]:
        """Get all 15 runs for a given query."""
        return [r for r in self.runs if r["query_id"] == query_id]

    def get_runs_for_retriever(self, retriever: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all runs for a specific retriever, optionally filtered by k."""
        if k is not None:
            return [r for r in self.runs if r["retriever"] == retriever and r["k"] == k]
        return [r for r in self.runs if r["retriever"] == retriever]


def load_retrieval_runs(
    path: str | Path = DEFAULT_RUNS_OUTPUT_PATH,
) -> RetrievalRunsCollection:
    """Load retrieval runs from JSON."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Retrieval runs file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RetrievalRunsCollection(data)


def load_retrieval_metrics(
    path: str | Path = DEFAULT_METRICS_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Load retrieval metrics from JSON."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Retrieval metrics file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Reconstructing retrieval runs across 5 systems and 3 K values...")
    runs_dict = reconstruct_retrieval_runs()
    runs_file = save_retrieval_runs(runs_dict)
    print(f"Saved {runs_dict['metadata']['total_runs']} runs to: {runs_file}")

    print("Computing metrics and validating against Stage 2 baselines...")
    metrics_dict = compute_retrieval_metrics(runs_dict)
    metrics_file = save_retrieval_metrics(metrics_dict)
    print(f"Saved metrics to: {metrics_file}")

    val = metrics_dict["stage2_validation"]
    print(f"\nStage 2 baseline reproduction status: {'ALL PASSED' if val['all_systems_passed'] else 'FAILED'}")
    for sys_id, details in val["systems"].items():
        print(f"  {sys_id:20s}: {details['status']}")
        for m, stat in details.items():
            if m != "status":
                print(f"    {m:10s} expected={stat['expected']:.4f} reconstructed={stat['reconstructed']:.4f} delta={stat['delta']:.5f}")
