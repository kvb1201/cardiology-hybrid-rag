"""
run_generation.py
-----------------
CLI entry point to execute the fixed downstream RAG generation experiment.

Usage:
    # Full dry-run (validates all 975 conditions with 0 API calls):
    python scripts/run_generation.py --dry-run

    # 3-condition pilot run:
    python scripts/run_generation.py --limit 3

    # Full execution across 975 conditions (once approved):
    python scripts/run_generation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.generator import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RUNS_PATH,
    RAGGenerator,
    load_completed_runs,
    load_generation_config,
    run_generation_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute fixed downstream RAG generation experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to generation YAML configuration file.",
    )
    parser.add_argument(
        "--runs-file",
        type=str,
        default=str(DEFAULT_RUNS_PATH),
        help="Path to stage3_retrieval_runs.json.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to stage3_generated_answers.jsonl output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate prompt construction and schemas across all 975 conditions without calling LLM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit execution to first N conditions.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Run only the 3 pilot conditions: Q001/bm25/K=5, Q001/dense/K=5, Q001/bm25_dense_rrf/K=5.",
    )
    parser.add_argument(
        "--batch1",
        action="store_true",
        default=False,
        help="Execute Generation Batch 1: target exactly 485 new incomplete experiment conditions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("Stage 3 Fixed Downstream RAG Generation")
    print("=" * 70)

    cfg = load_generation_config(args.config)
    provider = cfg.get("provider", "unspecified")
    model = cfg.get("model", "unspecified")

    print(f"Provider:    {provider}")
    print(f"Model:       {model}")
    print(f"Temperature: {cfg.get('temperature')}")
    print(f"Top P:       {cfg.get('top_p')}")
    print(f"Max tokens:  {cfg.get('max_tokens')}")
    print(f"System ver:  {cfg.get('system_prompt_version')}")
    print(f"Prompt ver:  {cfg.get('generation_prompt_version')}")
    print(f"Dry run:     {args.dry_run}")
    print(f"Pilot mode:  {args.pilot}")
    print(f"Batch 1:     {args.batch1}")
    print(f"Limit:       {args.limit}")

    if not args.dry_run and (provider == "unspecified" or not provider):
        print("\n" + "!" * 70)
        print("STOP: Generator model/provider is unspecified and requires explicit project configuration.")
        print("!" * 70)
        sys.exit(1)

    generator = RAGGenerator(config=cfg)

    filter_keys = None
    limit = args.limit
    if args.pilot:
        filter_keys = [
            ("Q001", "bm25", 5),
            ("Q001", "dense", 5),
            ("Q001", "bm25_dense_rrf", 5),
        ]
        limit = 3
    elif args.batch1:
        limit = 485

    t0 = time.perf_counter()
    summary = run_generation_experiment(
        generator=generator,
        runs_path=args.runs_file,
        output_path=args.output_file,
        dry_run=args.dry_run,
        limit=limit,
        filter_keys=filter_keys,
    )
    elapsed = time.perf_counter() - t0

    # Key validation
    all_completed = load_completed_runs(args.output_file) if not args.dry_run else set()
    output_records: List[Dict[str, Any]] = []
    out_path = Path(args.output_file)
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    output_records.append(json.loads(line))

    # Calculate token usage from output records
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_tokens = 0
    for r in output_records:
        u = r.get("usage_metadata", {})
        total_prompt_tokens += u.get("promptTokenCount", 0)
        total_candidate_tokens += u.get("candidatesTokenCount", 0)
        total_tokens += u.get("totalTokenCount", 0)

    # Check for duplicate and missing keys in output
    all_success_keys = [(r["query_id"], r["retriever"], int(r["k"])) for r in output_records if r.get("status") == "success"]
    duplicates = len(all_success_keys) - len(set(all_success_keys))
    missing_count = summary['total_conditions'] - len(set(all_success_keys))

    print("\n" + "=" * 70)
    print("TASK 4D: GENERATION EXECUTION & VALIDATION REPORT")
    print("=" * 70)
    print(f"Generator model:                  {model}")
    print(f"Previously completed conditions:  {summary.get('previously_completed', 0)}")
    print(f"NEW successful conditions:        {summary['new_success_count']}")
    print(f"Failed conditions:                {summary['failed_count']}")
    print(f"Retry attempts:                   {generator.caller.total_retries}")
    print(f"Quota failures:                   {1 if summary.get('quota_stopped') else 0}")
    if summary.get("quota_stopped"):
        print(f"Quota error message:              {summary.get('quota_error_msg')}")
    print(f"Total completed conditions:       {summary['total_completed']}")
    print(f"Remaining conditions:             {summary['remaining_conditions']}")
    print(f"Duplicate experiment keys:        {duplicates}")
    print(f"Missing experiment keys:          {missing_count}")
    print(f"Output JSONL path:                {args.output_file}")
    print(f"Elapsed time:                     {elapsed:.2f}s")
    if total_tokens > 0:
        print(f"Total prompt tokens:              {total_prompt_tokens:,}")
        print(f"Total candidate tokens:           {total_candidate_tokens:,}")
        print(f"Total tokens:                     {total_tokens:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
