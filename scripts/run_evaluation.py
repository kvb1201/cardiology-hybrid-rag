"""
run_evaluation.py
-----------------
CLI entry point to execute the downstream LLM evaluation for the Cardiology-Chat research study.

Evaluator:
    Model: Cohere Command R (command-r-08-2024)
    Dimensions:
        - evidence_coverage (0.0 to 1.0)
        - factual_consistency (0.0 to 1.0)
        - answer_relevance (0.0 to 1.0)
        - composite_quality = mean(coverage, consistency, relevance) [programmatic]

Methodological Controls:
    - Evaluator is strictly BLIND: receives ONLY Query, Answer, and Reference Evidence.
    - Zero retriever identity, cutoff K, or generation metadata exposed.

Usage:
    # Dry-run validation (validates pipeline logic across 10 pilot conditions with 0 API calls):
    python scripts/run_evaluation.py --dry-run --pilot

    # Run the 10-condition live pilot:
    python scripts/run_evaluation.py --pilot

    # Full execution across 975 conditions (when authorized):
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_GENERATED_ANSWERS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PILOT_OUTPUT_PATH,
    DEFAULT_REFERENCE_EVIDENCE_PATH,
    DownstreamEvaluator,
    load_completed_evaluations,
    load_evaluator_config,
    load_generated_answers,
    load_reference_evidence_map,
)

# Standard 10-condition pilot matrix covering all 5 retrievers, 3 K values, and 7 topics
PILOT_CONDITIONS: List[Tuple[str, str, int]] = [
    ("Q001", "bm25", 5),
    ("Q001", "dense", 5),
    ("Q009", "dense", 10),
    ("Q009", "bm25_dense_rrf", 10),
    ("Q018", "bm25_dense_rrf", 20),
    ("Q018", "bm25_dense_lsh_rrf", 20),
    ("Q027", "lsh", 5),
    ("Q037", "lsh", 10),
    ("Q044", "bm25", 20),
    ("Q049", "bm25_dense_lsh_rrf", 10),
]

SENSITIVITY_QUERIES = {"Q007", "Q013", "Q017", "Q021", "Q055", "Q061"}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute downstream LLM evaluation for Cardiology-Chat."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to evaluator YAML configuration file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output JSONL file. Defaults to pilot or main output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate evaluation pipeline logic using mock client without API calls.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Run only the 10 balanced pilot conditions across retrievers, K values, and topics.",
    )
    parser.add_argument(
        "--primary",
        action="store_true",
        default=True,
        help="Run the 885 primary conditions (59 grade-2 queries, excluding 6 sensitivity queries).",
    )
    parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        default=False,
        help="Include the 90 sensitivity conditions (all 975 conditions).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit execution to first N conditions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    print("=" * 70)
    print("Stage 3 Downstream LLM Evaluator (Task 4E)")
    print("=" * 70)

    cfg = load_evaluator_config(args.config)
    provider = cfg.get("provider", "cohere")
    model = cfg.get("model", "command-r-08-2024")
    temp = cfg.get("temperature", 0.0)

    print(f"Evaluator Provider: {provider}")
    print(f"Evaluator Model:    {model}")
    print(f"Temperature:        {temp}")
    print(f"Execution Mode:     {'DRY-RUN (Mock)' if args.dry_run else 'LIVE'}")

    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    elif args.pilot:
        output_path = DEFAULT_PILOT_OUTPUT_PATH
    else:
        output_path = DEFAULT_OUTPUT_PATH

    print(f"Output File:        {output_path}")

    # Determine condition matrix
    if args.pilot:
        target_conditions = list(PILOT_CONDITIONS)
        print(f"Selected Pilot:     {len(target_conditions)} conditions")
    else:
        # Load all conditions from generated answers
        answers_map = load_generated_answers(DEFAULT_GENERATED_ANSWERS_PATH)
        all_conditions = sorted(list(answers_map.keys()))
        if args.include_sensitivity:
            target_conditions = all_conditions
            print(f"Full Matrix (inc. sensitivity): {len(target_conditions)} conditions")
        else:
            # Primary cohort: 59 queries with grade-2 evidence (885 conditions)
            target_conditions = [c for c in all_conditions if c[0] not in SENSITIVITY_QUERIES]
            print(f"Primary Grade-2 Matrix: {len(target_conditions)} conditions (59 queries)")
            print(f"Sensitivity Excluded:   {len(all_conditions) - len(target_conditions)} conditions (6 queries)")

    if args.limit:
        target_conditions = target_conditions[: args.limit]
        print(f"Applied Limit:      {len(target_conditions)} conditions")

    # Initialize evaluator
    evaluator = DownstreamEvaluator(
        config=cfg,
        use_mock=args.dry_run,
    )

    t0 = time.time()
    summary = evaluator.run(
        conditions=target_conditions,
        output_path=output_path,
        reference_evidence_path=DEFAULT_REFERENCE_EVIDENCE_PATH,
        generated_answers_path=DEFAULT_GENERATED_ANSWERS_PATH,
    )
    elapsed = time.time() - t0

    # Read output records to compute statistics
    output_records = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        output_records.append(rec)
                    except json.JSONDecodeError:
                        pass

    success_records = [r for r in output_records if r.get("status") == "success"]
    total_tokens = sum(
        r.get("tokens", {}).get("input_tokens", 0) + r.get("tokens", {}).get("output_tokens", 0)
        for r in success_records
    )

    print("\n" + "=" * 70)
    print("EVALUATION EXECUTION & VALIDATION REPORT")
    print("=" * 70)
    print(f"Evaluator model:                  {model}")
    print(f"Conditions targeted:              {summary['total_conditions']}")
    print(f"Previously skipped conditions:    {summary['skipped']}")
    print(f"Newly succeeded conditions:       {summary['succeeded']}")
    print(f"Failed conditions:                {summary['failed']}")
    print(f"Total completed in output:        {len(success_records)}")
    print(f"Output JSONL path:                {output_path}")
    print(f"Elapsed time:                     {elapsed:.2f}s")
    if total_tokens > 0:
        print(f"Approximate tokens recorded:      {total_tokens:,}")

    if success_records:
        avg_cov = sum(r["scores"]["evidence_coverage"] for r in success_records) / len(success_records)
        avg_con = sum(r["scores"]["factual_consistency"] for r in success_records) / len(success_records)
        avg_rel = sum(r["scores"]["answer_relevance"] for r in success_records) / len(success_records)
        avg_comp = sum(r["scores"]["composite_quality"] for r in success_records) / len(success_records)
        print("-" * 70)
        print(f"Mean Evidence Coverage:           {avg_cov:.4f}")
        print(f"Mean Factual Consistency:         {avg_con:.4f}")
        print(f"Mean Answer Relevance:            {avg_rel:.4f}")
        print(f"Mean Composite Quality:           {avg_comp:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
