"""
run_h2_correction.py
--------------------
CLI runner for Task 4F-C: H2 Methodological Correction and Reanalysis.

Generates:
- experiments/results/stage3_h2_corrected.json
- experiments/results/stage3_h2_corrected.csv
- experiments/results/stage3_h2_bootstrap.csv
- experiments/results/stage3_daq_configuration_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.h2_correction import (
    DEFAULT_DAQ_SUMMARY_CSV,
    DEFAULT_EVALUATOR_RESULTS_PATH,
    DEFAULT_H2_BOOTSTRAP_CSV,
    DEFAULT_H2_CORRECTED_CSV,
    DEFAULT_H2_CORRECTED_JSON,
    DEFAULT_REFERENCE_EVIDENCE_PATH,
    DEFAULT_RETRIEVAL_METRICS_PATH,
    RANDOM_SEED,
    run_h2_reanalysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Task 4F-C H2 methodological correction and reanalysis."
    )
    parser.add_argument(
        "--reference-evidence",
        type=str,
        default=str(DEFAULT_REFERENCE_EVIDENCE_PATH),
    )
    parser.add_argument(
        "--retrieval-metrics",
        type=str,
        default=str(DEFAULT_RETRIEVAL_METRICS_PATH),
    )
    parser.add_argument(
        "--evaluator-results",
        type=str,
        default=str(DEFAULT_EVALUATOR_RESULTS_PATH),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_H2_CORRECTED_JSON),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("Task 4F-C: H2 Methodological Correction & Reanalysis")
    print("=" * 70)
    print(f"Downstream Metric: DAQ = (Factual Consistency + Answer Relevance) / 2.0")
    print(f"Bootstrap Samples: {args.bootstrap_samples}")
    print(f"Random Seed:       {args.seed}")
    print("=" * 70)

    results = run_h2_reanalysis(
        reference_evidence_path=args.reference_evidence,
        retrieval_metrics_path=args.retrieval_metrics,
        evaluator_results_path=args.evaluator_results,
        output_json_path=args.output_json,
        output_correlations_csv=DEFAULT_H2_CORRECTED_CSV,
        output_bootstrap_csv=DEFAULT_H2_BOOTSTRAP_CSV,
        output_daq_summary_csv=DEFAULT_DAQ_SUMMARY_CSV,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("TASK 4F-C H2 CORRECTION REANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Master JSON:     {args.output_json}")
    print(f"Correlations:    {DEFAULT_H2_CORRECTED_CSV}")
    print(f"Bootstrap CSV:   {DEFAULT_H2_BOOTSTRAP_CSV}")
    print(f"DAQ Summary:     {DEFAULT_DAQ_SUMMARY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
