"""
run_statistical_analysis.py
---------------------------
CLI runner for Task 4F: Frozen Results Statistical Analysis.

Executes:
- Data join and validation (885 primary conditions, 59 queries, 15 configurations)
- Configuration-level summary with 95% bootstrap CIs
- Query-level Spearman rank correlations for all 15 configurations
- H2 paired bootstrap comparisons (2000 resamples)
- Exploratory system-level correlations (all systems and non-LSH)
- Paired nonparametric Wilcoxon signed-rank tests with FDR correction
- Publication-quality figure generation (6 figures)
- Export of CSV and JSON statistical artifacts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_analysis import (
    DEFAULT_CONFIG_SUMMARY_CSV,
    DEFAULT_EVALUATOR_RESULTS_PATH,
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_PAIRWISE_CSV,
    DEFAULT_QUERY_CORR_CSV,
    DEFAULT_REFERENCE_EVIDENCE_PATH,
    DEFAULT_RETRIEVAL_METRICS_PATH,
    RANDOM_SEED,
    run_complete_statistical_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Task 4F statistical analysis across frozen experiment results."
    )
    parser.add_argument(
        "--reference-evidence",
        type=str,
        default=str(DEFAULT_REFERENCE_EVIDENCE_PATH),
        help="Path to reference evidence JSON.",
    )
    parser.add_argument(
        "--retrieval-metrics",
        type=str,
        default=str(DEFAULT_RETRIEVAL_METRICS_PATH),
        help="Path to frozen retrieval metrics JSON.",
    )
    parser.add_argument(
        "--evaluator-results",
        type=str,
        default=str(DEFAULT_EVALUATOR_RESULTS_PATH),
        help="Path to completed evaluator results JSONL.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output path for master statistical JSON.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(DEFAULT_FIGURES_DIR),
        help="Directory to save generated publication figures.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for bootstrap reproducibility (default: 42).",
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("Task 4F: Frozen Results Statistical Analysis Pipeline")
    print("=" * 70)
    print(f"Reference Evidence: {args.reference_evidence}")
    print(f"Retrieval Metrics:  {args.retrieval_metrics}")
    print(f"Evaluator Results:  {args.evaluator_results}")
    print(f"Output JSON:        {args.output_json}")
    print(f"Figures Dir:        {args.figures_dir}")
    print(f"Bootstrap Samples:  {args.bootstrap_samples}")
    print(f"Random Seed:        {args.seed}")
    print("=" * 70)

    results = run_complete_statistical_analysis(
        reference_evidence_path=args.reference_evidence,
        retrieval_metrics_path=args.retrieval_metrics,
        evaluator_results_path=args.evaluator_results,
        output_json_path=args.output_json,
        config_summary_csv=DEFAULT_CONFIG_SUMMARY_CSV,
        query_corr_csv=DEFAULT_QUERY_CORR_CSV,
        pairwise_csv=DEFAULT_PAIRWISE_CSV,
        figures_dir=args.figures_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("TASK 4F STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    meta = results["metadata"]
    print(f"Primary Queries:       {meta['n_primary_queries']}")
    print(f"Configurations:        {meta['n_configurations']}")
    print(f"Total Conditions:      {meta['n_conditions']}")
    print(f"Figures Created:       {len(results['figures_generated'])}")
    for name, p in results["figures_generated"].items():
        print(f"  - {name}: {p}")
    print(f"CSV Summaries:")
    print(f"  - {DEFAULT_CONFIG_SUMMARY_CSV}")
    print(f"  - {DEFAULT_QUERY_CORR_CSV}")
    print(f"  - {DEFAULT_PAIRWISE_CSV}")
    print(f"JSON Master:")
    print(f"  - {args.output_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
