"""
build_stage3_infrastructure.py
------------------------------
Command-line runner to construct the Stage 3 deterministic reference evidence
and reconstruct the retrieval runs across all 5 systems at K = 5, 10, 20.

Usage:
    python scripts/build_stage3_infrastructure.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reference_evidence import (
    build_reference_evidence,
    save_reference_evidence,
)
from src.evaluation.retrieval_reconstruction import (
    compute_retrieval_metrics,
    reconstruct_retrieval_runs,
    save_retrieval_metrics,
    save_retrieval_runs,
)


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 70)
    print("Stage 3 Infrastructure Builder")
    print("=" * 70)

    # 1. Reference Evidence Builder
    print("\n[Step 1/3] Building deterministic reference evidence...")
    evidence_data = build_reference_evidence()
    ref_file = save_reference_evidence(evidence_data)
    print(f"  -> Saved reference evidence to: {ref_file}")
    print(f"  -> Total queries: {len(evidence_data['queries'])}")
    print(f"  -> Grade-2 queries (primary): {evidence_data['metadata']['primary_analysis_queries_count']}")
    print(f"  -> Grade-1 fallback queries: {evidence_data['metadata']['sensitivity_fallback_queries_count']}")

    # 2. Retrieval Run Reconstruction
    print("\n[Step 2/3] Reconstructing retrieval runs (65 queries x 5 retrievers x 3 K values)...")
    runs_data = reconstruct_retrieval_runs()
    runs_file = save_retrieval_runs(runs_data)
    print(f"  -> Saved {runs_data['metadata']['total_runs']} runs to: {runs_file}")

    # 3. Retrieval Metrics Computation & Stage 2 Validation
    print("\n[Step 3/3] Computing retrieval metrics and validating against Stage 2...")
    metrics_data = compute_retrieval_metrics(runs_data)
    metrics_file = save_retrieval_metrics(metrics_data)
    print(f"  -> Saved metrics to: {metrics_file}")

    val = metrics_data["stage2_validation"]
    print("\n" + "=" * 70)
    print("STAGE 2 BASELINE REPRODUCTION VALIDATION")
    print("=" * 70)
    print(f"Overall Status: {'ALL PASSED' if val['all_systems_passed'] else 'FAILED'}\n")

    header = f"{'System':<20} {'Metric':<10} {'Expected':<10} {'Reconstructed':<15} {'Delta':<10} {'Status':<8}"
    print(header)
    print("-" * len(header))

    for sys_id, details in val["systems"].items():
        for m, stat in details.items():
            if m != "status":
                print(
                    f"{sys_id:<20} {m:<10} {stat['expected']:<10.4f} "
                    f"{stat['reconstructed']:<15.4f} {stat['delta']:<10.5f} "
                    f"{'OK' if stat['passed'] else 'FAIL':<8}"
                )

    elapsed = time.perf_counter() - t_start
    print(f"\nStage 3 infrastructure build completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
