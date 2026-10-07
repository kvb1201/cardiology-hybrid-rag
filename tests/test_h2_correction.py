"""
test_h2_correction.py
---------------------
Automated unit and validation tests for Task 4F-C: H2 Methodological Correction and Reanalysis.

Verifies:
1. DAQ mathematical formula: exactly equals (Factual Consistency + Answer Relevance) / 2.0.
2. DAQ bounds: strictly finite and bounded in [0.0, 1.0].
3. Primary population: exactly 885 rows, 59 queries per configuration, 15 configurations.
4. Sensitivity query exclusion: six grade-1-only queries (Q007, Q013, Q017, Q021, Q055, Q061) strictly absent.
5. Key uniqueness: exactly 885 unique (query_id, retriever, k) condition keys (0 duplicates).
6. Evidence Coverage completely excluded from DAQ.
7. Corrected H2 correlation values are finite, in [-1, 1], with valid p-values in [0, 1].
8. Bootstrap reproducibility with seed 42.
9. Artifact integrity: JSON and CSV files exist and adhere to declared schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.h2_correction import (
    DEFAULT_DAQ_SUMMARY_CSV,
    DEFAULT_EVALUATOR_RESULTS_PATH,
    DEFAULT_H2_BOOTSTRAP_CSV,
    DEFAULT_H2_CORRECTED_CSV,
    DEFAULT_H2_CORRECTED_JSON,
    DEFAULT_REFERENCE_EVIDENCE_PATH,
    DEFAULT_RETRIEVAL_METRICS_PATH,
    EXPECTED_K_VALUES,
    EXPECTED_RETRIEVERS,
    RANDOM_SEED,
    SENSITIVITY_QUERIES,
    compute_daq_configuration_summary,
    compute_daq_system_level_analysis,
    compute_h2_corrected_bootstrap,
    compute_h2_corrected_correlations,
    load_and_compute_daq,
)


@pytest.fixture(scope="module")
def daq_dataset():
    """Load and compute the DAQ dataset across the 885 primary conditions."""
    return load_and_compute_daq(
        reference_evidence_path=DEFAULT_REFERENCE_EVIDENCE_PATH,
        retrieval_metrics_path=DEFAULT_RETRIEVAL_METRICS_PATH,
        evaluator_results_path=DEFAULT_EVALUATOR_RESULTS_PATH,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. DAQ Metric Definition & Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_daq_formula_and_range(daq_dataset):
    df = daq_dataset
    # Exact definition: DAQ = (FC + AR) / 2.0
    expected_daq = (df["factual_consistency"] + df["answer_relevance"]) / 2.0
    np.testing.assert_allclose(df["daq"], expected_daq, atol=1e-6)

    # Range check
    assert (df["daq"] >= 0.0).all()
    assert (df["daq"] <= 1.0).all()
    assert df["daq"].notna().all()
    assert np.isfinite(df["daq"]).all()


def test_daq_population_and_query_exclusion(daq_dataset):
    df = daq_dataset
    assert len(df) == 885, f"Expected 885 rows, got {len(df)}"

    unique_keys = df[["query_id", "retriever", "k"]].drop_duplicates()
    assert len(unique_keys) == 885, f"Expected 885 unique condition keys, got {len(unique_keys)}"

    queries = set(df["query_id"].unique())
    assert len(queries) == 59, f"Expected 59 queries, got {len(queries)}"
    assert queries.isdisjoint(SENSITIVITY_QUERIES), "Found excluded sensitivity queries in DAQ dataset!"

    retrievers = set(df["retriever"].unique())
    assert retrievers == set(EXPECTED_RETRIEVERS)

    k_vals = set(df["k"].unique())
    assert k_vals == set(EXPECTED_K_VALUES)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Corrected H2 Correlations Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_corrected_h2_correlations(daq_dataset):
    corrs = compute_h2_corrected_correlations(daq_dataset)
    assert len(corrs) == 15, f"Expected 15 configuration correlation rows, got {len(corrs)}"

    for _, row in corrs.iterrows():
        assert row["n_queries"] == 59
        # Boundedness
        assert -1.0 <= row["rho_coverage_daq"] <= 1.0
        assert 0.0 <= row["pvalue_coverage_daq"] <= 1.0
        assert -1.0 <= row["rho_ndcg_daq"] <= 1.0
        assert 0.0 <= row["pvalue_ndcg_daq"] <= 1.0

        # Delta calculation
        expected_delta = round(row["rho_coverage_daq"] - row["rho_ndcg_daq"], 4)
        assert abs(row["delta"] - expected_delta) < 1e-3

    # In all 15 configurations, Coverage -> DAQ is strictly positive
    assert (corrs["rho_coverage_daq"] > 0.75).all()

    # In all 15 configurations, Delta is positive
    assert (corrs["delta"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Corrected H2 Bootstrap Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_corrected_h2_bootstrap_reproducibility(daq_dataset):
    b1 = compute_h2_corrected_bootstrap(daq_dataset, n_bootstrap=200, seed=RANDOM_SEED)
    b2 = compute_h2_corrected_bootstrap(daq_dataset, n_bootstrap=200, seed=RANDOM_SEED)

    pd.testing.assert_frame_equal(b1, b2)
    assert len(b1) == 15

    for _, row in b1.iterrows():
        assert row["delta_ci_lower"] <= row["delta_ci_upper"]
        assert 0.0 <= row["p_bootstrap_delta_positive"] <= 1.0

    # Verify that in all 15 configurations, bootstrap CI excludes zero
    assert (b1["ci_excludes_zero"] == True).all()
    assert (b1["p_bootstrap_delta_positive"] > 0.95).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Configuration Summaries & System-Level Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_daq_configuration_summary(daq_dataset):
    summary = compute_daq_configuration_summary(daq_dataset, n_bootstrap=100, seed=RANDOM_SEED)
    assert len(summary) == 15

    for _, row in summary.iterrows():
        assert row["n_queries"] == 59
        assert row["daq_ci_lower"] <= row["daq_mean"] <= row["daq_ci_upper"]

    sys_analysis = compute_daq_system_level_analysis(summary)
    assert "all_configurations" in sys_analysis
    assert "non_lsh_configurations" in sys_analysis
    assert sys_analysis["all_configurations"]["n_configurations"] == 15
    assert sys_analysis["non_lsh_configurations"]["n_configurations"] == 12

    # In non-LSH configs, Coverage -> DAQ remains strong (> 0.70) while NDCG -> DAQ collapses (< 0.15)
    r_cov_no_lsh = sys_analysis["non_lsh_configurations"]["coverage_vs_daq"]["rho"]
    r_ndcg_no_lsh = sys_analysis["non_lsh_configurations"]["ndcg_vs_daq"]["rho"]
    assert r_cov_no_lsh > 0.75
    assert abs(r_ndcg_no_lsh) < 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 5. Output Artifacts Existence & Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exported_h2_artifacts_exist_and_valid():
    assert DEFAULT_H2_CORRECTED_JSON.exists(), f"Missing {DEFAULT_H2_CORRECTED_JSON}"
    assert DEFAULT_H2_CORRECTED_CSV.exists(), f"Missing {DEFAULT_H2_CORRECTED_CSV}"
    assert DEFAULT_H2_BOOTSTRAP_CSV.exists(), f"Missing {DEFAULT_H2_BOOTSTRAP_CSV}"
    assert DEFAULT_DAQ_SUMMARY_CSV.exists(), f"Missing {DEFAULT_DAQ_SUMMARY_CSV}"

    with open(DEFAULT_H2_CORRECTED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["n_primary_queries"] == 59
    assert meta["n_configurations"] == 15
    assert meta["n_conditions"] == 885
    assert meta["evidence_coverage_excluded_from_daq"] is True
    assert meta["random_seed"] == RANDOM_SEED
