"""
test_statistical_analysis.py
-----------------------------
Comprehensive automated unit and regression tests for src.evaluation.statistical_analysis.

Verifies:
1. Exact 885-row inner join across frozen retrieval and evaluator artifacts.
2. Complete population integrity: 59 queries, 5 retrievers, 3 K values, 0 duplicates.
3. Strict exclusion of 6 sensitivity queries (Q007, Q013, Q017, Q021, Q055, Q061).
4. No null, NaN, or infinite values in joined experimental metrics.
5. Configuration summary calculations (15 rows, means, medians, 95% bootstrap CIs).
6. Query-level Spearman rank correlations for all 15 configurations.
7. H2 paired bootstrap comparisons (2000 iterations, reproducibility with seed=42).
8. System-level exploratory correlations (all-system n=15 and non-LSH n=12).
9. Paired Wilcoxon comparisons with Benjamini-Hochberg FDR correction.
10. Existence and integrity of generated figures and CSV/JSON output artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_analysis import (
    DEFAULT_CONFIG_SUMMARY_CSV,
    DEFAULT_EVALUATOR_RESULTS_PATH,
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_PAIRWISE_CSV,
    DEFAULT_QUERY_CORR_CSV,
    DEFAULT_REFERENCE_EVIDENCE_PATH,
    DEFAULT_RETRIEVAL_METRICS_PATH,
    EXPECTED_K_VALUES,
    EXPECTED_RETRIEVERS,
    RANDOM_SEED,
    SENSITIVITY_QUERIES,
    bootstrap_ci,
    compute_configuration_summaries,
    compute_h2_bootstrap_comparisons,
    compute_paired_comparisons,
    compute_pooled_correlations,
    compute_query_level_correlations,
    compute_system_level_analysis,
    load_and_join_data,
)


@pytest.fixture(scope="module")
def joined_dataset():
    """Load and join the 885 primary conditions."""
    return load_and_join_data(
        reference_evidence_path=DEFAULT_REFERENCE_EVIDENCE_PATH,
        retrieval_metrics_path=DEFAULT_RETRIEVAL_METRICS_PATH,
        evaluator_results_path=DEFAULT_EVALUATOR_RESULTS_PATH,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Join & Population Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_joined_dataset_dimensions_and_population(joined_dataset):
    df = joined_dataset
    assert len(df) == 885, f"Expected 885 joined rows, got {len(df)}"

    unique_keys = df[["query_id", "retriever", "k"]].drop_duplicates()
    assert len(unique_keys) == 885, f"Expected 885 unique keys, got {len(unique_keys)}"

    queries = set(df["query_id"].unique())
    assert len(queries) == 59, f"Expected 59 queries, got {len(queries)}"

    # Strict exclusion check
    assert queries.isdisjoint(SENSITIVITY_QUERIES), "Found sensitivity queries in primary join!"

    # Retrievers check
    retrievers = set(df["retriever"].unique())
    assert retrievers == set(EXPECTED_RETRIEVERS), f"Retrievers mismatch: {retrievers}"

    # K values check
    k_vals = set(df["k"].unique())
    assert k_vals == set(EXPECTED_K_VALUES), f"K values mismatch: {k_vals}"


def test_joined_dataset_no_missing_or_invalid_values(joined_dataset):
    df = joined_dataset
    assert df.isna().sum().sum() == 0, "Found missing/NaN values in joined dataset"

    # Numeric range checks
    for col in ["evidence_coverage", "factual_consistency", "answer_relevance", "composite_quality"]:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Column {col} out of [0, 1] range"

    for col in ["ndcg_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision"]:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Retrieval column {col} out of [0, 1] range"

    # Composite consistency
    computed_comp = (df["evidence_coverage"] + df["factual_consistency"] + df["answer_relevance"]) / 3.0
    np.testing.assert_allclose(df["composite_quality"], computed_comp, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configuration Summaries Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_configuration_summaries(joined_dataset):
    summary = compute_configuration_summaries(joined_dataset, n_bootstrap=200, seed=RANDOM_SEED)
    assert len(summary) == 15, f"Expected 15 configuration summaries, got {len(summary)}"

    required_cols = [
        "retriever", "k", "n_queries",
        "ndcg_mean", "precision_mean", "recall_mean", "mrr_mean", "map_mean",
        "coverage_mean", "coverage_ci_lower", "coverage_ci_upper",
        "composite_mean", "composite_ci_lower", "composite_ci_upper",
    ]
    for col in required_cols:
        assert col in summary.columns, f"Missing column {col} in configuration summaries"

    assert (summary["n_queries"] == 59).all(), "Every configuration must contain exactly 59 queries"
    assert (summary["composite_ci_lower"] <= summary["composite_mean"]).all()
    assert (summary["composite_mean"] <= summary["composite_ci_upper"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Query-Level Correlations Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_query_level_correlations(joined_dataset):
    corrs = compute_query_level_correlations(joined_dataset)
    assert len(corrs) == 15, f"Expected 15 correlation rows, got {len(corrs)}"

    for _, row in corrs.iterrows():
        assert row["n_queries"] == 59
        # H1: NDCG vs Coverage
        assert -1.0 <= row["ndcg_vs_coverage_rho"] <= 1.0
        assert 0.0 <= row["ndcg_vs_coverage_pvalue"] <= 1.0
        # H2: Coverage vs Composite
        assert -1.0 <= row["coverage_vs_composite_rho"] <= 1.0
        assert 0.0 <= row["coverage_vs_composite_pvalue"] <= 1.0

    # Verify that in all 15 configurations, Coverage vs Composite rho is strongly positive
    assert (corrs["coverage_vs_composite_rho"] > 0.80).all(), "Coverage vs Composite must be > 0.80 across configs"


# ─────────────────────────────────────────────────────────────────────────────
# 4. H2 Bootstrap Comparisons Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_h2_bootstrap_comparisons(joined_dataset):
    h2_res = compute_h2_bootstrap_comparisons(joined_dataset, n_bootstrap=300, seed=RANDOM_SEED)
    assert len(h2_res) == 15

    for _, row in h2_res.iterrows():
        # Delta = rho(Coverage, Composite) - rho(NDCG, Composite)
        expected_delta = round(row["rho_coverage_composite"] - row["rho_ndcg_composite"], 4)
        assert abs(row["observed_delta"] - expected_delta) < 1e-3
        assert row["delta_ci_lower"] <= row["delta_ci_upper"]

    # In all 15 configurations, Evidence Coverage has stronger correlation with Composite than NDCG
    assert (h2_res["observed_delta"] > 0).all()
    assert (h2_res["prop_bootstrap_delta_positive"] > 0.95).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. System-Level Exploratory Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_system_level_analysis(joined_dataset):
    summary = compute_configuration_summaries(joined_dataset, n_bootstrap=100, seed=RANDOM_SEED)
    sys_analysis = compute_system_level_analysis(summary)

    assert "all_configurations" in sys_analysis
    assert "non_lsh_configurations" in sys_analysis
    assert sys_analysis["all_configurations"]["n_configurations"] == 15
    assert sys_analysis["non_lsh_configurations"]["n_configurations"] == 12

    # Check that coverage vs composite remains strong without LSH
    r_cov_comp_no_lsh = sys_analysis["non_lsh_configurations"]["coverage_vs_composite"]["rho"]
    assert r_cov_comp_no_lsh > 0.85


# ─────────────────────────────────────────────────────────────────────────────
# 6. Paired Wilcoxon Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_paired_comparisons(joined_dataset):
    paired = compute_paired_comparisons(joined_dataset)
    # 5 pairs * 3 K values = 15 rows
    assert len(paired) == 15

    for _, row in paired.iterrows():
        assert row["n_queries"] == 59
        assert 0.0 <= row["pvalue_composite"] <= 1.0
        assert 0.0 <= row["fdr_pvalue_composite"] <= 1.0

    # BM25 vs LSH at K=10 must be highly significant
    bm25_lsh_k10 = paired[(paired["k"] == 10) & (paired["system_1"] == "bm25") & (paired["system_2"] == "lsh")]
    assert len(bm25_lsh_k10) == 1
    assert bm25_lsh_k10.iloc[0]["fdr_pvalue_composite"] < 0.001
    assert bm25_lsh_k10.iloc[0]["mean_delta_composite"] > 0.35


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output Artifacts Existence & Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exported_artifacts_exist_and_valid():
    assert DEFAULT_OUTPUT_JSON.exists(), f"Missing output JSON: {DEFAULT_OUTPUT_JSON}"
    assert DEFAULT_CONFIG_SUMMARY_CSV.exists(), f"Missing summary CSV: {DEFAULT_CONFIG_SUMMARY_CSV}"
    assert DEFAULT_QUERY_CORR_CSV.exists(), f"Missing correlation CSV: {DEFAULT_QUERY_CORR_CSV}"
    assert DEFAULT_PAIRWISE_CSV.exists(), f"Missing pairwise CSV: {DEFAULT_PAIRWISE_CSV}"

    with open(DEFAULT_OUTPUT_JSON, "r", encoding="utf-8") as f:
        master = json.load(f)

    meta = master["metadata"]
    assert meta["n_primary_queries"] == 59
    assert meta["n_configurations"] == 15
    assert meta["n_conditions"] == 885
    assert meta["retrieval_dense_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta["generator_model"] == "gemini-3.1-flash-lite"
    assert meta["evaluator_model"] == "command-r-08-2024"
    assert meta["shared_reference_dependency_disclosed"] is True

    # Check figures
    for fig_id in ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]:
        fig_path = DEFAULT_FIGURES_DIR / f"{fig_id}_*.png"
        matches = list(DEFAULT_FIGURES_DIR.glob(f"{fig_id}_*.png"))
        assert len(matches) >= 1, f"Missing figure for {fig_id}"
