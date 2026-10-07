"""
h2_correction.py
----------------
Task 4F-C: Methodological Correction and Reanalysis for Hypothesis 2 (H2).

Methodological Rationale:
    The original composite quality metric was defined as:
        Composite Quality = (Evidence Coverage + Factual Consistency + Answer Relevance) / 3
    Correlating Evidence Coverage with Composite Quality introduced a structural dependency,
    as Evidence Coverage was itself one-third of the outcome variable.

    To eliminate this circularity and establish an independent evaluation of H2,
    this module implements:
        Downstream Answer Quality (DAQ) = (Factual Consistency + Answer Relevance) / 2.0
    where Evidence Coverage is completely decoupled from the downstream target.

Revised H2:
    "Evidence Coverage has a stronger association with downstream answer quality
     than conventional retrieval quality alone."

Controls & Safeguards:
    - Strictly OFFLINE statistical reanalysis: zero new LLM or evaluator calls.
    - Uses only the frozen evaluator results (885 primary conditions, 59 grade-2 queries).
    - Excluded queries (Q007, Q013, Q017, Q021, Q055, Q061) remain strictly omitted.
    - Does NOT modify historical experimental artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr, wilcoxon

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_REFERENCE_EVIDENCE_PATH = ROOT / "data" / "evaluation" / "reference_evidence.json"
DEFAULT_RETRIEVAL_METRICS_PATH = ROOT / "experiments" / "results" / "stage3_retrieval_metrics.json"
DEFAULT_EVALUATOR_RESULTS_PATH = ROOT / "experiments" / "results" / "stage3_evaluator_results.jsonl"

DEFAULT_H2_CORRECTED_JSON = ROOT / "experiments" / "results" / "stage3_h2_corrected.json"
DEFAULT_H2_CORRECTED_CSV = ROOT / "experiments" / "results" / "stage3_h2_corrected.csv"
DEFAULT_H2_BOOTSTRAP_CSV = ROOT / "experiments" / "results" / "stage3_h2_bootstrap.csv"
DEFAULT_DAQ_SUMMARY_CSV = ROOT / "experiments" / "results" / "stage3_daq_configuration_summary.csv"

SENSITIVITY_QUERIES = {"Q007", "Q013", "Q017", "Q021", "Q055", "Q061"}
EXPECTED_RETRIEVERS = ["bm25", "dense", "lsh", "bm25_dense_rrf", "bm25_dense_lsh_rrf"]
EXPECTED_K_VALUES = [5, 10, 20]
RANDOM_SEED = 42


def load_and_compute_daq(
    reference_evidence_path: Path | str = DEFAULT_REFERENCE_EVIDENCE_PATH,
    retrieval_metrics_path: Path | str = DEFAULT_RETRIEVAL_METRICS_PATH,
    evaluator_results_path: Path | str = DEFAULT_EVALUATOR_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Load frozen retrieval and evaluator artifacts, perform strict inner join,
    and compute Downstream Answer Quality (DAQ) = (Factual Consistency + Answer Relevance) / 2.0.

    Ensures:
      - 59 grade-2 primary queries only (6 sensitivity queries strictly excluded)
      - Exactly 885 conditions joined on (query_id, retriever, k)
      - DAQ is finite and in [0.0, 1.0]
      - Evidence Coverage is completely decoupled from DAQ
    """
    ref_p = Path(reference_evidence_path)
    ret_p = Path(retrieval_metrics_path)
    eval_p = Path(evaluator_results_path)

    # 1. Load reference evidence
    with open(ref_p, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    grade2_queries = {q["query_id"] for q in ref_data.get("queries", []) if q.get("grade_2_available", True)}
    assert grade2_queries.isdisjoint(SENSITIVITY_QUERIES), "Grade-2 set contains sensitivity queries!"
    assert len(grade2_queries) == 59, f"Expected 59 grade-2 queries, found {len(grade2_queries)}"

    # 2. Load retrieval metrics
    with open(ret_p, "r", encoding="utf-8") as f:
        ret_data = json.load(f)
    pq_metrics = ret_data.get("per_query_metrics", [])
    ret_rows = [r for r in pq_metrics if r["query_id"] in grade2_queries]

    # 3. Load evaluator results (filtering on status == 'success')
    eval_rows = []
    with open(eval_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "success" and rec["query_id"] in grade2_queries:
                eval_rows.append(rec)

    # Deduplicate by key taking latest success
    eval_map = {(r["query_id"], r["retriever"], int(r["k"])): r for r in eval_rows}

    df_ret = pd.DataFrame(ret_rows)
    df_eval = pd.DataFrame([
        {
            "query_id": r["query_id"],
            "retriever": r["retriever"],
            "k": int(r["k"]),
            "evidence_coverage": float(r["scores"]["evidence_coverage"]),
            "factual_consistency": float(r["scores"]["factual_consistency"]),
            "answer_relevance": float(r["scores"]["answer_relevance"]),
            "composite_quality": float(r["scores"]["composite_quality"]),  # preserved for historical comparison
        }
        for r in eval_map.values()
    ])

    df = pd.merge(df_ret, df_eval, on=["query_id", "retriever", "k"], how="inner").copy()

    # Compute DAQ
    fc = df["factual_consistency"].to_numpy()
    ar = df["answer_relevance"].to_numpy()
    df = df.assign(daq=(fc + ar) / 2.0)

    # Integrity assertions
    assert len(df) == 885, f"Expected 885 rows, got {len(df)}"
    assert len(df[["query_id", "retriever", "k"]].drop_duplicates()) == 885, "Duplicate keys detected!"
    assert set(df["query_id"].unique()) == grade2_queries, "Mismatch in query IDs!"
    assert set(df["retriever"].unique()) == set(EXPECTED_RETRIEVERS), "Mismatch in retrievers!"
    assert set(df["k"].unique()) == set(EXPECTED_K_VALUES), "Mismatch in K values!"
    assert df["daq"].notna().all(), "DAQ contains null/NaN values!"
    assert ((df["daq"] >= 0.0) & (df["daq"] <= 1.0)).all(), "DAQ values out of [0, 1] range!"

    return df


def bootstrap_ci(
    data: np.ndarray,
    stat_func: Any = np.mean,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """Calculate percentile bootstrap confidence interval for a 1D array."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_func(sample)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    return round(lower, 4), round(upper, 4)


def compute_daq_configuration_summary(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compute configuration-level summaries across the 15 configurations using DAQ.
    Preserves old Composite Quality strictly for provenance/comparison.
    """
    rows = []
    for ret in EXPECTED_RETRIEVERS:
        for k in EXPECTED_K_VALUES:
            sub = df[(df["retriever"] == ret) & (df["k"] == k)].copy()
            assert len(sub) == 59, f"Expected 59 queries for ({ret}, {k}), got {len(sub)}"

            daq_vals = sub["daq"].to_numpy()
            daq_ci_low, daq_ci_high = bootstrap_ci(daq_vals, np.mean, n_bootstrap, seed=seed)

            rows.append({
                "retriever": ret,
                "k": k,
                "n_queries": 59,
                "ndcg_mean": round(float(sub["ndcg_at_k"].mean()), 4),
                "ndcg_std": round(float(sub["ndcg_at_k"].std()), 4),
                "coverage_mean": round(float(sub["evidence_coverage"].mean()), 4),
                "coverage_std": round(float(sub["evidence_coverage"].std()), 4),
                "factual_consistency_mean": round(float(sub["factual_consistency"].mean()), 4),
                "factual_consistency_std": round(float(sub["factual_consistency"].std()), 4),
                "answer_relevance_mean": round(float(sub["answer_relevance"].mean()), 4),
                "answer_relevance_std": round(float(sub["answer_relevance"].std()), 4),
                "daq_mean": round(float(sub["daq"].mean()), 4),
                "daq_median": round(float(sub["daq"].median()), 4),
                "daq_std": round(float(sub["daq"].std()), 4),
                "daq_ci_lower": daq_ci_low,
                "daq_ci_upper": daq_ci_high,
                "old_composite_mean": round(float(sub["composite_quality"].mean()), 4),
            })
    return pd.DataFrame(rows)


def compute_h2_corrected_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute corrected query-level Spearman rank correlations for revised H2:
      - rho(Evidence Coverage, DAQ)
      - rho(NDCG@K, DAQ)
      - Delta = rho(Coverage, DAQ) - rho(NDCG@K, DAQ)
      - Dimension-level breakdown: Coverage vs FC, Coverage vs AR, NDCG vs FC, NDCG vs AR.
    """
    rows = []
    for ret in EXPECTED_RETRIEVERS:
        for k in EXPECTED_K_VALUES:
            sub = df[(df["retriever"] == ret) & (df["k"] == k)]
            assert len(sub) == 59

            cov = sub["evidence_coverage"].to_numpy()
            ndcg = sub["ndcg_at_k"].to_numpy()
            daq = sub["daq"].to_numpy()
            fc = sub["factual_consistency"].to_numpy()
            ar = sub["answer_relevance"].to_numpy()

            # Primary H2 comparison
            res_cov_daq = spearmanr(cov, daq)
            res_ndcg_daq = spearmanr(ndcg, daq)
            r_cov_daq, p_cov_daq = float(res_cov_daq.statistic), float(res_cov_daq.pvalue)
            r_ndcg_daq, p_ndcg_daq = float(res_ndcg_daq.statistic), float(res_ndcg_daq.pvalue)
            delta = r_cov_daq - r_ndcg_daq

            # Dimension-level correlations
            res_cov_fc = spearmanr(cov, fc)
            res_cov_ar = spearmanr(cov, ar)
            res_ndcg_fc = spearmanr(ndcg, fc)
            res_ndcg_ar = spearmanr(ndcg, ar)

            rows.append({
                "retriever": ret,
                "k": k,
                "n_queries": 59,
                "rho_coverage_daq": round(r_cov_daq, 4),
                "pvalue_coverage_daq": round(p_cov_daq, 5),
                "rho_ndcg_daq": round(r_ndcg_daq, 4),
                "pvalue_ndcg_daq": round(p_ndcg_daq, 5),
                "delta": round(delta, 4),
                "rho_coverage_fc": round(float(res_cov_fc.statistic), 4),
                "pvalue_coverage_fc": round(float(res_cov_fc.pvalue), 5),
                "rho_coverage_ar": round(float(res_cov_ar.statistic), 4),
                "pvalue_coverage_ar": round(float(res_cov_ar.pvalue), 5),
                "rho_ndcg_fc": round(float(res_ndcg_fc.statistic), 4),
                "pvalue_ndcg_fc": round(float(res_ndcg_fc.pvalue), 5),
                "rho_ndcg_ar": round(float(res_ndcg_ar.statistic), 4),
                "pvalue_ndcg_ar": round(float(res_ndcg_ar.pvalue), 5),
            })
    return pd.DataFrame(rows)


def compute_h2_corrected_bootstrap(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Perform paired query-level bootstrap comparison for revised H2:
    Delta = rho(Coverage, DAQ) - rho(NDCG@K, DAQ) across 2,000 resamples.

    Reports:
      - observed Delta
      - bootstrap mean Delta
      - bootstrap median Delta
      - 95% percentile CI
      - P(Delta > 0) [bootstrap directional probability]
      - excludes_zero [Boolean]
    """
    rng = np.random.default_rng(seed)
    rows = []

    for ret in EXPECTED_RETRIEVERS:
        for k in EXPECTED_K_VALUES:
            sub = df[(df["retriever"] == ret) & (df["k"] == k)].reset_index(drop=True)
            n_queries = len(sub)

            cov = sub["evidence_coverage"].to_numpy()
            ndcg = sub["ndcg_at_k"].to_numpy()
            daq = sub["daq"].to_numpy()

            obs_r_cov = float(spearmanr(cov, daq).statistic)
            obs_r_ndcg = float(spearmanr(ndcg, daq).statistic)
            obs_delta = obs_r_cov - obs_r_ndcg

            boot_deltas = np.empty(n_bootstrap)
            boot_r_cov = np.empty(n_bootstrap)
            boot_r_ndcg = np.empty(n_bootstrap)

            for b in range(n_bootstrap):
                idx = rng.choice(n_queries, size=n_queries, replace=True)
                b_cov = cov[idx]
                b_ndcg = ndcg[idx]
                b_daq = daq[idx]

                r_c = spearmanr(b_cov, b_daq).statistic if (np.std(b_cov) > 0 and np.std(b_daq) > 0) else 0.0
                r_n = spearmanr(b_ndcg, b_daq).statistic if (np.std(b_ndcg) > 0 and np.std(b_daq) > 0) else 0.0

                boot_r_cov[b] = r_c
                boot_r_ndcg[b] = r_n
                boot_deltas[b] = r_c - r_n

            boot_mean_delta = float(np.mean(boot_deltas))
            boot_med_delta = float(np.median(boot_deltas))
            ci_low = float(np.percentile(boot_deltas, 2.5))
            ci_high = float(np.percentile(boot_deltas, 97.5))
            prop_positive = float(np.mean(boot_deltas > 0))
            excludes_zero = bool(ci_low > 0)

            rows.append({
                "retriever": ret,
                "k": k,
                "n_queries": 59,
                "rho_coverage_daq": round(obs_r_cov, 4),
                "rho_ndcg_daq": round(obs_r_ndcg, 4),
                "observed_delta": round(obs_delta, 4),
                "bootstrap_mean_delta": round(boot_mean_delta, 4),
                "bootstrap_median_delta": round(boot_med_delta, 4),
                "delta_ci_lower": round(ci_low, 4),
                "delta_ci_upper": round(ci_high, 4),
                "p_bootstrap_delta_positive": round(prop_positive, 4),
                "ci_excludes_zero": excludes_zero,
                "h2_supported": bool(obs_delta > 0 and excludes_zero),
            })
    return pd.DataFrame(rows)


def compute_daq_system_level_analysis(config_summary: pd.DataFrame) -> Dict[str, Any]:
    """
    Configuration-level exploratory analysis of DAQ across all 15 configurations
    and non-LSH 12 configurations.
    """
    # 1. All configurations (n=15)
    r_ndcg_all, p_ndcg_all = spearmanr(config_summary["ndcg_mean"], config_summary["daq_mean"])
    r_cov_all, p_cov_all = spearmanr(config_summary["coverage_mean"], config_summary["daq_mean"])

    # 2. Non-LSH configurations (n=12)
    non_lsh = config_summary[config_summary["retriever"] != "lsh"].copy()
    r_ndcg_no, p_ndcg_no = spearmanr(non_lsh["ndcg_mean"], non_lsh["daq_mean"])
    r_cov_no, p_cov_no = spearmanr(non_lsh["coverage_mean"], non_lsh["daq_mean"])

    return {
        "all_configurations": {
            "n_configurations": 15,
            "ndcg_vs_daq": {"rho": round(float(r_ndcg_all), 4), "pvalue": round(float(p_ndcg_all), 5)},
            "coverage_vs_daq": {"rho": round(float(r_cov_all), 4), "pvalue": round(float(p_cov_all), 5)},
        },
        "non_lsh_configurations": {
            "n_configurations": 12,
            "ndcg_vs_daq": {"rho": round(float(r_ndcg_no), 4), "pvalue": round(float(p_ndcg_no), 5)},
            "coverage_vs_daq": {"rho": round(float(r_cov_no), 4), "pvalue": round(float(p_cov_no), 5)},
        },
    }


def compute_bm25_dense_daq_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reanalyze BM25 vs Dense using DAQ at K=5, 10, 20 with paired Wilcoxon tests.
    """
    records = []
    for k in EXPECTED_K_VALUES:
        sub_bm25 = df[(df["retriever"] == "bm25") & (df["k"] == k)].sort_values("query_id").reset_index(drop=True)
        sub_dense = df[(df["retriever"] == "dense") & (df["k"] == k)].sort_values("query_id").reset_index(drop=True)

        diff_cov = sub_bm25["evidence_coverage"].to_numpy() - sub_dense["evidence_coverage"].to_numpy()
        diff_fc = sub_bm25["factual_consistency"].to_numpy() - sub_dense["factual_consistency"].to_numpy()
        diff_ar = sub_bm25["answer_relevance"].to_numpy() - sub_dense["answer_relevance"].to_numpy()
        diff_daq = sub_bm25["daq"].to_numpy() - sub_dense["daq"].to_numpy()

        res_daq = wilcoxon(diff_daq, alternative="two-sided") if np.any(diff_daq != 0) else None
        p_daq = float(res_daq.pvalue) if res_daq else 1.0
        w_daq = float(res_daq.statistic) if res_daq else 0.0

        records.append({
            "k": k,
            "n_queries": 59,
            "bm25_coverage_mean": round(float(sub_bm25["evidence_coverage"].mean()), 4),
            "dense_coverage_mean": round(float(sub_dense["evidence_coverage"].mean()), 4),
            "mean_delta_coverage": round(float(np.mean(diff_cov)), 4),
            "bm25_fc_mean": round(float(sub_bm25["factual_consistency"].mean()), 4),
            "dense_fc_mean": round(float(sub_dense["factual_consistency"].mean()), 4),
            "mean_delta_fc": round(float(np.mean(diff_fc)), 4),
            "bm25_ar_mean": round(float(sub_bm25["answer_relevance"].mean()), 4),
            "dense_ar_mean": round(float(sub_dense["answer_relevance"].mean()), 4),
            "mean_delta_ar": round(float(np.mean(diff_ar)), 4),
            "bm25_daq_mean": round(float(sub_bm25["daq"].mean()), 4),
            "dense_daq_mean": round(float(sub_dense["daq"].mean()), 4),
            "mean_delta_daq": round(float(np.mean(diff_daq)), 4),
            "wilcoxon_stat_daq": round(w_daq, 2),
            "pvalue_daq": round(p_daq, 5),
        })
    return pd.DataFrame(records)


def run_h2_reanalysis(
    reference_evidence_path: Path | str = DEFAULT_REFERENCE_EVIDENCE_PATH,
    retrieval_metrics_path: Path | str = DEFAULT_RETRIEVAL_METRICS_PATH,
    evaluator_results_path: Path | str = DEFAULT_EVALUATOR_RESULTS_PATH,
    output_json_path: Path | str = DEFAULT_H2_CORRECTED_JSON,
    output_correlations_csv: Path | str = DEFAULT_H2_CORRECTED_CSV,
    output_bootstrap_csv: Path | str = DEFAULT_H2_BOOTSTRAP_CSV,
    output_daq_summary_csv: Path | str = DEFAULT_DAQ_SUMMARY_CSV,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute complete Task 4F-C H2 correction and save all new artifacts."""
    logger.info("Loading data and computing DAQ...")
    df = load_and_compute_daq(reference_evidence_path, retrieval_metrics_path, evaluator_results_path)

    logger.info("Computing DAQ configuration summary...")
    daq_summary = compute_daq_configuration_summary(df, n_bootstrap=n_bootstrap, seed=seed)

    logger.info("Computing corrected H2 query-level correlations...")
    h2_corrs = compute_h2_corrected_correlations(df)

    logger.info("Computing corrected H2 bootstrap comparisons (2000 iterations)...")
    h2_boot = compute_h2_corrected_bootstrap(df, n_bootstrap=n_bootstrap, seed=seed)

    logger.info("Computing configuration-level exploratory analysis...")
    sys_analysis = compute_daq_system_level_analysis(daq_summary)

    logger.info("Computing BM25 vs Dense DAQ comparisons...")
    bm25_dense = compute_bm25_dense_daq_comparisons(df)

    # Export CSVs
    daq_summary.to_csv(output_daq_summary_csv, index=False)
    h2_corrs.to_csv(output_correlations_csv, index=False)
    h2_boot.to_csv(output_bootstrap_csv, index=False)

    # Master JSON output
    out_json = {
        "metadata": {
            "task": "Task 4F-C H2 Methodological Correction and Reanalysis",
            "study": "Cardiology-Chat Cardiovascular RAG Research Study",
            "revised_h2_statement": "Evidence Coverage has a stronger association with downstream answer quality than conventional retrieval quality alone.",
            "downstream_answer_quality_definition": "DAQ = (Factual Consistency + Answer Relevance) / 2.0",
            "evidence_coverage_excluded_from_daq": True,
            "n_primary_queries": 59,
            "n_configurations": 15,
            "n_conditions": 885,
            "random_seed": seed,
            "n_bootstrap_iterations": n_bootstrap,
            "claims_restricted_to_automated_quality": True,
        },
        "daq_configuration_summary": daq_summary.to_dict(orient="records"),
        "h2_corrected_correlations": h2_corrs.to_dict(orient="records"),
        "h2_corrected_bootstrap": h2_boot.to_dict(orient="records"),
        "system_level_exploratory": sys_analysis,
        "bm25_vs_dense_daq_comparisons": bm25_dense.to_dict(orient="records"),
    }

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    logger.info("Task 4F-C H2 reanalysis complete. Artifacts written successfully.")
    return out_json
