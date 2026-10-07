"""
statistical_analysis.py
------------------------
Task 4F: Frozen Results Statistical Analysis Pipeline for Cardiology-Chat.

Primary Research Question:
    "How well do conventional retrieval metrics predict evidence sufficiency
     and downstream answer quality in cardiovascular RAG?"

Methodological Controls & Constraints:
    - Primary population: 59 queries with grade-2 reference evidence.
    - Sensitivity queries (Q007, Q013, Q017, Q021, Q055, Q061) are strictly EXCLUDED.
    - 5 retrievers: bm25, dense, lsh, bm25_dense_rrf, bm25_dense_lsh_rrf.
    - 3 cutoffs: K in {5, 10, 20}.
    - 15 configurations, 885 primary conditions joined on (query_id, retriever, k).
    - Dense model: sentence-transformers/all-MiniLM-L6-v2 (NOT PubMedBERT).
    - Evaluator scores measure AUTOMATED ANSWER QUALITY against provided reference evidence,
      NOT clinical correctness, clinical validity, or clinical safety.
    - Discloses shared-reference dependency between retrieval metrics and evidence coverage.
    - Analysis is strictly non-destructive: zero modification of frozen artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr, wilcoxon

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_REFERENCE_EVIDENCE_PATH = ROOT / "data" / "evaluation" / "reference_evidence.json"
DEFAULT_RETRIEVAL_METRICS_PATH = ROOT / "experiments" / "results" / "stage3_retrieval_metrics.json"
DEFAULT_EVALUATOR_RESULTS_PATH = ROOT / "experiments" / "results" / "stage3_evaluator_results.jsonl"
DEFAULT_FIGURES_DIR = ROOT / "experiments" / "results" / "stage3_figures"
DEFAULT_OUTPUT_JSON = ROOT / "experiments" / "results" / "stage3_statistical_analysis.json"
DEFAULT_CONFIG_SUMMARY_CSV = ROOT / "experiments" / "results" / "stage3_configuration_summary.csv"
DEFAULT_QUERY_CORR_CSV = ROOT / "experiments" / "results" / "stage3_query_level_correlations.csv"
DEFAULT_PAIRWISE_CSV = ROOT / "experiments" / "results" / "stage3_pairwise_comparisons.csv"

SENSITIVITY_QUERIES = {"Q007", "Q013", "Q017", "Q021", "Q055", "Q061"}
EXPECTED_RETRIEVERS = ["bm25", "dense", "lsh", "bm25_dense_rrf", "bm25_dense_lsh_rrf"]
EXPECTED_K_VALUES = [5, 10, 20]
RANDOM_SEED = 42


def load_and_join_data(
    reference_evidence_path: Path | str = DEFAULT_REFERENCE_EVIDENCE_PATH,
    retrieval_metrics_path: Path | str = DEFAULT_RETRIEVAL_METRICS_PATH,
    evaluator_results_path: Path | str = DEFAULT_EVALUATOR_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Load frozen retrieval and evaluator artifacts and perform strict inner join on (query_id, retriever, k).

    Enforces:
      - 59 grade-2 primary queries only
      - Strict exclusion of 6 sensitivity queries
      - Successful evaluator records only (ignoring historical Pass-1 error entries)
      - Exact 885-row population with 885 unique condition keys
      - No null, NaN, or infinite values in joined metrics
    """
    ref_p = Path(reference_evidence_path)
    ret_p = Path(retrieval_metrics_path)
    eval_p = Path(evaluator_results_path)

    # 1. Load reference evidence
    with open(ref_p, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    grade2_queries = {q["query_id"] for q in ref_data.get("queries", []) if q.get("grade_2_available", True)}

    # Ensure sensitivity queries are excluded
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

    # Deduplicate evaluator records by key if needed (taking latest success)
    eval_map = {(r["query_id"], r["retriever"], int(r["k"])): r for r in eval_rows}

    # Format into tabular rows
    df_ret = pd.DataFrame(ret_rows)
    df_eval = pd.DataFrame([
        {
            "query_id": r["query_id"],
            "retriever": r["retriever"],
            "k": int(r["k"]),
            "evidence_coverage": float(r["scores"]["evidence_coverage"]),
            "factual_consistency": float(r["scores"]["factual_consistency"]),
            "answer_relevance": float(r["scores"]["answer_relevance"]),
            "composite_quality": float(r["scores"]["composite_quality"]),
            "evaluator_latency": float(r.get("latency_seconds", 0.0)),
        }
        for r in eval_map.values()
    ])

    # Inner join
    df = pd.merge(df_ret, df_eval, on=["query_id", "retriever", "k"], how="inner")

    # Integrity assertions
    assert len(df) == 885, f"Expected exactly 885 joined rows, got {len(df)}"
    assert len(df[["query_id", "retriever", "k"]].drop_duplicates()) == 885, "Duplicate keys in joined dataset!"
    assert set(df["query_id"].unique()) == grade2_queries, "Mismatch in query IDs!"
    assert set(df["retriever"].unique()) == set(EXPECTED_RETRIEVERS), "Mismatch in retrievers!"
    assert set(df["k"].unique()) == set(EXPECTED_K_VALUES), "Mismatch in K values!"
    assert df.isna().sum().sum() == 0, "Joined dataset contains missing/NaN values!"

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


def compute_configuration_summaries(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compute retrieval and downstream summary statistics for each of the 15 configurations.
    Includes means, medians, standard deviations, and 95% bootstrap CIs for downstream metrics.
    """
    configs = []
    for ret in EXPECTED_RETRIEVERS:
        for k in EXPECTED_K_VALUES:
            sub = df[(df["retriever"] == ret) & (df["k"] == k)].copy()
            n_queries = len(sub)
            assert n_queries == 59, f"Config ({ret}, {k}) has {n_queries} queries, expected 59"

            # Downstream bootstrap CIs
            cov_vals = sub["evidence_coverage"].to_numpy()
            con_vals = sub["factual_consistency"].to_numpy()
            rel_vals = sub["answer_relevance"].to_numpy()
            comp_vals = sub["composite_quality"].to_numpy()

            cov_ci_low, cov_ci_high = bootstrap_ci(cov_vals, np.mean, n_bootstrap, seed=seed)
            con_ci_low, con_ci_high = bootstrap_ci(con_vals, np.mean, n_bootstrap, seed=seed)
            rel_ci_low, rel_ci_high = bootstrap_ci(rel_vals, np.mean, n_bootstrap, seed=seed)
            comp_ci_low, comp_ci_high = bootstrap_ci(comp_vals, np.mean, n_bootstrap, seed=seed)

            row = {
                "retriever": ret,
                "k": k,
                "n_queries": n_queries,
                # Retrieval metrics
                "ndcg_mean": round(float(sub["ndcg_at_k"].mean()), 4),
                "ndcg_std": round(float(sub["ndcg_at_k"].std()), 4),
                "precision_mean": round(float(sub["precision_at_k"].mean()), 4),
                "precision_std": round(float(sub["precision_at_k"].std()), 4),
                "recall_mean": round(float(sub["recall_at_k"].mean()), 4),
                "recall_std": round(float(sub["recall_at_k"].std()), 4),
                "mrr_mean": round(float(sub["reciprocal_rank"].mean()), 4),
                "mrr_std": round(float(sub["reciprocal_rank"].std()), 4),
                "map_mean": round(float(sub["average_precision"].mean()), 4),
                "map_std": round(float(sub["average_precision"].std()), 4),
                # Downstream metrics: Evidence Coverage
                "coverage_mean": round(float(sub["evidence_coverage"].mean()), 4),
                "coverage_median": round(float(sub["evidence_coverage"].median()), 4),
                "coverage_std": round(float(sub["evidence_coverage"].std()), 4),
                "coverage_ci_lower": cov_ci_low,
                "coverage_ci_upper": cov_ci_high,
                # Downstream metrics: Factual Consistency
                "consistency_mean": round(float(sub["factual_consistency"].mean()), 4),
                "consistency_median": round(float(sub["factual_consistency"].median()), 4),
                "consistency_std": round(float(sub["factual_consistency"].std()), 4),
                "consistency_ci_lower": con_ci_low,
                "consistency_ci_upper": con_ci_high,
                # Downstream metrics: Answer Relevance
                "relevance_mean": round(float(sub["answer_relevance"].mean()), 4),
                "relevance_median": round(float(sub["answer_relevance"].median()), 4),
                "relevance_std": round(float(sub["answer_relevance"].std()), 4),
                "relevance_ci_lower": rel_ci_low,
                "relevance_ci_upper": rel_ci_high,
                # Downstream metrics: Composite Quality
                "composite_mean": round(float(sub["composite_quality"].mean()), 4),
                "composite_median": round(float(sub["composite_quality"].median()), 4),
                "composite_std": round(float(sub["composite_quality"].std()), 4),
                "composite_ci_lower": comp_ci_low,
                "composite_ci_upper": comp_ci_high,
            }
            configs.append(row)

    return pd.DataFrame(configs)


def compute_query_level_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute query-level Spearman rank correlations within each of the 15 configurations.
    Evaluates:
      - Pair A: NDCG@K vs Evidence Coverage (Primary H1)
      - Pair B: NDCG@K vs Composite Quality
      - Pair C: Evidence Coverage vs Composite Quality (Primary H2)
      - Pair D: Precision@K vs Evidence Coverage & Composite Quality
      - Pair E: Recall@K vs Evidence Coverage & Composite Quality
      - Pair F: MRR vs Evidence Coverage & Composite Quality
      - Pair G: MAP vs Evidence Coverage & Composite Quality
      - Secondary: NDCG@K vs Factual Consistency & Answer Relevance
    """
    results = []
    pairs = [
        ("ndcg_vs_coverage", "ndcg_at_k", "evidence_coverage"),
        ("ndcg_vs_composite", "ndcg_at_k", "composite_quality"),
        ("coverage_vs_composite", "evidence_coverage", "composite_quality"),
        ("precision_vs_coverage", "precision_at_k", "evidence_coverage"),
        ("precision_vs_composite", "precision_at_k", "composite_quality"),
        ("recall_vs_coverage", "recall_at_k", "evidence_coverage"),
        ("recall_vs_composite", "recall_at_k", "composite_quality"),
        ("mrr_vs_coverage", "reciprocal_rank", "evidence_coverage"),
        ("mrr_vs_composite", "reciprocal_rank", "composite_quality"),
        ("map_vs_coverage", "average_precision", "evidence_coverage"),
        ("map_vs_composite", "average_precision", "composite_quality"),
        ("ndcg_vs_consistency", "ndcg_at_k", "factual_consistency"),
        ("ndcg_vs_relevance", "ndcg_at_k", "answer_relevance"),
    ]

    for ret in EXPECTED_RETRIEVERS:
        for k in EXPECTED_K_VALUES:
            sub = df[(df["retriever"] == ret) & (df["k"] == k)]
            row = {"retriever": ret, "k": k, "n_queries": len(sub)}

            for pair_name, col_x, col_y in pairs:
                x = sub[col_x].to_numpy()
                y = sub[col_y].to_numpy()
                # Check for variance to avoid degenerate spearmanr warning
                if np.std(x) == 0 or np.std(y) == 0:
                    rho, p_val = 0.0, 1.0
                else:
                    res = spearmanr(x, y)
                    rho, p_val = float(res.statistic), float(res.pvalue)

                row[f"{pair_name}_rho"] = round(rho, 4)
                row[f"{pair_name}_pvalue"] = round(p_val, 5)

            results.append(row)

    return pd.DataFrame(results)


def compute_h2_bootstrap_comparisons(
    df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Perform paired query-level bootstrap comparison for H2:
    Does Evidence Coverage have a stronger association with Composite Quality than NDCG@K?

    For each configuration:
      1. Resample the 59 query IDs with replacement.
      2. Compute rho(Coverage, Composite) and rho(NDCG@K, Composite).
      3. Compute difference Delta = rho(Coverage, Composite) - rho(NDCG@K, Composite).
      4. Compute observed Delta, 95% bootstrap CI, and proportion of Delta > 0.
    """
    rng = np.random.default_rng(seed)
    h2_records = []

    for ret in EXPECTED_RETRIEVERS:
        for k in EXPECTED_K_VALUES:
            sub = df[(df["retriever"] == ret) & (df["k"] == k)].copy().reset_index(drop=True)
            n_queries = len(sub)

            cov = sub["evidence_coverage"].to_numpy()
            ndcg = sub["ndcg_at_k"].to_numpy()
            comp = sub["composite_quality"].to_numpy()

            # Observed correlations and difference
            obs_rho_cov = float(spearmanr(cov, comp).statistic)
            obs_rho_ndcg = float(spearmanr(ndcg, comp).statistic)
            obs_delta = obs_rho_cov - obs_rho_ndcg

            boot_deltas = np.empty(n_bootstrap)
            boot_rhos_cov = np.empty(n_bootstrap)
            boot_rhos_ndcg = np.empty(n_bootstrap)

            for b in range(n_bootstrap):
                idx = rng.choice(n_queries, size=n_queries, replace=True)
                b_cov = cov[idx]
                b_ndcg = ndcg[idx]
                b_comp = comp[idx]

                r_cov = spearmanr(b_cov, b_comp).statistic if (np.std(b_cov) > 0 and np.std(b_comp) > 0) else 0.0
                r_ndcg = spearmanr(b_ndcg, b_comp).statistic if (np.std(b_ndcg) > 0 and np.std(b_comp) > 0) else 0.0

                boot_rhos_cov[b] = r_cov
                boot_rhos_ndcg[b] = r_ndcg
                boot_deltas[b] = r_cov - r_ndcg

            ci_low = float(np.percentile(boot_deltas, 2.5))
            ci_high = float(np.percentile(boot_deltas, 97.5))
            prop_pos = float(np.mean(boot_deltas > 0))

            h2_records.append({
                "retriever": ret,
                "k": k,
                "n_queries": n_queries,
                "rho_coverage_composite": round(obs_rho_cov, 4),
                "rho_ndcg_composite": round(obs_rho_ndcg, 4),
                "observed_delta": round(obs_delta, 4),
                "delta_ci_lower": round(ci_low, 4),
                "delta_ci_upper": round(ci_high, 4),
                "prop_bootstrap_delta_positive": round(prop_pos, 4),
                "h2_supported": bool(obs_delta > 0 and ci_low > 0),
            })

    return pd.DataFrame(h2_records)


def compute_system_level_analysis(config_summary: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform exploratory system-level (configuration-level) correlation analysis across the 15 configurations.
    Also computes non-LSH robustness analysis across 12 configurations.
    """
    # 1. All-system (n=15)
    r_ndcg_cov, p_ndcg_cov = spearmanr(config_summary["ndcg_mean"], config_summary["coverage_mean"])
    r_ndcg_comp, p_ndcg_comp = spearmanr(config_summary["ndcg_mean"], config_summary["composite_mean"])
    r_cov_comp, p_cov_comp = spearmanr(config_summary["coverage_mean"], config_summary["composite_mean"])
    r_rec_cov, p_rec_cov = spearmanr(config_summary["recall_mean"], config_summary["coverage_mean"])
    r_rec_comp, p_rec_comp = spearmanr(config_summary["recall_mean"], config_summary["composite_mean"])

    # 2. Non-LSH robustness (n=12)
    non_lsh = config_summary[config_summary["retriever"] != "lsh"].copy()
    r_ndcg_cov_no_lsh, p_ndcg_cov_no_lsh = spearmanr(non_lsh["ndcg_mean"], non_lsh["coverage_mean"])
    r_ndcg_comp_no_lsh, p_ndcg_comp_no_lsh = spearmanr(non_lsh["ndcg_mean"], non_lsh["composite_mean"])
    r_cov_comp_no_lsh, p_cov_comp_no_lsh = spearmanr(non_lsh["coverage_mean"], non_lsh["composite_mean"])
    r_rec_cov_no_lsh, p_rec_cov_no_lsh = spearmanr(non_lsh["recall_mean"], non_lsh["coverage_mean"])
    r_rec_comp_no_lsh, p_rec_comp_no_lsh = spearmanr(non_lsh["recall_mean"], non_lsh["composite_mean"])

    return {
        "all_configurations": {
            "n_configurations": 15,
            "ndcg_vs_coverage": {"rho": round(float(r_ndcg_cov), 4), "pvalue": round(float(p_ndcg_cov), 5)},
            "ndcg_vs_composite": {"rho": round(float(r_ndcg_comp), 4), "pvalue": round(float(p_ndcg_comp), 5)},
            "coverage_vs_composite": {"rho": round(float(r_cov_comp), 4), "pvalue": round(float(p_cov_comp), 5)},
            "recall_vs_coverage": {"rho": round(float(r_rec_cov), 4), "pvalue": round(float(p_rec_cov), 5)},
            "recall_vs_composite": {"rho": round(float(r_rec_comp), 4), "pvalue": round(float(p_rec_comp), 5)},
        },
        "non_lsh_configurations": {
            "n_configurations": 12,
            "ndcg_vs_coverage": {"rho": round(float(r_ndcg_cov_no_lsh), 4), "pvalue": round(float(p_ndcg_cov_no_lsh), 5)},
            "ndcg_vs_composite": {"rho": round(float(r_ndcg_comp_no_lsh), 4), "pvalue": round(float(p_ndcg_comp_no_lsh), 5)},
            "coverage_vs_composite": {"rho": round(float(r_cov_comp_no_lsh), 4), "pvalue": round(float(p_cov_comp_no_lsh), 5)},
            "recall_vs_coverage": {"rho": round(float(r_rec_cov_no_lsh), 4), "pvalue": round(float(p_rec_cov_no_lsh), 5)},
            "recall_vs_composite": {"rho": round(float(r_rec_comp_no_lsh), 4), "pvalue": round(float(p_rec_comp_no_lsh), 5)},
        },
    }


def compute_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform paired nonparametric tests (Wilcoxon signed-rank) across retrievers at fixed K.
    Accounts for repeated-measures design across the 59 queries.
    Applies Benjamini-Hochberg FDR correction.
    """
    comparisons = [
        ("bm25", "dense"),
        ("bm25", "bm25_dense_rrf"),
        ("dense", "bm25_dense_rrf"),
        ("bm25_dense_rrf", "bm25_dense_lsh_rrf"),
        ("bm25", "lsh"),
    ]

    records = []
    pvals_cov = []
    pvals_comp = []

    for k in EXPECTED_K_VALUES:
        for sys1, sys2 in comparisons:
            sub1 = df[(df["retriever"] == sys1) & (df["k"] == k)].sort_values("query_id").reset_index(drop=True)
            sub2 = df[(df["retriever"] == sys2) & (df["k"] == k)].sort_values("query_id").reset_index(drop=True)

            assert len(sub1) == 59 and len(sub2) == 59, "Mismatch in query count for paired test"

            # Difference: sys1 - sys2
            diff_cov = sub1["evidence_coverage"].to_numpy() - sub2["evidence_coverage"].to_numpy()
            diff_comp = sub1["composite_quality"].to_numpy() - sub2["composite_quality"].to_numpy()

            # Wilcoxon signed-rank test
            # If all differences are zero, p-value is 1.0
            if np.all(diff_cov == 0):
                w_stat_cov, p_cov = 0.0, 1.0
            else:
                res_cov = wilcoxon(diff_cov, alternative="two-sided")
                w_stat_cov, p_cov = float(res_cov.statistic), float(res_cov.pvalue)

            if np.all(diff_comp == 0):
                w_stat_comp, p_comp = 0.0, 1.0
            else:
                res_comp = wilcoxon(diff_comp, alternative="two-sided")
                w_stat_comp, p_comp = float(res_comp.statistic), float(res_comp.pvalue)

            # Effect size r = Z / sqrt(N), using normal approximation rank-biserial
            n_nonzero_cov = np.count_nonzero(diff_cov)
            n_nonzero_comp = np.count_nonzero(diff_comp)
            r_effect_cov = float(np.mean(diff_cov) / (np.std(diff_cov) + 1e-9)) if n_nonzero_cov > 0 else 0.0
            r_effect_comp = float(np.mean(diff_comp) / (np.std(diff_comp) + 1e-9)) if n_nonzero_comp > 0 else 0.0

            pvals_cov.append(p_cov)
            pvals_comp.append(p_comp)

            records.append({
                "k": k,
                "system_1": sys1,
                "system_2": sys2,
                "n_queries": 59,
                "mean_delta_coverage": round(float(np.mean(diff_cov)), 4),
                "wilcoxon_stat_coverage": round(w_stat_cov, 2),
                "pvalue_coverage": p_cov,
                "mean_delta_composite": round(float(np.mean(diff_comp)), 4),
                "wilcoxon_stat_composite": round(w_stat_comp, 2),
                "pvalue_composite": p_comp,
                "effect_size_composite": round(r_effect_comp, 4),
            })

    # FDR Benjamini-Hochberg correction
    fdr_p_cov = false_discovery_control(np.array(pvals_cov))
    fdr_p_comp = false_discovery_control(np.array(pvals_comp))

    for idx, r in enumerate(records):
        r["fdr_pvalue_coverage"] = round(float(fdr_p_cov[idx]), 5)
        r["fdr_pvalue_composite"] = round(float(fdr_p_comp[idx]), 5)
        r["pvalue_coverage"] = round(r["pvalue_coverage"], 5)
        r["pvalue_composite"] = round(r["pvalue_composite"], 5)

    return pd.DataFrame(records)


def compute_pooled_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Supplementary descriptive diagnostic: pooled correlations across all 885 rows and 708 non-LSH rows.
    Explicitly labeled as descriptive diagnostics, NOT independent-sample inference.
    """
    # 1. All rows (N=885)
    r_ndcg_cov, p_ndcg_cov = spearmanr(df["ndcg_at_k"], df["evidence_coverage"])
    r_ndcg_comp, p_ndcg_comp = spearmanr(df["ndcg_at_k"], df["composite_quality"])
    r_cov_comp, p_cov_comp = spearmanr(df["evidence_coverage"], df["composite_quality"])
    r_rec_cov, p_rec_cov = spearmanr(df["recall_at_k"], df["evidence_coverage"])
    r_rec_comp, p_rec_comp = spearmanr(df["recall_at_k"], df["composite_quality"])

    # 2. Non-LSH (N=708)
    no_lsh = df[df["retriever"] != "lsh"].copy()
    r_ndcg_cov_no, p_ndcg_cov_no = spearmanr(no_lsh["ndcg_at_k"], no_lsh["evidence_coverage"])
    r_ndcg_comp_no, p_ndcg_comp_no = spearmanr(no_lsh["ndcg_at_k"], no_lsh["composite_quality"])
    r_cov_comp_no, p_cov_comp_no = spearmanr(no_lsh["evidence_coverage"], no_lsh["composite_quality"])
    r_rec_cov_no, p_rec_cov_no = spearmanr(no_lsh["recall_at_k"], no_lsh["evidence_coverage"])
    r_rec_comp_no, p_rec_comp_no = spearmanr(no_lsh["recall_at_k"], no_lsh["composite_quality"])

    return {
        "pooled_all_observations": {
            "n_observations": 885,
            "ndcg_vs_coverage": {"rho": round(float(r_ndcg_cov), 4), "pvalue": float(p_ndcg_cov)},
            "ndcg_vs_composite": {"rho": round(float(r_ndcg_comp), 4), "pvalue": float(p_ndcg_comp)},
            "coverage_vs_composite": {"rho": round(float(r_cov_comp), 4), "pvalue": float(p_cov_comp)},
            "recall_vs_coverage": {"rho": round(float(r_rec_cov), 4), "pvalue": float(p_rec_cov)},
            "recall_vs_composite": {"rho": round(float(r_rec_comp), 4), "pvalue": float(p_rec_comp)},
        },
        "pooled_non_lsh_observations": {
            "n_observations": 708,
            "ndcg_vs_coverage": {"rho": round(float(r_ndcg_cov_no), 4), "pvalue": float(p_ndcg_cov_no)},
            "ndcg_vs_composite": {"rho": round(float(r_ndcg_comp_no), 4), "pvalue": float(p_ndcg_comp_no)},
            "coverage_vs_composite": {"rho": round(float(r_cov_comp_no), 4), "pvalue": float(p_cov_comp_no)},
            "recall_vs_coverage": {"rho": round(float(r_rec_cov_no), 4), "pvalue": float(p_rec_cov_no)},
            "recall_vs_composite": {"rho": round(float(r_rec_comp_no), 4), "pvalue": float(p_rec_comp_no)},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def generate_figures(
    df: pd.DataFrame,
    config_summary: pd.DataFrame,
    output_dir: Path | str = DEFAULT_FIGURES_DIR,
) -> Dict[str, Path]:
    """Generate publication-quality figures for the research report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig_paths = {}
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {
        "bm25": "#2b5c8f",
        "dense": "#d95f02",
        "bm25_dense_rrf": "#7570b3",
        "bm25_dense_lsh_rrf": "#1b9e77",
        "lsh": "#e7298a",
    }
    retriever_labels = {
        "bm25": "BM25 (Sparse)",
        "dense": "Dense (all-MiniLM-L6-v2)",
        "bm25_dense_rrf": "BM25 + Dense RRF",
        "bm25_dense_lsh_rrf": "BM25 + Dense + LSH RRF",
        "lsh": "LSH (Approximate)",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 1: Retrieval NDCG@K vs Mean Evidence Coverage (Configuration-level)
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8.5, 6), dpi=300)
    for ret, grp in config_summary.groupby("retriever"):
        plt.scatter(
            grp["ndcg_mean"],
            grp["coverage_mean"],
            label=retriever_labels.get(ret, ret),
            color=palette.get(ret, "#333333"),
            s=120,
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        for _, row in grp.iterrows():
            plt.annotate(
                f"K={int(row['k'])}",
                (row["ndcg_mean"], row["coverage_mean"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                fontweight="semibold",
            )

    plt.title("Figure 1: Configuration-Level NDCG@K vs. Evidence Coverage (n=15)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Conventional Retrieval Quality (NDCG@K)", fontsize=11, fontweight="semibold")
    plt.ylabel("Mean Downstream Evidence Coverage", fontsize=11, fontweight="semibold")
    plt.xlim(-0.02, 0.65)
    plt.ylim(0.20, 0.88)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5, loc="lower right")
    plt.tight_layout()
    f1_path = out_path / "fig1_ndcg_vs_coverage_config.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    fig_paths["fig1"] = f1_path

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 2: Mean Evidence Coverage vs Mean Composite Quality (Configuration-level)
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8.5, 6), dpi=300)
    for ret, grp in config_summary.groupby("retriever"):
        plt.scatter(
            grp["coverage_mean"],
            grp["composite_mean"],
            label=retriever_labels.get(ret, ret),
            color=palette.get(ret, "#333333"),
            s=120,
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        for _, row in grp.iterrows():
            plt.annotate(
                f"K={int(row['k'])}",
                (row["coverage_mean"], row["composite_mean"]),
                textcoords="offset points",
                xytext=(6, -7),
                fontsize=9,
                fontweight="semibold",
            )

    # Reference trend line
    cov_arr = config_summary["coverage_mean"].values
    comp_arr = config_summary["composite_mean"].values
    m, b = np.polyfit(cov_arr, comp_arr, 1)
    plt.plot(np.linspace(0.25, 0.85, 50), m * np.linspace(0.25, 0.85, 50) + b, "--", color="#888888", alpha=0.7, label=f"Linear Fit (slope={m:.2f})")

    plt.title("Figure 2: Configuration-Level Evidence Coverage vs. Composite Answer Quality", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Evidence Coverage", fontsize=11, fontweight="semibold")
    plt.ylabel("Mean Downstream Composite Answer Quality", fontsize=11, fontweight="semibold")
    plt.xlim(0.20, 0.86)
    plt.ylim(0.32, 0.86)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5, loc="lower right")
    plt.tight_layout()
    f2_path = out_path / "fig2_coverage_vs_composite_config.png"
    plt.savefig(f2_path, dpi=300)
    plt.close()
    fig_paths["fig2"] = f2_path

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 3: Retrieval NDCG@K vs Mean Composite Quality (Configuration-level)
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8.5, 6), dpi=300)
    for ret, grp in config_summary.groupby("retriever"):
        plt.scatter(
            grp["ndcg_mean"],
            grp["composite_mean"],
            label=retriever_labels.get(ret, ret),
            color=palette.get(ret, "#333333"),
            s=120,
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        for _, row in grp.iterrows():
            plt.annotate(
                f"K={int(row['k'])}",
                (row["ndcg_mean"], row["composite_mean"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
                fontweight="semibold",
            )

    plt.title("Figure 3: Configuration-Level NDCG@K vs. Composite Answer Quality (n=15)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Conventional Retrieval Quality (NDCG@K)", fontsize=11, fontweight="semibold")
    plt.ylabel("Mean Downstream Composite Answer Quality", fontsize=11, fontweight="semibold")
    plt.xlim(-0.02, 0.65)
    plt.ylim(0.32, 0.86)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5, loc="lower right")
    plt.tight_layout()
    f3_path = out_path / "fig3_ndcg_vs_composite_config.png"
    plt.savefig(f3_path, dpi=300)
    plt.close()
    fig_paths["fig3"] = f3_path

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 4: Retriever x K Heatmaps (NDCG, Evidence Coverage, Composite Quality)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    metric_configs = [
        ("ndcg_mean", "Retrieval NDCG@K", "YlGnBu", axes[0]),
        ("coverage_mean", "Evidence Coverage", "Greens", axes[1]),
        ("composite_mean", "Composite Answer Quality", "Purples", axes[2]),
    ]
    pivot_index = ["bm25", "dense", "bm25_dense_rrf", "bm25_dense_lsh_rrf", "lsh"]
    display_index = [retriever_labels[r] for r in pivot_index]

    for col_metric, title, cmap, ax in metric_configs:
        pivot = config_summary.pivot(index="retriever", columns="k", values=col_metric).reindex(pivot_index)
        pivot.index = display_index
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            cbar=True,
            linewidths=0.8,
            linecolor="white",
            ax=ax,
            annot_kws={"fontsize": 10, "fontweight": "bold"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Cutoff K", fontsize=10, fontweight="semibold")
        ax.set_ylabel("")

    plt.suptitle("Figure 4: Performance Across Retriever Architectures and Cutoff Depths", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    f4_path = out_path / "fig4_heatmaps_retriever_k.png"
    plt.savefig(f4_path, dpi=300, bbox_inches="tight")
    plt.close()
    fig_paths["fig4"] = f4_path

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 5: Query-Level Scatter Plots at K=10 (Representative Systems)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    rep_systems = [
        ("bm25", 10, axes[0, 0]),
        ("dense", 10, axes[0, 1]),
        ("bm25_dense_rrf", 10, axes[1, 0]),
        ("lsh", 10, axes[1, 1]),
    ]
    for sys_name, k_val, ax in rep_systems:
        sub = df[(df["retriever"] == sys_name) & (df["k"] == k_val)]
        x = sub["ndcg_at_k"].values
        y_cov = sub["evidence_coverage"].values
        y_comp = sub["composite_quality"].values

        ax.scatter(x, y_comp, alpha=0.75, color=palette[sys_name], s=55, label="Composite Quality", edgecolor="k", linewidth=0.6)
        ax.scatter(x, y_cov, alpha=0.6, color="#2ca02c", marker="^", s=45, label="Evidence Coverage")

        r_ndcg_comp, _ = spearmanr(x, y_comp)
        r_cov_comp, _ = spearmanr(y_cov, y_comp)
        ax.set_title(f"{retriever_labels[sys_name]} (K={k_val})\nρ(NDCG, Comp)={r_ndcg_comp:+.2f} | ρ(Cov, Comp)={r_cov_comp:+.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Query NDCG@10", fontsize=10)
        ax.set_ylabel("Downstream Score", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.suptitle("Figure 5: Query-Level Retrieval vs. Downstream Scores (K=10, n=59 Queries)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    f5_path = out_path / "fig5_query_level_scatters.png"
    plt.savefig(f5_path, dpi=300, bbox_inches="tight")
    plt.close()
    fig_paths["fig5"] = f5_path

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 6: K Sensitivity Curves for Downstream Quality
    # ─────────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    for ret in EXPECTED_RETRIEVERS:
        sub = config_summary[config_summary["retriever"] == ret].sort_values("k")
        ax1.plot(
            sub["k"],
            sub["composite_mean"],
            marker="o",
            linewidth=2.2,
            markersize=8,
            color=palette[ret],
            label=retriever_labels[ret],
        )
        ax2.plot(
            sub["k"],
            sub["coverage_mean"],
            marker="s",
            linewidth=2.2,
            markersize=8,
            color=palette[ret],
            label=retriever_labels[ret],
        )

    ax1.set_title("Downstream Composite Answer Quality vs. K", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Cutoff K", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Mean Composite Quality", fontsize=11, fontweight="semibold")
    ax1.set_xticks([5, 10, 20])
    ax1.set_ylim(0.35, 0.85)
    ax1.legend(loc="lower right", fontsize=9, frameon=True)

    ax2.set_title("Downstream Evidence Coverage vs. K", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Cutoff K", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Mean Evidence Coverage", fontsize=11, fontweight="semibold")
    ax2.set_xticks([5, 10, 20])
    ax2.set_ylim(0.25, 0.85)
    ax2.legend(loc="lower right", fontsize=9, frameon=True)

    plt.suptitle("Figure 6: Downstream Quality Dynamics Across Retrieval Depths K ∈ {5, 10, 20}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    f6_path = out_path / "fig6_k_sensitivity_curves.png"
    plt.savefig(f6_path, dpi=300, bbox_inches="tight")
    plt.close()
    fig_paths["fig6"] = f6_path

    return fig_paths


# ─────────────────────────────────────────────────────────────────────────────
# Complete Execution Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_complete_statistical_analysis(
    reference_evidence_path: Path | str = DEFAULT_REFERENCE_EVIDENCE_PATH,
    retrieval_metrics_path: Path | str = DEFAULT_RETRIEVAL_METRICS_PATH,
    evaluator_results_path: Path | str = DEFAULT_EVALUATOR_RESULTS_PATH,
    output_json_path: Path | str = DEFAULT_OUTPUT_JSON,
    config_summary_csv: Path | str = DEFAULT_CONFIG_SUMMARY_CSV,
    query_corr_csv: Path | str = DEFAULT_QUERY_CORR_CSV,
    pairwise_csv: Path | str = DEFAULT_PAIRWISE_CSV,
    figures_dir: Path | str = DEFAULT_FIGURES_DIR,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute complete Task 4F statistical analysis pipeline and save all artifacts."""
    logger.info("Loading and joining frozen experiment data...")
    df = load_and_join_data(reference_evidence_path, retrieval_metrics_path, evaluator_results_path)

    logger.info("Computing per-configuration summaries (n=15)...")
    config_summary = compute_configuration_summaries(df, n_bootstrap=n_bootstrap, seed=seed)

    logger.info("Computing query-level Spearman correlations...")
    query_correlations = compute_query_level_correlations(df)

    logger.info("Computing H2 bootstrap comparisons (2000 iterations)...")
    h2_bootstrap = compute_h2_bootstrap_comparisons(df, n_bootstrap=n_bootstrap, seed=seed)

    logger.info("Computing system-level exploratory correlations...")
    system_level = compute_system_level_analysis(config_summary)

    logger.info("Computing paired Wilcoxon comparisons...")
    paired_comp = compute_paired_comparisons(df)

    logger.info("Computing pooled descriptive correlations...")
    pooled_corr = compute_pooled_correlations(df)

    logger.info("Generating publication figures...")
    fig_paths = generate_figures(df, config_summary, figures_dir)

    # Export CSVs
    config_summary.to_csv(config_summary_csv, index=False)
    query_correlations.to_csv(query_corr_csv, index=False)
    paired_comp.to_csv(pairwise_csv, index=False)

    # Master machine-readable JSON output
    results_json = {
        "metadata": {
            "task": "Task 4F Frozen Results Statistical Analysis",
            "study": "Cardiology-Chat Cardiovascular RAG Research Study",
            "research_question": "How well do conventional retrieval metrics predict evidence sufficiency and downstream answer quality in cardiovascular RAG?",
            "n_primary_queries": 59,
            "n_configurations": 15,
            "n_conditions": 885,
            "random_seed": seed,
            "n_bootstrap_iterations": n_bootstrap,
            "retrieval_dense_model": "sentence-transformers/all-MiniLM-L6-v2",
            "generator_model": "gemini-3.1-flash-lite",
            "evaluator_model": "command-r-08-2024",
            "shared_reference_dependency_disclosed": True,
            "claims_restricted_to_automated_quality": True,
        },
        "configuration_summaries": config_summary.to_dict(orient="records"),
        "query_level_correlations": query_correlations.to_dict(orient="records"),
        "h2_bootstrap_comparisons": h2_bootstrap.to_dict(orient="records"),
        "system_level_exploratory": system_level,
        "paired_comparisons": paired_comp.to_dict(orient="records"),
        "pooled_descriptive_correlations": pooled_corr,
        "figures_generated": {k: str(v) for k, v in fig_paths.items()},
    }

    out_json_path = Path(output_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    logger.info("Statistical analysis complete. Artifacts written successfully.")
    return results_json

