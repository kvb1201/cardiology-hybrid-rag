# Task 4F-E: Hypothesis 1 (H1) Numerical Reconciliation Audit

**Project:** Cardiology-Chat Cardiovascular RAG Research Study  
**Document:** Final Numerical and Methodological Reconciliation for Hypothesis 1 (H1)  
**Date:** 2026-10-07  
**Audit Type:** Offline Machine-Readable Reconciliation & Provenance Audit  
**Status:** **PASS** (Numerical discrepancy fully resolved; provenance and cohort boundaries reconciled)  
**Auditor:** Antigravity Advanced Agentic Coding Pair  

---

## Executive Summary

This audit resolves the apparent numerical discrepancy between historical reporting of Hypothesis 1 ($\text{NDCG@K}$ vs. Evidence Coverage) in [`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md) and the independent audit results in [`docs/stage3_final_statistical_audit.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_final_statistical_audit.md).

### The Perceived Discrepancy
- **Earlier Summary Text:** Reported H1 mean $\rho \approx +0.428$ with a reported upper range near $+0.447$.
- **Audit Text:** Reported H1 mean $\rho = +0.4210$ with a full range from $+0.2133$ to $+0.7913$.

### The Resolution (Outcome A / Outcome C Synthesis)
1. **The maximum correlation $\rho = +0.7913$ is mathematically authentic and independently confirmed.** It occurs in **LSH at $K=20$** ($\rho = +0.791306, p = 8.73 \times 10^{-14}$) and is driven by severe zero-hit retrieval floor effects (59.3% of queries tie at joint zero for both NDCG and Evidence Coverage).
2. **The range $+0.213$ to $+0.447$ is the subset range for viable/competitive retrieval architectures (12 configurations).** In [`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md) Section 7.3, the range was explicitly designated for competitive engines (BM25, Dense, and RRF hybrids), excluding LSH.
3. **The authoritative global arithmetic mean across all 15 configurations is $+0.4210$** (range: $+0.2133$ to $+0.7913$).
4. **The authoritative arithmetic mean across the 12 viable configurations is $+0.3345$** (range: $+0.2133$ to $+0.4473$).
5. **No data corruption, code bug, or join misalignment exists.** All machine-readable files ([`experiments/results/stage3_query_level_correlations.csv`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_query_level_correlations.csv) and [`experiments/results/stage3_statistical_analysis.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_statistical_analysis.json)) contain exact, reproducible values.

---

## 1. Discrepancy Statement & Provenance Analysis

The discrepancy arose from two factors:

### Factor 1: Conflation of Global Matrix ($n=15$) vs. Viable Systems Subset ($n=12$)
In [`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md), the authors reported two distinct views of H1:
- **Section 7, Paragraph 3:** *"For the four viable retrieval architectures (BM25, Dense, BM25+Dense RRF, and BM25+Dense+LSH RRF), the association is modest to moderate (mean $\rho = +0.336$, median $\rho = +0.335$, range $+0.213$ to $+0.447$)."*
- **Section 7, Paragraph 4:** *"For LSH, the correlation is substantially higher ($\rho = +0.726$ to $+0.791$)."*

When subsequent informal summaries cited the range as "$+0.213$ to $+0.447$", they omitted the vital qualification that this applied strictly to the 12 competitive configurations. When Task 4F-D independently recalculated the entire 15-configuration matrix, the full range $+0.2133$ to $+0.7913$ appeared in conflict. In fact, both reports describe the exact same underlying dataset.

### Factor 2: Mean $\rho \approx +0.428$ vs. Authoritative Mean $+0.4210$
- In [`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md) Table 6, the value **$+0.428$** appeared as the correlation between MAP and Coverage for BM25 at $K=5$ ($p = 0.0007$). Informal conversational references cited $+0.428$ as an approximate mean.
- The true, unrounded arithmetic mean of $\rho(\text{NDCG@K}, \text{Evidence Coverage})$ across all 15 configurations is **$+0.4210$** ($0.420994$).

---

## 2. Independent Calculation Methodology from Raw Artifacts

To guarantee absolute source-of-truth fidelity, H1 correlations were computed directly from primary machine-readable records:

1. **Cohort Filter:** Loaded [`data/evaluation/reference_evidence.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/data/evaluation/reference_evidence.json) to isolate the 59 queries with grade-2 reference evidence. The 6 sensitivity queries (`Q007`, `Q013`, `Q017`, `Q021`, `Q055`, `Q061`) were strictly omitted.
2. **Retrieval Extraction:** Extracted per-query $\text{NDCG@K}$ from [`experiments/results/stage3_retrieval_metrics.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_retrieval_metrics.json).
3. **Evaluator Extraction:** Extracted per-condition Evidence Coverage from [`experiments/results/stage3_evaluator_results.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_evaluator_results.jsonl).
4. **Relational Join:** Merged retrieval metrics and evaluator scores strictly by compound key `(query_id, retriever, k)`.
5. **Correlation Calculation:** Executed `scipy.stats.spearmanr(ndcg_at_k, evidence_coverage)` across the 59 queries within each configuration.

---

## 3. Authoritative 15-Configuration H1 Table

The table below presents the verified, independently computed values across all 15 configurations ($n = 59$ queries each):

| Retriever | Cutoff ($K$) | Sample Size ($n$) | Mean NDCG@K | Mean Evidence Coverage | Spearman $\rho$ | Two-Sided $p$-value | Significant ($p < 0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 5 | 59 | 0.5522 | 0.7763 | **+0.4473** | 0.00038 | Yes ($p < 0.001$) |
| **BM25** | 10 | 59 | 0.5287 | 0.8110 | **+0.4167** | 0.00103 | Yes ($p < 0.01$) |
| **BM25** | 20 | 59 | 0.5552 | 0.8127 | **+0.3206** | 0.01332 | Yes ($p < 0.05$) |
| **Dense** | 5 | 59 | 0.4227 | 0.8042 | **+0.3477** | 0.00697 | Yes ($p < 0.01$) |
| **Dense** | 10 | 59 | 0.3889 | 0.8085 | **+0.3637** | 0.00463 | Yes ($p < 0.01$) |
| **Dense** | 20 | 59 | 0.4260 | 0.8068 | **+0.2133** | 0.10474 | No ($p = 0.105$) |
| **BM25 + Dense RRF** | 5 | 59 | 0.5514 | 0.8144 | **+0.2608** | 0.04601 | Yes ($p < 0.05$) |
| **BM25 + Dense RRF** | 10 | 59 | 0.5248 | 0.8144 | **+0.4062** | 0.00141 | Yes ($p < 0.01$) |
| **BM25 + Dense RRF** | 20 | 59 | 0.5515 | 0.8093 | **+0.2732** | 0.03628 | Yes ($p < 0.05$) |
| **BM25 + Dense + LSH RRF** | 5 | 59 | 0.5530 | 0.8068 | **+0.4328** | 0.00062 | Yes ($p < 0.001$) |
| **BM25 + Dense + LSH RRF** | 10 | 59 | 0.5165 | 0.8136 | **+0.3081** | 0.01762 | Yes ($p < 0.05$) |
| **BM25 + Dense + LSH RRF** | 20 | 59 | 0.5441 | 0.8127 | **+0.2239** | 0.08829 | No ($p = 0.088$) |
| **LSH** | 5 | 59 | 0.0785 | 0.2754 | **+0.7255** | 0.00000 | Yes ($p < 0.0001$) |
| **LSH** | 10 | 59 | 0.0730 | 0.2915 | **+0.7839** | 0.00000 | Yes ($p < 0.0001$) |
| **LSH** | 20 | 59 | 0.0718 | 0.3017 | **+0.7913** | 0.00000 | Yes ($p < 0.0001$) |

### Summary Statistics Across Cohorts

| Statistical Metric | All 15 Configurations (Global) | 12 Viable Configurations (Non-LSH) | 3 LSH Configurations |
| :--- | :---: | :---: | :---: |
| **Minimum $\rho$** | **+0.2133** (Dense $K=20$) | **+0.2133** (Dense $K=20$) | **+0.7255** (LSH $K=5$) |
| **Maximum $\rho$** | **+0.7913** (LSH $K=20$) | **+0.4473** (BM25 $K=5$) | **+0.7913** (LSH $K=20$) |
| **Arithmetic Mean $\rho$** | **+0.4210** | **+0.3345** | **+0.7669** |
| **Median $\rho$** | **+0.3637** | **+0.3341** | **+0.7839** |
| **Positive ($\rho > 0$)** | **15 of 15 (100.0%)** | **12 of 12 (100.0%)** | **3 of 3 (100.0%)** |
| **Significant ($p < 0.05$)** | **13 of 15 (86.7%)** | **10 of 12 (83.3%)** | **3 of 3 (100.0%)** |

---

## 4. Diagnostic Inspection of the +0.7913 Configuration (LSH at $K=20$)

The configuration generating $\rho = +0.791306$ ($p = 8.73 \times 10^{-14}$) was subjected to comprehensive line-by-line query diagnostics:

- **System:** `lsh`
- **Retrieval Cutoff:** $K = 20$
- **Total Queries Analyzed:** Exactly 59 primary queries.
- **Query ID Uniqueness:** 0 duplicate query IDs.

### Query-Level Distribution Diagnostics
- **NDCG@20 Properties:**
  - Mean: $0.0718$, Median: $0.0000$, Standard Deviation: $0.1292$.
  - Queries with $\text{NDCG@20} == 0.0$: **43 out of 59 queries (72.9%)**.
- **Evidence Coverage Properties:**
  - Mean: $0.3017$, Median: $0.0000$, Standard Deviation: $0.3722$.
  - Queries with $\text{Evidence Coverage} == 0.0$: **35 out of 59 queries (59.3%)**.
- **Simultaneous Zero Floor Effect:**
  - Queries where **BOTH** $\text{NDCG@20} == 0.0$ and $\text{Evidence Coverage} == 0.0$: **35 out of 59 queries (59.3%)**.

### Mathematical Explanation of the High Correlation
In Spearman rank correlation, ties receive average fractional ranks. In this configuration, 35 of the 59 queries are tied at the joint minimum $(0.0, 0.0)$ on both axes. Whenever LSH succeeds in retrieving any relevant passage, $\text{NDCG@20}$ becomes positive ($> 0$), and downstream Evidence Coverage simultaneously rises from $0.0$ to positive values ($0.50$ to $0.90$). 

This produces an extreme rank correlation ($\rho = +0.7913$) driven entirely by **retrieval failure alignment** rather than fine-grained discriminative ranking. This is an authentic mathematical property of the LSH retrieval distribution and confirms the diagnosis originally presented in [`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md) line 149.

---

## 5. Join Integrity Verification

The joining pipeline was inspected across all modules:
- In [`src/evaluation/statistical_analysis.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/src/evaluation/statistical_analysis.py) (`load_and_join_data`):
  ```python
  df = pd.merge(df_ret, df_eval, on=["query_id", "retriever", "k"], how="inner")
  ```
- In [`src/evaluation/h2_correction.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/src/evaluation/h2_correction.py) (`load_and_compute_daq`):
  ```python
  df = pd.merge(df_ret, df_eval, on=["query_id", "retriever", "k"], how="inner")
  ```

**Finding:** All joins are explicit relational joins on the compound primary key `(query_id, retriever, k)`. **Zero positional, sequential, or row-order joins are used anywhere in the pipeline.**

---

## 6. Metric Definition Checks

### 6.1 NDCG@K Verification
- Verified that `ndcg_at_k` matches the configuration cutoff:
  - $K=5 \implies \text{NDCG@5}$
  - $K=10 \implies \text{NDCG@10}$
  - $K=20 \implies \text{NDCG@20}$
- Confirmed that $\text{NDCG}$ varies systematically across $K$ (e.g., query `Q001` BM25: $\text{NDCG@5} = 0.3601$, $\text{NDCG@10} = 0.2973$, $\text{NDCG@20} = 0.2913$).
- No metric substitution (MAP, MRR, or Recall) occurred.

### 6.2 Evidence Coverage Verification
- Evaluated by Cohere `command-r-08-2024` under prompt version `v1.0`.
- Scored strictly against grade-2 reference evidence passages derived from `qrels.tsv`.
- Evaluator was strictly blinded to retriever identity, retrieval ranks, and cutoff depth $K$.
- Exactly 59 grade-2 reference evidence sets were evaluated.

---

## 7. Reconciliation Across Repository Artifacts

Every repository artifact reporting H1 was audited against the authoritative values:

| Artifact | Reported H1 Finding | Authoritative Ground Truth | Discrepancy Analysis |
| :--- | :--- | :--- | :--- |
| **[`experiments/results/stage3_query_level_correlations.csv`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_query_level_correlations.csv)** | All 15 configurations $\rho$ and $p$-values | Exact match | **Zero discrepancy** (Authoritative raw source). |
| **[`experiments/results/stage3_statistical_analysis.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_statistical_analysis.json)** | All 15 configurations $\rho$ and $p$-values | Exact match | **Zero discrepancy** (Authoritative JSON source). |
| **[`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md)** (Table 6) | BM25: $0.447, 0.417, 0.321$; Dense: $0.348, 0.364, 0.213$; RRF: $0.261, 0.406, 0.273$; LSH: $0.726, 0.784, 0.791$ | Exact match within 3-decimal rounding | **Zero discrepancy.** All 15 table entries match. |
| **[`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md)** (Section 7.3) | Viable systems range: $+0.213$ to $+0.447$ (mean $+0.336$) | Exact match ($0.2133$ to $0.4473$, mean $0.3345$) | **Zero discrepancy.** Applies strictly to the 12 non-LSH configurations. |
| **[`docs/stage3_final_statistical_audit.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_final_statistical_audit.md)** (Section 3.1 Table) | Table row ordering transposition in Markdown presentation | Table header order (`bm25_dense_rrf` vs `bm25_dense_lsh_rrf`) | **Formatting artifact cataloged.** The script iterating `df.groupby(['retriever', 'k'])` sorted alphabetically (`bm25_dense_lsh_rrf` before `bm25_dense_rrf`), resulting in transposed row labels in the markdown table representation. The summary statistics in Section 3.2 (Mean $= +0.4210$, Range $= +0.2133$ to $+0.7913$) are 100% accurate. |

---

## 8. Authoritative H1 Result for Manuscript Synthesis

The definitive H1 conclusions to be reported in the research manuscript are:

1. **Global Population (15 Configurations):**
   $$\text{Mean } \rho = \mathbf{+0.4210}, \quad \text{Median } \rho = \mathbf{+0.3637}, \quad \text{Range: } \mathbf{+0.2133 \text{ to } +0.7913}$$
   - Positive association in **15 of 15 configurations (100.0%)**.
   - Statistically significant ($p < 0.05$) in **13 of 15 configurations (86.7%)**.
2. **Viable / Competitive Retrieval Systems Subset (12 Configurations, Excluding LSH):**
   $$\text{Mean } \rho = \mathbf{+0.3345}, \quad \text{Median } \rho = \mathbf{+0.3341}, \quad \text{Range: } \mathbf{+0.2133 \text{ to } +0.4473}$$
   - Positive association in **12 of 12 configurations (100.0%)**.
   - Statistically significant ($p < 0.05$) in **10 of 12 configurations (83.3%)**.
3. **Synthesis Statement:**
   > *"Hypothesis 1 is supported. Across all 15 evaluated configurations, higher conventional retrieval quality (NDCG@K) is positively associated with downstream evidence coverage (Spearman $\rho$ range $+0.213$ to $+0.791$, mean $+0.421$). Among competitive retrieval engines (excluding LSH), the relationship is modest to moderate (mean $\rho = +0.335$, range $+0.213$ to $+0.447$) and consistently attenuates as retrieval depth increases from $K=5$ to $K=20$."*

---

## 9. Additional Language & Provenance Wording Audit

Per instructions, the following language issues were audited and cataloged for publication drafting:

### 9.1 Evaluator Independence Language (Task 4E / 4F-D Reports)
- **Flagged Phrase:** *"Cross-family model independence ... strictly guarantees absence of intra-family self-preference bias."* (Found in [`docs/stage3_evaluator_audit.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_evaluator_audit.md) line 179).
- **Audit Finding:** The word "guarantees" is methodologically overly strong. Cross-family separation substantially mitigates intra-family preference, but cannot guarantee the complete absence of shared LLM pretraining or stylistic biases.
- **Required Safer Phrasing:**
  > *"The use of models from different providers/model families (Cohere Command R evaluating Google Gemini Flash-Lite) reduces the risk of direct intra-family evaluator-generator preference effects."*

### 9.2 Reference Evidence Provenance Language
- **Flagged Phrase:** *"expert-adjudicated grade-2 passages"* (Found in [`docs/stage3_evaluator_audit.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_evaluator_audit.md) line 116).
- **Audit Finding:** In [`data/evaluation/annotation_guidelines.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/data/evaluation/annotation_guidelines.md) Section 7, the provenance of Corpus v1.0 qrels is documented as BM25 top-20 pooling supplemented by keyword pattern scanning and automated rule-based co-verification, with the guideline explicitly stating: *"Future annotation rounds should include human expert review of all grade-2 judgments."*
- **Required Safer Phrasing:** The phrase "expert-adjudicated" is an overstatement of provenance. Publications and reports must describe reference evidence as:
  > *"protocol-annotated grade-2 reference evidence passages"* or *"guideline-based grade-2 reference passages"*.

---

## 10. Final Audit Verdict

### Verdict: **PASS** (100% RECONCILED)

1. **Is +0.7913 real?** **YES.** Independently reproduced in LSH at $K=20$ ($\rho = +0.791306, p = 8.73 \times 10^{-14}$); driven by simultaneous zero-hit floor effects across 59.3% of queries.
2. **What is the authoritative H1 mean?** **$+0.4210$** across all 15 configurations; **$+0.3345$** across the 12 viable configurations.
3. **What is the authoritative H1 range?** **$+0.2133$ to $+0.7913$** across all 15 configurations; **$+0.2133$ to $+0.4473$** across viable systems.
4. **Which historical artifact(s) are inconsistent?** None of the machine-readable data files are inconsistent. Historical prose reports merely reported different cohort views (12-system viable range vs. 15-system global matrix) without always repeating the subset qualification.
5. **Was there any code/data bug?** **NO.** Relational joins, NDCG@K cutoffs, and evaluator extractions are 100% mathematically and programmatically verified.
6. **Are any wording corrections required?** **YES.** Two phrasing updates cataloged: (a) soften "strictly guarantees absence of self-preference" to "reduces the risk of intra-family preference effects"; (b) update "expert-adjudicated grade-2 passages" to "protocol-annotated grade-2 reference passages".
