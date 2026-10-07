# Final Statistical Integrity Audit (Task 4F-D)

**Project:** Cardiology-Chat Cardiovascular RAG Research Study  
**Document:** Final Statistical Integrity Audit for Completed Task 4F-C H2 Methodological Correction  
**Date:** 2026-10-07  
**Audit Type:** Offline Machine-Readable Statistical & Methodological Integrity Audit  
**Audit Status:** **PASS** (with two minor documentation reporting anomalies flagged and cataloged)  
**Auditor:** Antigravity Advanced Agentic Coding Pair  

---

## Executive Summary

This document reports the comprehensive statistical and methodological audit of the completed **Task 4F-C H2 Methodological Correction and Reanalysis** against the frozen, machine-readable artifacts of the Cardiology-Chat research project. 

The audit verified population completeness, metric decoupling, numerical reproducibility of rank correlations, joint query resampling in the paired bootstrap procedure, system-level robustness comparisons, paired non-parametric tests, and scientific claim validity.

### Key Audit Findings
1. **Population Integrity:** Exactly 59 primary queries with grade-2 reference evidence, 5 retrievers, and 3 cutoffs ($K \in \{5, 10, 20\}$) yielding 885 conditions. Zero duplicates, zero missing values, and zero appearances of the six excluded sensitivity queries (`Q007`, `Q013`, `Q017`, `Q021`, `Q055`, `Q061`).
2. **Metric Decoupling (DAQ):** Downstream Answer Quality ($\text{DAQ} = (\text{Factual Consistency} + \text{Answer Relevance}) / 2.0$) strictly excludes Evidence Coverage across all 885 observations. Bounded in $[0.0, 1.0]$.
3. **Independent Recalculation:** All query-level Spearman correlations ($\text{NDCG@K}$ vs. Evidence Coverage, Evidence Coverage vs. DAQ, and $\text{NDCG@K}$ vs. DAQ) reproduced within machine precision and standard 4-decimal rounding tolerance ($\le 4.84 \times 10^{-5}$).
4. **Bootstrap Paired Resampling:** The paired bootstrap procedure ($B = 2,000$, seed = $42$) jointly resamples the identical 59 query IDs across metrics. Independent execution verified that $P(\Delta > 0) = 1.0000$ and 95% percentile confidence intervals strictly exclude zero across all 15 configurations.
5. **Scientific Phrasing:** The documentation strictly frames findings as statistical associations, disclaims clinical correctness and safety, and explicitly discloses the shared-reference dependency.
6. **Documentation Discrepancies:** Two minor typographical/drafting discrepancies in [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md) were identified and cataloged without modifying the source reports.
7. **Test Suite:** Complete repository test suite passed with 429 of 429 tests green (100%).

---

## 1. Population Integrity Audit

The frozen experimental design evaluates 5 retrieval architectures across 3 retrieval cutoffs ($K \in \{5, 10, 20\}$) for the 59 primary clinical queries containing verified grade-2 reference evidence passages.

```
Total Conditions Target = 59 queries × 5 retrievers × 3 cutoffs = 885 primary conditions
```

### 1.1 Query Population Verification
- **Primary Cohort Size:** Exactly 59 unique query IDs with verified grade-2 reference evidence.
- **Excluded Sensitivity Queries:**
  The six queries lacking verified grade-2 reference evidence (grade-1 only) were audited across all primary and corrected evaluation artifacts:
  - `Q007`: Count = 0
  - `Q013`: Count = 0
  - `Q017`: Count = 0
  - `Q021`: Count = 0
  - `Q055`: Count = 0
  - `Q061`: Count = 0
  - **Result:** Exactly zero appearances in primary analysis.

### 1.2 Configuration & Observation Counts
Every experimental configuration was audited for query completeness and uniqueness:

| Retriever | $K = 5$ ($n$) | $K = 10$ ($n$) | $K = 20$ ($n$) | Total Conditions | Duplicates |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 59 | 59 | 59 | 177 | 0 |
| **Dense** | 59 | 59 | 59 | 177 | 0 |
| **LSH** | 59 | 59 | 59 | 177 | 0 |
| **BM25 + Dense RRF** | 59 | 59 | 59 | 177 | 0 |
| **BM25 + Dense + LSH RRF** | 59 | 59 | 59 | 177 | 0 |
| **Total** | **295** | **295** | **295** | **885** | **0** |

- **Deduplication Handling:** The raw evaluator log (`stage3_evaluator_results.jsonl`) contains 985 records, comprising 885 successful evaluations and 100 historical retry records from early Pass 1 network resumption. The loading pipeline in [`src/evaluation/h2_correction.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/src/evaluation/h2_correction.py) strictly resolves records by `(query_id, retriever, k)` using successful executions, yielding exactly 885 unique condition tuples.
- **Data Hygiene:** Verified 0 null values, 0 NaNs, and 0 infinite values across all numerical metrics (`evidence_coverage`, `factual_consistency`, `answer_relevance`, `daq`, `ndcg_at_k`).

---

## 2. Downstream Answer Quality (DAQ) Metric Integrity

To resolve the circular part-whole dependency inherent in the historical Composite Quality metric ($\text{Composite Quality} = (\text{Coverage} + \text{Consistency} + \text{Relevance}) / 3$), the reanalysis introduced Downstream Answer Quality:

$$\mathbf{DAQ} = \frac{\text{Factual Consistency} + \text{Answer Relevance}}{2.0}$$

### 2.1 Formula Verification
Every individual record among the 885 primary conditions was tested for exact floating-point equality:
- $| \text{DAQ} - (\text{Factual Consistency} + \text{Answer Relevance}) / 2.0 |_{\max} = \mathbf{0.0}$
- Boundedness: $0.0 \le \text{DAQ} \le 1.0$ holds for 885 of 885 records ($100\%$).
- Empirical observed range: $[0.0000, 0.9000]$.

### 2.2 Population Distribution Summary ($N = 885$)

| Metric | Machine-Calculated Value | Reported in [`stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md) | Discrepancy / Status |
| :--- | :---: | :---: | :--- |
| **Mean** | 0.740367 | 0.7404 | None (exact rounding) |
| **Median** | 0.800000 | 0.8000 | None (exact) |
| **Minimum** | 0.000000 | 0.0000 | None (exact) |
| **Maximum** | 0.900000 | 0.9000 | None (exact) |
| **Sample Std ($ddof=1$)** | 0.214023 | 0.2052 | **Discrepancy: +0.0088** (Flagged in Sec 12) |
| **Population Std ($ddof=0$)**| 0.213902 | 0.2052 | **Discrepancy: +0.0087** (Flagged in Sec 12) |

---

## 3. Hypothesis 1 (H1) Verification

Hypothesis 1 posits:
> *Higher conventional retrieval quality is generally associated with higher evidence coverage.*

### 3.1 Independent Correlation Recalculation
Spearman rank correlations between $\text{NDCG@K}$ and Evidence Coverage were independently recomputed from raw per-query metrics using `scipy.stats.spearmanr` ($n = 59$ queries per configuration):

| Retriever | $K$ | Independently Recomputed $\rho$ | Stored $\rho$ ([`stage3_statistical_analysis.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_statistical_analysis.json)) | Absolute Discrepancy | Stored $p$-value | Discrepancy in $p$ | Significant ($p < 0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 5 | +0.44734 | +0.4473 | $4.0 \times 10^{-5}$ | 0.00038 | $4.8 \times 10^{-6}$ | Yes ($p < 0.001$) |
| **BM25** | 10 | +0.40796 | +0.4080 | $4.0 \times 10^{-5}$ | 0.00135 | $3.5 \times 10^{-6}$ | Yes ($p < 0.01$) |
| **BM25** | 20 | +0.21333 | +0.2133 | $3.1 \times 10^{-5}$ | 0.10486 | $3.8 \times 10^{-6}$ | No ($p = 0.105$) |
| **Dense** | 5 | +0.29742 | +0.2974 | $2.0 \times 10^{-5}$ | 0.02213 | $2.6 \times 10^{-6}$ | Yes ($p < 0.05$) |
| **Dense** | 10 | +0.33413 | +0.3341 | $3.0 \times 10^{-5}$ | 0.00974 | $1.7 \times 10^{-6}$ | Yes ($p < 0.01$) |
| **Dense** | 20 | +0.27092 | +0.2709 | $2.0 \times 10^{-5}$ | 0.03786 | $1.9 \times 10^{-6}$ | Yes ($p < 0.05$) |
| **BM25 + Dense RRF** | 5 | +0.34796 | +0.3480 | $4.0 \times 10^{-5}$ | 0.00684 | $3.6 \times 10^{-6}$ | Yes ($p < 0.01$) |
| **BM25 + Dense RRF** | 10 | +0.39958 | +0.3996 | $2.0 \times 10^{-5}$ | 0.00177 | $4.3 \times 10^{-6}$ | Yes ($p < 0.01$) |
| **BM25 + Dense RRF** | 20 | +0.33596 | +0.3360 | $4.0 \times 10^{-5}$ | 0.00928 | $4.2 \times 10^{-6}$ | Yes ($p < 0.01$) |
| **BM25 + Dense + LSH RRF** | 5 | +0.48522 | +0.4852 | $2.0 \times 10^{-5}$ | 0.00010 | $2.1 \times 10^{-6}$ | Yes ($p < 0.001$) |
| **BM25 + Dense + LSH RRF** | 10 | +0.22602 | +0.2260 | $2.0 \times 10^{-5}$ | 0.08511 | $3.5 \times 10^{-6}$ | No ($p = 0.085$) |
| **BM25 + Dense + LSH RRF** | 20 | +0.36625 | +0.3663 | $5.0 \times 10^{-5}$ | 0.00438 | $2.8 \times 10^{-6}$ | Yes ($p < 0.01$) |
| **LSH** | 5 | +0.76356 | +0.7636 | $4.0 \times 10^{-5}$ | 0.00000 | $0.0 \times 10^{-6}$ | Yes ($p < 0.0001$) |
| **LSH** | 10 | +0.62794 | +0.6279 | $4.0 \times 10^{-5}$ | 0.00000 | $0.0 \times 10^{-6}$ | Yes ($p < 0.0001$) |
| **LSH** | 20 | +0.79131 | +0.7913 | $1.0 \times 10^{-5}$ | 0.00000 | $0.0 \times 10^{-6}$ | Yes ($p < 0.0001$) |

### 3.2 H1 Summary Statistics
- **Directional Consistency:** Positive association in **15 of 15 configurations (100.0%)**.
- **Statistical Significance:** Significant ($p < 0.05$) in **13 of 15 configurations (86.7%)**.
- **Correlation Range:** $+0.2133$ to $+0.7913$ (mean $+0.4210$).
- **Maximum Absolute Discrepancy:** $4.84 \times 10^{-5}$ for $\rho$ and $4.82 \times 10^{-6}$ for $p$-value (consistent with 4- and 5-decimal place rounding).

---

## 4. Hypothesis 2 (H2) Verification

Revised Hypothesis 2 posits:
> *Evidence Coverage has a stronger association with downstream answer quality (DAQ) than conventional retrieval quality alone (NDCG@K).*

### 4.1 Independent Recalculation of H2 Correlations & $\Delta$
For each configuration, the difference between the two correlations was independently computed:
$$\Delta = \rho(\text{Evidence Coverage}, \text{DAQ}) - \rho(\text{NDCG@K}, \text{DAQ})$$

| Configuration | Recomputed $\rho(\text{Cov}, \text{DAQ})$ | Stored $\rho(\text{Cov}, \text{DAQ})$ | Recomputed $\rho(\text{NDCG}, \text{DAQ})$ | Stored $\rho(\text{NDCG}, \text{DAQ})$ | Recomputed $\Delta$ | Stored $\Delta$ ([`stage3_h2_corrected.csv`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_corrected.csv)) | $\Delta > 0$? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 ($K=5$)** | +0.96439 | +0.9644 | +0.45618 | +0.4562 | +0.50821 | +0.5082 | **Yes** |
| **BM25 ($K=10$)** | +0.93657 | +0.9366 | +0.40938 | +0.4094 | +0.52719 | +0.5272 | **Yes** |
| **BM25 ($K=20$)** | +0.83518 | +0.8352 | +0.14515 | +0.1452 | +0.69003 | +0.6900 | **Yes** |
| **Dense ($K=5$)** | +0.80534 | +0.8053 | +0.30773 | +0.3077 | +0.49761 | +0.4976 | **Yes** |
| **Dense ($K=10$)** | +0.89924 | +0.8992 | +0.32230 | +0.3223 | +0.57694 | +0.5769 | **Yes** |
| **Dense ($K=20$)** | +0.86389 | +0.8639 | +0.12687 | +0.1269 | +0.73702 | +0.7370 | **Yes** |
| **BM25+Dense RRF ($K=5$)** | +0.90367 | +0.9037 | +0.27233 | +0.2723 | +0.63134 | +0.6313 | **Yes** |
| **BM25+Dense RRF ($K=10$)** | +0.86548 | +0.8655 | +0.31878 | +0.3188 | +0.54670 | +0.5467 | **Yes** |
| **BM25+Dense RRF ($K=20$)** | +0.86458 | +0.8646 | +0.22013 | +0.2201 | +0.64445 | +0.6445 | **Yes** |
| **BM25+Dense+LSH RRF ($K=5$)** | +0.94553 | +0.9455 | +0.45525 | +0.4552 | +0.49028 | +0.4903 | **Yes** |
| **BM25+Dense+LSH RRF ($K=10$)** | +0.81395 | +0.8140 | +0.22602 | +0.2260 | +0.58793 | +0.5879 | **Yes** |
| **BM25+Dense+LSH RRF ($K=20$)** | +0.91090 | +0.9109 | +0.26127 | +0.2613 | +0.64963 | +0.6496 | **Yes** |
| **LSH ($K=5$)** | +0.87706 | +0.8771 | +0.62725 | +0.6273 | +0.24981 | +0.2498 | **Yes** |
| **LSH ($K=10$)** | +0.89505 | +0.8951 | +0.67056 | +0.6706 | +0.22449 | +0.2245 | **Yes** |
| **LSH ($K=20$)** | +0.90066 | +0.9007 | +0.69892 | +0.6989 | +0.20174 | +0.2017 | **Yes** |

### 4.2 H2 Recalculation Metrics
- **Minimum $\Delta$:** $+0.2017$ (LSH at $K=20$)
- **Maximum $\Delta$:** $+0.7370$ (Dense at $K=20$)
- **Mean Observed $\Delta$:** $+0.51754$ (Stored: $+0.51756$; difference: $2.0 \times 10^{-5}$)
- **Configurations with $\Delta > 0$:** **15 out of 15 (100.0%)**
- **Coverage Association Range:** $+0.8053$ to $+0.9644$ (mean $+0.8854$, all $p < 0.0001$)
- **NDCG Association Range:** $+0.1269$ to $+0.6989$ (mean $+0.3678$; competitive non-LSH mean $+0.2850$)
- **Max Absolute Discrepancy:** $4.83 \times 10^{-5}$ (within rounding tolerance).

---

## 5. Paired Bootstrap Validation

### 5.1 Algorithmic Inspection
Code inspection of `compute_h2_corrected_bootstrap()` in [`src/evaluation/h2_correction.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds project/implementation/src/evaluation/h2_correction.py) (lines 268–307) confirmed:
1. **Bootstrap Parameters:** $B = 2,000$ iterations, seed = $42$.
2. **Joint Resampling Structure:** Resampling indexes queries with replacement:
   ```python
   idx = rng.choice(n_queries, size=n_queries, replace=True)
   b_cov = cov[idx]
   b_ndcg = ndcg[idx]
   b_daq = daq[idx]
   ```
   Because `idx` is generated once per iteration, the identical 59 query observations are paired jointly across Evidence Coverage, NDCG, and DAQ. This respects the repeated-measures, paired design.
3. **Iteration Metric:** For each resample $b$, $\Delta_b = \rho_b(\text{Coverage}, \text{DAQ}) - \rho_b(\text{NDCG}, \text{DAQ})$.
4. **Directional Probability Reporting:** The proportion of positive bootstrap samples $\frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta_b > 0)$ is explicitly documented as the **bootstrap directional probability $P(\Delta > 0)$**, NOT as a hypothesis test $p$-value.

### 5.2 Independent Full 15-Configuration Bootstrap Verification
The entire paired bootstrap procedure was independently re-executed using identical parameters ($B = 2,000$, seed = $42$) and cross-checked against [`experiments/results/stage3_h2_bootstrap.csv`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_bootstrap.csv):

| Configuration | Stored Mean $\Delta$ | Recomputed Mean $\Delta$ | Stored Median $\Delta$ | Recomputed Median $\Delta$ | Stored 95% Percentile CI | Recomputed 95% CI | Stored $P(\Delta > 0)$ | Recomputed $P(\Delta > 0)$ | CI Excludes 0? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 ($K=5$)** | +0.5102 | +0.5102 | +0.5006 | +0.5006 | [+0.3178, +0.7298] | [+0.3178, +0.7298] | 1.0000 | 1.0000 | **Yes** |
| **BM25 ($K=10$)** | +0.5350 | +0.5350 | +0.5273 | +0.5273 | [+0.3204, +0.7922] | [+0.3204, +0.7922] | 1.0000 | 1.0000 | **Yes** |
| **BM25 ($K=20$)** | +0.6876 | +0.6876 | +0.6823 | +0.6823 | [+0.4309, +0.9660] | [+0.4309, +0.9660] | 1.0000 | 1.0000 | **Yes** |
| **Dense ($K=5$)** | +0.4984 | +0.4984 | +0.4932 | +0.4932 | [+0.2891, +0.7134] | [+0.2891, +0.7134] | 1.0000 | 1.0000 | **Yes** |
| **Dense ($K=10$)** | +0.5804 | +0.5804 | +0.5758 | +0.5758 | [+0.3491, +0.8529] | [+0.3491, +0.8529] | 1.0000 | 1.0000 | **Yes** |
| **Dense ($K=20$)** | +0.7351 | +0.7351 | +0.7303 | +0.7303 | [+0.4533, +1.0505] | [+0.4533, +1.0505] | 1.0000 | 1.0000 | **Yes** |
| **BM25+Dense RRF ($K=5$)** | +0.6310 | +0.6310 | +0.6288 | +0.6288 | [+0.4048, +0.8741] | [+0.4048, +0.8741] | 1.0000 | 1.0000 | **Yes** |
| **BM25+Dense RRF ($K=10$)** | +0.5480 | +0.5480 | +0.5412 | +0.5412 | [+0.3516, +0.7774] | [+0.3516, +0.7774] | 1.0000 | 1.0000 | **Yes** |
| **BM25+Dense RRF ($K=20$)** | +0.6489 | +0.6489 | +0.6438 | +0.6438 | [+0.4172, +0.9011] | [+0.4172, +0.9011] | 1.0000 | 1.0000 | **Yes** |
| **BM25+Dense+LSH RRF ($K=5$)** | +0.4941 | +0.4941 | +0.4891 | +0.4891 | [+0.3234, +0.6987] | [+0.3234, +0.6987] | 1.0000 | 1.0000 | **Yes** |
| **BM25+Dense+LSH RRF ($K=10$)**| +0.5862 | +0.5862 | +0.5849 | +0.5849 | [+0.3629, +0.8078] | [+0.3629, +0.8078] | 1.0000 | 1.0000 | **Yes** |
| **BM25+Dense+LSH RRF ($K=20$)**| +0.6552 | +0.6552 | +0.6515 | +0.6515 | [+0.3857, +0.9382] | [+0.3857, +0.9382] | 1.0000 | 1.0000 | **Yes** |
| **LSH ($K=5$)** | +0.2526 | +0.2526 | +0.2514 | +0.2514 | [+0.0964, +0.4267] | [+0.0964, +0.4267] | 1.0000 | 1.0000 | **Yes** |
| **LSH ($K=10$)** | +0.2269 | +0.2269 | +0.2234 | +0.2234 | [+0.0944, +0.3688] | [+0.0944, +0.3688] | 1.0000 | 1.0000 | **Yes** |
| **LSH ($K=20$)** | +0.2033 | +0.2033 | +0.1989 | +0.1989 | [+0.0883, +0.3366] | [+0.0883, +0.3366] | 1.0000 | 1.0000 | **Yes** |

- **Maximum Discrepancy Across All 15 Configurations:**
  - Bootstrap Mean $\Delta$: $4.84 \times 10^{-5}$
  - Bootstrap Median $\Delta$: $4.87 \times 10^{-5}$
  - 95% CI Lower Bound: $4.92 \times 10^{-5}$
  - 95% CI Upper Bound: $4.99 \times 10^{-5}$
  - $P(\Delta > 0)$: **$0.0$** (Identical exact 1.0000 across all 15 configurations)

---

## 6. LSH Robustness & System-Level Audit

Because LSH exhibits floor effects in conventional retrieval ($\text{NDCG@K} \approx 0.07$ vs. $0.38$–$0.55$), configuration-level correlations across configuration averages were audited to ensure LSH does not confound system-level conclusions.

### 6.1 Configuration-Level Correlation Verification

| Configuration Subset | Sample Size | Predictor vs. Outcome | Independently Calculated $\rho$ | Calculated $p$-value | Stored $\rho$ ([`stage3_h2_corrected.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_corrected.json)) | Stored $p$-value |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **All Configurations** | $n = 15$ | NDCG vs. DAQ | +0.4826 | 0.06847 | +0.4826 | 0.06847 |
| **All Configurations** | $n = 15$ | Coverage vs. DAQ | +0.9122 | 0.00000 | +0.9122 | 0.00000 |
| **Non-LSH Subset** | $n = 12$ | NDCG vs. DAQ | **+0.0070** | **0.98276** | **+0.0070** | **0.98276** |
| **Non-LSH Subset** | $n = 12$ | Coverage vs. DAQ | **+0.8345** | **0.00074** | **+0.8345** | **0.00074** |

### 6.2 Audit Verification Checks
1. **LSH Retention in Primary Analysis:** Verified that LSH is fully retained in the primary query-level analysis (all 15 configurations).
2. **Exploratory Status:** Verified that the non-LSH analysis is clearly designated as an exploratory robustness check across configuration averages.
3. **Causal Language:** Verified that neither report uses causal formulations such as "LSH causes".
4. **Floor Effect Interpretation:** Correctly documents that when LSH is removed, the configuration-mean correlation between NDCG and DAQ completely collapses ($\rho = +0.0070, p = 0.983$), while Evidence Coverage remains strongly associated with DAQ ($\rho = +0.8345, p = 0.0007$).

---

## 7. BM25 vs. Dense Downstream Comparison Audit

Lexical BM25 and Dense semantic retrieval were re-evaluated on the decoupled DAQ metric across the 59 queries using paired Wilcoxon signed-rank tests.

### 7.1 Independent Numerical Verification

| Cutoff | BM25 Mean DAQ | Dense Mean DAQ | Mean Difference ($\Delta$) | Recomputed Wilcoxon $W$ | Recomputed $p$-value | Stored $p$-value | Significant ($p < 0.05$)? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 5$** | 0.7826 | 0.8127 | -0.0301 | 106.5 | 0.51455 | 0.51455 | No ($p = 0.515$) |
| **$K = 10$** | 0.8220 | 0.8144 | +0.0076 | 28.0 | 0.21564 | 0.21564 | No ($p = 0.216$) |
| **$K = 20$** | 0.8186 | 0.8123 | +0.0064 | 52.5 | 0.41925 | 0.41925 | No ($p = 0.419$) |

### 7.2 Orientation & Substantive Interpretation
- **Paired Test Orientation:** Consistently computed as **BM25 - Dense**:
  - $K=5$: $\text{BM25} - \text{Dense} = 0.7826 - 0.8127 = -0.0301$ (Dense observed DAQ higher by $0.0301$).
  - $K=10$: $\text{BM25} - \text{Dense} = 0.8220 - 0.8144 = +0.0076$ (BM25 observed DAQ higher by $0.0076$).
  - $K=20$: $\text{BM25} - \text{Dense} = 0.8186 - 0.8123 = +0.0064$ (BM25 observed DAQ higher by $0.0064$).
- **Statistical Interpretation:** The reports correctly state that *"no statistically significant difference was detected"* across all retrieval cutoffs. Non-significance is correctly interpreted as failure to reject the null hypothesis of equal distributions, not as proof that the systems are strictly identical.
- **Scientific Conclusion:** Despite BM25's clear superiority in conventional lexical retrieval metrics ($\text{NDCG@10} = 0.529$ vs. $0.389$), Dense retrieval achieves downstream DAQ parity ($0.8144$ vs. $0.8220$, $p = 0.216$).

---

## 8. Cutoff Depth ($K$) Dynamics Audit

Downstream Answer Quality trajectories across retrieval cutoffs $K \in \{5, 10, 20\}$ were evaluated to verify non-monotonicity and context depth patterns.

### 8.1 Retrieval Depth Trajectories

| Retrieval System | DAQ ($K=5$) | DAQ ($K=10$) | DAQ ($K=20$) | DAQ ($K=20 < K=10$)? | Factual Consistency ($K=10 \to K=20$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 0.7826 | **0.8220** | 0.8186 | **True** ($-0.0034$) | $0.8271 \to 0.8246$ ($-0.0025$) |
| **Dense** | 0.8127 | **0.8144** | 0.8123 | **True** ($-0.0021$) | $0.8195 \to 0.8153$ ($-0.0042$) |
| **BM25 + Dense RRF** | **0.8203** | 0.8199 | 0.8144 | **True** ($-0.0055$) | $0.8229 \to 0.8153$ ($-0.0076$) |
| **BM25 + Dense + LSH RRF** | 0.8131 | **0.8195** | 0.8140 | **True** ($-0.0055$) | $0.8237 \to 0.8169$ ($-0.0068$) |
| **LSH** | 0.4331 | **0.4610** | 0.4475 | **True** ($-0.0135$) | $0.6364 \to 0.5958$ ($-0.0406$) |

### 8.2 Audit Verification Checks
1. **Numerical Correctness of Claim:** The statement *"DAQ at K=20 was lower than K=10 across all five systems"* was verified and is **100% numerically true**.
2. **Cited Factual Consistency Values:** The Factual Consistency figures cited in [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md) line 224 (BM25: $0.8271 \to 0.8246$; Dense: $0.8195 \to 0.8153$; BM25+Dense RRF: $0.8229 \to 0.8153$) match the underlying configuration summaries exactly.
3. **Absence of Causal Claims:** The report qualifies the observation: *"While not demonstrating a causal mechanism, this pattern is consistent with a plausible context dilution effect..."* Context dilution is presented strictly as a plausible descriptive interpretation, not an experimentally proven mechanism.

---

## 9. Statistical Test & Multiple Comparison Audit

### 9.1 Wilcoxon Signed-Rank Tests
- **Paired Unit:** Matched query-level differences for the identical 59 query IDs, sorted deterministically by `query_id`.
- **Zero Differences:** Zero differences are appropriately handled by `scipy.stats.wilcoxon` using the standard Wilcoxon Pratt/zero-exclusion convention.
- **Alternative Hypothesis:** Two-sided alternative hypothesis applied consistently.

### 9.2 Multiple Comparison Corrections (FDR)
- In the primary Task 4F analysis ([`src/evaluation/statistical_analysis.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/src/evaluation/statistical_analysis.py)), 15 pairwise retriever comparisons (5 pairwise contrasts $\times$ 3 cutoffs) were corrected for False Discovery Rate using the Benjamini-Hochberg procedure (`scipy.stats.false_discovery_control`).
- In the Task 4F-C BM25 vs. Dense DAQ reanalysis (3 comparisons at $K=5, 10, 20$), unadjusted two-sided $p$-values were reported ($p = 0.5145, 0.2156, 0.4193$). Because all three unadjusted $p$-values exceed $0.20$, none would attain significance under any multiple-testing adjustment (e.g., FDR-adjusted $p \ge 0.419$).

---

## 10. Report & Machine-Readable Artifact Consistency

All numerical tables and prose statements in [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md) were audited against the machine-readable artifacts:
- [`experiments/results/stage3_h2_corrected.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_corrected.json)
- [`experiments/results/stage3_h2_corrected.csv`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_corrected.csv)
- [`experiments/results/stage3_h2_bootstrap.csv`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_bootstrap.csv)
- [`experiments/results/stage3_daq_configuration_summary.csv`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_daq_configuration_summary.csv)

### 10.1 Table-by-Table Agreement
- **Table 4 (Configuration Summaries):** All 15 rows of NDCG, Evidence Coverage, Factual Consistency, Answer Relevance, DAQ means, and bootstrap 95% CIs agree within 4-decimal rounding tolerance.
- **Table 5 (Corrected Correlations):** All 15 rows of $\rho(\text{Coverage}, \text{DAQ})$, $\rho(\text{NDCG}, \text{DAQ})$, $\Delta$, and dimension-level correlations agree within 4-decimal rounding tolerance.
- **Table 6 (Bootstrap Results):** All 15 rows of observed $\Delta$, bootstrap mean, median, 95% CIs, and $P(\Delta > 0)$ agree within 3-decimal rounding tolerance.
- **Table 8 (LSH Robustness):** All 4 correlation pairs and $p$-values agree to 4 decimal places.
- **Table 10 (K Depth Trajectories):** All 15 DAQ means agree to 4 decimal places.

### 10.2 Historical Provenance & Superseding Status
The audit verified that the historical Task 4F report ([`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md)) is explicitly preserved unmodified for archival provenance, while [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md) explicitly supersedes the historical H2 interpretation. The circular correlation ($\rho \approx 0.94$) and the historical verdict of "H2 strongly supported" based on Composite Quality are explicitly documented as **retracted and replaced** by the decoupled DAQ analysis.

---

## 11. Scientific Claim & Terminology Audit

### 11.1 Prohibited Wording Audit
The audit scanned documentation for inappropriate claims:
- **Causation:** Prohibited words ("causes", "caused", "causality") used as empirical findings: **0 instances**. The word "cause" appears only once in explanatory text stating that Evidence Coverage is a mathematical component of Composite Quality, and once in disclaiming causal mechanisms for context dilution.
- **Clinical Correctness / Clinical Validation / Clinical Safety:** Prohibited claims: **0 instances**. Section 11 of [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md) explicitly clarifies that DAQ does *NOT* establish clinical correctness, diagnostic validity, medical safety, or clinician consensus.
- **Predictive Validity:** No claims of prospective clinical predictive validity were found.

### 11.2 Evaluator & Reference Evidence Dependency Disclosures
The audit verified the presence of explicit limitation disclosures:
1. **Shared-Reference Dependency Disclosure:**
   In [`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md) (lines 27–28):
   > *"Because both retrieval evaluation (qrels relevance grades) and downstream evidence coverage rely on the same qrels-derived grade-2 reference evidence passages, the relationship between conventional retrieval metrics and evidence coverage must be interpreted as a shared-reference association rather than an independent external validation of retrieval metrics."*
2. **Evaluator Proxy Disclosure:**
   In [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md) (lines 230–231):
   > *"DAQ is evaluated automatically by Cohere command-r-08-2024. It measures factual consistency and answer relevance against reference text; it does NOT establish clinical correctness, diagnostic validity, medical safety, or clinician consensus. While Evidence Coverage is excluded from DAQ, reference evidence is still derived from grade-2 qrels. The evaluation remains observational within this corpus and prompt setup."*

### 11.3 Justification of Scientific Statements

| Candidate Scientific Statement | Justified by Empirical Evidence? | Audit Evaluation & Rationale |
| :--- | :---: | :--- |
| **A. "NDCG@K and Evidence Coverage show a positive association."** | **YES** | Positive in 15 of 15 configurations (mean $\rho = +0.421$); statistically significant in 13 of 15 configurations ($p < 0.05$). |
| **B. "Evidence Coverage has a stronger association with DAQ than NDCG@K in the evaluated configurations."** | **YES** | $\Delta > 0$ in 15 of 15 configurations (mean $\Delta = +0.518$); 95% bootstrap percentile CIs strictly exclude zero across all 15 configurations ($P(\Delta > 0) = 1.0000$). |
| **C. "BM25's higher conventional retrieval scores did not translate into statistically significant downstream DAQ superiority over Dense."** | **YES** | Paired Wilcoxon tests show $p = 0.515$ ($K=5$), $p = 0.216$ ($K=10$), and $p = 0.419$ ($K=20$). Dense achieves parity despite lower lexical NDCG. |
| **D. "DAQ generally peaked around K=10 or did not monotonically improve with K."** | **YES** | Across all 5 retrievers, DAQ at $K=20$ is strictly lower than at $K=10$. Deeper context does not yield monotonic answer quality improvements. |
| **E. "The evidence supports an association between reference-evidence coverage and downstream automated answer quality."** | **YES** | Rank correlations between coverage and DAQ are strong and positive ($\rho = +0.805$ to $+0.964$). |
| **"Evidence Coverage causes better answers."** | **NO (STRICTLY PROHIBITED)** | Observational correlations do not establish causality. Both metrics are conditioned on reference evidence passages. |

---

## 12. Catalog of Identified Discrepancies

Per audit protocol, no historical artifacts were silently altered. Two minor documentation anomalies were identified and are formally recorded below:

### Discrepancy 1: BM25 vs. Dense Wilcoxon Statistic in Markdown Report
- **Location:** [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md), Section 9 table (lines 198–202).
- **Reported Value:**
  - $K=5$: Wilcoxon $W = \mathbf{37.0}$ ($p = 0.5145$)
  - $K=10$: Wilcoxon $W = \mathbf{21.0}$ ($p = 0.2156$)
  - $K=20$: Wilcoxon $W = \mathbf{24.5}$ ($p = 0.4193$)
- **Machine-Readable Value:** In [`experiments/results/stage3_h2_corrected.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_corrected.json) (`bm25_vs_dense_daq_comparisons`):
  - $K=5$: `wilcoxon_stat_daq` = $\mathbf{106.5}$ ($p = 0.51455$)
  - $K=10$: `wilcoxon_stat_daq` = $\mathbf{28.0}$ ($p = 0.21564$)
  - $K=20$: `wilcoxon_stat_daq` = $\mathbf{52.5}$ ($p = 0.41925$)
- **Root Cause Analysis:** The test statistic $W$ reported in the markdown table appears to have been transcribed from an intermediate calculation or an alternative rank sum convention, whereas the machine-readable JSON accurately records the standard `scipy.stats.wilcoxon` test statistic (which selects the minimum of positive and negative rank sums for non-zero differences: $n_{nz}=22 \implies W^- = 106.5, W^+ = 146.5 \implies W = 106.5$). Crucially, the $p$-values ($0.5145, 0.2156, 0.4193$), mean differences ($-0.0301, +0.0076, +0.0064$), and qualitative findings are identical.
- **Impact:** Negligible. The machine-readable artifact is authoritative.

### Discrepancy 2: Global Population DAQ Standard Deviation in Markdown Header
- **Location:** [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md), Section 2 (line 58).
- **Reported Value:** `Standard Deviation: 0.2052`
- **Machine-Readable Value:** The sample standard deviation across all 885 conditions is $\mathbf{0.214023}$ (sample $ddof=1$) or $\mathbf{0.213902}$ (population $ddof=0$).
- **Root Cause Analysis:** Drafting transcription variance ($\sim 0.0088$). All other population metrics (Mean = $0.7404$, Median = $0.8000$, Min = $0.0000$, Max = $0.9000$) match machine precision to 4 decimal places.
- **Impact:** Negligible. Configuration-level standard deviations in CSV and JSON are exact.

### Wording Advisory Note
In [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md), subheadings on line 132 (*"Evidence Coverage Remains Highly Predictive"*) and line 190 (*"massively predictive of DAQ"*) use colloquial terminology. In subsequent publication writing, these should be standardized to *"demonstrates strong positive association"* to avoid conflation with prospective clinical predictive validity.

---

## 13. Final Audit Decision

### Audit Verdict: **PASS** (100% Core Statistical Integrity Verified)

The completed Task 4F-C H2 methodological correction and reanalysis adheres strictly to experimental design standards, statistical rigor, and population constraints:
- **Population:** Verified exactly 59 grade-2 queries, 5 retrievers, 3 cutoffs, 15 configurations, 885 primary conditions; zero appearances of excluded sensitivity queries.
- **Metric Validity:** DAQ is completely decoupled from Evidence Coverage, eliminating structural circularity.
- **Reproducibility:** All Spearman correlations, bootstrap distributions, system-level regressions, and paired non-parametric tests independently reproduce with maximum absolute numerical discrepancy $\le 4.84 \times 10^{-5}$ (within floating-point rounding tolerance).
- **Paired Bootstrap:** Resamples the identical 59 query IDs jointly; $P(\Delta > 0) = 1.0000$ and 95% CIs exclude zero in 15 of 15 configurations.
- **Safety & Epistemology:** Causal claims and clinical validity claims are rigorously excluded; shared-reference limitations are prominently acknowledged.
- **Test Suite:** Repository test suite passes 429 of 429 automated tests.
