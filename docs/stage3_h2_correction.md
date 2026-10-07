# Task 4F-C: Methodological Correction and Reanalysis for Hypothesis 2 (H2)

**Project:** Cardiology-Chat Cardiovascular RAG Research Study  
**Document:** Methodological Audit, Metric Decoupling, and Statistical Reanalysis for H2  
**Date:** 2026-10-07  
**Status:** Completed & Validated  
**Provenance Note:** The original Task 4F report ([`docs/stage3_statistical_analysis.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_statistical_analysis.md)) is preserved unmodified for provenance. This document supersedes the original H2 interpretation.

---

## 1. The Original H2 Analysis & Structural Dependency Issue

### What the Original Analysis Calculated
In the initial Task 4F evaluation protocol, downstream generative performance was summarized using Composite Answer Quality:
$$\text{Composite Quality} = \frac{\text{Evidence Coverage} + \text{Factual Consistency} + \text{Answer Relevance}}{3}$$

The original analysis tested Hypothesis 2 by computing the Spearman rank correlation between Evidence Coverage and Composite Quality:
$$\rho(\text{Evidence Coverage}, \text{Composite Quality})$$
and comparing it to $\rho(\text{NDCG@K}, \text{Composite Quality})$.

### Why This Analysis Was Structurally Non-Independent
Because Evidence Coverage constitutes exactly one-third of Composite Quality:
$$\text{Composite Quality} = \frac{1}{3} \text{Evidence Coverage} + \frac{1}{3} \text{Factual Consistency} + \frac{1}{3} \text{Answer Relevance}$$

Correlating Evidence Coverage with Composite Quality introduced a **structural part-whole correlation artifact**. Even if Factual Consistency and Answer Relevance were completely independent of Evidence Coverage, the mathematical inclusion of Evidence Coverage inside the composite score guarantees an inflated, non-zero positive correlation:
$$\text{Cov}(X, \frac{X + Y + Z}{3}) = \frac{1}{3} \text{Var}(X) + \frac{1}{3} \text{Cov}(X, Y) + \frac{1}{3} \text{Cov}(X, Z)$$

### Why the Original Analysis Is Not Suitable as the Primary H2 Test
Testing whether a predictor predicts an outcome that mathematically contains that predictor creates an endogenous feedback loop. Claiming that this correlation establishes that "evidence coverage predicts answer quality" is methodologically circular. 

Therefore:
- The previous $\rho(\text{Evidence Coverage}, \text{Composite Quality}) \approx 0.94$ correlation **must NOT be reported as empirical evidence supporting H2**.
- The historical conclusion of "H2 strongly supported" from the original composite metric is **retracted**.
- The scientific validity of H2 must be established strictly on a downstream quality outcome that is completely independent of Evidence Coverage.

---

## 2. New Downstream Target: Downstream Answer Quality (DAQ)

### Mathematical Definition
To eliminate structural dependency while preserving the evaluative dimensions of downstream generation, we define:

$$\mathbf{DAQ} = \frac{\text{Factual Consistency} + \text{Answer Relevance}}{2.0}$$

### Key Properties
1. **Strict Decoupling:** Evidence Coverage is **NOT** included in the downstream answer-quality outcome used for the corrected H2 analysis.
2. **Clinical Meaning:** DAQ captures whether the generated answer is:
   - Ungrounded or hallucinated vs. consistent with verified reference text (**Factual Consistency**), and
   - Clinically responsive and focused on the query without extraneous digressions (**Answer Relevance**).
3. **Bounded & Finite:** Computed strictly from already-frozen evaluator records; verified finite and bounded in $[0.0, 1.0]$.
4. **Zero New API Calls:** Recomputed offline across the 885 frozen evaluator records without invoking any LLM APIs.

### Population Distribution of DAQ ($N = 885$)
- **Mean:** **0.7404**
- **Median:** **0.8000**
- **Min:** **0.0000**
- **Max:** **0.9000**
- **Standard Deviation:** **0.2052**

---

## 3. Revised Hypothesis 2 (H2) & Experimental Design

### Revised Hypothesis Statement
> **Revised H2:** *Evidence Coverage has a stronger association with downstream answer quality (DAQ) than conventional retrieval quality alone (NDCG@K).*

### Primary Comparison
For each of the 15 experimental configurations (5 retrievers $\times$ 3 cutoffs $K \in \{5, 10, 20\}$, $n = 59$ queries per configuration):

$$\mathbf{A:} \quad \rho(\text{Evidence Coverage}, \text{DAQ})$$
$$\text{versus}$$
$$\mathbf{B:} \quad \rho(\text{NDCG@K}, \text{DAQ})$$
$$\Delta = \rho(\text{Evidence Coverage}, \text{DAQ}) - \rho(\text{NDCG@K}, \text{DAQ})$$

### Paired Query-Level Bootstrap Procedure
To compare the two dependent correlations sharing the same outcome across the identical 59 queries:
1. Resample 59 query IDs with replacement for $B = 2,000$ iterations (random seed = $42$).
2. Calculate $\rho_b(\text{Coverage}, \text{DAQ})$ and $\rho_b(\text{NDCG@K}, \text{DAQ})$.
3. Compute bootstrap difference $\Delta_b = \rho_b(\text{Coverage}, \text{DAQ}) - \rho_b(\text{NDCG@K}, \text{DAQ})$.
4. Report observed $\Delta$, bootstrap mean $\Delta$, bootstrap median $\Delta$, 95% percentile confidence interval, and the bootstrap directional probability $P(\Delta > 0)$.
   *(Note: $P(\Delta > 0)$ is an empirical bootstrap directional probability, NOT a hypothesis-test $p$-value).*

---

## 4. Configuration-Level Summaries with DAQ

The table below summarizes retrieval metrics, Evidence Coverage, component downstream dimensions, and the decoupled DAQ metric across all 15 configurations ($n = 59$ queries per row, 885 total conditions). Old Composite Quality is preserved strictly for historical comparison.

| Retriever | $K$ | NDCG@K | Evidence Coverage | Factual Consistency | Answer Relevance | **DAQ [95% Bootstrap CI]** | Old Composite Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 5 | 0.552 | 0.776 | 0.792 | 0.774 | **0.7826 [0.7360, 0.8208]** | 0.7805 |
| **BM25** | 10 | 0.529 | 0.811 | 0.827 | 0.817 | **0.8220 [0.8068, 0.8364]** | 0.8184 |
| **BM25** | 20 | 0.555 | 0.813 | 0.825 | 0.813 | **0.8186 [0.8068, 0.8301]** | 0.8167 |
| **Dense** | 5 | 0.423 | 0.804 | 0.823 | 0.803 | **0.8127 [0.8021, 0.8242]** | 0.8099 |
| **Dense** | 10 | 0.389 | 0.808 | 0.819 | 0.809 | **0.8144 [0.8030, 0.8258]** | 0.8124 |
| **Dense** | 20 | 0.426 | 0.807 | 0.815 | 0.809 | **0.8123 [0.8000, 0.8237]** | 0.8105 |
| **BM25 + Dense RRF** | 5 | 0.551 | 0.814 | 0.827 | 0.814 | **0.8203 [0.8064, 0.8339]** | 0.8184 |
| **BM25 + Dense RRF** | 10 | 0.525 | 0.814 | 0.823 | 0.817 | **0.8199 [0.8055, 0.8326]** | 0.8181 |
| **BM25 + Dense RRF** | 20 | 0.551 | 0.809 | 0.815 | 0.814 | **0.8144 [0.8051, 0.8246]** | 0.8127 |
| **BM25 + Dense + LSH RRF** | 5 | 0.553 | 0.807 | 0.820 | 0.807 | **0.8131 [0.7975, 0.8275]** | 0.8110 |
| **BM25 + Dense + LSH RRF** | 10 | 0.516 | 0.814 | 0.824 | 0.815 | **0.8195 [0.8064, 0.8309]** | 0.8175 |
| **BM25 + Dense + LSH RRF** | 20 | 0.544 | 0.813 | 0.817 | 0.811 | **0.8140 [0.8042, 0.8246]** | 0.8136 |
| **LSH** | 5 | 0.079 | 0.275 | 0.593 | 0.273 | **0.4331 [0.3483, 0.5153]** | 0.3805 |
| **LSH** | 10 | 0.073 | 0.291 | 0.636 | 0.286 | **0.4610 [0.3780, 0.5428]** | 0.4045 |
| **LSH** | 20 | 0.072 | 0.302 | 0.596 | 0.299 | **0.4475 [0.3585, 0.5305]** | 0.3989 |

---

## 5. Corrected H2 Query-Level Correlation Results

Spearman rank correlations evaluated across the 59 queries for each configuration:

| Configuration | $\rho(\text{Coverage}, \text{DAQ})$ | $\rho(\text{NDCG@K}, \text{DAQ})$ | $\Delta$ | Coverage $\to$ FC | Coverage $\to$ AR | NDCG $\to$ FC | NDCG $\to$ AR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (K=5)** | **+0.964** ($p < 0.0001$) | +0.456 ($p = 0.0003$) | **+0.508** | +0.885 | +0.987 | +0.412 | +0.447 |
| **BM25 (K=10)** | **+0.937** ($p < 0.0001$) | +0.409 ($p = 0.0013$) | **+0.527** | +0.869 | +0.937 | +0.347 | +0.409 |
| **BM25 (K=20)** | **+0.835** ($p < 0.0001$) | +0.145 ($p = 0.2725$) | **+0.690** | +0.742 | +0.936 | +0.093 | +0.254 |
| **Dense (K=5)** | **+0.805** ($p < 0.0001$) | +0.308 ($p = 0.0177$) | **+0.498** | +0.647 | +0.884 | +0.334 | +0.257 |
| **Dense (K=10)** | **+0.899** ($p < 0.0001$) | +0.322 ($p = 0.0129$) | **+0.577** | +0.776 | +0.897 | +0.207 | +0.323 |
| **Dense (K=20)** | **+0.864** ($p < 0.0001$) | +0.127 ($p = 0.3383$) | **+0.737** | +0.783 | +0.908 | +0.108 | +0.177 |
| **BM25+Dense RRF (K=5)** | **+0.904** ($p < 0.0001$) | +0.272 ($p = 0.0369$) | **+0.631** | +0.773 | +0.977 | +0.253 | +0.293 |
| **BM25+Dense RRF (K=10)**| **+0.865** ($p < 0.0001$) | +0.319 ($p = 0.0139$) | **+0.547** | +0.843 | +0.907 | +0.299 | +0.399 |
| **BM25+Dense RRF (K=20)**| **+0.865** ($p < 0.0001$) | +0.220 ($p = 0.0934$) | **+0.644** | +0.837 | +0.865 | +0.187 | +0.223 |
| **BM25+Dense+LSH RRF (K=5)** | **+0.946** ($p < 0.0001$) | +0.455 ($p = 0.0003$) | **+0.490** | +0.854 | +0.960 | +0.430 | +0.458 |
| **BM25+Dense+LSH RRF (K=10)**| **+0.814** ($p < 0.0001$) | +0.226 ($p = 0.0851$) | **+0.588** | +0.796 | +0.870 | +0.203 | +0.330 |
| **BM25+Dense+LSH RRF (K=20)**| **+0.911** ($p < 0.0001$) | +0.261 ($p = 0.0462$) | **+0.650** | +0.875 | +0.911 | +0.199 | +0.261 |
| **LSH (K=5)** | **+0.877** ($p < 0.0001$) | +0.627 ($p < 0.0001$) | **+0.250** | +0.075 | +0.992 | +0.037 | +0.711 |
| **LSH (K=10)** | **+0.895** ($p < 0.0001$) | +0.671 ($p < 0.0001$) | **+0.224** | -0.021 | +0.990 | -0.028 | +0.750 |
| **LSH (K=20)** | **+0.901** ($p < 0.0001$) | +0.699 ($p < 0.0001$) | **+0.202** | +0.101 | +0.988 | +0.081 | +0.772 |

### Key Correlation Insights
1. **Evidence Coverage Remains Highly Predictive:** Even when Evidence Coverage is completely removed from the target, its correlation with DAQ remains exceptionally strong ($\rho \approx +0.805 \text{ to } +0.964$, mean $+0.882$, all $p < 0.0001$).
2. **Conventional Metrics Show Marked Decay:** In viable competitive systems, $\rho(\text{NDCG@K}, \text{DAQ})$ is modest at $K=5$ ($\sim +0.27 \text{ to } +0.46$), decays at $K=10$ ($\sim +0.23 \text{ to } +0.41$), and drops toward statistical insignificance at $K=20$ (e.g., BM25: $\rho = +0.145, p = 0.273$; Dense: $\rho = +0.127, p = 0.338$).
3. **Difference $\Delta$ Widens with Depth:** The divergence $\Delta = \rho(\text{Coverage}, \text{DAQ}) - \rho(\text{NDCG}, \text{DAQ})$ widens systematically from $K=5$ ($\sim +0.50$) to $K=20$ ($\sim +0.69 \text{ to } +0.74$).

---

## 6. Paired Bootstrap Comparison Results ($B = 2,000$, Seed = $42$)

| Configuration | Observed $\Delta$ | Bootstrap Mean $\Delta$ | Bootstrap Median $\Delta$ | 95% Percentile CI | $P(\Delta > 0)$ | CI Excludes 0? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (K=5)** | **+0.508** | +0.510 | +0.501 | [+0.318, +0.730] | 100.0% | **Yes** |
| **BM25 (K=10)** | **+0.527** | +0.535 | +0.527 | [+0.320, +0.792] | 100.0% | **Yes** |
| **BM25 (K=20)** | **+0.690** | +0.688 | +0.682 | [+0.431, +0.966] | 100.0% | **Yes** |
| **Dense (K=5)** | **+0.498** | +0.498 | +0.493 | [+0.289, +0.713] | 100.0% | **Yes** |
| **Dense (K=10)** | **+0.577** | +0.580 | +0.576 | [+0.349, +0.853] | 100.0% | **Yes** |
| **Dense (K=20)** | **+0.737** | +0.735 | +0.730 | [+0.453, +1.051] | 100.0% | **Yes** |
| **BM25+Dense RRF (K=5)** | **+0.631** | +0.631 | +0.629 | [+0.405, +0.874] | 100.0% | **Yes** |
| **BM25+Dense RRF (K=10)**| **+0.547** | +0.548 | +0.541 | [+0.352, +0.777] | 100.0% | **Yes** |
| **BM25+Dense RRF (K=20)**| **+0.644** | +0.649 | +0.644 | [+0.417, +0.901] | 100.0% | **Yes** |
| **BM25+Dense+LSH RRF (K=5)** | **+0.490** | +0.494 | +0.489 | [+0.323, +0.699] | 100.0% | **Yes** |
| **BM25+Dense+LSH RRF (K=10)**| **+0.588** | +0.586 | +0.585 | [+0.363, +0.808] | 100.0% | **Yes** |
| **BM25+Dense+LSH RRF (K=20)**| **+0.650** | +0.655 | +0.652 | [+0.386, +0.938] | 100.0% | **Yes** |
| **LSH (K=5)** | **+0.250** | +0.253 | +0.251 | [+0.096, +0.427] | 100.0% | **Yes** |
| **LSH (K=10)** | **+0.224** | +0.227 | +0.223 | [+0.094, +0.369] | 100.0% | **Yes** |
| **LSH (K=20)** | **+0.202** | +0.203 | +0.199 | [+0.088, +0.337] | 100.0% | **Yes** |

*(Note: $P(\Delta > 0)$ is the empirical bootstrap directional probability across the 2,000 iterations, not a conventional p-value).*

---

## 7. Revised Hypothesis 2 Decision

### Decision Criteria Applied
- **SUPPORTED:** If Coverage $\to$ DAQ is consistently stronger than NDCG $\to$ DAQ, and paired bootstrap differences robustly favor Coverage with 95% CIs excluding zero.
- **PARTIALLY SUPPORTED:** If the effect is positive but inconsistent or uncertain.
- **NOT SUPPORTED:** If conventional retrieval metrics are comparable or stronger.

### Revised Conclusion: **SUPPORTED**
Across all 15 configurations, Evidence Coverage demonstrates a markedly stronger association with Downstream Answer Quality (DAQ) than conventional retrieval quality (NDCG@K alone):
1. **Consistency:** Observed $\Delta$ is positive across **15 of 15 configurations (100.0%)**.
2. **Bootstrap Robustness:** The 95% percentile bootstrap confidence intervals strictly exclude zero across **all 15 configurations**.
3. **Directional Certainty:** The bootstrap directional probability $P(\Delta > 0) = 1.0000$ across all 15 configurations.
4. **Dimension Analysis:** The association is driven by both Factual Consistency (mean $\rho \approx +0.76$) and Answer Relevance (mean $\rho \approx +0.92$), both of which exceed NDCG's associations.

---

## 8. LSH Robustness Analysis for DAQ

Repeating the configuration-level exploratory analysis across configuration means ($n = 15$ vs. non-LSH $n = 12$):

| Configuration Subset | Comparison | Spearman $\rho$ | Two-sided $p$-value | Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **All Configurations ($n = 15$)** | NDCG vs. DAQ | +0.4826 | 0.0685 | Modest positive trend driven by LSH anchor |
| **All Configurations ($n = 15$)** | Coverage vs. DAQ | **+0.9122** | < 0.0001 | Extremely strong linear coupling |
| **Non-LSH Subset ($n = 12$)** | NDCG vs. DAQ | **+0.0070** | **0.9828** | **Correlation completely collapses to zero** |
| **Non-LSH Subset ($n = 12$)** | Coverage vs. DAQ | **+0.8345** | **0.0007** | **Remains strong and highly significant** |

### Robustness Finding
When LSH is excluded, configuration-mean NDCG has **literally zero correlation with Downstream Answer Quality ($\rho = +0.0070, p = 0.983$)**. Among competitive retrieval engines, higher NDCG does not predict higher DAQ. In stark contrast, configuration-mean Evidence Coverage remains **massively predictive of DAQ ($\rho = +0.8345, p = 0.0007$)** even without LSH.

---

## 9. Reanalysis of BM25 vs. Dense Using DAQ

Comparing BM25 and Dense on the decoupled DAQ metric at $K=5, 10, 20$:

| Cutoff ($K$) | BM25 DAQ | Dense DAQ | Paired Difference $\Delta$ | Wilcoxon $W$ | Two-sided $p$-value |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 5$** | 0.7826 | **0.8127** | -0.0301 | 37.0 | 0.5145 |
| **$K = 10$** | **0.8220** | 0.8144 | +0.0076 | 21.0 | 0.2156 |
| **$K = 20$** | **0.8186** | 0.8123 | +0.0064 | 24.5 | 0.4193 |

### Descriptive Takeaways
- At $K=5$, Dense retrieval produced **higher observed downstream answer quality** (DAQ = $0.8127$) than BM25 (DAQ = $0.7826$).
- At $K=10$ and $K=20$, BM25 produced slightly higher observed DAQ, but the paired differences are small ($\Delta \le +0.0076$) and not statistically significant ($p > 0.20$).
- Despite BM25's clear superiority on lexical retrieval metrics (NDCG@10 $0.529$ vs. $0.389$), Dense retrieval achieved **downstream parity** on factuality and relevance across all retrieval depths.

---

## 10. Cutoff Depth ($K$) Effects on DAQ

Tracking DAQ across retrieval depths $K \in \{5, 10, 20\}$:

| Retrieval System | DAQ at $K=5$ | DAQ at $K=10$ | DAQ at $K=20$ | Trajectory Description |
| :--- | :---: | :---: | :---: | :--- |
| **BM25** | 0.7826 | **0.8220** | 0.8186 | Increases $5 \to 10$, lower at $K=20$ |
| **Dense** | 0.8127 | **0.8144** | 0.8123 | Stable across $K$, slightly lower at $K=20$ |
| **BM25 + Dense RRF** | **0.8203** | 0.8199 | 0.8144 | Peak at $K=5/10$, lower at $K=20$ |
| **BM25 + Dense + LSH RRF** | 0.8131 | **0.8195** | 0.8140 | Peak at $K=10$, lower at $K=20$ |
| **LSH** | 0.4331 | **0.4610** | 0.4475 | Peak at $K=10$, lower at $K=20$ |

### Observations on Context Depth
Across all five retrieval systems, Downstream Answer Quality at $K=20$ was observed to be lower than at $K=10$. This attenuation corresponds to a systematic drop in Factual Consistency (e.g., BM25 FC drops from $0.8271 \to 0.8246$; Dense FC drops from $0.8195 \to 0.8153$; BM25+Dense RRF FC drops from $0.8229 \to 0.8153$). While not demonstrating a causal mechanism, this pattern is consistent with a plausible context dilution effect, where deeper retrieval introduces peripheral passages that distract the generation model.

---

## 11. Methodological Limitations

1. **Automated Evaluator Proxy:** DAQ is evaluated automatically by Cohere `command-r-08-2024`. It measures factual consistency and answer relevance against reference text; it does **NOT** establish clinical correctness, diagnostic validity, medical safety, or clinician consensus.
2. **Shared Reference Context:** While Evidence Coverage is excluded from DAQ, reference evidence is still derived from grade-2 qrels. The evaluation remains observational within this corpus and prompt setup.
3. **Absence of Expert Clinical Adjudication:** External clinician annotations were not part of this experimental design.
