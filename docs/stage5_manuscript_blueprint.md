# Research Manuscript Blueprint (Task 5A)

**Working Title:** Do Conventional Retrieval Metrics Predict Answer Quality in Medical RAG? An Empirical Evaluation on Cardiovascular Guidelines  
**Project:** Cardiology-Chat Cardiovascular RAG Research Study  
**Document Type:** Fact-Locked Publication Blueprint & Synthesis Architecture  
**Status:** Frozen Experimental Synthesis (All Experiments Complete & Audited)  
**Target Venue:** Medical Informatics / Information Retrieval (e.g., *Journal of the American Medical Informatics Association*, *ACM Transactions on Information Systems*, or *BioNLP / EMNLP Clinical Track*)  
**Date:** 2026-10-07  
**Authoritative Sources:**  
1. [`docs/stage3_final_statistical_audit.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_final_statistical_audit.md)  
2. [`docs/stage3_h1_reconciliation.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h1_reconciliation.md)  
3. [`docs/stage3_h2_correction.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_h2_correction.md)  
4. [`experiments/results/stage3_h2_corrected.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_h2_corrected.json)  
5. [`experiments/results/stage3_statistical_analysis.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_statistical_analysis.json)  

---

## Executive Framing & Epistemological Boundaries

### Core Research Question
> *"How well do conventional retrieval metrics predict evidence sufficiency and downstream answer quality in cardiovascular RAG?"*

### Epistemological Rules for Synthesis
1. **Associational Language:** All statistical relationships must be framed as **associations**, not causal mechanisms. Words such as *"causes"*, *"governs"*, or *"determines"* are strictly prohibited.
2. **Empirical Evaluation Contribution:** The contribution is strictly an **empirical benchmark and methodological audit**. It does **NOT** present a novel retrieval algorithm, a new dense embedding model, a new fusion operator, or a validated clinical RAG system.
3. **No Clinical Efficacy Claims:** The study evaluates automated proxy dimensions scored by an LLM judge relative to reference text. It does **NOT** demonstrate clinical correctness, diagnostic accuracy, medical consensus, or patient safety.
4. **Decoupled Downstream Target (DAQ):** Downstream Answer Quality ($\text{DAQ} = (\text{Factual Consistency} + \text{Answer Relevance}) / 2.0$) strictly excludes Evidence Coverage to prevent circular part-whole artifacts.
5. **Shared-Reference Disclosure:** Conventional retrieval metrics and Evidence Coverage share an underlying reference evidence context derived from the same qrels.
6. **Reference Provenance:** Passages are described exclusively as *"protocol-annotated grade-2 reference evidence passages"*—never as "expert-adjudicated".

---

## Blueprint Architecture by Manuscript Section

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MANUSCRIPT BLUEPRINT                            │
├────────────────────────────────────────────────────────────────────────┤
│ Title & Structured Abstract                                            │
│ 1. Introduction                                                        │
│ 2. Research Questions & Formal Hypotheses                              │
│ 3. Related Work & Methodological Foundations                           │
│ 4. Materials & Experimental Methods                                    │
│    4.1 Cardiovascular Corpus (Corpus V1)                               │
│    4.2 Clinical Query Benchmark & Exclusion Protocol                   │
│    4.3 Protocol-Annotated Reference Evidence                           │
│    4.4 Retrieval Architectures                                         │
│    4.5 RAG Generation Protocol                                         │
│    4.6 Cross-Family Downstream Evaluation                              │
│    4.7 Decoupled Statistical Testing & Paired Bootstrap               │
│ 5. Results                                                             │
│    5.1 Retrieval Engine Performance Benchmark                          │
│    5.2 Hypothesis 1: Retrieval Quality vs. Evidence Coverage           │
│    5.3 Hypothesis 2: Evidence Coverage vs. Downstream Quality (DAQ)     │
│    5.4 Lexical vs. Dense Divergence: BM25 vs. Dense Parity             │
│    5.5 Retrieval Cutoff Dynamics: Non-Monotonic Depth Attenuation      │
│    5.6 Exploratory Robustness: Disentangling LSH Failure Floors        │
│ 6. Discussion                                                          │
│ 7. Methodological Limitations                                          │
│ 8. Conclusion & Future Directions                                      │
│ 9. Comprehensive Publication Tables (Tables 1–8)                       │
│ 10. Publication Figure Architecture (Figures 1–6)                      │
│ 11. Final Frozen Numbers Compendium                                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Structured Abstract

- **Background:** Retrieval-augmented generation (RAG) is widely applied to clinical reasoning, under the implicit assumption that optimizing standard information retrieval (IR) metrics (e.g., $\text{NDCG@K}$) ensures sufficient evidence and faithful downstream answers.
- **Objective:** Empirically investigate how well conventional IR metrics associate with reference-evidence coverage and downstream automated answer quality in a cardiovascular guideline corpus.
- **Methods:** Evaluated 5 retrieval architectures (lexical BM25, dense `all-MiniLM-L6-v2`, locality-sensitive hashing [LSH], and Reciprocal Rank Fusion [RRF] hybrids) across 3 cutoffs ($K \in \{5, 10, 20\}$) on 59 guideline queries with protocol-annotated reference evidence (885 conditions). Generation used `gemini-3.1-flash-lite`. Downstream answer quality was evaluated by cross-family LLM judge `command-r-08-2024` on Evidence Coverage, Factual Consistency, and Answer Relevance. To eliminate structural dependency, Downstream Answer Quality ($\text{DAQ}$) was defined as the average of Factual Consistency and Answer Relevance, completely decoupling Evidence Coverage from the outcome.
- **Results:**
  - **H1:** $\text{NDCG@K}$ was positively associated with Evidence Coverage across all 15 configurations (global mean $\rho = +0.421$, range $+0.213$ to $+0.791$; competitive systems mean $\rho = +0.335$, range $+0.213$ to $+0.447$).
  - **H2:** Evidence Coverage showed a markedly stronger association with DAQ ($\rho = +0.805$ to $+0.964$, mean $+0.885$) than $\text{NDCG@K}$ ($\rho = +0.127$ to $+0.699$, mean $+0.368$). Paired bootstrap testing ($B=2,000$) showed $\Delta = \rho(\text{Coverage}, \text{DAQ}) - \rho(\text{NDCG}, \text{DAQ}) > 0$ in 15 of 15 configurations ($P(\Delta > 0) = 1.000$, all 95% CIs exclude zero).
  - **Lexical vs. Dense Parity:** Despite BM25's conventional retrieval superiority ($\text{NDCG@10} = 0.529$ vs. $0.389$), Dense retrieval achieved downstream DAQ parity ($0.814$ vs. $0.822$, $p = 0.216$).
  - **Depth Trajectory:** Across all five retrieval architectures, DAQ was lower at $K=20$ than at $K=10$, mirroring systematic declines in factual consistency.
- **Conclusions:** In cardiovascular RAG, conventional retrieval metrics explain reference evidence coverage only moderately and decouple from downstream answer quality among competitive systems. Downstream evaluation and passage evidence coverage are essential beyond standard ranking scores.

---

### Section 1: Introduction

#### Key Claims
1. RAG is increasingly deployed in biomedical and cardiovascular reasoning to ground LLMs in verified literature.
2. Clinical RAG pipelines typically optimize retrieval using rank-based IR metrics ($\text{NDCG@K}$, $\text{MAP}$, $\text{MRR}$) assuming these optimize the generation context.
3. Ranking metrics evaluate passage positions in a ranked list; generation models, however, condition on the concatenated context set as an unordered or flat context window.
4. This study investigates whether ranking superiority translates to evidence sufficiency and downstream generation quality.

#### Exact Supporting Numbers & Sources
- 5 retrieval architectures, 3 cutoffs ($K \in \{5, 10, 20\}$), 59 primary clinical queries, 885 RAG conditions. Source: [`docs/stage3_evaluator_protocol.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_evaluator_protocol.md).

#### Caveats & Boundaries
- Must not claim RAG is unsafe or broken; frame as a metric alignment evaluation.
- Frame LLM downstream evaluation strictly as automated surrogate quality proxies.

---

### Section 2: Research Questions & Hypotheses

#### Core Formulation
- **Primary Research Question:** *"How well do conventional retrieval metrics predict evidence sufficiency and downstream answer quality in cardiovascular RAG?"*

#### Formal Hypotheses
- **Hypothesis 1 (H1 — Associational):** *Higher conventional retrieval quality ($\text{NDCG@K}$) is positively associated with reference-evidence coverage.*
- **Hypothesis 2 (H2 — Associational, Decoupled):** *Reference-evidence coverage has a stronger association with downstream answer quality than conventional retrieval quality alone.*

#### Metric Decoupling Formula
$$\mathbf{DAQ} = \frac{\text{Factual Consistency} + \text{Answer Relevance}}{2.0}$$
- **Methodological Rule:** Evidence Coverage is strictly decoupled from DAQ. The historical composite metric ($\text{Composite Quality} = (\text{Coverage} + \text{Consistency} + \text{Relevance}) / 3$) is documented as retracted due to circular part-whole correlation artifacts.

---

### Section 3: Related Work

#### Foundational Themes
1. **Biomedical RAG & Guideline Adherence:** Prior work in clinical question answering (MedQA, PubMedQA, BioASQ) and clinical guideline grounding.
2. **The Retrieval-Generation Gap in RAG:** Studies showing discrepancies between retrieval benchmark scores and downstream task performance (e.g., "Lost in the Middle", context dilution effects).
3. **Automated LLM-as-a-Judge Evaluation:** Validation of LLM evaluators, self-preference mitigation, and multi-dimensional quality decomposition.
4. **Methodological Independence in Benchmark Design:** Avoiding circular target metrics in empirical IR evaluations.

---

### Section 4: Materials and Methods

#### 4.1 Cardiovascular Corpus (Corpus V1)
- **Document Count:** 22 peer-reviewed clinical guidelines and cardiovascular reviews under open CC BY / CC BY 4.0 licenses.
- **Corpus Volume:** 387 total pages, 248,894 cleaned tokens, 1,142 chunks.
- **Chunking Pipeline:** Target chunk size of 256 tokens, overlap of 32 tokens, sentence-boundary aligned chunking.
- **Thin Source Limitation:** HTN02 (2-page editorial) contributes few chunks, leading to sparse candidate density for hypertension queries.
- **Artifact Source:** [`data/evaluation/evaluation_dataset_report.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/data/evaluation/evaluation_dataset_report.md).

#### 4.2 Query Benchmark & Cohort Stratification
- **Total Custom Queries:** 65 custom-authored queries spanning 9 clinical cardiovascular domains (heart failure, atrial fibrillation, myocardial infarction, coronary artery disease, hypertension, cardiomyopathy, dyslipidemia, valvular disease, prevention).
- **Primary Downstream Cohort ($n = 59$):** Queries possessing protocol-verified grade-2 reference evidence passages.
- **Excluded Sensitivity Queries ($n = 6$):** Six queries (`Q007`, `Q013`, `Q017`, `Q021`, `Q055`, `Q061`) possessed only grade-1 (partially relevant) passages. These were strictly excluded from primary downstream evaluation to prevent confounding by incomplete reference ground truth.
- **Total Primary Experimental Units:** $59 \text{ queries} \times 5 \text{ retrievers} \times 3 \text{ cutoffs} = 885 \text{ conditions}$.

#### 4.3 Reference Evidence & Qrels Provenance
- **Accurate Terminology:** *"protocol-annotated grade-2 reference evidence passages"* or *"guideline-based grade-2 reference passages"*.
- **Construction Protocol:** BM25 top-20 pooling combined with keyword pattern scanning, verified against chunk text under formal annotation guidelines ([`data/evaluation/annotation_guidelines.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/data/evaluation/annotation_guidelines.md)).
- **Limitation Disclosure:** Qrels were not established through independent prospective clinical adjudication; future work requires formal clinician annotation panels.

#### 4.4 Retrieval Architectures
1. **BM25:** Lexical BM25 using standard Okapi BM25 parameters ($k_1 = 1.5, b = 0.75$).
2. **Dense:** Semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, FAISS `IndexFlatIP` over $L_2$-normalized embeddings).
   *(CRITICAL: Formally noted that earlier documentation erroneously referred to PubMedBERT; all-MiniLM-L6-v2 is the verified frozen model).*
3. **LSH:** Locality-Sensitive Hashing with MinHash (128 permutation hashes, 16 bands $\times$ 8 rows).
4. **BM25 + Dense RRF:** Reciprocal Rank Fusion ($k_{rrf} = 60$) combining BM25 and Dense rankings.
5. **BM25 + Dense + LSH RRF:** Three-way RRF ($k_{rrf} = 60$) combining BM25, Dense, and LSH.

#### 4.5 Generation Protocol
- **Generator Model:** Google `gemini-3.1-flash-lite`.
- **Inference Configuration:** Temperature $= 0.0$, top-$p = 1.0$, max output $= 512$ tokens.
- **Strict Blindness:** Generator receives query and top-$K$ passages; blind to retriever identity, ranking scores, qrels, and reference evidence. Exactly 975/975 conditions generated with zero errors.

#### 4.6 Downstream Evaluation Protocol
- **Evaluator Model:** Cohere `command-r-08-2024` (prompt version `v1.0`, temperature $= 0.0$).
- **Cross-Family Independence:** Evaluator belongs to a separate architectural family from the generator.
  *(Methodological Phrasing: "The use of models from different providers/model families reduces the risk of direct intra-family evaluator-generator preference effects." Avoids claiming guaranteed bias absence).*
- **Evaluator Blindness:** Receives strictly: (1) Query text, (2) Generated answer text, (3) Reference evidence text. Blind to retriever, cutoff $K$, retrieval ranks, and generator identity.
- **Scored Dimensions:** Evidence Coverage $[0, 1]$, Factual Consistency $[0, 1]$, Answer Relevance $[0, 1]$.

#### 4.7 Statistical Analysis Framework
- **Primary Inferential Unit:** The query-level observation within each of the 15 experimental configurations ($n = 59$).
- **H1 Evaluation:** Query-level Spearman rank correlation $\rho(\text{NDCG@K}, \text{Evidence Coverage})$.
- **H2 Evaluation:** Dependent correlation contrast $\Delta = \rho(\text{Coverage}, \text{DAQ}) - \rho(\text{NDCG@K}, \text{DAQ})$.
- **Paired Bootstrap:** $B = 2,000$ resamples (seed $= 42$), resampling identical 59 query IDs jointly across Evidence Coverage, NDCG, and DAQ. Reports bootstrap mean, median, 95% percentile CI, and bootstrap directional probability $P(\Delta > 0)$.
- **Pairwise System Tests:** Non-parametric Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction.

---

### Section 5: Results

#### 5.1 Retrieval Performance Benchmark
- **Finding (DIRECTLY SUPPORTED):** BM25 demonstrates the strongest conventional lexical retrieval performance among individual architectures ($\text{NDCG@5} = 0.5522$, $\text{NDCG@10} = 0.5287$, $\text{NDCG@20} = 0.5552$).
- **Dense Retrieval:** Achieves moderate ranking scores ($\text{NDCG@5} = 0.4227$, $\text{NDCG@10} = 0.3889$, $\text{NDCG@20} = 0.4260$).
- **Hybrid RRF:** Yields competitive retrieval performance comparable to BM25 alone ($\text{NDCG@5} = 0.5514$, $\text{NDCG@10} = 0.5248$, $\text{NDCG@20} = 0.5515$).
- **LSH Floor Effect:** LSH exhibits severe retrieval failure floors ($\text{NDCG@K} \approx 0.072$–$0.079$, $\text{Recall@20} = 0.0607$).

#### 5.2 Hypothesis 1 Results: NDCG vs. Evidence Coverage
- **Finding (ASSOCIATIONAL):** Higher conventional retrieval quality is positively associated with Evidence Coverage across all 15 configurations ($100.0\%$ positive).
- **Global Population ($n = 15$ configs):** Mean $\rho = \mathbf{+0.4210}$, median $\rho = \mathbf{+0.3637}$, range $\mathbf{+0.2133 \text{ to } +0.7913}$. 13 of 15 configurations statistically significant ($p < 0.05$).
- **Viable Retrieval Systems ($n = 12$ configs, excluding LSH):** Mean $\rho = \mathbf{+0.3345}$, median $\rho = \mathbf{+0.3341}$, range $\mathbf{+0.2133 \text{ to } +0.4473}$. 10 of 12 configurations significant ($p < 0.05$).
- **LSH Inflation Mechanism:** LSH correlations ($\rho = +0.7255 \text{ to } +0.7913$) are inflated by joint zero-hit floor effects (at $K=20$, 35 of 59 queries tie at joint zero for both NDCG and Coverage).

#### 5.3 Hypothesis 2 Results: Evidence Coverage vs. DAQ
- **Finding (ASSOCIATIONAL):** Reference-evidence coverage exhibits a dramatically stronger association with downstream answer quality than conventional retrieval metrics alone.
- **Coverage $\to$ DAQ:** $\rho$ ranges from $\mathbf{+0.8053 \text{ to } +0.9644}$ (mean $\mathbf{+0.8854}$, all $p < 0.0001$).
- **NDCG $\to$ DAQ:** $\rho$ ranges from $\mathbf{+0.1269 \text{ to } +0.6989}$ (global mean $\mathbf{+0.3678}$; viable systems mean $\mathbf{+0.2850}$).
- **Observed Difference ($\Delta$):** Positive in **15 of 15 configurations (100.0%)**, ranging from $+0.2017$ to $+0.7370$ (mean $\mathbf{+0.5175}$).
- **Paired Bootstrap:** $P(\Delta > 0) = \mathbf{1.0000}$ in all 15 configurations; 95% percentile CIs strictly exclude zero (lowest lower bound $+0.0883$, highest upper bound $+1.0505$).

#### 5.4 Lexical vs. Dense Downstream Divergence
- **Finding (DIRECTLY SUPPORTED):** Despite BM25's clear superiority on lexical retrieval metrics ($\text{NDCG@10} = 0.5287$ vs. $0.3889$), Dense retrieval achieved downstream DAQ parity across all cutoffs:
  - $K=5$: $\text{BM25} = 0.7826$, $\text{Dense} = 0.8127$, $\Delta = -0.0301$, Wilcoxon $W = 106.5$, $p = 0.5145$.
  - $K=10$: $\text{BM25} = 0.8220$, $\text{Dense} = 0.8144$, $\Delta = +0.0076$, Wilcoxon $W = 28.0$, $p = 0.2156$.
  - $K=20$: $\text{BM25} = 0.8186$, $\text{Dense} = 0.8123$, $\Delta = +0.0064$, Wilcoxon $W = 52.5$, $p = 0.4193$.
- **Interpretation:** No statistically significant difference was detected; ranking metric deficits in dense semantic retrieval do not impair downstream answer fidelity.

#### 5.5 Retrieval Cutoff Depth Dynamics
- **Finding (DIRECTLY SUPPORTED):** Downstream answer quality does not monotonically improve with retrieval depth. Across all five retrieval architectures, DAQ was lower at $K=20$ than at $K=10$:
  - BM25: $0.8220 \to 0.8186$ ($-0.0034$)
  - Dense: $0.8144 \to 0.8123$ ($-0.0021$)
  - BM25 + Dense RRF: $0.8199 \to 0.8144$ ($-0.0055$)
  - BM25 + Dense + LSH RRF: $0.8195 \to 0.8140$ ($-0.0055$)
  - LSH: $0.4610 \to 0.4475$ ($-0.0135$)
- **Underlying Driver:** Systematic attenuation in Factual Consistency (e.g., BM25 FC drops $0.8271 \to 0.8246$; Dense FC drops $0.8195 \to 0.8153$).
- **Hypothesis (EXPLORATORY):** Consistent with context dilution effects, where peripheral passages at $K=20$ distract the generator without providing marginal useful evidence.

#### 5.6 Exploratory Robustness: Disentangling LSH Floors
- **Finding (EXPLORATORY):** Configuration-level correlations across configuration averages ($n = 15$ vs. non-LSH $n = 12$):
  - All 15 Configs: $\text{NDCG vs. DAQ } \rho = +0.4826$ ($p = 0.0685$); $\text{Coverage vs. DAQ } \rho = +0.9122$ ($p < 0.0001$).
  - Non-LSH 12 Configs: $\text{NDCG vs. DAQ } \rho = \mathbf{+0.0070}$ ($p = 0.9828$); $\text{Coverage vs. DAQ } \rho = \mathbf{+0.8345}$ ($p = 0.0007$).
- **Interpretation:** The apparent configuration-level association between NDCG and DAQ completely collapses to zero among viable systems; Evidence Coverage remains robustly coupled to answer quality.

---

### Section 6: Discussion

#### Principal Interpretations
1. **The Retrieval-Generation Metric Mismatch:** IR metrics prioritize top ranks, but LLM generators consume passage sets. Once sufficient reference evidence is captured in context, ranking variations have negligible impact on generation fidelity.
2. **Dense Semantic Retrieval Parity:** Dense retrieval captures semantically paraphrased guideline passages that receive lower lexical overlap scores but successfully provide sufficient evidence for generation.
3. **The Danger of High-K Retrieval:** Deeper retrieval cutoffs ($K=20$) increase recall marginally but introduce distractors, systematically attenuating factual consistency.
4. **Methodological Independence in Benchmark Evaluations:** Highlights the critical necessity of decoupling evaluative targets (avoiding circular part-whole composite scores).

---

### Section 7: Methodological Limitations

1. **Shared-Reference Dependency:** Reference evidence and retrieval relevance judgments rely on the same qrels passages, establishing a shared-reference association rather than fully independent external validation.
2. **Protocol-Annotated Qrels:** Qrels were derived from BM25 pooling and keyword scanning co-verified by rule-based heuristics; they were not independently adjudicated by an external clinician consensus panel.
3. **Automated Evaluator Proxies:** LLM evaluator scores quantify textual adherence to reference passages; they do not establish clinical correctness, safety, or diagnostic validity.
4. **Custom Benchmark Scale:** 65 custom queries and 22 guideline papers; findings require replication on larger multi-specialty corpora.
5. **Observational Evidence:** Correlations demonstrate association, not causal mechanisms.
6. **Cross-Family LLM Bias:** Cross-family separation reduces intra-family preference but does not guarantee complete absence of evaluator bias.
7. **LSH Failure-Floor Confounding:** Global correlation statistics are sensitive to LSH floor effects.

---

### Section 8: Conclusion

- Conventional ranking metrics are incomplete surrogates for downstream generation quality in clinical guideline RAG.
- Once competitive retrieval engines are considered, ranking differences have negligible association with downstream answer quality, while evidence coverage remains strongly coupled.
- Clinical RAG evaluation frameworks must incorporate passage evidence sufficiency metrics and decoupled downstream generation auditing rather than relying solely on conventional IR ranking metrics.

---

## Publication Tables Plan (Tables 1–8)

### Table 1: Cardiovascular Corpus V1 Characteristics
- **Purpose:** Document corpus volume, chunking schema, and guideline coverage.
- **Data Content:**
  - Papers: 22 documents (CC BY / CC BY 4.0).
  - Pages: 387.
  - Chunks: 1,142 chunks.
  - Cleaned Tokens: 248,894.
  - Target Chunk Size: 256 tokens (32 overlap).
  - Source Note: HTN02 editorial thin-source documentation.

### Table 2: Evaluated Retrieval Architectures
- **Purpose:** Formally document the 5 retrieval configurations.
- **Data Content:**
  - BM25: Okapi BM25 ($k_1=1.5, b=0.75$).
  - Dense: `all-MiniLM-L6-v2` (384-dim, FAISS `IndexFlatIP`).
  - LSH: MinHash LSH (128 hashes, 16 bands $\times$ 8 rows).
  - BM25+Dense RRF: $k_{rrf}=60$.
  - BM25+Dense+LSH RRF: $k_{rrf}=60$.

### Table 3: Conventional Retrieval Benchmark Metrics ($n = 59$ Primary Queries)
- **Purpose:** Authoritative retrieval ranking performance across cutoffs.
- **Data Content (Exact Frozen Numbers):**

| Retriever | $K$ | NDCG@K | Recall@K | Precision@K | MRR | MAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 5 | 0.5522 | 0.2357 | 0.5424 | 0.7506 | 0.2019 |
| **BM25** | 10 | 0.5287 | 0.3846 | 0.4627 | 0.7587 | 0.2980 |
| **BM25** | 20 | 0.5552 | 0.5537 | 0.3492 | 0.7597 | 0.3844 |
| **Dense** | 5 | 0.4227 | 0.1869 | 0.4508 | 0.5975 | 0.1388 |
| **Dense** | 10 | 0.3889 | 0.2840 | 0.3542 | 0.6036 | 0.1905 |
| **Dense** | 20 | 0.4260 | 0.4484 | 0.2907 | 0.6090 | 0.2547 |
| **BM25+Dense RRF** | 5 | 0.5514 | 0.2349 | 0.5593 | 0.7898 | 0.1966 |
| **BM25+Dense RRF** | 10 | 0.5248 | 0.3833 | 0.4644 | 0.7927 | 0.2881 |
| **BM25+Dense RRF** | 20 | 0.5515 | 0.5574 | 0.3466 | 0.7938 | 0.3706 |
| **BM25+Dense+LSH RRF** | 5 | 0.5530 | 0.2401 | 0.5661 | 0.7780 | 0.1976 |
| **BM25+Dense+LSH RRF** | 10 | 0.5165 | 0.3744 | 0.4559 | 0.7827 | 0.2835 |
| **BM25+Dense+LSH RRF** | 20 | 0.5441 | 0.5482 | 0.3364 | 0.7827 | 0.3615 |
| **LSH** | 5 | 0.0785 | 0.0338 | 0.0644 | 0.1794 | 0.0254 |
| **LSH** | 10 | 0.0730 | 0.0493 | 0.0559 | 0.1843 | 0.0315 |
| **LSH** | 20 | 0.0718 | 0.0607 | 0.0381 | 0.1843 | 0.0358 |

### Table 4: Hypothesis 1 Correlation Matrix (NDCG@K vs. Evidence Coverage)
- **Purpose:** Full query-level Spearman correlation evaluation of H1.
- **Data Content (Exact Frozen Numbers):**

| Retriever | $K$ | $n$ | Mean NDCG@K | Mean Coverage | Spearman $\rho$ | $p$-value | Significant ($p < 0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 5 | 59 | 0.5522 | 0.7763 | **+0.4473** | 0.00038 | Yes |
| **BM25** | 10 | 59 | 0.5287 | 0.8110 | **+0.4167** | 0.00103 | Yes |
| **BM25** | 20 | 59 | 0.5552 | 0.8127 | **+0.3206** | 0.01332 | Yes |
| **Dense** | 5 | 59 | 0.4227 | 0.8042 | **+0.3477** | 0.00697 | Yes |
| **Dense** | 10 | 59 | 0.3889 | 0.8085 | **+0.3637** | 0.00463 | Yes |
| **Dense** | 20 | 59 | 0.4260 | 0.8068 | **+0.2133** | 0.10474 | No |
| **BM25+Dense RRF** | 5 | 59 | 0.5514 | 0.8144 | **+0.2608** | 0.04601 | Yes |
| **BM25+Dense RRF** | 10 | 59 | 0.5248 | 0.8144 | **+0.4062** | 0.00141 | Yes |
| **BM25+Dense RRF** | 20 | 59 | 0.5515 | 0.8093 | **+0.2732** | 0.03628 | Yes |
| **BM25+Dense+LSH RRF** | 5 | 59 | 0.5530 | 0.8068 | **+0.4328** | 0.00062 | Yes |
| **BM25+Dense+LSH RRF** | 10 | 59 | 0.5165 | 0.8136 | **+0.3081** | 0.01762 | Yes |
| **BM25+Dense+LSH RRF** | 20 | 59 | 0.5441 | 0.8127 | **+0.2239** | 0.08829 | No |
| **LSH** | 5 | 59 | 0.0785 | 0.2754 | **+0.7255** | < 0.00001 | Yes |
| **LSH** | 10 | 59 | 0.0730 | 0.2915 | **+0.7839** | < 0.00001 | Yes |
| **LSH** | 20 | 59 | 0.0718 | 0.3017 | **+0.7913** | < 0.00001 | Yes |

### Table 5: Downstream Generation Quality Across Configurations (DAQ)
- **Purpose:** Document decoupled downstream answer quality components.
- **Data Content (Exact Frozen Numbers):**

| Retriever | $K$ | Factual Consistency | Answer Relevance | **DAQ Mean [95% Bootstrap CI]** | Old Composite Quality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 5 | 0.7915 | 0.7737 | **0.7826 [0.7360, 0.8208]** | 0.7805 |
| **BM25** | 10 | 0.8271 | 0.8169 | **0.8220 [0.8068, 0.8364]** | 0.8184 |
| **BM25** | 20 | 0.8246 | 0.8127 | **0.8186 [0.8068, 0.8301]** | 0.8167 |
| **Dense** | 5 | 0.8229 | 0.8025 | **0.8127 [0.8021, 0.8242]** | 0.8099 |
| **Dense** | 10 | 0.8195 | 0.8093 | **0.8144 [0.8030, 0.8258]** | 0.8124 |
| **Dense** | 20 | 0.8153 | 0.8093 | **0.8123 [0.8000, 0.8237]** | 0.8105 |
| **BM25+Dense RRF** | 5 | 0.8267 | 0.8139 | **0.8203 [0.8064, 0.8339]** | 0.8184 |
| **BM25+Dense RRF** | 10 | 0.8229 | 0.8169 | **0.8199 [0.8055, 0.8326]** | 0.8181 |
| **BM25+Dense RRF** | 20 | 0.8153 | 0.8136 | **0.8144 [0.8051, 0.8246]** | 0.8127 |
| **BM25+Dense+LSH RRF** | 5 | 0.8195 | 0.8068 | **0.8131 [0.7975, 0.8275]** | 0.8110 |
| **BM25+Dense+LSH RRF** | 10 | 0.8237 | 0.8153 | **0.8195 [0.8064, 0.8309]** | 0.8175 |
| **BM25+Dense+LSH RRF** | 20 | 0.8169 | 0.8110 | **0.8140 [0.8042, 0.8246]** | 0.8136 |
| **LSH** | 5 | 0.5932 | 0.2729 | **0.4331 [0.3483, 0.5153]** | 0.3805 |
| **LSH** | 10 | 0.6364 | 0.2856 | **0.4610 [0.3780, 0.5428]** | 0.4045 |
| **LSH** | 20 | 0.5958 | 0.2992 | **0.4475 [0.3585, 0.5305]** | 0.3989 |

### Table 6: Hypothesis 2 Paired Bootstrap Comparison (Coverage vs. NDCG $\to$ DAQ)
- **Purpose:** Primary inferential test of H2 across all configurations.
- **Data Content (Exact Frozen Numbers):**

| Configuration | $\rho(\text{Coverage}, \text{DAQ})$ | $\rho(\text{NDCG}, \text{DAQ})$ | Observed $\Delta$ | Bootstrap Mean $\Delta$ | 95% Percentile CI | $P(\Delta > 0)$ | CI Excludes 0? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 ($K=5$)** | +0.9644 | +0.4562 | **+0.5082** | +0.5102 | [+0.3178, +0.7298] | 1.0000 | Yes |
| **BM25 ($K=10$)** | +0.9366 | +0.4094 | **+0.5272** | +0.5350 | [+0.3204, +0.7922] | 1.0000 | Yes |
| **BM25 ($K=20$)** | +0.8352 | +0.1452 | **+0.6900** | +0.6876 | [+0.4309, +0.9660] | 1.0000 | Yes |
| **Dense ($K=5$)** | +0.8053 | +0.3077 | **+0.4976** | +0.4984 | [+0.2891, +0.7134] | 1.0000 | Yes |
| **Dense ($K=10$)** | +0.8992 | +0.3223 | **+0.5769** | +0.5804 | [+0.3491, +0.8529] | 1.0000 | Yes |
| **Dense ($K=20$)** | +0.8639 | +0.1269 | **+0.7370** | +0.7351 | [+0.4533, +1.0505] | 1.0000 | Yes |
| **BM25+Dense RRF ($K=5$)** | +0.9037 | +0.2723 | **+0.6313** | +0.6310 | [+0.4048, +0.8741] | 1.0000 | Yes |
| **BM25+Dense RRF ($K=10$)**| +0.8655 | +0.3188 | **+0.5467** | +0.5480 | [+0.3516, +0.7774] | 1.0000 | Yes |
| **BM25+Dense RRF ($K=20$)**| +0.8646 | +0.2201 | **+0.6445** | +0.6489 | [+0.4172, +0.9011] | 1.0000 | Yes |
| **BM25+Dense+LSH RRF ($K=5$)** | +0.9455 | +0.4552 | **+0.4903** | +0.4941 | [+0.3234, +0.6987] | 1.0000 | Yes |
| **BM25+Dense+LSH RRF ($K=10$)**| +0.8140 | +0.2260 | **+0.5879** | +0.5862 | [+0.3629, +0.8078] | 1.0000 | Yes |
| **BM25+Dense+LSH RRF ($K=20$)**| +0.9109 | +0.2613 | **+0.6496** | +0.6552 | [+0.3857, +0.9382] | 1.0000 | Yes |
| **LSH ($K=5$)** | +0.8771 | +0.6273 | **+0.2498** | +0.2526 | [+0.0964, +0.4267] | 1.0000 | Yes |
| **LSH ($K=10$)** | +0.8951 | +0.6706 | **+0.2245** | +0.2269 | [+0.0944, +0.3688] | 1.0000 | Yes |
| **LSH ($K=20$)** | +0.9007 | +0.6989 | **+0.2017** | +0.2033 | [+0.0883, +0.3366] | 1.0000 | Yes |

### Table 7: BM25 vs. Dense Downstream Parity Analysis
- **Purpose:** Statistical contrast of BM25 and Dense on DAQ.
- **Data Content (Exact Frozen Numbers):**

| Cutoff ($K$) | BM25 Mean DAQ | Dense Mean DAQ | Mean Paired Difference ($\text{BM25} - \text{Dense}$) | Wilcoxon Statistic ($W$) | Two-Sided $p$-value | Significant ($p < 0.05$)? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 5$** | 0.7826 | 0.8127 | -0.0301 | 106.5 | 0.51455 | No |
| **$K = 10$** | 0.8220 | 0.8144 | +0.0076 | 28.0 | 0.21564 | No |
| **$K = 20$** | 0.8186 | 0.8123 | +0.0064 | 52.5 | 0.41925 | No |

### Table 8: Retrieval Depth Trajectory Analysis ($K \in \{5, 10, 20\}$)
- **Purpose:** Document non-monotonic downstream answer quality across cutoffs.
- **Data Content (Exact Frozen Numbers):**

| Retrieval Architecture | DAQ at $K=5$ | DAQ at $K=10$ | DAQ at $K=20$ | Peak Depth | Lower at $K=20$ than $K=10$? | $\Delta$ Factual Consistency ($K=10 \to K=20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 0.7826 | **0.8220** | 0.8186 | $K=10$ | **Yes** ($-0.0034$) | $0.8271 \to 0.8246$ ($-0.0025$) |
| **Dense** | 0.8127 | **0.8144** | 0.8123 | $K=10$ | **Yes** ($-0.0021$) | $0.8195 \to 0.8153$ ($-0.0042$) |
| **BM25 + Dense RRF** | **0.8203** | 0.8199 | 0.8144 | $K=5$ | **Yes** ($-0.0055$) | $0.8229 \to 0.8153$ ($-0.0076$) |
| **BM25 + Dense + LSH RRF** | 0.8131 | **0.8195** | 0.8140 | $K=10$ | **Yes** ($-0.0055$) | $0.8237 \to 0.8169$ ($-0.0068$) |
| **LSH** | 0.4331 | **0.4610** | 0.4475 | $K=10$ | **Yes** ($-0.0135$) | $0.6364 \to 0.5958$ ($-0.0406$) |

---

## Publication Figures Plan (Figures 1–6)

### Figure 1: Overall Experimental Pipeline Architecture
- **Description:** System diagram illustrating: (1) Corpus V1 ingestion & chunking; (2) 5-system retrieval generating top-$K$ candidate sets; (3) Generator (`gemini-3.1-flash-lite`, $T=0$) generating clinical answers; (4) Cross-family Evaluator (`command-r-08-2024`, $T=0$) scoring Coverage, Consistency, and Relevance; (5) Decoupled metric derivation ($\text{DAQ} = (\text{FC} + \text{AR}) / 2$).
- **Visual Distinction:** Highlights strict blindness barriers and metric decoupling boundaries.

### Figure 2: Conventional Retrieval Performance Across Cutoffs
- **Reference File:** [`experiments/results/stage3_figures/fig4_heatmaps_retriever_k.png`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_figures/fig4_heatmaps_retriever_k.png)
- **Description:** Multi-panel bar/heatmap illustrating NDCG@K, Recall@K, and Precision@K across the 5 architectures at $K \in \{5, 10, 20\}$.
- **Visual Distinction:** LSH clearly demarcated as a baseline anchor demonstrating retrieval floor effects.

### Figure 3: Hypothesis 1 — NDCG@K vs. Evidence Coverage Association
- **Reference File:** [`experiments/results/stage3_figures/fig1_ndcg_vs_coverage_config.png`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_figures/fig1_ndcg_vs_coverage_config.png)
- **Description:** Scatter plots of query-level $\text{NDCG@K}$ vs. Evidence Coverage across configurations, showing regression lines and Spearman $\rho$.
- **Visual Distinction:** Points color-coded by retrieval engine; callout box noting the LSH joint zero-floor tie cluster.

### Figure 4: Hypothesis 2 — Evidence Coverage vs. DAQ Association
- **Reference File:** [`experiments/results/stage3_figures/fig2_coverage_vs_composite_config.png`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_figures/fig2_coverage_vs_composite_config.png) (updated with DAQ axis)
- **Description:** Strong linear/rank alignment between Evidence Coverage and decoupled Downstream Answer Quality (DAQ) ($\rho \approx +0.805$ to $+0.964$).

### Figure 5: Contrast — NDCG@K vs. DAQ vs. Coverage vs. DAQ ($\Delta$ Distribution)
- **Reference File:** [`experiments/results/stage3_figures/fig3_ndcg_vs_composite_config.png`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_figures/fig3_ndcg_vs_composite_config.png)
- **Description:** Forest plot of paired bootstrap distributions for $\Delta = \rho(\text{Coverage}, \text{DAQ}) - \rho(\text{NDCG}, \text{DAQ})$ across all 15 configurations, displaying observed $\Delta$, bootstrap median, and 95% percentile confidence intervals strictly above 0.

### Figure 6: Retrieval Cutoff Dynamics and Downstream Quality Trajectories
- **Reference File:** [`experiments/results/stage3_figures/fig6_k_sensitivity_curves.png`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_figures/fig6_k_sensitivity_curves.png)
- **Description:** Line plots showing trajectory of DAQ and Factual Consistency across $K=5 \to 10 \to 20$, illustrating the universal drop at $K=20$.

---

## Claim Strength & Scientific Epistemology Taxonomy

| Manuscript Claim / Statement | Classification | Epistemological Basis & Justification |
| :--- | :---: | :--- |
| **"NDCG@K and Evidence Coverage show a positive rank association."** | **ASSOCIATIONAL** | Confirmed positive in 15/15 configs (mean $\rho = +0.421$); significant in 13/15. |
| **"Evidence Coverage exhibits a stronger rank association with DAQ than NDCG@K."** | **ASSOCIATIONAL** | Observed $\Delta > 0$ in 15/15 configs; paired bootstrap $P(\Delta > 0) = 1.000$ and 95% CIs exclude zero. |
| **"BM25's conventional retrieval superiority did not translate to downstream DAQ superiority over Dense."** | **DIRECTLY SUPPORTED** | Paired Wilcoxon tests show no statistically significant difference at any cutoff ($p = 0.515, 0.216, 0.419$). |
| **"Downstream answer quality at K=20 was lower than at K=10 across all five retrieval systems."** | **DIRECTLY SUPPORTED** | Verified numerically for 5 of 5 retrieval engines without exception. |
| **"Context dilution caused the downstream quality decline at K=20."** | **EXPLORATORY** | Plausible explanatory hypothesis consistent with observed declines in factual consistency, but observational design cannot prove causal mechanism. |
| **"Evidence Coverage causes higher downstream answer quality."** | **PROHIBITED CLAIM** | Strictly rejected. Observational correlations do not establish causation, and both variables share reference context. |
| **"The system produces clinically verified or safe medical answers."** | **LIMITATION** | Prohibited claim. DAQ evaluates text adherence proxies scored by an LLM, not clinical validity or safety. |

---

## FINAL FROZEN NUMBERS COMPENDIUM

These values are locked from the authoritative machine-readable artifacts and final statistical audits:

### 1. Corpus & Benchmark Totals
- **Corpus V1 Papers:** $22$
- **Total Corpus Pages:** $387$
- **Total Chunks:** $1,142$
- **Total Cleaned Tokens:** $248,894$
- **Target Chunk Size:** $256$ tokens ($32$ overlap)
- **Total Query Benchmark:** $65$ queries
- **Primary Evaluated Cohort:** $59$ queries with protocol-annotated grade-2 evidence
- **Excluded Sensitivity Queries:** $6$ queries (`Q007`, `Q013`, `Q017`, `Q021`, `Q055`, `Q061`)
- **Total Primary Experimental Conditions:** $885$ ($59 \times 5 \times 3$)

### 2. Downstream Answer Quality (DAQ) Population Parameters ($N = 885$)
- **Formula:** $\text{DAQ} = (\text{Factual Consistency} + \text{Answer Relevance}) / 2.0$
- **Population Mean:** $0.7404$
- **Population Median:** $0.8000$
- **Population Minimum:** $0.0000$
- **Population Maximum:** $0.9000$
- **Population Sample Standard Deviation ($ddof=1$):** $0.2140$

### 3. Hypothesis 1 Summary Statistics (NDCG@K vs. Evidence Coverage)
- **Global Population ($n = 15$ configurations):**
  - Mean $\rho$: $\mathbf{+0.4210}$ ($0.420994$)
  - Median $\rho$: $\mathbf{+0.3637}$
  - Range: $\mathbf{+0.2133 \text{ to } +0.7913}$
  - Positive Configurations: $\mathbf{15 \text{ of } 15}$ ($100.0\%$)
  - Statistically Significant Configurations ($p < 0.05$): $\mathbf{13 \text{ of } 15}$ ($86.7\%$)
- **Viable / Competitive Subset ($n = 12$ configurations, non-LSH):**
  - Mean $\rho$: $\mathbf{+0.3345}$
  - Median $\rho$: $\mathbf{+0.3341}$
  - Range: $\mathbf{+0.2133 \text{ to } +0.4473}$
  - Positive Configurations: $\mathbf{12 \text{ of } 12}$ ($100.0\%$)
  - Statistically Significant Configurations ($p < 0.05$): $\mathbf{10 \text{ of } 12}$ ($83.3\%$)
- **LSH Cutoff Correlation Anchor ($K=20$):**
  - Spearman $\rho$: $\mathbf{+0.7913}$ ($p = 8.73 \times 10^{-14}$)
  - $\text{NDCG@20} == 0.0$: $43 \text{ of } 59 \text{ queries}$ ($72.9\%$)
  - $\text{Evidence Coverage} == 0.0$: $35 \text{ of } 59 \text{ queries}$ ($59.3\%$)
  - Joint Ties at $(0.0, 0.0)$: $35 \text{ of } 59 \text{ queries}$ ($59.3\%$)

### 4. Hypothesis 2 Summary Statistics (Coverage vs. NDCG $\to$ DAQ)
- **Coverage $\to$ DAQ Range:** $\mathbf{+0.8053 \text{ to } +0.9644}$ (Mean $\mathbf{+0.8854}$, all $15 \text{ configs } p < 0.0001$)
- **NDCG $\to$ DAQ Range:** $\mathbf{+0.1269 \text{ to } +0.6989}$ (Global Mean $\mathbf{+0.3678}$; Viable Mean $\mathbf{+0.2850}$)
- **Difference $\Delta = \rho(\text{Coverage}, \text{DAQ}) - \rho(\text{NDCG}, \text{DAQ})$:**
  - Range: $\mathbf{+0.2017 \text{ to } +0.7370}$
  - Mean: $\mathbf{+0.5175}$
  - Positive Configurations: $\mathbf{15 \text{ of } 15}$ ($100.0\%$)
- **Paired Bootstrap ($B = 2,000$, Seed $= 42$):**
  - Directional Probability $P(\Delta > 0)$: $\mathbf{1.0000}$ across all 15 configurations
  - 95% Percentile Confidence Intervals: Strictly exclude zero across all 15 configurations (Lowest lower bound $\mathbf{+0.0883}$, highest upper bound $\mathbf{+1.0505}$)

### 5. Configuration-Level Exploratory Analysis (Across Averages)
- **All 15 Configurations:**
  - $\text{NDCG vs. DAQ } \rho = +0.4826$ ($p = 0.0685$)
  - $\text{Coverage vs. DAQ } \rho = +0.9122$ ($p < 0.0001$)
- **Non-LSH 12 Configurations:**
  - $\text{NDCG vs. DAQ } \rho = \mathbf{+0.0070}$ ($p = 0.9828$) — collapses to zero
  - $\text{Coverage vs. DAQ } \rho = \mathbf{+0.8345}$ ($p = 0.0007$) — remains strong

### 6. BM25 vs. Dense Downstream DAQ Contrasts
- **$K = 5$:** $\text{BM25} = 0.7826$, $\text{Dense} = 0.8127$, $\Delta = -0.0301$, Wilcoxon $W = 106.5$, $p = 0.51455$
- **$K = 10$:** $\text{BM25} = 0.8220$, $\text{Dense} = 0.8144$, $\Delta = +0.0076$, Wilcoxon $W = 28.0$, $p = 0.21564$
- **$K = 20$:** $\text{BM25} = 0.8186$, $\text{Dense} = 0.8123$, $\Delta = +0.0064$, Wilcoxon $W = 52.5$, $p = 0.41925$
- **Status:** Downstream parity; no statistically significant differences detected.

### 7. Retrieval Depth Cutoff Trajectories (DAQ across $K \in \{5, 10, 20\}$)
- **BM25:** $0.7826 \to \mathbf{0.8220} \to 0.8186$ ($K=20 < K=10$: True)
- **Dense:** $0.8127 \to \mathbf{0.8144} \to 0.8123$ ($K=20 < K=10$: True)
- **BM25 + Dense RRF:** $\mathbf{0.8203} \to 0.8199 \to 0.8144$ ($K=20 < K=10$: True)
- **BM25 + Dense + LSH RRF:** $0.8131 \to \mathbf{0.8195} \to 0.8140$ ($K=20 < K=10$: True)
- **LSH:** $0.4331 \to \mathbf{0.4610} \to 0.4475$ ($K=20 < K=10$: True)
- **Universal Trajectory Property:** $\text{DAQ}(K=20) < \text{DAQ}(K=10)$ holds across **5 of 5 architectures ($100.0\%$)**.
