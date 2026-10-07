# Conventional Retrieval Metrics and Downstream Answer Quality in Cardiovascular RAG

An empirical evaluation of the relationship between conventional retrieval metrics, reference-evidence coverage, and downstream answer quality in cardiovascular retrieval-augmented generation.

---

## Overview

Retrieval-augmented generation (RAG) is widely explored for clinical decision-support to ground large language models (LLMs) in verifiable medical literature. Clinical RAG implementations standardly evaluate and optimize retrieval using classical Information Retrieval (IR) metrics—such as Normalized Discounted Cumulative Gain ($\text{NDCG@K}$), Mean Reciprocal Rank ($\text{MRR}$), Recall, and Mean Average Precision ($\text{MAP}$).

However, these metrics primarily evaluate **ordinal ranking quality** of candidate passages. In contrast, generative language models consume retrieved passages as a joint, concatenated context set. Once an essential clinical fact is present within the prompt, its specific ranking position may exert negligible influence on whether the model incorporates it. Conversely, expanding context depth to maximize passage recall can introduce peripheral passages, diluting attention and increasing hallucinations.

This project empirically investigates this **retrieval-generation metric alignment problem** in cardiovascular RAG. We evaluate whether conventional ranking metrics reliably reflect reference-evidence coverage and downstream generation quality across multiple retrieval architectures and depths.

> **Scope Note:** This study does **not** introduce a novel retrieval algorithm, a new dense embedding model, or a clinical diagnostic tool. It is an empirical and methodological investigation of metric alignment in clinical RAG benchmarks.

---

## Research Question

> **"How strongly are conventional retrieval metrics associated with evidence sufficiency and downstream answer quality in cardiovascular RAG?"**

Because the study design is observational and non-interventional, all analyses evaluate statistical **associations** rather than asserting causal or predictive claims.

---

## Hypotheses

- **Hypothesis 1 (H1 — Associational):** Higher conventional retrieval quality ($\text{NDCG@K}$) is positively associated with reference-evidence coverage.
- **Hypothesis 2 (H2 — Associational, Decoupled):** Reference-evidence coverage has a stronger association with downstream answer quality than conventional retrieval quality alone.

---

## Research Design

The study follows a frozen, fully blinded evaluation pipeline:

```
[Corpus V1: 22 Publications, 1,142 Chunks]
                      │
                      ▼
[5 Retrieval Systems @ K ∈ {5, 10, 20}] ──► 15 Experimental Configurations
                      │
                      ▼
[Generator: Gemini 3.1 Flash-Lite (T=0)] ──► Blinded Generation (885 Primary Conditions)
                      │
                      ▼
[Evaluator: Cohere Command-R (T=0)] ──────► Cross-Family Blinded Scoring
                      │
                      ▼
[Decoupled Target: DAQ = (FC + AR) / 2] ──► Statistical Association Analysis & Bootstrap
```

---

## Corpus

**Corpus V1** consists of license-verified, open-access cardiovascular literature:

- **22 open-access cardiovascular publications** spanning clinical reviews, guideline/consensus-related documents, systematic reviews, and related cardiovascular literature.
- **Licensing:** 100% verified Creative Commons Attribution (CC BY / CC BY 4.0).
- **Volume:** 387 publication pages, 248,894 cleaned tokens.
- **Passage Chunks:** 1,142 discrete text passages.
- **Chunking Parameters:** Target size of 256 tokens, 32-token overlap, sentence-boundary-aware partitioning.
- **Clinical Topics ($n=9$):** Heart Failure ($n=4$), Arrhythmia/Atrial Fibrillation ($n=4$), Myocardial Infarction ($n=3$), Coronary Artery Disease ($n=3$), Hypertension ($n=2$), Cardiomyopathy ($n=2$), Prevention ($n=2$), Dyslipidemia ($n=1$), Valvular Heart Disease ($n=1$).
- **Thin-Source Limitation:** Document `HTN02` is a two-page editorial contributing few chunks (sparse passage candidate density for certain hypertension queries).

---

## Query Benchmark

- **Total Queries:** 65 clinical queries formulated to reflect authentic inquiries in cardiovascular medicine.
- **Cognitive Inquiry Types:**
  - Treatment selection ($n=27$)
  - Diagnostic criteria and workup ($n=17$)
  - Pathophysiological mechanisms ($n=10$)
  - Clinical prognosis ($n=6$)
  - Risk stratification ($n=4$)
  - Acute complications ($n=1$)
- **Primary Evaluation Cohort ($n=59$ queries):** Queries with at least one verified grade-2 (essential reference evidence) passage in the corpus.
- **Excluded Sensitivity Cohort ($n=6$ queries):** Queries containing only grade-1 (supporting/partial) passages (`Q007`, `Q013`, `Q017`, `Q021`, `Q055`, `Q061`), strictly excluded from primary downstream evaluation to prevent reference-incompleteness confounding.
- **Primary Experimental Matrix:** 59 queries $\times$ 5 retrieval architectures $\times$ 3 cutoff depths = **885 conditions**.

---

## Reference Evidence

Reference evidence consists of **protocol-annotated grade-2 reference evidence passages** from the frozen relevance dataset (`data/evaluation/qrels.csv`).

- **Provenance:** Candidate passages were generated via BM25 top-20 pooling and keyword scanning, followed by rule-based and textual verification under formal annotation criteria (`data/evaluation/annotation_guidelines.md`).
- **Nature of Evidence:** These labels represent an internally consistent literature reference standard for this corpus. They are **not** prospective adjudications by an independent panel of clinical cardiologists.

---

## Retrieval Architectures

Five retrieval architectures representing common RAG retrieval paradigms were evaluated at cutoff depths $K \in \{5, 10, 20\}$:

1. **BM25 (Lexical Baseline):** Okapi BM25 ($k_1 = 1.5, b = 0.75$) over tokenized chunk text.
2. **Dense Semantic Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings, $L_2$-normalized, exact inner product search via FAISS `IndexFlatIP`).
3. **Locality-Sensitive Hashing (LSH):** Word-level 3-shingle MinHash LSH (128 permutation hash functions, 16 bands $\times$ 8 rows, indexing threshold $= 0.05$, exact Jaccard similarity reranking).
4. **BM25 + Dense RRF (Hybrid):** Reciprocal Rank Fusion combining BM25 and Dense with smoothing parameter $k_{\text{rrf}} = 60$.
5. **BM25 + Dense + LSH RRF (Three-Way Hybrid):** Reciprocal Rank Fusion combining BM25, Dense, and LSH with $k_{\text{rrf}} = 60$.

---

## Generation Protocol

- **Generator Model:** Google `gemini-3.1-flash-lite`.
- **Inference Configuration:** Greedy decoding (temperature $= 0.0$, top-$p = 1.0$, maximum output $= 512$ tokens).
- **Conditioning Prompt:** Clinical query concatenated with top-$K$ retrieved passages in retrieved order.
- **Blinding:** Generator received zero metadata regarding retriever identity, cutoff depth $K$, retrieval similarity scores, relevance grades, or evaluation criteria.
- **Execution:** All $65 \times 5 \times 3 = 975$ conditions completed successfully with zero API failures.

---

## Downstream Evaluation

- **Evaluator Model:** Cohere `command-r-08-2024` via Cohere v2 API (`/v2/chat`).
- **Inference Configuration:** Greedy decoding (temperature $= 0.0$, structured JSON output schema).
- **Evaluator Blinding:** Received strictly: (1) clinical query, (2) generated answer, and (3) protocol-annotated reference evidence text. Did not receive retriever identity, cutoff depth $K$, retrieval scores, or generator identity.
- **Evaluator Independence:** Generator and evaluator models were drawn from different commercial providers and architectural families, reducing the risk of direct intra-family evaluator-generator preference effects (though not eliminating all evaluator bias).
- **Evaluation Dimensions ($[0.0, 1.0]$):**
  1. *Evidence Coverage:* Degree to which the answer captures essential factual assertions from reference evidence.
  2. *Factual Consistency:* Degree to which answer assertions are strictly supported by reference evidence without hallucination.
  3. *Answer Relevance:* How directly and appropriately the answer addresses the clinical query.

---

## Downstream Answer Quality (DAQ)

To avoid the **part-whole correlation artifact**—where correlating Evidence Coverage with a composite score that contains Evidence Coverage produces mathematical auto-correlation—we strictly decouple the target outcome:

$$\mathbf{DAQ} = \frac{\text{Factual Consistency} + \text{Answer Relevance}}{2.0}$$

Evidence Coverage is completely excluded from $\text{DAQ}$, allowing unbiased evaluation of how passage coverage relates to downstream generative quality.

---

## Statistical Analysis

- **Inferential Unit:** Query-level observations ($n=59$) within each of the 15 experimental configurations.
- **H1 Analysis:** Query-level Spearman rank correlation $\rho(\text{NDCG@K}, \text{Evidence Coverage})$.
- **H2 Analysis:** Dependent correlation contrast $\Delta = \rho(\text{Coverage}, \text{DAQ}) - \rho(\text{NDCG@K}, \text{DAQ})$.
- **Bootstrap Resampling:** Paired query bootstrap ($B = 2{,}000$ iterations, seed $= 42$), resampling 59 query IDs with replacement.
- **Pairwise Contrasts:** Paired Wilcoxon signed-rank tests across queries at fixed $K$.

---

## Main Results

### H1: Retrieval Metrics vs. Evidence Coverage

Higher conventional retrieval quality ($\text{NDCG@K}$) is positively associated with Evidence Coverage across configurations:

- **Global Population ($n=15$ configurations):** Mean Spearman $\rho = \mathbf{+0.4210}$, median $\rho = \mathbf{+0.3637}$, range $\mathbf{+0.2133 \text{ to } +0.7913}$ (15/15 positive, 13/15 statistically significant at $p < 0.05$).
- **Competitive Viable Systems ($n=12$ configurations, excluding LSH):** Mean $\rho = \mathbf{+0.3345}$, median $\rho = \mathbf{+0.3341}$, range $\mathbf{+0.2133 \text{ to } +0.4473}$ (12/12 positive, 10/12 significant).

#### LSH Floor Effect Diagnosis
While LSH at $K=20$ achieved the nominal maximum correlation ($\rho = +0.7913, p < 10^{-13}$), this is driven by an extreme zero-hit floor effect:
- 43 of 59 queries had $\text{NDCG@20} = 0.0$.
- 35 of 59 queries had $\text{Evidence Coverage} = 0.0$.
- Exactly 35 queries tied at joint zero $(0.0, 0.0)$.
Because ties receive identical ranks, this shared failure mode produces an inflated correlation. LSH is not a competitive retrieval engine; the correlation reflects aligned failure modes rather than retrieval superiority.

### H2: Evidence Coverage vs. Downstream Answer Quality (DAQ)

Evidence Coverage demonstrates a substantially stronger association with $\text{DAQ}$ than conventional ranking metrics ($\text{NDCG@K}$):

- **$\rho(\text{Coverage}, \text{DAQ})$:** Range $+0.8053$ to $+0.9644$, mean $\mathbf{+0.8854}$ (all $p < 0.0001$).
- **$\rho(\text{NDCG}, \text{DAQ})$:** Range $+0.1269$ to $+0.6989$, global mean $+0.3678$, non-LSH mean $+0.2850$.
- **Correlation Difference ($\Delta$):** Positive in **15 of 15 configurations ($100.0\%$)**, range $+0.2017$ to $+0.7370$, mean $\mathbf{+0.5175}$.
- **Paired Bootstrap ($B=2{,}000$, Seed $= 42$):** $P(\Delta > 0) = \mathbf{1.0000}$ for all 15 configurations; all 95% percentile confidence intervals strictly exclude zero.

*Decision:* **H2 Supported (Associational).**

### BM25 vs. Dense Downstream Parity

In conventional retrieval ranking, BM25 substantially outperformed Dense retrieval at $K=10$ ($\text{NDCG@10} = 0.5287$ vs. $0.3889$; $\text{Recall@10} = 0.3846$ vs. $0.2840$).

Downstream, however, paired Wilcoxon signed-rank tests detected **no statistically significant difference** in $\text{DAQ}$ across queries:

| Cutoff | BM25 Mean DAQ | Dense Mean DAQ | Difference | Wilcoxon $W$ | Two-Sided $p$-value | Significant? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $K = 5$ | 0.7826 | 0.8127 | -0.0301 | 106.5 | 0.51455 | No |
| $K = 10$ | 0.8220 | 0.8144 | +0.0076 | 28.0 | 0.21564 | No |
| $K = 20$ | 0.8186 | 0.8123 | +0.0064 | 52.5 | 0.41925 | No |

Dense semantic retrieval achieved downstream answer quality parity despite lower conventional ranking scores. (Note: Semantic paraphrasing is an interpretation of this parity, not a directly tested mechanism.)

### Retrieval Depth Dynamics ($K \in \{5, 10, 20\}$)

- $K=10$ achieved the highest $\text{DAQ}$ for 3 of the 5 architectures (BM25: $0.8220$; Dense: $0.8144$; BM25+Dense+LSH RRF: $0.8195$).
- BM25 + Dense RRF peaked at $K=5$ ($0.8203$ vs. $0.8199$ at $K=10$).
- Across all five retrieval architectures without exception, $\text{DAQ}$ at $K=20$ was lower than at $K=10$:
  - BM25: $0.8220 \to 0.8186$ ($-0.0034$)
  - Dense: $0.8144 \to 0.8123$ ($-0.0021$)
  - BM25 + Dense RRF: $0.8199 \to 0.8144$ ($-0.0055$)
  - BM25 + Dense + LSH RRF: $0.8195 \to 0.8140$ ($-0.0055$)
  - LSH: $0.4610 \to 0.4475$ ($-0.0135$)
- This drop mirrored systematic decreases in Factual Consistency from $K=10$ to $K=20$.
- *Interpretation:* This pattern is consistent with the context dilution hypothesis (peripheral passages introducing distractors), but generator attention allocation was not directly measured. $K=10$ is not claimed to be universally optimal across all RAG systems.

---

## Publication Figures & Artifact Policy

- **Included Verified Publication Figure:**
  - `paper/figures/fig1_ndcg_vs_coverage_config.png` — Configuration-level scatter plot of mean $\text{NDCG@K}$ vs. mean Evidence Coverage ($n=15$). Verified to plot only pure retrieval and coverage metrics.
- **Withheld Historical Composite Figures:**
  - Preliminary exploratory figure artifacts (`fig2` to `fig6`) generated during earlier stages plotted downstream performance against a retracted 3-component composite score. To prevent presenting historical composite data as decoupled $\text{DAQ}$, these figures are withheld from the formal publication manuscript pending reproduction with decoupled outcomes.

---

## Limitations

1. **Shared-Reference Dependency:** Evidence Coverage and retrieval relevance judgments both rely on the same underlying reference evidence passages.
2. **Protocol-Annotated Relevance Labels:** Labels were established via BM25 pooling, keyword scanning, and rule-based verification, not prospective clinical cardiologist adjudication.
3. **Automated LLM Proxy Evaluation:** Scored automatically by Cohere Command-R; automated scores evaluate textual adherence and do not establish clinical correctness or patient safety.
4. **No Direct Assessment of Clinical Efficacy:** Evaluated factual consistency against literature, not real-world diagnostic or therapeutic outcomes.
5. **Benchmark Scale:** The benchmark comprises 65 clinical queries and 22 open-access publications, providing 885 evaluated conditions across the experimental matrix. Generalizability to other specialties remains to be demonstrated.
6. **Cohort Exclusion:** Six queries with only grade-1 evidence were excluded from primary downstream evaluation.
7. **Observational Study Design:** Correlational analysis; does not establish causal mechanisms.
8. **Incomplete Elimination of Evaluator Bias:** Cross-family separation reduces intra-family self-preference risk, but cannot eliminate all shared LLM bias.
9. **LSH Floor Confounding:** Global correlation statistics are sensitive to LSH floor ties.
10. **Thin-Source Document:** Document `HTN02` is a two-page editorial with sparse chunk density.
11. **Untested Generative Mechanisms:** Semantic paraphrasing and attention allocation are interpretations, not directly verified mechanisms.
12. **Historical Figure Artifact Retraction:** Downstream composite figures withheld from publication.

---

## Final Research Contribution

1. **Empirical Evaluation of Metric Alignment:** Demonstrates that conventional IR ranking metrics ($\text{NDCG@K}$) correlate only modestly with passage evidence coverage ($\rho \approx +0.33$) and decouple from downstream answer quality among competitive retrieval architectures.
2. **Target Metric Decoupling:** Proves the necessity of decoupling passage coverage from downstream outcomes to prevent part-whole correlation artifacts.
3. **Downstream Retrieval Parity:** Demonstrates that dense semantic retrieval achieves downstream answer quality parity with BM25 despite substantial deficits in lexical ranking metrics.
4. **Depth Attenuation Dynamics:** Documents non-monotonic retrieval depth behavior where deeper retrieval ($K=20$) consistently degrades factual consistency across all architectures.

---

## Project Status

**Status: FINAL / FACT-LOCKED RESEARCH FREEZE**

- **Corpus V1:** Frozen (22 documents, 1,142 chunks)
- **Evaluation Benchmark:** Frozen (65 queries, 59 primary)
- **Retrieval Experiments:** Complete (5 architectures $\times$ 3 cutoffs)
- **Generation:** Complete ($975 / 975$ conditions, 100% success rate)
- **Downstream Evaluation:** Complete ($885$ primary conditions)
- **Statistical Analysis:** Complete (H1, decoupled H2, bootstrap $B=2{,}000$, Wilcoxon parity)
- **Manuscript:** Complete ([`docs/stage5_manuscript_final.md`](docs/stage5_manuscript_final.md))
- **LaTeX Paper:** Complete ([`paper/main.tex`](paper/main.tex), [`paper/references.bib`](paper/references.bib))
- **Test Suite:** Passing (429 passed, 1 warning in test suite)

---

## Repository Structure

```
.
├── data/
│   ├── processed/                       # Cleaned document texts
│   ├── chunks/                          # Serialised chunks (chunks.jsonl, 1,142 chunks)
│   └── evaluation/
│       ├── queries.csv                  # 65 clinical queries
│       ├── qrels.csv                    # Graded passage relevance judgments
│       ├── reference_evidence.json      # Compiled grade-2 reference passages
│       ├── annotation_guidelines.md     # Relevance annotation guidelines
│       └── evaluation_dataset_report.md # Dataset composition report
│
├── docs/
│   ├── stage3_final_statistical_audit.md # Final statistical integrity audit
│   ├── stage3_h1_reconciliation.md       # H1 reconciliation report
│   ├── stage3_h2_correction.md           # H2 decoupling methodology
│   ├── stage5_manuscript_blueprint.md    # Fact-locked manuscript blueprint
│   ├── stage5_manuscript_final.md        # Corrected final research manuscript
│   └── stage5_final_manuscript_audit.md  # Final manuscript compliance audit
│
├── experiments/
│   └── results/
│       ├── stage3_retrieval_runs.json    # Full retrieved passage runs
│       ├── stage3_retrieval_metrics.json # Conventional IR metrics (NDCG, Recall, MRR)
│       ├── stage3_generated_answers.jsonl# 975 generated clinical responses
│       ├── stage3_evaluator_results.jsonl# Blinded Cohere evaluator outputs
│       ├── stage3_h2_corrected.json      # Corrected decoupled DAQ correlations
│       ├── stage3_h2_corrected.csv       # Summary CSV for corrected H2
│       ├── stage3_h2_bootstrap.csv       # 2,000-iteration paired bootstrap distributions
│       ├── stage3_daq_configuration_summary.csv # Configuration-level DAQ means
│       ├── stage3_statistical_analysis.json # Full statistical artifacts
│       ├── stage3_configuration_summary.csv
│       ├── stage3_query_level_correlations.csv
│       ├── stage3_pairwise_comparisons.csv
│       └── stage3_figures/              # Generated figures archive
│
├── indexes/
│   ├── bm25/                            # Serialized BM25 Okapi index
│   ├── faiss/                           # Serialized FAISS IndexFlatIP dense index
│   └── lsh/                             # Serialized MinHash LSH index
│
├── paper/
│   ├── main.tex                         # Publication-quality LaTeX paper
│   ├── references.bib                   # BibTeX bibliography (21 verified entries)
│   └── figures/
│       └── fig1_ndcg_vs_coverage_config.png # Verified publication figure
│
├── src/
│   ├── ingestion/                       # Document ingestion loaders
│   ├── preprocessing/                   # Text cleaner and chunker
│   ├── retrieval/                       # BM25, Dense, LSH, and Hybrid fusion retrievers
│   ├── generation/                      # Gemini generation pipeline
│   └── evaluation/                      # Evaluator, DAQ calculation, and statistical tools
│
├── tests/                               # 429 comprehensive unit and integration tests
├── scripts/
│   └── validate_latex.py                # LaTeX static syntax and structural validator
│
├── paper.zip                            # Portable LaTeX archive for Overleaf recompilation
└── requirements.txt                     # Python dependencies
```

---

## Reproducibility

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests

```bash
pytest tests/ -q
```

### Validating LaTeX Paper

```bash
python3 scripts/validate_latex.py
```

> **Security Note:** API credentials for Gemini and Cohere must never be committed to source control. Ensure `.env` is listed in `.gitignore`.

---

## Citation

```bibtex
@article{cardiology_rag_2026,
  title   = {Conventional Retrieval Metrics and Downstream Answer Quality in Cardiovascular RAG: An Empirical Evaluation},
  author  = {Anonymous Authors},
  journal = {Working Manuscript},
  year    = {2026}
}
```

---

## Disclaimer

This repository contains an empirical research prototype and evaluation study. It is **not** a clinical decision-support system and **must not** be used for clinical diagnosis, treatment decisions, patient care, or medical advice.
