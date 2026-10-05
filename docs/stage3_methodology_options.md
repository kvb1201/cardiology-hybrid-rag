# Stage 3 — Methodology Redesign

## 1. Problem Created by Missing Reference Answers

The original pipeline required **reference answers** — expert-authored gold-standard answers for each of the 65 queries — to compute Reference-Evidence Coverage (REC) and answer correctness:

```
retrieval → REC against reference answer → LLM generation → correctness vs reference answer → correlation
```

The Stage 3 audit established that:

- **No reference answers exist** anywhere in the repository.
- The 65 queries are **custom-authored** for Corpus V1; they cannot be mapped to an external QA benchmark (MedQA, PubMedQA, etc.).
- Creating clinical reference answers requires domain expertise and is explicitly forbidden by the project constraints (no manual clinical annotation, no clinician annotation, 2-day deadline).
- **LLM-generated "gold answers" are not ground truth.** Treating them as such would be scientifically indefensible.

However, the repository **does** contain 828 frozen qrels mapping each of the 65 queries to relevant corpus chunks with graded relevance (530 grade-2, 298 grade-1). These qrels represent **reference evidence** — the set of chunks that a correct retriever should surface — rather than reference answers.

**The methodological pivot:** Instead of comparing generated answers against reference *answers*, we evaluate generated answers against reference *evidence*. The research question becomes:

> *"Do conventional retrieval metrics (computed against qrels) predict the quality of LLM-generated answers (evaluated against the same qrels evidence)?"*

This is a genuine empirical question because the LLM generation step is a non-trivial, lossy transformation: a retriever can surface all the right evidence and the LLM can still hallucinate, fail to synthesize, or drop key information. Conversely, a retriever with lower recall might surface compact, high-quality chunks that the LLM synthesizes effectively.

---

## 2. Candidate Method A — Qrels-Grounded Evaluation

**Design:** Use all qrels chunks (grade ≥ 1) for a given query as the reference evidence set. An LLM evaluator judges whether the generated answer is supported by and covers the information in this evidence set.

**Inputs:**
- Query text
- Generated answer (from retrieved top-K chunks via a generator LLM)
- Reference evidence set: all qrels chunks for that query (grade 1 + grade 2)

**Evaluation question for the LLM judge:**
> "Given these authoritative evidence passages, does the generated answer accurately cover the key clinical information?"

**Strengths:**
- Uses the full qrels set — maximizes the reference evidence available per query.
- All 65 queries have at least 3 relevant chunks (no zero-evidence queries).
- Evaluator never sees the *retrieved* chunks — only the qrels evidence. This separates retrieval evaluation from answer evaluation.

**Weaknesses:**
- Mixes grade-1 (partial, tangential) and grade-2 (direct, substantive) chunks. Grade-1 chunks may contain peripheral information that inflates the reference set with content the answer shouldn't be expected to cover.
- A query with 20 relevant chunks (e.g., Q047) and a query with 3 relevant chunks (e.g., Q065) produce very different evaluation contexts.
- The evaluator must judge coverage against a potentially large and heterogeneous evidence set.

**Assessment:** Usable but noisy. The grade-1 chunks introduce ambiguity about what constitutes "sufficient" coverage.

---

## 3. Candidate Method B — Full-Corpus-Grounded Evaluation

**Design:** Use the entire frozen corpus (1,142 chunks) as the reference knowledge source. The evaluator checks the generated answer against all corpus knowledge.

**Strengths:**
- Maximum reference information available.
- No dependence on qrels at all for evaluation — eliminates any circularity with retrieval metrics.

**Weaknesses:**
- **Impractical.** 1,142 chunks × ~240 words each ≈ 274,000 words per evaluation call. This far exceeds practical LLM context limits and is cost-prohibitive (65 queries × 5 retrievers × 3 K = 975 calls, each requiring the full corpus).
- **Unfair evaluation.** The full corpus contains information on 9 topics. The evaluator might penalize an answer for not covering tangentially related information the query never asked about.
- **Leakage risk.** If the evaluator has access to *all* corpus information, it can always detect what's missing — but the generator only saw K chunks. This creates an asymmetry that doesn't test retrieval quality; it tests the trivial fact that K < 1,142.

**Assessment:** Not viable. Rejected.

---

## 4. Candidate Method C — Grade-2 Reference Evidence

**Design:** Use only grade-2 ("highly relevant") qrels chunks as the reference evidence set. The LLM evaluator judges the generated answer against this focused, high-quality evidence set.

**Inputs:**
- Query text
- Generated answer (from retrieved top-K chunks)
- Reference evidence set: only grade-2 qrels chunks for that query

**Rationale from the annotation guidelines:**

Grade-2 is defined as: *"The chunk directly and substantively addresses the query. A clinician trying to answer the query would find this chunk essential or primary reading."* (annotation_guidelines.md, §2)

This means grade-2 chunks contain the **core clinical information** a correct answer should include. They represent the ideal evidence pool — what a perfect retriever would surface as its most valuable results.

**Empirical properties (from analysis):**
- 59 of 65 queries have at least one grade-2 chunk.
- 6 queries have zero grade-2 chunks: Q007, Q013, Q017, Q021, Q055, Q061. These have grade-1 chunks only (range: 4–13).
- For the 59 grade-2-available queries: min=2, max=20, mean=9.0, median=8 grade-2 chunks.
- Grade-2 chunks average ~242 words each. For a query with 9 grade-2 chunks, the reference evidence set is ~2,200 words — well within LLM context limits.

**Handling the 6 zero-grade-2 queries:**
- **Option 1 (recommended):** For these 6 queries, fall back to using grade-1 chunks as reference evidence and flag them in the analysis. Report results both including and excluding these queries.
- **Option 2:** Exclude these 6 queries entirely (analyze 59/65 = 91%). Report the exclusion explicitly.

**Strengths:**
- Focused, high-quality reference set. Grade-2 chunks are defined as "directly and substantively addressing the query," making them the closest available proxy for the information a correct answer should contain.
- Practical context size (~2,200 words per query on average).
- Available for 91% of queries without fallback.
- Separates what the evaluator sees (grade-2 evidence) from what the retrieval metrics measure (overlap between retrieved chunks and the full qrels set including grade 1).

**Weaknesses:**
- Partial circularity with retrieval metrics (see §5).
- 6 queries require fallback or exclusion.

**Assessment:** The strongest candidate. Proceed to circularity analysis.

---

## 5. Circularity and Leakage Analysis

This is the critical methodological question. We need to determine whether evaluating generated answers against grade-2 qrels chunks creates a circular dependency with retrieval metrics that are also computed against qrels.

### 5.1 The Circularity Concern

**Retrieval metrics** (NDCG@K, Recall@K, etc.) measure the overlap between retrieved chunk IDs and qrels chunk IDs.

**Answer quality** (as proposed) would be evaluated by an LLM judge comparing generated text against grade-2 qrels chunk text.

If a retriever has high Recall@K (retrieves many qrels chunks), the generator sees more of the same text the evaluator uses as reference evidence. This could create a trivially positive correlation: high recall → answer generated from reference evidence text → evaluator sees the same text in both the answer and the reference → high score.

### 5.2 Why This is NOT Fully Circular

The concern above is valid but overstated. The following intervening factors break the direct circularity:

1. **The LLM generation step is a lossy transformation.** The generator must *synthesize* 5–20 retrieved chunks into a coherent clinical answer. It may:
   - Drop key details from retrieved evidence.
   - Misinterpret or distort clinical concepts.
   - Hallucinate information not present in any retrieved chunk.
   - Fail to integrate complementary information from multiple chunks.
   - Successfully compress and synthesize information from just a few high-quality chunks.

2. **Retrieval metrics and answer quality measure fundamentally different things.** Recall@10 measures: "How many of the 13 relevant chunks did you retrieve?" Answer quality measures: "Does the *synthesized text* cover the *clinical information* in the reference evidence?" These are related but distinct. An answer can achieve high coverage from fewer retrieved chunks if those chunks contain the key information. An answer can fail despite high recall if the LLM hallucinates.

3. **Grade-1 chunks contribute to retrieval metrics but are excluded from the reference evidence.** NDCG@K gives credit for retrieving grade-1 chunks. The answer evaluator only checks against grade-2 chunks. A retriever that scores well on NDCG partly by retrieving grade-1 chunks may still produce answers that fail to cover grade-2 information.

4. **Rank position matters for retrieval metrics but not for answer quality.** NDCG discounts by rank position. The LLM generator sees all K retrieved chunks with equal prominence (or at least in a flat list). Rank-position effects create divergence between retrieval metrics and downstream quality.

5. **The same grade-2 information may exist in non-qrels chunks.** A retriever might surface chunks not in the qrels that nonetheless contain the same clinical information as grade-2 chunks (from the same or different documents). The answer evaluator would give credit for covering this information, even though retrieval metrics give no credit for these chunks.

### 5.3 Residual Circularity — Honest Assessment

There IS a baseline positive correlation baked in: a retriever that retrieves zero relevant chunks will produce answers with near-zero evidence coverage, and a retriever that retrieves all grade-2 chunks will produce answers with near-maximal coverage, *all else being equal*. This baseline correlation is not a methodological flaw — it is the expected behavior. The research question is about the *strength, shape, and reliability* of this relationship, not whether it exists at all.

**The genuinely interesting findings would be:**
- Queries where high retrieval metrics do NOT produce high answer quality (LLM failure modes).
- Queries where moderate retrieval metrics produce high answer quality (information redundancy in the corpus).
- Whether the correlation holds uniformly across topics and query types.
- Whether certain retrieval metrics (e.g., NDCG@10 vs. Recall@10) are better predictors than others.
- Whether the correlation changes across K values.

### 5.4 Mitigation Strategies

1. **Report the circularity explicitly** in the paper. Frame the experiment as testing "how reliably retrieval quality translates to answer quality through the LLM generation step," not as discovering an unknown relationship.
2. **Analyze residuals.** Identify queries where retrieval quality and answer quality diverge. These cases are the most informative.
3. **Include non-trivial evaluation dimensions.** Beyond coverage, evaluate factual consistency (no hallucination) and clinical coherence. These dimensions are NOT correlated with retrieval metrics by construction.
4. **Compare metrics.** If NDCG@10 correlates more strongly with answer quality than MRR does, that is a non-trivial finding even with baseline circularity.

### 5.5 Leakage Analysis

**Generator leakage:** The generator sees only retrieved top-K chunks. It does NOT see qrels or grade labels. No leakage.

**Evaluator leakage:** The evaluator sees grade-2 chunks and the generated answer. It does NOT see the retrieved chunk IDs, retrieval scores, or retrieval metrics. No leakage into retrieval evaluation.

**Cross-leakage:** The evaluator does not know which retriever produced the answer. No system-identification leakage.

### 5.6 Verdict

The circularity is **partial and acknowledged, not disqualifying**. The LLM generation step introduces genuine noise, the evaluation dimensions extend beyond retrieval overlap, and the analysis structure can identify informative divergence cases. For an undergraduate empirical study, this is defensible with explicit disclosure.

---

## 6. LLM-as-Judge Design

### 6.1 Generator vs. Evaluator Model Overlap

**Risk:** If the same LLM serves as both generator and evaluator, the evaluator may exhibit familiarity bias — preferring text it "would have written itself," or failing to detect hallucinations it would also produce.

**Recommendation:**
- **Ideal:** Use different models (e.g., GPT-4o-mini as generator, Gemini 1.5 Flash as evaluator, or vice versa). This eliminates self-evaluation bias entirely.
- **Acceptable fallback:** Use the same model with explicit disclosure. Temperature=0 for both. Justify in the paper: "We acknowledge that generator-evaluator overlap may introduce bias; however, using a fixed deterministic configuration and structured evaluation rubrics mitigates prompt-level variation."

### 6.2 Temperature and Determinism

- **Generator:** temperature=0, fixed system prompt, fixed passage formatting. Every retriever's output for the same query must be generated under identical conditions.
- **Evaluator:** temperature=0, fixed evaluation rubric, structured output (JSON with numerical scores and brief rationale).
- **Seed/determinism:** If the API supports a seed parameter, use it. Otherwise, temperature=0 provides near-deterministic outputs.

### 6.3 Evaluation Rubric

The evaluator should score on three dimensions (1–5 scale each):

| Dimension | Definition | What it Measures |
|---|---|---|
| **Comprehensiveness** | How much of the key clinical information in the reference evidence is covered by the answer? | Evidence coverage (distinct from Recall@K because it operates on synthesized text, not chunk IDs) |
| **Factual Consistency** | Is every claim in the answer supported by the reference evidence? Are there any contradictions or fabricated claims? | Hallucination detection (not correlated with retrieval metrics by construction) |
| **Clinical Utility** | Does the answer directly, coherently, and completely address the clinical question? | Overall quality beyond simple overlap |

**Composite score:** Mean of the three dimensions. Report individual dimensions as well.

### 6.4 Prompt Sensitivity

**Risk:** Small changes to the evaluation prompt can shift scores by 0.5–1.0 points on a 5-point scale.

**Mitigation:**
- Fix the evaluation prompt before running any experiments. Do not iterate on the prompt after seeing results.
- Pilot the prompt on 3–5 queries to verify the rubric produces sensible score distributions. Document the pilot but do not tune on the full dataset.
- Report the exact prompt in an appendix.

### 6.5 Known Limitations — To Be Reported

- LLM evaluators tend toward "generous" scoring (scores cluster in 3–5 range rather than 1–5).
- LLM evaluators may struggle to detect subtle clinical errors.
- Evaluation scores are an **automated proxy** for answer quality, NOT clinical expert judgment.
- We **cannot claim clinical correctness** of any generated answer.
- We can only claim: "As measured by an automated LLM evaluator, retrieval quality [does/does not] predict downstream answer quality."

---

## 7. Statistical Design

### 7.1 Experimental Design Matrix

| Factor | Levels | Count |
|---|---|---|
| Queries | Q001–Q065 | 65 |
| Retrieval systems | BM25, Dense, LSH, BM25+Dense RRF, BM25+Dense+LSH RRF | 5 |
| K values | 5, 10, 20 | 3 |
| **Total observations** | | **975** |

Each observation consists of:
- A retrieval metric vector (NDCG@K, Recall@K, Precision@K, MRR, MAP — computed from existing frozen infrastructure)
- A generated answer (from the generator LLM)
- An answer quality score vector (Comprehensiveness, Factual Consistency, Clinical Utility, Composite — from the evaluator LLM)

### 7.2 Dependency Structure — Why 975 ≠ 975 Independent Observations

These 975 observations are **NOT independent.** Three dependency sources:

1. **Within-query clustering.** Each query appears 15 times (5 retrievers × 3 K values). The difficulty and topic of the query create correlated outcomes. For example, Q047 (hypertriglyceridaemia) has 20 grade-2 chunks — nearly any retriever will produce a good answer. Q065 (risk scores) has 3 grade-2 chunks — all retrievers may struggle.

2. **K-value nesting.** Top-5 retrieved chunks are a strict subset of top-10, which are a strict subset of top-20. For a given query and retriever, the K=5 result is contained within the K=10 result.

3. **Within-retriever correlation.** Each retriever has systematic biases (BM25 favors keyword overlap, Dense favors semantic similarity). Performance across queries for a given retriever is correlated.

### 7.3 Recommended Analysis Structure

**Analysis Level 1 — System-Level Correlation (most conservative, lowest power)**

Aggregate across all 65 queries to produce one mean retrieval metric and one mean answer quality score per system configuration (retriever × K). This yields **15 data points**.

- Compute Spearman's ρ between mean NDCG@K and mean composite answer score across the 15 configurations.
- Report: ρ, p-value, 95% bootstrap CI.
- Low statistical power (n=15) but eliminates all within-query dependency.

**Analysis Level 2 — Per-Query Correlation at Fixed K (primary analysis)**

For each query at a fixed K (K=10 as primary), rank the 5 retrievers by retrieval metric and by answer quality.

- Compute per-query Spearman's ρ (n=5 per query — very low power per query).
- Report: distribution of per-query ρ values, mean ρ, % of queries where ρ > 0.
- More informative but requires acknowledging the low per-query sample size.

**Alternative for Level 2:** Instead of per-query rank correlation, report **concordance rate** — "For what percentage of queries does the retriever with the highest NDCG@10 also produce the highest-quality answer?" This is intuitive and does not require independence assumptions.

**Analysis Level 3 — Mixed-Effects Model (most sophisticated, if time permits)**

Fit a linear mixed-effects model:
```
answer_quality ~ retrieval_metric + (1 | query_id) + (1 | retriever)
```

This accounts for query-level and retriever-level random effects. Report the fixed-effect coefficient for `retrieval_metric` and its significance.

**Recommended for this project:** Report Levels 1 and 2 (system-level Spearman + per-query concordance rate). Level 3 is a stretch goal.

### 7.4 Metrics to Correlate

| Retrieval Metric | Answer Quality Metric |
|---|---|
| NDCG@5, NDCG@10, NDCG@20 | Composite score (mean of 3 dimensions) |
| Recall@5, Recall@10, Recall@20 | Comprehensiveness |
| MRR | Factual Consistency |
| MAP | Clinical Utility |
| Precision@5, Precision@10, Precision@20 | Individual dimensions |

**Primary comparison:** NDCG@10 vs. Composite answer quality (the project's declared primary retrieval metric is NDCG@10; see `evaluation.yaml`).

### 7.5 Total LLM Calls Required

- Generator: 65 queries × 5 retrievers × 3 K = **975 calls**
- Evaluator: 975 calls (one per generated answer)
- **Total: 1,950 LLM calls**
- At ~2 seconds per call: ~65 minutes of API time. Feasible in 2 days.
- At ~$0.01 per call (mid-tier model): ~$20. Feasible for an undergraduate project.

---

## 8. Research Novelty Assessment

### 8.1 What Cardiology-Chat Already Does

The Cardiology-Chat system (which this project extends) evaluates retrieval performance and downstream diagnostic performance. However, it does so **separately** — reporting retrieval metrics (NDCG, Recall) in one analysis and generation quality in another, without formally measuring the **correlation** between the two.

### 8.2 What This Experiment Adds

Our experiment explicitly and quantitatively measures: *"Does a higher NDCG@10 actually predict a better generated answer in cardiovascular RAG?"*

This is a **non-trivial empirical contribution** because:

1. **The relationship is assumed but rarely tested.** Most RAG papers report retrieval metrics alone as a proxy for downstream quality. Our experiment tests whether this proxy is reliable.

2. **The LLM generation step is a black box.** We empirically characterize whether this black box preserves, amplifies, or attenuates the retrieval quality signal.

3. **We test across 5 retriever architectures** (sparse, approximate, dense, two hybrid configurations) — a comparative analysis of whether the retrieval-to-generation quality link differs by retriever type.

4. **We analyze failure modes** — queries where good retrieval produces bad answers and vice versa — providing actionable insight for RAG system design.

### 8.3 Honest Assessment of Novelty

This is a **solid empirical contribution for an undergraduate research project**. It is not a methodological breakthrough. The approach of using LLM-as-judge for RAG evaluation is established (RAGAS, ARES, etc.). Our contribution is the **specific empirical analysis** in the cardiovascular domain with a controlled multi-retriever comparison.

The experiment is most valuable if it produces **non-obvious findings** — for example:
- NDCG@10 is a poor predictor of answer quality (LLM generation noise dominates).
- NDCG@10 is a good predictor for treatment queries but poor for mechanism queries.
- BM25 has a stronger retrieval-to-generation quality link than Dense retrieval.

Even a "null result" (strong positive correlation, as expected) is publishable if reported honestly — it empirically validates the common practice of using retrieval metrics as a proxy for downstream quality.

---

## 9. Recommended Method

**Method C — Grade-2 Reference Evidence** with the following specifications:

### 9.1 Reference Evidence Set

For each query Qᵢ:
- **Primary (59 queries):** Reference evidence = all grade-2 qrels chunks for Qᵢ.
- **Fallback (6 queries: Q007, Q013, Q017, Q021, Q055, Q061):** Reference evidence = all grade-1 qrels chunks for Qᵢ, flagged as `grade_1_fallback=True`.
- **Sensitivity analysis:** Report results both for all 65 queries and for the 59 grade-2-only queries.

### 9.2 What Counts as What

| Artifact | Role | Status |
|---|---|---|
| 65 queries | Fixed experimental input | Frozen |
| 828 qrels | Source of reference evidence sets; source of retrieval metrics | Frozen |
| Grade-2 chunks (text) | **Reference evidence** (NOT reference answers) | Frozen, derived from frozen qrels + frozen corpus |
| Retrieval systems (5×) | Fixed experimental conditions | Frozen |
| Retrieved top-K chunks | Generator input (varies by retriever, K) | Deterministically reconstructed from frozen indexes |
| Generated answers | Experimental output | Newly produced |
| LLM evaluator scores | **Automated proxy for answer quality** (NOT clinical ground truth) | Newly produced |
| Retrieval metrics | Independent variable for correlation | Computed from frozen infrastructure |
| Correlation analysis | Primary finding | Newly computed |

### 9.3 What Claims We CAN Make

- "In our 65-query cardiovascular RAG benchmark, NDCG@10 shows a [strength] rank correlation (Spearman ρ = X, p = Y) with automated answer quality scores."
- "Automated evaluation suggests that retrieval quality [is / is not] a reliable indicator of downstream answer quality in this setting."
- "The retriever with the highest NDCG@10 produces the highest-rated answer in X% of queries."
- "Answer quality [does / does not] plateau beyond Recall@K = Z, suggesting [diminishing / sustained] returns from additional evidence retrieval."
- "Among the five retrieval systems tested, [system] exhibits the strongest / weakest retrieval-to-generation quality coupling."

### 9.4 What Claims We CANNOT Make

- ~~"Our generated answers are clinically correct."~~ (No clinical validation.)
- ~~"LLM evaluation scores equal expert judgment."~~ (Automated proxy only.)
- ~~"Higher NDCG *causes* better answers."~~ (Correlation ≠ causation; confounders exist.)
- ~~"This generalizes to all medical RAG systems."~~ (Single corpus, single domain, single generator.)
- ~~"Grade-2 qrels constitute ground truth answers."~~ (They are evidence chunks, not synthesized answers.)

---

## 10. Exact Experimental Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FROZEN (do not modify)                         │
│                                                                        │
│  queries.csv ──── qrels.csv ──── chunks.jsonl ──── indexes/*           │
│     (65)           (828)          (1142)         (BM25/LSH/FAISS)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                            │
       ▼                            ▼                            ▼
  Step 1: BUILD                Step 2: BUILD               Step 3: COMPUTE
  REFERENCE EVIDENCE           RETRIEVAL RUNS              RETRIEVAL METRICS
  SETS                                                     
                                                           
  For each query Qᵢ:          For each (Qᵢ, retriever,   For each (Qᵢ, retriever,
  - Extract grade-2 chunks     K):                         K):
    from qrels + corpus        - Run retriever on Qᵢ       - P@K, R@K, NDCG@K,
  - Store as ref_evidence_i    - Store top-K chunk IDs      MRR, MAP from existing
                                 and text                    code
       │                            │                            │
       │                            ▼                            │
       │                     Step 4: GENERATE                    │
       │                     ANSWERS                             │
       │                                                         │
       │                     For each (Qᵢ, retriever, K):       │
       │                     - Fixed generator LLM               │
       │                     - temperature=0                     │
       │                     - Input: query + retrieved chunks   │
       │                     - Output: answer text               │
       │                            │                            │
       ▼                            ▼                            │
  ┌──────────────────────────────────────┐                       │
  │         Step 5: EVALUATE             │                       │
  │                                      │                       │
  │  For each (Qᵢ, retriever, K):       │                       │
  │  - Fixed evaluator LLM              │                       │
  │  - temperature=0                    │                       │
  │  - Input: query + answer +          │                       │
  │    ref_evidence_i                   │                       │
  │  - Output: scores (1-5) for        │                       │
  │    Comprehensiveness,              │                       │
  │    Factual Consistency,            │                       │
  │    Clinical Utility                │                       │
  └──────────────────────────────────────┘                       │
                    │                                            │
                    ▼                                            ▼
           ┌─────────────────────────────────────────────────────────┐
           │              Step 6: CORRELATION ANALYSIS               │
           │                                                         │
           │  System-level: Spearman(mean_retrieval, mean_quality)   │
           │  Query-level:  concordance rate, per-query rank corr.   │
           │  Breakdown:    by topic, by query_type, by K            │
           │  Divergence:   identify queries with high retrieval      │
           │                but low answer quality (and vice versa)   │
           └─────────────────────────────────────────────────────────┘
```

### 10.1 What is Frozen

- `data/evaluation/queries.csv` (65 queries)
- `data/evaluation/qrels.csv` (828 judgments)
- `data/chunks/chunks.jsonl` (1,142 chunks)
- `indexes/bm25/`, `indexes/lsh/`, `indexes/faiss/`
- `src/retrieval/bm25.py`, `src/retrieval/dense.py`, `src/retrieval/lsh.py`, `src/retrieval/hybrid.py`
- `src/evaluation/retrieval_metrics.py`
- `experiments/configs/evaluation.yaml`, `experiments/configs/retrieval.yaml`

### 10.2 What is Newly Implemented

| Component | Description | Estimated effort |
|---|---|---|
| Reference evidence extractor | Read qrels + chunks.jsonl → per-query grade-2 chunk text sets | ~1 hour |
| Retrieval run reconstructor | Re-run 5 retrievers at K=5,10,20 on 65 queries → store ranked chunk ID + text lists | ~2 hours |
| Per-query retrieval metric computer | Compute NDCG@K, Recall@K, etc. per (query, retriever, K) | ~1 hour (uses existing `retrieval_metrics.py`) |
| Generator module | Fixed LLM API call: query + passages → answer | ~2 hours |
| Evaluator module | Fixed LLM API call: query + answer + ref evidence → structured scores | ~2 hours |
| Correlation analysis script | System-level + query-level analysis, plots | ~3 hours |
| Report generation | Results summary, figures, tables | ~3 hours |
| **Total** | | **~14 hours** |

### 10.3 What is Considered Ground Truth

**Nothing is treated as clinical ground truth.** Explicitly:

- Grade-2 qrels chunks are **reference evidence**, not reference answers.
- LLM evaluator scores are **automated proxies**, not expert assessments.
- Generated answers are **experimental outputs**, not clinical recommendations.

### 10.4 What is Only an Automated Proxy

- All three evaluation dimensions (Comprehensiveness, Factual Consistency, Clinical Utility).
- The composite answer quality score.
- The resulting correlation coefficients describe the relationship between retrieval metrics and *automated* answer quality, not clinically validated quality.

---

## 11. Threats to Validity

### 11.1 Internal Validity

| Threat | Severity | Mitigation |
|---|---|---|
| Circularity between retrieval metrics and answer evaluation (both depend on qrels) | Medium | Acknowledged explicitly; answer evaluation operates on synthesized text, not chunk IDs; factual consistency dimension is independent of retrieval overlap; residual analysis examines divergence cases |
| LLM evaluator unreliability | Medium | Use structured rubric; temperature=0; pilot on 3–5 queries; report exact prompt |
| Generator-evaluator model overlap (if same model) | Low–Medium | Prefer different models; if same model, disclose and discuss |
| K-value nesting (top-5 ⊂ top-10 ⊂ top-20) | Low | Treat K as a fixed condition; do not pool across K; report per-K results separately |

### 11.2 External Validity

| Threat | Severity | Mitigation |
|---|---|---|
| Single domain (cardiovascular only) | High | Acknowledge; do not claim generalization beyond cardiology |
| Small corpus (22 documents, 1,142 chunks) | High | Acknowledge; note that results may differ on larger corpora |
| Custom queries (not from a standard benchmark) | Medium | Acknowledge; note bespoke nature; report query type breakdown |
| Single generator LLM | Medium | Acknowledge; future work could test multiple generators |

### 11.3 Construct Validity

| Threat | Severity | Mitigation |
|---|---|---|
| LLM scores ≠ clinical quality | High | Never claim clinical correctness; label all scores as automated proxies |
| Grade-2 evidence ≠ comprehensive clinical truth | Medium | Grade-2 represents "essential reading" per annotation guidelines, not exhaustive truth; acknowledge |
| 6 queries with no grade-2 evidence | Low | Fallback to grade-1 + sensitivity analysis excluding these queries |

---

## 12. Go / No-Go Decision

### GO.

**The method is defensible and implementable in 2 days.**

**Rationale:**

1. **Defensible.** The approach uses frozen reference evidence (grade-2 qrels chunks) rather than fabricated reference answers. The circularity with retrieval metrics is partial and acknowledged, not disqualifying — the LLM generation step introduces genuine variance, and the factual consistency dimension is independent of retrieval overlap. The experiment is framed as measuring automated proxies, not clinical truth.

2. **Implementable.** The pipeline requires ~14 hours of implementation across 6 clearly scoped modules. All frozen infrastructure is in place. The total computational cost is ~1,950 LLM API calls (~$20, ~65 minutes).

3. **Novel enough.** The explicit quantitative analysis of retrieval-to-generation quality correlation across 5 retrieval architectures is a genuine empirical contribution that extends beyond what Cardiology-Chat already reports.

4. **Failure modes are informative.** Even if the experiment finds a strong positive correlation (the expected result), the per-query divergence analysis and the per-topic breakdown provide actionable findings. A weak or inconsistent correlation would be a genuinely surprising and publishable result.

**Conditions for the GO:**

- The paper must explicitly state that evaluator scores are automated proxies, not clinical assessments.
- The circularity between retrieval metrics and answer evaluation must be disclosed and discussed.
- The 6 fallback queries must be flagged and sensitivity-analyzed.
- The exact evaluation prompt must be frozen before any experiment runs and reported in full.
- All claims must be appropriately hedged (correlation, not causation; this domain, not general).
