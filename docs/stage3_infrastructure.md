# Stage 3 — Reference-Evidence & Retrieval-Run Infrastructure

## 1. Purpose

This document details the deterministic infrastructure implemented for **Stage 3** of the Cardiology-Chat research project.

The overarching research question of this study is:

> **"Do conventional retrieval metrics reliably predict automated downstream answer quality in cardiovascular RAG?"**

Because no human reference answers exist for the 65 evaluation queries and no external clinical QA benchmark can be substituted without distorting query-corpus provenance, the approved methodology leverages frozen relevance judgments (qrels) as **reference evidence**.

> [!IMPORTANT]
> **Methodological Disclaimer:**  
> **Reference evidence is not clinical ground truth.**  
> It represents passage-level relevance judgments extracted deterministically from the frozen qrels dataset. Evaluator scores comparing downstream answers against this evidence measure automated proxy concordance with retrieved source material, not clinical correctness or medical validity.

---

## 2. Absolute Frozen Artifacts

The following artifacts are frozen and were untouched, unmodified, and preserved with zero changes:

- `data/evaluation/queries.csv` (65 queries)
- `data/evaluation/qrels.csv` (828 relevance judgments)
- `data/evaluation/annotation_guidelines.md`
- `data/evaluation/evaluation_dataset_report.md`
- `data/evaluation/qrels_schema.md`
- `data/chunks/chunks.jsonl` (1,142 chunks from 22 open-access cardiovascular papers)
- `data/source_manifest.csv`
- `indexes/bm25/*` (frozen BM25 inverted index)
- `indexes/faiss/*` (frozen dense IndexFlatIP FAISS index)
- `indexes/lsh/*` (frozen MinHash-LSH index, threshold=0.05)
- `src/retrieval/bm25.py`
- `src/retrieval/dense.py`
- `src/retrieval/lsh.py`
- `src/retrieval/hybrid.py`
- `src/evaluation/retrieval_metrics.py`
- `experiments/configs/retrieval.yaml`
- `experiments/configs/evaluation.yaml`
- Existing Stage 2 experiment result files (`stage2c_lsh_vs_bm25.json`, `stage2d_dense.json`, `stage2e_hybrid_rrf.json`)

---

## 3. Reference Evidence Construction

Reference evidence was constructed deterministically in `src/evaluation/reference_evidence.py` adhering to the following rules:

1. **Query Resolution:** All 65 evaluation queries from `data/evaluation/queries.csv` are loaded.
2. **Relevance Filtering:** For each query, qrels rows are partitioned by relevance grade:
   - Grade 2: Highly relevant chunks.
   - Grade 1: Partially relevant chunks.
3. **Primary Set Selection:** If a query possesses at least one grade-2 chunk, all grade-2 chunks constitute its reference evidence set (`grade_2_available = True`, `grade_1_fallback = False`).
4. **Corpus Text & Metadata Binding:** Each chunk ID is resolved against `data/chunks/chunks.jsonl`, binding the exact, unaltered chunk text and document metadata (`doc_id`, `start_char`, `end_char`, `title`, `source`, `section`, `chunk_index`, `total_chunks`).
5. **Deterministic Ordering:** Chunks within each query's reference evidence set are sorted lexicographically by `chunk_id`.
6. **Integrity Enforcement:** Validated that no duplicate chunk IDs exist within any query's evidence set, and 100% of chunk IDs resolve to the frozen corpus.

### Statistics of the Reference Evidence Set:
- **Total queries:** 65
- **Primary analysis queries (Grade-2 available):** 59 queries (90.8%)
- **Total Grade-2 chunks across primary queries:** 530 chunk occurrences
- **Mean Grade-2 chunks per primary query:** 8.98 chunks (range: 1 – 20 chunks)

---

## 4. The Six Grade-1-Only Queries

Exactly six queries have zero grade-2 chunks in the frozen qrels:

| Query ID | Topic | Query Type | Available Grade-1 Chunks |
| :--- | :--- | :--- | :--- |
| **Q007** | Myocardial Infarction | Complication | 7 |
| **Q013** | Coronary Artery Disease | Diagnostic Criteria | 12 |
| **Q017** | Heart Failure | Diagnostic Criteria | 4 |
| **Q021** | Heart Failure | Complication | 13 |
| **Q055** | Cardiomyopathy | Mechanism | 8 |
| **Q061** | Cardiovascular Prevention | Biomarker | 10 |

### Approved Protocol for These Queries:
- They are **NOT silently mixed** into the primary analysis.
- In `reference_evidence.json`, they are explicitly tagged:
  ```json
  "grade_2_available": false,
  "grade_1_fallback": true
  ```
- Their reference evidence is composed strictly of their `relevance == 1` chunks.
- In downstream evaluation, they are segregated and reserved exclusively for secondary sensitivity analysis, preserving the integrity of the 59-query primary dataset.

---

## 5. Retrieval Run Reconstruction

Retrieval run reconstruction is implemented in `src/evaluation/retrieval_reconstruction.py`. It executes the frozen retrievers and packages ranked passages across all permutations required for downstream generation:

- **Queries:** All 65 evaluation queries.
- **Retrievers:**
  1. `bm25`: BM25 baseline ($k_1=1.5, b=0.75$)
  2. `dense`: Sentence-Transformers `all-MiniLM-L6-v2` + FAISS `IndexFlatIP`
  3. `lsh`: MinHash-LSH ($k=3$ word shingles, 128 permutations, threshold 0.05)
  4. `bm25_dense_rrf`: System A RRF fusion ($RRF_k=60$, candidate pool depth = 20)
  5. `bm25_dense_lsh_rrf`: System B RRF fusion ($RRF_k=60$, candidate pool depth = 20)
- **Cutoffs ($K$):** $K \in \{5, 10, 20\}$
- **Total reconstructed runs:** $65 \times 5 \times 3 = 975$ runs.

Each reconstructed run captures:
- `query_id`, `query_text`, `topic`, `query_type`
- `retriever` ID and `retriever_name`
- `k`
- `n_results`
- `results`: Ranked list (1..$K$) containing `rank`, `chunk_id`, `score`, `text`, `doc_id`, and `metadata` (including `constituent_ranks` and `rrf_k` for hybrid systems).

---

## 6. Metric Computation

Metric computation strictly reuses the frozen `src/evaluation/retrieval_metrics.py` pure functions. No metric logic was rewritten or altered:

- **Per-run metrics:** Computed for each $(q, \text{retriever}, K)$:
  - $\text{Precision}@K$: Binary relevance ($\text{grade} \ge 1$)
  - $\text{Recall}@K$: Binary relevance ($\text{grade} \ge 1$)
  - $\text{NDCG}@K$: Graded relevance ($\text{grades} \in \{0, 1, 2\}$, logarithmic discount)
  - $\text{Reciprocal Rank (RR)}$: First relevant passage rank
  - $\text{Average Precision (AP)}$: Ranked precision at each hit
- **System-level aggregates:** Computed across all 65 queries:
  - Mean Precision@5, 10, 20
  - Mean Recall@5, 10, 20
  - Mean NDCG@5, 10, 20
  - Mean Reciprocal Rank (MRR) over top-20 pool (Stage 2 IR protocol)
  - Mean Average Precision (MAP) over top-20 pool (Stage 2 IR protocol)

---

## 7. Deterministic Guarantees

The entire pipeline is guaranteed to be 100% deterministic:
1. **Corpus & Index Loading:** Indexes are loaded from frozen serialized binaries.
2. **Deterministic Tie-Breaking:** In `HybridRetriever`, ties are broken alphabetically by `chunk_id`.
3. **Prefix Invariance:** Verified that for all retrievers and queries, the top 5 and top 10 results are exact prefixes of the top 20 candidate pool.
4. **Repeated Execution Invariance:** Executing retrieval reconstruction repeatedly produces identical ranked IDs, ranks, and scores.

---

## 8. Output Schemas

### 1. `data/evaluation/reference_evidence.json`
```json
{
  "metadata": {
    "schema_version": "1.0",
    "description": "Deterministic reference evidence for Stage 3 downstream RAG evaluation.",
    "methodology_disclaimer": "Reference evidence is not clinical ground truth.",
    "total_queries": 65,
    "primary_analysis_queries_count": 59,
    "sensitivity_fallback_queries_count": 6,
    "sensitivity_fallback_query_ids": ["Q007", "Q013", "Q017", "Q021", "Q055", "Q061"]
  },
  "queries": [
    {
      "query_id": "Q001",
      "query_text": "What pharmacological treatments are recommended after myocardial infarction?",
      "topic": "myocardial_infarction",
      "query_type": "treatment",
      "grade_2_available": true,
      "grade_1_fallback": false,
      "n_evidence_chunks": 12,
      "reference_evidence": [
        {
          "chunk_id": "CAD01-0001",
          "relevance": 2,
          "text": "...",
          "doc_id": "CAD01",
          "start_char": 0,
          "end_char": 1899,
          "title": "...",
          "source": "...",
          "section": "",
          "strategy": "sentence",
          "chunk_index": 0,
          "total_chunks": 50
        }
      ]
    }
  ]
}
```

### 2. `experiments/results/stage3_retrieval_runs.json`
```json
{
  "metadata": {
    "schema_version": "1.0",
    "total_runs": 975,
    "n_queries": 65,
    "retrievers": ["bm25", "dense", "lsh", "bm25_dense_rrf", "bm25_dense_lsh_rrf"],
    "k_values": [5, 10, 20]
  },
  "runs": [
    {
      "query_id": "Q001",
      "query_text": "...",
      "retriever": "bm25",
      "retriever_name": "BM25",
      "k": 10,
      "n_results": 10,
      "results": [
        {
          "rank": 1,
          "chunk_id": "CAD01-0001",
          "score": 12.345,
          "text": "...",
          "doc_id": "CAD01",
          "metadata": { ... }
        }
      ]
    }
  ]
}
```

### 3. `experiments/results/stage3_retrieval_metrics.json`
Contains:
- `stage2_validation`: Status of baseline reproduction per system and metric.
- `aggregate_metrics`: Summary metrics per retriever and K.
- `per_query_metrics`: Granular scores for all 975 $(q, \text{retriever}, K)$ combinations.

---

## 9. Validation Against Stage 2 Baseline

Reconstructed retrieval results were aggregated and compared against the frozen Stage 2 reports (`stage2c_lsh_vs_bm25.json`, `stage2d_dense.json`, `stage2e_hybrid_rrf.json`).

All 25 metric comparisons passed validation within acceptable numerical tolerance ($\Delta \le 0.0001$):

| System | Metric | Stage 2 Expected | Reconstructed Value | Absolute $\Delta$ | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | NDCG@5 | 0.5389 | 0.5389 | 0.00000 | **PASSED** |
| | NDCG@10 | 0.5188 | 0.5188 | 0.00000 | **PASSED** |
| | NDCG@20 | 0.5514 | 0.5514 | 0.00000 | **PASSED** |
| | MRR | 0.7452 | 0.7452 | 0.00000 | **PASSED** |
| | MAP | 0.3783 | 0.3783 | 0.00000 | **PASSED** |
| **Dense** | NDCG@5 | 0.4095 | 0.4095 | 0.00000 | **PASSED** |
| | NDCG@10 | 0.3781 | 0.3781 | 0.00000 | **PASSED** |
| | NDCG@20 | 0.4175 | 0.4175 | 0.00000 | **PASSED** |
| | MRR | 0.5928 | 0.5928 | 0.00000 | **PASSED** |
| | MAP | 0.2478 | 0.2478 | 0.00000 | **PASSED** |
| **LSH** | NDCG@5 | 0.0814 | 0.0814 | 0.00000 | **PASSED** |
| | NDCG@10 | 0.0739 | 0.0740 | 0.00010 | **PASSED** |
| | NDCG@20 | 0.0738 | 0.0739 | 0.00010 | **PASSED** |
| | MRR | 0.1750 | 0.1750 | 0.00000 | **PASSED** |
| | MAP | 0.0386 | 0.0385 | 0.00010 | **PASSED** |
| **SysA_BD (BM25+Dense)** | NDCG@5 | 0.5309 | 0.5309 | 0.00000 | **PASSED** |
| | NDCG@10 | 0.5097 | 0.5097 | 0.00000 | **PASSED** |
| | NDCG@20 | 0.5395 | 0.5395 | 0.00000 | **PASSED** |
| | MRR | 0.7560 | 0.7560 | 0.00000 | **PASSED** |
| | MAP | 0.3610 | 0.3610 | 0.00000 | **PASSED** |
| **SysB_BDL (BM25+Dense+LSH)** | NDCG@5 | 0.5301 | 0.5301 | 0.00000 | **PASSED** |
| | NDCG@10 | 0.5018 | 0.5018 | 0.00000 | **PASSED** |
| | NDCG@20 | 0.5349 | 0.5349 | 0.00000 | **PASSED** |
| | MRR | 0.7459 | 0.7459 | 0.00000 | **PASSED** |
| | MAP | 0.3544 | 0.3544 | 0.00000 | **PASSED** |

---

## 10. Test Suite Execution

A dedicated suite of 28 unit and integration tests was introduced:

- `tests/test_reference_evidence.py` (11 tests):
  - Query coverage (65/65 queries present)
  - 59 queries flagged with `grade_2_available = True`
  - 6 queries flagged with `grade_1_fallback = True` (Q007, Q013, Q017, Q021, Q055, Q061)
  - Fallback queries contain exclusively Grade 1 chunks
  - Primary queries contain exclusively Grade 2 chunks
  - Zero unresolvable chunk IDs; text matches frozen corpus bit-for-bit
  - Deterministic serialization invariance
  - Absence of forbidden clinical ground-truth terminology
  - Reader class lookup and properties
- `tests/test_retrieval_reconstruction.py` (17 tests):
  - Run matrix count ($65 \times 5 \times 3 = 975$ runs)
  - Query, retriever, and K coverage
  - Exact K result count where possible
  - Sequential 1..K ranking without gaps
  - Uniqueness of retrieved chunks within each run
  - Chunk ID resolution to frozen corpus
  - Repeated execution determinism
  - Stage 2 baseline metric reproduction for all 5 systems
  - Reader class lookup and query filtering

### Full Test Suite Results:
- Prior baseline: 354 passed
- Added tests: 28 passed
- Total suite: **382 passed, 0 failed, 1 warning (expected)**

---

## 11. Known Limitations

1. **LSH Candidate Sparseness:** As established in Stage 2C, MinHash-LSH yields 0 candidates for 27 queries due to word 3-gram vocabulary mismatch on short queries. For these queries, LSH runs correctly contain 0 results, resulting in Precision=0, Recall=0, NDCG=0, RR=0, and AP=0.
2. **Evaluation Scope:** Retrieval metrics assess ranking and coverage of qrels-annotated chunks. They do not evaluate clinical validity, semantic coherence, or hallucination rates of generated text.
3. **Automated Proxies:** Subsequent downstream generation and evaluation will measure automated proxy alignment against reference evidence, strictly following the approved reference-answer-free design.
