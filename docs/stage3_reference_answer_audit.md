# Stage 3 — Reference Answer Audit

## 1. Executive Summary

REFERENCE ANSWERS NOT AVAILABLE. 

The evaluation dataset contains well-structured queries and chunk-level relevance judgments (qrels) for retrieval evaluation, but it lacks any synthesized, "gold standard" reference answers for downstream generation evaluation.

## 2. Query Dataset

- **File:** `data/evaluation/queries.csv`
- **Row count:** 65 queries
- **Columns:** `query_id`, `query_text`, `topic`, `query_type`
- **Representative examples:**
  1. `Q001,What pharmacological treatments are recommended after myocardial infarction?,myocardial_infarction,treatment`
  2. `Q020,What risk prediction models exist for heart failure with preserved ejection fraction?,heart_failure,prognosis`
  3. `Q054,What is the genetic basis of dilated cardiomyopathy?,cardiomyopathy,mechanism`

## 3. Qrels Dataset

- **File:** `data/evaluation/qrels.csv`
- **Row count:** 828 judgments
- **Columns:** `query_id`, `chunk_id`, `relevance`
- **Grade distribution:** Grade 2 (Highly relevant): 530; Grade 1 (Partially relevant): 298
- **Interpretation:** The qrels map queries to specific document chunks that contain relevant clinical evidence. As stated in `annotation_guidelines.md` and `qrels_schema.md`, these are purely relevance judgments used for IR metrics (NDCG@k, Recall@k). They are retrieval evidence, not synthesized text, and cannot be used directly as reference answers for text generation. 

## 4. Reference Answer Search

A repository-wide search was conducted for terms such as `reference_answer`, `gold_answer`, `ground_truth`, `expected_answer`, `MedQA`, `PubMedQA`, etc. 
No files containing reference answers were found. The keyword searches only yielded matches in:
- `src/evaluation/diagnostic_metrics.py` (which contains only `NotImplementedError` stubs for evaluating downstream answers and uses terms like "Ground-truth answer" in docstrings).
- `src/rag/pipeline.py` (which contains unimplemented stubs for the RAG generation step).
- `data/evaluation/annotation_guidelines.md` and `qrels_schema.md` (which use the term "ground truth" to describe the qrels relevance judgments for retrieval, not for generated answers).

## 5. Query Provenance

The origin of the 65 queries can be established, but they do not originate from an external public benchmark. 
**Inference:** The queries were custom-authored specifically for the 22 review documents in Corpus V1. Many queries directly mirror the titles of the source documents. For example, Q061 ("What strategies does the World Heart Federation recommend for secondary prevention of CVD?") corresponds perfectly with document `PREV01` ("World Heart Federation Roadmap for Secondary Prevention of Cardiovascular Disease"). The queries are entirely bespoke to this dataset.

## 6. Retrieval Result Availability

Current retrieval results are stored as JSON files containing aggregated metrics (e.g., NDCG, MRR, P@k), latencies, and candidate counts, but they do not explicitly store the top-k chunk IDs or the retrieved text:
- `experiments/results/stage2c_lsh_vs_bm25.json`
- `experiments/results/stage2d_dense.json`
- `experiments/results/stage2e_hybrid_rrf.json`

However, the retrieved top-k chunk IDs and text **can be deterministically reconstructed** because the frozen corpus (`data/chunks/chunks.jsonl`), indexes (`indexes/`), configurations (`experiments/configs/retrieval.yaml`), and retriever implementations (`src/retrieval/`) are all present and deterministic. Supported K values evaluated and configured are 5, 10, and 20.

## 7. Evaluation Infrastructure

- **Retrieval Metrics:** Implemented in `src/evaluation/retrieval_metrics.py`. It includes standard IR functions for `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `average_precision`, `mean_average_precision`, and `ndcg_at_k`. Tests are fully passing.
- **Diagnostic Metrics:** Outlined in `src/evaluation/diagnostic_metrics.py`. It defines functions for Exact Match, token F1, and Faithfulness, but all functions are currently empty and raise `NotImplementedError`.

## 8. Can We Implement Reference-Evidence Coverage?

**NO.**

**Explanation:** We cannot evaluate reference-answer evidence coverage because we do not have reference answers. To compute this coverage, the pipeline needs to check if the retrieved evidence encompasses the information contained in a known "gold standard" medical answer. Since the current dataset only links queries to chunks without providing a synthesized gold answer, there is nothing against which to measure the coverage.

## 9. Recommended Next Step

Since reference answers do not exist:
- **Do NOT invent them or use LLM generation to fake them.**
- **Missing Information:** We lack expert-synthesized, concise medical reference answers (gold answers) for the 65 benchmark queries.
- **Public Benchmark Linkage:** Because the queries are custom-designed for the 22 documents in Corpus V1, they cannot be readily linked to or replaced by an existing public benchmark like MedQA without abandoning the frozen corpus and relevance judgments.
- **Smallest Defensible Next Step:** Have domain experts (clinicians) manually review the 65 queries alongside their grade-2 (highly relevant) chunks and author concise, evidence-grounded reference answers. These should be saved in a new file (e.g., `data/evaluation/reference_answers.csv`) before proceeding with the diagnostic pipeline implementation.

## 10. Frozen Artifacts

The following artifacts and files were strictly audited and **NOT modified**:
- `data/evaluation/queries.csv`
- `data/evaluation/qrels.csv`
- `data/evaluation/annotation_guidelines.md`
- `data/evaluation/evaluation_dataset_report.md`
- `data/evaluation/qrels_schema.md`
- `data/chunks/chunks.jsonl`
- `data/corpus_acquisition_report.md`
- `data/source_manifest.csv`
- `experiments/configs/evaluation.yaml`
- `experiments/configs/retrieval.yaml`
- `experiments/results/stage2c_lsh_vs_bm25.json`
- `experiments/results/stage2d_dense.json`
- `experiments/results/stage2e_hybrid_rrf.json`
- `src/retrieval/*`
- `src/evaluation/*`
- All indexes in `indexes/`
- All test files in `tests/`
