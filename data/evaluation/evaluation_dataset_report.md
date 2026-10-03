# Evaluation Dataset Report — Corpus V1
**Project:** Hybrid Text Retrieval for Evidence-Grounded Cardiovascular Diagnostic Reasoning  
**Dataset version:** v1.0 (Frozen)  
**Date:** 2026-10-01

---

## Summary

| Metric | Value |
|---|---|
| Total queries | **65** |
| Total qrel entries | **828** |
| Grade-2 (highly relevant) entries | 530 (64.0%) |
| Grade-1 (partially relevant) entries | 298 (36.0%) |
| Mean relevant chunks per query | 12.7 |
| Median relevant chunks per query | 13.0 |
| Min relevant chunks per query | 3 |
| Max relevant chunks per query | 20 |
| Queries with zero relevant chunks | 0 |
| Corpus documents represented in qrels | 22 / 22 (all) |

---

## 1. Query Count

**65 queries** across 9 cardiovascular topics.

---

## 2. Queries per Topic

| Topic | Queries | Target |
|---|---|---|
| Atrial fibrillation / arrhythmia | 10 | 10 |
| Heart failure | 10 | 10 |
| Myocardial infarction | 8 | 8 |
| Coronary artery disease | 8 | 8 |
| Hypertension | 7 | 7 |
| Cardiomyopathy | 7 | 7 |
| Dyslipidemia | 5 | 5 |
| Valvular heart disease | 5 | 5 |
| Cardiovascular prevention | 5 | 5 |
| **Total** | **65** | **65** |

✓ All topic targets met exactly.

---

## 3. Queries per Query Type

| Query type | Count |
|---|---|
| treatment | 27 |
| diagnostic_criteria | 17 |
| mechanism | 10 |
| prognosis | 6 |
| risk_factors | 4 |
| complication | 1 |

Treatment and diagnostic queries are most common, reflecting the clinical focus of the corpus.

---

## 4. Total Positive Qrels

**828 qrel entries**, all positive (grade ≥ 1):
- Grade 2 (highly relevant): 530 entries (64%)
- Grade 1 (partially relevant): 298 entries (36%)

Grade-0 entries are not stored (standard IR evaluation practice).

---

## 5. Relevant Chunks per Query

| Statistic | Value |
|---|---|
| Mean | 12.7 |
| Median | 13.0 |
| Minimum | 3 (Q039, Q065) |
| Maximum | 20 (capped at 20 for over-represented queries) |

All 65 queries have at least 3 relevant chunks — the minimum required for meaningful NDCG and Recall@k computation at k=5,10,20.

---

## 6. Queries with Zero Relevant Chunks

**0 queries.** All 65 queries have at least 3 relevant chunks.

---

## 7. Duplicate Queries

- Duplicate query IDs: **0**
- Duplicate query texts: **0**

---

## 8. Documents Represented in Qrels

**All 22 Corpus V1 documents** are represented in the qrels. Every document contributes at least one relevant chunk to at least one query.

| Document | Topic |
|---|---|
| MI01, MI02, MI03 | myocardial_infarction |
| CAD01, CAD02, CAD03 | coronary_artery_disease |
| HF01, HF02, HF03, HF04 | heart_failure |
| AF01, AF02, AF03, AF04 | atrial_fibrillation_arrhythmia |
| HTN01, HTN02 | hypertension |
| LIPID01 | dyslipidemia |
| VHD01 | valvular_heart_disease |
| DCM01, CMY02 | cardiomyopathy |
| PREV01, PREV02 | cardiovascular_prevention |

---

## 9. Chunks Represented in Qrels

828 unique (query, chunk) pairs across 1,142 total corpus chunks.

---

## 10. Single-Document Dominance (Noted, Not a Problem)

11 queries draw ≥95% of their relevant chunks from a single document. This is an expected property of a small, focused corpus where some topics are covered by only one source (e.g., Q020 "HFpEF risk prediction models" → only HF03 covers this topic; Q049 "valvular heart disease epidemiology" → only VHD01 covers this topic).

This does not invalidate the evaluation dataset — it accurately reflects the corpus coverage. Retrieval experiments should note this limitation when interpreting per-topic results for dominated queries.

---

## 11. Annotation Method

**Pool construction:**
- BM25 top-20 per query
- Keyword-augmented scan of up to 10 additional chunks per query

This dual-source pool reduces BM25 pool bias so that retrievers that rank differently from BM25 can still be credited for finding relevant chunks.

**Grading method:**
- Grade 2: regex co-occurrence of all must-have terms + at least one nice-to-have term in top-10 BM25 rank; or top-5 with score >15
- Grade 1: regex co-occurrence of all must-have terms in top-20 BM25 rank
- Manually verified and supplemented for queries with fewer than 3 candidates

**Limitation:** Initial annotation used automated rule-based patterns verified against actual chunk text. Human expert review of all grade-2 judgments is recommended before reporting final retrieval experiment results.

---

## 12. Known Issues

| Issue | Severity | Detail |
|---|---|---|
| HTN02 is a 2-page editorial | Minor | Hypertension queries rely mainly on HTN01 (53 chunks). HTN02 contributes only 6 chunks to a few queries. |
| Single-source dominance in 11 queries | Noted | Expected corpus property; see Section 10. |
| No human expert review yet | To-do | Automated annotation should be validated by a cardiologist for grade-2 assignments. |
| Reference-list chunks in LIPID01/HF03 | Marginal | Some annotated chunks are citation-dense; these were filtered where >60% of lines were citations. |

---

## 13. Evaluation Configuration

| Setting | Value |
|---|---|
| k values | 5, 10, 20 |
| Primary metric | NDCG@10 |
| Relevance threshold (binary) | ≥ 1 |
| Supports graded NDCG | Yes (grades 0/1/2) |
| Config file | `experiments/configs/evaluation.yaml` |

---

## 14. Test Results

All 33 evaluation dataset tests pass:
```
tests/test_evaluation_dataset.py  33/33 ✓
Full suite                       248/248 ✓
```

One expected UserWarning (single-document dominance — documented).

---

## 15. Files

| File | Description |
|---|---|
| `data/evaluation/queries.csv` | 65 queries with topic and query_type labels |
| `data/evaluation/qrels.csv` | 828 relevance judgments (grade 1 and 2) |
| `data/evaluation/annotation_guidelines.md` | Annotation protocol and grading rules |
| `data/evaluation/evaluation_dataset_report.md` | This document |
| `experiments/configs/evaluation.yaml` | Evaluation protocol configuration |

---

## Freeze Decision

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     EVALUATION DATASET — READY TO FREEZE                    ║
║                                                              ║
║  65 queries  ·  9 topics  ·  828 qrel entries               ║
║  Min 3 rel/query  ·  All 22 documents represented           ║
║  0 duplicate queries  ·  0 queries with zero relevance      ║
║  248/248 tests passing                                       ║
║                                                              ║
║  Note: Human expert review of grade-2 judgments is          ║
║  recommended before reporting final experiment results.      ║
║  The dataset is structurally valid and ready for use.       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```
