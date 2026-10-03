# Relevance Judgments (qrels) — Schema Documentation

**Project:** Hybrid Text Retrieval for Evidence-Grounded Cardiovascular Diagnostic Reasoning  
**Stage:** 2A — Schema defined; judgments NOT yet created  
**Status:** ⚠ EMPTY — do not fabricate relevance judgments

---

## Purpose

Qrels (query-relevance judgments) map each evaluation query to a set of
chunks and their relevance grades.  They are the ground truth used to compute
Precision@k, Recall@k, MRR, MAP, and NDCG@k.

**Judgments must be created by domain experts or by pooling retrieval runs
and manually grading the pooled results.  They must never be fabricated.**

---

## File Format

### queries.csv

```
query_id,query_text,topic,notes
Q001,"What are the pharmacological treatments for HFrEF?",heart_failure,
Q002,...
```

| Column | Type | Description |
|---|---|---|
| `query_id` | string | Unique query identifier, e.g. `Q001` |
| `query_text` | string | Full natural-language query |
| `topic` | string | Cardiovascular topic (matches corpus topic labels) |
| `notes` | string | Optional annotation notes |

### qrels.csv

```
query_id,chunk_id,relevance,judge_id,notes
Q001,AF01-0003,2,judge1,Directly discusses HFrEF pharmacotherapy
Q001,HF01-0012,2,judge1,Core treatment section
Q001,HF02-0007,1,judge1,Partial — mentions GDMT but not HFrEF specifically
Q001,CAD01-0042,0,judge1,Not relevant
```

| Column | Type | Description |
|---|---|---|
| `query_id` | string | Foreign key → queries.csv |
| `chunk_id` | string | Foreign key → data/chunks/chunks.jsonl |
| `relevance` | int | Graded relevance: **0** not relevant, **1** partially relevant, **2** highly relevant |
| `judge_id` | string | Annotator identifier (for inter-annotator agreement tracking) |
| `notes` | string | Optional free-text annotation notes |

---

## Relevance Grade Definitions

| Grade | Label | Definition |
|---|---|---|
| 2 | Highly relevant | The chunk directly and substantively answers the query. A clinician would consider it essential reading for this question. |
| 1 | Partially relevant | The chunk is related to the query topic but does not directly answer it, or provides only peripheral evidence. |
| 0 | Not relevant | The chunk has no meaningful connection to the query. |

---

## Annotation Guidelines

1. Annotators should be cardiology-trained or working from validated clinical resources.
2. Each query should be judged by at least two independent annotators.
3. Inter-annotator agreement (Cohen's κ) should be reported before freezing qrels.
4. Disagreements should be resolved by consensus or a third judge.
5. Partial relevance (grade 1) should be used sparingly; prefer a clear 0/2 split where possible.
6. The annotation pool should be constructed by pooling the top-10 results from each retriever (BM25, LSH, Dense, Hybrid) to avoid pool bias.

---

## Planned Evaluation Queries

Minimum target: **30 queries** covering all 9 corpus topics.

| Topic | Target queries |
|---|---|
| Myocardial infarction | 4 |
| Coronary artery disease | 3 |
| Heart failure | 5 |
| Atrial fibrillation / arrhythmia | 4 |
| Hypertension | 3 |
| Dyslipidemia | 2 |
| Valvular heart disease | 2 |
| Cardiomyopathy | 4 |
| Cardiovascular prevention | 3 |
| **Total** | **30** |

---

## Current Status

- [x] Schema defined
- [x] Template queries created (`queries_template.csv`)
- [ ] Real evaluation queries written and reviewed
- [ ] Chunk pool constructed from retriever runs
- [ ] Manual relevance judgments created
- [ ] Inter-annotator agreement computed
- [ ] Qrels frozen for evaluation

---

## File Locations

```
data/evaluation/
├── queries_template.csv    # Template — NOT the real eval set
├── qrels_schema.md         # This document
├── queries.csv             # (to be created — real evaluation queries)
└── qrels.csv               # (to be created — manual relevance judgments)
```

**Do not proceed to scientific retrieval evaluation until `qrels.csv` is frozen.**
