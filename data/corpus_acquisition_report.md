# Corpus V1 Acquisition Report
**Project:** Hybrid Text Retrieval for Evidence-Grounded Cardiovascular Diagnostic Reasoning  
**Date:** 2026-10-01  
**Status:** CORPUS V1 — READY TO FREEZE

---

## 1. Overview

| Metric | Value |
|---|---|
| Target documents | 22 |
| Successfully verified | 22 |
| Successfully downloaded | 22 |
| Failed downloads | 0 |
| Ambiguous license records | 0 |
| Duplicate documents removed | 1 (`Systematic Review Opportunities...` — identical SHA-256 to PREV02) |
| Missing documents resolved | 2 (HTN01, HTN02 — manually downloaded) |

---

## 2. License Verification

All 22 documents were independently verified via the NCBI PMC E-utilities XML API (`efetch`).  
License was confirmed via the `ali:license_ref` element in each article's `<permissions>` block.

| Document | PMCID | License | License URL |
|---|---|---|---|
| MI01 | PMC12429806 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| MI02 | PMC11432415 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| MI03 | PMC11267324 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| CAD01 | PMC11354290 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| CAD02 | PMC11273976 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| CAD03 | PMC9773289 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| HF01 | PMC10502425 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| HF02 | PMC11107472 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| HF03 | PMC10614113 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| HF04 | PMC11172756 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| AF01 | PMC10518883 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| AF02 | PMC10902941 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| AF03 | PMC11264032 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| AF04 | PMC11759983 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| HTN01 | PMC11263885 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| HTN02 | PMC10277911 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| LIPID01 | PMC10975559 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| VHD01 | PMC10054046 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| DCM01 | PMC11546582 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| CMY02 | PMC11642006 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| PREV01 | PMC10809857 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |
| PREV02 | PMC12059770 | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ |

**All 22 documents are confirmed CC BY 4.0.** This license explicitly permits use, adaptation, and reproduction for any purpose (including computational and AI research) with attribution.

---

## 3. Topic Coverage

| Topic | Documents | IDs |
|---|---|---|
| Myocardial infarction | 3 | MI01, MI02, MI03 |
| Coronary artery disease | 3 | CAD01, CAD02, CAD03 |
| Heart failure | 4 | HF01, HF02, HF03, HF04 |
| Atrial fibrillation / arrhythmia | 4 | AF01, AF02, AF03, AF04 |
| Hypertension | 2 | HTN01, HTN02 |
| Dyslipidemia | 1 | LIPID01 |
| Valvular heart disease | 1 | VHD01 |
| Cardiomyopathy | 2 | DCM01, CMY02 |
| Cardiovascular prevention | 2 | PREV01, PREV02 |

---

## 4. Corpus Extraction Statistics

| Metric | Value |
|---|---|
| Total documents | 22 |
| Total pages | 387 |
| Total raw characters | 1,876,248 |
| Total cleaned characters | 1,843,593 |
| Total tokens (cleaned) | 248,894 |
| Total chunks | 1,142 |
| Chunking strategy | sentence |
| Chunk size target | 256 tokens |
| Overlap | 32 tokens |

### Chunk Token-Length Distribution

| Statistic | Value |
|---|---|
| Min | 39 |
| P25 | 237 |
| Median | 246.0 |
| Mean | 239.7 |
| P75 | 253 |
| Max | 337 |
| Std Dev | 26.2 |
| CV (std/mean) | 0.11 |

---

## 5. Per-Document Statistics

| ID | Pages | Tokens | Chunks | Mean Tok | Notes |
|---|---|---|---|---|---|
| MI01 | 14 | 7,221 | 33 | 247.2 | |
| MI02 | 16 | 10,452 | 48 | 241.4 | |
| MI03 | 11 | 4,814 | 22 | 235.0 | |
| CAD01 | 34 | 23,091 | 107 | 241.6 | |
| CAD02 | 8 | 4,391 | 20 | 239.1 | |
| CAD03 | 10 | 9,427 | 47 | 219.0 | |
| HF01 | 12 | 9,141 | 42 | 229.8 | |
| HF02 | 10 | 4,825 | 22 | 239.6 | |
| HF03 | 17 | 14,799 | 68 | 234.3 | |
| HF04 | 18 | 11,557 | 53 | 237.4 | |
| AF01 | 14 | 11,337 | 50 | 243.6 | |
| AF02 | 10 | 6,880 | 31 | 240.6 | |
| AF03 | 10 | 6,791 | 31 | 240.3 | |
| AF04 | 18 | 9,182 | 42 | 242.8 | |
| HTN01 | 24 | 11,855 | 53 | 247.1 | |
| HTN02 | 2 | 1,247 | 6 | 221.3 | ⚠ Short document (2-page editorial) |
| LIPID01 | 39 | 23,912 | 112 | 235.2 | |
| VHD01 | 20 | 13,607 | 63 | 239.6 | |
| DCM01 | 31 | 23,124 | 106 | 246.9 | |
| CMY02 | 16 | 10,359 | 47 | 244.4 | |
| PREV01 | 33 | 18,801 | 84 | 242.4 | |
| PREV02 | 20 | 12,081 | 55 | 240.9 | |
| **TOTAL** | **387** | **248,894** | **1,142** | **239.7** | |

---

## 6. Data Quality Issues

| Severity | Issue | Detail |
|---|---|---|
| ⚠ WARNING | HTN02 is a short document | 2 pages, 1,247 tokens, 6 chunks. This is a research news/editorial piece (eBioMedicine 2023). Content is valid and on-topic; the document is included but retrieval experiments should note its limited coverage of the HTN topic. |
| ⚠ MINOR | 2 chunks exceed 300 tokens | 0.2% of corpus — negligible. Caused by unusually long sentences in LIPID01. No action required. |
| ℹ INFO | 0% section coverage | PDF extraction does not produce Markdown headings. All section metadata fields are empty. This is expected for the existing PDF loader — section-aware chunking is not active for PDF sources. Downstream retrieval is unaffected. |
| ✓ PASS | No very short chunks | 0 chunks below 20 tokens |
| ✓ PASS | No empty chunks | 0 empty chunks |
| ✓ PASS | No duplicate chunks | 0 duplicate chunk bodies |
| ✓ PASS | No duplicate documents | 22 unique doc_ids, no duplicate SHA-256 hashes |
| ✓ PASS | All PDFs have valid magic bytes | %PDF header confirmed for all 22 |
| ✓ PASS | No extraction failures | All 22 documents loaded and chunked successfully |

---

## 7. Git Safety

| Item | Status |
|---|---|
| `data/raw/*.pdf` gitignored | ✓ Rule `data/raw/*.pdf` added to `.gitignore` |
| `data/source_manifest.csv` tracked | ✓ Not gitignored |
| `data/source_checksums.csv` tracked | ✓ Not gitignored |
| `data/corpus_acquisition_report.md` tracked | ✓ Not gitignored |
| PDFs committed to git | ✗ None committed |

---

## 8. Test Suite

**Previous:** 129/129 tests passing  
**Current:** 129/129 tests passing  
**No regressions introduced.**

---

## 9. Files Created / Modified

### Created
- `data/source_manifest.csv` — 22 rows, full provenance
- `data/source_checksums.csv` — 22 rows, SHA-256 + file sizes
- `data/corpus_acquisition_report.md` — this report
- `data/processed/` — 22 cleaned `.txt` files (one per document)
- `data/chunks/chunks.jsonl` — 1,142 chunks (replaced Stage-1 smoke-test chunks)

### Modified
- `.gitignore` — added rules for `data/raw/*.pdf`, `data/processed/`, `data/chunks/`

### Not modified
- All source code (`src/`, `scripts/`, `tests/`) — unchanged
- All chunking configuration — unchanged
- All retrieval stubs — unchanged

---

## 10. Known Limitations for Retrieval Stage

1. **HTN02 is thin** (6 chunks). HTN topic coverage relies heavily on HTN01 (53 chunks). Consider whether to supplement the hypertension topic at corpus v1.1 time.
2. **No section headings** in any chunk (all PDFs — section metadata is empty). The sentence-chunking strategy does not depend on headings, so retrieval is unaffected. Future work may explore PDF heading extraction with a specialised parser (e.g., PyMuPDF).
3. **Corpus is skewed toward larger documents**: CAD01 (107 chunks), DCM01 (106 chunks), LIPID01 (112 chunks) each contribute ~10% of all chunks. Retrieval evaluation should weight by query coverage, not raw chunk count.

---

## 11. Freeze Decision

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         CORPUS V1 — READY TO FREEZE                         ║
║                                                              ║
║  22/22 documents verified (CC BY 4.0)                       ║
║  22/22 documents successfully ingested                       ║
║  1,142 chunks produced                                       ║
║  0 extraction failures                                       ║
║  0 empty or duplicate chunks                                 ║
║  0 duplicate documents                                       ║
║  129/129 tests passing                                       ║
║                                                              ║
║  One minor warning: HTN02 is a 2-page editorial.            ║
║  This does NOT block the freeze.                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

The corpus is ready for the retrieval implementation stage (BM25, MinHash+LSH, Dense+FAISS, RRF, cross-encoder reranking).
