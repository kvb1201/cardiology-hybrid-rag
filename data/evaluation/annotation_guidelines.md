# Retrieval Evaluation — Annotation Guidelines
**Project:** Hybrid Text Retrieval for Evidence-Grounded Cardiovascular Diagnostic Reasoning  
**Version:** 1.0 (Corpus V1 — Frozen)  
**Date:** 2026-10-01

---

## 1. Purpose

These guidelines define how relevance is assigned between a retrieval query and a corpus chunk. They govern the creation and future extension of `qrels.csv` and must be applied consistently across all annotators and annotation rounds.

The same qrels will be used to evaluate BM25, MinHash+LSH, Dense+FAISS, Hybrid RRF, and the Cross-Encoder reranker. Relevance judgments must therefore be **retriever-agnostic** — do not judge whether a chunk would be found by a particular retrieval method; judge only whether the chunk contains useful evidence for the query.

---

## 2. Relevance Scale

### Grade 2 — Highly Relevant

The chunk **directly and substantively** addresses the query.

A clinician trying to answer the query would find this chunk **essential** or **primary** reading.

**Criteria (must satisfy at least one):**
- The chunk directly defines, describes, or explains the main clinical concept in the query
- The chunk presents clinical evidence, trial data, or guideline recommendations that directly answer the query
- The chunk provides a clinical classification, diagnostic criterion, or treatment algorithm for the query topic
- The chunk discusses mechanisms that are the specific subject of the query

**Example:**  
Query: *"What pharmacological treatments are used for HFrEF?"*  
Chunk: *"The four pillars of GDMT for HFrEF are: ARNI (sacubitril/valsartan), beta-blocker, MRA (spironolactone/eplerenone), and SGLT2 inhibitor. Titration to target doses is recommended in all stable patients."*  
→ **Grade 2** — directly answers the query with all main drug classes.

### Grade 1 — Partially Relevant

The chunk **provides useful supporting context** but does not directly answer the query, or addresses only one aspect of a multi-part query.

**Criteria (must satisfy at least one):**
- The chunk mentions the query concept but its primary focus is a related (not identical) topic
- The chunk provides background or epidemiological context relevant to the query
- The chunk discusses a sub-component or consequence of the query topic
- The chunk contains a reference list that cites work directly relevant to the query
- The chunk describes a treatment comparison where one of the treatments is the subject of the query

**Example:**  
Query: *"What pharmacological treatments are used for HFrEF?"*  
Chunk: *"Patients with HFrEF have a significantly higher mortality rate than HFpEF. GDMT is under-prescribed in real-world practice."*  
→ **Grade 1** — relevant context but does not describe the treatments themselves.

### Grade 0 — Not Relevant

The chunk provides **no meaningful evidence** for answering the query.

**Criteria:**
- The chunk's primary topic is entirely unrelated to the query
- The chunk mentions a query term only in passing (e.g., in a comparison table row for a different disease)
- The chunk is a reference list entry with no clinical content
- The chunk discusses a different disease with no connection to the query

**Grade 0 chunks are NOT stored in qrels.csv** (only grades 1 and 2 are recorded).

---

## 3. Special Cases

### 3.1 Reference-list chunks

Chunks that consist primarily of bibliographic references (author names, journal names, DOIs, year) with minimal clinical text are generally **grade 0** unless the reference is specifically about the query topic and provides meaningful context through its title and abstract snippet.

Rule: if >60% of lines follow the pattern `[N] Author et al., Journal Year` and the chunk has fewer than 50 words of clinical content → **grade 0**.

### 3.2 Chunks with overlapping content

When two chunks from the same document cover similar content (e.g., consecutive chunks from a long treatment section), both may receive the same grade if both independently provide useful evidence. Do not artificially limit to one relevant chunk per document per query.

### 3.3 Chunks from different documents supporting the same answer

If multiple documents in the corpus cover the same clinical concept (e.g., both HF01 and HF02 discuss SGLT2 inhibitors), relevant chunks from all documents should be graded independently. Cross-document redundancy is acceptable and expected.

### 3.4 Highly technical or statistical chunks

Chunks presenting statistical tables, hazard ratios, or model performance metrics are **grade 2** if the statistics directly quantify the outcome relevant to the query, and **grade 1** if the statistics are for a related but different outcome.

### 3.5 Query-specific terminology

Annotators must match the **clinical intent** of the query, not just keyword matching. For example:
- Query: *"left ventricular hypertrophy"* → also grade chunks discussing *"LVH"*, *"ventricular remodelling"*, *"increased wall thickness"*
- Query: *"anticoagulation in AF"* → also grade chunks discussing *"stroke prevention in atrial fibrillation"*, *"DOAC"*, *"warfarin AF"*

### 3.6 Chunks that partially answer one sub-question

Some queries have multiple implicit sub-questions (e.g., *"diagnosis AND management"*). A chunk that only answers one sub-question (e.g., only diagnosis, not management) should receive **grade 1**, not grade 2.

### 3.7 Mechanism vs treatment queries

Queries explicitly asking for *mechanisms* should receive grade 2 for mechanistic explanations and grade 1 for treatment chunks (which imply the mechanism). The reverse also applies.

---

## 4. Pool Construction

The candidate pool for annotation was constructed using:
1. **BM25 top-20** results for each query
2. **Keyword-augmented scan**: additional chunks identified by keyword pattern matching on the query's key terms

This dual-source pool prevents evaluation bias toward BM25 (if only BM25 results were judged, BM25 would appear artificially better than other retrievers).

For future annotation rounds, the pool should also include:
- MinHash-LSH top-20 results
- Dense retrieval top-20 results
- Any cross-system union of top-20 from all retrievers

---

## 5. Annotation Consistency Rules

1. **Unit of judgment**: judge the individual chunk, not the document as a whole.
2. **Context independence**: assume the chunk is the only thing the retriever would show a user. Do not use knowledge of surrounding chunks to upgrade a grade.
3. **Clinical accuracy**: do not downgrade a chunk because it presents data you personally disagree with — grade based on topical relevance, not clinical accuracy.
4. **Annotator independence**: if two annotators are used, each must complete grading independently before discussion.
5. **Disagreement resolution**: for grade 2 vs grade 1 disagreements, use the lower grade (conservative). For grade 1 vs grade 0 disagreements, use the higher grade (inclusive).

---

## 6. Inter-Annotator Agreement

Target Cohen's κ ≥ 0.6 (moderate-to-substantial agreement) on a 20% random sample before finalising qrels. Report κ at both binary (relevant/not) and graded levels.

---

## 7. Corpus V1 Annotation Notes

The initial qrels (v1.0) were constructed using:
- BM25 top-20 per query + keyword augmentation
- Graded relevance assigned by automated rule-based patterns co-verified against actual chunk text
- All zero-coverage queries were manually supplemented by direct corpus inspection

Known limitations:
- HTN02 (2-page editorial) contributes few chunks to hypertension queries — HTN topic may be under-represented at evaluation time
- Some reference-heavy chunks from LIPID01 and HF03 may have been incorrectly included if reference text happened to contain target terms

Future annotation rounds should include human expert review of all grade-2 judgments.
