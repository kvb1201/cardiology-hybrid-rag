# Stage 3 — Downstream RAG Generation Protocol

## 1. Experimental Design & Purpose

This protocol specifies the fixed downstream generation setup for evaluating:

> **"Do conventional retrieval metrics reliably predict automated downstream answer quality in cardiovascular RAG?"**

### Experimental Variable:
The experiment varies **ONLY retrieval**. Across all five retrieval configurations, the generation pipeline is strictly held constant:
- Identical generator model
- Identical system prompt
- Identical user prompt template
- Identical decoding parameters ($\text{temperature}=0.0$, $\text{max\_tokens}=512$)
- Identical evidence context formatting
- Identical retry and timeout policies

The **only** distinction between generation inputs for the same query and cutoff $K$ is the set of retrieved evidence passages provided.

---

## 2. Configuration & Centralization

All generation parameters are centrally defined in [`experiments/configs/generation.yaml`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/configs/generation.yaml).

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Provider** | Google Gemini (`gemini`) | Explicitly selected in Task 4B. |
| **Model** | Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) | Deliberately selected and frozen after Gemini 3.8 Flash encountered the project-level Free Tier quota limit (20 req/day). |
| **Temperature** | `0.0` | Eliminates decoding stochasticity to ensure deterministic comparability. |
| **Max Output Tokens** | `512` | Sufficient for concise, evidence-grounded clinical responses. |
| **Top P** | `1.0` | Standard greedy decoding under temperature 0. |
| **System Prompt Version** | `v1.0` | Frozen versioned clinical QA system prompt. |
| **Generation Prompt Version**| `v1.0` | Frozen versioned query + context prompt template. |
| **Concurrency** | `5` | Controlled rate limiting to prevent provider throttling. |
| **Timeout** | `30s` | Per-call HTTP timeout. |
| **Max Retries** | `3` | Exponential backoff (1.0s, 2.0s, 4.0s; cap at 10s). |

---

## 3. Frozen Versioned Prompts

### System Prompt (`v1.0`):
```text
You are a clinical information question-answering assistant specializing in cardiovascular medicine.
Your task is to provide an accurate, concise, and evidence-grounded answer to the user's clinical question based strictly on the provided evidence passages.

Guidelines:
1. Base your answer solely on the provided evidence passages.
2. Do not fabricate, extrapolate, or invent information not directly supported by the evidence.
3. If the evidence passages do not contain sufficient information to answer the question completely, clearly state the limitations and uncertainty based on the available evidence.
4. Do not cite external sources or mention retrieval systems, search ranks, scores, or evaluation processes.
5. Maintain an objective, professional, and clear clinical communication tone.
```

### User Prompt Template (`v1.0`):
```text
Evidence Passages:
{context}

Question: {query}

Please provide a concise, evidence-grounded answer:
```

### Neutrality Controls Enforced:
- The generator **NEVER** receives:
  - Retriever identity (BM25, Dense, LSH, BD-RRF, BDL-RRF)
  - Cutoff $K$ value
  - Retrieval scores or rank positions
  - Qrels or relevance grades (Grade 0, 1, or 2)
  - Reference evidence passages
  - Downstream evaluation rubrics or metrics

---

## 4. Context Formatting

Passages retrieved by each system are formatted into a uniform block:

```text
[Evidence 1]
Source: <Document Title>
<Chunk Text>

[Evidence 2]
Source: <Document Title>
<Chunk Text>
```

- **Ordering:** Strictly follows retrieval ranking from rank 1 to $K$.
- **Empty Context (e.g. LSH zero-candidate queries):**
  Formatted as `"[No evidence passages retrieved for this query.]"`.
- **Integrity:** Chunk IDs, numerical scores, and system names are excluded.

---

## 5. Experiment Matrix (975 Conditions)

The generation matrix encompasses all permutations of:
- **Queries:** 65 clinical queries ($Q_{001} \dots Q_{065}$)
- **Retrievers:**
  1. `bm25`: BM25 baseline ($k_1=1.5, b=0.75$)
  2. `dense`: Dense FAISS `all-MiniLM-L6-v2`
  3. `lsh`: MinHash-LSH ($k=3$ word shingles, 128 permutations, threshold 0.05)
  4. `bm25_dense_rrf`: System A RRF ($RRF_k=60$)
  5. `bm25_dense_lsh_rrf`: System B RRF ($RRF_k=60$)
- **Cutoffs ($K$):** $K \in \{5, 10, 20\}$
- **Total Conditions:** $65 \times 5 \times 3 = \mathbf{975\text{ conditions}}$

---

## 6. Output Schema & Resumability

Results are appended line-by-line to [`experiments/results/stage3_generated_answers.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_generated_answers.jsonl).

### Success Record Schema:
```json
{
  "query_id": "Q001",
  "query_text": "What pharmacological treatments are recommended after myocardial infarction?",
  "topic": "myocardial_infarction",
  "query_type": "treatment",
  "retriever": "bm25",
  "k": 5,
  "retrieved_chunk_ids": ["CAD01-0001", "CAD01-0098", "CAD03-0012", "CAD03-0020", "CAD03-0021"],
  "generated_answer": "...",
  "model": "...",
  "temperature": 0.0,
  "prompt_version": "v1.0",
  "generation_timestamp": "2026-10-05T10:45:00Z",
  "status": "success"
}
```

### Failure Record Schema:
```json
{
  "query_id": "Q001",
  "query_text": "What pharmacological treatments are recommended after myocardial infarction?",
  "topic": "myocardial_infarction",
  "query_type": "treatment",
  "retriever": "bm25",
  "k": 5,
  "retrieved_chunk_ids": ["CAD01-0001", "CAD01-0098", "CAD03-0012", "CAD03-0020", "CAD03-0021"],
  "model": "...",
  "temperature": 0.0,
  "prompt_version": "v1.0",
  "generation_timestamp": "2026-10-05T10:45:00Z",
  "status": "failed",
  "error_type": "RateLimitError",
  "error_message": "Rate limit exceeded. Try again later."
}
```

### Resumability Key:
Each run is indexed by the deterministic triplet `(query_id, retriever, k)`. Before invoking the LLM, the generator scans the output JSONL file; if a record with `status: "success"` exists for that key, it is skipped automatically.

---

## 7. Execution CLI Commands

- **Dry Run (Validates 975 conditions without API calls):**
  ```bash
  python scripts/run_generation.py --dry-run
  ```

- **Pilot Run (3 conditions):**
  ```bash
  python scripts/run_generation.py --limit 3
  ```

- **Full Run (975 conditions once provider is configured):**
  ```bash
  python scripts/run_generation.py
  ```
