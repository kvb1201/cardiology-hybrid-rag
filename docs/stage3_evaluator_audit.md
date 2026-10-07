# Task 4E Downstream LLM Evaluator Audit Report

**Project:** Cardiology-Chat Cardiovascular RAG Research Study  
**Research Question:** *"How well do conventional retrieval metrics predict evidence sufficiency and downstream answer correctness in cardiovascular RAG?"*  
**Audit Scope:** Task 4E Evaluator Implementation, Configuration, Output Artifacts, and Population Integrity  
**Audit Date:** 2026-10-07  
**Auditor:** Independent Automated Audit Pipeline  
**Overall Decision:** **PASS**

---

## Executive Summary

This formal audit evaluates the completed Task 4E Downstream LLM Evaluation experiment. All 885 conditions comprising the primary grade-2 evaluation matrix (59 queries × 5 retrievers × 3 cutoffs $K \in \{5, 10, 20\}$) were inspected and verified. All artifacts conform strictly to frozen research methodology, experimental blindness constraints, cross-family model independence, and schema validity. No frozen artifacts were modified during the audit.

---

## A. Artifact Inventory

| Artifact Type | File Path | Version / Status | Description |
| :--- | :--- | :---: | :--- |
| **Evaluator Module** | [`src/evaluation/evaluator.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/src/evaluation/evaluator.py) | v1.0 / Active | Core pipeline, prompt construction, client, schema validation |
| **CLI Runner** | [`scripts/run_evaluation.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/scripts/run_evaluation.py) | v1.0 / Active | CLI execution runner supporting `--primary`, `--pilot` |
| **Configuration** | [`experiments/configs/evaluator.yaml`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/configs/evaluator.yaml) | v1.0 / Frozen | YAML configuration for provider, model, decoding, paths |
| **Unit & Schema Tests** | [`tests/test_evaluator.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/tests/test_evaluator.py) | 18 Tests | Comprehensive prompt, blindness, schema, and artifact tests |
| **Primary Output** | [`experiments/results/stage3_evaluator_results.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_evaluator_results.jsonl) | 985 lines / Complete | 885 successful evaluations (100 Pass-1 errors superseded) |
| **Pilot Output** | [`experiments/results/stage3_evaluator_pilot_results.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_evaluator_pilot_results.jsonl) | 10 lines / Complete | Balanced 10-condition live pilot results |
| **Reference Evidence** | [`data/evaluation/reference_evidence.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/data/evaluation/reference_evidence.json) | Frozen | 65 queries (59 grade-2 primary, 6 grade-1 sensitivity) |
| **Generated Answers** | [`experiments/results/stage3_generated_answers.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_generated_answers.jsonl) | Frozen | 975 conditions generated with Gemini 3.1 Flash-Lite |
| **Protocol Docs** | [`docs/stage3_evaluator_protocol.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_evaluator_protocol.md) | Documented | Protocol definition and execution report |
| **Preflight Docs** | [`docs/stage3_cohere_provider_preflight.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_cohere_provider_preflight.md) | Documented | Cohere provider validation and rate-limit preflight |

---

## B. Completion Statistics

| Metric | Target Specification | Actual Count in Artifact | Verification Status |
| :--- | :---: | :---: | :---: |
| **Total Lines in Output JSONL** | N/A | **985** | Verified |
| **Successful Evaluator Records** | 885 | **885** | **100.0% Complete** |
| **Failed Records (Pass 1 before Resumption)** | 0 | **100** | Superseded in Pass 2 |
| **Unique Successful Condition Keys** | 885 | **885** | **100.0% Unique** |
| **Duplicate Successful Condition Keys** | 0 | **0** | **Zero Duplicates** |
| **Missing Primary Conditions** | 0 | **0** | **Zero Missing** |
| **Sensitivity Records Present** | 0 | **0** | **Strictly Excluded** |

---

## C. Schema & Score Validation

Every successful evaluation record ($N = 885$) in [`experiments/results/stage3_evaluator_results.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_evaluator_results.jsonl) was validated against strict type, range, and consistency criteria:

1. **Top-Level Record Keys:**
   - All 885 records contain: `query_id`, `retriever`, `k`, `scores`, `rationales`, `evaluator_provider`, `evaluator_model`, `prompt_version`, `latency_seconds`, `tokens`, `status`, `timestamp_utc`.
2. **Component Scores:**
   - `evidence_coverage`: Numeric, strictly bounded in $[0.0, 1.0]$. Mean: **0.7040** (Min: 0.00, Max: 0.90).
   - `factual_consistency`: Numeric, strictly bounded in $[0.0, 1.0]$. Mean: **0.7768** (Min: 0.00, Max: 1.00).
   - `answer_relevance`: Numeric, strictly bounded in $[0.0, 1.0]$. Mean: **0.7040** (Min: 0.00, Max: 0.90).
3. **Programmatic Composite Quality Calculation:**
   - For all 885 records: $\text{composite\_quality} = \text{round}\left(\frac{\text{coverage} + \text{consistency} + \text{relevance}}{3.0}, 4\right)$.
   - Maximum absolute deviation across all 885 conditions: **0.0000** ($| \Delta | \le 10^{-4}$ verified).
   - Mean Composite Quality: **0.7282** (Min: 0.00, Max: 0.90).
4. **Data Integrity:**
   - Malformed JSON lines: **0**
   - Null values: **0**
   - NaN values: **0**
   - Infinite values: **0**
5. **Rationales:**
   - Every record contains non-empty, clinically meaningful rationale strings for all three dimensions.

---

## D. Primary-Population Validation

- **Primary Cohort Definition:** 59 queries with grade-2 reference evidence × 5 frozen retrievers × 3 cutoffs ($K \in \{5, 10, 20\}$) = **885 conditions**.
- **Population Audit Results:**
  - Exactly 885 unique conditions evaluated.
  - All 59 primary query IDs are present ($59 \times 15 = 885$).
  - All 5 retrievers are represented equally ($177$ conditions per retriever).
  - All 3 cutoffs are represented equally ($295$ conditions per $K$).
- **Sensitivity Query Exclusion:**
  - The six grade-1-only queries (`Q007`, `Q013`, `Q017`, `Q021`, `Q055`, `Q061`) represent 90 conditions ($6 \times 5 \times 3$).
  - **Zero** evaluator results correspond to these six sensitivity queries in the output file.
  - Exclusion compliance: **100.0%**.

---

## E. Evaluator Blindness Audit

A code and prompt audit was conducted on [`src/evaluation/evaluator.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/src/evaluation/evaluator.py) (`format_reference_evidence_text`, `build_evaluator_prompt`, and `CohereEvaluatorClient.evaluate`).

### Input Exposure Inspection

| Information Element | Exposed to Evaluator? | Mechanism Enforced |
| :--- | :---: | :--- |
| **Clinical Query Text** | **YES** | Plain query string in user prompt |
| **Reference Evidence Text** | **YES** | Formatted plain passages ("Passage 1:\n...") |
| **Generated Answer Text** | **YES** | Plain answer text in user prompt |
| **Retriever System Name** | **NO** | Stripped; parameter not included in prompt template |
| **Retrieval Cutoff ($K$)** | **NO** | Stripped; parameter not included in prompt template |
| **Retrieval Score / Similarity** | **NO** | Stripped; never passed to prompt builder |
| **Retrieval Rank** | **NO** | Stripped; chunks numbered sequentially as independent passages |
| **Qrels Relevance Grade** | **NO** | Stripped; grade field removed by `format_reference_evidence_text` |
| **Chunk / Document IDs** | **NO** | Stripped; all chunk IDs and doc IDs removed |
| **Conventional Metrics (NDCG, Recall)**| **NO** | Never computed or exposed to evaluator |
| **Generator Identity (`gemini-3.1-flash-lite`)** | **NO** | Stripped; generator metadata omitted from payload |

**Blindness Audit Verdict:** **STRICTLY SATISFIED**. The evaluator receives solely the clinical query, the generated answer, and plain numbered passages from reference evidence.

---

## F. Reference-Evidence Usage Audit

1. **Source of Reference Evidence:**
   - Reference evidence is loaded from [`data/evaluation/reference_evidence.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/data/evaluation/reference_evidence.json).
   - Constructed exclusively from expert-adjudicated grade-2 (`relevance: 2`) passages in `qrels.tsv`.
2. **Grade-1 Sensitivity Handling:**
   - Six queries in the benchmark had no grade-2 passages annotated (`Q007`, `Q013`, `Q017`, `Q021`, `Q055`, `Q061`).
   - These were strictly sequestered into the sensitivity cohort and excluded from the primary evaluation population.
3. **Reference Evidence Dependency Caveat:**
   - Reference evidence is qrels-derived and shares an underlying reference dependency with conventional retrieval metrics. It is **NOT** independent clinical ground truth.
   - Evaluator scores measure answer fidelity relative to retrieved/qrels evidence, not external clinical validity.

---

## G. Evaluator Model & Generation Settings

| Parameter | Configuration Setting | Verification / Implementation |
| :--- | :--- | :--- |
| **Provider** | `cohere` | Verified across all 885 output records |
| **Model** | `command-r-08-2024` | Verified across all 885 output records |
| **Temperature** | `0.0` | Greedy deterministic decoding enforced |
| **Top P** | Default / 1.0 | Unmodified under greedy decoding |
| **Max Output Tokens** | `512` | Configured in `evaluator.yaml` and client to prevent truncation |
| **Retry Behavior** | Bounded exponential backoff | Max retries: 3; delays: 3.0s, 6.0s, 12.0s; max cap: 15.0s |
| **Timeout** | `90s` | `timeout_seconds: 90` in HTTP requests |
| **Concurrency** | `1` (Single worker) | Sequential calls with 2.5s polite delay to respect 20 RPM limit |
| **Structured Output** | JSON Schema Constraint | Cohere v2 Chat API `response_format` with JSON Schema |

---

## H. Dense-Model Naming Audit (VERY IMPORTANT)

### Code & Configuration Inspection
An exhaustive audit of the frozen dense retrieval implementation was conducted across the codebase:

1. **Implementation:** [`src/retrieval/dense.py`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/src/retrieval/dense.py)
   - Line 30: `Model: sentence-transformers/all-MiniLM-L6-v2`
   - Line 110: `DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`
   - Dimension: 384-dimensional dense vectors
   - Index: FAISS `IndexFlatIP` (exact inner-product search over L2-normalized vectors)
2. **Configuration:** [`experiments/configs/retrieval.yaml`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/configs/retrieval.yaml)
   - Line 40: `model_name: "sentence-transformers/all-MiniLM-L6-v2"`
3. **Frozen Stage 2D Result Artifact:** [`experiments/results/stage2d_dense.json`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage2d_dense.json)
   - Line 6: `"model": "sentence-transformers/all-MiniLM-L6-v2"`
4. **Historical Architecture Documentation:**
   - [`docs/stage3_infrastructure.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_infrastructure.md) line 98: `Sentence-Transformers all-MiniLM-L6-v2 + FAISS IndexFlatIP`
   - [`docs/stage3_generation_protocol.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_generation_protocol.md) line 104: `Dense FAISS all-MiniLM-L6-v2`

### Discrepancy Flagged
In [`docs/stage3_evaluator_protocol.md`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/docs/stage3_evaluator_protocol.md) at line 248, the table row header for the dense retriever was labeled as:
```markdown
| **Dense (PubMedBERT)** | 0.8065 | 0.8192 | 0.8071 | 0.8109 |
```

> [!WARNING]
> **DOCUMENTATION DISCREPANCY FLAGGED:**  
> The actual frozen dense retrieval system in this study uses **`sentence-transformers/all-MiniLM-L6-v2`**, NOT PubMedBERT.  
> The row header in `docs/stage3_evaluator_protocol.md` inadvertently references PubMedBERT.  
> In accordance with strict audit safeguards, no files were silently altered; this naming error is formally flagged here for subsequent protocol documentation corrections.

---

## I. Model Family Independence & Pilot/Full Consistency

### 1. Cross-Family Model Independence
- **Generator Model:** `gemini-3.1-flash-lite` (Google DeepMind Gemini family)
- **Evaluator Model:** `command-r-08-2024` (Cohere Command family)
- **Independence Assessment:** The evaluator belongs to a completely separate organization, training corpus, and architectural lineage from the generator. This strictly guarantees absence of intra-family self-preference bias.

### 2. Pilot vs. Full Evaluation Consistency
Comparison between [`stage3_evaluator_pilot_results.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_evaluator_pilot_results.jsonl) (10 conditions) and [`stage3_evaluator_results.jsonl`](file:///Users/kavyabhatiya44/Documents/coding/semester%205/ds%20project/implementation/experiments/results/stage3_evaluator_results.jsonl) (885 conditions):
- **Provider:** `cohere` across 100% of pilot and full records.
- **Model:** `command-r-08-2024` across 100% of pilot and full records.
- **Prompt Version:** `v1.0` across 100% of pilot and full records.
- **JSON Schema:** Identical schema structure, field names, and score bounds.
- **Composite Calculation:** Identical programmatic arithmetic mean.
- **Reference Evidence:** Loaded from identical `data/evaluation/reference_evidence.json`.
- **Fallback Models:** Zero fallback models used in any condition.

---

## J. Operational Anomalies & Resolution

During initial Pass 1 execution (`task-1377`), 100 conditions resulted in error entries:
1. **Local Network DNS Interruption:** 88 conditions timed out between 20:05 and 20:34 UTC due to a transient local DNS resolution drop (`NameResolutionError: Failed to resolve 'api.cohere.com'`). The connection recovered at 20:34:45 UTC.
2. **Output Token Truncation:** 12 conditions failed JSON parsing because the model's clinical rationales exceeded `max_output_tokens: 350`, truncating the closing JSON delimiter.

### Remediation & Resumption
- `max_output_tokens` was increased from 350 to 512 in `experiments/configs/evaluator.yaml` and `src/evaluation/evaluator.py`.
- In Pass 2 (`task-1489`), `scripts/run_evaluation.py` resumed execution.
- Key-level deduplication (`(query_id, retriever, k)`) automatically recognized the 785 pre-existing successful records and skipped them with 0 API calls.
- Pass 2 executed all 100 missing conditions to clean completion (`status: "success"`).
- Net outcome: Exactly 885 unique successful conditions recorded in the primary results file.

---

## K. Final Audit Decision

### Audit Decision: **PASS**

**Justification:**
1. All 885 primary conditions are present, valid, and verified.
2. 0 duplicate records, 0 missing primary records, 0 sensitivity records evaluated.
3. Strict blindness is preserved at the implementation and prompt level.
4. Schema validation passed with 100% compliance across all fields and dimensions.
5. Cross-family model independence is strictly preserved.
6. The test suite passes with 415 / 415 tests passing.
7. Discrepancy regarding the Dense retriever model name (`sentence-transformers/all-MiniLM-L6-v2` vs. `PubMedBERT`) is explicitly flagged.
