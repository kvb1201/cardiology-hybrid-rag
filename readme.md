# Hybrid Text Retrieval for Evidence-Grounded Cardiovascular Diagnostic Reasoning

A research prototype extending the retrieval component of the **Cardiology-Chat** paper.  
The system combines sparse, approximate, and dense retrieval methods in a hybrid pipeline,
with cross-encoder reranking and end-to-end RAG evaluation on cardiovascular diagnostic tasks.

---

## Overview

| Component | Method |
|---|---|
| Sparse retrieval | BM25 (rank-bm25) |
| Approximate retrieval | MinHash + LSH (datasketch) |
| Dense retrieval | Sentence-Transformers + FAISS |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder (sentence-transformers) |
| Generation | RAG pipeline (LLM to be configured) |
| Evaluation | Retrieval metrics + downstream diagnostic metrics |

---

## Repository Structure

```
cardiology-hybrid-rag/
├── data/
│   ├── raw/            # Original source documents (PDF/HTML/TXT/MD — not committed)
│   ├── processed/      # Cleaned plain-text files (one per document, named by doc_id)
│   ├── chunks/         # Serialised text chunks (chunks.jsonl)
│   └── evaluation/     # Queries + ground-truth relevance judgements
│
├── indexes/
│   ├── bm25/           # Serialised BM25 index
│   ├── lsh/            # Serialised MinHash-LSH index
│   └── faiss/          # FAISS vector index + chunk-ID mapping
│
├── src/
│   ├── ingestion/
│   │   ├── base.py             # RawDocument dataclass + BaseLoader ABC
│   │   ├── pdf_loader.py       # PDF extraction via pypdf
│   │   ├── html_loader.py      # HTML extraction via BeautifulSoup + lxml
│   │   ├── text_loader.py      # Plain-text / Markdown / RST loader
│   │   └── factory.py          # get_loader() + load_directory()
│   ├── preprocessing/
│   │   ├── cleaner.py          # 5-stage text cleaning pipeline
│   │   └── chunker.py          # Chunking strategies + Chunk dataclass + JSONL I/O
│   ├── retrieval/
│   │   ├── base.py             # BaseRetriever ABC + RetrievalResult dataclass  [implemented]
│   │   ├── bm25.py             # BM25Retriever (rank-bm25, k1=1.5, b=0.75)  [implemented]
│   │   ├── lsh.py              # LSHRetriever (MinHash)  [stub — Stage 2B]
│   │   ├── dense.py            # DenseRetriever (FAISS)  [stub — Stage 2C]
│   │   └── hybrid.py           # HybridRetriever + reciprocal_rank_fusion()  [stub — Stage 2D]
│   ├── reranking/
│   │   └── reranker.py         # CrossEncoderReranker  [stub]
│   ├── rag/
│   │   └── pipeline.py         # RAGPipeline  [stub]
│   └── evaluation/
│       ├── retrieval_metrics.py    # Precision@k, Recall@k, MRR, MAP, NDCG@k  [implemented]
│       └── diagnostic_metrics.py  # EM, F1, Faithfulness, Clinical accuracy  [stub]
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Corpus statistics & vocabulary analysis
│   ├── 02_chunking_analysis.ipynb      # Chunking strategy comparison
│   ├── 03_retrieval_comparison.ipynb   # BM25 vs LSH vs Dense vs Hybrid
│   ├── 04_hybrid_analysis.ipynb        # RRF tuning & reranker ablation
│   └── 05_results_visualization.ipynb  # Final paper figures
│
├── experiments/
│   ├── configs/        # YAML/JSON experiment configuration files
│   └── results/        # Per-run metric JSON files + exported figures
│
├── scripts/
│   ├── build_corpus.py     # CLI: ingest raw docs → clean → chunk → JSONL  [implemented]
│   ├── build_indexes.py    # CLI: build BM25 / LSH / FAISS indexes  [stub]
│   └── run_evaluation.py   # CLI: run retrieval + diagnostic evaluation  [stub]
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (sample files, text constants)
│   ├── test_cleaner.py             # 45 tests — cleaning pipeline
│   ├── test_chunker.py             # 46 tests — chunking strategies + JSONL I/O
│   ├── test_ingestion.py           # 38 tests — loaders + factory
│   ├── test_bm25.py                # 37 tests — tokenizer + BM25Retriever
│   └── test_retrieval_metrics.py   # 49 tests — Precision@k, Recall@k, MRR, MAP, NDCG@k
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd cardiology-hybrid-rag

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add your corpus

Place raw documents in `data/raw/`. Supported formats:

| Format | Extensions | Loader |
|--------|-----------|--------|
| PDF | `.pdf` | `pypdf` — extracts text per page |
| HTML | `.html`, `.htm` | `BeautifulSoup + lxml` — strips nav/footer/script, preserves headings |
| Plain text | `.txt`, `.md`, `.rst` | UTF-8 decode with fallback encoding detection |

> The sample documents in `data/raw/` (`sample_atrial_fibrillation.txt`,
> `sample_heart_failure.md`, `sample_hypertension.html`) are synthetic
> educational documents included only for testing. Replace them with your
> real cardiology corpus before running experiments.

### 3. Build the corpus

Run the full ingestion → cleaning → chunking pipeline:

```bash
python scripts/build_corpus.py \
    --raw-dir    data/raw \
    --proc-dir   data/processed \
    --chunk-dir  data/chunks \
    --chunk-size 256 \
    --overlap    32 \
    --strategy   sentence
```

#### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--raw-dir` | `data/raw` | Input directory of raw documents |
| `--proc-dir` | `data/processed` | Output directory for cleaned `.txt` files |
| `--chunk-dir` | `data/chunks` | Output directory for `chunks.jsonl` |
| `--output-file` | `chunks.jsonl` | Name of the JSONL output file |
| `--chunk-size` | `256` | Target chunk size in whitespace-delimited tokens |
| `--overlap` | `32` | Token overlap between consecutive chunks |
| `--strategy` | `sentence` | Chunking strategy: `sentence`, `paragraph`, or `sliding_window` |
| `--recursive` | off | Recurse into subdirectories of `--raw-dir` |

#### Chunking strategies

| Strategy | Description | Best for |
|----------|-------------|----------|
| `sentence` | Accumulates sentences up to `chunk_size` tokens; respects sentence boundaries | General use; good balance of context and granularity |
| `paragraph` | Splits on blank lines; Markdown headings start new chunks | Documents with clear section structure |
| `sliding_window` | Fixed token window with stride `chunk_size - overlap` | Quick baseline; fast but ignores linguistic structure |

#### Example output (sentence strategy, 128 tokens, 16 overlap)

```
Documents loaded      : 3
Documents skipped     : 0  (empty after cleaning)
Documents failed      : 0
Total chunks produced : 19
Avg tokens per chunk  : 93.6
Elapsed               : 0.27s
```

### 4. JSONL chunk format

Each line of `data/chunks/chunks.jsonl` is a flat JSON object:

```json
{
  "chunk_id":    "f9f3a4b74a20dbbb-0005",
  "text":        "### Anticoagulation\n\nDirect oral anticoagulants (DOACs) ...",
  "doc_id":      "f9f3a4b74a20dbbb",
  "start_char":  2949,
  "end_char":    3470,
  "title":       "Atrial Fibrillation: Diagnosis and Management",
  "source":      "/abs/path/to/data/raw/sample_atrial_fibrillation.txt",
  "section":     "Rhythm Control",
  "strategy":    "sentence",
  "chunk_index": 5,
  "total_chunks": 7
}
```

| Field | Description |
|-------|-------------|
| `chunk_id` | `"{doc_id}-{index:04d}"` — globally unique |
| `text` | Chunk content (cleaned, section headings preserved) |
| `doc_id` | SHA-256 (first 16 hex chars) of the source file's absolute path |
| `start_char` / `end_char` | Character offsets within the cleaned document text |
| `title` | Document title (from PDF metadata, HTML `<title>`, or Markdown H1) |
| `source` | Absolute path to the originating file |
| `section` | Nearest preceding Markdown heading (`""` if none) |
| `strategy` | Chunking strategy used |
| `chunk_index` | 0-based position within this document |
| `total_chunks` | Total chunk count for this document |

### 5. Build the BM25 index

Build the BM25 sparse retrieval index from the frozen corpus:

```bash
python - << 'EOF'
from src.preprocessing.chunker import load_chunks
from src.retrieval.bm25 import BM25Retriever

chunks = load_chunks("data/chunks/chunks.jsonl")
r = BM25Retriever(index_path="indexes/bm25", k1=1.5, b=0.75)
r.build(chunks)
r.save()
print(f"Index built: {r.corpus_size} chunks → indexes/bm25/bm25_index.pkl")
EOF
```

Or use the scripts CLI once `build_indexes.py` is fully implemented:

```bash
python scripts/build_indexes.py --chunk-file data/chunks/chunks.jsonl --build-bm25
```

### 6. Query the BM25 index

```python
from src.retrieval.bm25 import BM25Retriever

r = BM25Retriever(index_path="indexes/bm25")
r.load()

results = r.search("pharmacological treatment of heart failure", top_k=5)
for res in results:
    print(f"Rank {res.rank}  score={res.score:.3f}  {res.chunk_id}  {res.text[:80]}...")
```

Each result contains: `rank`, `chunk_id`, `doc_id`, `score`, `text`, `metadata`.

### 7. Build indexes *(LSH + Dense — not yet implemented)*

```bash
python scripts/build_indexes.py \
    --chunk-file data/chunks/chunks.jsonl \
    --build-bm25 --build-lsh --build-dense
```

### 8. Run evaluation *(not yet implemented — qrels required)*

```bash
python scripts/run_evaluation.py \
    --query-file data/evaluation/queries.jsonl \
    --run-name   baseline_hybrid \
    --use-bm25 --use-lsh --use-dense \
    --use-reranker
```

---

## BM25 Retrieval (Stage 2A)

### How BM25 works

BM25 (Best Match 25) is a probabilistic sparse retrieval model that ranks chunks
by term-frequency/inverse-document-frequency scores.  For a query token `t` and
chunk `d`:

```
score(t,d) = IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
```

where `k1` saturates term frequency and `b` normalises for chunk length.

### Tokenizer

Function `src.retrieval.bm25.tokenize(text)`:
- Lowercase the input
- Extract alphanumeric runs that contain at least one letter (preserves `SGLT-2`, `HFrEF`, `CHA2DS2-VASc`, `INR`, `mg/kg`)
- **No stemming, no lemmatisation, no stopword removal**
- Deterministic and side-effect-free

```python
from src.retrieval.bm25 import tokenize
tokenize("SGLT-2 inhibitor reduces HF hospitalisation by 25%.")
# → ['sglt-2', 'inhibitor', 'reduces', 'hf', 'hospitalisation', 'by']
```

### Baseline parameters

| Parameter | Value | Description |
|---|---|---|
| `k1` | 1.5 | Term-frequency saturation |
| `b` | 0.75 | Document-length normalisation |

These are the standard BM25Okapi defaults.  **Do not tune until a frozen evaluation set (qrels) exists.**  Parameters are configurable in `experiments/configs/retrieval.yaml`.

### Index facts (Corpus V1)

| Metric | Value |
|---|---|
| Indexed chunks | 1,142 |
| Build time | 0.124 s |
| Index file | `indexes/bm25/bm25_index.pkl` |
| Index size | 4.5 MB |
| Implementation | `rank-bm25==0.2.2` BM25Okapi |

### Smoke-test results (sanity check only — not scientific evaluation)

| Query | Rank 1 chunk | Doc |
|---|---|---|
| "pharmacological treatments for heart failure" | HF02-0000 | HF02 |
| "risk factors for atrial fibrillation" | AF04-0030 | AF04 |
| "hypertensive heart disease diagnosed" | HTN01-0021 | HTN01 |
| "mechanisms of myocardial infarction" | MI03-0000 | MI03 |
| "imaging modalities in cardiomyopathy" | CMY02-0003 | CMY02 |

All top-1 results are topically correct.  Scientific evaluation requires frozen relevance judgments (qrels) — see `data/evaluation/qrels_schema.md`.

### Retrieval metrics

`src/evaluation/retrieval_metrics.py` implements:

| Metric | Function |
|---|---|
| Precision@k | `precision_at_k(retrieved, relevant, k)` |
| Recall@k | `recall_at_k(retrieved, relevant, k)` |
| Reciprocal Rank | `reciprocal_rank(retrieved, relevant)` |
| Mean Reciprocal Rank | `mean_reciprocal_rank(all_retrieved, all_relevant)` |
| Average Precision | `average_precision(retrieved, relevant)` |
| Mean Average Precision | `mean_average_precision(all_retrieved, all_relevant)` |
| NDCG@k | `ndcg_at_k(retrieved, relevant, k, graded_relevance=None)` |

All functions accept generic Python types and support both binary and graded relevance (0/1/2).  They are retriever-agnostic and will evaluate BM25, LSH, Dense, Hybrid, and Reranker with identical code.

---

`src/preprocessing/cleaner.py` applies these stages in order:

1. **Ligature / Unicode normalisation** — decodes `ﬁ`→`fi`, `ﬀ`→`ff`, en-dash→`-`, `\u00a0`→space, Greek letter expansions, etc.
2. **PDF artefact removal** — strips form feeds (`\f`), DOI lines, copyright notices, "Downloaded from …", "Received / Accepted" publication metadata.
3. **Soft-hyphen repair** — rejoins words split across lines by PDF column layout (`cardio-\nvascular` → `cardiovascular`).
4. **Per-line cleaning** — removes standalone page numbers, single-character lines, symbol-only lines; normalises tabs and multiple spaces.
5. **Blank-line collapsing** — reduces 3+ consecutive newlines to at most 2.

Preserved intentionally: compound-term hyphens (`ST-elevation`), dose slashes (`mg/kg`), lab-value colons (`Na: 138 mEq/L`), abbreviation parentheses (`(AF)`), citation brackets (`[1,2]`), `±`, `%`, blood pressure notation, Markdown headings.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_cleaner.py -v
python -m pytest tests/test_chunker.py -v
python -m pytest tests/test_ingestion.py -v
```

Current status: **215 / 215 tests passing**.

---

## Development Stages

- [x] **Stage 0** — Repository scaffold
- [x] **Stage 1** — Ingestion + cleaning + chunking pipeline
- [x] **Stage 2A** — BM25 sparse retrieval baseline + retrieval metrics *(current)*
- [ ] **Stage 2B** — MinHash-LSH retrieval
- [ ] **Stage 2C** — Dense (FAISS) retrieval
- [ ] **Stage 2D** — Hybrid RRF fusion
- [ ] **Stage 3** — Cross-encoder reranking
- [ ] **Stage 4** — RAG pipeline + LLM integration
- [ ] **Stage 5** — Retrieval evaluation (requires frozen qrels)
- [ ] **Stage 6** — Downstream diagnostic evaluation
- [ ] **Stage 7** — Ablation study + error analysis
- [ ] **Stage 10** — Ablation study + error analysis

---

## Citation

This prototype extends the retrieval methodology described in:

> *Cardiology-Chat* — [citation to be added]

---

## License

[To be determined]
