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
│   ├── raw/            # Original source documents (PDFs, HTML — not committed)
│   ├── processed/      # Cleaned plain-text documents
│   ├── chunks/         # Serialised text chunks (JSONL)
│   └── evaluation/     # Queries + ground-truth relevance judgements
│
├── indexes/
│   ├── bm25/           # Serialised BM25 index
│   ├── lsh/            # Serialised MinHash-LSH index
│   └── faiss/          # FAISS vector index + chunk-ID mapping
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py          # Raw text cleaning
│   │   └── chunker.py          # Chunking strategies + Chunk dataclass
│   ├── retrieval/
│   │   ├── bm25.py             # BM25Retriever
│   │   ├── lsh.py              # LSHRetriever (MinHash)
│   │   ├── dense.py            # DenseRetriever (FAISS)
│   │   └── hybrid.py           # HybridRetriever + reciprocal_rank_fusion()
│   ├── reranking/
│   │   └── reranker.py         # CrossEncoderReranker
│   ├── rag/
│   │   └── pipeline.py         # RAGPipeline (retrieval → reranking → generation)
│   └── evaluation/
│       ├── retrieval_metrics.py    # MRR, MAP, NDCG@k, Recall@k, Precision@k
│       └── diagnostic_metrics.py  # EM, F1, Faithfulness, Clinical accuracy
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
│   ├── build_corpus.py     # CLI: clean raw docs → chunks
│   ├── build_indexes.py    # CLI: build BM25 / LSH / FAISS indexes
│   └── run_evaluation.py   # CLI: run retrieval + diagnostic evaluation
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

Place raw documents (PDF or plain-text) in `data/raw/`.  
See `data/raw/README_data.md` (to be created) for corpus sourcing notes.

### 3. Build the corpus

```bash
python scripts/build_corpus.py \
    --raw-dir   data/raw \
    --proc-dir  data/processed \
    --chunk-dir data/chunks \
    --chunk-size 512 \
    --overlap    64
```

### 4. Build indexes

```bash
python scripts/build_indexes.py \
    --chunk-file data/chunks/chunks.jsonl \
    --build-bm25 --build-lsh --build-dense
```

### 5. Run evaluation

```bash
python scripts/run_evaluation.py \
    --query-file data/evaluation/queries.jsonl \
    --run-name   baseline_hybrid \
    --use-bm25 --use-lsh --use-dense \
    --use-reranker
```

> **Note:** Steps 3–5 are not yet implemented. See `NotImplementedError` stubs in `src/`.

---

## Development Stages

- [x] **Stage 0** — Repository scaffold (current)
- [ ] **Stage 1** — Data preprocessing (cleaner + chunker)
- [ ] **Stage 2** — BM25 retrieval
- [ ] **Stage 3** — MinHash-LSH retrieval
- [ ] **Stage 4** — Dense (FAISS) retrieval
- [ ] **Stage 5** — Hybrid RRF fusion
- [ ] **Stage 6** — Cross-encoder reranking
- [ ] **Stage 7** — RAG pipeline + LLM integration
- [ ] **Stage 8** — Retrieval evaluation
- [ ] **Stage 9** — Downstream diagnostic evaluation
- [ ] **Stage 10** — Ablation study + error analysis

---

## Citation

This prototype extends the retrieval methodology described in:

> *Cardiology-Chat* — [citation to be added]

---

## License

[To be determined]
