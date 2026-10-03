"""
dense.py
--------
Dense vector retrieval using sentence-transformer embeddings + FAISS exact search.

Architecture
------------
                      chunks.jsonl
                           │
                    load_chunks()
                           │
               ┌───────────▼───────────┐
               │  SentenceTransformer  │  all-MiniLM-L6-v2
               │  encode(batch=64)     │  → 384-dim L2-normalised vectors
               └───────────┬───────────┘
                           │  numpy array (N × 384)
               ┌───────────▼───────────┐
               │  faiss.IndexFlatIP    │  exact inner-product search
               │  (= cosine on unit    │  vectors)
               └───────────┬───────────┘
                           │  .save()  ──► indexes/faiss/
                           │  .load()  ◄── indexes/faiss/
                           │
               ┌───────────▼───────────┐
               │       search()        │  → List[RetrievalResult]
               └───────────────────────┘

Model
-----
Model:       sentence-transformers/all-MiniLM-L6-v2
Version:     as published on HuggingFace Hub (sentence-transformers 3.4.1)
Dimension:   384
Normalised:  Yes — L2-normalised (unit vectors) via
             SentenceTransformer.encode(normalize_embeddings=True)
Metric:      Inner product (equivalent to cosine similarity on unit vectors)
             Score range: [−1, 1]; higher is more similar.

Choice rationale
----------------
all-MiniLM-L6-v2 is the de-facto standard small English sentence-encoder.
It achieves competitive semantic similarity performance at a small footprint
(90 MB), runs on CPU in under 2 minutes for 1,142 chunks, and is widely
used as a retrieval baseline, making results directly comparable to published
work.  No fine-tuning or domain adaptation is applied — this is a zero-shot
baseline, which is the correct starting point for the ablation study.

FAISS index type
----------------
IndexFlatIP — exact inner-product search over L2-normalised vectors.
Equivalent to exact cosine-similarity search.
Chosen because:
  * Corpus size (1,142) is small enough that exact search is instantaneous
    (~0.2 ms/query).
  * Exact search eliminates HNSW/IVF approximation error, making the
    comparison with BM25 and LSH cleaner.
  * Reproducibility: IndexFlatIP produces identical results for identical
    inputs with no random state.

Index persistence
-----------------
Three files are saved under indexes/faiss/:
  dense_index.faiss  — the FAISS IndexFlatIP binary
  dense_meta.pkl     — chunk_ids, chunk_texts, chunk_meta lists
  dense_config.json  — model name, dimension, normalisation flag

Rebuild command::

    python scripts/build_indexes.py --build-dense

or programmatically::

    from src.retrieval.dense import DenseRetriever
    from src.preprocessing.chunker import load_chunks

    chunks = load_chunks("data/chunks/chunks.jsonl")
    r = DenseRetriever()
    r.build(chunks)
    r.save()
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.retrieval.base import BaseRetriever, RetrievalResult

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required: pip install faiss-cpu==1.15.1"
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required: pip install sentence-transformers==3.4.1"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384
NORMALIZE          = True        # L2-normalise → inner product = cosine


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class DenseRetriever(BaseRetriever):
    """Dense semantic retriever: sentence-transformer embeddings + FAISS IndexFlatIP.

    Parameters
    ----------
    index_path:
        Directory for FAISS index and metadata files.
    model_name:
        HuggingFace sentence-transformer model identifier.
        Default: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim).
    batch_size:
        Encoding batch size.  64 is fast on CPU without exceeding memory.
    chunks_path:
        Path to the frozen corpus JSONL used by :meth:`build`.

    Notes
    -----
    The model is loaded **lazily** — only when :meth:`encode`,
    :meth:`build`, or :meth:`search` is first called.  This avoids the
    ~3-second model-load cost during unit-test collection.
    """

    FAISS_FILENAME  = "dense_index.faiss"
    META_FILENAME   = "dense_meta.pkl"
    CONFIG_FILENAME = "dense_config.json"

    DEFAULT_INDEX_PATH  = Path("indexes/faiss")
    DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    ) -> None:
        self.index_path   = Path(index_path)
        self.model_name   = model_name
        self.batch_size   = batch_size
        self.chunks_path  = Path(chunks_path)

        self._model: Optional[SentenceTransformer] = None   # lazy load
        self._index: Optional[faiss.Index]          = None
        self._chunk_ids:   List[str]  = []
        self._chunk_texts: List[str]  = []
        self._chunk_meta:  List[dict] = []
        self._dim: int = EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Model (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence-transformer model if not already loaded."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* into L2-normalised dense vectors.

        Parameters
        ----------
        texts:
            List of strings to embed.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(texts), embedding_dim)``, dtype ``float32``,
            L2-normalised (unit vectors).

        Raises
        ------
        ValueError
            If *texts* is empty.
        """
        if not texts:
            raise ValueError("encode() requires at least one text string.")
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks=None) -> None:
        """Encode all chunks and build the FAISS IndexFlatIP index.

        Parameters
        ----------
        chunks:
            List of ``Chunk`` objects.  If *None*, loads from
            ``self.chunks_path``.

        Notes
        -----
        Progress is printed during encoding (1,142 chunks ≈ 60–90 s on CPU).
        Call :meth:`save` afterwards to persist the index.
        """
        if chunks is None:
            from src.preprocessing.chunker import load_chunks
            chunks = load_chunks(self.chunks_path)

        if not chunks:
            raise ValueError("Cannot build Dense index from an empty chunk list.")

        texts = [c.text for c in chunks]
        ids   = [c.chunk_id for c in chunks]
        meta  = [
            {
                "doc_id":       c.doc_id,
                "title":        c.metadata.get("title", ""),
                "source":       c.metadata.get("source", ""),
                "section":      c.metadata.get("section", ""),
                "chunk_index":  c.metadata.get("chunk_index", 0),
                "total_chunks": c.metadata.get("total_chunks", 0),
                "strategy":     c.metadata.get("strategy", ""),
            }
            for c in chunks
        ]

        print(f"  Encoding {len(texts)} chunks with {self.model_name} ...")
        embeddings = self.encode(texts)   # (N, 384) float32

        self._dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(self._dim)
        index.add(embeddings)

        self._index       = index
        self._chunk_ids   = ids
        self._chunk_texts = texts
        self._chunk_meta  = meta

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the FAISS index and metadata to ``<index_path>/``.

        Files written:
          * ``dense_index.faiss``  — binary FAISS index
          * ``dense_meta.pkl``     — chunk_ids, texts, meta
          * ``dense_config.json``  — model, dim, normalisation

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called yet.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")

        self.index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_path / self.FAISS_FILENAME))

        with (self.index_path / self.META_FILENAME).open("wb") as fh:
            pickle.dump(
                {
                    "chunk_ids":   self._chunk_ids,
                    "chunk_texts": self._chunk_texts,
                    "chunk_meta":  self._chunk_meta,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        config = {
            "model_name":    self.model_name,
            "dim":           self._dim,
            "normalize":     NORMALIZE,
            "metric":        "inner_product (cosine on unit vectors)",
            "faiss_index":   "IndexFlatIP",
            "corpus_size":   len(self._chunk_ids),
        }
        with (self.index_path / self.CONFIG_FILENAME).open("w") as fh:
            json.dump(config, fh, indent=2)

    def load(self) -> None:
        """Load a previously saved index from ``<index_path>/``.

        Raises
        ------
        FileNotFoundError
            If the index files do not exist.
        """
        faiss_path = self.index_path / self.FAISS_FILENAME
        meta_path  = self.index_path / self.META_FILENAME

        if not faiss_path.exists():
            raise FileNotFoundError(
                f"Dense index not found at '{faiss_path}'. "
                "Run build() + save() or: python scripts/build_indexes.py --build-dense"
            )

        self._index = faiss.read_index(str(faiss_path))

        with meta_path.open("rb") as fh:
            payload = pickle.load(fh)

        self._chunk_ids   = payload["chunk_ids"]
        self._chunk_texts = payload["chunk_texts"]
        self._chunk_meta  = payload["chunk_meta"]
        self._dim         = self._index.d

        # Load config (best-effort)
        cfg_path = self.index_path / self.CONFIG_FILENAME
        if cfg_path.exists():
            with cfg_path.open() as fh:
                cfg = json.load(fh)
            self.model_name = cfg.get("model_name", self.model_name)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k chunks most semantically similar to *query*.

        Parameters
        ----------
        query:
            Free-text query string.
        top_k:
            Number of results.  Clamped to corpus size automatically.

        Returns
        -------
        List[RetrievalResult]
            Sorted descending by inner-product (cosine) score.
            Empty list if *query* is blank.

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded.
        """
        if self._index is None:
            raise RuntimeError(
                "Dense index not available. Call build() or load() first."
            )

        query = query.strip()
        if not query:
            return []

        top_k = min(top_k, len(self._chunk_ids))
        if top_k == 0:
            return []

        query_vec = self.encode([query])          # (1, dim) float32
        scores_arr, idxs_arr = self._index.search(query_vec, top_k)

        scores = scores_arr[0]
        idxs   = idxs_arr[0]

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(idxs, scores), start=1):
            if idx < 0:          # FAISS sentinel for no result
                continue
            meta = dict(self._chunk_meta[idx])
            results.append(
                RetrievalResult(
                    chunk_id=self._chunk_ids[idx],
                    doc_id=meta.pop("doc_id", ""),
                    score=float(score),
                    rank=rank,
                    text=self._chunk_texts[idx],
                    metadata=meta,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Legacy alias
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Alias for search() returning (chunk_id, score) tuples.

        Provided for backward compatibility with the original stub signature.
        Prefer :meth:`search` for new code.
        """
        return [(r.chunk_id, r.score) for r in self.search(query, top_k=top_k)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """True if the FAISS index is ready for querying."""
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the index (0 if not yet built)."""
        return len(self._chunk_ids)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (0 if not yet built)."""
        return self._dim if self._index is not None else 0
