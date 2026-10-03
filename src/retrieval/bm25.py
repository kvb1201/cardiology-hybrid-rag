"""
bm25.py
-------
BM25 sparse retriever using BM25Okapi from the rank-bm25 library.

Architecture
------------
                      chunks.jsonl
                           │
                    load_chunks()
                           │
                    ┌──────▼──────┐
                    │  tokenize   │  (lowercase, alphanumeric filter)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  BM25Okapi  │  (rank-bm25, k1=1.5, b=0.75)
                    └──────┬──────┘
                           │  .save()  ──►  indexes/bm25/bm25_index.pkl
                           │  .load()  ◄──  indexes/bm25/bm25_index.pkl
                           │
                    ┌──────▼──────┐
                    │   search()  │  → List[RetrievalResult]
                    └─────────────┘

Tokenization
------------
Function: ``tokenize(text)``
- Lowercase the text.
- Extract runs of alphanumeric characters and hyphens that contain at
  least one letter (preserves drug names like "SGLT-2", scores like
  "CHA2DS2-VASc", lab values like "INR", abbreviations like "LVEF").
- No stemming, no lemmatisation, no stopword removal.
- Deterministic and side-effect-free (pure function).

BM25 Parameters (baseline — do not tune)
-----------------------------------------
k1 = 1.5   (term-frequency saturation)
b  = 0.75  (document-length normalisation)

These are configurable via the constructor and via
``experiments/configs/retrieval.yaml``.

Index persistence
-----------------
The index is serialised as a single pickle file:
    indexes/bm25/bm25_index.pkl

It stores the BM25Okapi object plus the ordered list of chunk IDs so
that rank → chunk_id lookups are O(1).  The raw chunk text and metadata
are NOT duplicated inside the index; they are re-loaded from the frozen
corpus on demand.

Rebuild command::

    python scripts/build_indexes.py --build-bm25

or programmatically::

    from src.retrieval.bm25 import BM25Retriever
    from src.preprocessing.chunker import load_chunks

    chunks = load_chunks("data/chunks/chunks.jsonl")
    r = BM25Retriever()
    r.build(chunks)
    r.save()
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import List, Optional

from src.retrieval.base import BaseRetriever, RetrievalResult

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rank-bm25 is required: pip install rank-bm25==0.2.2"
    ) from exc


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Matches runs of alphanumeric characters and hyphens that contain at
# least one ASCII letter.  This keeps:
#   "SGLT-2", "CHA2DS2-VASc", "beta-blocker", "HFrEF", "LVEF", "INR"
# but drops pure-number tokens like "2023", "10" and pure-punctuation.
_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?")
_HAS_LETTER = re.compile(r"[A-Za-z]")


def tokenize(text: str) -> List[str]:
    """Tokenize *text* for BM25 indexing and querying.

    Rules
    -----
    1. Lowercase the entire string.
    2. Extract every run of alphanumeric characters and hyphens that
       contains at least one letter.  This preserves clinical
       abbreviations (``LVEF``, ``INR``), hyphenated drug names
       (``beta-blocker``, ``SGLT-2``), and score names
       (``CHA2DS2-VASc``) while discarding bare numbers and punctuation.
    3. Single-character tokens are kept (e.g. the letter "a" as a
       stopword is intentionally preserved — stopword removal is out of
       scope for this stage).

    Parameters
    ----------
    text:
        Raw string — may be a chunk body or a query.

    Returns
    -------
    List[str]
        Ordered list of lowercase tokens.  Empty list if *text* is blank.

    Examples
    --------
    >>> tokenize("SGLT-2 inhibitor reduces HF hospitalisation by 25%.")
    ['sglt-2', 'inhibitor', 'reduces', 'hf', 'hospitalisation', 'by']
    >>> tokenize("")
    []
    """
    if not text:
        return []
    lowered = text.lower()
    return [
        tok
        for tok in _TOKEN_RE.findall(lowered)
        if _HAS_LETTER.search(tok)
    ]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever over the frozen Corpus V1 chunks.

    Parameters
    ----------
    index_path:
        Directory where the serialised index is stored.
        Defaults to ``indexes/bm25`` relative to the working directory.
    k1:
        BM25 term-frequency saturation parameter.
        Baseline value: ``1.5``.  Do not tune without a frozen eval set.
    b:
        BM25 document-length normalisation parameter.
        Baseline value: ``0.75``.
    chunks_path:
        Path to the frozen corpus JSONL file used by :meth:`build`.

    Attributes
    ----------
    _index:
        The underlying ``BM25Okapi`` object after :meth:`build` or
        :meth:`load`.
    _chunk_ids:
        Ordered list of chunk IDs parallel to the BM25 corpus rows.
    _chunk_texts:
        Ordered list of chunk texts for result construction.
    _chunk_meta:
        Ordered list of chunk metadata dicts.
    """

    DEFAULT_INDEX_PATH = Path("indexes/bm25")
    DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
    INDEX_FILENAME = "bm25_index.pkl"

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        k1: float = 1.5,
        b: float = 0.75,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    ) -> None:
        self.index_path   = Path(index_path)
        self.k1           = k1
        self.b            = b
        self.chunks_path  = Path(chunks_path)

        self._index: Optional[BM25Okapi]  = None
        self._chunk_ids:   List[str]  = []
        self._chunk_texts: List[str]  = []
        self._chunk_meta:  List[dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks=None) -> None:
        """Build the BM25 index from a list of Chunk objects.

        Parameters
        ----------
        chunks:
            List of ``Chunk`` objects (from
            ``src.preprocessing.chunker.load_chunks``).
            If *None* the method loads from ``self.chunks_path``.

        Notes
        -----
        Calling ``build()`` replaces any previously built in-memory index.
        The index is NOT automatically saved — call :meth:`save` afterwards.
        """
        if chunks is None:
            from src.preprocessing.chunker import load_chunks  # lazy import
            chunks = load_chunks(self.chunks_path)

        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunk list.")

        self._chunk_ids   = [c.chunk_id for c in chunks]
        self._chunk_texts = [c.text     for c in chunks]
        self._chunk_meta  = [
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

        tokenised_corpus = [tokenize(t) for t in self._chunk_texts]
        self._index = BM25Okapi(tokenised_corpus, k1=self.k1, b=self.b)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Serialise the index to ``<index_path>/bm25_index.pkl``.

        The pickle contains a dict with four keys:
        ``index``, ``chunk_ids``, ``chunk_texts``, ``chunk_meta``.
        The raw chunk text is stored so that :meth:`search` can return
        full result objects without re-reading the JSONL file at query
        time.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called yet.
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        self.index_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "index":       self._index,
            "chunk_ids":   self._chunk_ids,
            "chunk_texts": self._chunk_texts,
            "chunk_meta":  self._chunk_meta,
            "k1":          self.k1,
            "b":           self.b,
        }
        out = self.index_path / self.INDEX_FILENAME
        with out.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """Load a previously saved index from ``<index_path>/bm25_index.pkl``.

        Raises
        ------
        FileNotFoundError
            If the index file does not exist.
        """
        pkl = self.index_path / self.INDEX_FILENAME
        if not pkl.exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{pkl}'. "
                "Run build() + save() or: python scripts/build_indexes.py --build-bm25"
            )
        with pkl.open("rb") as fh:
            payload = pickle.load(fh)

        self._index       = payload["index"]
        self._chunk_ids   = payload["chunk_ids"]
        self._chunk_texts = payload["chunk_texts"]
        self._chunk_meta  = payload["chunk_meta"]
        self.k1           = payload.get("k1", self.k1)
        self.b            = payload.get("b",  self.b)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k chunks most relevant to *query* via BM25.

        Parameters
        ----------
        query:
            Free-text query string.  Tokenized with :func:`tokenize`
            before scoring.
        top_k:
            Number of results to return.  Clamped to the corpus size if
            larger.

        Returns
        -------
        List[RetrievalResult]
            Results sorted descending by BM25 score (rank 1 = best).
            If *query* is empty or tokenizes to nothing, returns an
            empty list.

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded.
        """
        if self._index is None:
            raise RuntimeError(
                "BM25 index not available. Call build() or load() first."
            )

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Clamp top_k to corpus size
        n = len(self._chunk_ids)
        top_k = min(top_k, n)

        # Partial sort: O(n) instead of O(n log n) for large corpora
        import numpy as np  # numpy is already in requirements

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            meta = dict(self._chunk_meta[idx])
            results.append(
                RetrievalResult(
                    chunk_id=self._chunk_ids[idx],
                    doc_id=meta.pop("doc_id", ""),
                    score=float(scores[idx]),
                    rank=rank,
                    text=self._chunk_texts[idx],
                    metadata=meta,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Legacy retrieve() alias — keeps backward compatibility with the
    # original stub signature used by other parts of the codebase.
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 100) -> List[tuple]:
        """Alias for search() returning (chunk_id, score) tuples.

        Provided for backward compatibility with the original stub.
        Prefer :meth:`search` for new code.
        """
        results = self.search(query, top_k=top_k)
        return [(r.chunk_id, r.score) for r in results]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """True if the index is ready for querying."""
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the index (0 if not yet built)."""
        return len(self._chunk_ids)
