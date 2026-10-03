"""
lsh.py
------
MinHash + Locality-Sensitive Hashing (LSH) retrieval using datasketch.

Architecture
------------
                        chunks.jsonl
                             │
                      load_chunks()
                             │
                    ┌────────▼────────┐
                    │  shingle(text)  │  word k-grams, lowercased
                    └────────┬────────┘
                             │  set of frozenset tuples
                    ┌────────▼────────┐
                    │   MinHash(128)  │  datasketch.MinHash per chunk
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  MinHashLSH     │  approximate neighbour index
                    │  (threshold=0.3)│  threshold = 0.3 (baseline)
                    └────────┬────────┘
                             │  .save()  ──► indexes/lsh/lsh_index.pkl
                             │  .load()  ◄── indexes/lsh/lsh_index.pkl
                             │
                    ┌────────▼────────┐
                    │    search()     │  returns List[RetrievalResult]
                    └─────────────────┘

Shingling
---------
Function: ``shingle(text, k)``
- Lowercase and split on whitespace (same normalisation as BM25 tokenizer)
- Produce overlapping k-word tuples (word k-grams)
- Default k = 3 (configurable, do NOT tune on evaluation results)
- Empty / very short texts produce an empty set; handled gracefully

MinHash
-------
- Uses datasketch.MinHash with ``num_perm`` permutations (default 128)
- Each chunk's shingle set is hashed into a MinHash signature
- MinHash provides an unbiased estimator of Jaccard similarity

LSH Candidate Retrieval
-----------------------
datasketch.MinHashLSH stores signatures in banded hash tables.
``threshold`` sets the *approximate* Jaccard similarity required to be a
candidate (default 0.3 — lower than BM25's term-overlap regime to cast a
wider net for recall comparison).

Because MinHashLSH returns a *candidate set* rather than a ranked list:

  1. query shingles → query MinHash
  2. MinHashLSH.query(query_minhash)  →  candidate set (unranked)
  3. For each candidate, compute **exact** Jaccard similarity between the
     query shingle set and the chunk shingle set stored at index time.
  4. Sort candidates by exact Jaccard descending.
  5. Return top-k.

If fewer than k candidates survive step 2, the method returns as many as are
available — this is documented behaviour, not a bug.  The caller (evaluation
code) treats missing results as 0 in precision/recall calculations.

Score semantics
---------------
``RetrievalResult.score`` = **exact Jaccard similarity** over word-shingle sets.
Range: [0.0, 1.0].
Do NOT compare numerically to BM25 scores (different scale and meaning).

Index persistence
-----------------
A single pickle stores: the MinHashLSH object, chunk_ids, chunk_texts,
chunk_meta, and chunk_shingle_sets (needed for exact Jaccard re-scoring).

Rebuild command::

    python scripts/build_indexes.py --build-lsh

or programmatically::

    from src.retrieval.lsh import LSHRetriever
    from src.preprocessing.chunker import load_chunks

    chunks = load_chunks("data/chunks/chunks.jsonl")
    r = LSHRetriever()
    r.build(chunks)
    r.save()
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Tuple

from src.retrieval.base import BaseRetriever, RetrievalResult

try:
    from datasketch import MinHash, MinHashLSH
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "datasketch is required: pip install datasketch==1.6.5"
    ) from exc


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------

# Reuse the same lowercasing + alphanumeric-hyphen filter as BM25 to ensure
# the two retrievers operate on the same lexical space.
_TOKEN_RE  = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?")
_HAS_LETTER = re.compile(r"[A-Za-z]")


def _tokenize_for_lsh(text: str) -> List[str]:
    """Tokenize text identically to the BM25 tokenizer.

    Lowercase, keep alphanumeric+hyphen runs that contain at least one letter.
    Preserves clinical abbreviations (SGLT-2, HFrEF, LVEF).
    """
    if not text:
        return []
    lowered = text.lower()
    return [tok for tok in _TOKEN_RE.findall(lowered) if _HAS_LETTER.search(tok)]


def shingle(text: str, k: int = 3) -> FrozenSet[Tuple[str, ...]]:
    """Produce a frozen set of word k-grams (k-shingles) from *text*.

    Parameters
    ----------
    text:
        Input string.  Normalised identically to the BM25 tokenizer:
        lowercased, alphanumeric+hyphen tokens, minimum 1 letter per token.
    k:
        Shingle size (number of consecutive words per shingle).
        Default: 3.  Baseline value — do not tune on evaluation results.

    Returns
    -------
    FrozenSet[Tuple[str, ...]]
        Immutable set of k-tuples.  Empty frozenset for text shorter than k
        tokens (handled gracefully by downstream code).

    Examples
    --------
    >>> shingle("heart failure treatment", k=2)
    frozenset({('heart', 'failure'), ('failure', 'treatment')})
    >>> shingle("short", k=3)     # fewer than k tokens → empty
    frozenset()
    >>> shingle("", k=3)
    frozenset()
    """
    if k <= 0:
        raise ValueError(f"shingle size k must be >= 1, got {k}")
    tokens = _tokenize_for_lsh(text)
    if len(tokens) < k:
        return frozenset()
    return frozenset(tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def _minhash_from_shingles(shingles: FrozenSet[Tuple[str, ...]], num_perm: int) -> MinHash:
    """Build a datasketch MinHash from a shingle set.

    Each shingle tuple is encoded to UTF-8 bytes before hashing.
    """
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(" ".join(s).encode("utf-8"))
    return mh


def _jaccard(a: FrozenSet, b: FrozenSet) -> float:
    """Exact Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0   # both empty → identical (degenerate case)
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# LSHRetriever
# ---------------------------------------------------------------------------


class LSHRetriever(BaseRetriever):
    """MinHash-LSH retriever for approximate Jaccard-similarity search.

    Parameters
    ----------
    index_path:
        Directory for serialised index files.
    shingle_size:
        Word k-gram size.  Default: 3.  Baseline — do not tune.
    num_perm:
        Number of MinHash permutations.  Default: 128.  Higher → more
        accurate but slower to build and query.
    threshold:
        Approximate Jaccard threshold for LSH candidate generation.
        Default: 0.3.  Lower = more candidates (higher recall, slower).
        This is an initial baseline value.
    chunks_path:
        Path to the frozen corpus JSONL file used by :meth:`build`.

    Notes
    -----
    The *threshold* parameter controls candidate generation, not final
    ranking.  After LSH candidate lookup, results are re-ranked by exact
    Jaccard similarity.  The threshold should be set low enough that the
    correct answer is included in the candidate set for most queries.

    Because LSH retrieval is set-similarity based and corpus chunks are
    256-token windows, a threshold of 0.3 is intentionally low to ensure
    adequate recall for comparison with BM25.
    """

    DEFAULT_INDEX_PATH  = Path("indexes/lsh")
    DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
    INDEX_FILENAME      = "lsh_index.pkl"

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        shingle_size: int = 3,
        num_perm: int = 128,
        threshold: float = 0.3,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    ) -> None:
        self.index_path   = Path(index_path)
        self.shingle_size = shingle_size
        self.num_perm     = num_perm
        self.threshold    = threshold
        self.chunks_path  = Path(chunks_path)

        self._lsh: Optional[MinHashLSH] = None
        self._chunk_ids:      List[str]            = []
        self._chunk_texts:    List[str]            = []
        self._chunk_meta:     List[dict]           = []
        self._chunk_shingles: List[FrozenSet]      = []   # exact Jaccard re-scoring

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks=None) -> None:
        """Build the MinHash-LSH index from a list of Chunk objects.

        Parameters
        ----------
        chunks:
            List of ``Chunk`` objects.  If *None*, loads from
            ``self.chunks_path``.

        Notes
        -----
        Chunks whose shingle set is empty (fewer than ``shingle_size``
        tokens) are still indexed using an empty MinHash so that the
        internal position arrays remain aligned with ``_chunk_ids``.
        They will never be returned as candidates because no query shingle
        will match them.

        Calling ``build()`` replaces any previously built in-memory index.
        Call :meth:`save` afterwards to persist.
        """
        if chunks is None:
            from src.preprocessing.chunker import load_chunks
            chunks = load_chunks(self.chunks_path)

        if not chunks:
            raise ValueError("Cannot build LSH index from an empty chunk list.")

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        chunk_ids:      List[str]       = []
        chunk_texts:    List[str]       = []
        chunk_meta:     List[dict]      = []
        chunk_shingles: List[FrozenSet] = []

        for c in chunks:
            cid      = c.chunk_id
            text     = c.text
            shingles = shingle(text, k=self.shingle_size)
            mh       = _minhash_from_shingles(shingles, self.num_perm)

            lsh.insert(cid, mh)

            chunk_ids.append(cid)
            chunk_texts.append(text)
            chunk_shingles.append(shingles)
            chunk_meta.append(
                {
                    "doc_id":       c.doc_id,
                    "title":        c.metadata.get("title", ""),
                    "source":       c.metadata.get("source", ""),
                    "section":      c.metadata.get("section", ""),
                    "chunk_index":  c.metadata.get("chunk_index", 0),
                    "total_chunks": c.metadata.get("total_chunks", 0),
                    "strategy":     c.metadata.get("strategy", ""),
                }
            )

        self._lsh           = lsh
        self._chunk_ids     = chunk_ids
        self._chunk_texts   = chunk_texts
        self._chunk_meta    = chunk_meta
        self._chunk_shingles = chunk_shingles

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the index to ``<index_path>/lsh_index.pkl``.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called yet.
        """
        if self._lsh is None:
            raise RuntimeError("Index not built. Call build() first.")

        self.index_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "lsh":            self._lsh,
            "chunk_ids":      self._chunk_ids,
            "chunk_texts":    self._chunk_texts,
            "chunk_meta":     self._chunk_meta,
            "chunk_shingles": self._chunk_shingles,
            "shingle_size":   self.shingle_size,
            "num_perm":       self.num_perm,
            "threshold":      self.threshold,
        }
        out = self.index_path / self.INDEX_FILENAME
        with out.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """Load a previously saved index from ``<index_path>/lsh_index.pkl``.

        Raises
        ------
        FileNotFoundError
            If the index file does not exist.
        """
        pkl = self.index_path / self.INDEX_FILENAME
        if not pkl.exists():
            raise FileNotFoundError(
                f"LSH index not found at '{pkl}'. "
                "Run build() + save() or: python scripts/build_indexes.py --build-lsh"
            )
        with pkl.open("rb") as fh:
            payload = pickle.load(fh)

        self._lsh            = payload["lsh"]
        self._chunk_ids      = payload["chunk_ids"]
        self._chunk_texts    = payload["chunk_texts"]
        self._chunk_meta     = payload["chunk_meta"]
        self._chunk_shingles = payload["chunk_shingles"]
        self.shingle_size    = payload.get("shingle_size", self.shingle_size)
        self.num_perm        = payload.get("num_perm",     self.num_perm)
        self.threshold       = payload.get("threshold",    self.threshold)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k chunks most similar to *query* by Jaccard similarity.

        Pipeline
        --------
        1. Shingle the query.
        2. Build a query MinHash.
        3. Query MinHashLSH to get approximate candidates.
        4. Compute **exact** Jaccard similarity between query shingles and
           each candidate's stored shingle set.
        5. Sort by exact Jaccard descending, return top-k.

        Parameters
        ----------
        query:
            Free-text query string.
        top_k:
            Maximum number of results.  If fewer candidates are found than
            *top_k*, the method returns all available candidates.
            **This is documented behaviour**: LSH may return fewer than
            top_k results when the query has low lexical overlap with the
            corpus.  Evaluation code treats missing results as rank > k.

        Returns
        -------
        List[RetrievalResult]
            Sorted descending by exact Jaccard similarity.  May be shorter
            than *top_k*.  Returns empty list for blank / all-number queries.

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded.
        """
        if self._lsh is None:
            raise RuntimeError(
                "LSH index not available. Call build() or load() first."
            )

        query_shingles = shingle(query, k=self.shingle_size)
        if not query_shingles:
            return []

        query_mh = _minhash_from_shingles(query_shingles, self.num_perm)

        # Step 1: approximate candidate set from LSH
        try:
            candidate_ids: List[str] = self._lsh.query(query_mh)
        except Exception:  # noqa: BLE001
            candidate_ids = []

        if not candidate_ids:
            return []

        # Build a fast lookup: chunk_id → positional index
        id_to_idx = {cid: i for i, cid in enumerate(self._chunk_ids)}

        # Step 2: exact Jaccard re-scoring
        scored: List[Tuple[float, int]] = []
        for cid in candidate_ids:
            idx = id_to_idx.get(cid)
            if idx is None:
                continue
            j = _jaccard(query_shingles, self._chunk_shingles[idx])
            scored.append((j, idx))

        # Sort descending by exact Jaccard
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: min(top_k, len(scored))]

        results = []
        for rank, (score, idx) in enumerate(scored, start=1):
            meta = dict(self._chunk_meta[idx])
            results.append(
                RetrievalResult(
                    chunk_id=self._chunk_ids[idx],
                    doc_id=meta.pop("doc_id", ""),
                    score=score,
                    rank=rank,
                    text=self._chunk_texts[idx],
                    metadata=meta,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Legacy alias
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """Alias for search() returning (chunk_id, jaccard_score) tuples.

        Provided for backward compatibility with the original stub signature.
        Prefer :meth:`search` for new code.
        """
        return [(r.chunk_id, r.score) for r in self.search(query, top_k=top_k)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """True if the index is ready for querying."""
        return self._lsh is not None

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the index (0 if not yet built)."""
        return len(self._chunk_ids)
