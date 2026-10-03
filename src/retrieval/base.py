"""
base.py
-------
Common retrieval interface shared by all retriever implementations:
BM25Retriever, LSHRetriever, DenseRetriever, and HybridRetriever.

Every retriever returns a list of RetrievalResult objects so that
downstream code (reranker, RAG pipeline, evaluation) is retriever-agnostic.

Design rules
------------
* BaseRetriever is an ABC — callers should type-hint against it.
* RetrievalResult is a plain frozen dataclass with no retriever-specific
  fields; extra metadata is stuffed into the `metadata` dict.
* `search()` is the ONLY method callers need; build/save/load are
  implementation details each retriever handles internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """A single ranked result returned by any retriever.

    Attributes
    ----------
    chunk_id:
        The unique identifier of the retrieved chunk
        (matches ``Chunk.chunk_id`` in the frozen corpus).
    doc_id:
        Source document identifier (``Chunk.doc_id``).
    score:
        Retriever-specific relevance score.
        Higher is always more relevant — individual retrievers are
        responsible for ensuring this invariant (e.g., BM25 scores are
        already higher-is-better; cosine similarities need no inversion).
    rank:
        1-based position in the ranked list (1 = most relevant).
    text:
        The chunk text.  Included so callers do not need to look up the
        chunk store for display or prompt construction.
    metadata:
        Optional extra fields (title, source path, section label, etc.)
        copied from the originating ``Chunk.metadata``.
    """

    chunk_id: str
    doc_id: str
    score: float
    rank: int
    text: str
    metadata: Dict = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for logging / JSON output)."""
        return {
            "rank":     self.rank,
            "chunk_id": self.chunk_id,
            "doc_id":   self.doc_id,
            "score":    self.score,
            "text":     self.text,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseRetriever(ABC):
    """Abstract base class for all retrievers in this project.

    Subclasses must implement :meth:`search`.

    The build / save / load lifecycle is left to each implementation
    because different backends have very different serialisation needs.
    Callers should always call :meth:`search` through this interface.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query:
            Free-text clinical or research query.
        top_k:
            Maximum number of results to return.
            If *top_k* exceeds the total number of indexed chunks the
            method must return all chunks rather than raising an error.

        Returns
        -------
        List[RetrievalResult]
            Ranked results, most relevant first (rank 1 = best).
            The list may be shorter than *top_k* if the corpus is small
            or if the query matches fewer than *top_k* chunks.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_ranks(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Return a new list with rank fields set to 1-based positions.

        The input list must already be sorted descending by score.
        This helper allows subclasses to build an unranked list first
        and assign ranks in one final pass.
        """
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                score=r.score,
                rank=i + 1,
                text=r.text,
                metadata=r.metadata,
            )
            for i, r in enumerate(results)
        ]
