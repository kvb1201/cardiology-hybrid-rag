"""
bm25.py
-------
BM25 sparse retrieval using the rank-bm25 library.

Responsibilities (to be implemented):
  - Build a BM25 index from a list of text chunks
  - Persist / load the index to/from indexes/bm25/
  - Query the index and return ranked (chunk_id, score) pairs
"""

from __future__ import annotations

from typing import List, Tuple


class BM25Retriever:
    """Sparse BM25 retriever over a corpus of text chunks.

    Attributes:
        index_path: Directory where the serialized index is stored.
    """

    def __init__(self, index_path: str = "indexes/bm25") -> None:
        self.index_path = index_path
        self._index = None  # populated by build() or load()

    def build(self, corpus: List[str], chunk_ids: List[str]) -> None:
        """Tokenize corpus and build the BM25 index.

        Args:
            corpus: List of chunk text strings.
            chunk_ids: Parallel list of unique chunk identifiers.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("BM25Retriever.build() is not yet implemented.")

    def save(self) -> None:
        """Persist the index to self.index_path.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("BM25Retriever.save() is not yet implemented.")

    def load(self) -> None:
        """Load a previously saved index from self.index_path.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("BM25Retriever.load() is not yet implemented.")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve the top-k chunks for a query.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted descending by score.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("BM25Retriever.retrieve() is not yet implemented.")
