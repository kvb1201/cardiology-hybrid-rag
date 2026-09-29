"""
dense.py
--------
Dense vector retrieval using sentence-transformers embeddings + FAISS index.

Responsibilities (to be implemented):
  - Encode text chunks into dense vectors via a sentence-transformer model
  - Build a FAISS index (e.g., IndexFlatIP or IndexHNSWFlat) over chunk embeddings
  - Persist / load the index and chunk-ID mapping to/from indexes/faiss/
  - Query the index and return ranked (chunk_id, cosine_score) pairs
"""

from __future__ import annotations

from typing import List, Tuple


class DenseRetriever:
    """Dense retriever: sentence-transformer embeddings + FAISS ANN search.

    Attributes:
        index_path: Directory where the FAISS index is stored.
        model_name: HuggingFace model name used for encoding.
        batch_size: Encoding batch size.
    """

    def __init__(
        self,
        index_path: str = "indexes/faiss",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        self.index_path = index_path
        self.model_name = model_name
        self.batch_size = batch_size
        self._index = None       # FAISS index populated by build() or load()
        self._chunk_ids: List[str] = []  # positional mapping index -> chunk_id

    def build(self, corpus: List[str], chunk_ids: List[str]) -> None:
        """Encode corpus and build the FAISS index.

        Args:
            corpus: List of chunk text strings.
            chunk_ids: Parallel list of unique chunk identifiers.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("DenseRetriever.build() is not yet implemented.")

    def save(self) -> None:
        """Persist the FAISS index and chunk-ID list to self.index_path.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("DenseRetriever.save() is not yet implemented.")

    def load(self) -> None:
        """Load a previously saved FAISS index from self.index_path.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("DenseRetriever.load() is not yet implemented.")

    def encode(self, texts: List[str]):
        """Encode a list of texts into dense vectors.

        Args:
            texts: Strings to encode.

        Returns:
            numpy.ndarray of shape (len(texts), embedding_dim).

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("DenseRetriever.encode() is not yet implemented.")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve the top-k chunks by cosine similarity to the query embedding.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted descending.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("DenseRetriever.retrieve() is not yet implemented.")
