"""
reranker.py
-----------
Cross-encoder reranking of candidate passages.

A cross-encoder scores (query, passage) pairs jointly, giving more accurate
relevance estimates than bi-encoder or sparse scores but at higher latency.
This module reranks the top-N candidates returned by HybridRetriever.

Responsibilities (to be implemented):
  - Load a cross-encoder model (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)
  - Score each (query, chunk_text) pair
  - Return a reranked list of (chunk_id, cross_encoder_score) tuples
"""

from __future__ import annotations

from typing import List, Tuple


class CrossEncoderReranker:
    """Cross-encoder reranker for refining hybrid retrieval results.

    Attributes:
        model_name: HuggingFace cross-encoder model name.
        batch_size: Inference batch size.
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None  # loaded lazily in load_model()

    def load_model(self) -> None:
        """Load the cross-encoder model into memory.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError(
            "CrossEncoderReranker.load_model() is not yet implemented."
        )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Score and rerank candidate (chunk_id, chunk_text) pairs.

        Args:
            query: Query string.
            candidates: List of (chunk_id, chunk_text) tuples to rerank.
            top_k: How many top results to return after reranking.

        Returns:
            List of (chunk_id, cross_encoder_score) tuples sorted descending,
            truncated to top_k.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError(
            "CrossEncoderReranker.rerank() is not yet implemented."
        )
