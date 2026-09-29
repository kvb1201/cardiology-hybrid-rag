"""
hybrid.py
---------
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines ranked lists from BM25, LSH, and Dense retrievers into a single
fused ranking. RRF score for a document d across rankers:

    RRF(d) = sum_r [ 1 / (k + rank_r(d)) ]

where k is a smoothing constant (default 60, per Cormack et al. 2009).

Responsibilities (to be implemented):
  - Accept ranked lists from any subset of retrievers
  - Compute RRF fusion scores
  - Return a unified, re-ranked candidate list
  - Support configurable per-retriever weights (weighted RRF variant)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each element is a list of (chunk_id, score) tuples
                      sorted descending by score, from one retriever.
        k: RRF smoothing constant. Higher k reduces the impact of top ranks.
        weights: Optional per-list weights (same length as ranked_lists).
                 Defaults to uniform weighting if None.

    Returns:
        Fused list of (chunk_id, rrf_score) tuples, sorted descending by
        rrf_score.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("reciprocal_rank_fusion() is not yet implemented.")


class HybridRetriever:
    """Orchestrates BM25, LSH, and Dense retrievers and fuses results with RRF.

    Attributes:
        bm25: BM25Retriever instance (optional).
        lsh: LSHRetriever instance (optional).
        dense: DenseRetriever instance (optional).
        rrf_k: RRF smoothing constant.
    """

    def __init__(
        self,
        bm25=None,
        lsh=None,
        dense=None,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = bm25
        self.lsh = lsh
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        use_bm25: bool = True,
        use_lsh: bool = True,
        use_dense: bool = True,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Run enabled retrievers and fuse results with RRF.

        Args:
            query: Query string.
            top_k: Number of candidates each sub-retriever returns before fusion.
            use_bm25: Include BM25 ranked list in fusion.
            use_lsh: Include LSH ranked list in fusion.
            use_dense: Include Dense ranked list in fusion.
            weights: Optional per-retriever weights for weighted RRF.

        Returns:
            Fused list of (chunk_id, rrf_score) tuples, sorted descending.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("HybridRetriever.retrieve() is not yet implemented.")
