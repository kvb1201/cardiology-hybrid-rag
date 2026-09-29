"""
retrieval_metrics.py
--------------------
Standard information-retrieval evaluation metrics.

Metrics to be implemented:
  - Precision@k
  - Recall@k
  - Mean Reciprocal Rank (MRR)
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (NDCG@k)

Each function accepts a list of retrieved chunk IDs and a set of relevant
chunk IDs, matching the standard IR evaluation convention.
"""

from __future__ import annotations

from typing import List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.
        k: Cut-off rank.

    Returns:
        Precision@k in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("precision_at_k() is not yet implemented.")


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved results.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.
        k: Cut-off rank.

    Returns:
        Recall@k in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("recall_at_k() is not yet implemented.")


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Reciprocal rank of the first relevant item in the retrieved list.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.

    Returns:
        1 / rank_of_first_relevant, or 0 if no relevant item is found.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("reciprocal_rank() is not yet implemented.")


def mean_reciprocal_rank(
    all_retrieved: List[List[str]], all_relevant: List[Set[str]]
) -> float:
    """Mean Reciprocal Rank over a set of queries.

    Args:
        all_retrieved: List of retrieved ID lists, one per query.
        all_relevant: List of relevant ID sets, one per query.

    Returns:
        MRR score in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("mean_reciprocal_rank() is not yet implemented.")


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Average Precision for a single query.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.

    Returns:
        Average Precision in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("average_precision() is not yet implemented.")


def mean_average_precision(
    all_retrieved: List[List[str]], all_relevant: List[Set[str]]
) -> float:
    """Mean Average Precision (MAP) over a set of queries.

    Args:
        all_retrieved: List of retrieved ID lists, one per query.
        all_relevant: List of relevant ID sets, one per query.

    Returns:
        MAP score in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("mean_average_precision() is not yet implemented.")


def ndcg_at_k(
    retrieved: List[str], relevant: Set[str], k: int
) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs (binary relevance).
        k: Cut-off rank.

    Returns:
        NDCG@k in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("ndcg_at_k() is not yet implemented.")
