"""
retrieval_metrics.py
--------------------
Standard information-retrieval evaluation metrics for the cardiology
hybrid RAG project.

All functions use **graded or binary relevance** depending on the caller:

Graded relevance (used with qrels from the evaluation dataset):
    0 = not relevant
    1 = partially relevant
    2 = highly relevant

Binary relevance (used when qrels only contain {0, 1}):
    A chunk is considered relevant if its relevance score >= 1.

Design constraints
------------------
* Every function is a pure function — no global state, no file I/O.
* Every function accepts generic Python types (lists/sets/dicts) so it
  can evaluate BM25, LSH, Dense, Hybrid, and Reranker results with
  identical code.
* Metric functions are designed for retrieval evaluation, not
  diagnostic evaluation (that lives in diagnostic_metrics.py).
* Implementations follow standard TREC/IR conventions.

Reference
---------
Manning, Raghavan, Schütze. Introduction to Information Retrieval.
Cambridge University Press. 2008.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_binary(relevant: Set[str], threshold: int = 1) -> Set[str]:
    """Return a plain set of relevant IDs (identity — already a set)."""
    return relevant


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved chunk IDs (most relevant first).
    relevant:
        Set of ground-truth relevant chunk IDs (binary relevance).
    k:
        Cut-off rank.

    Returns
    -------
    float
        Precision@k in [0, 1].  Returns 0.0 if *retrieved* is empty.

    Examples
    --------
    >>> precision_at_k(["a","b","c","d"], {"a","c"}, k=4)
    0.5
    >>> precision_at_k(["a","b","c"], {"a","c"}, k=2)
    0.5
    """
    _check_k(k)
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for cid in top_k if cid in relevant)
    return hits / k


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of all relevant items found in the top-k retrieved results.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved chunk IDs.
    relevant:
        Set of ground-truth relevant chunk IDs.
    k:
        Cut-off rank.

    Returns
    -------
    float
        Recall@k in [0, 1].  Returns 0.0 if *relevant* is empty.

    Examples
    --------
    >>> recall_at_k(["a","b","c","d"], {"a","c","e"}, k=4)
    0.6666666666666666
    """
    _check_k(k)
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


# ---------------------------------------------------------------------------
# Reciprocal Rank (RR)
# ---------------------------------------------------------------------------


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Reciprocal rank of the first relevant item in *retrieved*.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved chunk IDs.
    relevant:
        Set of ground-truth relevant chunk IDs.

    Returns
    -------
    float
        1 / rank_of_first_relevant, or 0.0 if no relevant item is found.

    Examples
    --------
    >>> reciprocal_rank(["a","b","c"], {"b"})
    0.5
    >>> reciprocal_rank(["a","b","c"], {"d"})
    0.0
    """
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank (MRR)
# ---------------------------------------------------------------------------


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant:  List[Set[str]],
) -> float:
    """Mean Reciprocal Rank (MRR) across a set of queries.

    Parameters
    ----------
    all_retrieved:
        List of retrieved ID lists, one per query.
    all_relevant:
        List of relevant ID sets, one per query (parallel to
        *all_retrieved*).

    Returns
    -------
    float
        MRR in [0, 1].  Returns 0.0 if the input lists are empty.

    Raises
    ------
    ValueError
        If *all_retrieved* and *all_relevant* have different lengths.
    """
    if len(all_retrieved) != len(all_relevant):
        raise ValueError(
            f"all_retrieved and all_relevant must have the same length; "
            f"got {len(all_retrieved)} and {len(all_relevant)}"
        )
    if not all_retrieved:
        return 0.0
    rr_scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    return sum(rr_scores) / len(rr_scores)


# ---------------------------------------------------------------------------
# Average Precision (AP)
# ---------------------------------------------------------------------------


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Average Precision (AP) for a single query.

    AP is the mean of Precision@k at every rank where a relevant
    document is retrieved.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved chunk IDs.
    relevant:
        Set of ground-truth relevant chunk IDs.

    Returns
    -------
    float
        AP in [0, 1].  Returns 0.0 if *relevant* is empty or no
        relevant item appears in *retrieved*.

    Examples
    --------
    >>> average_precision(["a","b","c","d"], {"a","c"})
    0.6666666666666666
    """
    if not relevant or not retrieved:
        return 0.0
    hits    = 0
    sum_prec = 0.0
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            hits += 1
            sum_prec += hits / rank
    if hits == 0:
        return 0.0
    return sum_prec / len(relevant)


# ---------------------------------------------------------------------------
# Mean Average Precision (MAP)
# ---------------------------------------------------------------------------


def mean_average_precision(
    all_retrieved: List[List[str]],
    all_relevant:  List[Set[str]],
) -> float:
    """Mean Average Precision (MAP) over a set of queries.

    Parameters
    ----------
    all_retrieved:
        List of retrieved ID lists, one per query.
    all_relevant:
        List of relevant ID sets, one per query.

    Returns
    -------
    float
        MAP in [0, 1].  Returns 0.0 if the input lists are empty.

    Raises
    ------
    ValueError
        If the two input lists have different lengths.
    """
    if len(all_retrieved) != len(all_relevant):
        raise ValueError(
            f"all_retrieved and all_relevant must have the same length; "
            f"got {len(all_retrieved)} and {len(all_relevant)}"
        )
    if not all_retrieved:
        return 0.0
    ap_scores = [
        average_precision(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    return sum(ap_scores) / len(ap_scores)


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------


def ndcg_at_k(
    retrieved: List[str],
    relevant:  Set[str],
    k: int,
    graded_relevance: Dict[str, int] | None = None,
) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    Supports both binary relevance (``relevant`` set) and graded
    relevance (``graded_relevance`` dict mapping chunk_id → int grade).
    When ``graded_relevance`` is provided, ``relevant`` is ignored.

    Parameters
    ----------
    retrieved:
        Ordered list of retrieved chunk IDs (most relevant first).
    relevant:
        Set of ground-truth relevant chunk IDs (binary relevance,
        gain = 1 for each).  Ignored when *graded_relevance* is given.
    k:
        Cut-off rank.
    graded_relevance:
        Optional dict mapping chunk_id → integer relevance grade
        (0 = not relevant, 1 = partial, 2 = highly relevant).
        If provided this takes precedence over *relevant*.

    Returns
    -------
    float
        NDCG@k in [0, 1].  Returns 0.0 if there are no relevant items
        or if *retrieved* is empty.

    Notes
    -----
    Uses the standard logarithmic discount: discount(i) = 1 / log2(i+1)
    for rank i (1-based).

    Examples
    --------
    >>> ndcg_at_k(["a","b","c","d"], {"a","c"}, k=4)
    0.9502...
    """
    _check_k(k)

    def _gain(cid: str) -> float:
        if graded_relevance is not None:
            return float(graded_relevance.get(cid, 0))
        return 1.0 if cid in relevant else 0.0

    # Compute DCG@k
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        g = _gain(cid)
        if g > 0:
            dcg += g / math.log2(i + 1)

    # Compute IDCG@k (ideal DCG — best possible ranking)
    if graded_relevance is not None:
        ideal_gains = sorted(graded_relevance.values(), reverse=True)
    else:
        ideal_gains = [1.0] * len(relevant)

    ideal_gains = ideal_gains[:k]
    idcg = sum(
        g / math.log2(i + 1)
        for i, g in enumerate(ideal_gains, start=1)
        if g > 0
    )

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
