"""
hybrid.py
---------
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF formula (Cormack et al. 2009):

    RRF(d) = Σ_r  1 / (rrf_k + rank_r(d))

where:
  - d       is a chunk/document
  - r       iterates over the constituent retrievers
  - rank_r  is the 1-based rank assigned by retriever r  (absent → no contribution)
  - rrf_k   is a smoothing constant  (fixed at 60 per the standard; do NOT tune)

Design decisions
----------------
* Raw scores from individual retrievers (BM25 TF-IDF, Jaccard, cosine) are
  NOT combined — only rank positions are used.  This is intentional: the
  scores are on incompatible scales and combining them would require tuning.
* rrf_k = 60 is fixed before any evaluation and must not be changed based
  on qrels results.  This is the value used in the original paper and in
  virtually all subsequent IR work.
* candidate_depth = 20 (configurable) — the number of candidates each
  constituent retriever contributes.  Deeper pools increase recall but
  also latency.
* If a constituent retriever returns zero candidates (e.g., LSH for a
  query with no shingle overlap), it simply contributes nothing to the
  fusion — no candidates are fabricated.
* The HybridRetriever implements BaseRetriever.search() so it is a
  drop-in replacement for any single retriever in the evaluation pipeline.

Stage 2E systems evaluated
---------------------------
System A:  BM25 + Dense          (two-retriever RRF)
System B:  BM25 + Dense + LSH    (three-retriever RRF)

The LSH component is included in System B to test whether its sparse
lexical-fragment signal provides any complementary boost despite its weak
standalone performance (NDCG@10 = 0.074, 41.5% zero-candidate queries).

Reference
---------
Cormack, G.V., Clarke, C.L.A., Buettcher, S. (2009).
"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
Learning Methods." SIGIR 2009.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.retrieval.base import BaseRetriever, RetrievalResult

# Fixed baseline constant.  Do not change after evaluation begins.
DEFAULT_RRF_K       = 60
DEFAULT_CAND_DEPTH  = 20


# ---------------------------------------------------------------------------
# Core RRF function (standalone, reusable)
# ---------------------------------------------------------------------------


def rrf_fuse(
    ranked_lists: List[List[RetrievalResult]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked RetrievalResult lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Each element is a list of :class:`~src.retrieval.base.RetrievalResult`
        objects from one retriever, already sorted descending by score.
        Empty lists are silently skipped (e.g., LSH returning 0 candidates).
    rrf_k:
        Smoothing constant.  Default: 60 (Cormack et al. 2009).
        Do not tune this value against evaluation results.
    top_k:
        If given, return only the top-*top_k* results.

    Returns
    -------
    List[Tuple[str, float]]
        ``(chunk_id, rrf_score)`` pairs sorted **descending** by RRF score.
        Ties are broken by chunk_id (alphabetical) for determinism.
    """
    # Accumulate RRF scores: chunk_id → cumulative RRF score
    scores: Dict[str, float] = {}

    for ranked in ranked_lists:
        if not ranked:
            continue  # retriever returned 0 candidates — skip silently
        for result in ranked:
            # rank is 1-based; RRF contribution = 1 / (rrf_k + rank)
            contrib = 1.0 / (rrf_k + result.rank)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + contrib

    # Sort: descending by score, then ascending by chunk_id for tie-breaking
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    if top_k is not None:
        fused = fused[:top_k]
    return fused


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever(BaseRetriever):
    """RRF-based hybrid retriever combining two or three constituent retrievers.

    Parameters
    ----------
    retrievers:
        Ordered list of already-loaded retriever instances.  Accepts any
        combination of :class:`~src.retrieval.bm25.BM25Retriever`,
        :class:`~src.retrieval.lsh.LSHRetriever`, and
        :class:`~src.retrieval.dense.DenseRetriever`.
    rrf_k:
        RRF smoothing constant.  **Fixed at 60 — do not tune on qrels.**
    candidate_depth:
        Number of candidates each constituent retriever contributes per
        query before fusion.  Default: 20.

    Notes
    -----
    The ``metadata`` dict of each returned :class:`RetrievalResult`
    contains a ``constituent_ranks`` sub-dict recording the rank assigned
    by each constituent retriever (or ``None`` if that retriever did not
    return the chunk).  This is useful for analysis but should not be used
    to further re-rank results.

    Example
    -------
    ::

        from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

        bm25  = BM25Retriever();  bm25.load()
        dense = DenseRetriever(); dense.load()

        hybrid = HybridRetriever([bm25, dense], rrf_k=60, candidate_depth=20)
        results = hybrid.search("heart failure treatment", top_k=10)
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        rrf_k: int = DEFAULT_RRF_K,
        candidate_depth: int = DEFAULT_CAND_DEPTH,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one constituent retriever.")
        self.retrievers      = retrievers
        self.rrf_k           = rrf_k
        self.candidate_depth = candidate_depth

    # ------------------------------------------------------------------
    # BaseRetriever.search implementation
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve top-k chunks via RRF fusion of constituent retrievers.

        Parameters
        ----------
        query:
            Free-text query string.
        top_k:
            Number of results to return after fusion.

        Returns
        -------
        List[RetrievalResult]
            Fused results sorted descending by RRF score.  The
            ``RetrievalResult.score`` is the RRF score (range > 0, no
            fixed upper bound — do NOT compare to BM25 or cosine scores).
            Each result's ``metadata`` includes ``constituent_ranks``
            mapping constituent retriever index to its rank (or None).

        Notes
        -----
        If all constituent retrievers return empty results for a query,
        an empty list is returned.
        """
        query = query.strip() if query else ""
        if not query:
            return []

        # Step 1: collect ranked lists from all constituent retrievers
        ranked_lists: List[List[RetrievalResult]] = []
        for ret in self.retrievers:
            results = ret.search(query, top_k=self.candidate_depth)
            ranked_lists.append(results)

        # Step 2: build a lookup: chunk_id → RetrievalResult (for text/meta)
        chunk_lookup: Dict[str, RetrievalResult] = {}
        for rl in ranked_lists:
            for r in rl:
                if r.chunk_id not in chunk_lookup:
                    chunk_lookup[r.chunk_id] = r

        if not chunk_lookup:
            return []

        # Step 3: build per-retriever rank maps: chunk_id → rank
        rank_maps: List[Dict[str, int]] = [
            {r.chunk_id: r.rank for r in rl} for rl in ranked_lists
        ]

        # Step 4: RRF fusion
        fused_pairs = rrf_fuse(ranked_lists, rrf_k=self.rrf_k, top_k=top_k)

        # Step 5: assemble RetrievalResult objects
        results_out: List[RetrievalResult] = []
        for final_rank, (chunk_id, rrf_score) in enumerate(fused_pairs, start=1):
            base = chunk_lookup[chunk_id]
            # Record per-constituent ranks for analysis
            constituent_ranks = {
                f"retriever_{i}": rank_maps[i].get(chunk_id)
                for i in range(len(self.retrievers))
            }
            meta = dict(base.metadata)
            meta["constituent_ranks"] = constituent_ranks
            meta["rrf_k"]             = self.rrf_k

            results_out.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=base.doc_id,
                    score=rrf_score,
                    rank=final_rank,
                    text=base.text,
                    metadata=meta,
                )
            )

        return results_out

    # ------------------------------------------------------------------
    # Legacy alias (backward compat with original stub)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        use_bm25: bool = True,
        use_lsh:  bool = True,
        use_dense: bool = True,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Alias for search() returning (chunk_id, rrf_score) tuples.

        The *use_bm25*, *use_lsh*, *use_dense*, and *weights* parameters
        are accepted for backward compatibility but ignored — the active
        retrievers are those supplied to ``__init__``.  Prefer
        :meth:`search` for new code.
        """
        return [(r.chunk_id, r.score) for r in self.search(query, top_k=top_k)]

    @property
    def n_retrievers(self) -> int:
        """Number of constituent retrievers."""
        return len(self.retrievers)
