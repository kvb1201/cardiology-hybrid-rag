"""
test_hybrid.py
--------------
Unit tests for src.retrieval.hybrid (rrf_fuse + HybridRetriever).

Coverage
--------
rrf_fuse()
    1.  Single list → same order, scores = 1/(k+rank)
    2.  Duplicate chunk_id across lists → scores accumulate correctly
    3.  Chunk absent from one list gets no contribution from that list
    4.  Empty constituent list is silently skipped
    5.  All empty lists return empty result
    6.  Results are sorted descending by RRF score
    7.  Tie-breaking is deterministic (alphabetical by chunk_id)
    8.  top_k truncates correctly
    9.  rrf_k=60 matches formula exactly

HybridRetriever
    10. Raises ValueError on empty retrievers list
    11. search() returns List[RetrievalResult]
    12. Ranks are 1-based and sequential
    13. RRF scores are positive floats
    14. top_k respected
    15. Empty query returns empty list
    16. Repeated identical queries produce identical rankings
    17. constituent_ranks metadata populated for each result
    18. Chunk present in both retrievers accumulates higher score
    19. BM25 + Dense fusion with real indices (integration, skipped if absent)
    20. LSH zero-candidate query handled gracefully
    21. retrieve() alias returns (chunk_id, score) tuples
    22. n_retrievers property correct
    23. HybridRetriever is subclass of BaseRetriever
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.base   import BaseRetriever, RetrievalResult
from src.retrieval.hybrid import HybridRetriever, rrf_fuse, DEFAULT_RRF_K

# ── Index availability flags ──────────────────────────────────────────────────
BM25_IDX   = ROOT / "indexes" / "bm25"  / "bm25_index.pkl"
DENSE_IDX  = ROOT / "indexes" / "faiss" / "dense_index.faiss"
LSH_IDX    = ROOT / "indexes" / "lsh"   / "lsh_index.pkl"
INDICES_OK = BM25_IDX.exists() and DENSE_IDX.exists() and LSH_IDX.exists()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(chunk_id: str, rank: int, score: float = 1.0,
                  doc_id: str = "doc0") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, doc_id=doc_id,
        score=score, rank=rank,
        text=f"text for {chunk_id}", metadata={},
    )


def _make_ranked(ids_scores: List[tuple]) -> List[RetrievalResult]:
    """Build a ranked list from (chunk_id, score) pairs (score ignored by RRF)."""
    return [
        _make_result(cid, rank=i + 1, score=sc)
        for i, (cid, sc) in enumerate(ids_scores)
    ]


class _FakeRetriever(BaseRetriever):
    """Deterministic fake retriever for unit tests."""

    def __init__(self, results: List[RetrievalResult]):
        self._results = results

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        if not query.strip():
            return []
        return self._results[:top_k]


# ── rrf_fuse() tests ──────────────────────────────────────────────────────────

class TestRRFFuse:
    def test_single_list_preserves_order(self):
        rl = _make_ranked([("A", 10), ("B", 8), ("C", 6)])
        result = rrf_fuse([rl], rrf_k=60)
        ids = [r[0] for r in result]
        assert ids == ["A", "B", "C"]

    def test_single_list_scores_match_formula(self):
        rl = _make_ranked([("A", 10), ("B", 8)])
        result = dict(rrf_fuse([rl], rrf_k=60))
        assert result["A"] == pytest.approx(1 / (60 + 1))
        assert result["B"] == pytest.approx(1 / (60 + 2))

    def test_duplicate_chunk_scores_accumulate(self):
        rl1 = _make_ranked([("A", 10), ("B", 8)])   # A rank 1, B rank 2
        rl2 = _make_ranked([("B", 9),  ("A", 7)])   # B rank 1, A rank 2
        result = dict(rrf_fuse([rl1, rl2], rrf_k=60))
        expected_A = 1/(60+1) + 1/(60+2)
        expected_B = 1/(60+2) + 1/(60+1)
        assert result["A"] == pytest.approx(expected_A)
        assert result["B"] == pytest.approx(expected_B)

    def test_absent_chunk_gets_no_contribution_from_that_list(self):
        rl1 = _make_ranked([("A", 10)])
        rl2 = _make_ranked([("B", 9)])
        result = dict(rrf_fuse([rl1, rl2], rrf_k=60))
        assert result["A"] == pytest.approx(1 / (60 + 1))
        assert result["B"] == pytest.approx(1 / (60 + 1))

    def test_empty_list_silently_skipped(self):
        rl1 = _make_ranked([("A", 10)])
        result = rrf_fuse([rl1, []], rrf_k=60)
        assert len(result) == 1
        assert result[0][0] == "A"

    def test_all_empty_returns_empty(self):
        assert rrf_fuse([[], []], rrf_k=60) == []

    def test_results_sorted_descending(self):
        rl1 = _make_ranked([("A", 10), ("B", 5)])
        rl2 = _make_ranked([("A", 9),  ("C", 3)])
        result = rrf_fuse([rl1, rl2], rrf_k=60)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_tie_breaking_deterministic(self):
        # A and B get same RRF score → sorted alphabetically
        rl1 = _make_ranked([("B", 10)])
        rl2 = _make_ranked([("A", 10)])
        result = rrf_fuse([rl1, rl2], rrf_k=60)
        ids = [r[0] for r in result]
        assert ids == ["A", "B"]   # A < B alphabetically

    def test_top_k_truncates(self):
        rl = _make_ranked([("A", 10), ("B", 8), ("C", 6), ("D", 4)])
        result = rrf_fuse([rl], rrf_k=60, top_k=2)
        assert len(result) == 2
        assert result[0][0] == "A"
        assert result[1][0] == "B"

    def test_rrf_k_60_formula(self):
        """Verify the formula: score = 1/(60+rank) for rank=1."""
        rl = _make_ranked([("X", 1.0)])
        result = dict(rrf_fuse([rl], rrf_k=60))
        assert result["X"] == pytest.approx(1.0 / 61.0)


# ── HybridRetriever tests ─────────────────────────────────────────────────────

class TestHybridRetriever:
    def _build(self, results_a, results_b, rrf_k=60):
        r1 = _FakeRetriever(_make_ranked(results_a))
        r2 = _FakeRetriever(_make_ranked(results_b))
        return HybridRetriever([r1, r2], rrf_k=rrf_k, candidate_depth=10)

    def test_raises_on_empty_retrievers(self):
        with pytest.raises(ValueError, match="at least one"):
            HybridRetriever([])

    def test_search_returns_retrieval_results(self):
        h = self._build([("A", 1), ("B", 2)], [("C", 1), ("A", 2)])
        results = h.search("heart failure", top_k=3)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_ranks_1_based_sequential(self):
        h = self._build([("A", 5), ("B", 4), ("C", 3)],
                         [("B", 5), ("A", 4), ("D", 3)])
        results = h.search("query", top_k=4)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_rrf_scores_positive(self):
        h = self._build([("A", 5), ("B", 4)], [("B", 5), ("C", 4)])
        results = h.search("test query", top_k=3)
        for r in results:
            assert r.score > 0.0

    def test_top_k_respected(self):
        h = self._build(
            [("A", 9), ("B", 8), ("C", 7), ("D", 6)],
            [("E", 9), ("F", 8), ("G", 7), ("H", 6)],
        )
        results = h.search("query", top_k=3)
        assert len(results) <= 3

    def test_empty_query_returns_empty(self):
        h = self._build([("A", 1)], [("B", 1)])
        assert h.search("") == []
        assert h.search("   ") == []

    def test_repeated_query_deterministic(self):
        h = self._build([("A", 9), ("B", 7), ("C", 5)],
                         [("B", 9), ("A", 7), ("D", 5)])
        q = "atrial fibrillation risk factors"
        r1 = [(r.chunk_id, r.score) for r in h.search(q, top_k=5)]
        r2 = [(r.chunk_id, r.score) for r in h.search(q, top_k=5)]
        assert r1 == r2

    def test_constituent_ranks_in_metadata(self):
        h = self._build([("A", 5), ("B", 4)], [("A", 5), ("C", 4)])
        results = h.search("query", top_k=3)
        for r in results:
            assert "constituent_ranks" in r.metadata
            cr = r.metadata["constituent_ranks"]
            assert "retriever_0" in cr
            assert "retriever_1" in cr

    def test_chunk_in_both_retrievers_scores_higher(self):
        """Chunk appearing in both ranked lists accumulates two RRF contributions."""
        h = self._build([("SHARED", 9), ("ONLY_A", 7)],
                         [("SHARED", 9), ("ONLY_B", 7)])
        results = h.search("query", top_k=5)
        shared_score = next(r.score for r in results if r.chunk_id == "SHARED")
        only_a_score = next(r.score for r in results if r.chunk_id == "ONLY_A")
        only_b_score = next(r.score for r in results if r.chunk_id == "ONLY_B")
        assert shared_score > only_a_score
        assert shared_score > only_b_score

    def test_zero_candidate_retriever_handled_gracefully(self):
        """A retriever returning 0 candidates should not break fusion."""
        r1 = _FakeRetriever(_make_ranked([("A", 5), ("B", 4)]))
        r2 = _FakeRetriever([])   # zero candidates (like LSH for some queries)
        h  = HybridRetriever([r1, r2], rrf_k=60, candidate_depth=10)
        results = h.search("query about heart failure", top_k=5)
        assert len(results) == 2  # only r1's candidates
        assert results[0].chunk_id == "A"

    def test_n_retrievers_property(self):
        r1 = _FakeRetriever([])
        r2 = _FakeRetriever([])
        h  = HybridRetriever([r1, r2])
        assert h.n_retrievers == 2

    def test_retrieve_alias_returns_tuples(self):
        h = self._build([("A", 5), ("B", 4)], [("B", 5), ("C", 4)])
        results = h.retrieve("heart failure", top_k=3)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_is_subclass_of_base_retriever(self):
        assert issubclass(HybridRetriever, BaseRetriever)

    @pytest.mark.skipif(not INDICES_OK, reason="Real indices not available")
    def test_bm25_dense_fusion_with_real_indices(self):
        from src.retrieval.bm25  import BM25Retriever
        from src.retrieval.dense import DenseRetriever
        bm25  = BM25Retriever(index_path=str(BM25_IDX.parent));  bm25.load()
        dense = DenseRetriever(index_path=str(DENSE_IDX.parent)); dense.load()
        h = HybridRetriever([bm25, dense], rrf_k=60, candidate_depth=20)
        results = h.search("pharmacological treatments for heart failure", top_k=10)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Scores should be positive (RRF)
        assert all(r.score > 0 for r in results)
        # Ranks sequential
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    @pytest.mark.skipif(not INDICES_OK, reason="Real indices not available")
    def test_bm25_dense_lsh_fusion_with_real_indices(self):
        from src.retrieval.bm25  import BM25Retriever
        from src.retrieval.lsh   import LSHRetriever
        from src.retrieval.dense import DenseRetriever
        bm25  = BM25Retriever(index_path=str(BM25_IDX.parent));  bm25.load()
        lsh   = LSHRetriever(index_path=str(LSH_IDX.parent));    lsh.load()
        dense = DenseRetriever(index_path=str(DENSE_IDX.parent)); dense.load()
        h = HybridRetriever([bm25, lsh, dense], rrf_k=60, candidate_depth=20)
        results = h.search("atrial fibrillation risk factors catheter ablation", top_k=10)
        assert len(results) > 0
        assert all(r.score > 0 for r in results)
