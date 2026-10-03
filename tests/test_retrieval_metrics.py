"""
test_retrieval_metrics.py
-------------------------
Unit tests for src.evaluation.retrieval_metrics.

Coverage
--------
precision_at_k     — perfect, partial, zero, edge cases
recall_at_k        — perfect, partial, zero, empty relevant
reciprocal_rank    — first pos, mid pos, last pos, not found
mean_reciprocal_rank — multiple queries, empty input, length mismatch
average_precision  — standard example, all relevant, none relevant
mean_average_precision — multiple queries, empty input
ndcg_at_k          — binary mode, graded mode, perfect ranking, zero DCG
input validation   — k <= 0, mismatched list lengths
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RETRIEVED     = ["a", "b", "c", "d", "e"]
RELEVANT_2    = {"a", "c"}          # 2 of 5 retrieved are relevant
RELEVANT_ALL  = {"a", "b", "c", "d", "e"}
RELEVANT_NONE = {"z", "y"}          # none of retrieved are relevant


# ── precision_at_k ────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a","b"], {"a","b"}, k=2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x","y"], {"a","b"}, k=2) == 0.0

    def test_partial_precision(self):
        # 2 relevant out of 4
        assert precision_at_k(RETRIEVED, RELEVANT_2, k=4) == pytest.approx(0.5)

    def test_precision_at_k1(self):
        assert precision_at_k(["a","b","c"], {"a"}, k=1) == 1.0

    def test_precision_first_not_relevant(self):
        assert precision_at_k(["x","a","b"], {"a"}, k=1) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        # k=10 but only 3 retrieved — denominator is still k
        result = precision_at_k(["a","b","c"], {"a","c"}, k=10)
        assert result == pytest.approx(2 / 10)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, k=0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, k=-1)


# ── recall_at_k ───────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a","b","c"], {"a","b","c"}, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x","y"], {"a","b"}, k=2) == 0.0

    def test_partial_recall(self):
        # 2 relevant retrieved out of 3 total relevant
        assert recall_at_k(RETRIEVED, {"a","c","z"}, k=5) == pytest.approx(2/3)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a","b"], set(), k=2) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], {"a"}, k=5) == 0.0

    def test_recall_at_k1_hit(self):
        assert recall_at_k(["a","b"], {"a"}, k=1) == 1.0

    def test_recall_at_k1_miss(self):
        assert recall_at_k(["x","a"], {"a"}, k=1) == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], {"a"}, k=0)


# ── reciprocal_rank ───────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_relevant_at_rank_1(self):
        assert reciprocal_rank(["a","b","c"], {"a"}) == pytest.approx(1.0)

    def test_relevant_at_rank_2(self):
        assert reciprocal_rank(["x","a","b"], {"a"}) == pytest.approx(0.5)

    def test_relevant_at_rank_3(self):
        assert reciprocal_rank(["x","y","a"], {"a"}) == pytest.approx(1/3)

    def test_no_relevant_returns_zero(self):
        assert reciprocal_rank(["x","y","z"], {"a"}) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_multiple_relevant_uses_first_hit(self):
        # "b" is at rank 2, "c" at rank 3; answer should be 1/2
        assert reciprocal_rank(["x","b","c"], {"b","c"}) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        assert reciprocal_rank(["a","b"], set()) == 0.0


# ── mean_reciprocal_rank ──────────────────────────────────────────────────────

class TestMRR:
    def test_basic_mrr(self):
        retrieved = [["a","b"], ["x","a"]]
        relevant  = [{"a"},     {"a"}]
        # RR = 1.0 and 0.5 → MRR = 0.75
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.75)

    def test_all_miss(self):
        retrieved = [["x","y"], ["p","q"]]
        relevant  = [{"a"},      {"b"}]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_single_query(self):
        assert mean_reciprocal_rank([["a","b"]], [{"b"}]) == pytest.approx(0.5)

    def test_empty_input_returns_zero(self):
        assert mean_reciprocal_rank([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])


# ── average_precision ─────────────────────────────────────────────────────────

class TestAP:
    def test_ap_standard_example(self):
        # retrieved = [a, b, c, d], relevant = {a, c}
        # P@1=1.0 (a hit), P@3=2/3 (c hit) → AP = (1.0 + 2/3) / 2
        result = average_precision(["a","b","c","d"], {"a","c"})
        assert result == pytest.approx((1.0 + 2/3) / 2)

    def test_all_relevant(self):
        assert average_precision(["a","b","c"], {"a","b","c"}) == pytest.approx(1.0)

    def test_none_relevant_in_retrieved(self):
        assert average_precision(["x","y"], {"a","b"}) == 0.0

    def test_empty_retrieved(self):
        assert average_precision([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert average_precision(["a","b"], set()) == 0.0

    def test_single_relevant_at_rank_1(self):
        assert average_precision(["a","x","y"], {"a"}) == pytest.approx(1.0)

    def test_single_relevant_at_rank_3(self):
        # P@3 = 1/3, only 1 relevant → AP = (1/3) / 1 = 1/3
        assert average_precision(["x","y","a"], {"a"}) == pytest.approx(1/3)


# ── mean_average_precision ────────────────────────────────────────────────────

class TestMAP:
    def test_basic_map(self):
        all_ret = [["a","b","c"], ["x","a","b"]]
        all_rel = [{"a","c"},     {"a"}]
        ap1 = average_precision(["a","b","c"], {"a","c"})
        ap2 = average_precision(["x","a","b"], {"a"})
        expected = (ap1 + ap2) / 2
        assert mean_average_precision(all_ret, all_rel) == pytest.approx(expected)

    def test_empty_input_returns_zero(self):
        assert mean_average_precision([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mean_average_precision([["a"]], [{"a"}, {"b"}])

    def test_all_miss_returns_zero(self):
        assert mean_average_precision([["x"],["y"]], [{"a"},{"b"}]) == 0.0


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────

class TestNDCG:
    def test_perfect_binary_ranking(self):
        # Both relevant at ranks 1 and 2 → NDCG = 1.0
        assert ndcg_at_k(["a","b","x"], {"a","b"}, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        # No relevant items retrieved
        assert ndcg_at_k(["x","y","z"], {"a","b"}, k=3) == 0.0

    def test_partial_binary(self):
        # relevant = {a, c}; retrieved = [a, b, c, d]
        # DCG@4 = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG@4 = 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.631 = 1.631
        result = ndcg_at_k(["a","b","c","d"], {"a","c"}, k=4)
        dcg  = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert result == pytest.approx(dcg / idcg)

    def test_graded_relevance_used(self):
        grades = {"a": 2, "b": 1, "c": 0}
        # Retrieved = [a, b, c] → DCG = 2/log2(2) + 1/log2(3)
        result = ndcg_at_k(["a","b","c"], set(), k=3, graded_relevance=grades)
        dcg  = 2.0/math.log2(2) + 1.0/math.log2(3)
        idcg = 2.0/math.log2(2) + 1.0/math.log2(3)
        assert result == pytest.approx(dcg / idcg)

    def test_graded_suboptimal_ranking(self):
        grades = {"a": 2, "b": 1}
        # Suboptimal: [b, a] instead of [a, b]
        sub  = ndcg_at_k(["b","a"], set(), k=2, graded_relevance=grades)
        opt  = ndcg_at_k(["a","b"], set(), k=2, graded_relevance=grades)
        assert sub < opt
        assert opt == pytest.approx(1.0)

    def test_empty_retrieved_returns_zero(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k(["a","b"], set(), k=2) == 0.0

    def test_k_truncates_to_top_k(self):
        # Relevant item at rank 5, k=3 → should not be counted
        result = ndcg_at_k(["x","y","z","w","a"], {"a"}, k=3)
        assert result == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], {"a"}, k=0)
