"""
test_lsh.py
-----------
Unit tests for src.retrieval.lsh (shingle function + LSHRetriever).

Coverage
--------
shingle()
    1.  Empty string returns empty frozenset
    2.  Text shorter than k returns empty frozenset
    3.  Text with exactly k tokens returns one shingle
    4.  Lowercasing applied consistently
    5.  Clinical abbreviations preserved (SGLT-2, HFrEF)
    6.  Pure numbers filtered (same as BM25 tokenizer)
    7.  Deterministic (same input → same output)
    8.  k=1 produces unigram shingles
    9.  k=0 raises ValueError

LSHRetriever — build / index
    10. Index builds successfully from synthetic corpus
    11. corpus_size equals number of indexed chunks
    12. Corpus V1 index size == 1142 (integration, skipped if absent)
    13. build() raises ValueError on empty chunk list
    14. is_built is False before build, True after

LSHRetriever — search
    15. Identical text returns Jaccard == 1.0 as top result
    16. Disjoint shingle sets return empty list (score → 0, not returned unless LSH returns candidate)
    17. top_k respected — returns at most top_k results
    18. Results sorted descending by Jaccard
    19. Ranks are 1-based and sequential
    20. Scores are floats in [0, 1]
    21. Empty query returns empty list
    22. Repeated identical queries give identical results
    23. Returned chunk_ids exist in the corpus
    24. search() before build raises RuntimeError
    25. top_k larger than candidates returns available (fewer than top_k OK)

LSHRetriever — save / load
    26. save() creates pickle file
    27. load() + search gives same results as in-memory
    28. load() raises FileNotFoundError when index absent
    29. save() before build raises RuntimeError

BaseRetriever interface
    30. LSHRetriever is a subclass of BaseRetriever
    31. retrieve() alias returns (chunk_id, score) tuples

Functional sanity
    32. Cardiology query returns non-empty results
    33. Jaccard score is 1.0 for exact-duplicate chunk
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.lsh import LSHRetriever, shingle, _jaccard
from src.preprocessing.chunker import Chunk

CORPUS_JSONL = ROOT / "data" / "chunks" / "chunks.jsonl"
CORPUS_AVAILABLE = CORPUS_JSONL.exists() and CORPUS_JSONL.stat().st_size > 0

# ── Synthetic test corpus ──────────────────────────────────────────────────────

CARDIO_TEXTS = [
    "Heart failure treatment includes ACE inhibitors beta-blockers and SGLT-2 inhibitors for HFrEF patients.",
    "Atrial fibrillation risk factors include hypertension obesity sleep apnoea and diabetes mellitus.",
    "Myocardial infarction is caused by coronary plaque rupture and subsequent thrombus formation.",
    "Hypertensive heart disease causes left ventricular hypertrophy and diastolic dysfunction.",
    "Cardiomyopathy imaging modalities include echocardiography cardiac MRI and nuclear scintigraphy.",
    "BM25 retrieval ranks documents by term frequency and inverse document frequency.",
    "Catheter ablation of atrial fibrillation achieves rhythm control via pulmonary vein isolation.",
    "Dyslipidemia is managed with statins ezetimibe and PCSK9 inhibitors to lower LDL cholesterol.",
    "Valvular heart disease requires echocardiographic assessment of gradients and regurgitation.",
    "Dilated cardiomyopathy has genetic aetiology involving titin and LMNA mutations.",
    "Coronary artery disease secondary prevention requires antiplatelet therapy and statin treatment.",
    "Heart failure with preserved ejection fraction HFpEF lacks robust mortality-reducing therapies.",
]


def _make_chunks(texts: List[str], prefix: str = "syn") -> List[Chunk]:
    chunks = []
    for i, text in enumerate(texts):
        doc_id = f"{prefix}{i // 3:02d}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{i:04d}",
                text=text,
                doc_id=doc_id,
                start_char=0,
                end_char=len(text),
                metadata={
                    "title":        f"Article {doc_id}",
                    "source":       f"/data/{doc_id}.pdf",
                    "section":      "",
                    "strategy":     "sentence",
                    "chunk_index":  i,
                    "total_chunks": len(texts),
                },
            )
        )
    return chunks


@pytest.fixture(scope="module")
def small_lsh() -> LSHRetriever:
    r = LSHRetriever(shingle_size=3, num_perm=64, threshold=0.1)
    r.build(_make_chunks(CARDIO_TEXTS))
    return r


# ── shingle() tests ────────────────────────────────────────────────────────────

class TestShingle:
    def test_empty_string(self):
        assert shingle("", k=3) == frozenset()

    def test_text_shorter_than_k(self):
        assert shingle("heart failure", k=3) == frozenset()  # 2 tokens, k=3

    def test_exactly_k_tokens_gives_one_shingle(self):
        result = shingle("heart failure treatment", k=3)
        assert result == frozenset({("heart", "failure", "treatment")})

    def test_lowercasing_applied(self):
        r1 = shingle("Heart Failure LVEF", k=2)
        r2 = shingle("heart failure lvef", k=2)
        assert r1 == r2

    def test_clinical_abbreviations_preserved(self):
        result = shingle("SGLT-2 inhibitor HFrEF treatment", k=2)
        assert ("sglt-2", "inhibitor") in result
        assert ("hfref", "treatment") in result

    def test_pure_numbers_filtered(self):
        result = shingle("age 72 years blood pressure 140", k=2)
        for s in result:
            for tok in s:
                assert any(c.isalpha() for c in tok), \
                    f"Pure-number token survived: {tok!r}"

    def test_deterministic(self):
        text = "Atrial fibrillation SGLT-2 HFrEF diagnosis 2024"
        assert shingle(text, k=3) == shingle(text, k=3)

    def test_k1_produces_unigrams(self):
        result = shingle("heart failure", k=1)
        assert result == frozenset({("heart",), ("failure",)})

    def test_k0_raises(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            shingle("heart failure", k=0)

    def test_result_is_frozenset(self):
        result = shingle("heart failure treatment", k=2)
        assert isinstance(result, frozenset)


# ── LSHRetriever — build / index ───────────────────────────────────────────────

class TestLSHBuild:
    def test_builds_successfully(self, small_lsh):
        assert small_lsh.is_built

    def test_corpus_size_matches_input(self, small_lsh):
        assert small_lsh.corpus_size == len(CARDIO_TEXTS)

    @pytest.mark.skipif(not CORPUS_AVAILABLE, reason="Corpus V1 not available")
    def test_corpus_v1_size_is_1142(self):
        r = LSHRetriever(shingle_size=3, num_perm=64, threshold=0.1)
        r.build()
        assert r.corpus_size == 1142

    def test_build_raises_on_empty_chunks(self):
        r = LSHRetriever()
        with pytest.raises(ValueError, match="empty"):
            r.build([])

    def test_is_built_false_before_build(self):
        r = LSHRetriever()
        assert r.is_built is False

    def test_is_built_true_after_build(self, small_lsh):
        assert small_lsh.is_built is True


# ── LSHRetriever — search ──────────────────────────────────────────────────────

class TestLSHSearch:
    def test_identical_text_returns_jaccard_1(self, small_lsh):
        """Searching with the exact text of a chunk should return it with Jaccard=1."""
        target_text = CARDIO_TEXTS[0]
        results = small_lsh.search(target_text, top_k=5)
        # The exact chunk should appear in results
        top_chunk_ids = [r.chunk_id for r in results]
        assert "syn00-0000" in top_chunk_ids
        # Its score should be 1.0
        top_result = next(r for r in results if r.chunk_id == "syn00-0000")
        assert top_result.score == pytest.approx(1.0)

    def test_top_k_respected(self, small_lsh):
        results = small_lsh.search("heart failure treatment", top_k=3)
        assert len(results) <= 3

    def test_results_sorted_descending_by_jaccard(self, small_lsh):
        results = small_lsh.search("atrial fibrillation risk factors", top_k=10)
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_ranks_1_based_sequential(self, small_lsh):
        results = small_lsh.search("cardiomyopathy imaging", top_k=5)
        if results:
            assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_in_unit_interval(self, small_lsh):
        results = small_lsh.search("coronary artery disease statin", top_k=8)
        for r in results:
            assert isinstance(r.score, float)
            assert 0.0 <= r.score <= 1.0

    def test_empty_query_returns_empty(self, small_lsh):
        assert small_lsh.search("") == []

    def test_whitespace_query_returns_empty(self, small_lsh):
        assert small_lsh.search("   ") == []

    def test_repeated_identical_queries_deterministic(self, small_lsh):
        q = "heart failure HFrEF SGLT-2 management"
        r1 = [(r.chunk_id, r.score) for r in small_lsh.search(q, top_k=5)]
        r2 = [(r.chunk_id, r.score) for r in small_lsh.search(q, top_k=5)]
        assert r1 == r2

    def test_chunk_ids_exist_in_corpus(self, small_lsh):
        valid_ids = set(small_lsh._chunk_ids)
        results = small_lsh.search("hypertension left ventricular hypertrophy", top_k=6)
        for r in results:
            assert r.chunk_id in valid_ids

    def test_search_before_build_raises(self):
        r = LSHRetriever()
        with pytest.raises(RuntimeError, match="not available"):
            r.search("heart failure")

    def test_top_k_larger_than_candidates_ok(self, small_lsh):
        """top_k > candidates should return available results without error."""
        results = small_lsh.search("dilated cardiomyopathy titin mutation", top_k=10_000)
        assert len(results) <= small_lsh.corpus_size  # upper bound
        assert isinstance(results, list)

    def test_result_text_non_empty(self, small_lsh):
        results = small_lsh.search("valvular heart disease echocardiography", top_k=3)
        for r in results:
            assert r.text.strip() != ""

    def test_result_doc_id_non_empty(self, small_lsh):
        results = small_lsh.search("myocardial infarction plaque rupture", top_k=5)
        for r in results:
            assert isinstance(r.doc_id, str)
            assert r.doc_id != ""


# ── LSHRetriever — save / load ─────────────────────────────────────────────────

class TestLSHPersistence:
    def test_save_creates_pickle(self, tmp_path):
        r = LSHRetriever(index_path=tmp_path, shingle_size=3, num_perm=32, threshold=0.1)
        r.build(_make_chunks(CARDIO_TEXTS))
        r.save()
        assert (tmp_path / "lsh_index.pkl").exists()

    def test_load_gives_same_results(self, tmp_path):
        chunks = _make_chunks(CARDIO_TEXTS)
        r1 = LSHRetriever(index_path=tmp_path, shingle_size=3, num_perm=32, threshold=0.1)
        r1.build(chunks)
        r1.save()

        r2 = LSHRetriever(index_path=tmp_path)
        r2.load()

        q = "heart failure SGLT-2 treatment HFrEF"
        res1 = sorted((r.chunk_id, r.score) for r in r1.search(q, top_k=5))
        res2 = sorted((r.chunk_id, r.score) for r in r2.search(q, top_k=5))
        assert res1 == res2

    def test_load_preserves_corpus_size(self, tmp_path):
        r1 = LSHRetriever(index_path=tmp_path, shingle_size=3, num_perm=32, threshold=0.1)
        r1.build(_make_chunks(CARDIO_TEXTS))
        r1.save()

        r2 = LSHRetriever(index_path=tmp_path)
        r2.load()
        assert r2.corpus_size == len(CARDIO_TEXTS)

    def test_load_raises_when_file_absent(self, tmp_path):
        r = LSHRetriever(index_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            r.load()

    def test_save_before_build_raises(self, tmp_path):
        r = LSHRetriever(index_path=tmp_path)
        with pytest.raises(RuntimeError, match="not built"):
            r.save()


# ── BaseRetriever interface ────────────────────────────────────────────────────

class TestBaseRetrieverInterface:
    def test_is_subclass_of_base_retriever(self):
        assert issubclass(LSHRetriever, BaseRetriever)

    def test_retrieve_alias_returns_tuples(self, small_lsh):
        results = small_lsh.retrieve("cardiomyopathy genetic testing", top_k=3)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)


# ── Jaccard helper ────────────────────────────────────────────────────────────

class TestJaccardHelper:
    def test_identical_sets(self):
        s = frozenset({("heart", "failure"), ("failure", "treatment")})
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = frozenset({("heart", "failure")})
        b = frozenset({("atrial", "fibrillation")})
        assert _jaccard(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = frozenset({("a", "b"), ("b", "c")})
        b = frozenset({("b", "c"), ("c", "d")})
        # intersection = {(b,c)}, union = 3 → Jaccard = 1/3
        assert _jaccard(a, b) == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert _jaccard(frozenset(), frozenset()) == pytest.approx(1.0)

    def test_one_empty(self):
        s = frozenset({("heart", "failure")})
        assert _jaccard(s, frozenset()) == pytest.approx(0.0)
        assert _jaccard(frozenset(), s) == pytest.approx(0.0)


# ── Functional sanity ─────────────────────────────────────────────────────────

class TestFunctionalSanity:
    def test_cardiology_query_returns_results(self, small_lsh):
        """Queries must have at least shingle_size=3 words to generate shingles."""
        for query in [
            "heart failure treatment",      # 3 words → k=3 shingles exist
            "atrial fibrillation risk factors",  # 4 words
            "cardiomyopathy imaging modalities",  # 3 words
        ]:
            results = small_lsh.search(query, top_k=5)
            assert len(results) > 0, f"No results for: {query!r}"

    def test_exact_duplicate_chunk_gives_jaccard_1(self, tmp_path):
        """A chunk whose text is identical to the query must score exactly 1.0."""
        target = "Heart failure treatment includes ACE inhibitors beta-blockers SGLT-2."
        chunks = _make_chunks([target] + CARDIO_TEXTS[1:5])
        r = LSHRetriever(
            index_path=tmp_path, shingle_size=3, num_perm=64, threshold=0.05
        )
        r.build(chunks)
        results = r.search(target, top_k=3)
        assert results, "No results returned for exact duplicate query"
        assert results[0].score == pytest.approx(1.0)
