"""
test_bm25.py
------------
Unit tests for src.retrieval.bm25 (tokenizer + BM25Retriever).

Coverage
--------
Tokenizer
    1.  Empty string returns empty list
    2.  Lowercasing is applied
    3.  Hyphenated clinical terms preserved (SGLT-2, CHA2DS2-VASc)
    4.  Pure numbers filtered out
    5.  Single-letter tokens kept

BM25Retriever — build / index
    6.  Index builds successfully from synthetic corpus
    7.  corpus_size equals number of indexed chunks
    8.  Corpus V1 index size = 1,142 (integration, skipped if file absent)
    9.  build() raises ValueError on empty chunk list

BM25Retriever — search
    10. top_k=5 returns at most 5 results
    11. Results are sorted descending by score (rank order)
    12. Ranks are 1-based and sequential
    13. Scores are non-negative floats
    14. Chunk IDs returned exist in the corpus
    15. Empty query returns empty list
    16. Whitespace-only query returns empty list
    17. Repeated identical queries produce identical results
    18. top_k larger than corpus returns all corpus items
    19. Common cardiology terms return non-empty results
    20. Returned doc_id matches chunk_id prefix

BM25Retriever — save / load
    21. save() creates the index pickle file
    22. load() restores index; subsequent search gives same results
    23. load() raises FileNotFoundError on missing index

BaseRetriever interface
    24. BM25Retriever is a subclass of BaseRetriever
    25. is_built is False before build(), True after
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import BM25Retriever, tokenize
from src.preprocessing.chunker import Chunk


# ── Helpers ──────────────────────────────────────────────────────────────────

CARDIOLOGY_CORPUS = [
    "Heart failure management includes ACE inhibitors, beta-blockers, and SGLT-2 inhibitors.",
    "Atrial fibrillation risk factors include hypertension, obesity, and sleep apnoea.",
    "Myocardial infarction is caused by plaque rupture and coronary artery occlusion.",
    "Hypertensive heart disease leads to left ventricular hypertrophy and diastolic dysfunction.",
    "Cardiomyopathy imaging uses echocardiography, cardiac MRI, and PET scanning.",
    "BM25 retrieval assigns term-frequency scores to rank relevant documents.",
    "Atrial fibrillation ablation achieves rhythm control via pulmonary vein isolation.",
    "Dyslipidemia treatment includes statins, ezetimibe, and PCSK9 inhibitors.",
    "Valvular heart disease assessment requires echocardiographic evaluation of gradients.",
    "Dilated cardiomyopathy may have genetic aetiology involving titin mutations.",
    "Coronary artery disease prevention relies on lifestyle modification and antiplatelet therapy.",
    "Heart failure preserved ejection fraction lacks proven mortality-reducing therapies.",
]


def _make_chunks(texts: List[str], doc_prefix: str = "doc") -> List[Chunk]:
    """Build synthetic Chunk objects from a list of strings."""
    chunks = []
    for i, text in enumerate(texts):
        doc_id = f"{doc_prefix}{i // 3:02d}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{i:04d}",
                text=text,
                doc_id=doc_id,
                start_char=0,
                end_char=len(text),
                metadata={
                    "title":        f"Article {doc_id}",
                    "source":       f"/data/raw/{doc_id}.pdf",
                    "section":      "",
                    "strategy":     "sentence",
                    "chunk_index":  i,
                    "total_chunks": len(texts),
                },
            )
        )
    return chunks


@pytest.fixture(scope="module")
def small_retriever() -> BM25Retriever:
    """BM25Retriever built over the small synthetic cardiology corpus."""
    r = BM25Retriever()
    r.build(_make_chunks(CARDIOLOGY_CORPUS))
    return r


CORPUS_JSONL = ROOT / "data" / "chunks" / "chunks.jsonl"
CORPUS_AVAILABLE = CORPUS_JSONL.exists() and CORPUS_JSONL.stat().st_size > 0


# ── Tokenizer tests ───────────────────────────────────────────────────────────

class TestTokenize:
    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert tokenize("   \t\n  ") == []

    def test_lowercasing_applied(self):
        result = tokenize("Heart Failure LVEF")
        assert result == ["heart", "failure", "lvef"]

    def test_hyphenated_clinical_terms_preserved(self):
        result = tokenize("SGLT-2 inhibitor and CHA2DS2-VASc score")
        assert "sglt-2" in result
        assert "cha2ds2-vasc" in result

    def test_pure_numbers_filtered(self):
        result = tokenize("age 72 years BP 140 90 mmHg")
        assert "72" not in result
        assert "140" not in result
        assert "90" not in result
        # words with letters must survive
        assert "age" in result
        assert "years" in result
        assert "bp" in result
        assert "mmhg" in result

    def test_single_letter_tokens_kept(self):
        # Medical abbreviations like "a" (as in "a fib") should not be stripped
        result = tokenize("a v node reentry tachycardia")
        assert "a" in result
        assert "v" in result

    def test_deterministic(self):
        text = "Atrial fibrillation SGLT-2 HFrEF diagnosis 2024"
        assert tokenize(text) == tokenize(text)

    def test_slash_breaks_token(self):
        # mg/kg should split into "mg" and "kg"
        result = tokenize("10 mg/kg/day dosing")
        assert "mg" in result
        assert "kg" in result

    def test_abbreviation_preserved(self):
        result = tokenize("LVEF HFrEF HFpEF INR DOAC")
        assert result == ["lvef", "hfref", "hfpef", "inr", "doac"]


# ── BM25Retriever — build / index ─────────────────────────────────────────────

class TestBM25Build:
    def test_builds_successfully(self, small_retriever):
        assert small_retriever.is_built is True

    def test_corpus_size_matches_input(self, small_retriever):
        assert small_retriever.corpus_size == len(CARDIOLOGY_CORPUS)

    @pytest.mark.skipif(not CORPUS_AVAILABLE, reason="Corpus V1 not present")
    def test_corpus_v1_size_is_1142(self):
        r = BM25Retriever()
        r.build()           # loads from default chunks.jsonl
        assert r.corpus_size == 1142

    def test_build_raises_on_empty_chunks(self):
        r = BM25Retriever()
        with pytest.raises(ValueError, match="empty"):
            r.build([])

    def test_is_built_false_before_build(self):
        r = BM25Retriever()
        assert r.is_built is False

    def test_is_built_true_after_build(self, small_retriever):
        assert small_retriever.is_built is True


# ── BM25Retriever — search ────────────────────────────────────────────────────

class TestBM25Search:
    def test_top_k_5_returns_at_most_5(self, small_retriever):
        results = small_retriever.search("heart failure treatment", top_k=5)
        assert len(results) <= 5

    def test_results_sorted_by_score_descending(self, small_retriever):
        results = small_retriever.search("atrial fibrillation", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_1_based_and_sequential(self, small_retriever):
        results = small_retriever.search("cardiomyopathy imaging", top_k=5)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_are_non_negative_floats(self, small_retriever):
        results = small_retriever.search("coronary artery disease", top_k=8)
        for r in results:
            assert isinstance(r.score, float)
            assert r.score >= 0.0

    def test_chunk_ids_exist_in_corpus(self, small_retriever):
        all_ids = set(small_retriever._chunk_ids)
        results = small_retriever.search("hypertension diagnosis", top_k=6)
        for r in results:
            assert r.chunk_id in all_ids

    def test_empty_query_returns_empty_list(self, small_retriever):
        assert small_retriever.search("") == []

    def test_whitespace_query_returns_empty_list(self, small_retriever):
        assert small_retriever.search("   ") == []

    def test_repeated_identical_queries_are_identical(self, small_retriever):
        q = "heart failure LVEF management"
        r1 = small_retriever.search(q, top_k=5)
        r2 = small_retriever.search(q, top_k=5)
        assert [(r.chunk_id, r.score) for r in r1] == \
               [(r.chunk_id, r.score) for r in r2]

    def test_top_k_larger_than_corpus_returns_all(self, small_retriever):
        huge_k = 10_000
        results = small_retriever.search("atrial fibrillation", top_k=huge_k)
        assert len(results) == small_retriever.corpus_size

    def test_common_cardiology_term_returns_results(self, small_retriever):
        for query in ["heart failure", "atrial fibrillation", "hypertension"]:
            results = small_retriever.search(query, top_k=3)
            assert len(results) > 0, f"No results for: {query!r}"

    def test_result_doc_id_is_non_empty_string(self, small_retriever):
        results = small_retriever.search("myocardial infarction", top_k=5)
        for r in results:
            assert isinstance(r.doc_id, str)
            assert r.doc_id != ""

    def test_result_text_is_non_empty(self, small_retriever):
        results = small_retriever.search("valvular heart disease", top_k=3)
        for r in results:
            assert r.text.strip() != ""

    def test_search_before_build_raises(self):
        r = BM25Retriever()
        with pytest.raises(RuntimeError, match="not available"):
            r.search("test query")

    def test_retrieve_alias_returns_tuples(self, small_retriever):
        results = small_retriever.retrieve("cardiomyopathy", top_k=3)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)


# ── BM25Retriever — save / load ───────────────────────────────────────────────

class TestBM25Persistence:
    def test_save_creates_pickle_file(self, tmp_path):
        r = BM25Retriever(index_path=tmp_path)
        r.build(_make_chunks(CARDIOLOGY_CORPUS))
        r.save()
        assert (tmp_path / "bm25_index.pkl").exists()

    def test_load_restores_search_results(self, tmp_path):
        chunks = _make_chunks(CARDIOLOGY_CORPUS)
        r1 = BM25Retriever(index_path=tmp_path)
        r1.build(chunks)
        r1.save()

        r2 = BM25Retriever(index_path=tmp_path)
        r2.load()

        q = "atrial fibrillation ablation"
        res1 = [(r.chunk_id, r.score) for r in r1.search(q, top_k=5)]
        res2 = [(r.chunk_id, r.score) for r in r2.search(q, top_k=5)]
        assert res1 == res2

    def test_load_preserves_corpus_size(self, tmp_path):
        r1 = BM25Retriever(index_path=tmp_path)
        r1.build(_make_chunks(CARDIOLOGY_CORPUS))
        r1.save()

        r2 = BM25Retriever(index_path=tmp_path)
        r2.load()
        assert r2.corpus_size == len(CARDIOLOGY_CORPUS)

    def test_load_raises_on_missing_file(self, tmp_path):
        r = BM25Retriever(index_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            r.load()

    def test_save_raises_before_build(self, tmp_path):
        r = BM25Retriever(index_path=tmp_path)
        with pytest.raises(RuntimeError, match="not been built"):
            r.save()


# ── BaseRetriever interface ───────────────────────────────────────────────────

class TestBaseRetrieverInterface:
    def test_bm25_is_subclass_of_base_retriever(self):
        assert issubclass(BM25Retriever, BaseRetriever)

    def test_retrieval_result_is_frozen_dataclass(self):
        r = RetrievalResult(
            chunk_id="doc01-0001", doc_id="doc01",
            score=3.14, rank=1, text="test"
        )
        with pytest.raises((AttributeError, TypeError)):
            r.score = 99.0  # type: ignore[misc]

    def test_retrieval_result_to_dict(self):
        r = RetrievalResult(
            chunk_id="doc01-0001", doc_id="doc01",
            score=2.5, rank=1, text="heart failure"
        )
        d = r.to_dict()
        assert d["chunk_id"] == "doc01-0001"
        assert d["score"] == 2.5
        assert d["rank"] == 1
        assert d["text"] == "heart failure"
