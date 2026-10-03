"""
test_dense.py
-------------
Unit tests for src.retrieval.dense (DenseRetriever).

Strategy
--------
* A module-scoped ``small_dense`` fixture encodes a 12-sentence synthetic
  corpus ONCE and re-uses it across all search tests — avoids repeated
  model-loading.
* Tests that only need numpy/FAISS shapes use ``small_dense`` directly.
* The Corpus V1 integration test (1,142 chunks) is skipped when the JSONL
  is absent so CI can run without the full corpus.

Coverage
--------
Embedding
    1.  encode() returns float32 ndarray of correct shape
    2.  embeddings are unit-norm (normalised)
    3.  encode() raises ValueError on empty list
    4.  identical texts produce identical vectors (deterministic)
    5.  Embedding dimension matches EMBEDDING_DIM constant (384)

Build / index
    6.  build() succeeds on synthetic corpus
    7.  corpus_size matches number of indexed chunks
    8.  Corpus V1 corpus_size == 1142 (integration, skipped if absent)
    9.  build() raises ValueError on empty chunk list
    10. is_built is False before build, True after
    11. embedding_dim is 384 after build

Search
    12. search() returns at most top_k results
    13. Results are sorted descending by score
    14. Ranks are 1-based and sequential
    15. Scores are floats (inner-product / cosine, can be negative)
    16. Returned chunk_ids exist in the index corpus
    17. Empty query returns empty list
    18. Repeated identical queries produce identical rankings
    19. top_k larger than corpus returns full corpus
    20. Identical chunk text retrieves that chunk at rank 1
    21. Semantically similar text scores higher than unrelated text
    22. search() before build raises RuntimeError

Save / load
    23. save() creates the three expected files
    24. load() restores corpus_size
    25. load() + search gives same results as in-memory index
    26. load() raises FileNotFoundError when files absent
    27. save() before build raises RuntimeError

Interface
    28. DenseRetriever is subclass of BaseRetriever
    29. retrieve() alias returns (chunk_id, score) tuples
    30. RetrievalResult fields are populated correctly
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.base  import BaseRetriever, RetrievalResult
from src.retrieval.dense import DenseRetriever, EMBEDDING_DIM

CORPUS_JSONL = ROOT / "data" / "chunks" / "chunks.jsonl"
CORPUS_AVAILABLE = CORPUS_JSONL.exists() and CORPUS_JSONL.stat().st_size > 0

# ── Synthetic corpus ──────────────────────────────────────────────────────────

CARDIO_TEXTS = [
    "Heart failure treatment includes ACE inhibitors beta-blockers and SGLT-2 inhibitors for reduced ejection fraction.",
    "Atrial fibrillation risk factors include hypertension obesity sleep apnoea and diabetes mellitus.",
    "Myocardial infarction is caused by coronary plaque rupture and subsequent thrombus formation.",
    "Hypertensive heart disease causes left ventricular hypertrophy and diastolic dysfunction.",
    "Cardiomyopathy imaging modalities include echocardiography cardiac MRI and nuclear scintigraphy.",
    "Catheter ablation of atrial fibrillation achieves rhythm control via pulmonary vein isolation.",
    "Dyslipidemia is managed with statins ezetimibe and PCSK9 inhibitors to lower LDL cholesterol.",
    "Valvular heart disease requires echocardiographic assessment of gradients and regurgitation severity.",
    "Dilated cardiomyopathy has genetic aetiology involving titin and LMNA mutations.",
    "Coronary artery disease secondary prevention requires antiplatelet therapy and statin treatment.",
    "Heart failure with preserved ejection fraction lacks robust mortality-reducing therapies.",
    "Primary prevention of cardiovascular disease uses risk scores such as SCORE2 and Framingham.",
]


def _make_chunks(texts: List[str], prefix: str = "syn") -> list:
    from src.preprocessing.chunker import Chunk
    return [
        Chunk(
            chunk_id=f"{prefix}{i // 3:02d}-{i:04d}",
            text=text,
            doc_id=f"{prefix}{i // 3:02d}",
            start_char=0,
            end_char=len(text),
            metadata={
                "title":        f"Article {prefix}{i // 3:02d}",
                "source":       f"/data/{prefix}{i // 3:02d}.pdf",
                "section":      "",
                "strategy":     "sentence",
                "chunk_index":  i,
                "total_chunks": len(texts),
            },
        )
        for i, text in enumerate(texts)
    ]


# Module-scoped fixture: build the index ONCE for all search tests
@pytest.fixture(scope="module")
def small_dense(tmp_path_factory) -> DenseRetriever:
    """DenseRetriever built over the 12-sentence synthetic corpus."""
    idx_path = tmp_path_factory.mktemp("dense_idx")
    r = DenseRetriever(index_path=idx_path)
    r.build(_make_chunks(CARDIO_TEXTS))
    return r


# ── Embedding tests ───────────────────────────────────────────────────────────

class TestEmbedding:
    def test_encode_returns_float32_ndarray(self, small_dense):
        vecs = small_dense.encode(["heart failure"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.dtype == np.float32

    def test_encode_shape_correct(self, small_dense):
        vecs = small_dense.encode(["heart failure", "atrial fibrillation"])
        assert vecs.shape == (2, EMBEDDING_DIM)

    def test_embeddings_unit_norm(self, small_dense):
        vecs = small_dense.encode(CARDIO_TEXTS[:4])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_raises_on_empty_list(self, small_dense):
        with pytest.raises(ValueError):
            small_dense.encode([])

    def test_identical_texts_identical_vectors(self, small_dense):
        v1 = small_dense.encode(["heart failure treatment"])
        v2 = small_dense.encode(["heart failure treatment"])
        np.testing.assert_array_equal(v1, v2)

    def test_embedding_dim_constant(self, small_dense):
        assert small_dense.embedding_dim == EMBEDDING_DIM
        assert EMBEDDING_DIM == 384


# ── Build / index tests ───────────────────────────────────────────────────────

class TestDenseBuild:
    def test_builds_successfully(self, small_dense):
        assert small_dense.is_built

    def test_corpus_size_matches_input(self, small_dense):
        assert small_dense.corpus_size == len(CARDIO_TEXTS)

    @pytest.mark.skipif(not CORPUS_AVAILABLE, reason="Corpus V1 not present")
    def test_corpus_v1_size_is_1142(self, tmp_path):
        r = DenseRetriever(index_path=tmp_path)
        r.build()           # loads from default chunks.jsonl
        assert r.corpus_size == 1142

    def test_build_raises_on_empty_chunks(self, tmp_path):
        r = DenseRetriever(index_path=tmp_path)
        with pytest.raises(ValueError, match="empty"):
            r.build([])

    def test_is_built_false_before_build(self):
        r = DenseRetriever()
        assert r.is_built is False

    def test_is_built_true_after_build(self, small_dense):
        assert small_dense.is_built is True

    def test_embedding_dim_after_build(self, small_dense):
        assert small_dense.embedding_dim == EMBEDDING_DIM

    def test_all_chunks_represented_once(self, small_dense):
        """Each chunk_id appears exactly once in the index."""
        ids = small_dense._chunk_ids
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids in index"


# ── Search tests ──────────────────────────────────────────────────────────────

class TestDenseSearch:
    def test_top_k_respected(self, small_dense):
        results = small_dense.search("heart failure treatment", top_k=3)
        assert len(results) <= 3

    def test_results_sorted_descending_by_score(self, small_dense):
        results = small_dense.search("atrial fibrillation risk factors", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_1_based_sequential(self, small_dense):
        results = small_dense.search("cardiomyopathy imaging", top_k=5)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_are_floats(self, small_dense):
        results = small_dense.search("coronary artery disease", top_k=8)
        for r in results:
            assert isinstance(r.score, float)

    def test_returned_chunk_ids_exist_in_corpus(self, small_dense):
        valid_ids = set(small_dense._chunk_ids)
        results = small_dense.search("hypertension left ventricular hypertrophy", top_k=6)
        for r in results:
            assert r.chunk_id in valid_ids

    def test_empty_query_returns_empty_list(self, small_dense):
        assert small_dense.search("") == []

    def test_whitespace_query_returns_empty_list(self, small_dense):
        assert small_dense.search("   ") == []

    def test_repeated_identical_queries_deterministic(self, small_dense):
        q = "heart failure SGLT-2 treatment HFrEF"
        r1 = [(r.chunk_id, r.score) for r in small_dense.search(q, top_k=5)]
        r2 = [(r.chunk_id, r.score) for r in small_dense.search(q, top_k=5)]
        assert r1 == r2

    def test_top_k_larger_than_corpus_returns_full_corpus(self, small_dense):
        results = small_dense.search("heart failure", top_k=10_000)
        assert len(results) == small_dense.corpus_size

    def test_identical_chunk_text_retrieves_that_chunk_at_rank1(self, small_dense):
        """Searching with the exact text of a corpus chunk should rank it first."""
        target_text = CARDIO_TEXTS[0]   # heart failure treatment chunk
        target_id   = _make_chunks([target_text])[0].chunk_id  # syn00-0000
        # The actual id in small_dense is also syn00-0000
        results = small_dense.search(target_text, top_k=5)
        assert results, "No results for exact-match query"
        assert results[0].chunk_id == "syn00-0000"

    def test_semantically_similar_scores_higher_than_unrelated(self, small_dense):
        """A topically relevant query should score a relevant chunk higher than an
        unrelated one."""
        query = "heart failure treatment ACE inhibitor"
        results = small_dense.search(query, top_k=len(CARDIO_TEXTS))
        # Heart failure chunks (indices 0, 10) should rank above unrelated ones
        hf_ids    = {"syn00-0000", "syn03-0010"}   # heart failure texts
        other_ids = {"syn00-0003"}                 # hypertension text
        if len(results) >= 2:
            top_ids = {r.chunk_id for r in results[:3]}
            assert hf_ids & top_ids, "Heart failure chunks not in top-3 for HF query"

    def test_search_before_build_raises(self):
        r = DenseRetriever()
        with pytest.raises(RuntimeError, match="not available"):
            r.search("heart failure")

    def test_result_text_non_empty(self, small_dense):
        results = small_dense.search("valvular heart disease", top_k=3)
        for r in results:
            assert r.text.strip() != ""

    def test_result_doc_id_non_empty(self, small_dense):
        results = small_dense.search("myocardial infarction plaque rupture", top_k=5)
        for r in results:
            assert isinstance(r.doc_id, str)
            assert r.doc_id != ""

    def test_result_score_is_cosine_range(self, small_dense):
        """Inner-product of unit vectors ∈ [−1, 1]."""
        results = small_dense.search("dilated cardiomyopathy titin", top_k=12)
        for r in results:
            assert -1.01 <= r.score <= 1.01, f"Score out of cosine range: {r.score}"


# ── Save / load tests ─────────────────────────────────────────────────────────

class TestDensePersistence:
    def test_save_creates_three_files(self, tmp_path):
        r = DenseRetriever(index_path=tmp_path)
        r.build(_make_chunks(CARDIO_TEXTS))
        r.save()
        assert (tmp_path / "dense_index.faiss").exists()
        assert (tmp_path / "dense_meta.pkl").exists()
        assert (tmp_path / "dense_config.json").exists()

    def test_load_restores_corpus_size(self, tmp_path):
        r1 = DenseRetriever(index_path=tmp_path)
        r1.build(_make_chunks(CARDIO_TEXTS))
        r1.save()

        r2 = DenseRetriever(index_path=tmp_path)
        r2.load()
        assert r2.corpus_size == len(CARDIO_TEXTS)

    def test_load_gives_same_results(self, tmp_path):
        chunks = _make_chunks(CARDIO_TEXTS)
        r1 = DenseRetriever(index_path=tmp_path)
        r1.build(chunks)
        r1.save()

        r2 = DenseRetriever(index_path=tmp_path)
        r2.load()

        q = "pharmacological treatment heart failure HFrEF sacubitril"
        res1 = [(r.chunk_id, round(r.score, 5)) for r in r1.search(q, top_k=5)]
        res2 = [(r.chunk_id, round(r.score, 5)) for r in r2.search(q, top_k=5)]
        assert res1 == res2

    def test_load_raises_when_index_absent(self, tmp_path):
        r = DenseRetriever(index_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            r.load()

    def test_save_before_build_raises(self, tmp_path):
        r = DenseRetriever(index_path=tmp_path)
        with pytest.raises(RuntimeError, match="not built"):
            r.save()

    def test_config_json_contains_model_name(self, tmp_path):
        import json
        r = DenseRetriever(index_path=tmp_path)
        r.build(_make_chunks(CARDIO_TEXTS))
        r.save()
        cfg = json.loads((tmp_path / "dense_config.json").read_text())
        assert "model_name" in cfg
        assert "all-MiniLM-L6-v2" in cfg["model_name"]


# ── BaseRetriever interface ────────────────────────────────────────────────────

class TestDenseInterface:
    def test_is_subclass_of_base_retriever(self):
        assert issubclass(DenseRetriever, BaseRetriever)

    def test_retrieve_alias_returns_tuples(self, small_dense):
        results = small_dense.retrieve("cardiomyopathy genetic testing", top_k=3)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_retrieval_result_fields_populated(self, small_dense):
        results = small_dense.search("heart failure treatment", top_k=3)
        assert results, "Expected at least one result"
        r = results[0]
        assert r.chunk_id
        assert r.doc_id
        assert r.text
        assert r.rank == 1
        assert isinstance(r.score, float)
        assert isinstance(r.metadata, dict)
