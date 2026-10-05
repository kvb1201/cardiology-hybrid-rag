"""
test_reference_evidence.py
--------------------------
Unit and integration tests for src.evaluation.reference_evidence.

Coverage:
1. Every query appears (all 65 queries).
2. Exactly 59 queries have grade-2 evidence.
3. Six expected queries have grade-2-empty status (Q007, Q013, Q017, Q021, Q055, Q061).
4. Fallback uses only grade-1 chunks.
5. Primary queries use only grade-2 chunks.
6. No unknown chunk IDs (all resolve to frozen chunks.jsonl).
7. Chunk text exactly matches frozen chunks.jsonl.
8. Deterministic output (repeated builds yield identical data).
9. No duplicate chunk IDs within any query's reference set.
10. Collection reader interface (lookup, iteration, properties).
11. No forbidden clinical ground-truth terminology.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reference_evidence import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_QRELS_PATH,
    DEFAULT_QUERIES_PATH,
    GRADE_1_FALLBACK_QUERY_IDS,
    ReferenceEvidenceCollection,
    build_reference_evidence,
    load_chunks_lookup,
    load_reference_evidence,
)


@pytest.fixture(scope="module")
def reference_data():
    """Load or build reference evidence data for testing."""
    if DEFAULT_OUTPUT_PATH.exists():
        with open(DEFAULT_OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_reference_evidence()


@pytest.fixture(scope="module")
def corpus_chunks():
    """Load corpus chunks lookup."""
    return load_chunks_lookup(DEFAULT_CHUNKS_PATH)


class TestReferenceEvidenceIntegrity:
    """Test structural integrity and correctness of the reference evidence."""

    def test_every_query_appears(self, reference_data):
        queries = reference_data["queries"]
        assert len(queries) == 65
        query_ids = [q["query_id"] for q in queries]
        expected_ids = [f"Q{i:03d}" for i in range(1, 66)]
        assert sorted(query_ids) == expected_ids

    def test_grade_2_available_count(self, reference_data):
        queries = reference_data["queries"]
        g2_queries = [q for q in queries if q["grade_2_available"]]
        assert len(g2_queries) == 59
        for q in g2_queries:
            assert q["grade_1_fallback"] is False
            assert q["n_evidence_chunks"] > 0
            assert len(q["reference_evidence"]) == q["n_evidence_chunks"]

    def test_six_expected_fallback_queries(self, reference_data):
        queries = reference_data["queries"]
        fallback_queries = [q for q in queries if q["grade_1_fallback"]]
        assert len(fallback_queries) == 6
        fallback_ids = {q["query_id"] for q in fallback_queries}
        assert fallback_ids == GRADE_1_FALLBACK_QUERY_IDS
        for q in fallback_queries:
            assert q["grade_2_available"] is False
            assert q["n_evidence_chunks"] > 0
            assert len(q["reference_evidence"]) == q["n_evidence_chunks"]

    def test_fallback_chunks_grade_1_only(self, reference_data):
        queries = reference_data["queries"]
        fallback_queries = [q for q in queries if q["query_id"] in GRADE_1_FALLBACK_QUERY_IDS]
        for q in fallback_queries:
            for chunk in q["reference_evidence"]:
                assert chunk["relevance"] == 1, (
                    f"Fallback query {q['query_id']} contains non-grade-1 chunk {chunk['chunk_id']} "
                    f"with grade {chunk['relevance']}"
                )

    def test_primary_chunks_grade_2_only(self, reference_data):
        queries = reference_data["queries"]
        primary_queries = [q for q in queries if q["query_id"] not in GRADE_1_FALLBACK_QUERY_IDS]
        for q in primary_queries:
            for chunk in q["reference_evidence"]:
                assert chunk["relevance"] == 2, (
                    f"Primary query {q['query_id']} contains non-grade-2 chunk {chunk['chunk_id']} "
                    f"with grade {chunk['relevance']}"
                )

    def test_no_unknown_chunk_ids(self, reference_data, corpus_chunks):
        queries = reference_data["queries"]
        for q in queries:
            for chunk in q["reference_evidence"]:
                cid = chunk["chunk_id"]
                assert cid in corpus_chunks, f"Query {q['query_id']} references unknown chunk: {cid}"

    def test_chunk_text_matches_frozen_corpus(self, reference_data, corpus_chunks):
        queries = reference_data["queries"]
        for q in queries:
            for chunk in q["reference_evidence"]:
                cid = chunk["chunk_id"]
                expected_text = corpus_chunks[cid]["text"]
                assert chunk["text"] == expected_text, f"Text mismatch for chunk {cid}"
                assert chunk["doc_id"] == corpus_chunks[cid]["doc_id"]

    def test_no_duplicate_chunk_ids_in_reference_set(self, reference_data):
        queries = reference_data["queries"]
        for q in queries:
            cids = [chunk["chunk_id"] for chunk in q["reference_evidence"]]
            assert len(cids) == len(set(cids)), f"Duplicate chunk IDs in query {q['query_id']}"

    def test_deterministic_output(self):
        run1 = build_reference_evidence()
        run2 = build_reference_evidence()
        # Drop created_at timestamps before comparison
        run1_clean = {k: v for k, v in run1.items() if k != "metadata"}
        run2_clean = {k: v for k, v in run2.items() if k != "metadata"}
        assert json.dumps(run1_clean, sort_keys=True) == json.dumps(run2_clean, sort_keys=True)

    def test_no_forbidden_ground_truth_terminology(self, reference_data):
        text = json.dumps(reference_data).lower()
        forbidden_phrases = [
            "gold answer",
            "ground-truth answer",
            "ground truth answer",
            "clinical truth",
        ]
        for phrase in forbidden_phrases:
            assert phrase not in text, f"Forbidden phrase '{phrase}' detected in reference evidence!"

    def test_collection_reader_interface(self):
        coll = load_reference_evidence(DEFAULT_OUTPUT_PATH)
        assert len(coll) == 65
        assert len(coll.grade_2_queries) == 59
        assert len(coll.fallback_queries) == 6

        q1 = coll["Q001"]
        assert q1["query_id"] == "Q001"
        assert q1["grade_2_available"] is True

        q7 = coll["Q007"]
        assert q7["query_id"] == "Q007"
        assert q7["grade_1_fallback"] is True

        assert coll.get("NONEXISTENT") is None
        assert coll.get("NONEXISTENT", "default") == "default"
