"""
test_chunker.py
---------------
Unit tests for src.preprocessing.chunker.

Coverage
--------
* Chunk dataclass: to_dict / from_dict round-trip
* chunk_document(): all three strategies
* chunk_document(): metadata fields populated correctly
* chunk_document(): overlap semantics
* chunk_document(): edge cases (empty, single token, invalid args)
* chunk_corpus(): batch function
* save_chunks() / load_chunks(): JSONL round-trip
* Section detection via Markdown headings
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.preprocessing.chunker import (
    Chunk,
    chunk_document,
    chunk_corpus,
    save_chunks,
    load_chunks,
)
from tests.conftest import CLEAN_CARDIOLOGY_TEXT


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

class TestChunkDataclass:
    def test_to_dict_contains_all_fields(self):
        c = Chunk(
            chunk_id="doc001-0000",
            text="Atrial fibrillation (AF) is common.",
            doc_id="doc001",
            start_char=0,
            end_char=35,
            metadata={
                "title": "AF Guide",
                "source": "/data/raw/af.txt",
                "section": "Introduction",
                "strategy": "sentence",
                "chunk_index": 0,
                "total_chunks": 5,
            },
        )
        d = c.to_dict()
        assert d["chunk_id"] == "doc001-0000"
        assert d["text"] == "Atrial fibrillation (AF) is common."
        assert d["doc_id"] == "doc001"
        assert d["start_char"] == 0
        assert d["end_char"] == 35
        # Metadata hoisted to top level
        assert d["title"] == "AF Guide"
        assert d["source"] == "/data/raw/af.txt"
        assert d["section"] == "Introduction"
        assert d["strategy"] == "sentence"
        assert d["chunk_index"] == 0
        assert d["total_chunks"] == 5

    def test_from_dict_round_trip(self):
        c1 = Chunk(
            chunk_id="doc001-0001",
            text="Heart failure treatment.",
            doc_id="doc001",
            start_char=100,
            end_char=124,
            metadata={"title": "HF", "source": "/hf.txt", "section": "Tx",
                      "strategy": "paragraph", "chunk_index": 1, "total_chunks": 3},
        )
        d = c1.to_dict()
        c2 = Chunk.from_dict(d)
        assert c2.chunk_id == c1.chunk_id
        assert c2.text == c1.text
        assert c2.doc_id == c1.doc_id
        assert c2.start_char == c1.start_char
        assert c2.end_char == c1.end_char
        assert c2.metadata["title"] == "HF"
        assert c2.metadata["strategy"] == "paragraph"


# ---------------------------------------------------------------------------
# chunk_document() — sliding window
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_returns_list_of_chunks(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=50, overlap=10, strategy="sliding_window",
        )
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_non_empty_text_produces_chunks(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=50, overlap=10, strategy="sliding_window",
        )
        assert len(chunks) > 0

    def test_chunk_ids_unique(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=50, overlap=10, strategy="sliding_window",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_contain_doc_id(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "mydoc",
            chunk_size=50, overlap=10, strategy="sliding_window",
        )
        for c in chunks:
            assert c.chunk_id.startswith("mydoc-")

    def test_token_count_roughly_respected(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=50, overlap=5, strategy="sliding_window",
        )
        for c in chunks:
            tokens = len(c.text.split())
            # All chunks except possibly the last should be at most chunk_size
            assert tokens <= 55, f"Chunk too large: {tokens} tokens"

    def test_overlap_means_repeated_tokens(self):
        text = " ".join(str(i) for i in range(200))
        chunks = chunk_document(
            text, "doc1",
            chunk_size=40, overlap=10, strategy="sliding_window",
        )
        assert len(chunks) >= 2
        end_of_first = chunks[0].text.split()[-10:]
        start_of_second = chunks[1].text.split()[:10]
        assert end_of_first == start_of_second, "Overlap tokens not carried forward"

    def test_total_chunks_backfilled(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=50, overlap=5, strategy="sliding_window",
        )
        expected = len(chunks)
        for c in chunks:
            assert c.metadata["total_chunks"] == expected

    def test_chunk_index_sequential(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=50, overlap=5, strategy="sliding_window",
        )
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i


# ---------------------------------------------------------------------------
# chunk_document() — sentence strategy
# ---------------------------------------------------------------------------

class TestSentenceStrategy:
    def test_produces_chunks(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sentence",
        )
        assert len(chunks) > 0

    def test_sentences_not_split_mid_sentence(self):
        """Each chunk should end with sentence-ending punctuation."""
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=80, overlap=10, strategy="sentence",
        )
        for c in chunks[:-1]:
            stripped = c.text.strip()
            # Allow chunks ending in colon (list introductions), period, ?, !
            assert stripped[-1] in ".?!:)", (
                f"Chunk appears to end mid-sentence: '…{stripped[-30:]}'"
            )

    def test_metadata_strategy_field(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sentence",
        )
        for c in chunks:
            assert c.metadata["strategy"] == "sentence"


# ---------------------------------------------------------------------------
# chunk_document() — paragraph strategy
# ---------------------------------------------------------------------------

class TestParagraphStrategy:
    def test_produces_chunks(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=100, overlap=20, strategy="paragraph",
        )
        assert len(chunks) > 0

    def test_metadata_strategy_field(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=100, overlap=20, strategy="paragraph",
        )
        for c in chunks:
            assert c.metadata["strategy"] == "paragraph"

    def test_heading_starts_new_chunk(self):
        """A Markdown heading should trigger a new chunk boundary."""
        text = (
            "First section content about atrial fibrillation treatment.\n\n"
            "## New Section\n\n"
            "Second section content about ventricular tachycardia management."
        )
        chunks = chunk_document(
            text, "doc1",
            chunk_size=200, overlap=0, strategy="paragraph",
        )
        # The heading must appear at the start of a chunk
        heading_in_start = any(
            c.text.startswith("## New Section") for c in chunks
        )
        assert heading_in_start, "Heading should start a new chunk"


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    def test_title_in_metadata(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sentence",
            title="AF Guidelines 2023",
        )
        for c in chunks:
            assert c.metadata["title"] == "AF Guidelines 2023"

    def test_source_in_metadata(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sentence",
            source="/data/raw/af_guidelines.pdf",
        )
        for c in chunks:
            assert c.metadata["source"] == "/data/raw/af_guidelines.pdf"

    def test_section_detected_from_heading(self):
        """Chunks after a heading should carry that heading as 'section'."""
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sentence",
        )
        # Find a chunk that follows a heading — at least one should have a section
        sections = [c.metadata.get("section", "") for c in chunks]
        non_empty_sections = [s for s in sections if s]
        assert len(non_empty_sections) > 0, (
            "Expected at least one chunk with a non-empty section heading"
        )

    def test_doc_id_in_all_chunks(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "docXYZ",
            chunk_size=60, overlap=10, strategy="sentence",
        )
        for c in chunks:
            assert c.doc_id == "docXYZ"

    def test_start_end_char_within_text_bounds(self):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sliding_window",
        )
        n = len(CLEAN_CARDIOLOGY_TEXT.strip())
        for c in chunks:
            assert 0 <= c.start_char, "start_char must be >= 0"
            assert c.end_char <= n + 10, "end_char out of bounds"
            assert c.start_char < c.end_char, "start_char must be < end_char"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_returns_empty_list(self):
        for strategy in ["sliding_window", "sentence", "paragraph"]:
            result = chunk_document("", "doc1", strategy=strategy)
            assert result == [], f"Strategy {strategy} should return [] for empty input"

    def test_whitespace_only_returns_empty_list(self):
        result = chunk_document("   \n\t\n  ", "doc1", strategy="sentence")
        assert result == []

    def test_single_sentence_produces_one_chunk(self):
        text = "Atrial fibrillation is the most common sustained cardiac arrhythmia."
        chunks = chunk_document(text, "doc1", chunk_size=100, overlap=10, strategy="sentence")
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text.strip()

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_document(CLEAN_CARDIOLOGY_TEXT, "doc1", strategy="magic")

    def test_chunk_size_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            chunk_document(CLEAN_CARDIOLOGY_TEXT, "doc1", chunk_size=0)

    def test_negative_overlap_raises_value_error(self):
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_document(CLEAN_CARDIOLOGY_TEXT, "doc1", overlap=-1)

    def test_overlap_gte_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="overlap.*must be < chunk_size"):
            chunk_document(CLEAN_CARDIOLOGY_TEXT, "doc1", chunk_size=10, overlap=10)


# ---------------------------------------------------------------------------
# chunk_corpus()
# ---------------------------------------------------------------------------

class TestChunkCorpus:
    def test_returns_flat_list(self):
        docs = [CLEAN_CARDIOLOGY_TEXT, "Heart failure affects millions."]
        chunks = chunk_corpus(docs, doc_ids=["d1", "d2"], chunk_size=60, overlap=10)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_doc_ids_propagated(self):
        docs = [CLEAN_CARDIOLOGY_TEXT, "Short text about MI."]
        chunks = chunk_corpus(docs, doc_ids=["alpha", "beta"], chunk_size=60, overlap=10)
        doc_ids_seen = {c.doc_id for c in chunks}
        assert "alpha" in doc_ids_seen
        assert "beta" in doc_ids_seen

    def test_auto_doc_ids_generated_when_none(self):
        docs = ["Content A.", "Content B."]
        chunks = chunk_corpus(docs, chunk_size=20, overlap=0)
        doc_ids = {c.doc_id for c in chunks}
        assert len(doc_ids) == 2  # Two distinct auto-generated IDs

    def test_empty_corpus_returns_empty_list(self):
        assert chunk_corpus([]) == []


# ---------------------------------------------------------------------------
# JSONL save / load
# ---------------------------------------------------------------------------

class TestJSONLRoundTrip:
    def test_save_and_load_identity(self, tmp_path):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sentence",
            title="Test Document",
            source="/data/raw/test.txt",
        )
        out = tmp_path / "chunks.jsonl"
        save_chunks(chunks, out)

        loaded = load_chunks(out)
        assert len(loaded) == len(chunks)
        for orig, read in zip(chunks, loaded):
            assert orig.chunk_id == read.chunk_id
            assert orig.text == read.text
            assert orig.doc_id == read.doc_id
            assert orig.start_char == read.start_char
            assert orig.end_char == read.end_char
            assert orig.metadata["title"] == read.metadata["title"]
            assert orig.metadata["source"] == read.metadata["source"]
            assert orig.metadata["strategy"] == read.metadata["strategy"]

    def test_each_line_is_valid_json(self, tmp_path):
        chunks = chunk_document(
            CLEAN_CARDIOLOGY_TEXT, "doc1",
            chunk_size=60, overlap=10, strategy="sentence",
        )
        out = tmp_path / "chunks.jsonl"
        save_chunks(chunks, out)
        for line in out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)  # raises if malformed
                assert "chunk_id" in obj
                assert "text" in obj
                assert "doc_id" in obj

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_chunks(tmp_path / "nonexistent.jsonl")

    def test_save_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deeply" / "nested" / "chunks.jsonl"
        chunks = chunk_document("Short text.", "doc1", chunk_size=10, overlap=0)
        save_chunks(chunks, out)
        assert out.exists()

    def test_empty_chunk_list_produces_empty_file(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        save_chunks([], out)
        assert out.exists()
        content = out.read_text().strip()
        assert content == ""

    def test_load_empty_file_returns_empty_list(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        out.write_text("")
        result = load_chunks(out)
        assert result == []
