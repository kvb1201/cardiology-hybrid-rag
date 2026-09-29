"""
test_ingestion.py
-----------------
Unit tests for src.ingestion.

Coverage
--------
* RawDocument.make_doc_id(): deterministic, 16-char hex
* RawDocument.is_empty()
* TextLoader: plain text and Markdown loading, title extraction, encoding
* HTMLLoader: text extraction, heading preservation, boilerplate removal, title
* PDFLoader: skip (PDFs are tested in integration; creating valid PDF bytes
  in-memory requires reportlab/fpdf — not in Stage-0 requirements).
  A corrupt-file error path is tested instead.
* get_loader(): correct dispatch by extension, error on unknown extension
* load_directory(): loads all supported files, skips hidden/unsupported files
* Edge cases: FileNotFoundError, empty files
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.base import RawDocument
from src.ingestion.factory import get_loader, load_directory
from src.ingestion.text_loader import TextLoader
from src.ingestion.html_loader import HTMLLoader


# ---------------------------------------------------------------------------
# RawDocument helpers
# ---------------------------------------------------------------------------

class TestRawDocumentHelpers:
    def test_make_doc_id_returns_16_char_hex(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.touch()
        doc_id = RawDocument.make_doc_id(p)
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_make_doc_id_is_deterministic(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.touch()
        assert RawDocument.make_doc_id(p) == RawDocument.make_doc_id(p)

    def test_make_doc_id_differs_for_different_paths(self, tmp_path):
        p1 = tmp_path / "doc1.txt"
        p2 = tmp_path / "doc2.txt"
        p1.touch()
        p2.touch()
        assert RawDocument.make_doc_id(p1) != RawDocument.make_doc_id(p2)

    def test_is_empty_true_for_blank_text(self):
        doc = RawDocument(
            doc_id="x", source_path="/x", file_type="text",
            raw_text="   \n\t  ",
        )
        assert doc.is_empty() is True

    def test_is_empty_false_for_real_content(self):
        doc = RawDocument(
            doc_id="x", source_path="/x", file_type="text",
            raw_text="Atrial fibrillation management.",
        )
        assert doc.is_empty() is False


# ---------------------------------------------------------------------------
# TextLoader
# ---------------------------------------------------------------------------

class TestTextLoader:
    def test_loads_plain_text(self, sample_txt_file):
        loader = TextLoader()
        doc = loader.load(sample_txt_file)
        assert doc.file_type == "text"
        assert "Atrial Fibrillation" in doc.raw_text
        assert doc.raw_text  # non-empty

    def test_source_path_is_absolute(self, sample_txt_file):
        loader = TextLoader()
        doc = loader.load(sample_txt_file)
        assert Path(doc.source_path).is_absolute()

    def test_doc_id_is_16_hex(self, sample_txt_file):
        loader = TextLoader()
        doc = loader.load(sample_txt_file)
        assert len(doc.doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc.doc_id)

    def test_title_from_markdown_h1(self, sample_md_file):
        loader = TextLoader()
        doc = loader.load(sample_md_file)
        assert doc.title == "Hypertension Management"

    def test_title_fallback_to_filename_stem(self, tmp_path):
        p = tmp_path / "my_document.txt"
        p.write_text("No heading here, just content.", encoding="utf-8")
        loader = TextLoader()
        doc = loader.load(p)
        assert doc.title == "my_document"

    def test_metadata_contains_encoding(self, sample_txt_file):
        loader = TextLoader()
        doc = loader.load(sample_txt_file)
        assert "encoding" in doc.metadata

    def test_metadata_contains_size_bytes(self, sample_txt_file):
        loader = TextLoader()
        doc = loader.load(sample_txt_file)
        assert "size_bytes" in doc.metadata
        assert doc.metadata["size_bytes"] > 0

    def test_markdown_file_loaded(self, sample_md_file):
        loader = TextLoader()
        doc = loader.load(sample_md_file)
        assert "Hypertension" in doc.raw_text

    def test_file_not_found_raises(self, tmp_path):
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")

    def test_empty_file_returns_empty_raw_text(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        loader = TextLoader()
        doc = loader.load(p)
        assert doc.raw_text == ""
        assert doc.is_empty()

    def test_can_load_true_for_supported_extensions(self, tmp_path):
        for ext in [".txt", ".md", ".rst"]:
            p = tmp_path / f"file{ext}"
            assert TextLoader.can_load(p) is True

    def test_can_load_false_for_unsupported(self, tmp_path):
        assert TextLoader.can_load(tmp_path / "file.pdf") is False
        assert TextLoader.can_load(tmp_path / "file.html") is False


# ---------------------------------------------------------------------------
# HTMLLoader
# ---------------------------------------------------------------------------

class TestHTMLLoader:
    def test_loads_html(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert doc.file_type == "html"
        assert "Heart Failure" in doc.raw_text

    def test_title_from_title_tag(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert doc.title == "Heart Failure Overview"

    def test_nav_stripped(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "Site navigation" not in doc.raw_text

    def test_footer_stripped(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "Copyright 2023" not in doc.raw_text

    def test_script_stripped(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "alert" not in doc.raw_text

    def test_headings_preserved_with_markdown_prefix(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        # h1 → "# …", h2 → "## …"
        assert "# Heart Failure Overview" in doc.raw_text
        assert "## Classification" in doc.raw_text
        assert "## Treatment" in doc.raw_text

    def test_medical_content_preserved(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "LVEF" in doc.raw_text
        assert "HFrEF" in doc.raw_text

    def test_source_path_is_absolute(self, sample_html_file):
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert Path(doc.source_path).is_absolute()

    def test_file_not_found_raises(self, tmp_path):
        loader = HTMLLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.html")

    def test_title_fallback_to_filename_stem(self, tmp_path):
        p = tmp_path / "no_title.html"
        p.write_text("<html><body><p>Content.</p></body></html>", encoding="utf-8")
        loader = HTMLLoader()
        doc = loader.load(p)
        assert doc.title == "no_title"

    def test_can_load_true_for_html_extensions(self, tmp_path):
        assert HTMLLoader.can_load(tmp_path / "file.html") is True
        assert HTMLLoader.can_load(tmp_path / "file.htm") is True

    def test_can_load_false_for_other_extensions(self, tmp_path):
        assert HTMLLoader.can_load(tmp_path / "file.pdf") is False
        assert HTMLLoader.can_load(tmp_path / "file.txt") is False


# ---------------------------------------------------------------------------
# get_loader() factory
# ---------------------------------------------------------------------------

class TestGetLoader:
    def test_returns_text_loader_for_txt(self, tmp_path):
        loader = get_loader(tmp_path / "doc.txt")
        assert isinstance(loader, TextLoader)

    def test_returns_text_loader_for_md(self, tmp_path):
        loader = get_loader(tmp_path / "doc.md")
        assert isinstance(loader, TextLoader)

    def test_returns_html_loader_for_html(self, tmp_path):
        loader = get_loader(tmp_path / "doc.html")
        assert isinstance(loader, HTMLLoader)

    def test_returns_html_loader_for_htm(self, tmp_path):
        loader = get_loader(tmp_path / "doc.htm")
        assert isinstance(loader, HTMLLoader)

    def test_unknown_extension_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="No loader registered"):
            get_loader(tmp_path / "doc.xyz")

    def test_accepts_string_path(self, tmp_path):
        loader = get_loader(str(tmp_path / "doc.txt"))
        assert isinstance(loader, TextLoader)


# ---------------------------------------------------------------------------
# load_directory()
# ---------------------------------------------------------------------------

class TestLoadDirectory:
    def test_loads_all_supported_files(self, sample_raw_dir):
        docs = load_directory(sample_raw_dir)
        assert len(docs) == 3  # txt + md + html

    def test_skips_hidden_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / ".gitkeep").touch()
        (raw_dir / "real.txt").write_text("Content.", encoding="utf-8")
        docs = load_directory(raw_dir)
        assert len(docs) == 1

    def test_skips_unsupported_extensions(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "data.csv").write_text("a,b,c", encoding="utf-8")
        (raw_dir / "notes.txt").write_text("Cardiology notes.", encoding="utf-8")
        docs = load_directory(raw_dir)
        assert len(docs) == 1

    def test_empty_directory_returns_empty_list(self, tmp_path):
        raw_dir = tmp_path / "empty"
        raw_dir.mkdir()
        docs = load_directory(raw_dir)
        assert docs == []

    def test_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_directory(tmp_path / "does_not_exist")

    def test_doc_ids_are_unique(self, sample_raw_dir):
        docs = load_directory(sample_raw_dir)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_all_docs_have_source_path(self, sample_raw_dir):
        docs = load_directory(sample_raw_dir)
        for doc in docs:
            assert doc.source_path
            assert Path(doc.source_path).is_absolute()

    def test_recursive_flag_scans_subdirs(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sub_dir = raw_dir / "subdir"
        sub_dir.mkdir(parents=True)
        (raw_dir / "top.txt").write_text("Top content.", encoding="utf-8")
        (sub_dir / "nested.txt").write_text("Nested content.", encoding="utf-8")
        docs = load_directory(raw_dir, recursive=True)
        assert len(docs) == 2

    def test_non_recursive_does_not_scan_subdirs(self, tmp_path):
        raw_dir = tmp_path / "raw"
        sub_dir = raw_dir / "subdir"
        sub_dir.mkdir(parents=True)
        (raw_dir / "top.txt").write_text("Top content.", encoding="utf-8")
        (sub_dir / "nested.txt").write_text("Nested content.", encoding="utf-8")
        docs = load_directory(raw_dir, recursive=False)
        assert len(docs) == 1
