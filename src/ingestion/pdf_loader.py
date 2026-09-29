"""
pdf_loader.py
-------------
PDF text extraction using pypdf.

Extraction strategy
-------------------
* Page text is extracted with pypdf's ``extract_text()``.
* Pages are joined with a form-feed sentinel (``\\f``) so downstream code
  can optionally split by page.
* PDF metadata fields (Title, Author, Subject, CreationDate) are stored in
  ``RawDocument.metadata`` unchanged — cleaning is not done here.
* If a page yields no text (scanned / image-only page), a placeholder comment
  is inserted so page numbering in metadata stays consistent.

Limitations
-----------
* pypdf cannot OCR scanned pages.  Image-only PDFs will produce mostly empty
  text.  Use an OCR step (e.g. pytesseract) if the corpus contains scans.
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.base import BaseLoader, RawDocument

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError as exc:  # pragma: no cover
    raise ImportError("pypdf is required for PDF loading: pip install pypdf") from exc


class PDFLoader(BaseLoader):
    """Load text from a PDF file using pypdf."""

    SUPPORTED_EXTENSIONS = (".pdf",)

    def load(self, path: Path) -> RawDocument:
        """Extract text from all pages of a PDF.

        Args:
            path: Path to the ``.pdf`` file.

        Returns:
            :class:`RawDocument` with per-page text joined by ``\\f``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If pypdf cannot parse the file.
        """
        self._require_exists(path)

        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise ValueError(f"Cannot parse PDF '{path}': {exc}") from exc

        # --- extract title from PDF metadata (best-effort) ----------------
        pdf_meta = reader.metadata or {}
        title = (
            pdf_meta.get("/Title")
            or pdf_meta.get("Title")
            or path.stem
        )
        # pypdf metadata values may be bytes-like objects in some versions
        if not isinstance(title, str):
            title = str(title)

        # --- extract per-page text ----------------------------------------
        pages: list[str] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
            if not text.strip():
                text = f"[PAGE {page_num}: no extractable text]"
            pages.append(text)

        raw_text = "\f".join(pages)

        # --- collect metadata ---------------------------------------------
        metadata: dict = {
            "page_count": len(reader.pages),
            "author":     str(pdf_meta.get("/Author") or pdf_meta.get("Author") or ""),
            "subject":    str(pdf_meta.get("/Subject") or pdf_meta.get("Subject") or ""),
            "creation_date": str(pdf_meta.get("/CreationDate") or ""),
        }

        return RawDocument(
            doc_id=RawDocument.make_doc_id(path),
            source_path=str(path.resolve()),
            file_type="pdf",
            raw_text=raw_text,
            title=title.strip(),
            metadata=metadata,
        )
