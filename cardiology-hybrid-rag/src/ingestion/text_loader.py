"""
text_loader.py
--------------
Plain-text and Markdown file loader.

Handles ``.txt``, ``.md``, and ``.rst`` files.

Title heuristic
---------------
1. If the file starts with a Markdown H1 (``# Title``), use that.
2. Otherwise fall back to the filename stem.

No structural transformation is applied — the raw bytes are decoded and
stored verbatim so the cleaner and chunker see the original content.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.ingestion.base import BaseLoader, RawDocument

# Matches a Markdown H1 on the very first non-blank line
_MD_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)

# Encodings to try in order before giving up
_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


class TextLoader(BaseLoader):
    """Load plain text, Markdown, or reStructuredText files."""

    SUPPORTED_EXTENSIONS = (".txt", ".md", ".markdown", ".rst")

    def load(self, path: Path) -> RawDocument:
        """Read a plain-text file and return a RawDocument.

        Args:
            path: Path to the text file.

        Returns:
            :class:`RawDocument` with raw file content as ``raw_text``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file cannot be decoded with any tried encoding.
        """
        self._require_exists(path)

        raw_bytes = path.read_bytes()
        raw_text: str | None = None
        detected_encoding = "utf-8"

        for enc in _ENCODINGS:
            try:
                raw_text = raw_bytes.decode(enc)
                detected_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if raw_text is None:
            # Last-resort: decode with replacement characters
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            detected_encoding = "utf-8-replace"

        # --- title heuristic: first Markdown H1 or filename stem ---------
        match = _MD_H1_RE.search(raw_text)
        title = match.group(1).strip() if match else path.stem

        metadata: dict = {
            "encoding": detected_encoding,
            "file_extension": path.suffix.lower(),
            "size_bytes": len(raw_bytes),
        }

        return RawDocument(
            doc_id=RawDocument.make_doc_id(path),
            source_path=str(path.resolve()),
            file_type="text",
            raw_text=raw_text,
            title=title,
            metadata=metadata,
        )
