"""
base.py
-------
Shared data contract and abstract base class for all document loaders.

Design principles
-----------------
* A loader's ONLY job is raw extraction — it must not clean, chunk, or
  interpret the text in any domain-specific way.
* Every loader returns a RawDocument so downstream code works uniformly
  regardless of source format.
* Loaders raise descriptive exceptions rather than silently returning empty
  results — callers decide whether to skip or fail.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RawDocument:
    """Normalised container for text extracted from any source file.

    Attributes
    ----------
    doc_id:
        Stable identifier derived from the source file path (SHA-256 of the
        resolved absolute path string, hex-encoded, first 16 chars).
    source_path:
        Absolute path to the originating file.
    file_type:
        One of ``'pdf'``, ``'html'``, ``'text'``.
    raw_text:
        Full extracted text, un-cleaned.  May contain extraction artefacts
        (ligatures, hyphenation, page markers, etc.).
    title:
        Best-effort document title.  Extracted from PDF metadata, the HTML
        ``<title>`` tag, or the filename stem — whichever is available first.
    metadata:
        Catch-all dict for format-specific extras (page count, author,
        charset, etc.).  Never relied upon by cleaning/chunking code.
    """

    doc_id: str
    source_path: str
    file_type: str
    raw_text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_doc_id(path: str | Path) -> str:
        """Derive a stable, collision-resistant doc_id from a file path.

        Uses the first 16 hex characters of the SHA-256 hash of the
        resolved absolute path string.

        Args:
            path: Absolute or relative path to the source file.

        Returns:
            16-character lowercase hex string.
        """
        resolved = str(Path(path).resolve())
        return hashlib.sha256(resolved.encode()).hexdigest()[:16]

    def is_empty(self) -> bool:
        """Return True if the extracted text is blank after stripping."""
        return not self.raw_text.strip()


class BaseLoader(ABC):
    """Abstract base class all format-specific loaders must implement.

    Subclasses override :meth:`load` and optionally :meth:`can_load`.
    """

    # File extensions this loader handles (lower-case, with dot).
    SUPPORTED_EXTENSIONS: tuple[str, ...] = ()

    @classmethod
    def can_load(cls, path: Path) -> bool:
        """Return True if this loader can handle *path* based on extension.

        Args:
            path: Path object pointing to the candidate file.

        Returns:
            Boolean.
        """
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @abstractmethod
    def load(self, path: Path) -> RawDocument:
        """Extract raw text and metadata from *path*.

        Args:
            path: Path to the source file.  Must exist.

        Returns:
            :class:`RawDocument` with at minimum ``raw_text`` populated.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file is corrupt or unreadable.
        """
        ...

    def _require_exists(self, path: Path) -> None:
        """Raise FileNotFoundError if *path* does not exist."""
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
