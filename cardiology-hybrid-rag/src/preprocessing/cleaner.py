"""
cleaner.py
----------
Raw text cleaning for cardiology literature.

Responsibilities (to be implemented):
  - Remove boilerplate headers/footers from PDF-extracted text
  - Normalize whitespace, Unicode, and special characters
  - Strip irrelevant metadata sections (e.g., journal footers, page numbers)
  - Sentence boundary detection / basic normalization
"""

from __future__ import annotations

from typing import List


def clean_document(raw_text: str) -> str:
    """Clean a single raw document string.

    Args:
        raw_text: Raw text extracted from a source document (PDF, HTML, etc.).

    Returns:
        Cleaned text string ready for chunking.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("clean_document() is not yet implemented.")


def clean_corpus(raw_documents: List[str]) -> List[str]:
    """Apply clean_document() to a list of raw documents.

    Args:
        raw_documents: List of raw text strings.

    Returns:
        List of cleaned text strings.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("clean_corpus() is not yet implemented.")
