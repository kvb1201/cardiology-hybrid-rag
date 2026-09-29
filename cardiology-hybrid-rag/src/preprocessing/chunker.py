"""
chunker.py
----------
Text chunking strategies for cleaned cardiology documents.

Responsibilities (to be implemented):
  - Fixed-size sliding window chunking (token or character based)
  - Sentence-aware chunking (respect sentence boundaries)
  - Paragraph / section-aware chunking
  - Overlap control to preserve context across chunk boundaries
  - Output serialization: save chunks with metadata to data/chunks/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """Represents a single text chunk with provenance metadata."""

    chunk_id: str
    text: str
    doc_id: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def chunk_document(
    text: str,
    doc_id: str,
    chunk_size: int = 512,
    overlap: int = 64,
    strategy: str = "sliding_window",
) -> List[Chunk]:
    """Split a single cleaned document into overlapping chunks.

    Args:
        text: Cleaned document text.
        doc_id: Unique identifier for the source document.
        chunk_size: Target chunk size in tokens/characters.
        overlap: Number of tokens/characters to overlap between chunks.
        strategy: Chunking strategy — one of 'sliding_window', 'sentence', 'paragraph'.

    Returns:
        Ordered list of Chunk objects.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("chunk_document() is not yet implemented.")


def chunk_corpus(
    documents: List[str],
    doc_ids: Optional[List[str]] = None,
    chunk_size: int = 512,
    overlap: int = 64,
    strategy: str = "sliding_window",
) -> List[Chunk]:
    """Chunk an entire corpus of cleaned documents.

    Args:
        documents: List of cleaned document strings.
        doc_ids: Optional list of document identifiers (auto-generated if None).
        chunk_size: Target chunk size.
        overlap: Overlap between consecutive chunks.
        strategy: Chunking strategy.

    Returns:
        Flat list of Chunk objects across all documents.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("chunk_corpus() is not yet implemented.")


def save_chunks(chunks: List[Chunk], output_path: str) -> None:
    """Serialize chunks to a JSONL file.

    Args:
        chunks: List of Chunk objects to save.
        output_path: Destination file path (e.g., data/chunks/chunks.jsonl).

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("save_chunks() is not yet implemented.")


def load_chunks(input_path: str) -> List[Chunk]:
    """Load chunks from a JSONL file produced by save_chunks().

    Args:
        input_path: Path to the JSONL file.

    Returns:
        List of Chunk objects.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("load_chunks() is not yet implemented.")
