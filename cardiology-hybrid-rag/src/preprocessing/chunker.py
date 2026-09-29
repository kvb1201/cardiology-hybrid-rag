"""
chunker.py
----------
Text chunking strategies for cleaned cardiology documents.

Strategies
----------
sliding_window
    Split on whitespace-delimited tokens.  Each chunk contains exactly
    ``chunk_size`` tokens with ``overlap`` tokens of context carried
    forward.  Fast and predictable; ignores sentence/paragraph structure.

sentence
    Use NLTK's Punkt sentence tokenizer to find sentence boundaries.
    Accumulate sentences until adding the next would exceed ``chunk_size``
    tokens, then emit and start a new chunk carrying the last ``overlap``
    tokens of the previous chunk as context.

paragraph
    Split the document on blank lines (two or more consecutive newlines).
    Paragraphs are accumulated the same way as sentences.  Preserves
    section/heading boundaries that the HTML loader marks with ``#`` prefixes.

Configuration
-------------
All numeric thresholds are passed explicitly as function arguments — nothing
is hard-coded.  Callers (build_corpus.py, unit tests) supply all values.

Chunk metadata contract
-----------------------
Every Chunk stores full provenance:

    chunk_id        globally unique: "{doc_id}-{zero_padded_index}"
    text            the chunk's text content
    doc_id          source document identifier
    start_char      character offset in the *cleaned* document text
    end_char        character offset in the *cleaned* document text
    metadata:
      title         document title (from RawDocument or caller-supplied)
      source        absolute path to the originating file
      section       nearest preceding heading (Markdown ``# …``) or ""
      strategy      chunking strategy used
      chunk_index   0-based position of this chunk within the document
      total_chunks  total chunk count for this document (back-filled)

JSONL serialization
-------------------
Each line of the output file is a JSON object with all Chunk fields
flattened (metadata fields hoisted to top level for easy downstream use).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, List, Optional

# ---------------------------------------------------------------------------
# NLTK sentence tokenizer — lazy import so the module loads without NLTK data
# ---------------------------------------------------------------------------
_sent_tokenizer = None


def _get_sent_tokenizer():
    """Return a cached NLTK Punkt sentence tokenizer, downloading if needed."""
    global _sent_tokenizer
    if _sent_tokenizer is None:
        import nltk  # noqa: PLC0415

        # Download punkt_tab (newer NLTK) or punkt (older) quietly
        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
                break
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    break
                except Exception:  # noqa: BLE001
                    continue

        try:
            _sent_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        except Exception:  # noqa: BLE001
            try:
                _sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            except Exception:  # noqa: BLE001
                # Absolute fallback: simple regex splitter
                _sent_tokenizer = None

    return _sent_tokenizer


# ---------------------------------------------------------------------------
# Heading detection helper
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)


def _nearest_heading_before(text: str, char_offset: int) -> str:
    """Return the text of the nearest Markdown heading before *char_offset*.

    Args:
        text: Full cleaned document text.
        char_offset: Character position of the chunk's start.

    Returns:
        Heading text (without the ``#`` prefix and leading space), or ``""``
        if no heading precedes this position.
    """
    best = ""
    for m in _HEADING_RE.finditer(text):
        if m.start() >= char_offset:
            break
        # Strip leading '#' characters and whitespace
        best = m.group().lstrip("#").strip()
    return best


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------


def _token_count(text: str) -> int:
    """Count whitespace-delimited tokens in *text*.

    This is intentionally simple and fast — no NLP tokenizer is used here
    because we are budgeting chunk size, not doing linguistic analysis.
    """
    return len(text.split())


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata.

    Attributes
    ----------
    chunk_id:
        Globally unique identifier: ``"{doc_id}-{index:04d}"``.
    text:
        The chunk's content.
    doc_id:
        Identifier of the source :class:`~src.ingestion.base.RawDocument`.
    start_char:
        Character offset of the first character of *text* within the
        cleaned document string.
    end_char:
        Character offset one past the last character of *text*.
    metadata:
        Dict with keys: ``title``, ``source``, ``section``, ``strategy``,
        ``chunk_index``, ``total_chunks``.
    """

    chunk_id: str
    text: str
    doc_id: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a flat dict suitable for JSONL output.

        Metadata keys are hoisted to the top level for readability.
        """
        d = {
            "chunk_id":    self.chunk_id,
            "text":        self.text,
            "doc_id":      self.doc_id,
            "start_char":  self.start_char,
            "end_char":    self.end_char,
        }
        d.update(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        """Reconstruct a Chunk from a flat dict (as produced by to_dict)). """
        known = {"chunk_id", "text", "doc_id", "start_char", "end_char"}
        metadata = {k: v for k, v in d.items() if k not in known}
        return cls(
            chunk_id=d["chunk_id"],
            text=d["text"],
            doc_id=d["doc_id"],
            start_char=d["start_char"],
            end_char=d["end_char"],
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Internal chunking generators
# ---------------------------------------------------------------------------


def _sliding_window_chunks(
    text: str,
    doc_id: str,
    chunk_size: int,
    overlap: int,
    title: str,
    source: str,
) -> List[Chunk]:
    """Token-based sliding window chunking."""
    tokens = text.split()
    if not tokens:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[Chunk] = []

    # Build a char-offset map: tokens[i] starts at char_offsets[i]
    # We do one pass through the text to record each token's start offset.
    char_offsets: List[int] = []
    pos = 0
    for tok in tokens:
        idx = text.find(tok, pos)
        char_offsets.append(idx)
        pos = idx + len(tok)

    token_start = 0
    while token_start < len(tokens):
        token_end = min(token_start + chunk_size, len(tokens))
        chunk_tokens = tokens[token_start:token_end]
        chunk_text = " ".join(chunk_tokens)

        start_char = char_offsets[token_start]
        # end_char: offset after the last token of this chunk
        last_tok_idx = token_end - 1
        end_char = char_offsets[last_tok_idx] + len(tokens[last_tok_idx])

        section = _nearest_heading_before(text, start_char)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{idx:04d}",
                text=chunk_text,
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "title":       title,
                    "source":      source,
                    "section":     section,
                    "strategy":    "sliding_window",
                    "chunk_index": idx,
                    "total_chunks": 0,  # back-filled below
                },
            )
        )

        if token_end == len(tokens):
            break
        token_start += step

    # Back-fill total_chunks
    total = len(chunks)
    for c in chunks:
        c.metadata["total_chunks"] = total

    return chunks


def _sentence_tokenize(text: str) -> List[tuple[str, int]]:
    """Return list of (sentence, start_char) using NLTK or fallback.

    Returns:
        List of (sentence_text, start_char_offset) tuples.
    """
    tokenizer = _get_sent_tokenizer()
    if tokenizer is not None:
        try:
            spans = list(tokenizer.span_tokenize(text))
            return [(text[s:e], s) for s, e in spans]
        except Exception:  # noqa: BLE001
            pass

    # Fallback: split on ". " / "? " / "! " followed by capital letter
    sentences: List[tuple[str, int]] = []
    pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    last = 0
    for m in pattern.finditer(text):
        sent = text[last : m.start() + 1].strip()
        if sent:
            sentences.append((sent, last))
        last = m.end()
    if last < len(text):
        sentences.append((text[last:].strip(), last))
    return [(s, o) for s, o in sentences if s]


def _sentence_chunks(
    text: str,
    doc_id: str,
    chunk_size: int,
    overlap: int,
    title: str,
    source: str,
) -> List[Chunk]:
    """Sentence-boundary-aware chunking."""
    sentences = _sentence_tokenize(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_sents: List[tuple[str, int]] = []
    current_tokens = 0

    def _emit() -> None:
        if not current_sents:
            return
        chunk_text = " ".join(s for s, _ in current_sents)
        start_char = current_sents[0][1]
        last_sent_text, last_sent_start = current_sents[-1]
        end_char = last_sent_start + len(last_sent_text)
        section = _nearest_heading_before(text, start_char)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{idx:04d}",
                text=chunk_text,
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "title":        title,
                    "source":       source,
                    "section":      section,
                    "strategy":     "sentence",
                    "chunk_index":  idx,
                    "total_chunks": 0,
                },
            )
        )

    for sent_text, sent_start in sentences:
        sent_tokens = _token_count(sent_text)

        if current_tokens + sent_tokens > chunk_size and current_sents:
            _emit()
            # Keep overlap worth of tokens from the end of the previous chunk
            overlap_sents: List[tuple[str, int]] = []
            overlap_count = 0
            for s, o in reversed(current_sents):
                tc = _token_count(s)
                if overlap_count + tc > overlap:
                    break
                overlap_sents.insert(0, (s, o))
                overlap_count += tc
            current_sents = overlap_sents
            current_tokens = overlap_count

        current_sents.append((sent_text, sent_start))
        current_tokens += sent_tokens

    _emit()

    total = len(chunks)
    for c in chunks:
        c.metadata["total_chunks"] = total

    return chunks


def _paragraph_chunks(
    text: str,
    doc_id: str,
    chunk_size: int,
    overlap: int,
    title: str,
    source: str,
) -> List[Chunk]:
    """Paragraph / section-aware chunking.

    Splits on blank lines (two or more consecutive newlines).  Heading lines
    (``# …``) always start a new chunk boundary.
    """
    # Split on blank lines
    raw_paras = re.split(r"\n{2,}", text)
    # Build (paragraph_text, start_char_in_original) pairs
    paras: List[tuple[str, int]] = []
    pos = 0
    for para in raw_paras:
        para_stripped = para.strip()
        if para_stripped:
            # Find actual start in the original text
            start = text.find(para_stripped, pos)
            if start == -1:
                start = pos
            paras.append((para_stripped, start))
            pos = start + len(para_stripped)

    if not paras:
        return []

    chunks: List[Chunk] = []
    current_paras: List[tuple[str, int]] = []
    current_tokens = 0

    def _emit() -> None:
        if not current_paras:
            return
        chunk_text = "\n\n".join(p for p, _ in current_paras)
        start_char = current_paras[0][1]
        last_p, last_start = current_paras[-1]
        end_char = last_start + len(last_p)
        section = _nearest_heading_before(text, start_char)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{idx:04d}",
                text=chunk_text,
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "title":        title,
                    "source":       source,
                    "section":      section,
                    "strategy":     "paragraph",
                    "chunk_index":  idx,
                    "total_chunks": 0,
                },
            )
        )

    for para_text, para_start in paras:
        para_tokens = _token_count(para_text)
        is_heading = para_text.startswith("#")

        # Headings always flush the current accumulator and start fresh
        if is_heading and current_paras:
            _emit()
            current_paras = []
            current_tokens = 0

        if current_tokens + para_tokens > chunk_size and current_paras:
            _emit()
            # Carry overlap: keep last paragraph(s) totalling <= overlap tokens
            overlap_paras: List[tuple[str, int]] = []
            overlap_count = 0
            for p, o in reversed(current_paras):
                tc = _token_count(p)
                if overlap_count + tc > overlap:
                    break
                overlap_paras.insert(0, (p, o))
                overlap_count += tc
            current_paras = overlap_paras
            current_tokens = overlap_count

        current_paras.append((para_text, para_start))
        current_tokens += para_tokens

    _emit()

    total = len(chunks)
    for c in chunks:
        c.metadata["total_chunks"] = total

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "sliding_window": _sliding_window_chunks,
    "sentence":       _sentence_chunks,
    "paragraph":      _paragraph_chunks,
}


def chunk_document(
    text: str,
    doc_id: str,
    chunk_size: int = 256,
    overlap: int = 32,
    strategy: str = "sentence",
    title: str = "",
    source: str = "",
) -> List[Chunk]:
    """Split a single cleaned document into overlapping chunks.

    Args:
        text:       Cleaned document text (output of ``clean_document()``).
        doc_id:     Unique identifier for the source document.
        chunk_size: Target chunk size in whitespace-delimited tokens.
        overlap:    Token overlap carried from one chunk to the next.
        strategy:   One of ``'sliding_window'``, ``'sentence'``,
                    ``'paragraph'``.
        title:      Document title stored in every chunk's metadata.
        source:     Originating file path stored in every chunk's metadata.

    Returns:
        Ordered list of :class:`Chunk` objects.  Empty list if *text* is
        blank after stripping.

    Raises:
        ValueError: If *strategy* is not one of the supported values.
        ValueError: If *chunk_size* <= 0 or *overlap* < 0 or
                    *overlap* >= *chunk_size*.
    """
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {sorted(_STRATEGY_MAP)}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
        )

    text = text.strip()
    if not text:
        return []

    fn = _STRATEGY_MAP[strategy]
    return fn(
        text=text,
        doc_id=doc_id,
        chunk_size=chunk_size,
        overlap=overlap,
        title=title,
        source=source,
    )


def chunk_corpus(
    documents: List[str],
    doc_ids: Optional[List[str]] = None,
    chunk_size: int = 256,
    overlap: int = 32,
    strategy: str = "sentence",
    titles: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> List[Chunk]:
    """Chunk an entire corpus of cleaned documents.

    Args:
        documents:  List of cleaned document strings.
        doc_ids:    Parallel list of unique document identifiers.
                    Auto-generated UUIDs are used if ``None``.
        chunk_size: Target chunk size in tokens.
        overlap:    Token overlap between consecutive chunks.
        strategy:   Chunking strategy.
        titles:     Optional parallel list of document titles.
        sources:    Optional parallel list of source file paths.

    Returns:
        Flat list of :class:`Chunk` objects across all documents, in
        document order.
    """
    n = len(documents)
    if doc_ids is None:
        doc_ids = [uuid.uuid4().hex[:16] for _ in range(n)]
    if titles is None:
        titles = [""] * n
    if sources is None:
        sources = [""] * n

    all_chunks: List[Chunk] = []
    for text, doc_id, title, source in zip(documents, doc_ids, titles, sources):
        all_chunks.extend(
            chunk_document(
                text=text,
                doc_id=doc_id,
                chunk_size=chunk_size,
                overlap=overlap,
                strategy=strategy,
                title=title,
                source=source,
            )
        )
    return all_chunks


def save_chunks(chunks: List[Chunk], output_path: str | Path) -> None:
    """Serialize chunks to a JSONL file (one JSON object per line).

    Each line is a flat JSON object with metadata fields hoisted to the top
    level (see :meth:`Chunk.to_dict`).

    Args:
        chunks:      List of :class:`Chunk` objects to save.
        output_path: Destination file path.  Parent directories are created
                     automatically.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(input_path: str | Path) -> List[Chunk]:
    """Load chunks from a JSONL file produced by :func:`save_chunks`.

    Args:
        input_path: Path to the JSONL file.

    Returns:
        List of :class:`Chunk` objects.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        json.JSONDecodeError: If any line is malformed JSON.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")
    chunks: List[Chunk] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Malformed JSON on line {lineno} of '{input_path}': {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc
            chunks.append(Chunk.from_dict(d))
    return chunks
