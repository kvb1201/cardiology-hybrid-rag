"""
cleaner.py
----------
Generic text cleaning for cardiology literature.

Design constraints
------------------
* No stemming or lemmatization — preserve original clinical terms.
* No stopword removal — stopwords matter for sentence coherence in RAG.
* Preserve medically meaningful punctuation:
    - Hyphens inside compound terms (e.g. ST-elevation, dose-response)
    - Slashes in dose notation (e.g. 10 mg/kg)
    - Colons, semicolons in lab values (e.g. Na: 138 mEq/L)
    - Parentheses for abbreviations (e.g. atrial fibrillation (AF))
    - Square brackets for citations (e.g. [1], [2,3])
    - Plus/minus for ranges (e.g. 72 ± 5 years)
    - Degree signs and Greek letters that appear in PDF extraction
* Preserve Markdown-style headings emitted by the HTML loader.
* The cleaner is format-agnostic — it does not know whether input came
  from a PDF, HTML, or plain-text file.

Cleaning pipeline (applied in order)
--------------------------------------
1.  Decode ligatures and common Unicode substitutions.
2.  Remove PDF page-break markers (\\f) and page-number lines.
3.  Remove journal boilerplate patterns (DOI lines, copyright notices,
    "Downloaded from …" lines, "Received / Accepted / Published" lines).
4.  Fix soft-hyphen line-break artefacts from PDF column layout.
5.  Collapse multiple blank lines to at most two.
6.  Normalise whitespace within lines (tabs → space, runs of spaces → one).
7.  Strip leading/trailing whitespace from each line.
8.  Drop lines that are pure noise (page numbers, single characters, etc.)
    while leaving structural markers (headings) intact.
9.  Re-join the cleaned lines.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

# PDF form-feed page separators
_FF_RE = re.compile(r"\f+")

# Standalone page number lines: optional whitespace, digits, optional whitespace
# Must be on their own line to avoid nuking lab values like "PaO2 = 80"
_PAGE_NUM_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Running headers / footers that repeat the journal name pattern:
# lines that are ALL CAPS and shorter than 80 chars (journal header heuristic)
_ALL_CAPS_LINE_RE = re.compile(r"^[A-Z][A-Z\s\d\-:,\.]{5,79}$")

# DOI lines
_DOI_RE = re.compile(
    r"^\s*(doi|DOI|https?://doi\.org)[:\s/].*$", re.MULTILINE
)

# Copyright / licence lines
_COPYRIGHT_RE = re.compile(
    r"^\s*(©|Copyright|Ó|All rights reserved|This article|Published by"
    r"|Elsevier|Springer|Wiley|JACC|AHA|ACC|BMJ|NEJM|Lancet).*$",
    re.IGNORECASE | re.MULTILINE,
)

# "Downloaded from / Received / Accepted / Published online" boilerplate
# Note: colon after "Received"/"Accepted" is optional — PDF extraction may omit it.
_JOURNAL_META_RE = re.compile(
    r"^\s*(Downloaded from|Received:?|Accepted:?|Published online|"
    r"Correspondence to|Address reprint|Address for correspondence"
    r"|ClinicalTrials\.gov|Trial registration|ORCID|Funding|"
    r"Disclosures:?|Conflict of interest|Author contributions).*$",
    re.IGNORECASE | re.MULTILINE,
)

# Soft-hyphen line-break artefact from PDF two-column layout:
#   "cardio-\nvascular" → "cardiovascular"
# Matches a hyphen at EOL (possibly with trailing spaces) followed by a newline
_SOFT_HYPHEN_RE = re.compile(r"-\s*\n\s*([a-z])")

# Two or more consecutive blank lines → exactly two newlines (one blank line)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Runs of spaces or tabs within a line
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")

# Lines that are just punctuation/symbols with no alphabetic content
_SYMBOL_ONLY_LINE_RE = re.compile(r"^[\W\d_]+$")

# Lines that are a single character (noise from column-border extraction)
_SINGLE_CHAR_LINE_RE = re.compile(r"^\S$")

# Unicode ligature map: common PDF extraction substitutions
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2019": "'",   # right single quotation mark
    "\u2018": "'",   # left single quotation mark
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u2013": "-",   # en dash → hyphen
    "\u2014": " - ", # em dash → spaced hyphen
    "\u00ad": "",    # soft hyphen (invisible) → remove
    "\u00a0": " ",   # non-breaking space → regular space
    "\u2212": "-",   # minus sign → hyphen-minus
    "\u00b1": "±",   # plus-minus (keep as-is; meaningful in clinical values)
    "\u00d7": "x",   # multiplication sign → x
    "\u2265": ">=",  # ≥
    "\u2264": "<=",  # ≤
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03bc": "mu",
    "\u03c9": "omega",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode_ligatures(text: str) -> str:
    """Replace Unicode ligatures and typographic characters."""
    for char, replacement in _LIGATURE_MAP.items():
        text = text.replace(char, replacement)
    # Normalize remaining Unicode to NFC (composed form)
    return unicodedata.normalize("NFC", text)


def _remove_pdf_artifacts(text: str) -> str:
    """Remove PDF extraction noise: form feeds, page numbers, boilerplate."""
    # Replace form-feed page separators with a blank line
    text = _FF_RE.sub("\n\n", text)
    # Remove DOI, copyright, journal metadata lines
    text = _DOI_RE.sub("", text)
    text = _COPYRIGHT_RE.sub("", text)
    text = _JOURNAL_META_RE.sub("", text)
    return text


def _fix_hyphenation(text: str) -> str:
    """Rejoin words split across lines by PDF column-layout hyphenation."""
    return _SOFT_HYPHEN_RE.sub(r"\1", text)


def _clean_lines(text: str) -> str:
    """Per-line cleaning: drop noise lines, normalise inline whitespace."""
    lines = text.split("\n")
    cleaned: list[str] = []

    for line in lines:
        # Normalise inline whitespace (tabs and runs of spaces)
        line = _INLINE_WHITESPACE_RE.sub(" ", line)
        line = line.strip()

        # Always keep Markdown headings produced by the HTML loader
        if line.startswith("#"):
            cleaned.append(line)
            continue

        # Drop pure page-number lines
        if _PAGE_NUM_LINE_RE.match(line):
            continue

        # Drop single-character lines (column border noise)
        if _SINGLE_CHAR_LINE_RE.match(line):
            continue

        # Drop symbol-only lines (table rule lines, separator noise)
        if line and _SYMBOL_ONLY_LINE_RE.match(line):
            continue

        # Keep everything else (including blank lines for paragraph breaks)
        cleaned.append(line)

    return "\n".join(cleaned)


def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive newlines to at most 2 (one blank line)."""
    return _MULTI_BLANK_RE.sub("\n\n", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_document(raw_text: str) -> str:
    """Clean a single raw document string extracted from any source format.

    Applies the cleaning pipeline in order:
    1. Ligature/Unicode normalisation
    2. PDF artefact removal (form feeds, DOI lines, copyright notices)
    3. Soft-hyphen line-break repair
    4. Per-line noise removal and inline whitespace normalisation
    5. Blank-line collapsing

    Args:
        raw_text: Raw text as produced by a loader (PDF, HTML, plain-text).
                  May contain Unicode ligatures, form-feed characters, or
                  extraction artefacts.

    Returns:
        Cleaned text string.  Empty string if input is blank.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = _decode_ligatures(raw_text)
    text = _remove_pdf_artifacts(text)
    text = _fix_hyphenation(text)
    text = _clean_lines(text)
    text = _collapse_blank_lines(text)
    return text.strip()


def clean_corpus(raw_documents: List[str]) -> List[str]:
    """Apply :func:`clean_document` to every document in a list.

    Args:
        raw_documents: List of raw text strings, one per document.

    Returns:
        List of cleaned text strings in the same order.  Documents that
        are empty after cleaning are represented as empty strings (not
        removed) so the caller's index alignment is preserved.
    """
    return [clean_document(doc) for doc in raw_documents]
