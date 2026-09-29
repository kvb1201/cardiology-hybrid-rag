"""
html_loader.py
--------------
HTML text extraction using BeautifulSoup + lxml.

Extraction strategy
-------------------
* The ``<title>`` tag is extracted as the document title.
* Boilerplate HTML elements are removed before text extraction:
    - ``<script>``, ``<style>``, ``<nav>``, ``<footer>``, ``<header>``
    - ``<aside>``, ``<form>``, ``<button>``, ``<iframe>``
* Heading tags (h1–h6) are preserved with a Markdown-style prefix
  (e.g. ``## Atrial Fibrillation``) so section-aware chunking can
  detect boundaries downstream.
* All remaining text is extracted via ``get_text(separator='\\n')``.
* No cleaning (whitespace normalisation, artefact removal) is done here —
  that is the cleaner's responsibility.

Charset handling
----------------
BeautifulSoup auto-detects encoding via the ``<meta charset>`` tag.
If detection fails, UTF-8 with ``errors='replace'`` is used as fallback.
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.base import BaseLoader, RawDocument

try:
    from bs4 import BeautifulSoup, Tag
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "beautifulsoup4 and lxml are required for HTML loading: "
        "pip install beautifulsoup4 lxml"
    ) from exc

# Tags whose full subtree should be removed before text extraction
_REMOVE_TAGS = {
    "script", "style", "nav", "footer", "header",
    "aside", "form", "button", "iframe", "noscript",
}

# Heading tags and their Markdown-style prefix
_HEADING_PREFIX = {
    "h1": "# ",
    "h2": "## ",
    "h3": "### ",
    "h4": "#### ",
    "h5": "##### ",
    "h6": "###### ",
}


def _extract_text_with_headings(soup: BeautifulSoup) -> str:
    """Walk the parse tree and emit text, prefixing headings.

    Preserving heading markers lets the paragraph/section chunker detect
    section boundaries without a second parse pass.

    Args:
        soup: Parsed BeautifulSoup tree (boilerplate already removed).

    Returns:
        Multi-line string with Markdown-prefixed headings.
    """
    lines: list[str] = []
    body = soup.find("body") or soup

    for element in body.descendants:
        if not isinstance(element, Tag):
            continue
        tag_name = element.name.lower() if element.name else ""
        if tag_name in _HEADING_PREFIX:
            heading_text = element.get_text(separator=" ", strip=True)
            if heading_text:
                lines.append(_HEADING_PREFIX[tag_name] + heading_text)
        elif tag_name == "p":
            para_text = element.get_text(separator=" ", strip=True)
            if para_text:
                lines.append(para_text)
        elif tag_name in {"li"}:
            item_text = element.get_text(separator=" ", strip=True)
            if item_text:
                lines.append("- " + item_text)

    # Fall back to full get_text if the structured walk yielded nothing
    if not lines:
        return body.get_text(separator="\n")

    return "\n".join(lines)


class HTMLLoader(BaseLoader):
    """Load text from an HTML file using BeautifulSoup."""

    SUPPORTED_EXTENSIONS = (".html", ".htm")

    def load(self, path: Path) -> RawDocument:
        """Extract text and metadata from an HTML file.

        Args:
            path: Path to the ``.html`` / ``.htm`` file.

        Returns:
            :class:`RawDocument` with heading-preserving plain text.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file cannot be parsed.
        """
        self._require_exists(path)

        raw_bytes = path.read_bytes()
        # lxml parser: fastest, most lenient with malformed HTML
        try:
            soup = BeautifulSoup(raw_bytes, "lxml")
        except Exception as exc:
            raise ValueError(f"Cannot parse HTML '{path}': {exc}") from exc

        # --- title -------------------------------------------------------
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else path.stem

        # --- remove boilerplate subtrees ---------------------------------
        for tag_name in _REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # --- extract text preserving headings ----------------------------
        raw_text = _extract_text_with_headings(soup)

        # --- metadata ----------------------------------------------------
        metadata: dict = {
            "charset": soup.original_encoding or "utf-8",
            "lang": str(soup.find("html", {}).get("lang", "")) if soup.find("html") else "",  # type: ignore[union-attr]
        }

        # Meta description if present
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and isinstance(desc_tag, Tag):
            metadata["description"] = str(desc_tag.get("content", ""))

        return RawDocument(
            doc_id=RawDocument.make_doc_id(path),
            source_path=str(path.resolve()),
            file_type="html",
            raw_text=raw_text,
            title=title.strip(),
            metadata=metadata,
        )
