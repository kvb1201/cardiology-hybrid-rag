"""
factory.py
----------
Loader factory: resolve the correct loader for a given file path.

Usage
-----
    from src.ingestion import get_loader, RawDocument
    from pathlib import Path

    loader = get_loader(Path("data/raw/paper.pdf"))
    doc: RawDocument = loader.load(Path("data/raw/paper.pdf"))

Adding new loaders
------------------
Register a new :class:`~src.ingestion.base.BaseLoader` subclass in
``_LOADER_REGISTRY`` below.  The factory walks the registry in order and
returns the first loader whose ``can_load()`` returns True.
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.base import BaseLoader
from src.ingestion.html_loader import HTMLLoader
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_loader import TextLoader

# Ordered list of loader classes.  More specific loaders should come first.
_LOADER_REGISTRY: list[type[BaseLoader]] = [
    PDFLoader,
    HTMLLoader,
    TextLoader,
]


def get_loader(path: Path | str) -> BaseLoader:
    """Return the appropriate loader instance for *path*.

    Args:
        path: Path to a file in ``data/raw/`` (or any location).

    Returns:
        An instantiated :class:`BaseLoader` subclass.

    Raises:
        ValueError: If no registered loader supports the file extension.
    """
    path = Path(path)
    for loader_cls in _LOADER_REGISTRY:
        if loader_cls.can_load(path):
            return loader_cls()
    supported = sorted(
        ext
        for cls in _LOADER_REGISTRY
        for ext in cls.SUPPORTED_EXTENSIONS
    )
    raise ValueError(
        f"No loader registered for extension '{path.suffix}' "
        f"(file: {path.name}). "
        f"Supported extensions: {supported}"
    )


def load_directory(
    directory: Path | str,
    recursive: bool = False,
) -> list:
    """Load all supported documents from a directory.

    Silently skips files whose extension is not supported and hidden files
    (names starting with ``'.'``).

    Args:
        directory: Path to the directory to scan.
        recursive: If True, recurse into subdirectories.

    Returns:
        List of :class:`~src.ingestion.base.RawDocument` objects, one per
        successfully loaded file.  Order follows filesystem sort order.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    from src.ingestion.base import RawDocument  # local import avoids circularity

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    glob_pattern = "**/*" if recursive else "*"
    files = sorted(directory.glob(glob_pattern))

    documents: list[RawDocument] = []
    errors: list[tuple[Path, Exception]] = []

    for file_path in files:
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue  # skip hidden files / .gitkeep
        try:
            loader = get_loader(file_path)
        except ValueError:
            continue  # unsupported extension — skip silently

        try:
            doc = loader.load(file_path)
            documents.append(doc)
        except Exception as exc:  # noqa: BLE001
            errors.append((file_path, exc))

    if errors:
        import warnings
        for file_path, exc in errors:
            warnings.warn(
                f"Failed to load '{file_path.name}': {exc}",
                stacklevel=2,
            )

    return documents
