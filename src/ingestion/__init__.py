"""
ingestion package
-----------------
Source-specific document loading layer.

Each loader handles extraction from one file type and returns a normalised
RawDocument object. Generic cleaning and chunking happen downstream in the
preprocessing package and are completely decoupled from extraction.

Public API
----------
from src.ingestion import get_loader, RawDocument
"""

from src.ingestion.base import RawDocument
from src.ingestion.factory import get_loader

__all__ = ["RawDocument", "get_loader"]
