"""
retrieval package
-----------------
Sparse, approximate, dense, and hybrid retrieval methods for the
cardiology RAG pipeline.

Stage implementation status
----------------------------
Stage 2A  BM25Retriever           ✓ implemented
Stage 2B  LSHRetriever            stub
Stage 2C  DenseRetriever (FAISS)  stub
Stage 2D  HybridRetriever (RRF)   stub

Public API
----------
    from src.retrieval import BaseRetriever, RetrievalResult, BM25Retriever
"""

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import BM25Retriever, tokenize

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "BM25Retriever",
    "tokenize",
]
