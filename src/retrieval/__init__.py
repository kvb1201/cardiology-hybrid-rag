"""
retrieval package
-----------------
Sparse, approximate, dense, and hybrid retrieval methods for the
cardiology RAG pipeline.

Stage implementation status
----------------------------
Stage 2A  BM25Retriever           ✓ implemented
Stage 2C  LSHRetriever            ✓ implemented
Stage 2D  DenseRetriever (FAISS)  ✓ implemented
Stage 2E  HybridRetriever (RRF)   stub

Public API
----------
    from src.retrieval import BaseRetriever, RetrievalResult
    from src.retrieval import BM25Retriever, tokenize
    from src.retrieval import LSHRetriever, shingle
    from src.retrieval import DenseRetriever
"""

from src.retrieval.base  import BaseRetriever, RetrievalResult
from src.retrieval.bm25  import BM25Retriever, tokenize
from src.retrieval.lsh   import LSHRetriever, shingle
from src.retrieval.dense import DenseRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "BM25Retriever",
    "tokenize",
    "LSHRetriever",
    "shingle",
    "DenseRetriever",
]
