"""
pipeline.py
-----------
End-to-end Retrieval-Augmented Generation (RAG) pipeline.

Flow:
  query
    └─> HybridRetriever  (BM25 + LSH + Dense → RRF)
          └─> CrossEncoderReranker  (top-N reranked)
                └─> LLM prompt construction  (query + evidence passages)
                      └─> LLM generation  (diagnostic answer)

Responsibilities (to be implemented):
  - Wire together retriever, reranker, and LLM caller
  - Build the prompt template with retrieved evidence
  - Call the LLM API / local model
  - Return structured output: answer + grounding passages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RAGResult:
    """Structured output from a single RAG query."""

    query: str
    answer: str
    retrieved_chunks: List[str] = field(default_factory=list)
    reranked_chunks: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """Full RAG pipeline: retrieval → reranking → generation.

    Attributes:
        retriever: HybridRetriever instance.
        reranker: CrossEncoderReranker instance (optional).
        llm: LLM caller instance (to be defined in a later module).
        top_k_retrieve: Candidates to fetch before reranking.
        top_k_rerank: Passages to pass to the LLM after reranking.
    """

    def __init__(
        self,
        retriever=None,
        reranker=None,
        llm=None,
        top_k_retrieve: int = 100,
        top_k_rerank: int = 5,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

    def build_prompt(self, query: str, passages: List[str]) -> str:
        """Construct a prompt from a query and retrieved evidence passages.

        Args:
            query: Clinical query / diagnostic question.
            passages: List of evidence passage strings.

        Returns:
            Formatted prompt string for the LLM.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("RAGPipeline.build_prompt() is not yet implemented.")

    def run(self, query: str) -> RAGResult:
        """Execute the full RAG pipeline for a single query.

        Args:
            query: Clinical query string.

        Returns:
            RAGResult containing the generated answer and grounding passages.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("RAGPipeline.run() is not yet implemented.")

    def run_batch(self, queries: List[str]) -> List[RAGResult]:
        """Run the pipeline over a batch of queries.

        Args:
            queries: List of clinical query strings.

        Returns:
            List of RAGResult objects.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError("RAGPipeline.run_batch() is not yet implemented.")
