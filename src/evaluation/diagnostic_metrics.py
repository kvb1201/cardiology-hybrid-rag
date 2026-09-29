"""
diagnostic_metrics.py
---------------------
Downstream diagnostic evaluation metrics for generated answers.

Metrics to be implemented:
  - Exact Match (EM) — strict string match after normalization
  - F1 token overlap — unigram F1 between prediction and reference
  - BERTScore — contextual similarity via BERT embeddings
  - Clinical accuracy — domain-specific label match (e.g., ICD-10 code match)
  - Faithfulness — grounding score: does the answer follow from retrieved evidence?
  - Answer Relevance — does the answer address the question?
"""

from __future__ import annotations

from typing import List, Optional


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for fair comparison.

    Args:
        text: Raw answer string.

    Returns:
        Normalized string.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("normalize_answer() is not yet implemented.")


def exact_match(prediction: str, reference: str) -> float:
    """Binary exact match after normalization.

    Args:
        prediction: Model-generated answer.
        reference: Ground-truth answer.

    Returns:
        1.0 if normalized strings match, else 0.0.

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("exact_match() is not yet implemented.")


def token_f1(prediction: str, reference: str) -> float:
    """Unigram token-level F1 between prediction and reference.

    Args:
        prediction: Model-generated answer.
        reference: Ground-truth answer.

    Returns:
        F1 score in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("token_f1() is not yet implemented.")


def corpus_exact_match(predictions: List[str], references: List[str]) -> float:
    """Mean Exact Match over a list of predictions and references.

    Args:
        predictions: List of model-generated answers.
        references: List of ground-truth answers.

    Returns:
        Mean EM score in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("corpus_exact_match() is not yet implemented.")


def corpus_token_f1(predictions: List[str], references: List[str]) -> float:
    """Mean token-level F1 over a list of predictions and references.

    Args:
        predictions: List of model-generated answers.
        references: List of ground-truth answers.

    Returns:
        Mean F1 score in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("corpus_token_f1() is not yet implemented.")


def faithfulness_score(
    answer: str,
    evidence_passages: List[str],
) -> float:
    """Estimate how well the answer is grounded in the retrieved evidence.

    Args:
        answer: Model-generated answer string.
        evidence_passages: List of retrieved passage strings used as context.

    Returns:
        Faithfulness score in [0, 1].

    Raises:
        NotImplementedError: Until this function is implemented.
    """
    raise NotImplementedError("faithfulness_score() is not yet implemented.")
