"""
rag package
-----------
Retrieval-Augmented Generation pipeline for evidence-grounded cardiovascular
diagnostic reasoning.
"""

from src.rag.generator import (
    RAGGenerator,
    build_generator_prompt,
    format_evidence_context,
    load_completed_runs,
    load_generation_config,
    run_generation_experiment,
)
from src.rag.pipeline import RAGPipeline, RAGResult

__all__ = [
    "RAGGenerator",
    "RAGPipeline",
    "RAGResult",
    "build_generator_prompt",
    "format_evidence_context",
    "load_completed_runs",
    "load_generation_config",
    "run_generation_experiment",
]
