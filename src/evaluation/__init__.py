"""
evaluation package
------------------
Retrieval-level metrics (MRR, NDCG, Recall@k) and downstream diagnostic
accuracy evaluation.
"""

from src.evaluation.reference_evidence import (
    ReferenceEvidenceCollection,
    build_reference_evidence,
    load_reference_evidence,
    save_reference_evidence,
)
from src.evaluation.retrieval_reconstruction import (
    RetrievalRunsCollection,
    compute_retrieval_metrics,
    load_all_retrievers,
    load_retrieval_metrics,
    load_retrieval_runs,
    reconstruct_retrieval_runs,
    save_retrieval_metrics,
    save_retrieval_runs,
)

__all__ = [
    "ReferenceEvidenceCollection",
    "build_reference_evidence",
    "load_reference_evidence",
    "save_reference_evidence",
    "RetrievalRunsCollection",
    "compute_retrieval_metrics",
    "load_all_retrievers",
    "load_retrieval_metrics",
    "load_retrieval_runs",
    "reconstruct_retrieval_runs",
    "save_retrieval_metrics",
    "save_retrieval_runs",
]
