"""
run_evaluation.py
-----------------
CLI entry point: run retrieval and diagnostic evaluation, save results.

Pipeline:
  data/evaluation/queries.jsonl
    -> HybridRetriever  (configurable: BM25 / LSH / Dense / all)
    -> CrossEncoderReranker  (optional)
    -> RAGPipeline.run_batch()
    -> retrieval_metrics  (MRR, NDCG@k, Recall@k, MAP)
    -> diagnostic_metrics (EM, F1, Faithfulness)
    -> experiments/results/<run_name>/

Also supports ablation mode: iterate over retriever subsets and log
per-configuration scores for comparison.

Usage (once implemented):
    python scripts/run_evaluation.py \\
        --query-file    data/evaluation/queries.jsonl \\
        --chunk-file    data/chunks/chunks.jsonl \\
        --index-dir     indexes \\
        --run-name      baseline_hybrid \\
        --use-bm25 --use-lsh --use-dense \\
        --use-reranker \\
        --top-k-retrieve 100 \\
        --top-k-rerank   10 \\
        --ablation

NOT YET IMPLEMENTED — scaffold only.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval + diagnostic evaluation and save results."
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default="data/evaluation/queries.jsonl",
        help="JSONL file with queries and ground-truth relevant chunk IDs.",
    )
    parser.add_argument(
        "--chunk-file",
        type=str,
        default="data/chunks/chunks.jsonl",
        help="JSONL file of all corpus chunks.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="indexes",
        help="Root directory containing built indexes.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory to write evaluation output.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run_001",
        help="Subdirectory name under results-dir for this run.",
    )
    parser.add_argument("--use-bm25",   action="store_true", default=True)
    parser.add_argument("--use-lsh",    action="store_true", default=True)
    parser.add_argument("--use-dense",  action="store_true", default=True)
    parser.add_argument("--use-reranker", action="store_true", default=False)
    parser.add_argument(
        "--top-k-retrieve", type=int, default=100,
        help="Candidates fetched per sub-retriever before RRF fusion.",
    )
    parser.add_argument(
        "--top-k-rerank", type=int, default=10,
        help="Passages passed to LLM after reranking.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        default=False,
        help="Run ablation study over all retriever subsets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "run_evaluation.py main() is not yet implemented.\n"
        f"Parsed args: {args}"
    )


if __name__ == "__main__":
    main()
