"""
build_indexes.py
----------------
CLI entry point: build BM25, LSH, and FAISS indexes from text chunks.

Pipeline:
  data/chunks/  ->  BM25Retriever.build()   ->  indexes/bm25/
             ->  LSHRetriever.build()    ->  indexes/lsh/
             ->  DenseRetriever.build()  ->  indexes/faiss/

Usage (once implemented):
    python scripts/build_indexes.py \\
        --chunk-file data/chunks/chunks.jsonl \\
        --index-dir  indexes \\
        --build-bm25 \\
        --build-lsh \\
        --build-dense \\
        --dense-model sentence-transformers/all-MiniLM-L6-v2

NOT YET IMPLEMENTED — scaffold only.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval indexes (BM25, LSH, FAISS) from chunk file."
    )
    parser.add_argument(
        "--chunk-file",
        type=str,
        default="data/chunks/chunks.jsonl",
        help="Path to JSONL file produced by build_corpus.py.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="indexes",
        help="Root directory for storing built indexes.",
    )
    parser.add_argument(
        "--build-bm25",
        action="store_true",
        default=True,
        help="Build BM25 index.",
    )
    parser.add_argument(
        "--build-lsh",
        action="store_true",
        default=True,
        help="Build MinHash-LSH index.",
    )
    parser.add_argument(
        "--build-dense",
        action="store_true",
        default=True,
        help="Build FAISS dense index.",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name for dense embeddings.",
    )
    parser.add_argument(
        "--lsh-num-perm",
        type=int,
        default=128,
        help="Number of MinHash permutations.",
    )
    parser.add_argument(
        "--lsh-threshold",
        type=float,
        default=0.5,
        help="Jaccard similarity threshold for LSH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "build_indexes.py main() is not yet implemented.\n"
        f"Parsed args: {args}"
    )


if __name__ == "__main__":
    main()
