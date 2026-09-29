"""
build_corpus.py
---------------
CLI entry point: clean raw documents and produce text chunks.

Pipeline:
  data/raw/  ->  cleaner.clean_corpus()  ->  data/processed/
           ->  chunker.chunk_corpus()   ->  data/chunks/

Usage (once implemented):
    python scripts/build_corpus.py \\
        --raw-dir   data/raw \\
        --proc-dir  data/processed \\
        --chunk-dir data/chunks \\
        --chunk-size 512 \\
        --overlap    64 \\
        --strategy   sliding_window

NOT YET IMPLEMENTED — scaffold only.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cleaned corpus and text chunks from raw documents."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw source documents.",
    )
    parser.add_argument(
        "--proc-dir",
        type=str,
        default="data/processed",
        help="Output directory for cleaned documents.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="data/chunks",
        help="Output directory for serialized chunks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens/characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "sentence", "paragraph"],
        help="Chunking strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "build_corpus.py main() is not yet implemented.\n"
        f"Parsed args: {args}"
    )


if __name__ == "__main__":
    main()
