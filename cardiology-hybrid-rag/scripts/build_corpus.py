"""
build_corpus.py
---------------
CLI entry point: ingest raw documents → clean → chunk → save JSONL.

Pipeline
--------
  data/raw/  (PDF / HTML / TXT / MD)
      |
      v  [src.ingestion.factory.load_directory]
  RawDocument list
      |
      v  [src.preprocessing.cleaner.clean_document]
  Cleaned text list  -->  data/processed/<doc_id>.txt
      |
      v  [src.preprocessing.chunker.chunk_document]
  Chunk list         -->  data/chunks/chunks.jsonl
                          (one JSONL file per run; appended per document)

Usage
-----
    python scripts/build_corpus.py \\
        --raw-dir    data/raw \\
        --proc-dir   data/processed \\
        --chunk-dir  data/chunks \\
        --chunk-size 256 \\
        --overlap    32 \\
        --strategy   sentence

    # Paragraph strategy, recursive scan:
    python scripts/build_corpus.py \\
        --raw-dir data/raw --recursive \\
        --strategy paragraph --chunk-size 512 --overlap 64

Exit codes
----------
    0  All documents processed successfully (or skipped cleanly).
    1  At least one document failed to load/process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# Resolve project root so the script works regardless of cwd
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.factory import load_directory          # noqa: E402
from src.preprocessing.cleaner import clean_document      # noqa: E402
from src.preprocessing.chunker import chunk_document, save_chunks, Chunk  # noqa: E402


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest raw cardiology documents, clean them, chunk them, "
            "and save the result as JSONL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw source documents (PDF/HTML/TXT/MD).",
    )
    parser.add_argument(
        "--proc-dir",
        type=str,
        default="data/processed",
        help="Output directory for cleaned plain-text files.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="data/chunks",
        help="Output directory for the JSONL chunk file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="chunks.jsonl",
        help="Name of the output JSONL file inside --chunk-dir.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Target chunk size in whitespace-delimited tokens.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Token overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sentence",
        choices=["sliding_window", "sentence", "paragraph"],
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recurse into subdirectories of --raw-dir.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="Skip documents that are empty after cleaning.",
    )
    parser.add_argument(
        "--no-skip-empty",
        dest="skip_empty",
        action="store_false",
        help="Do not skip empty documents (emit a warning instead).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _print_stats(
    n_loaded: int,
    n_empty: int,
    n_failed: int,
    n_chunks: int,
    total_tokens: int,
    elapsed: float,
    output_path: Path,
) -> None:
    _print_separator()
    print("CORPUS BUILD STATISTICS")
    _print_separator()
    print(f"  Documents loaded      : {n_loaded}")
    print(f"  Documents skipped     : {n_empty}  (empty after cleaning)")
    print(f"  Documents failed      : {n_failed}")
    print(f"  Total chunks produced : {n_chunks}")
    if n_chunks > 0:
        print(f"  Avg tokens per chunk  : {total_tokens / n_chunks:.1f}")
    print(f"  Output file           : {output_path}")
    print(f"  Elapsed               : {elapsed:.2f}s")
    _print_separator()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    raw_dir   = Path(args.raw_dir)
    proc_dir  = Path(args.proc_dir)
    chunk_dir = Path(args.chunk_dir)
    output_path = chunk_dir / args.output_file

    # Validate raw directory
    if not raw_dir.exists():
        print(f"[ERROR] --raw-dir '{raw_dir}' does not exist.", file=sys.stderr)
        return 1

    proc_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print()
    _print_separator("=")
    print("CARDIOLOGY CORPUS BUILDER")
    _print_separator("=")
    print(f"  raw-dir    : {raw_dir.resolve()}")
    print(f"  proc-dir   : {proc_dir.resolve()}")
    print(f"  chunk-dir  : {chunk_dir.resolve()}")
    print(f"  strategy   : {args.strategy}")
    print(f"  chunk-size : {args.chunk_size} tokens")
    print(f"  overlap    : {args.overlap} tokens")
    print(f"  recursive  : {args.recursive}")
    _print_separator("=")
    print()

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    print("[1/3] Loading documents from raw directory …")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        raw_docs = load_directory(raw_dir, recursive=args.recursive)

    n_loaded  = len(raw_docs)
    n_failed  = len(caught_warnings)

    if n_failed:
        for w in caught_warnings:
            print(f"  [WARN] {w.message}", file=sys.stderr)

    if n_loaded == 0:
        print(
            "[WARN] No supported documents found in "
            f"'{raw_dir}'. Add PDF/HTML/TXT/MD files and re-run.",
            file=sys.stderr,
        )
        _print_stats(0, 0, n_failed, 0, 0, time.perf_counter() - t0, output_path)
        return 0

    print(f"  Loaded {n_loaded} document(s).\n")

    # ------------------------------------------------------------------
    # 2. Clean + save processed text
    # ------------------------------------------------------------------
    print("[2/3] Cleaning documents …")
    all_chunks: list[Chunk] = []
    n_empty   = 0
    total_tokens = 0

    for doc in raw_docs:
        cleaned = clean_document(doc.raw_text)

        if not cleaned.strip():
            n_empty += 1
            msg = f"  [SKIP] '{Path(doc.source_path).name}' — empty after cleaning."
            if args.skip_empty:
                print(msg)
                continue
            else:
                print(f"  [WARN] '{Path(doc.source_path).name}' — empty after cleaning.")

        # Save cleaned text to data/processed/
        proc_file = proc_dir / f"{doc.doc_id}.txt"
        proc_file.write_text(cleaned, encoding="utf-8")

        # ------------------------------------------------------------------
        # 3. Chunk
        # ------------------------------------------------------------------
        chunks = chunk_document(
            text=cleaned,
            doc_id=doc.doc_id,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            strategy=args.strategy,
            title=doc.title,
            source=doc.source_path,
        )

        all_chunks.extend(chunks)
        doc_tokens = sum(len(c.text.split()) for c in chunks)
        total_tokens += doc_tokens

        print(
            f"  ✓  {Path(doc.source_path).name:<40}  "
            f"{len(chunks):>4} chunk(s)  |  "
            f"~{doc_tokens} tokens"
        )

    print()

    # ------------------------------------------------------------------
    # 4. Save JSONL
    # ------------------------------------------------------------------
    print(f"[3/3] Saving {len(all_chunks)} chunks to '{output_path}' …")
    save_chunks(all_chunks, output_path)
    print("  Done.\n")

    elapsed = time.perf_counter() - t0
    _print_stats(
        n_loaded=n_loaded,
        n_empty=n_empty,
        n_failed=n_failed,
        n_chunks=len(all_chunks),
        total_tokens=total_tokens,
        elapsed=elapsed,
        output_path=output_path,
    )

    # Print the first chunk as a sample
    if all_chunks:
        print("\nSAMPLE — first chunk:")
        _print_separator()
        sample = all_chunks[0].to_dict()
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        _print_separator()
        print()

    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
