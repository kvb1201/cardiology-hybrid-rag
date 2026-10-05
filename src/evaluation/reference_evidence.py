"""
reference_evidence.py
---------------------
Deterministic reference-evidence builder for Stage 3 downstream evaluation.

In the Cardiology-Chat research experiment, no human reference answers exist.
Instead, frozen qrels relevance judgments are used to construct reference evidence:
- Primary analysis (59 queries): all grade-2 (highly relevant) chunks.
- Fallback sensitivity set (6 queries: Q007, Q013, Q017, Q021, Q055, Q061):
  constructed using grade-1 (partially relevant) chunks.

CRITICAL METHODOLOGICAL PRINCIPLE:
Reference evidence represents qrels-derived relevant passages and is NOT clinical
ground truth, gold answers, or clinical truth.

This module is 100% deterministic and operates on frozen inputs:
  - data/evaluation/queries.csv
  - data/evaluation/qrels.csv
  - data/chunks/chunks.jsonl
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

# ── Frozen input paths ────────────────────────────────────────────────────────
DEFAULT_QUERIES_PATH = Path("data/evaluation/queries.csv")
DEFAULT_QRELS_PATH   = Path("data/evaluation/qrels.csv")
DEFAULT_CHUNKS_PATH  = Path("data/chunks/chunks.jsonl")
DEFAULT_OUTPUT_PATH  = Path("data/evaluation/reference_evidence.json")

# ── Locked zero-grade-2 queries ───────────────────────────────────────────────
GRADE_1_FALLBACK_QUERY_IDS: Set[str] = {
    "Q007", "Q013", "Q017", "Q021", "Q055", "Q061"
}


# ---------------------------------------------------------------------------
# Corpus chunk loader
# ---------------------------------------------------------------------------

def load_chunks_lookup(chunks_path: str | Path = DEFAULT_CHUNKS_PATH) -> Dict[str, Dict[str, Any]]:
    """Load all chunks from chunks.jsonl into a dictionary keyed by chunk_id.

    Parameters
    ----------
    chunks_path:
        Path to data/chunks/chunks.jsonl.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from chunk_id to full chunk record.
    """
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus chunks file not found at: {path}")

    chunks: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            cid = chunk.get("chunk_id")
            if not cid:
                raise ValueError(f"Chunk at line {line_num} missing 'chunk_id'")
            chunks[cid] = chunk
    return chunks


# ---------------------------------------------------------------------------
# Core Reference Evidence Builder
# ---------------------------------------------------------------------------

def build_reference_evidence(
    queries_path: str | Path = DEFAULT_QUERIES_PATH,
    qrels_path: str | Path = DEFAULT_QRELS_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
) -> Dict[str, Any]:
    """Construct deterministic reference evidence for all 65 queries.

    Parameters
    ----------
    queries_path:
        Path to queries.csv.
    qrels_path:
        Path to qrels.csv.
    chunks_path:
        Path to chunks.jsonl.

    Returns
    -------
    Dict[str, Any]
        Structured reference evidence dictionary conforming to the locked schema.
    """
    queries_file = Path(queries_path)
    qrels_file = Path(qrels_path)
    chunks_file = Path(chunks_path)

    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found at: {queries_file}")
    if not qrels_file.exists():
        raise FileNotFoundError(f"Qrels file not found at: {qrels_file}")

    # Step 1: Load chunks lookup
    chunks_lookup = load_chunks_lookup(chunks_file)

    # Step 2: Load queries
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    if len(queries) != 65:
        raise ValueError(f"Expected exactly 65 queries, found {len(queries)}")

    # Step 3: Load qrels
    qrels_by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(qrels_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qrels_by_query[row["query_id"]].append(row)

    # Step 4: Build evidence per query
    query_evidence_list: List[Dict[str, Any]] = []
    grade_2_query_count = 0
    grade_1_fallback_count = 0

    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        topic = q.get("topic", "")
        query_type = q.get("query_type", "")

        qrels_rows = qrels_by_query.get(qid, [])
        if not qrels_rows:
            raise ValueError(f"Query {qid} has zero qrels entries.")

        # Partition into grade-2 and grade-1
        g2_rows = [r for r in qrels_rows if int(r["relevance"]) == 2]
        g1_rows = [r for r in qrels_rows if int(r["relevance"]) == 1]

        if len(g2_rows) > 0:
            if qid in GRADE_1_FALLBACK_QUERY_IDS:
                raise ValueError(
                    f"Query {qid} was expected to be in GRADE_1_FALLBACK_QUERY_IDS, "
                    f"but has {len(g2_rows)} grade-2 chunks!"
                )
            grade_2_available = True
            grade_1_fallback = False
            selected_rows = g2_rows
            grade_2_query_count += 1
        else:
            if qid not in GRADE_1_FALLBACK_QUERY_IDS:
                raise ValueError(
                    f"Query {qid} has 0 grade-2 chunks but is not in GRADE_1_FALLBACK_QUERY_IDS!"
                )
            if len(g1_rows) == 0:
                raise ValueError(f"Fallback query {qid} has 0 grade-1 chunks!")
            grade_2_available = False
            grade_1_fallback = True
            selected_rows = g1_rows
            grade_1_fallback_count += 1

        # Sort selected chunks deterministically by chunk_id
        selected_rows = sorted(selected_rows, key=lambda x: x["chunk_id"])

        # Check for duplicates
        cids = [r["chunk_id"] for r in selected_rows]
        if len(cids) != len(set(cids)):
            raise ValueError(f"Duplicate chunk IDs found for query {qid}: {cids}")

        evidence_chunks: List[Dict[str, Any]] = []
        for r in selected_rows:
            cid = r["chunk_id"]
            if cid not in chunks_lookup:
                raise KeyError(f"Chunk ID '{cid}' from qrels not found in chunks.jsonl")

            chunk_obj = chunks_lookup[cid]
            evidence_chunks.append({
                "chunk_id": cid,
                "relevance": int(r["relevance"]),
                "text": chunk_obj["text"],
                "doc_id": chunk_obj.get("doc_id", ""),
                "start_char": chunk_obj.get("start_char"),
                "end_char": chunk_obj.get("end_char"),
                "title": chunk_obj.get("title", ""),
                "source": chunk_obj.get("source", ""),
                "section": chunk_obj.get("section", ""),
                "strategy": chunk_obj.get("strategy", ""),
                "chunk_index": chunk_obj.get("chunk_index"),
                "total_chunks": chunk_obj.get("total_chunks"),
            })

        query_record = {
            "query_id": qid,
            "query_text": qtext,
            "topic": topic,
            "query_type": query_type,
            "grade_2_available": grade_2_available,
            "grade_1_fallback": grade_1_fallback,
            "n_evidence_chunks": len(evidence_chunks),
            "reference_evidence": evidence_chunks,
        }
        query_evidence_list.append(query_record)

    # Validate global counts
    if grade_2_query_count != 59:
        raise ValueError(
            f"Expected exactly 59 grade-2 queries, found {grade_2_query_count}"
        )
    if grade_1_fallback_count != 6:
        raise ValueError(
            f"Expected exactly 6 fallback queries, found {grade_1_fallback_count}"
        )

    output_data: Dict[str, Any] = {
        "metadata": {
            "schema_version": "1.0",
            "description": "Deterministic reference evidence for Stage 3 downstream RAG evaluation.",
            "methodology_disclaimer": "Reference evidence is not clinical ground truth.",
            "total_queries": len(query_evidence_list),
            "primary_analysis_queries_count": grade_2_query_count,
            "sensitivity_fallback_queries_count": grade_1_fallback_count,
            "sensitivity_fallback_query_ids": sorted(list(GRADE_1_FALLBACK_QUERY_IDS)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "queries_path": str(queries_file),
                "qrels_path": str(qrels_file),
                "chunks_path": str(chunks_file),
            },
            "evidence_selection_rules": {
                "primary": "relevance == 2 (highly relevant) chunks for 59 queries",
                "fallback": "relevance == 1 (partially relevant) chunks for 6 queries with zero grade-2 chunks",
                "ordering": "lexicographical by chunk_id",
            },
        },
        "queries": query_evidence_list,
    }

    return output_data


def save_reference_evidence(
    data: Dict[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save reference evidence data to JSON deterministically.

    Parameters
    ----------
    data:
        Data returned by :func:`build_reference_evidence`.
    output_path:
        Output file path.

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out


# ---------------------------------------------------------------------------
# Reference Evidence Collection Reader
# ---------------------------------------------------------------------------

class ReferenceEvidenceCollection:
    """Reader and lookup wrapper for reference_evidence.json."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.metadata = data.get("metadata", {})
        self.queries: List[Dict[str, Any]] = data.get("queries", [])
        self._by_id: Dict[str, Dict[str, Any]] = {
            q["query_id"]: q for q in self.queries
        }

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.queries)

    def __getitem__(self, query_id: str) -> Dict[str, Any]:
        return self._by_id[query_id]

    def get(self, query_id: str, default: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        return self._by_id.get(query_id, default)

    @property
    def grade_2_queries(self) -> List[Dict[str, Any]]:
        """Return the 59 queries eligible for primary analysis."""
        return [q for q in self.queries if q["grade_2_available"]]

    @property
    def fallback_queries(self) -> List[Dict[str, Any]]:
        """Return the 6 fallback queries reserved for sensitivity analysis."""
        return [q for q in self.queries if q["grade_1_fallback"]]


def load_reference_evidence(
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> ReferenceEvidenceCollection:
    """Load reference evidence from JSON.

    Parameters
    ----------
    path:
        Path to reference_evidence.json.

    Returns
    -------
    ReferenceEvidenceCollection
        Loaded reference evidence collection.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Reference evidence file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ReferenceEvidenceCollection(data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building reference evidence...")
    result = build_reference_evidence()
    out_file = save_reference_evidence(result)
    print(f"Saved reference evidence to: {out_file}")
    print(f"Total queries: {len(result['queries'])}")
    print(f"Grade-2 queries: {result['metadata']['primary_analysis_queries_count']}")
    print(f"Fallback queries: {result['metadata']['sensitivity_fallback_queries_count']}")
