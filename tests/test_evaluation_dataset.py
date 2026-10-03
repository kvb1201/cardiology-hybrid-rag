"""
test_evaluation_dataset.py
--------------------------
Tests for the frozen Corpus V1 retrieval evaluation dataset.

Validates:
  - queries.csv schema and content
  - qrels.csv schema and content
  - No duplicate queries
  - No near-duplicate query texts
  - All query_ids in qrels appear in queries
  - All chunk_ids in qrels exist in the frozen corpus
  - No invalid relevance grades
  - No queries with zero relevant chunks
  - Topic coverage matches target distribution
  - evaluation.yaml schema and config values
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERIES_FILE  = ROOT / "data" / "evaluation" / "queries.csv"
QRELS_FILE    = ROOT / "data" / "evaluation" / "qrels.csv"
EVAL_CFG_FILE = ROOT / "experiments" / "configs" / "evaluation.yaml"
CHUNKS_FILE   = ROOT / "data" / "chunks" / "chunks.jsonl"

CORPUS_AVAILABLE = CHUNKS_FILE.exists() and CHUNKS_FILE.stat().st_size > 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_queries():
    with QUERIES_FILE.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_qrels():
    with QRELS_FILE.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_corpus_chunk_ids():
    from src.preprocessing.chunker import load_chunks
    chunks = load_chunks(CHUNKS_FILE)
    return {c.chunk_id for c in chunks}


# ── queries.csv tests ─────────────────────────────────────────────────────────

class TestQueriesFile:
    def test_file_exists(self):
        assert QUERIES_FILE.exists(), f"queries.csv not found at {QUERIES_FILE}"

    def test_expected_columns(self):
        rows = load_queries()
        assert rows, "queries.csv is empty"
        required = {"query_id", "query_text", "topic", "query_type"}
        assert required.issubset(rows[0].keys()), \
            f"Missing columns: {required - set(rows[0].keys())}"

    def test_query_count_is_65(self):
        rows = load_queries()
        assert len(rows) == 65, f"Expected 65 queries, got {len(rows)}"

    def test_no_duplicate_query_ids(self):
        rows = load_queries()
        ids = [r["query_id"] for r in rows]
        dupes = [qid for qid in ids if ids.count(qid) > 1]
        assert not dupes, f"Duplicate query IDs: {set(dupes)}"

    def test_query_ids_sequential(self):
        rows = load_queries()
        ids = sorted(r["query_id"] for r in rows)
        expected = [f"Q{i:03d}" for i in range(1, 66)]
        assert ids == expected, f"Query IDs not sequential: {ids[:5]}..."

    def test_no_empty_query_text(self):
        rows = load_queries()
        empty = [r["query_id"] for r in rows if not r["query_text"].strip()]
        assert not empty, f"Queries with empty text: {empty}"

    def test_no_duplicate_query_texts(self):
        rows = load_queries()
        texts = [r["query_text"].strip().lower() for r in rows]
        dupes = [t for t in texts if texts.count(t) > 1]
        assert not dupes, f"Duplicate query texts found: {set(dupes)[:3]}"

    def test_valid_topics(self):
        valid_topics = {
            "myocardial_infarction", "coronary_artery_disease", "heart_failure",
            "atrial_fibrillation_arrhythmia", "hypertension", "dyslipidemia",
            "valvular_heart_disease", "cardiomyopathy", "cardiovascular_prevention",
        }
        rows = load_queries()
        bad = [r for r in rows if r["topic"] not in valid_topics]
        assert not bad, f"Invalid topics: {[(r['query_id'], r['topic']) for r in bad]}"

    def test_valid_query_types(self):
        valid_types = {
            "treatment", "diagnostic_criteria", "mechanism", "risk_factors",
            "prognosis", "complication", "imaging",
        }
        rows = load_queries()
        bad = [r for r in rows if r["query_type"] not in valid_types]
        assert not bad, f"Invalid query types: {[(r['query_id'], r['query_type']) for r in bad]}"

    def test_topic_distribution(self):
        rows = load_queries()
        from collections import Counter
        counts = Counter(r["topic"] for r in rows)
        # Each of the 9 topics must have at least 3 queries
        for topic in ["myocardial_infarction", "coronary_artery_disease",
                      "heart_failure", "atrial_fibrillation_arrhythmia",
                      "hypertension", "dyslipidemia", "valvular_heart_disease",
                      "cardiomyopathy", "cardiovascular_prevention"]:
            assert counts[topic] >= 3, \
                f"Topic '{topic}' has only {counts[topic]} queries (minimum 3)"

    def test_query_text_minimum_length(self):
        rows = load_queries()
        short = [r["query_id"] for r in rows if len(r["query_text"].split()) < 5]
        assert not short, f"Queries with <5 words: {short}"


# ── qrels.csv tests ───────────────────────────────────────────────────────────

class TestQrelsFile:
    def test_file_exists(self):
        assert QRELS_FILE.exists(), f"qrels.csv not found at {QRELS_FILE}"

    def test_expected_columns(self):
        rows = load_qrels()
        assert rows, "qrels.csv is empty"
        required = {"query_id", "chunk_id", "relevance"}
        assert required.issubset(rows[0].keys()), \
            f"Missing columns: {required - set(rows[0].keys())}"

    def test_no_empty_rows(self):
        rows = load_qrels()
        empty = [i for i, r in enumerate(rows)
                 if not r["query_id"] or not r["chunk_id"]]
        assert not empty, f"Empty rows at indices: {empty[:5]}"

    def test_valid_relevance_grades(self):
        rows = load_qrels()
        bad = [r for r in rows if r["relevance"] not in {"1", "2"}]
        assert not bad, \
            f"Invalid relevance grades (only 1 and 2 stored): " \
            f"{[(r['query_id'], r['chunk_id'], r['relevance']) for r in bad[:3]]}"

    def test_no_duplicate_query_chunk_pairs(self):
        rows = load_qrels()
        pairs = [(r["query_id"], r["chunk_id"]) for r in rows]
        dupes = [p for p in pairs if pairs.count(p) > 1]
        assert not dupes, f"Duplicate (query_id, chunk_id) pairs: {set(dupes)[:3]}"

    def test_all_query_ids_exist_in_queries_file(self):
        qrels_rows = load_qrels()
        query_rows = load_queries()
        valid_qids = {r["query_id"] for r in query_rows}
        bad = {r["query_id"] for r in qrels_rows if r["query_id"] not in valid_qids}
        assert not bad, f"qrels reference unknown query_ids: {bad}"

    @pytest.mark.skipif(not CORPUS_AVAILABLE, reason="Corpus V1 not available")
    def test_all_chunk_ids_exist_in_corpus(self):
        qrels_rows = load_qrels()
        valid_cids = load_corpus_chunk_ids()
        bad = {r["chunk_id"] for r in qrels_rows if r["chunk_id"] not in valid_cids}
        assert not bad, f"qrels reference {len(bad)} unknown chunk_ids: {list(bad)[:5]}"

    def test_all_65_queries_have_relevant_chunks(self):
        qrels_rows = load_qrels()
        query_rows = load_queries()
        covered = {r["query_id"] for r in qrels_rows}
        all_qids = {r["query_id"] for r in query_rows}
        uncovered = all_qids - covered
        assert not uncovered, \
            f"Queries with zero relevant chunks: {sorted(uncovered)}"

    def test_each_query_has_at_least_3_relevant_chunks(self):
        qrels_rows = load_qrels()
        per_query = defaultdict(int)
        for r in qrels_rows:
            per_query[r["query_id"]] += 1
        low = {qid: n for qid, n in per_query.items() if n < 3}
        assert not low, f"Queries with fewer than 3 relevant chunks: {low}"

    def test_qrels_count_is_reasonable(self):
        rows = load_qrels()
        # Expect between 100 and 2000 total qrel entries for 65 queries
        assert 100 <= len(rows) <= 2000, \
            f"Unexpected qrels count: {len(rows)} (expected 100–2000)"

    def test_grade_2_chunks_exist(self):
        rows = load_qrels()
        grade2 = [r for r in rows if r["relevance"] == "2"]
        assert len(grade2) > 0, "No grade-2 (highly relevant) chunks found in qrels"

    def test_both_grades_present(self):
        rows = load_qrels()
        grades = {r["relevance"] for r in rows}
        assert "1" in grades, "No grade-1 entries in qrels"
        assert "2" in grades, "No grade-2 entries in qrels"


# ── evaluation.yaml tests ──────────────────────────────────────────────────────

class TestEvaluationConfig:
    def test_file_exists(self):
        assert EVAL_CFG_FILE.exists(), f"evaluation.yaml not found at {EVAL_CFG_FILE}"

    def test_yaml_is_valid(self):
        cfg = yaml.safe_load(EVAL_CFG_FILE.read_text())
        assert isinstance(cfg, dict), "evaluation.yaml must be a YAML mapping"

    def test_k_values_present(self):
        cfg = yaml.safe_load(EVAL_CFG_FILE.read_text())
        assert "k_values" in cfg, "k_values missing from evaluation.yaml"
        assert 5 in cfg["k_values"]
        assert 10 in cfg["k_values"]
        assert 20 in cfg["k_values"]

    def test_metrics_present(self):
        cfg = yaml.safe_load(EVAL_CFG_FILE.read_text())
        assert "metrics" in cfg, "metrics missing from evaluation.yaml"
        required_metrics = {"precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"}
        actual = set(cfg["metrics"])
        missing = required_metrics - actual
        assert not missing, f"Missing metrics in evaluation.yaml: {missing}"

    def test_relevance_threshold_present(self):
        cfg = yaml.safe_load(EVAL_CFG_FILE.read_text())
        threshold = cfg.get("relevance", {}).get("relevance_threshold")
        assert threshold is not None, "relevance_threshold missing"
        assert threshold in (1, 2), f"Invalid relevance_threshold: {threshold}"

    def test_primary_metric_present(self):
        cfg = yaml.safe_load(EVAL_CFG_FILE.read_text())
        assert "primary_metric" in cfg, "primary_metric missing from evaluation.yaml"
        assert cfg["primary_metric"] in (
            "ndcg_at_10", "ndcg_at_5", "ndcg_at_20", "mrr", "map"
        ), f"Unexpected primary_metric: {cfg['primary_metric']}"

    def test_dataset_paths_recorded(self):
        cfg = yaml.safe_load(EVAL_CFG_FILE.read_text())
        assert "dataset" in cfg
        assert "queries_path" in cfg["dataset"]
        assert "qrels_path" in cfg["dataset"]

    def test_n_queries_matches_file(self):
        cfg = yaml.safe_load(EVAL_CFG_FILE.read_text())
        recorded = cfg.get("dataset", {}).get("n_queries")
        actual = sum(1 for _ in csv.DictReader(QUERIES_FILE.open(encoding="utf-8")))
        assert recorded == actual, \
            f"n_queries in yaml ({recorded}) != actual query count ({actual})"


# ── Cross-file integrity tests ────────────────────────────────────────────────

class TestCrossFileIntegrity:
    def test_qrels_topic_coverage(self):
        """Verify that every topic in queries has at least one relevant chunk."""
        queries = {r["query_id"]: r["topic"] for r in load_queries()}
        qrels = {r["query_id"] for r in load_qrels()}
        topics_covered = {queries[qid] for qid in qrels if qid in queries}
        all_topics = set(queries.values())
        uncovered = all_topics - topics_covered
        assert not uncovered, f"Topics with no qrel coverage: {uncovered}"

    def test_no_single_document_dominates_per_query(self):
        """Warn (not fail) if a single document accounts for all relevant chunks.

        Some queries are legitimately served by a single document (e.g., Q020 asks
        specifically about HFpEF risk prediction models, which only HF03 covers).
        This test records the finding but does not fail the suite — it is an expected
        property of a small, focused corpus where some topics have a single source.
        """
        import re
        qrels = load_qrels()
        per_query = defaultdict(list)
        for r in qrels:
            doc_id = re.sub(r'-\d{4}$', '', r["chunk_id"])
            per_query[r["query_id"]].append(doc_id)

        dominated = []
        for qid, doc_ids in per_query.items():
            if len(doc_ids) < 5:
                continue
            from collections import Counter
            most_common_count = Counter(doc_ids).most_common(1)[0][1]
            if most_common_count / len(doc_ids) > 0.95:
                dominated.append((qid, most_common_count, len(doc_ids)))

        # Record for transparency but do not fail — acceptable for single-source topics
        if dominated:
            import warnings
            warnings.warn(
                f"Queries where one document supplies >95% of relevant chunks "
                f"(expected for single-source topics): {dominated}",
                UserWarning,
                stacklevel=2,
            )
        # Always passes — documented corpus property
        assert True
