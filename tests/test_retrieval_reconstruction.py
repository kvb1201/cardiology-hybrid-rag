"""
test_retrieval_reconstruction.py
--------------------------------
Unit and integration tests for src.evaluation.retrieval_reconstruction.

Coverage:
1. All 65 queries processed across all 5 retrievers and 3 K values (975 runs).
2. All 5 retrievers present: BM25, Dense, LSH, SysA_BD, SysB_BDL.
3. K = 5, 10, 20.
4. Exactly K results where possible (BM25, Dense, Hybrid always return K; LSH returns <= K).
5. Ranks are 1..len(results).
6. No duplicate chunk IDs within any ranked list.
7. Deterministic repeated execution.
8. All retrieved chunk IDs resolve to the frozen corpus.
9. Metrics reproduce Stage 2 aggregate values within tolerance.
10. Collection reader interface (lookup, filtering).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reference_evidence import load_chunks_lookup, DEFAULT_CHUNKS_PATH
from src.evaluation.retrieval_reconstruction import (
    DEFAULT_METRICS_OUTPUT_PATH,
    DEFAULT_RUNS_OUTPUT_PATH,
    K_VALUES,
    RETRIEVER_ORDER,
    STAGE2_BASELINE_METRICS,
    RetrievalRunsCollection,
    compute_retrieval_metrics,
    load_all_retrievers,
    load_retrieval_metrics,
    load_retrieval_runs,
    reconstruct_retrieval_runs,
)


@pytest.fixture(scope="module")
def runs_data():
    """Load or generate retrieval runs data."""
    if DEFAULT_RUNS_OUTPUT_PATH.exists():
        with open(DEFAULT_RUNS_OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return reconstruct_retrieval_runs()


@pytest.fixture(scope="module")
def metrics_data(runs_data):
    """Load or compute retrieval metrics data."""
    if DEFAULT_METRICS_OUTPUT_PATH.exists():
        with open(DEFAULT_METRICS_OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return compute_retrieval_metrics(runs_data)


@pytest.fixture(scope="module")
def corpus_chunks():
    """Load corpus chunks lookup."""
    return load_chunks_lookup(DEFAULT_CHUNKS_PATH)


class TestRetrievalReconstructionIntegrity:
    """Test the structure and deterministic properties of reconstructed retrieval runs."""

    def test_total_runs_count(self, runs_data):
        runs = runs_data["runs"]
        # 65 queries * 5 retrievers * 3 K values = 975 runs
        assert len(runs) == 975
        assert runs_data["metadata"]["total_runs"] == 975

    def test_all_65_queries_processed(self, runs_data):
        expected_qids = {f"Q{i:03d}" for i in range(1, 66)}
        runs = runs_data["runs"]
        for ret_id in RETRIEVER_ORDER:
            for k in K_VALUES:
                matching_qids = {
                    r["query_id"] for r in runs if r["retriever"] == ret_id and r["k"] == k
                }
                assert matching_qids == expected_qids, f"Missing queries for {ret_id} at k={k}"

    def test_all_5_retrievers_processed(self, runs_data):
        runs = runs_data["runs"]
        retrievers_found = {r["retriever"] for r in runs}
        assert retrievers_found == set(RETRIEVER_ORDER)

    def test_k_values_present(self, runs_data):
        runs = runs_data["runs"]
        k_values_found = {r["k"] for r in runs}
        assert k_values_found == {5, 10, 20}

    def test_results_count_where_possible(self, runs_data):
        """BM25, Dense, and Hybrid should have exactly K results. LSH may have fewer."""
        runs = runs_data["runs"]
        for r in runs:
            k = r["k"]
            ret_id = r["retriever"]
            results = r["results"]
            if ret_id in {"bm25", "dense", "bm25_dense_rrf", "bm25_dense_lsh_rrf"}:
                assert len(results) == k, (
                    f"Expected exactly {k} results for {ret_id} on query {r['query_id']}, got {len(results)}"
                )
            elif ret_id == "lsh":
                assert len(results) <= k, (
                    f"Expected at most {k} results for LSH on query {r['query_id']}, got {len(results)}"
                )

    def test_ranks_are_sequential_1_to_len(self, runs_data):
        runs = runs_data["runs"]
        for r in runs:
            results = r["results"]
            ranks = [item["rank"] for item in results]
            expected_ranks = list(range(1, len(results) + 1))
            assert ranks == expected_ranks, (
                f"Non-sequential ranks for query {r['query_id']} ({r['retriever']}, k={r['k']}): {ranks}"
            )

    def test_no_duplicate_chunks_within_ranked_list(self, runs_data):
        runs = runs_data["runs"]
        for r in runs:
            cids = [item["chunk_id"] for item in r["results"]]
            assert len(cids) == len(set(cids)), (
                f"Duplicate chunk IDs found in query {r['query_id']} ({r['retriever']}, k={r['k']})"
            )

    def test_retrieved_chunks_resolve_to_corpus(self, runs_data, corpus_chunks):
        runs = runs_data["runs"]
        for r in runs:
            for item in r["results"]:
                cid = item["chunk_id"]
                assert cid in corpus_chunks, (
                    f"Unknown chunk {cid} retrieved by {r['retriever']} for query {r['query_id']}"
                )
                assert item["text"] == corpus_chunks[cid]["text"], (
                    f"Text mismatch for retrieved chunk {cid}"
                )
                assert item["doc_id"] == corpus_chunks[cid]["doc_id"], (
                    f"Doc ID mismatch for retrieved chunk {cid}"
                )

    def test_deterministic_repeated_execution(self):
        """Verify that searching the same query twice produces bit-for-bit identical chunk IDs and scores."""
        retrievers = load_all_retrievers()
        test_queries = [
            "pharmacological treatments for heart failure",
            "atrial fibrillation catheter ablation recurrence risk",
        ]
        for q in test_queries:
            for ret_id, ret in retrievers.items():
                res1 = [(r.chunk_id, r.score) for r in ret.search(q, top_k=10)]
                res2 = [(r.chunk_id, r.score) for r in ret.search(q, top_k=10)]
                assert res1 == res2, f"Nondeterministic search on {ret_id} for query '{q}'"


class TestStage2MetricReproduction:
    """Validate that reconstructed retrieval runs reproduce frozen Stage 2 baseline metrics."""

    def test_stage2_validation_flag(self, metrics_data):
        val = metrics_data["stage2_validation"]
        assert val["all_systems_passed"] is True, "Stage 2 baseline validation failed!"

    @pytest.mark.parametrize("system_id", RETRIEVER_ORDER)
    def test_individual_system_baseline_metrics(self, metrics_data, system_id):
        val_system = metrics_data["stage2_validation"]["systems"][system_id]
        assert val_system["status"] == "PASSED", f"System {system_id} failed baseline validation"

        expected = STAGE2_BASELINE_METRICS[system_id]
        agg = metrics_data["aggregate_metrics"][system_id]

        for metric_name, expected_value in expected.items():
            actual_value = agg[metric_name]
            delta = abs(actual_value - expected_value)
            # Allow tolerance for standard float rounding
            assert delta <= 0.0015, (
                f"Metric {metric_name} for {system_id}: expected {expected_value}, "
                f"got {actual_value} (delta {delta:.5f} exceeds tolerance 0.0015)"
            )

    def test_per_query_metrics_count(self, metrics_data):
        per_q = metrics_data["per_query_metrics"]
        assert len(per_q) == 975, f"Expected 975 per-query metric records, got {len(per_q)}"


class TestRetrievalRunsCollectionLoader:
    """Test the RetrievalRunsCollection lookup interface."""

    def test_collection_access(self):
        coll = load_retrieval_runs(DEFAULT_RUNS_OUTPUT_PATH)
        assert len(coll) == 975

        run = coll.get_run("Q001", "bm25", 10)
        assert run is not None
        assert run["query_id"] == "Q001"
        assert run["retriever"] == "bm25"
        assert run["k"] == 10
        assert len(run["results"]) == 10

        q_runs = coll.get_runs_for_query("Q001")
        assert len(q_runs) == 15  # 5 retrievers * 3 K values

        ret_runs = coll.get_runs_for_retriever("dense", k=5)
        assert len(ret_runs) == 65  # 65 queries
