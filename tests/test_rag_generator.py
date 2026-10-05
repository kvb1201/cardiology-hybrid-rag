"""
test_rag_generator.py
---------------------
Unit and integration tests for src.rag.generator.

Coverage:
1. Experiment matrix = 975 conditions (65 queries x 5 retrievers x 3 K values).
2. Deterministic experiment IDs / keys (query_id, retriever, k).
3. Correct K-specific context construction (K=5 has 5 passages, K=10 has 10, K=20 has 20).
4. Identical prompt structure across retrievers.
5. Retriever identity is completely absent from generator prompt.
6. Qrels are completely absent from generator prompt.
7. Reference evidence is completely absent from generator prompt.
8. Successful result can be resumed/skipped.
9. Failed result is recorded explicitly with error details.
10. Dry-run performs zero API calls.
11. Output schema validation (all required fields present and valid).
12. Duplicate experiment keys are rejected / handled.
13. Context contains exactly K retrieved passages where available.
14. Context ordering matches retrieval ranking (rank 1 to K).
15. Generation configuration is centralized and enforces temperature=0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.generator import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RUNS_PATH,
    RAGGenerator,
    append_generation_record,
    build_generator_prompt,
    format_evidence_context,
    load_completed_runs,
    load_generation_config,
    run_generation_experiment,
)


@pytest.fixture(scope="module")
def runs_data():
    """Load the 975 reconstructed retrieval runs."""
    with open(DEFAULT_RUNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mock_config():
    """Config with mock provider for deterministic offline testing."""
    return {
        "provider": "mock",
        "model": "mock-clinical-model",
        "temperature": 0.0,
        "max_tokens": 512,
        "system_prompt_version": "v1.0",
        "generation_prompt_version": "v1.0",
        "concurrency": 1,
        "timeout_seconds": 10,
        "retry": {
            "max_retries": 2,
            "initial_delay_seconds": 0.01,
            "backoff_factor": 1.0,
            "max_delay_seconds": 0.05,
        },
    }


class TestGeneratorExperimentMatrix:
    """Validate the 975 condition matrix and experimental controls."""

    def test_experiment_matrix_975_conditions(self, runs_data):
        runs = runs_data["runs"]
        assert len(runs) == 975
        qids = {r["query_id"] for r in runs}
        retrievers = {r["retriever"] for r in runs}
        k_values = {r["k"] for r in runs}
        assert len(qids) == 65
        assert len(retrievers) == 5
        assert k_values == {5, 10, 20}

    def test_deterministic_experiment_ids(self, runs_data):
        runs = runs_data["runs"]
        keys = [(r["query_id"], r["retriever"], r["k"]) for r in runs]
        assert len(keys) == len(set(keys)) == 975

    def test_centralized_configuration(self):
        cfg = load_generation_config(DEFAULT_CONFIG_PATH)
        assert "provider" in cfg
        assert "model" in cfg
        assert cfg["temperature"] == 0.0
        assert cfg["max_tokens"] > 0
        assert cfg["system_prompt_version"] == "v1.0"
        assert cfg["generation_prompt_version"] == "v1.0"


class TestPromptAndContextIntegrity:
    """Validate prompt construction, neutrality, and context formatting."""

    def test_correct_k_specific_context_construction(self, runs_data):
        runs = runs_data["runs"]
        bm25_q1_k5 = next(r for r in runs if r["query_id"] == "Q001" and r["retriever"] == "bm25" and r["k"] == 5)
        bm25_q1_k10 = next(r for r in runs if r["query_id"] == "Q001" and r["retriever"] == "bm25" and r["k"] == 10)
        bm25_q1_k20 = next(r for r in runs if r["query_id"] == "Q001" and r["retriever"] == "bm25" and r["k"] == 20)

        ctx5 = format_evidence_context(bm25_q1_k5["results"])
        ctx10 = format_evidence_context(bm25_q1_k10["results"])
        ctx20 = format_evidence_context(bm25_q1_k20["results"])

        assert ctx5.count("[Evidence ") == 5
        assert ctx10.count("[Evidence ") == 10
        assert ctx20.count("[Evidence ") == 20

    def test_identical_prompt_structure_across_retrievers(self, runs_data):
        runs = runs_data["runs"]
        systems = ["bm25", "dense", "lsh", "bm25_dense_rrf", "bm25_dense_lsh_rrf"]
        for s in systems:
            run_s = next(r for r in runs if r["query_id"] == "Q001" and r["retriever"] == s and r["k"] == 5)
            ctx = format_evidence_context(run_s["results"])
            prompts = build_generator_prompt(run_s["query_text"], ctx)
            assert "Question: What pharmacological treatments are recommended after myocardial infarction?" in prompts["user_prompt"]
            assert "Evidence Passages:\n" in prompts["user_prompt"]
            assert "Please provide a concise, evidence-grounded answer:" in prompts["user_prompt"]

    def test_retriever_identity_is_absent_from_generator_prompt(self, runs_data):
        runs = runs_data["runs"]
        systems = ["bm25", "dense", "lsh", "bm25_dense_rrf", "bm25_dense_lsh_rrf"]
        for s in systems:
            run_s = next(r for r in runs if r["query_id"] == "Q001" and r["retriever"] == s and r["k"] == 5)
            ctx = format_evidence_context(run_s["results"])
            prompts = build_generator_prompt(run_s["query_text"], ctx)
            combined_prompt = (prompts["system_prompt"] + " " + prompts["user_prompt"]).lower()
            assert s not in combined_prompt
            assert "rrf" not in combined_prompt
            assert "dense" not in combined_prompt
            assert "bm25" not in combined_prompt
            assert "minhash" not in combined_prompt

    def test_qrels_and_grades_are_absent_from_generator_prompt(self, runs_data):
        runs = runs_data["runs"]
        for r in runs[:20]:
            ctx = format_evidence_context(r["results"])
            prompts = build_generator_prompt(r["query_text"], ctx)
            combined = (prompts["system_prompt"] + " " + prompts["user_prompt"]).lower()
            assert "qrels" not in combined
            assert "relevance grade" not in combined
            assert "relevance score" not in combined
            assert "relevance:" not in combined
            assert "grade 2" not in combined
            assert "grade 1" not in combined

    def test_reference_evidence_is_absent_from_generator_prompt(self, runs_data):
        runs = runs_data["runs"]
        for r in runs[:20]:
            ctx = format_evidence_context(r["results"])
            prompts = build_generator_prompt(r["query_text"], ctx)
            combined = (prompts["system_prompt"] + " " + prompts["user_prompt"]).lower()
            assert "reference evidence" not in combined
            assert "gold answer" not in combined
            assert "ground truth" not in combined

    def test_context_contains_exactly_k_passages_where_available(self, runs_data):
        runs = runs_data["runs"]
        for r in runs:
            k = r["k"]
            results = r["results"]
            ctx = format_evidence_context(results)
            if len(results) > 0:
                assert ctx.count("[Evidence ") == len(results)
                if r["retriever"] != "lsh":
                    assert ctx.count("[Evidence ") == k
            else:
                assert ctx == "[No evidence passages retrieved for this query.]"

    def test_context_ordering_matches_retrieval_ranking(self, runs_data):
        runs = runs_data["runs"]
        for r in runs[:30]:
            results = r["results"]
            if not results:
                continue
            ctx = format_evidence_context(results)
            for idx, item in enumerate(results, start=1):
                assert f"[Evidence {idx}]" in ctx


class TestExecutionResumabilityAndOutputSchema:
    """Validate dry-run, output schema, resumability, and error handling."""

    def test_dry_run_performs_zero_api_calls(self, mock_config, runs_data):
        generator = RAGGenerator(config=mock_config)
        first_run = runs_data["runs"][0]
        out = generator.execute_condition(first_run, dry_run=True)
        assert out["status"] == "dry_run"
        assert "prompts_preview" in out
        assert out["query_id"] == first_run["query_id"]

    def test_output_schema_validation(self, mock_config, runs_data):
        generator = RAGGenerator(config=mock_config)
        run_rec = runs_data["runs"][0]
        out = generator.execute_condition(run_rec, dry_run=False)

        required_keys = {
            "query_id",
            "query_text",
            "topic",
            "query_type",
            "retriever",
            "k",
            "retrieved_chunk_ids",
            "generated_answer",
            "model",
            "temperature",
            "prompt_version",
            "generation_timestamp",
            "status",
        }
        assert required_keys.issubset(out.keys())
        assert out["status"] == "success"
        assert out["temperature"] == 0.0
        assert len(out["generated_answer"]) > 0

    def test_resumability_skips_successful_runs(self, tmp_path, mock_config, runs_data):
        generator = RAGGenerator(config=mock_config)
        temp_output = tmp_path / "test_answers.jsonl"

        # Record a successful output for (Q001, bm25, 5)
        completed_rec = {
            "query_id": "Q001",
            "query_text": "text",
            "topic": "mi",
            "query_type": "treatment",
            "retriever": "bm25",
            "k": 5,
            "retrieved_chunk_ids": ["CAD01-0001"],
            "generated_answer": "pre-existing answer",
            "model": "mock",
            "temperature": 0.0,
            "prompt_version": "v1.0",
            "generation_timestamp": "2026-10-05T00:00:00Z",
            "status": "success",
        }
        append_generation_record(completed_rec, temp_output)

        completed_keys = load_completed_runs(temp_output)
        assert ("Q001", "bm25", 5) in completed_keys

        # Run experiment with limit=2
        summary = run_generation_experiment(
            generator=generator,
            runs_path=DEFAULT_RUNS_PATH,
            output_path=temp_output,
            dry_run=False,
            limit=2,
        )
        # Should execute only runs not already in completed_keys
        assert summary["success_count"] == 2
        new_completed = load_completed_runs(temp_output)
        assert len(new_completed) == 3

    def test_failed_result_recorded_explicitly(self, runs_data):
        broken_config = {
            "provider": "unspecified",
            "model": "none",
            "temperature": 0.0,
            "max_tokens": 512,
            "system_prompt_version": "v1.0",
            "generation_prompt_version": "v1.0",
        }
        generator = RAGGenerator(config=broken_config)
        run_rec = runs_data["runs"][0]
        out = generator.execute_condition(run_rec, dry_run=False)

        assert out["status"] == "failed"
        assert "error_type" in out
        assert "error_message" in out
        assert "RuntimeError" in out["error_type"]
        assert "unspecified" in out["error_message"].lower()

    def test_duplicate_experiment_keys_handled(self, tmp_path):
        temp_output = tmp_path / "test_dupes.jsonl"
        rec = {
            "query_id": "Q001",
            "retriever": "bm25",
            "k": 5,
            "status": "success",
        }
        append_generation_record(rec, temp_output)
        append_generation_record(rec, temp_output)
        completed = load_completed_runs(temp_output)
        # Set deduplicates keys
        assert len(completed) == 1
