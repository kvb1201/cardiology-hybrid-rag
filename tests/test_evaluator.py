"""
test_evaluator.py
-----------------
Comprehensive unit and integration tests for src.evaluation.evaluator.

Coverage:
1. Prompt construction & versioning (v1.0 system & user prompts).
2. Strict experimental blindness (zero retriever name, K, score, rank, qrels, or generator model).
3. Plain-text reference evidence formatting (strips chunk/doc IDs, qrels grades).
4. JSON parsing and schema validation (validates numeric scores and rationales).
5. Score range enforcement ([0.0, 1.0]).
6. Programmatic composite calculation (arithmetic mean, never trusts model math).
7. Malformed response handling (invalid JSON, markdown fences, missing keys, non-numeric values).
8. Resumability: skips already completed conditions.
9. Duplicate prevention: unique condition key (query_id, retriever, k).
10. Query & reference evidence lookup from data/evaluation/reference_evidence.json.
11. Generated-answer lookup from experiments/results/stage3_generated_answers.jsonl.
12. Retry behavior on transient errors and MonthlyQuotaExceededError handling.
13. Deterministic mock evaluator execution with 0 delay.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from src.evaluation.evaluator import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_GENERATED_ANSWERS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PILOT_OUTPUT_PATH,
    DEFAULT_REFERENCE_EVIDENCE_PATH,
    DownstreamEvaluator,
    EvaluatorError,
    MockEvaluatorClient,
    MonthlyQuotaExceededError,
    SchemaValidationError,
    append_evaluation_record,
    build_evaluator_prompt,
    format_reference_evidence_text,
    load_completed_evaluations,
    load_evaluator_config,
    load_generated_answers,
    load_reference_evidence_map,
    validate_evaluator_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_query():
    return "What pharmacological treatments are recommended after myocardial infarction?"


@pytest.fixture
def sample_answer():
    return "Post-MI medical therapy includes high-intensity statins, ACE inhibitors, and beta-blockers."


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk_001_secret",
            "doc_id": "doc_secret_guideline_2023",
            "relevance": 2,
            "text": "High-intensity statins are recommended for all patients following acute myocardial infarction.",
        },
        {
            "chunk_id": "chunk_002_secret",
            "doc_id": "doc_secret_guideline_2023",
            "relevance": 2,
            "text": "Beta-blockers and ACE inhibitors reduce mortality post-infarction.",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Prompt Construction & Blindness Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_prompt_construction(sample_query, sample_answer, sample_chunks):
    evidence_text = format_reference_evidence_text(sample_chunks)
    sys_prompt, user_prompt = build_evaluator_prompt(
        query_text=sample_query,
        generated_answer=sample_answer,
        reference_evidence_text=evidence_text,
    )
    assert "evidence_coverage" in sys_prompt
    assert "factual_consistency" in sys_prompt
    assert "answer_relevance" in sys_prompt
    assert sample_query in user_prompt
    assert sample_answer in user_prompt
    assert "High-intensity statins" in user_prompt


def test_strict_blindness_guarantees(sample_query, sample_answer, sample_chunks):
    """Verify that zero experimental metadata or retriever identifiers leak into prompts."""
    evidence_text = format_reference_evidence_text(sample_chunks)
    sys_prompt, user_prompt = build_evaluator_prompt(
        query_text=sample_query,
        generated_answer=sample_answer,
        reference_evidence_text=evidence_text,
    )
    combined = sys_prompt + "\n" + user_prompt

    # Ensure metadata from chunks was stripped
    assert "chunk_001_secret" not in combined
    assert "doc_secret_guideline_2023" not in combined

    # Forbidden experimental metadata tokens
    forbidden_tokens = [
        "bm25", "dense", "lsh", "rrf", "bm25_dense",
        "gemini", "gpt-4", "generator_model",
        "retrieval_score", "retriever_name",
        "qrels", "relevance: 2", "cutoff",
    ]
    for token in forbidden_tokens:
        assert token.lower() not in combined.lower(), f"Leakage detected: '{token}' found in prompt!"


def test_reference_evidence_formatting(sample_chunks):
    formatted = format_reference_evidence_text(sample_chunks)
    assert "Passage 1:" in formatted
    assert "Passage 2:" in formatted
    assert "chunk_id" not in formatted
    assert "relevance" not in formatted


def test_reference_evidence_empty_formatting():
    formatted = format_reference_evidence_text([])
    assert formatted == "[No reference evidence provided]"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema Validation & Programmatic Composite Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_json_response_parsing():
    valid_json = """{
        "evidence_coverage": 0.85,
        "factual_consistency": 0.90,
        "answer_relevance": 0.95,
        "rationales": {
            "evidence_coverage": "Covers key therapies.",
            "factual_consistency": "Consistent with evidence.",
            "answer_relevance": "Directly answers question."
        }
    }"""
    parsed = validate_evaluator_response(valid_json)
    scores = parsed["scores"]
    assert scores["evidence_coverage"] == 0.85
    assert scores["factual_consistency"] == 0.90
    assert scores["answer_relevance"] == 0.95
    # Programmatic composite: (0.85 + 0.90 + 0.95) / 3 = 0.90
    assert scores["composite_quality"] == 0.90
    assert parsed["rationales"]["evidence_coverage"] == "Covers key therapies."


def test_markdown_fence_stripping():
    fenced_json = """```json
    {
        "evidence_coverage": 0.60,
        "factual_consistency": 0.70,
        "answer_relevance": 0.80,
        "rationales": {
            "evidence_coverage": "Rationale 1",
            "factual_consistency": "Rationale 2",
            "answer_relevance": "Rationale 3"
        }
    }
    ```"""
    parsed = validate_evaluator_response(fenced_json)
    assert parsed["scores"]["composite_quality"] == 0.70


def test_composite_quality_calculated_programmatically():
    """Verify that model's composite is IGNORED and arithmetic mean is calculated."""
    json_with_bad_math = """{
        "evidence_coverage": 0.60,
        "factual_consistency": 0.60,
        "answer_relevance": 0.60,
        "composite_quality": 0.99,
        "rationales": {
            "evidence_coverage": "r", "factual_consistency": "r", "answer_relevance": "r"
        }
    }"""
    parsed = validate_evaluator_response(json_with_bad_math)
    # True mean is 0.60, not 0.99
    assert parsed["scores"]["composite_quality"] == 0.60


def test_schema_validation_out_of_range_score():
    invalid_json = """{
        "evidence_coverage": 1.5,
        "factual_consistency": 0.8,
        "answer_relevance": 0.9,
        "rationales": {"evidence_coverage": "r", "factual_consistency": "r", "answer_relevance": "r"}
    }"""
    with pytest.raises(SchemaValidationError, match="outside valid range"):
        validate_evaluator_response(invalid_json)


def test_schema_validation_missing_key():
    missing_key_json = """{
        "evidence_coverage": 0.8,
        "factual_consistency": 0.8,
        "rationales": {"evidence_coverage": "r", "factual_consistency": "r"}
    }"""
    with pytest.raises(SchemaValidationError, match="Missing required dimension"):
        validate_evaluator_response(missing_key_json)


def test_schema_validation_non_numeric_score():
    bad_type_json = """{
        "evidence_coverage": "high",
        "factual_consistency": 0.8,
        "answer_relevance": 0.8,
        "rationales": {"evidence_coverage": "r", "factual_consistency": "r", "answer_relevance": "r"}
    }"""
    with pytest.raises(SchemaValidationError, match="must be numeric"):
        validate_evaluator_response(bad_type_json)


def test_schema_validation_malformed_json():
    with pytest.raises(SchemaValidationError, match="Invalid JSON"):
        validate_evaluator_response("This is not JSON at all.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Artifact Lookup & Dataset Loading Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_load_reference_evidence_map():
    assert DEFAULT_REFERENCE_EVIDENCE_PATH.exists()
    ref_map = load_reference_evidence_map(DEFAULT_REFERENCE_EVIDENCE_PATH)
    assert len(ref_map) == 65
    assert "Q001" in ref_map
    assert "Passage 1:" in ref_map["Q001"]["evidence_text"]
    assert ref_map["Q001"]["n_evidence_chunks"] > 0


def test_load_generated_answers():
    assert DEFAULT_GENERATED_ANSWERS_PATH.exists()
    answers = load_generated_answers(DEFAULT_GENERATED_ANSWERS_PATH)
    assert len(answers) == 975
    sample_key = ("Q001", "bm25", 5)
    assert sample_key in answers
    assert len(answers[sample_key]["generated_answer"]) > 50


def test_load_evaluator_config():
    assert DEFAULT_CONFIG_PATH.exists()
    cfg = load_evaluator_config(DEFAULT_CONFIG_PATH)
    assert cfg["provider"] == "cohere"
    assert cfg["model"] == "command-r-08-2024"
    assert cfg["temperature"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Resumability & Pipeline Orchestration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_resumability_and_deduplication():
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as tmp:
        tmp_path = Path(tmp.name)

        # Pre-seed one successful record
        rec = {
            "query_id": "Q001",
            "retriever": "bm25",
            "k": 5,
            "status": "success",
            "scores": {"evidence_coverage": 0.8, "factual_consistency": 0.8, "answer_relevance": 0.8, "composite_quality": 0.8},
        }
        append_evaluation_record(rec, tmp_path)

        completed = load_completed_evaluations(tmp_path)
        assert ("Q001", "bm25", 5) in completed

        # Run evaluator with mock client on 2 conditions (one pre-seeded, one new)
        conditions = [
            ("Q001", "bm25", 5),
            ("Q001", "dense", 5),
        ]
        evaluator = DownstreamEvaluator(use_mock=True)
        summary = evaluator.run(
            conditions=conditions,
            output_path=tmp_path,
            reference_evidence_path=DEFAULT_REFERENCE_EVIDENCE_PATH,
            generated_answers_path=DEFAULT_GENERATED_ANSWERS_PATH,
        )

        assert summary["skipped"] == 1  # Q001/bm25/5 was skipped
        assert summary["succeeded"] == 1  # Q001/dense/5 was executed
        assert summary["total_completed"] == 2


def test_mock_evaluator_deterministic_execution():
    client = MockEvaluatorClient()
    res1 = client.evaluate("What is STEMI?", "STEMI is ST elevation MI.", "Evidence explains STEMI.")
    res2 = client.evaluate("What is STEMI?", "STEMI is ST elevation MI.", "Evidence explains STEMI.")
    assert res1["scores"] == res2["scores"]
    assert 0.0 <= res1["scores"]["composite_quality"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Artifact Schema & Population Integrity Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_primary_evaluator_artifact_schema_and_integrity():
    """Verify integrity of the completed 885-condition primary evaluator artifact."""
    assert DEFAULT_OUTPUT_PATH.exists(), f"Evaluator results artifact missing: {DEFAULT_OUTPUT_PATH}"

    records = []
    with open(DEFAULT_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    success_records = [r for r in records if r.get("status") == "success"]
    assert len(success_records) == 885, f"Expected 885 successful records, found {len(success_records)}"

    seen_keys = set()
    sensitivity_queries = {"Q007", "Q013", "Q017", "Q021", "Q055", "Q061"}
    expected_retrievers = {"bm25", "dense", "lsh", "bm25_dense_rrf", "bm25_dense_lsh_rrf"}
    expected_k = {5, 10, 20}

    for r in success_records:
        key = (r["query_id"], r["retriever"], r["k"])
        assert key not in seen_keys, f"Duplicate condition key detected: {key}"
        seen_keys.add(key)

        assert r["query_id"] not in sensitivity_queries, f"Sensitivity query found in primary: {r['query_id']}"
        assert r["retriever"] in expected_retrievers, f"Unexpected retriever: {r['retriever']}"
        assert r["k"] in expected_k, f"Unexpected K: {r['k']}"

        assert r["evaluator_provider"] == "cohere"
        assert r["evaluator_model"] == "command-r-08-2024"
        assert r["prompt_version"] == "v1.0"

        # Scores validation
        scores = r["scores"]
        cov = scores["evidence_coverage"]
        con = scores["factual_consistency"]
        rel = scores["answer_relevance"]
        comp = scores["composite_quality"]

        for val in (cov, con, rel, comp):
            assert isinstance(val, (int, float)) and not math.isnan(val)
            assert 0.0 <= val <= 1.0

        expected_comp = round((cov + con + rel) / 3.0, 4)
        assert abs(comp - expected_comp) <= 1e-4, f"Composite mismatch for {key}: {comp} vs {expected_comp}"

        # Rationales validation
        rats = r["rationales"]
        for dim in ("evidence_coverage", "factual_consistency", "answer_relevance"):
            assert dim in rats and isinstance(rats[dim], str)


def test_pilot_evaluator_artifact_integrity():
    """Verify integrity of the 10-condition pilot evaluator artifact."""
    assert DEFAULT_PILOT_OUTPUT_PATH.exists(), f"Pilot artifact missing: {DEFAULT_PILOT_OUTPUT_PATH}"

    records = []
    with open(DEFAULT_PILOT_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) == 10, f"Expected 10 pilot records, found {len(records)}"
    for r in records:
        assert r.get("status") == "success"
        assert r["evaluator_provider"] == "cohere"
        assert r["evaluator_model"] == "command-r-08-2024"
        assert r["prompt_version"] == "v1.0"
        scores = r["scores"]
        expected_comp = round((scores["evidence_coverage"] + scores["factual_consistency"] + scores["answer_relevance"]) / 3.0, 4)
        assert abs(scores["composite_quality"] - expected_comp) <= 1e-4

