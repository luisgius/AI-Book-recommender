"""
Tests for evaluation run artifacts and structured tracing.

These tests verify:
1. LLMCallTrace creation and immutability
2. generate_run_id() produces correct format
3. save_run_artifact() writes valid JSON with nested types
4. list_run_artifacts() reads and summarizes saved runs
5. _EvaluationEncoder handles UUID, datetime, set
6. Round-trip: save -> load -> verify fields
"""

import json
import re
from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

import pytest

from app.evaluation.types import (
    EvaluationRunArtifact,
    EvaluationResult,
    QueryEvaluationResult,
    LLMCallTrace,
    _EvaluationEncoder,
    generate_run_id,
    save_run_artifact,
    list_run_artifacts,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_aggregate() -> EvaluationResult:
    """Create a minimal valid EvaluationResult for testing."""
    return EvaluationResult(
        ndcg_at_10=0.72,
        recall_at_100=0.85,
        mrr=0.65,
        ild_at_10=0.45,
        num_queries=5,
        per_query_metrics={
            "q1": {"ndcg_at_10": 0.8, "mrr": 0.5},
        },
    )


def _make_query_result() -> QueryEvaluationResult:
    """Create a minimal QueryEvaluationResult for testing."""
    return QueryEvaluationResult(
        query_id="q1",
        query_text="books like 1984",
        ndcg_at_10=0.8,
        precision_at_5=0.6,
        recall_at_10=0.7,
        mrr=0.5,
        ild=0.4,
        latency_ms=120,
        retrieved_book_ids=[str(uuid4()), str(uuid4())],
    )


def _make_artifact(run_id: str = "2026-01-31_abc12345") -> EvaluationRunArtifact:
    """Create a complete artifact for testing."""
    return EvaluationRunArtifact(
        run_id=run_id,
        timestamp=datetime(2026, 1, 31, 14, 30, 0, tzinfo=UTC),
        prompt_versions={"query_understanding": "v1.0"},
        model_name="gpt-4o-mini",
        per_query_results=[_make_query_result()],
        aggregate_metrics=_make_aggregate(),
        git_commit="abc1234",
        failures=[],
        llm_traces=[
            LLMCallTrace(
                operation="generate_explanation",
                model="gpt-4o-mini",
                latency_ms=1250.5,
                success=True,
                query_id="q1",
                book_id=str(uuid4()),
            ),
        ],
    )


# =============================================================================
# Tests: LLMCallTrace
# =============================================================================


class TestLLMCallTrace:
    """Tests for LLMCallTrace dataclass."""

    def test_create_successful_trace(self):
        """Should create a trace for a successful LLM call."""
        trace = LLMCallTrace(
            operation="generate_explanation",
            model="gpt-4o-mini",
            latency_ms=1200.0,
            success=True,
            query_id="q1",
        )
        assert trace.operation == "generate_explanation"
        assert trace.success is True
        assert trace.error is None

    def test_create_failed_trace(self):
        """Should create a trace for a failed LLM call."""
        trace = LLMCallTrace(
            operation="judge_explanation",
            model="gpt-4o-mini",
            latency_ms=500.0,
            success=False,
            error="Rate limit exceeded",
        )
        assert trace.success is False
        assert trace.error == "Rate limit exceeded"

    def test_trace_is_immutable(self):
        """LLMCallTrace should be frozen."""
        trace = LLMCallTrace(
            operation="test",
            model="test",
            latency_ms=0,
            success=True,
        )
        with pytest.raises(Exception):
            trace.operation = "changed"

    def test_optional_fields_default_to_none(self):
        """Optional fields should default to None."""
        trace = LLMCallTrace(
            operation="test",
            model="test",
            latency_ms=0,
            success=True,
        )
        assert trace.query_id is None
        assert trace.book_id is None
        assert trace.error is None
        assert trace.prompt_version is None
        assert trace.input_tokens is None
        assert trace.output_tokens is None


# =============================================================================
# Tests: generate_run_id
# =============================================================================


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_format_matches_pattern(self):
        """Run ID should match YYYY-MM-DD_<8hex> pattern."""
        run_id = generate_run_id()
        pattern = r"^\d{4}-\d{2}-\d{2}_[0-9a-f]{8}$"
        assert re.match(pattern, run_id), f"'{run_id}' doesn't match expected pattern"

    def test_ids_are_unique(self):
        """Two consecutive calls should produce different IDs."""
        id1 = generate_run_id()
        id2 = generate_run_id()
        assert id1 != id2

    def test_date_prefix_is_today(self):
        """The date prefix should be today's date."""
        run_id = generate_run_id()
        date_part = run_id.split("_")[0]
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert date_part == today


# =============================================================================
# Tests: _EvaluationEncoder
# =============================================================================


class TestEvaluationEncoder:
    """Tests for the custom JSON encoder."""

    def test_encodes_datetime(self):
        """Should serialize datetime to ISO 8601 string."""
        dt = datetime(2026, 1, 31, 14, 30, 0, tzinfo=UTC)
        result = json.dumps({"ts": dt}, cls=_EvaluationEncoder)
        assert "2026-01-31T14:30:00" in result

    def test_encodes_uuid(self):
        """Should serialize UUID to string."""
        uid = uuid4()
        result = json.dumps({"id": uid}, cls=_EvaluationEncoder)
        assert str(uid) in result

    def test_encodes_set(self):
        """Should serialize set to sorted list."""
        result = json.dumps({"s": {"c", "a", "b"}}, cls=_EvaluationEncoder)
        parsed = json.loads(result)
        assert parsed["s"] == ["a", "b", "c"]

    def test_raises_for_unknown_type(self):
        """Should raise TypeError for unsupported types."""
        with pytest.raises(TypeError):
            json.dumps({"x": object()}, cls=_EvaluationEncoder)


# =============================================================================
# Tests: save_run_artifact and list_run_artifacts
# =============================================================================


class TestSaveAndListArtifacts:
    """Tests for artifact persistence (save + list)."""

    def test_save_creates_json_file(self, tmp_path):
        """save_run_artifact should create a JSON file in the runs dir."""
        artifact = _make_artifact()
        runs_dir = str(tmp_path / "runs")

        result_path = save_run_artifact(artifact, runs_dir=runs_dir)

        assert result_path.exists()
        assert result_path.suffix == ".json"
        assert result_path.name == f"{artifact.run_id}.json"

    def test_saved_json_is_valid(self, tmp_path):
        """The saved file should contain valid JSON."""
        artifact = _make_artifact()
        runs_dir = str(tmp_path / "runs")

        result_path = save_run_artifact(artifact, runs_dir=runs_dir)

        with open(result_path, "r") as f:
            data = json.load(f)

        assert data["run_id"] == artifact.run_id
        assert data["model_name"] == "gpt-4o-mini"
        assert data["git_commit"] == "abc1234"

    def test_saved_json_contains_nested_types(self, tmp_path):
        """Nested datetime and UUID should be serialized as strings."""
        artifact = _make_artifact()
        runs_dir = str(tmp_path / "runs")

        result_path = save_run_artifact(artifact, runs_dir=runs_dir)

        with open(result_path, "r") as f:
            data = json.load(f)

        # timestamp should be an ISO string, not a datetime object
        assert isinstance(data["timestamp"], str)
        assert "2026-01-31" in data["timestamp"]

        # per_query_results should have book IDs as strings
        assert len(data["per_query_results"]) == 1
        assert isinstance(data["per_query_results"][0]["retrieved_book_ids"][0], str)

    def test_saved_json_contains_llm_traces(self, tmp_path):
        """LLM traces should be included in the saved artifact."""
        artifact = _make_artifact()
        runs_dir = str(tmp_path / "runs")

        result_path = save_run_artifact(artifact, runs_dir=runs_dir)

        with open(result_path, "r") as f:
            data = json.load(f)

        assert len(data["llm_traces"]) == 1
        assert data["llm_traces"][0]["operation"] == "generate_explanation"
        assert data["llm_traces"][0]["success"] is True

    def test_creates_directory_if_missing(self, tmp_path):
        """Should create the runs directory if it doesn't exist."""
        deep_path = str(tmp_path / "a" / "b" / "c" / "runs")

        artifact = _make_artifact()
        result_path = save_run_artifact(artifact, runs_dir=deep_path)

        assert result_path.exists()

    def test_list_empty_directory(self, tmp_path):
        """list_run_artifacts should return empty list for empty dir."""
        runs_dir = str(tmp_path / "empty_runs")
        result = list_run_artifacts(runs_dir=runs_dir)
        assert result == []

    def test_list_nonexistent_directory(self, tmp_path):
        """list_run_artifacts should return empty list for missing dir."""
        result = list_run_artifacts(runs_dir=str(tmp_path / "doesnt_exist"))
        assert result == []

    def test_round_trip_save_and_list(self, tmp_path):
        """Saving an artifact and listing should return its summary."""
        runs_dir = str(tmp_path / "runs")
        artifact = _make_artifact("2026-01-31_test1234")

        save_run_artifact(artifact, runs_dir=runs_dir)
        runs = list_run_artifacts(runs_dir=runs_dir)

        assert len(runs) == 1
        assert runs[0]["run_id"] == "2026-01-31_test1234"
        assert runs[0]["model_name"] == "gpt-4o-mini"
        assert runs[0]["git_commit"] == "abc1234"
        assert runs[0]["num_queries"] == 1

    def test_list_multiple_runs_sorted_newest_first(self, tmp_path):
        """Multiple runs should be listed newest first (reverse sorted)."""
        runs_dir = str(tmp_path / "runs")

        # Save two artifacts with different dates in the run_id
        artifact_old = _make_artifact("2026-01-01_old00001")
        artifact_new = _make_artifact("2026-01-31_new00001")

        save_run_artifact(artifact_old, runs_dir=runs_dir)
        save_run_artifact(artifact_new, runs_dir=runs_dir)

        runs = list_run_artifacts(runs_dir=runs_dir)

        assert len(runs) == 2
        # Newest first (reverse alpha sort on filename)
        assert runs[0]["run_id"] == "2026-01-31_new00001"
        assert runs[1]["run_id"] == "2026-01-01_old00001"

    def test_list_skips_corrupt_json(self, tmp_path):
        """Corrupt JSON files should be skipped without error."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # Write a corrupt file
        (runs_dir / "corrupt.json").write_text("not valid json{{{")

        # Write a valid artifact
        artifact = _make_artifact("2026-01-31_valid001")
        save_run_artifact(artifact, runs_dir=str(runs_dir))

        runs = list_run_artifacts(runs_dir=str(runs_dir))

        # Should only return the valid one
        assert len(runs) == 1
        assert runs[0]["run_id"] == "2026-01-31_valid001"
