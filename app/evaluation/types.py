"""
Value objects for evaluation domain.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID, uuid4


@dataclass(frozen=True)
class TestQuery:
    """
    A test query for evaluation.
    """
    query_id: str
    text: str
    category: str | None = None


@dataclass(frozen=True)
class RelevanceJudgment:
    """
    Relevance judgment for a query-book pair.

    Maps book IDs to relevance scores (0-3):
    - 3: Perfectly relevant
    - 2: Relevant
    - 1: Somewhat relevant
    - 0: Not relevant
    """
    query_id: str
    judgments: Dict[UUID, int]  # book_id -> relevance (0-3)

    def __post_init__(self) -> None:
        """Validate relevance scores are in valid range."""
        if not self.query_id or not self.query_id.strip():
            raise ValueError("query_id cannot be empty")

        for book_id, relevance in self.judgments.items():
            if not (0 <= relevance <= 3):
                raise ValueError(
                    f"Relevance score must be 0-3, got {relevance} for book {book_id}"
                )


@dataclass(frozen=True)
class EvaluationResult:
    """
    Aggregated evaluation metrics for a test run.
    """
    ndcg_at_10: float
    recall_at_100: float
    mrr: float
    ild_at_10: float
    num_queries: int
    per_query_metrics: Dict[str, Dict[str, float]]  # query_id -> metrics

    def __post_init__(self) -> None:
        """Validate metric values are in valid ranges."""
        if not (0.0 <= self.ndcg_at_10 <= 1.0):
            raise ValueError(f"ndcg_at_10 must be in [0, 1], got {self.ndcg_at_10}")

        if not (0.0 <= self.recall_at_100 <= 1.0):
            raise ValueError(f"recall_at_100 must be in [0, 1], got {self.recall_at_100}")

        if not (0.0 <= self.mrr <= 1.0):
            raise ValueError(f"mrr must be in [0, 1], got {self.mrr}")

        if not (0.0 <= self.ild_at_10 <= 1.0):
            raise ValueError(f"ild_at_10 must be in [0, 1], got {self.ild_at_10}")

        if self.num_queries < 0:
            raise ValueError(f"num_queries cannot be negative, got {self.num_queries}")

@dataclass(frozen=True)
class QueryEvaluationResult:
    """
    Results for a single test query (for run artifacts).

    Contains both IR metrics and LLM-as-judge scores.
    """
    query_id: str
    query_text: str

    # IR metrics
    ndcg_at_10: float
    precision_at_5: float
    recall_at_10: float
    mrr: float
    ild: float  # Intra-List Diversity

    # Timing
    latency_ms: int

    # Raw outputs for debugging
    retrieved_book_ids: List[str]  # UUIDs as strings for JSON serialization

    # LLM-as-judge metrics (optional - None if not evaluated)
    groundedness_score: float | None = None
    clarity_score: float | None = None
    relevance_score: float | None = None


@dataclass
class EvaluationRunArtifact:
    """
    Complete record of an evaluation run (for reproducibility).

    Stored at: data/evaluation/runs/{run_id}.json
    """
    # Run metadata - REQUIRED
    run_id: str  # UUIDv7
    timestamp: datetime

    # Configuration - REQUIRED
    prompt_versions: Dict[str, str]  # {"query_understanding": "v1.2", ...}
    model_name: str

    # Results - REQUIRED
    per_query_results: List[QueryEvaluationResult]
    aggregate_metrics: EvaluationResult  # Reuse existing type

    # Optional fields with defaults
    git_commit: str | None = None
    model_temperature: float = 0.0
    failures: List[str] = field(default_factory=list)  # Error messages
    llm_traces: List["LLMCallTrace"] = field(default_factory=list)  # Individual LLM call records


@dataclass(frozen=True)
class ExplanationJudgmentResult:
    """
    Detailed judgment result for a single explanation.

    Combines LLM-as-judge scores with deterministic citation metrics.
    """

    # Identifiers - REQUIRED
    query_id: str
    book_id: UUID

    # Citation metrics - REQUIRED (deterministic, always work)
    citation_precision: float
    citation_recall: float

    # LLM scores - OPTIONAL (may fail)
    groundedness_score: float | None = None
    groundedness_reasoning: str | None = None
    clarity_score: float | None = None
    clarity_reasoning: str | None = None
    relevance_score: float | None = None
    relevance_reasoning: str | None = None


@dataclass(frozen=True)
class LLMCallTrace:
    """
    Record of a single LLM invocation during an evaluation run.

    Captures enough information to:
    - Estimate cost (model + token counts)
    - Debug latency issues (latency_ms)
    - Track reliability (success + error)
    - Reproduce (operation + prompt_version)
    """

    operation: str
    """What was being done: 'generate_explanation', 'judge_explanation', 'extract_intent'"""

    model: str
    """Model identifier (e.g., 'gpt-4o-mini')"""

    latency_ms: float
    """Wall-clock time for this call in milliseconds"""

    success: bool
    """Whether the call completed without error"""

    query_id: str | None = None
    """Which test query this call was for (None if not query-specific)"""

    book_id: str | None = None
    """Which book this call was for (None if not book-specific)"""

    error: str | None = None
    """Error message if success=False"""

    prompt_version: str | None = None
    """Version tag of the prompt used (e.g., 'v1.2')"""

    input_tokens: int | None = None
    """Approximate input token count (None if not tracked)"""

    output_tokens: int | None = None
    """Approximate output token count (None if not tracked)"""


# =============================================================================
# Run Artifact Serialization
# =============================================================================


class _EvaluationEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for evaluation artifacts.

    Handles types that json.dump() can't serialize by default:
    - datetime -> ISO 8601 string
    - UUID -> string
    - set -> sorted list

    Used internally by save_run_artifact().
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def generate_run_id() -> str:
    """
    Generate a human-readable run ID: {date}_{short_uuid}.

    Examples: '2026-01-31_a3f7bc12', '2026-02-15_9e4d1f0a'

    The date prefix makes files sortable chronologically.
    The UUID suffix guarantees uniqueness.
    """
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    short_id = uuid4().hex[:8]
    return f"{date_str}_{short_id}"


def save_run_artifact(
    artifact: EvaluationRunArtifact,
    runs_dir: str = "data/evaluation/runs",
) -> Path:
    """
    Persist an evaluation run artifact as a JSON file.

    Creates the runs directory if it doesn't exist.
    File name: {run_id}.json

    Args:
        artifact: The completed run artifact to save
        runs_dir: Directory for run artifacts

    Returns:
        Path to the saved JSON file
    """
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    file_path = runs_path / f"{artifact.run_id}.json"

    artifact_dict = asdict(artifact)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(artifact_dict, f, indent=2, ensure_ascii=False, cls=_EvaluationEncoder)

    return file_path


def list_run_artifacts(runs_dir: str = "data/evaluation/runs") -> List[dict]:
    """
    List all saved evaluation run artifacts with summary info.

    Returns a list of dicts with: run_id, timestamp, model_name, num_queries, file_path.
    Sorted by timestamp (most recent first).

    Args:
        runs_dir: Directory containing run artifact JSON files

    Returns:
        List of summary dicts, or empty list if no runs found
    """
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        return []

    summaries = []
    for json_file in sorted(runs_path.glob("*.json"), reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            summaries.append({
                "run_id": data.get("run_id", json_file.stem),
                "timestamp": data.get("timestamp", "unknown"),
                "model_name": data.get("model_name", "unknown"),
                "git_commit": data.get("git_commit", "unknown"),
                "num_queries": len(data.get("per_query_results", [])),
                "num_failures": len(data.get("failures", [])),
                "file_path": str(json_file),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return summaries


# =============================================================================
# Negative / Adversarial Testing
# =============================================================================


NEGATIVE_TEST_CATEGORIES = frozenset({
    "out_of_catalog",
    "ambiguous",
    "contradictory",
    "gibberish",
    "prompt_injection",
})


@dataclass(frozen=True)
class NegativeTestCase:
    """
    A single test case designed to probe system robustness.

    Each case belongs to a category that defines what kind of adversarial
    input it represents and what "passing" means for that category.

    Loaded from: app/evaluation/negative_tests.json
    """

    query_id: str
    text: str
    category: str  # Must be one of NEGATIVE_TEST_CATEGORIES
    expected_behavior: str  # Human-readable description of correct handling

    def __post_init__(self) -> None:
        if self.category not in NEGATIVE_TEST_CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}', "
                f"must be one of {sorted(NEGATIVE_TEST_CATEGORIES)}"
            )
        if not self.text or not self.text.strip():
            raise ValueError("Negative test text cannot be empty")


@dataclass
class NegativeTestResult:
    """
    Result of running a single negative test case against the search pipeline.

    Captures whether the system handled the adversarial input gracefully:
    - crashed: did the pipeline raise an unhandled exception?
    - num_results: how many results were returned (0 is often correct)?
    - latency_ms: did it complete in reasonable time?
    - passed: overall verdict based on category-specific criteria
    """

    query_id: str
    category: str
    query_text: str
    crashed: bool
    error_message: str | None
    num_results: int
    latency_ms: float
    passed: bool
    failure_reason: str | None