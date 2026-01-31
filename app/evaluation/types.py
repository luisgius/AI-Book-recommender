"""
Value objects for evaluation domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
from uuid import UUID


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