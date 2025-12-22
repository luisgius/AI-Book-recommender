"""
Value objects for evaluation domain.
"""

from dataclasses import dataclass
from typing import Dict
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