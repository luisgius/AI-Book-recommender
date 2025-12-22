"""
Evaluation domain module for IR metrics.
"""

from .types import TestQuery, RelevanceJudgment, EvaluationResult
from .evaluation_service import EvaluationService

__all__ = [
    "TestQuery",
    "RelevanceJudgment",
    "EvaluationResult",
    "EvaluationService",
]
