"""Tests for EvaluationService.compute_precision_at_k (Precision@k with denominator k)."""

from uuid import uuid4

import pytest

from app.domain.entities import Book, SearchResult
from app.evaluation.evaluation_service import EvaluationService
from app.evaluation.types import RelevanceJudgment


def ranked(n: int):
    books = [Book(id=uuid4(), title=f"B{i}", authors=["A"]) for i in range(n)]
    return books, [SearchResult(book=b, final_score=1.0, rank=i + 1, source="hybrid") for i, b in enumerate(books)]


def test_counts_relevant_books_in_top_k():
    books, results = ranked(5)
    judgment = RelevanceJudgment("q1", {books[0].id: 3, books[2].id: 1, books[4].id: 0})
    assert EvaluationService().compute_precision_at_k(results, judgment, k=5) == pytest.approx(2 / 5)


def test_short_result_lists_are_penalised():
    books, results = ranked(2)
    judgment = RelevanceJudgment("q1", {books[0].id: 2, books[1].id: 2})
    assert EvaluationService().compute_precision_at_k(results, judgment, k=5) == pytest.approx(2 / 5)


def test_differs_from_ndcg():
    """Regression: the evaluation job stored NDCG@5 in the precision_at_5 field."""
    books, results = ranked(5)
    judgment = RelevanceJudgment("q1", {books[4].id: 3})
    service = EvaluationService()
    assert service.compute_precision_at_k(results, judgment, k=5) == pytest.approx(0.2)
    assert service.compute_ndcg(results, judgment, k=5) != pytest.approx(0.2)


def test_rejects_non_positive_k():
    _, results = ranked(1)
    with pytest.raises(ValueError):
        EvaluationService().compute_precision_at_k(results, RelevanceJudgment("q1", {}), k=0)
