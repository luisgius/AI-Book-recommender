"""
Tests for LLM Judge Service.

These tests verify that the LLMJudgeService correctly:
1. Computes citation metrics (deterministic)
2. Calls LLM judge and extracts scores
3. Handles LLM failures gracefully (partial results)
4. Combines results into ExplanationJudgmentResult
"""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4
from datetime import datetime, UTC

from app.domain.entities import Book, Explanation, Citation
from app.evaluation.llm_judge_service import LLMJudgeService
from app.evaluation.evaluation_service import EvaluationService
from app.evaluation.types import ExplanationJudgmentResult
from app.infrastructure.llm.schemas_judge import (
    ExplanationJudgmentLLM,
    JudgmentDimension,
)


# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture
def sample_book() -> Book:
    """Create a sample book for testing."""
    return Book(
        id=uuid4(),
        title="The Pragmatic Programmer",
        authors=["Andrew Hunt", "David Thomas"],
        description="A guide to software craftsmanship and best practices for modern developers.",
        language="en",
        categories=["Programming", "Software Engineering"],
        published_date="1999-10-20",
        source="test",
        source_id="test-123",
        metadata={}
    )


@pytest.fixture
def sample_explanation(sample_book) -> Explanation:
    """Create a sample explanation with valid citations."""
    return Explanation(
        book_id=sample_book.id,
        query_text="software development books",
        text="This book is highly relevant for software development practices.",
        citations=[
            Citation(
                book_id=sample_book.id,
                chunk_id="title",
                snippet="Pragmatic Programmer",
                relevance_score=0.9,
            ),
            Citation(
                book_id=sample_book.id,
                chunk_id="description",
                snippet="software craftsmanship",
                relevance_score=0.85,
            ),
        ],
        model="gpt-4o-mini",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_explanation_no_citations(sample_book) -> Explanation:
    """Create a sample explanation with no citations."""
    return Explanation(
        book_id=sample_book.id,
        query_text="software development books",
        text="Unable to provide a grounded explanation. No valid evidence found.",
        citations=[],
        model="gpt-4o-mini",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_llm_client_success():
    """Mock LLM client that returns successful judgments."""
    mock = Mock()

    def mock_judge_explanation(query_text, book, explanation):
        return ExplanationJudgmentLLM(
            groundedness=JudgmentDimension(
                score=4,
                reasoning="Most claims are well supported by citations"
            ),
            clarity=JudgmentDimension(
                score=5,
                reasoning="Very clear and well-structured explanation"
            ),
            relevance=JudgmentDimension(
                score=4,
                reasoning="Directly addresses the query intent"
            ),
        )

    mock.judge_explanation = mock_judge_explanation
    return mock


@pytest.fixture
def mock_llm_client_failure():
    """Mock LLM client that raises exception on judge call."""
    mock = Mock()
    mock.judge_explanation = Mock(side_effect=RuntimeError("API connection failed"))
    return mock


@pytest.fixture
def eval_service():
    """Real evaluation service for citation metrics."""
    return EvaluationService()


# ================================================================================
# TEST: LLMJudgeService initialization
# ================================================================================

def test_judge_service_initialization(mock_llm_client_success, eval_service):
    """Test that service initializes correctly."""
    service = LLMJudgeService(mock_llm_client_success, eval_service)

    assert service.llm_client is mock_llm_client_success
    assert service.eval_service is eval_service


# ================================================================================
# TEST: judge_explanation - Success path
# ================================================================================

def test_judge_explanation_success(
    mock_llm_client_success, eval_service, sample_book, sample_explanation
):
    """Test successful judgment with full results."""
    service = LLMJudgeService(mock_llm_client_success, eval_service)

    result = service.judge_explanation(
        query_id="q01",
        query_text="software development books",
        book=sample_book,
        explanation=sample_explanation,
    )

    # Verify result type
    assert isinstance(result, ExplanationJudgmentResult)

    # Verify identifiers
    assert result.query_id == "q01"
    assert result.book_id == sample_book.id

    # Verify LLM scores (from mock)
    assert result.groundedness_score == 4
    assert result.groundedness_reasoning == "Most claims are well supported by citations"
    assert result.clarity_score == 5
    assert result.clarity_reasoning == "Very clear and well-structured explanation"
    assert result.relevance_score == 4
    assert result.relevance_reasoning == "Directly addresses the query intent"

    # Verify citation metrics (computed by eval_service)
    # Both citations are valid substrings of book fields
    assert result.citation_precision == 1.0  # 2/2 valid
    assert result.citation_recall == 0.5  # 2/4 fields cited (title, description out of title, description, authors, categories)


def test_judge_explanation_returns_correct_metrics_for_valid_citations(
    mock_llm_client_success, eval_service, sample_book
):
    """Test that citation metrics are computed correctly for all-valid citations."""
    # Create explanation that cites all fields
    explanation = Explanation(
        book_id=sample_book.id,
        query_text="test query",
        text="Comprehensive explanation covering all book aspects.",
        citations=[
            Citation(
                book_id=sample_book.id,
                chunk_id="title",
                snippet="Pragmatic Programmer",
                relevance_score=0.9,
            ),
            Citation(
                book_id=sample_book.id,
                chunk_id="description",
                snippet="software craftsmanship",
                relevance_score=0.85,
            ),
            Citation(
                book_id=sample_book.id,
                chunk_id="authors",
                snippet="Andrew Hunt",
                relevance_score=0.8,
            ),
            Citation(
                book_id=sample_book.id,
                chunk_id="categories",
                snippet="Programming",
                relevance_score=0.75,
            ),
        ],
        model="gpt-4o-mini",
    )

    service = LLMJudgeService(mock_llm_client_success, eval_service)
    result = service.judge_explanation(
        query_id="q02",
        query_text="test query",
        book=sample_book,
        explanation=explanation,
    )

    # All 4 citations are valid
    assert result.citation_precision == 1.0
    # All 4 fields with content are cited
    assert result.citation_recall == 1.0


# ================================================================================
# TEST: judge_explanation - LLM Failure (Graceful Degradation)
# ================================================================================

def test_judge_explanation_llm_failure_returns_partial_result(
    mock_llm_client_failure, eval_service, sample_book, sample_explanation
):
    """Test graceful degradation when LLM call fails."""
    service = LLMJudgeService(mock_llm_client_failure, eval_service)

    result = service.judge_explanation(
        query_id="q03",
        query_text="software development books",
        book=sample_book,
        explanation=sample_explanation,
    )

    # Verify result type - should still return a result
    assert isinstance(result, ExplanationJudgmentResult)

    # Verify identifiers are present
    assert result.query_id == "q03"
    assert result.book_id == sample_book.id

    # LLM scores should be None (call failed)
    assert result.groundedness_score is None
    assert result.groundedness_reasoning is None
    assert result.clarity_score is None
    assert result.clarity_reasoning is None
    assert result.relevance_score is None
    assert result.relevance_reasoning is None

    # Citation metrics should STILL be present (deterministic, computed before LLM)
    assert result.citation_precision == 1.0  # 2/2 valid
    assert result.citation_recall == 0.5  # 2/4 fields cited


def test_judge_explanation_llm_failure_does_not_raise(
    mock_llm_client_failure, eval_service, sample_book, sample_explanation
):
    """Test that LLM failure does not raise exception."""
    service = LLMJudgeService(mock_llm_client_failure, eval_service)

    # Should not raise
    result = service.judge_explanation(
        query_id="q04",
        query_text="test",
        book=sample_book,
        explanation=sample_explanation,
    )

    # Should return a result, not raise
    assert result is not None


# ================================================================================
# TEST: judge_explanation - Edge cases
# ================================================================================

def test_judge_explanation_no_citations(
    mock_llm_client_success, eval_service, sample_book, sample_explanation_no_citations
):
    """Test judgment when explanation has no citations."""
    service = LLMJudgeService(mock_llm_client_success, eval_service)

    result = service.judge_explanation(
        query_id="q05",
        query_text="software development books",
        book=sample_book,
        explanation=sample_explanation_no_citations,
    )

    # LLM scores should still be present (LLM can judge even without citations)
    assert result.groundedness_score == 4
    assert result.clarity_score == 5
    assert result.relevance_score == 4

    # Citation metrics should reflect no citations
    assert result.citation_precision == 0.0  # No citations to validate
    assert result.citation_recall == 0.0  # No fields cited


def test_judge_explanation_with_hallucinated_citation(
    mock_llm_client_success, eval_service, sample_book
):
    """Test that hallucinated citations result in lower precision."""
    # Create explanation with one valid and one hallucinated citation
    explanation = Explanation(
        book_id=sample_book.id,
        query_text="test query",
        text="Explanation with mixed citations.",
        citations=[
            Citation(
                book_id=sample_book.id,
                chunk_id="title",
                snippet="Pragmatic Programmer",  # Valid
                relevance_score=0.9,
            ),
            Citation(
                book_id=sample_book.id,
                chunk_id="description",
                snippet="machine learning algorithms",  # HALLUCINATED - not in book
                relevance_score=0.7,
            ),
        ],
        model="gpt-4o-mini",
    )

    service = LLMJudgeService(mock_llm_client_success, eval_service)
    result = service.judge_explanation(
        query_id="q06",
        query_text="test query",
        book=sample_book,
        explanation=explanation,
    )

    # 1 valid out of 2 = 50% precision
    assert result.citation_precision == 0.5
    # 1 field cited (title) out of 4 = 25% recall (but hallucinated one counts for recall tracking)
    # Actually recall counts unique chunk_ids, so title + description = 2 fields
    assert result.citation_recall == 0.5  # 2 fields cited out of 4


def test_judge_explanation_book_without_description(
    mock_llm_client_success, eval_service
):
    """Test judgment for a book without description."""
    book_no_desc = Book(
        id=uuid4(),
        title="Short Book",
        authors=["Author"],
        description=None,  # No description
        language="en",
        categories=[],  # No categories
        source="test",
        source_id="test-456",
    )

    explanation = Explanation(
        book_id=book_no_desc.id,
        query_text="test",
        text="This book matches your query.",
        citations=[
            Citation(
                book_id=book_no_desc.id,
                chunk_id="title",
                snippet="Short Book",
                relevance_score=0.8,
            ),
        ],
        model="gpt-4o-mini",
    )

    service = LLMJudgeService(mock_llm_client_success, eval_service)
    result = service.judge_explanation(
        query_id="q07",
        query_text="test",
        book=book_no_desc,
        explanation=explanation,
    )

    # Book has 2 fields with content: title, authors
    # 1 citation (title) out of 1 = 100% precision
    assert result.citation_precision == 1.0
    # 1 field cited (title) out of 2 fields with content = 50% recall
    assert result.citation_recall == 0.5


# ================================================================================
# TEST: Citation metrics edge cases
# ================================================================================

def test_citation_precision_case_insensitive(
    mock_llm_client_success, eval_service, sample_book
):
    """Test that citation validation is case-insensitive."""
    explanation = Explanation(
        book_id=sample_book.id,
        query_text="test",
        text="Test explanation.",
        citations=[
            Citation(
                book_id=sample_book.id,
                chunk_id="title",
                snippet="PRAGMATIC PROGRAMMER",  # Different case but should match
                relevance_score=0.9,
            ),
        ],
        model="gpt-4o-mini",
    )

    service = LLMJudgeService(mock_llm_client_success, eval_service)
    result = service.judge_explanation(
        query_id="q08",
        query_text="test",
        book=sample_book,
        explanation=explanation,
    )

    # Should still be valid (case-insensitive matching)
    assert result.citation_precision == 1.0


def test_citation_precision_whitespace_tolerant(
    mock_llm_client_success, eval_service, sample_book
):
    """Test that citation validation tolerates whitespace differences."""
    explanation = Explanation(
        book_id=sample_book.id,
        query_text="test",
        text="Test explanation.",
        citations=[
            Citation(
                book_id=sample_book.id,
                chunk_id="description",
                snippet="software    craftsmanship",  # Extra whitespace
                relevance_score=0.9,
            ),
        ],
        model="gpt-4o-mini",
    )

    service = LLMJudgeService(mock_llm_client_success, eval_service)
    result = service.judge_explanation(
        query_id="q09",
        query_text="test",
        book=sample_book,
        explanation=explanation,
    )

    # Should still be valid (whitespace normalized)
    assert result.citation_precision == 1.0


# ================================================================================
# TEST: Different LLM scores
# ================================================================================

def test_judge_explanation_varies_with_llm_response(eval_service, sample_book, sample_explanation):
    """Test that different LLM responses produce different scores."""
    # Create mock that returns low scores
    mock_llm_low = Mock()
    mock_llm_low.judge_explanation = Mock(
        return_value=ExplanationJudgmentLLM(
            groundedness=JudgmentDimension(score=2, reasoning="Weak grounding"),
            clarity=JudgmentDimension(score=3, reasoning="Somewhat clear"),
            relevance=JudgmentDimension(score=1, reasoning="Off topic"),
        )
    )

    service_low = LLMJudgeService(mock_llm_low, eval_service)
    result_low = service_low.judge_explanation(
        query_id="q10",
        query_text="test",
        book=sample_book,
        explanation=sample_explanation,
    )

    # Create mock that returns high scores
    mock_llm_high = Mock()
    mock_llm_high.judge_explanation = Mock(
        return_value=ExplanationJudgmentLLM(
            groundedness=JudgmentDimension(score=5, reasoning="Perfect grounding"),
            clarity=JudgmentDimension(score=5, reasoning="Crystal clear"),
            relevance=JudgmentDimension(score=5, reasoning="Perfect match"),
        )
    )

    service_high = LLMJudgeService(mock_llm_high, eval_service)
    result_high = service_high.judge_explanation(
        query_id="q11",
        query_text="test",
        book=sample_book,
        explanation=sample_explanation,
    )

    # Verify different scores
    assert result_low.groundedness_score == 2
    assert result_high.groundedness_score == 5

    assert result_low.clarity_score == 3
    assert result_high.clarity_score == 5

    assert result_low.relevance_score == 1
    assert result_high.relevance_score == 5

    # Citation metrics should be the same (deterministic, independent of LLM)
    assert result_low.citation_precision == result_high.citation_precision
    assert result_low.citation_recall == result_high.citation_recall
