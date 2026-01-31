"""
Shared fixtures for LLM infrastructure tests.

This module provides common fixtures for testing LLM components,
with proper mocking to avoid actual API calls.
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, patch

from app.domain.entities import Book, Explanation, Citation
from app.infrastructure.llm.schemas import GroundedExplanationLLM, CitationLLM
from app.infrastructure.llm.schemas_judge import ExplanationJudgmentLLM, JudgmentDimension
from app.infrastructure.llm.schemas_query import QueryIntentLLM, ExtractedFiltersLLM, ReformulatedQueryLLM


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
def sample_query_text() -> str:
    """Sample query text for testing."""
    return "books about software development"


@pytest.fixture
def sample_explanation(sample_book, sample_query_text) -> Explanation:
    """Create a sample explanation for testing."""
    return Explanation(
        book_id=sample_book.id,
        query_text=sample_query_text,
        text="This book is highly relevant for software development.",
        citations=[
            Citation(
                book_id=sample_book.id,
                chunk_id="title",
                snippet="Pragmatic Programmer",
                relevance_score=0.9
            )
        ],
        model="gpt-4o-mini"
    )


@pytest.fixture
def mock_grounded_explanation_llm(sample_book) -> GroundedExplanationLLM:
    """Create a mock LLM response for grounded explanation."""
    return GroundedExplanationLLM(
        summary="This book is highly relevant for software development.",
        reasoning="The title 'Pragmatic Programmer' and description about software craftsmanship match the query.",
        citations=[
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="title",
                snippet="Pragmatic Programmer",
                relevance_score=0.9
            ),
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="description",
                snippet="software craftsmanship",
                relevance_score=0.85
            )
        ],
        confidence=0.9
    )


@pytest.fixture
def mock_judgment() -> ExplanationJudgmentLLM:
    """Create a mock LLM judgment response."""
    return ExplanationJudgmentLLM(
        groundedness=JudgmentDimension(score=5, reasoning="All claims supported by citations"),
        clarity=JudgmentDimension(score=4, reasoning="Clear and well-structured"),
        relevance=JudgmentDimension(score=5, reasoning="Directly addresses the query")
    )


@pytest.fixture
def mock_query_intent_llm() -> QueryIntentLLM:
    """Create a mock query intent LLM response."""
    return QueryIntentLLM(
        intent_type="recommendation",
        confidence=0.9,
        reasoning="User is looking for similar books"
    )


@pytest.fixture
def mock_extracted_filters() -> ExtractedFiltersLLM:
    """Create mock extracted filters."""
    return ExtractedFiltersLLM(
        language="en",
        category="Programming",
        min_year=None,
        max_year=None
    )


@pytest.fixture
def mock_reformulated_query() -> ReformulatedQueryLLM:
    """Create a mock reformulated query."""
    return ReformulatedQueryLLM(
        optimized_query="software development programming best practices",
        keywords=["software", "development", "programming"]
    )
