"""
Tests for LLM guardrails (deterministic validation of citations).

These tests verify that the guardrail functions correctly validate citations
and prevent hallucinations by checking if snippets are actual substrings
from the book data.
"""

import pytest
from uuid import uuid4
from datetime import datetime, UTC

from app.domain.entities import Book, Explanation, Citation
from app.domain.citation_validation import (
    normalize_text,
    is_valid_snippet,
    get_book_field_text,
)
from app.infrastructure.llm.schemas import CitationLLM, GroundedExplanationLLM
from app.infrastructure.llm.guardrails import (
    filter_valid_citations,
    validate_grounded_explanation,
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
def sample_query_text() -> str:
    """Sample query text for testing."""
    return "books about software development"


# ================================================================================
# TEST: normalize_text()
# ================================================================================

def test_normalize_text_lowercase():
    """Test that normalization converts to lowercase."""
    assert normalize_text("Hello World") == "hello world"
    assert normalize_text("UPPERCASE") == "uppercase"


def test_normalize_text_whitespace_collapse():
    """Test that normalization collapses multiple whitespaces."""
    assert normalize_text("hello    world") == "hello world"
    assert normalize_text("hello\n\tworld") == "hello world"
    assert normalize_text("  hello  world  ") == "hello world"


def test_normalize_text_empty():
    """Test normalization of empty strings."""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


# ================================================================================
# TEST: get_book_field_text()
# ================================================================================

def test_get_book_field_text_title(sample_book):
    """Test retrieving title field."""
    assert get_book_field_text(sample_book, "title") == "The Pragmatic Programmer"


def test_get_book_field_text_authors(sample_book):
    """Test retrieving authors field (joined)."""
    assert get_book_field_text(sample_book, "authors") == "Andrew Hunt, David Thomas"


def test_get_book_field_text_categories(sample_book):
    """Test retrieving categories field (joined)."""
    assert get_book_field_text(sample_book, "categories") == "Programming, Software Engineering"


def test_get_book_field_text_description(sample_book):
    """Test retrieving description field."""
    expected = "A guide to software craftsmanship and best practices for modern developers."
    assert get_book_field_text(sample_book, "description") == expected


# ================================================================================
# TEST: is_valid_snippet()
# ================================================================================

def test_is_valid_snippet_exact_match():
    """Test that exact substring matches are valid."""
    field_text = "The Pragmatic Programmer"
    assert is_valid_snippet("Pragmatic", field_text) is True
    assert is_valid_snippet("The Pragmatic", field_text) is True


def test_is_valid_snippet_case_insensitive():
    """Test that validation is case-insensitive."""
    field_text = "The Pragmatic Programmer"
    assert is_valid_snippet("pragmatic", field_text) is True
    assert is_valid_snippet("PRAGMATIC", field_text) is True
    assert is_valid_snippet("PrAgMaTiC", field_text) is True


def test_is_valid_snippet_whitespace_tolerant():
    """Test that validation handles extra whitespace."""
    field_text = "The Pragmatic Programmer"
    assert is_valid_snippet("The   Pragmatic", field_text) is True
    assert is_valid_snippet("Pragmatic\nProgrammer", field_text) is True


def test_is_valid_snippet_invalid():
    """Test that non-substring snippets are invalid."""
    field_text = "The Pragmatic Programmer"
    assert is_valid_snippet("Java Expert", field_text) is False
    assert is_valid_snippet("Machine Learning", field_text) is False


def test_is_valid_snippet_empty():
    """Test that empty snippets are invalid."""
    field_text = "The Pragmatic Programmer"
    assert is_valid_snippet("", field_text) is False
    assert is_valid_snippet("   ", field_text) is False


def test_is_valid_snippet_partial_word():
    """Test that partial words are considered valid (substring matching)."""
    field_text = "The Pragmatic Programmer"
    # "Pragma" is a substring of "Pragmatic"
    assert is_valid_snippet("Pragma", field_text) is True


# ================================================================================
# TEST: filter_valid_citations()
# ================================================================================

def test_filter_valid_citations_all_valid(sample_book):
    """Test filtering when all citations are valid."""
    citations_llm = [
        CitationLLM(
            book_id=str(sample_book.id),
            chunk_id="title",
            snippet="Pragmatic Programmer",
            relevance_score=0.9
        ),
        CitationLLM(
            book_id=str(sample_book.id),
            chunk_id="authors",
            snippet="Andrew Hunt",
            relevance_score=0.8
        ),
    ]

    valid = filter_valid_citations(citations_llm, sample_book)

    assert len(valid) == 2
    assert valid[0].snippet == "Pragmatic Programmer"
    assert valid[1].snippet == "Andrew Hunt"


def test_filter_valid_citations_some_invalid(sample_book):
    """Test filtering when some citations are invalid (hallucinations)."""
    citations_llm = [
        CitationLLM(
            book_id=str(sample_book.id),
            chunk_id="title",
            snippet="Pragmatic Programmer",  # Valid
            relevance_score=0.9
        ),
        CitationLLM(
            book_id=str(sample_book.id),
            chunk_id="description",
            snippet="This book teaches machine learning",  # INVALID - hallucination
            relevance_score=0.7
        ),
        CitationLLM(
            book_id=str(sample_book.id),
            chunk_id="authors",
            snippet="Andrew Hunt",  # Valid
            relevance_score=0.8
        ),
    ]

    valid = filter_valid_citations(citations_llm, sample_book)

    # Only 2 valid citations should remain
    assert len(valid) == 2
    assert valid[0].snippet == "Pragmatic Programmer"
    assert valid[1].snippet == "Andrew Hunt"


def test_filter_valid_citations_all_invalid(sample_book):
    """Test filtering when all citations are invalid."""
    citations_llm = [
        CitationLLM(
            book_id=str(sample_book.id),
            chunk_id="title",
            snippet="Java Programming Expert",  # INVALID
            relevance_score=0.9
        ),
        CitationLLM(
            book_id=str(sample_book.id),
            chunk_id="description",
            snippet="Deep learning frameworks",  # INVALID
            relevance_score=0.8
        ),
    ]

    valid = filter_valid_citations(citations_llm, sample_book)

    assert len(valid) == 0


def test_filter_valid_citations_empty_list(sample_book):
    """Test filtering with empty citation list."""
    valid = filter_valid_citations([], sample_book)
    assert len(valid) == 0


# ================================================================================
# TEST: validate_grounded_explanation() - Main function
# ================================================================================

def test_validate_grounded_explanation_success(sample_book, sample_query_text):
    """Test successful validation with valid citations."""
    explanation_llm = GroundedExplanationLLM(
        summary="This book is highly relevant for software development.",
        reasoning="The title and description match the query about software development.",
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
            ),
        ],
        confidence=0.9
    )

    result = validate_grounded_explanation(
        explanation_llm=explanation_llm,
        book=sample_book,
        query_text=sample_query_text,
        model_name="gpt-4o-mini"
    )

    # Verify result type and basic properties
    assert isinstance(result, Explanation)
    assert result.book_id == sample_book.id
    assert result.query_text == sample_query_text
    assert result.model == "gpt-4o-mini"

    # Verify citations
    assert len(result.citations) == 2
    assert result.citations[0].snippet == "Pragmatic Programmer"
    assert result.citations[1].snippet == "software craftsmanship"

    # Verify explanation text contains reasoning (not summary)
    # validate_grounded_explanation() uses only the reasoning field
    assert "software development" in result.text.lower()


def test_validate_grounded_explanation_filters_hallucinations(sample_book, sample_query_text):
    """Test that hallucinated citations are filtered out."""
    explanation_llm = GroundedExplanationLLM(
        summary="This book is about programming.",
        reasoning="It covers software development topics.",
        citations=[
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="title",
                snippet="Pragmatic Programmer",  # Valid
                relevance_score=0.9
            ),
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="description",
                snippet="machine learning algorithms",  # INVALID - hallucination
                relevance_score=0.7
            ),
        ],
        confidence=0.8
    )

    result = validate_grounded_explanation(
        explanation_llm=explanation_llm,
        book=sample_book,
        query_text=sample_query_text,
        model_name="gpt-4o-mini"
    )

    # Only 1 valid citation should remain
    assert len(result.citations) == 1
    assert result.citations[0].snippet == "Pragmatic Programmer"


def test_validate_grounded_explanation_all_invalid_citations(sample_book, sample_query_text):
    """Test fallback when all citations are invalid (hallucinations)."""
    explanation_llm = GroundedExplanationLLM(
        summary="This book is great for software development.",
        reasoning="It has many topics related to programming and software engineering best practices.",
        citations=[
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="title",
                snippet="Java Expert Guide",  # INVALID
                relevance_score=0.8
            ),
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="description",
                snippet="deep neural networks",  # INVALID
                relevance_score=0.7
            ),
        ],
        confidence=0.6
    )

    result = validate_grounded_explanation(
        explanation_llm=explanation_llm,
        book=sample_book,
        query_text=sample_query_text,
        model_name="gpt-4o-mini"
    )

    # Should return fallback Explanation
    assert len(result.citations) == 0
    assert "Unable to provide a grounded explanation" in result.text
    assert "No valid evidence found" in result.text


def test_validate_grounded_explanation_empty_citations(sample_book, sample_query_text):
    """Test fallback when all citations are filtered out (effectively empty after guardrails)."""
    # Note: Schema requires at least 1 citation, so we provide 1 invalid citation
    # After guardrail filtering, this will result in 0 valid citations (same as empty)
    explanation_llm = GroundedExplanationLLM(
        summary="This book is relevant to your search query.",
        reasoning="It matches the query based on various factors and characteristics.",
        citations=[
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="title",
                snippet="Complete Guide to Machine Learning",  # INVALID - hallucination
                relevance_score=0.6
            )
        ],
        confidence=0.5
    )

    result = validate_grounded_explanation(
        explanation_llm=explanation_llm,
        book=sample_book,
        query_text=sample_query_text,
        model_name="gpt-4o-mini"
    )

    # Should return fallback (all citations were filtered out)
    assert len(result.citations) == 0
    assert "Unable to provide a grounded explanation" in result.text


def test_validate_grounded_explanation_low_relevance_citations(sample_book, sample_query_text):
    """Test that confidence is adjusted when citation relevance is low."""
    explanation_llm = GroundedExplanationLLM(
        summary="Somewhat relevant to the query about software development.",
        reasoning="This book shows a partial match with the search query terms.",
        citations=[
            CitationLLM(
                book_id=str(sample_book.id),
                chunk_id="title",
                snippet="Pragmatic",
                relevance_score=0.2  # Very low relevance
            ),
        ],
        confidence=0.9
    )

    result = validate_grounded_explanation(
        explanation_llm=explanation_llm,
        book=sample_book,
        query_text=sample_query_text,
        model_name="gpt-4o-mini"
    )

    # Confidence should be capped at 0.3 when avg relevance < 0.3
    assert result.get_confidence() <= 0.3
