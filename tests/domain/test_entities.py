"""
Tests for domain entities.
"""

import pytest
from datetime import datetime
from uuid import UUID

from app.domain.utils.uuid7 import uuid7

from app.domain.entities import Book, SearchResult, Explanation
from app.domain.value_objects import BookMetadata


class TestBook:
    """Tests for the Book entity."""

    def test_create_book_with_minimum_data(self):
        """Test creating a book with only required fields."""
        book_id = uuid7()
        book = Book(
            id=book_id,
            title="Clean Code",
            authors=["Robert C. Martin"],
        )

        assert book.id == book_id
        assert book.title == "Clean Code"
        assert book.authors == ["Robert C. Martin"]
        assert book.source == "unknown"
        assert book.categories == []
        assert book.description is None

    def test_create_book_factory_method(self):
        """Test creating a book using the factory method."""
        book = Book.create_new(
            title="The Pragmatic Programmer",
            authors=["Andrew Hunt", "David Thomas"],
            source="google_books",
            description="A classic programming book",
            language="en",
        )

        assert isinstance(book.id, UUID)
        assert book.title == "The Pragmatic Programmer"
        assert len(book.authors) == 2
        assert book.source == "google_books"
        assert book.description == "A classic programming book"

    def test_book_validation_empty_title(self):
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            Book(
                id=uuid7(),
                title="",
                authors=["Author"],
            )

    def test_book_validation_no_authors(self):
        """Test that book without authors raises ValueError."""
        with pytest.raises(ValueError, match="at least one author"):
            Book(
                id=uuid7(),
                title="Book Title",
                authors=[],
            )

    def test_book_validation_invalid_language(self):
        """Test that invalid language code raises ValueError."""
        with pytest.raises(ValueError, match="2-letter ISO 639-1 code"):
            Book(
                id=uuid7(),
                title="Book Title",
                authors=["Author"],
                language="english",  # Should be "en"
            )

    def test_book_equality(self):
        """Test that books with same ID are equal."""
        book_id = uuid7()
        book1 = Book(id=book_id, title="Title 1", authors=["Author 1"])
        book2 = Book(id=book_id, title="Title 2", authors=["Author 2"])

        assert book1 == book2

    def test_book_inequality(self):
        """Test that books with different IDs are not equal."""
        book1 = Book(id=uuid7(), title="Title", authors=["Author"])
        book2 = Book(id=uuid7(), title="Title", authors=["Author"])

        assert book1 != book2

    def test_book_get_published_year(self):
        """Test extracting publication year."""
        book = Book(
            id=uuid7(),
            title="Book",
            authors=["Author"],
            published_date=datetime(2020, 5, 15),
        )

        assert book.get_published_year() == 2020

    def test_book_get_published_year_none(self):
        """Test getting year when no date is set."""
        book = Book(
            id=uuid7(),
            title="Book",
            authors=["Author"],
        )

        assert book.get_published_year() is None

    def test_book_has_description(self):
        """Test description check."""
        book1 = Book(
            id=uuid7(),
            title="Book",
            authors=["Author"],
            description="A good book",
        )
        book2 = Book(
            id=uuid7(),
            title="Book",
            authors=["Author"],
            description="",
        )
        book3 = Book(
            id=uuid7(),
            title="Book",
            authors=["Author"],
        )

        assert book1.has_description() is True
        assert book2.has_description() is False
        assert book3.has_description() is False

    def test_book_get_searchable_text(self):
        """Test searchable text generation."""
        book = Book(
            id=uuid7(),
            title="Clean Code",
            authors=["Robert C. Martin"],
            description="A handbook of agile software craftsmanship",
            categories=["Programming", "Software Engineering"],
        )

        searchable = book.get_searchable_text()

        assert "Clean Code" in searchable
        assert "Robert C. Martin" in searchable
        assert "handbook of agile" in searchable
        assert "Programming" in searchable
        assert "Software Engineering" in searchable


class TestSearchResult:
    """Tests for the SearchResult entity."""

    def test_create_search_result(self):
        """Test creating a search result."""
        book = Book.create_new(title="Book", authors=["Author"])
        result = SearchResult(
            book=book,
            final_score=0.95,
            rank=1,
            source="lexical",
            lexical_score=0.95,
        )

        assert result.book == book
        assert result.final_score == 0.95
        assert result.rank == 1
        assert result.source == "lexical"
        assert result.explanation is None

    def test_search_result_with_explanation(self):
        """Test search result with explanation."""
        book = Book.create_new(title="Book", authors=["Author"])
        result = SearchResult(
            book=book,
            final_score=0.85,
            rank=2,
            source="vector",
            vector_score=0.85,
            explanation="This book matches because...",
        )

        assert result.has_explanation() is True
        assert "This book matches" in result.explanation

    def test_search_result_validation_invalid_rank(self):
        """Test that rank < 1 raises ValueError."""
        book = Book.create_new(title="Book", authors=["Author"])

        with pytest.raises(ValueError, match="rank must be >= 1"):
            SearchResult(book=book, final_score=0.5, rank=0, source="lexical")

    def test_search_result_allows_negative_score(self):
        """Test that negative scores are allowed (for some similarity metrics)."""
        book = Book.create_new(title="Book", authors=["Author"])

        # This should NOT raise an error - some metrics can produce negative scores
        result = SearchResult(book=book, final_score=-0.5, rank=1, source="vector", vector_score=-0.5)

        assert result.final_score == -0.5
        assert result.vector_score == -0.5


class TestExplanation:
    """Tests for the Explanation entity."""

    def test_create_explanation(self):
        """Test creating an explanation."""
        book_id = uuid7()
        explanation = Explanation(
            book_id=book_id,
            query_text="python programming books",
            text="This book is relevant because it covers advanced Python topics.",
            model="gpt-4",
        )

        assert explanation.book_id == book_id
        assert explanation.query_text == "python programming books"
        assert "advanced Python" in explanation.text
        assert explanation.model == "gpt-4"

    def test_explanation_validation_empty_text(self):
        """Test that empty explanation text raises ValueError."""
        with pytest.raises(ValueError, match="Explanation text cannot be empty"):
            Explanation(
                book_id=uuid7(),
                query_text="query",
                text="",
            )

    def test_explanation_validation_empty_query(self):
        """Test that empty query text raises ValueError."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            Explanation(
                book_id=uuid7(),
                query_text="",
                text="Some explanation",
            )

    def test_explanation_get_short_summary(self):
        """Test short summary generation."""
        explanation = Explanation(
            book_id=uuid7(),
            query_text="query",
            text="This is a very long explanation that should be truncated when we ask for a short summary.",
        )

        short = explanation.get_short_summary(max_length=30)

        assert len(short) <= 30
        assert short.endswith("...")
        assert "This is a very long" in short

    def test_explanation_get_short_summary_no_truncation(self):
        """Test that short text is not truncated."""
        explanation = Explanation(
            book_id=uuid7(),
            query_text="query",
            text="Short explanation.",
        )

        short = explanation.get_short_summary(max_length=100)

        assert short == "Short explanation."
        assert not short.endswith("...")
