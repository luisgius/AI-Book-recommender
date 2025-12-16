"""
Tests for domain value objects.
"""

import pytest

from app.domain.value_objects import SearchFilters, SearchQuery, BookMetadata


class TestSearchFilters:
    """Tests for the SearchFilters value object."""

    def test_create_empty_filters(self):
        """Test creating filters with no restrictions."""
        filters = SearchFilters()

        assert filters.language is None
        assert filters.category is None
        assert filters.min_year is None
        assert filters.max_year is None
        assert filters.is_empty() is True

    def test_create_filters_with_language(self):
        """Test creating filters with language restriction."""
        filters = SearchFilters(language="es")

        assert filters.language == "es"
        assert filters.is_empty() is False

    def test_create_filters_with_year_range(self):
        """Test creating filters with year range."""
        filters = SearchFilters(min_year=2000, max_year=2020)

        assert filters.min_year == 2000
        assert filters.max_year == 2020

    def test_filters_validation_invalid_year_range(self):
        """Test that min_year > max_year raises ValueError."""
        with pytest.raises(ValueError, match="min_year.*cannot be greater"):
            SearchFilters(min_year=2020, max_year=2000)

    def test_filters_validation_invalid_language(self):
        """Test that invalid language code raises ValueError."""
        with pytest.raises(ValueError, match="2-letter ISO 639-1 code"):
            SearchFilters(language="spanish")

    def test_filters_immutability(self):
        """Test that filters are immutable (frozen dataclass)."""
        filters = SearchFilters(language="en")

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            filters.language = "es"


class TestSearchQuery:
    """Tests for the SearchQuery value object."""

    def test_create_query_with_text_only(self):
        """Test creating a query with just text."""
        query = SearchQuery(text="python programming")

        assert query.text == "python programming"
        assert query.filters.is_empty()
        assert query.max_results == 10
        assert query.use_explanations is False

    def test_create_query_with_filters(self):
        """Test creating a query with filters."""
        filters = SearchFilters(language="en", category="Programming")
        query = SearchQuery(
            text="clean code",
            filters=filters,
            max_results=20,
            use_explanations=True,
        )

        assert query.text == "clean code"
        assert query.filters.language == "en"
        assert query.filters.category == "Programming"
        assert query.max_results == 20
        assert query.use_explanations is True

    def test_query_validation_empty_text(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SearchQuery(text="")

    def test_query_validation_whitespace_only_text(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SearchQuery(text="   ")

    def test_query_validation_max_results_too_low(self):
        """Test that max_results < 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            SearchQuery(text="query", max_results=0)

    def test_query_validation_max_results_too_high(self):
        """Test that max_results > 100 raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed 100"):
            SearchQuery(text="query", max_results=101)

    def test_query_immutability(self):
        """Test that queries are immutable."""
        query = SearchQuery(text="test")

        with pytest.raises(Exception):
            query.text = "changed"


class TestBookMetadata:
    """Tests for the BookMetadata value object."""

    def test_create_empty_metadata(self):
        """Test creating metadata with no data."""
        metadata = BookMetadata()

        assert metadata.isbn is None
        assert metadata.publisher is None
        assert metadata.page_count is None
        assert metadata.average_rating is None

    def test_create_metadata_with_all_fields(self):
        """Test creating metadata with all fields."""
        metadata = BookMetadata(
            isbn="0132350882",
            isbn13="9780132350884",
            publisher="Prentice Hall",
            page_count=464,
            average_rating=4.5,
            ratings_count=1500,
            thumbnail_url="http://example.com/cover.jpg",
            preview_link="http://example.com/preview",
        )

        assert metadata.isbn == "0132350882"
        assert metadata.publisher == "Prentice Hall"
        assert metadata.page_count == 464
        assert metadata.average_rating == 4.5
        assert metadata.ratings_count == 1500

    def test_metadata_validation_invalid_rating_too_low(self):
        """Test that rating < 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 5.0"):
            BookMetadata(average_rating=-1.0)

    def test_metadata_validation_invalid_rating_too_high(self):
        """Test that rating > 5 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 5.0"):
            BookMetadata(average_rating=6.0)

    def test_metadata_validation_negative_page_count(self):
        """Test that negative page count raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            BookMetadata(page_count=-10)

    def test_metadata_validation_negative_ratings_count(self):
        """Test that negative ratings count raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            BookMetadata(ratings_count=-5)

    def test_metadata_immutability(self):
        """Test that metadata is immutable."""
        metadata = BookMetadata(isbn="123456")

        with pytest.raises(Exception):
            metadata.isbn = "654321"
