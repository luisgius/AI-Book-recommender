"""
Tests for IngestionService.

This test suite validates the ingestion pipeline orchestration:
- Fetching books from external providers
- Deduplication logic
- Persistence to catalog repository
- Summary reporting
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from uuid import uuid4

from app.ingestion.ingestion_service import IngestionService
from app.domain.entities import Book
from app.domain.value_objects import BookMetadata, IngestionSummary


@pytest.fixture
def mock_provider():
    """Mock ExternalBooksProvider."""
    return Mock()


@pytest.fixture
def mock_catalog_repo():
    """Mock BookCatalogRepository."""
    return Mock()


@pytest.fixture
def ingestion_service(mock_provider, mock_catalog_repo):
    """IngestionService with mocked dependencies."""
    return IngestionService(
        books_provider=mock_provider,
        catalog_repo=mock_catalog_repo,
    )


@pytest.fixture
def sample_books():
    """Sample books for testing."""
    return [
        Book(
            id=uuid4(),
            title="The Pragmatic Programmer",
            authors=["Andrew Hunt", "David Thomas"],
            description="A guide to pragmatic programming",
            language="en",
            categories=["Programming"],
            published_date=datetime(1999, 10, 20),
            source="google_books",
            source_id="pragmatic-001",
            metadata=BookMetadata(isbn="0201616224"),
        ),
        Book(
            id=uuid4(),
            title="Clean Code",
            authors=["Robert C. Martin"],
            description="A handbook of agile software craftsmanship",
            language="en",
            categories=["Programming"],
            published_date=datetime(2008, 8, 1),
            source="google_books",
            source_id="clean-code-001",
            metadata=BookMetadata(isbn="0132350882"),
        ),
    ]


class TestIngestionServiceSuccess:
    """Test cases for successful ingestion scenarios."""

    def test_ingest_books_all_new(self, ingestion_service, mock_provider, mock_catalog_repo, sample_books):
        """Test ingesting books when all are new (no duplicates)."""
        # Arrange
        query = "programming"
        mock_provider.search_books.return_value = sample_books
        mock_catalog_repo.get_by_source_id.return_value = None  # No duplicates

        # Act
        summary = ingestion_service.ingest_books(query, max_results=10)

        # Assert
        assert isinstance(summary, IngestionSummary)
        assert summary.n_fetched == 2
        assert summary.n_inserted == 2
        assert summary.n_skipped == 0
        assert summary.n_errors == 0
        assert summary.query == query
        assert summary.language is None
        assert len(summary.errors) == 0

        # Verify provider was called
        mock_provider.search_books.assert_called_once_with(query, 10, None)

        # Verify catalog was queried for duplicates
        assert mock_catalog_repo.get_by_source_id.call_count == 2

        # Verify save was called for each book
        assert mock_catalog_repo.save.call_count == 2

    def test_ingest_books_with_language_filter(self, ingestion_service, mock_provider, mock_catalog_repo, sample_books):
        """Test ingesting books with language filter."""
        # Arrange
        query = "programming"
        language = "en"
        mock_provider.search_books.return_value = sample_books
        mock_catalog_repo.get_by_source_id.return_value = None

        # Act
        summary = ingestion_service.ingest_books(query, max_results=10, language=language)

        # Assert
        assert summary.n_fetched == 2
        assert summary.n_inserted == 2
        assert summary.language == language
        mock_provider.search_books.assert_called_once_with(query, 10, language)

    def test_ingest_books_with_duplicates(self, ingestion_service, mock_provider, mock_catalog_repo, sample_books):
        """Test ingesting books when some are duplicates."""
        # Arrange
        query = "programming"
        mock_provider.search_books.return_value = sample_books

        # First book is duplicate, second is new
        def get_by_source_id_side_effect(source, source_id):
            if source_id == "pragmatic-001":
                return sample_books[0]  # Duplicate
            return None  # New book

        mock_catalog_repo.get_by_source_id.side_effect = get_by_source_id_side_effect

        # Act
        summary = ingestion_service.ingest_books(query, max_results=10)

        # Assert
        assert summary.n_fetched == 2
        assert summary.n_inserted == 1  # Only second book inserted
        assert summary.n_skipped == 1  # First book skipped
        assert summary.n_errors == 0

        # Verify save was called only once (for the new book)
        assert mock_catalog_repo.save.call_count == 1

    def test_ingest_books_no_results_from_api(self, ingestion_service, mock_provider, mock_catalog_repo):
        """Test ingesting when API returns no books."""
        # Arrange
        query = "nonexistent topic"
        mock_provider.search_books.return_value = []

        # Act
        summary = ingestion_service.ingest_books(query, max_results=10)

        # Assert
        assert summary.n_fetched == 0
        assert summary.n_inserted == 0
        assert summary.n_skipped == 0
        assert summary.n_errors == 0
        assert summary.query == query

        # Verify catalog was not touched
        mock_catalog_repo.get_by_source_id.assert_not_called()
        mock_catalog_repo.save.assert_not_called()


class TestIngestionServiceErrors:
    """Test cases for error handling."""

    def test_ingest_books_api_failure(self, ingestion_service, mock_provider):
        """Test handling of API failure."""
        # Arrange
        query = "programming"
        mock_provider.search_books.side_effect = Exception("API timeout")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to fetch books from external API"):
            ingestion_service.ingest_books(query, max_results=10)

    def test_ingest_books_partial_save_failures(self, ingestion_service, mock_provider, mock_catalog_repo, sample_books):
        """Test handling of partial save failures."""
        # Arrange
        query = "programming"
        mock_provider.search_books.return_value = sample_books
        mock_catalog_repo.get_by_source_id.return_value = None  # No duplicates

        # First save succeeds, second fails
        def save_side_effect(book):
            if book.source_id == "clean-code-001":
                raise Exception("Database constraint violation")

        mock_catalog_repo.save.side_effect = save_side_effect

        # Act
        summary = ingestion_service.ingest_books(query, max_results=10)

        # Assert
        assert summary.n_fetched == 2
        assert summary.n_inserted == 1  # First book succeeded
        assert summary.n_skipped == 0
        assert summary.n_errors == 1  # Second book failed
        assert len(summary.errors) == 1
        assert "Clean Code" in summary.errors[0]
        assert "Database constraint violation" in summary.errors[0]

    def test_ingest_books_all_save_failures(self, ingestion_service, mock_provider, mock_catalog_repo, sample_books):
        """Test when all saves fail."""
        # Arrange
        query = "programming"
        mock_provider.search_books.return_value = sample_books
        mock_catalog_repo.get_by_source_id.return_value = None  # No duplicates
        mock_catalog_repo.save.side_effect = Exception("Database unavailable")

        # Act
        summary = ingestion_service.ingest_books(query, max_results=10)

        # Assert
        assert summary.n_fetched == 2
        assert summary.n_inserted == 0
        assert summary.n_skipped == 0
        assert summary.n_errors == 2
        assert len(summary.errors) == 2

    def test_ingest_books_google_books_missing_source_id_is_counted_as_error(
        self,
        ingestion_service,
        mock_provider,
        mock_catalog_repo,
    ):
        query = "programming"
        book_missing_source_id = Book(
            id=uuid4(),
            title="Missing Source Id",
            authors=["Author"],
            source="google_books",
            source_id=None,
        )
        mock_provider.search_books.return_value = [book_missing_source_id]

        summary = ingestion_service.ingest_books(query, max_results=10)

        assert summary.n_fetched == 1
        assert summary.n_inserted == 0
        assert summary.n_skipped == 0
        assert summary.n_errors == 1
        assert len(summary.errors) == 1
        assert "google_books" in summary.errors[0]
        assert "without source_id" in summary.errors[0]

        mock_catalog_repo.get_by_source_id.assert_not_called()
        mock_catalog_repo.save.assert_not_called()


class TestIngestionServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_ingest_books_max_results_zero(self, ingestion_service, mock_provider, mock_catalog_repo):
        """Test ingesting with max_results=0."""
        # Arrange
        query = "programming"
        mock_provider.search_books.return_value = []

        # Act
        summary = ingestion_service.ingest_books(query, max_results=0)

        # Assert
        mock_provider.search_books.assert_called_once_with(query, 0, None)

    def test_ingest_books_large_batch(self, ingestion_service, mock_provider, mock_catalog_repo):
        """Test ingesting a large batch of books."""
        # Arrange
        query = "programming"
        large_batch = [
            Book(
                id=uuid4(),
                title=f"Book {i}",
                authors=["Author"],
                description="Description",
                language="en",
                categories=["Programming"],
                published_date=datetime(2020, 1, 1),
                source="google_books",
                source_id=f"book-{i}",
                metadata=BookMetadata(),
            )
            for i in range(100)
        ]
        mock_provider.search_books.return_value = large_batch
        mock_catalog_repo.get_by_source_id.return_value = None

        # Act
        summary = ingestion_service.ingest_books(query, max_results=100)

        # Assert
        assert summary.n_fetched == 100
        assert summary.n_inserted == 100
        assert summary.n_skipped == 0
        assert summary.n_errors == 0
        assert mock_catalog_repo.save.call_count == 100
