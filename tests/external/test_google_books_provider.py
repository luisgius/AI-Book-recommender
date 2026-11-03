"""
Tests for GoogleBooksProvider.

These tests use mocks to avoid making actual HTTP requests to Google Books API.
"""

import pytest
from unittest.mock import Mock, patch
from uuid import UUID

from app.infrastructure.external.google_books_provider import GoogleBooksProvider


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def provider():
    """Create a GoogleBooksProvider instance for testing."""
    return GoogleBooksProvider(api_key=None)


@pytest.fixture
def sample_api_response():
    """Sample Google Books API response with one complete book."""
    return {
        "kind": "books#volumes",
        "totalItems": 1,
        "items": [
            {
                "id": "abc123",
                "volumeInfo": {
                    "title": "Don Quijote de la Mancha",
                    "authors": ["Miguel de Cervantes"],
                    "description": "La novela más famosa de la literatura española.",
                    "language": "es",
                    "categories": ["Fiction", "Classic"],
                    "publishedDate": "1605",
                    "publisher": "Editorial Castilla",
                    "pageCount": 1200,
                    "averageRating": 4.5,
                    "ratingsCount": 10000,
                    "industryIdentifiers": [
                        {"type": "ISBN_10", "identifier": "1234567890"},
                        {"type": "ISBN_13", "identifier": "1234567890123"},
                    ],
                    "imageLinks": {
                        "thumbnail": "https://example.com/quijote.jpg"
                    },
                    "previewLink": "https://books.google.com/quijote"
                }
            }
        ]
    }


@pytest.fixture
def minimal_api_response():
    """API response with minimal required fields only."""
    return {
        "items": [
            {
                "id": "min123",
                "volumeInfo": {
                    "title": "Minimal Book",
                    "authors": ["Unknown Author"],
                }
            }
        ]
    }


@pytest.fixture
def empty_api_response():
    """API response with no items."""
    return {
        "kind": "books#volumes",
        "totalItems": 0,
        "items": []
    }


@pytest.fixture
def incomplete_items_response():
    """API response with incomplete items (missing title or authors)."""
    return {
        "items": [
            {"id": "no_title", "volumeInfo": {"authors": ["Author"]}},
            {"id": "no_authors", "volumeInfo": {"title": "No Authors Book"}},
            {"id": "valid", "volumeInfo": {"title": "Valid Book", "authors": ["Author"]}},
        ]
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Tests for provider initialization."""

    def test_init_without_api_key(self):
        """Should initialize without API key for public access."""
        provider = GoogleBooksProvider()
        assert provider._api_key is None
        assert provider._base_url == "https://www.googleapis.com/books/v1/volumes"

    def test_init_with_api_key(self):
        """Should store API key when provided."""
        provider = GoogleBooksProvider(api_key="test_key_123")
        assert provider._api_key == "test_key_123"

    def test_get_source_name(self, provider):
        """Should return 'google_books' as source identifier."""
        assert provider.get_source_name() == "google_books"


# ============================================================================
# SEARCH VALIDATION TESTS
# ============================================================================

class TestSearchValidation:
    """Tests for search input validation."""

    def test_empty_query_raises_error(self, provider):
        """Should raise ValueError for empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            provider.search_books("")

    def test_whitespace_query_raises_error(self, provider):
        """Should raise ValueError for whitespace-only query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            provider.search_books("   ")

    def test_none_query_raises_error(self, provider):
        """Should raise ValueError for None query."""
        with pytest.raises(ValueError):
            provider.search_books(None)


# ============================================================================
# SEARCH BOOKS TESTS (WITH MOCKS)
# ============================================================================

class TestSearchBooks:
    """Tests for search_books() method with mocked HTTP requests."""

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_search_returns_books(self, mock_session_class, sample_api_response):
        """Should return list of Book entities from API response."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        # First call returns data, second call returns empty to stop pagination
        mock_response.json.side_effect = [sample_api_response, {"items": []}]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act
        books = provider.search_books("Don Quijote", max_results=1)
        
        # Assert
        assert len(books) == 1
        assert books[0].title == "Don Quijote de la Mancha"
        assert books[0].authors == ["Miguel de Cervantes"]
        assert books[0].source == "google_books"
        assert books[0].source_id == "abc123"

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_search_with_language_filter(self, mock_session_class, sample_api_response):
        """Should pass language parameter to API."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.side_effect = [sample_api_response, {"items": []}]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act
        provider.search_books("test", max_results=1, language="es")
        
        # Assert: Verify langRestrict was passed
        call_args = mock_session.get.call_args_list[0]
        params = call_args[1]["params"]
        assert params["langRestrict"] == "es"

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_search_with_api_key(self, mock_session_class, sample_api_response):
        """Should include API key in request when provided."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.side_effect = [sample_api_response, {"items": []}]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider(api_key="my_secret_key")
        
        # Act
        provider.search_books("test", max_results=1)
        
        # Assert: Verify key was passed
        call_args = mock_session.get.call_args_list[0]
        params = call_args[1]["params"]
        assert params["key"] == "my_secret_key"

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_search_returns_empty_list_when_no_results(self, mock_session_class, empty_api_response):
        """Should return empty list when API returns no items."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = empty_api_response
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act
        books = provider.search_books("nonexistent book xyz123", max_results=1)
        
        # Assert
        assert books == []

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_search_skips_incomplete_items(self, mock_session_class, incomplete_items_response):
        """Should skip items missing required fields (title or authors)."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        # Return incomplete items, then empty to stop pagination
        mock_response.json.side_effect = [incomplete_items_response, {"items": []}]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act: max_results=3 to match the 3 items in response
        books = provider.search_books("test", max_results=3)
        
        # Assert: Only valid item should be returned
        assert len(books) == 1
        assert books[0].title == "Valid Book"

    @patch("app.infrastructure.external.google_books_provider.time.sleep")
    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_search_returns_partial_results_when_later_page_fails_after_retries(
        self,
        mock_session_class,
        mock_sleep,
        sample_api_response,
    ):
        import requests

        mock_session = Mock()

        # Page 1: OK
        mock_response_ok = Mock()
        mock_response_ok.raise_for_status = Mock()
        mock_response_ok.json.return_value = sample_api_response

        # Page 2: repeated 503 -> triggers retries -> then raises
        mock_response_503 = Mock()
        mock_response_503.status_code = 503

        http_error = requests.exceptions.HTTPError(
            "503 Server Error: Service Unavailable",
            response=mock_response_503,
        )

        mock_response_503.raise_for_status = Mock(side_effect=http_error)

        # One successful call + (max_retries + 1) failures for page 2
        mock_session.get.side_effect = [
            mock_response_ok,
            mock_response_503,
            mock_response_503,
            mock_response_503,
            mock_response_503,
        ]
        mock_session_class.return_value = mock_session

        provider = GoogleBooksProvider()

        # Ask for more than one page (40/page) to force pagination
        books = provider.search_books("test", max_results=41)

        assert len(books) == 1
        assert books[0].title == "Don Quijote de la Mancha"
        assert mock_sleep.call_count == 3


# ============================================================================
# ITEM TO BOOK CONVERSION TESTS
# ============================================================================

class TestItemToBook:
    """Tests for _item_to_book() conversion method."""

    def test_converts_complete_item(self, provider, sample_api_response):
        """Should convert a complete API item to Book entity."""
        # Arrange
        item = sample_api_response["items"][0]
        
        # Act
        book = provider._item_to_book(item)
        
        # Assert
        assert book is not None
        assert book.title == "Don Quijote de la Mancha"
        assert book.authors == ["Miguel de Cervantes"]
        assert book.description == "La novela más famosa de la literatura española."
        assert book.language == "es"
        assert book.categories == ["Fiction", "Classic"]
        assert book.source == "google_books"
        assert book.source_id == "abc123"
        
        # Metadata
        assert book.metadata is not None
        assert book.metadata.isbn == "1234567890"
        assert book.metadata.isbn13 == "1234567890123"
        assert book.metadata.publisher == "Editorial Castilla"
        assert book.metadata.page_count == 1200
        assert book.metadata.average_rating == 4.5
        assert book.metadata.ratings_count == 10000

    def test_converts_minimal_item(self, provider, minimal_api_response):
        """Should convert item with only required fields."""
        # Arrange
        item = minimal_api_response["items"][0]
        
        # Act
        book = provider._item_to_book(item)
        
        # Assert
        assert book is not None
        assert book.title == "Minimal Book"
        assert book.authors == ["Unknown Author"]
        assert book.description is None
        assert book.metadata is None

    def test_returns_none_for_missing_title(self, provider):
        """Should return None when title is missing."""
        item = {"id": "x", "volumeInfo": {"authors": ["Author"]}}
        assert provider._item_to_book(item) is None

    def test_returns_none_for_missing_authors(self, provider):
        """Should return None when authors list is missing."""
        item = {"id": "x", "volumeInfo": {"title": "A Book"}}
        assert provider._item_to_book(item) is None

    def test_returns_none_for_empty_authors(self, provider):
        """Should return None when authors list is empty."""
        item = {"id": "x", "volumeInfo": {"title": "A Book", "authors": []}}
        assert provider._item_to_book(item) is None


# ============================================================================
# GET BOOK BY ID TESTS
# ============================================================================

class TestGetBookById:
    """Tests for get_book_by_id() method."""

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_returns_book_when_found(self, mock_session_class, sample_api_response):
        """Should return Book entity when ID exists."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_api_response["items"][0]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act
        book = provider.get_book_by_id("abc123")
        
        # Assert
        assert book is not None
        assert book.title == "Don Quijote de la Mancha"
        assert book.source_id == "abc123"

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_returns_none_when_not_found(self, mock_session_class):
        """Should return None when book ID doesn't exist (404)."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act
        book = provider.get_book_by_id("nonexistent_id")
        
        # Assert
        assert book is None

    def test_returns_none_for_empty_id(self, provider):
        """Should return None for empty ID without making API call."""
        assert provider.get_book_by_id("") is None
        assert provider.get_book_by_id("   ") is None

    def test_returns_none_for_none_id(self, provider):
        """Should return None for None ID."""
        assert provider.get_book_by_id(None) is None


# ============================================================================
# DATE PARSING TESTS
# ============================================================================

class TestDateParsing:
    """Tests for _parse_published_date() method."""

    def test_parse_full_date(self, provider):
        """Should parse full date format like '2023-05-15'."""
        result = provider._parse_published_date("2023-05-15")
        assert result is not None
        assert result.year == 2023
        assert result.month == 5
        assert result.day == 15

    def test_parse_year_month(self, provider):
        """Should parse year-month format like '2023-05'."""
        result = provider._parse_published_date("2023-05")
        assert result is not None
        assert result.year == 2023
        assert result.month == 5

    def test_parse_year_only(self, provider):
        """Should parse year-only format like '2023'."""
        result = provider._parse_published_date("2023")
        assert result is not None
        assert result.year == 2023

    def test_parse_invalid_returns_none(self, provider):
        """Should return None for invalid date formats."""
        assert provider._parse_published_date("not-a-date") is None
        assert provider._parse_published_date("") is None
        assert provider._parse_published_date(None) is None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in API calls."""

    @patch("app.infrastructure.external.google_books_provider.time.sleep")
    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_http_error_raises_runtime_error(self, mock_session_class, mock_sleep):
        """Should raise RuntimeError when HTTP request fails."""
        import requests
        
        # Arrange
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.RequestException("Connection failed")
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Google Books API request failed"):
            provider.search_books("test")

    @patch("app.infrastructure.external.google_books_provider.requests.Session")
    def test_invalid_json_raises_runtime_error(self, mock_session_class):
        """Should raise RuntimeError when JSON parsing fails."""
        # Arrange
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        provider = GoogleBooksProvider()
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            provider.search_books("test")
