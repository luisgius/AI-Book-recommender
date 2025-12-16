"""
Tests for GoogleBooksClient adapter.

Uses FakeSession and FakeResponse to test without network calls.
Covers: query validation, parsing, date formats, 404 handling.
"""

import pytest
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from app.infrastructure.external.google_books_client import GoogleBooksClient


# =============================================================================
# Fake HTTP Session and Response for testing
# =============================================================================


class FakeResponse:
    """Fake HTTP response for testing."""

    def __init__(
        self,
        json_data: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        raise_on_json: bool = False,
    ):
        self._json_data = json_data or {}
        self.status_code = status_code
        self._raise_on_json = raise_on_json

    def json(self) -> Dict[str, Any]:
        if self._raise_on_json:
            raise ValueError("Invalid JSON")
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 404:
            raise Exception(f"HTTP {self.status_code}")


class FakeSession:
    """Fake HTTP session for testing GoogleBooksClient."""

    def __init__(self, response: Optional[FakeResponse] = None):
        self._response = response or FakeResponse()
        self.last_url: Optional[str] = None
        self.last_params: Optional[Dict[str, Any]] = None

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> FakeResponse:
        self.last_url = url
        self.last_params = params
        return self._response


# =============================================================================
# Test fixtures
# =============================================================================


def _make_volume(
    volume_id: str = "vol123",
    title: str = "Test Book",
    authors: Optional[list] = None,
    description: Optional[str] = None,
    language: Optional[str] = None,
    categories: Optional[list] = None,
    published_date: Optional[str] = None,
    publisher: Optional[str] = None,
    page_count: Optional[int] = None,
    isbn_10: Optional[str] = None,
    isbn_13: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper to create a Google Books volume JSON structure."""
    volume_info: Dict[str, Any] = {"title": title}

    if authors is not None:
        volume_info["authors"] = authors
    if description is not None:
        volume_info["description"] = description
    if language is not None:
        volume_info["language"] = language
    if categories is not None:
        volume_info["categories"] = categories
    if published_date is not None:
        volume_info["publishedDate"] = published_date
    if publisher is not None:
        volume_info["publisher"] = publisher
    if page_count is not None:
        volume_info["pageCount"] = page_count

    if isbn_10 or isbn_13:
        identifiers = []
        if isbn_10:
            identifiers.append({"type": "ISBN_10", "identifier": isbn_10})
        if isbn_13:
            identifiers.append({"type": "ISBN_13", "identifier": isbn_13})
        volume_info["industryIdentifiers"] = identifiers

    return {"id": volume_id, "volumeInfo": volume_info}


# =============================================================================
# Tests: Input validation
# =============================================================================


class TestSearchBooksValidation:
    """Tests for search_books input validation."""

    def test_empty_query_raises_value_error(self):
        """Empty query should raise ValueError."""
        client = GoogleBooksClient(session=FakeSession())

        with pytest.raises(ValueError, match="query cannot be empty"):
            client.search_books("")

    def test_whitespace_query_raises_value_error(self):
        """Whitespace-only query should raise ValueError."""
        client = GoogleBooksClient(session=FakeSession())

        with pytest.raises(ValueError, match="query cannot be empty"):
            client.search_books("   ")

    def test_none_query_raises_value_error(self):
        """None query should raise ValueError."""
        client = GoogleBooksClient(session=FakeSession())

        with pytest.raises(ValueError, match="query cannot be empty"):
            client.search_books(None)  # type: ignore


# =============================================================================
# Tests: API request building
# =============================================================================


class TestSearchBooksRequestBuilding:
    """Tests for correct API request construction."""

    def test_basic_query_params(self):
        """Basic search should include q and maxResults params."""
        session = FakeSession(FakeResponse({"items": []}))
        client = GoogleBooksClient(session=session)

        client.search_books("python", max_results=10)

        assert session.last_url == "https://www.googleapis.com/books/v1/volumes"
        assert session.last_params["q"] == "python"
        assert session.last_params["maxResults"] == 10

    def test_query_is_stripped(self):
        """Query should be stripped of whitespace."""
        session = FakeSession(FakeResponse({"items": []}))
        client = GoogleBooksClient(session=session)

        client.search_books("  python programming  ", max_results=5)

        assert session.last_params["q"] == "python programming"

    def test_language_filter_added(self):
        """Language filter should be added to params."""
        session = FakeSession(FakeResponse({"items": []}))
        client = GoogleBooksClient(session=session)

        client.search_books("python", language="en")

        assert session.last_params["langRestrict"] == "en"

    def test_api_key_added_when_provided(self):
        """API key should be added to params when configured."""
        session = FakeSession(FakeResponse({"items": []}))
        client = GoogleBooksClient(api_key="test-key", session=session)

        client.search_books("python")

        assert session.last_params["key"] == "test-key"

    def test_api_key_and_language_combined(self):
        """API key and language filter should both be included in params."""
        session = FakeSession(FakeResponse({"items": []}))
        client = GoogleBooksClient(api_key="my-api-key", session=session)

        client.search_books("python", language="es")

        assert session.last_params["key"] == "my-api-key"
        assert session.last_params["langRestrict"] == "es"
        assert session.last_params["q"] == "python"

    def test_max_results_capped_at_40(self):
        """max_results should be capped at API limit of 40."""
        session = FakeSession(FakeResponse({"items": []}))
        client = GoogleBooksClient(session=session)

        client.search_books("python", max_results=100)

        assert session.last_params["maxResults"] == 40


# =============================================================================
# Tests: Response parsing
# =============================================================================


class TestSearchBooksResponseParsing:
    """Tests for parsing Google Books API responses."""

    def test_empty_items_returns_empty_list(self):
        """Empty items array should return empty list."""
        session = FakeSession(FakeResponse({"items": []}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("python")

        assert books == []

    def test_missing_items_returns_empty_list(self):
        """Missing items key should return empty list."""
        session = FakeSession(FakeResponse({}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("python")

        assert books == []

    def test_parses_single_book(self):
        """Should parse a single book correctly."""
        volume = _make_volume(
            volume_id="abc123",
            title="Python Programming",
            authors=["John Doe"],
            description="A great book",
            language="en",
            categories=["Computers"],
        )
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("python")

        assert len(books) == 1
        book = books[0]
        assert book.title == "Python Programming"
        assert book.authors == ["John Doe"]
        assert book.description == "A great book"
        assert book.language == "en"
        assert book.categories == ["Computers"]
        assert book.source == "google_books"
        assert book.source_id == "abc123"

    def test_parses_multiple_books(self):
        """Should parse multiple books."""
        volumes = [
            _make_volume(volume_id="vol1", title="Book One"),
            _make_volume(volume_id="vol2", title="Book Two"),
            _make_volume(volume_id="vol3", title="Book Three"),
        ]
        session = FakeSession(FakeResponse({"items": volumes}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 3
        assert books[0].title == "Book One"
        assert books[1].title == "Book Two"
        assert books[2].title == "Book Three"

    def test_missing_authors_defaults_to_unknown(self):
        """Missing authors should default to ['Unknown']."""
        # Create volume without authors key
        volume = {"id": "vol1", "volumeInfo": {"title": "No Author Book"}}
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].authors == ["Unknown"]

    def test_missing_title_skips_book(self):
        """Book without title should be skipped."""
        volume = {"id": "vol1", "volumeInfo": {}}  # No title
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert books == []

    def test_missing_volume_id_skips_book(self):
        """Book without volume ID should be skipped."""
        volume = {"volumeInfo": {"title": "No ID Book"}}  # No id
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert books == []

    def test_parses_metadata_isbn(self):
        """Should parse ISBN from metadata."""
        volume = _make_volume(
            volume_id="vol1",
            title="Book with ISBN",
            isbn_10="0123456789",
            isbn_13="9780123456789",
        )
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].metadata is not None
        assert books[0].metadata.isbn == "0123456789"
        assert books[0].metadata.isbn13 == "9780123456789"

    def test_metadata_none_when_no_metadata_fields(self):
        """Should return None metadata when no metadata fields present."""
        # Volume with only required fields, no metadata
        volume = {"id": "vol1", "volumeInfo": {"title": "Minimal Book"}}
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].metadata is None

    def test_missing_categories_defaults_to_empty_list(self):
        """Missing categories key should default to empty list."""
        # Volume without categories key
        volume = {"id": "vol1", "volumeInfo": {"title": "No Categories"}}
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].categories == []


# =============================================================================
# Tests: Date parsing
# =============================================================================


class TestPublishedDateParsing:
    """Tests for parsing various date formats from Google Books."""

    def test_full_date_format_yyyy_mm_dd(self):
        """Should parse YYYY-MM-DD format."""
        volume = _make_volume(
            volume_id="vol1",
            title="Book",
            published_date="2024-06-15",
        )
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].published_date == datetime(2024, 6, 15, tzinfo=timezone.utc)

    def test_year_month_format_yyyy_mm(self):
        """Should parse YYYY-MM format."""
        volume = _make_volume(
            volume_id="vol1",
            title="Book",
            published_date="2024-06",
        )
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].published_date == datetime(2024, 6, 1, tzinfo=timezone.utc)

    def test_year_only_format_yyyy(self):
        """Should parse YYYY format."""
        volume = _make_volume(
            volume_id="vol1",
            title="Book",
            published_date="2024",
        )
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].published_date == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_missing_date_returns_none(self):
        """Missing published date should result in None."""
        volume = _make_volume(volume_id="vol1", title="Book")
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].published_date is None

    def test_invalid_date_format_returns_none(self):
        """Invalid date format should result in None."""
        volume = _make_volume(
            volume_id="vol1",
            title="Book",
            published_date="June 2024",  # Invalid format
        )
        session = FakeSession(FakeResponse({"items": [volume]}))
        client = GoogleBooksClient(session=session)

        books = client.search_books("test")

        assert len(books) == 1
        assert books[0].published_date is None


# =============================================================================
# Tests: get_book_by_id
# =============================================================================


class TestGetBookById:
    """Tests for get_book_by_id method."""

    def test_returns_book_when_found(self):
        """Should return book when found."""
        volume = _make_volume(
            volume_id="abc123",
            title="Found Book",
            authors=["Author"],
        )
        session = FakeSession(FakeResponse(volume))
        client = GoogleBooksClient(session=session)

        book = client.get_book_by_id("abc123")

        assert book is not None
        assert book.title == "Found Book"
        assert book.source_id == "abc123"

    def test_returns_none_on_404(self):
        """Should return None when book not found (404)."""
        session = FakeSession(FakeResponse(status_code=404))
        client = GoogleBooksClient(session=session)

        book = client.get_book_by_id("nonexistent")

        assert book is None

    def test_returns_none_for_empty_id(self):
        """Should return None for empty ID."""
        session = FakeSession()
        client = GoogleBooksClient(session=session)

        assert client.get_book_by_id("") is None
        assert client.get_book_by_id("   ") is None

    def test_constructs_correct_url(self):
        """Should construct correct URL with volume ID."""
        volume = _make_volume(volume_id="xyz789", title="Book")
        session = FakeSession(FakeResponse(volume))
        client = GoogleBooksClient(session=session)

        client.get_book_by_id("xyz789")

        assert session.last_url == "https://www.googleapis.com/books/v1/volumes/xyz789"


# =============================================================================
# Tests: Error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_search_wraps_session_exception_in_runtime_error(self):
        """Session exceptions should be wrapped in RuntimeError."""

        class FailingSession:
            def get(self, url, params=None):
                raise ConnectionError("Network error")

        client = GoogleBooksClient(session=FailingSession())

        with pytest.raises(RuntimeError, match="Google Books API request failed"):
            client.search_books("test")

    def test_get_by_id_wraps_session_exception_in_runtime_error(self):
        """Session exceptions in get_by_id should be wrapped in RuntimeError."""

        class FailingSession:
            def get(self, url, params=None):
                raise TimeoutError("Request timeout")

        client = GoogleBooksClient(session=FailingSession())

        with pytest.raises(RuntimeError, match="Google Books API request failed"):
            client.get_book_by_id("vol123")

    def test_http_error_raises_runtime_error(self):
        """HTTP errors should raise RuntimeError."""
        session = FakeSession(FakeResponse(status_code=500))
        client = GoogleBooksClient(session=session)

        with pytest.raises(RuntimeError, match="Google Books API request failed"):
            client.search_books("test")


# =============================================================================
# Tests: get_source_name
# =============================================================================


class TestGetSourceName:
    """Tests for get_source_name method."""

    def test_returns_google_books(self):
        """Should return 'google_books' as source name."""
        client = GoogleBooksClient(session=FakeSession())

        assert client.get_source_name() == "google_books"
