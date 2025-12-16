"""
Google Books API client implementing ExternalBooksProvider port.

=============================================================================
TEACHING NOTES: Infrastructure Adapter Pattern
=============================================================================

This class is an ADAPTER in Hexagonal Architecture. It:
1. Implements a domain PORT (ExternalBooksProvider)
2. Handles infrastructure concerns (HTTP, JSON parsing)
3. Translates external data formats into domain entities (Book)

The domain layer calls methods like search_books() without knowing that:
- We're using HTTP to talk to Google
- Responses are JSON
- Field names differ from our domain model

This separation allows:
- Testing domain logic without network calls
- Swapping to a different book API (Open Library) with a new adapter
- Mocking this adapter for fast tests

=============================================================================
TEACHING NOTES: Dependency Injection for Testability
=============================================================================

The constructor accepts an optional `session` parameter:
- In production: uses requests.Session() by default
- In tests: inject a fake session that returns canned responses

This pattern avoids network calls in tests while keeping the real
implementation clean and simple.

=============================================================================
"""

import re
from datetime import datetime, timezone
from typing import List, Optional, Any
from app.domain.utils.uuid7 import uuid7

import requests

from app.domain.entities import Book
from app.domain.value_objects import BookMetadata
from app.domain.ports import ExternalBooksProvider


class GoogleBooksClient(ExternalBooksProvider):
    """
    Google Books API client for fetching book data.

    This adapter implements the ExternalBooksProvider port, translating
    Google Books API responses into domain Book entities.

    Features:
    - Search for books by query, with optional language filter
    - Fetch individual books by Google volume ID
    - Graceful handling of missing/partial data from API
    - Dependency-injected HTTP session for testability

    Usage:
        # Production
        client = GoogleBooksClient(api_key="your-api-key")
        books = client.search_books("python programming", max_results=10)

        # Testing (with fake session)
        client = GoogleBooksClient(session=fake_session)
        books = client.search_books("test query")
    """

    # Google Books API base URL
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Google Books client.

        Args:
            api_key: Optional Google API key for higher rate limits.
                    Without a key, requests are limited but still work.
            session: Optional HTTP session for dependency injection.
                    If None, creates a new requests.Session().
                    Pass a fake session in tests to avoid network calls.
        """
        self._api_key = api_key
        # Dependency injection: use provided session or create default
        self._session = session if session is not None else requests.Session()

    def search_books(
        self,
        query: str,
        max_results: int = 10,
        language: Optional[str] = None,
    ) -> List[Book]:
        """
        Search for books in Google Books API.

        Args:
            query: Search query (e.g., "machine learning python")
            max_results: Maximum number of books to return (1-40)
            language: Optional language filter (ISO 639-1 code)

        Returns:
            List of Book entities parsed from API response

        Raises:
            ValueError: If query is empty or blank
            RuntimeError: If API request fails
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        # Build request parameters
        params = {
            "q": query.strip(),
            "maxResults" : min(max_results, 40), # API limit is 40
        }

        if language:
            params["langRestrict"] = language

        if self._api_key:
            params["key"] = self._api_key

        # Make API request
        try:
            response = self._session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Google Books API request failed: {e}") from e

        # Parse response items into Book entities
        items = data.get("items", [])
        books = []

        for item in items:
            book = self._parse_volume_to_book(item)
            if book is not None:
                books.append(book)
        
        return books

    def get_book_by_id(self, external_id: str) -> Optional[Book]:
        """
        Fetch a specific book by its Google volume ID.

        Args:
            external_id: Google Books volume ID

        Returns:
            Book entity if found, None otherwise

        Raises:
            RuntimeError: If API request fails (other than 404)
        """
        if not external_id or not external_id.strip():
            return None
        
        url = f"{self.BASE_URL}/{external_id.strip()}"
        params = {}

        if self._api_key:
            params["key"]= self._api_key
        
        try:
            response = self._session.get(url, params=params)

            # 404 means book not found - return None
            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

        except Exception as e:
            raise RuntimeError(f"Google Books API request failed: {e}") from e

        return self._parse_volume_to_book(data)


    def get_source_name(self) -> str:
        """
        Get the source identifier for this provider.

        Returns:
            "google_books" - used as the `source` field in Book entities
        """
        return "google_books"

    # =========================================================================
    # Private helper methods
    # =========================================================================


    def _parse_volume_to_book(self, volume: dict) -> Optional[Book]:
        """
        Parse a Google Books volume JSON object into a Book entity.

        Handles missing fields gracefully - the API response structure
        is not always complete. Required field (title) missing will cause
        the book to be skipped. Missing authors default to ["Unknown"].

        Args:
            volume: Raw JSON object from Google Books API

        Returns:
            Book entity if parsing succeeds, None if required data is missing
        """
        try:

            volume_id = volume.get("id")
            if not volume_id:
                return None
            
            volume_info = volume.get("volumeInfo", {})

            # Required: title
            title = volume_info.get("title")
            if not title:
                return None

            # Required: authors (default to ["Unknown"] if missing)
            authors = volume_info.get("authors")
            if not authors:
                authors = ["Unknown"]

            # Optional fields - use None or empty list if missing
            description = volume_info.get("description")
            language = volume_info.get("language")
            categories = volume_info.get("categories", [])

            # Parse published date (may be YYYY, YYYY-MM, or YYYY-MM-DD)
            published_date_str = volume_info.get("publishedDate")
            published_date = self._parse_published_date(published_date_str)

            # Build BookMetadata from available fields
            metadata = self._extract_metadata(volume_info)

            # Create Book entity with a new UUID
            # Note: If this book already exists in SQLite (same source+source_id),
            # the repository's upsert will preserve the existing UUID
            return Book(
                id=uuid7(),
                title=title,
                authors=authors,
                description=description,
                language=language,
                categories=categories,
                published_date=published_date,
                source=self.get_source_name(),
                source_id=volume_id,
                metadata=metadata,
            )
        except Exception:
            # If parsing fails for any reason, skip this book
            # In production, you might want to log this
            return None

    def _parse_published_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse Google Books published date string.

        Google Books returns dates in various formats:
        - "2024" (year only)
        - "2024-06" (year and month)
        - "2024-06-15" (full date)

        Args:
            date_str: Date string from API, or None

        Returns:
            Timezone-aware datetime in UTC, or None if parsing fails
        """

        if not date_str:
            return None

        # Try full date format: YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # Try year-month format: YYYY-MM
        if re.match(r"^\d{4}-\d{2}$", date_str):
            try:
                return datetime.strptime(date_str, "%Y-%m").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # Try year-only format: YYYY
        if re.match(r"^\d{4}$", date_str):
            try:
                return datetime.strptime(date_str, "%Y").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        return None

    
    def _extract_metadata(self, volume_info: dict) -> Optional[BookMetadata]:
        """
        Extract BookMetadata from Google Books volumeInfo.

        Maps Google Books fields to our BookMetadata value object:
        - industryIdentifiers -> isbn, isbn13
        - publisher -> publisher
        - pageCount -> page_count
        - averageRating -> average_rating
        - ratingsCount -> ratings_count
        - imageLinks.thumbnail -> thumbnail_url
        - previewLink -> preview_link

        Args:
            volume_info: The volumeInfo object from API response

        Returns:
            BookMetadata if any fields are present, None otherwise
        """
        # Extract ISBN identifiers
        isbn = None
        isbn13 = None

        identifiers = volume_info.get("industryIdentifiers", [])
        for identifier in identifiers:
            id_type = identifier.get("type", "")
            id_value = identifier.get("identifier", "")

            if id_type == "ISBN_10":
                isbn = id_value
            elif id_type == "ISBN_13":
                isbn13 = id_value

        # Extract other metadata fields
        publisher = volume_info.get("publisher")
        page_count = volume_info.get("pageCount")
        average_rating = volume_info.get("averageRating")
        ratings_count = volume_info.get("ratingsCount")

        # Extract image URL
        image_links = volume_info.get("imageLinks", {})
        thumbnail_url = image_links.get("thumbnail")

        preview_link = volume_info.get("previewLink")

        # Only create metadata if at least one field is present
        has_any_field = any([
            isbn, isbn13, publisher, page_count,
            average_rating, ratings_count, thumbnail_url, preview_link
        ])

        if not has_any_field:
            return None

        return BookMetadata(
            isbn=isbn,
            isbn13=isbn13,
            publisher=publisher,
            page_count=page_count,
            average_rating=average_rating,
            ratings_count=ratings_count,
            thumbnail_url=thumbnail_url,
            preview_link=preview_link,
        )

