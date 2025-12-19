"""
Google Books API provider implementation.

This module implements the ExternalBooksProvider protocol for fetching
books from the Google Books API.
"""

from datetime import datetime
import logging
import random
import time
from typing import Optional, List

import requests

from app.domain.entities import Book
from app.domain.ports import ExternalBooksProvider
from app.domain.value_objects import BookMetadata


logger = logging.getLogger(__name__)

class GoogleBooksProvider(ExternalBooksProvider):
    
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google Books provider.
        
        Args:
            api_key: Optional API key for higher rate limits.
                    For MVP, this can be None (uses public access).
        """
        self._api_key = api_key
        self._base_url = self.BASE_URL
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "BookRecommendationSystem/1.0",
            "Accept": "application/json",
        })

    def get_source_name(self) -> str:
        """Get the source identifier."""
        return "google_books"

    def _get_with_retries(self, url: str, *, params: dict, timeout: int = 10) -> requests.Response:
        max_retries = 3
        base_backoff_s = 0.5
        retryable_statuses = {429, 500, 502, 503, 504}

        attempt = 0
        while True:
            try:
                response = self._session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code in retryable_statuses and attempt < max_retries:
                    sleep_s = base_backoff_s * (2**attempt) + random.uniform(0, 0.2)
                    logger.warning(
                        "Google Books API transient HTTP %s; retrying in %.2fs (attempt %s/%s)",
                        status_code,
                        sleep_s,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(sleep_s)
                    attempt += 1
                    continue
                raise RuntimeError(f"Google Books API request failed: {e}") from e
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    sleep_s = base_backoff_s * (2**attempt) + random.uniform(0, 0.2)
                    logger.warning(
                        "Google Books API request failed (%s); retrying in %.2fs (attempt %s/%s)",
                        type(e).__name__,
                        sleep_s,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(sleep_s)
                    attempt += 1
                    continue
                raise RuntimeError(f"Google Books API request failed: {e}") from e

    def search_books(self, query: str, max_results: int = 10, language: Optional[str] = None) -> List[Book]:
        """
        Search for books using the Google Books API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            language: Optional ISO 639-1 language code to filter results
            
        Returns:
            List of Book entities matching the query
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If API request fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        params= {
            "q" :query,
        }

        # aggregate API key
        if self._api_key:
            params["key"] = self._api_key

        # language filter
        if language:
            params["langRestrict"] = language

        # Pagination
        MAX_RESULTS_PER_PAGE = 40

        books = []
        fetched = 0

        while fetched < max_results:

            remaining = max_results - fetched
            page_size = min(MAX_RESULTS_PER_PAGE, remaining)

            params["maxResults"] = page_size
            params["startIndex"] = fetched

            # HTTP Call
            try:
                response = self._get_with_retries(self._base_url, params=params, timeout=10)
            except RuntimeError:
                if books:
                    logger.warning(
                        "Stopping pagination early due to repeated API failure; returning %s partial results",
                        len(books),
                    )
                    break
                raise

            # JSON Parser
            try:
                data = response.json()
            except ValueError as e:
                raise RuntimeError(f"Invalid JSON response from Google Books API: {e}") from e

            # items to book entities
            items = data.get("items",[])

            if not items:
                # No results, so break loop
                break

            for item in items:
                book = self._item_to_book(item)
                if book is not None:
                    books.append(book)

            fetched += len(items)

        return books

    def get_book_by_id(self, external_id: str) -> Optional[Book]:
        """
        Fetch a single book by its Google Books ID.
        
        Args:
            external_id: The Google Books volume ID (e.g., 'abc123')
            
        Returns:
            Book entity if found, None otherwise
            
        Raises:
            RuntimeError: If API request fails
        """
        if not external_id or not external_id.strip():
            return None
        
        url = f"{self._base_url}/{external_id}"
        params = {}
        
        if self._api_key:
            params["key"] = self._api_key
        
        try:
            response = self._session.get(url, params=params, timeout=10)
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Google Books API request failed: {e}") from e
        
        try:
            item = response.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON response from Google Books API: {e}") from e
        
        return self._item_to_book(item)



    def _item_to_book(self, item: dict) -> Optional[Book]:
        """
        Convert a Google Books API item to a Book entity.
        
        Args:
            item: A single item from the API response
            
        Returns:
            Book entity, or None if the item is invalid/incomplete
        """
        # Google Books items have this structure:
        # {
        #   "id": "abc123",
        #   "volumeInfo": { ... all the book data ... }
        # }
        volume_info = item.get("volumeInfo", {})
    
        # MANDATORY FIELDS
        title = volume_info.get("title")
        if not title:
            return None

        authors = volume_info.get("authors", [])
        if not authors:
            return None

        # OPTIONAL FIELDS
        description = volume_info.get("description")
        language = volume_info.get("language")
        categories = volume_info.get("categories", [])
        
        published_date = None
        published_date_str = volume_info.get("publishedDate")
        if published_date_str:
            published_date = self._parse_published_date(published_date_str)

        # METADATA
        metadata = None
        try:
            # ISBN
            isbn = None
            isbn13 = None
            industry_identifiers = volume_info.get("industryIdentifiers", [])
            for identifier in industry_identifiers:
                id_type = identifier.get("type", "")
                if id_type == "ISBN_13":
                    isbn13 = identifier.get("identifier")
                elif id_type == "ISBN_10":
                    isbn = identifier.get("identifier")

            # Other fields
            publisher = volume_info.get("publisher")
            page_count = volume_info.get("pageCount")
            average_rating = volume_info.get("averageRating")
            ratings_count = volume_info.get("ratingsCount")
            
            # Thumbnail
            image_links = volume_info.get("imageLinks", {})
            thumbnail_url = image_links.get("thumbnail")
            
            # Preview link
            preview_link = volume_info.get("previewLink")

            # Only create metadata if at least one field
            if any([isbn, isbn13, publisher, page_count, average_rating, 
                    ratings_count, thumbnail_url, preview_link]):
                metadata = BookMetadata(
                    isbn=isbn,
                    isbn13=isbn13,
                    publisher=publisher,
                    page_count=page_count,
                    average_rating=average_rating,
                    ratings_count=ratings_count,
                    thumbnail_url=thumbnail_url,
                    preview_link=preview_link,
                )
        except Exception:
            # If metadata creation fails, continue without it
            pass


        # BOOK CREATION
        try:
            book = Book.create_new(
            title=title,
            authors=authors,
            source=self.get_source_name(),
            source_id=item.get("id"),
            description=description,
            language=language,
            categories=categories,
            published_date=published_date,
            metadata=metadata,
            )
            return book
        except Exception:
            # If Book validation fails, return None
            return None

    def _parse_published_date(self, date_str: str) -> Optional[datetime]:  # noqa: C901
        """
        Parse Google Books publishedDate field.
        
        Google Books returns dates in various formats:
        - "2023" (year only)
        - "2023-05" (year-month)
        - "2023-05-15" (full date)
        
        Args:
            date_str: Date string from API
            
        Returns:
            datetime object, or None if parsing fails
        """
        if not date_str:
            return None
        
        # Try full ISO format first: "2023-05-15"
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            pass
        
        # Try year-month: "2023-05"
        try:
            return datetime.strptime(date_str, "%Y-%m")
        except ValueError:
            pass
        
        # Try year only: "2023"
        try:
            return datetime.strptime(date_str, "%Y")
        except ValueError:
            pass
        
        return None
           