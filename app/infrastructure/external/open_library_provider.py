"""
Open Library provider implementation (second ExternalBooksProvider).

Uses the public Open Library APIs:
- search:  https://openlibrary.org/search.json  (works matching a query)
- works:   https://openlibrary.org/works/{id}.json  (description, subjects)
- authors: https://openlibrary.org/authors/{id}.json  (names, for get_book_by_id)

Open Library is complementary to Google Books: it is strong on older and
Spanish-language works (e.g. Spanish Civil War novels missing from the
current catalog) and needs no API key.

Data licensing: Open Library publishes no Creative Commons licence for its
catalog data (see review/06-research-improvements-2026-11.md, section 3.8).
This adapter is meant for small academic ingestions with attribution
(source="open_library" on every book). For a large catalog, use the bulk
data dumps or a source with an explicit licence such as the BNE catalogue.

Open Library asks API clients to send an identifying User-Agent with a
contact address and to keep request rates low.
"""

import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

from app.domain.entities import Book
from app.domain.ports import ExternalBooksProvider
from app.domain.value_objects import BookMetadata

logger = logging.getLogger(__name__)

# Open Library uses MARC 21 language codes; the domain uses ISO 639-1.
ISO_TO_MARC = {"es": "spa", "en": "eng", "ca": "cat", "fr": "fre", "de": "ger", "it": "ita", "pt": "por"}
MARC_TO_ISO = {marc: iso for iso, marc in ISO_TO_MARC.items()}

SEARCH_FIELDS = ",".join([
    "key", "title", "author_name", "first_publish_year", "language", "subject",
    "isbn", "publisher", "number_of_pages_median", "cover_i", "first_sentence",
])
MAX_SUBJECTS = 5
PAGE_SIZE = 100


class OpenLibraryProvider(ExternalBooksProvider):
    """ExternalBooksProvider backed by the Open Library search and works APIs."""

    BASE_URL = "https://openlibrary.org"
    COVERS_URL = "https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"

    def __init__(
        self,
        contact_email: str = "lgimenezuso@gmail.com",
        fetch_descriptions: bool = True,
        request_delay_s: float = 0.25,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Args:
            contact_email: Contact sent in the User-Agent, as Open Library asks
            fetch_descriptions: Fetch each work's description (one extra request
                per book). Without it books only get the first sentence, if any.
            request_delay_s: Pause between description requests (politeness)
            session: Optional requests session (tests inject a fake one)
        """
        self._fetch_descriptions = fetch_descriptions
        self._delay = request_delay_s
        self._session = session or requests.Session()
        self._session.headers.update({
            "User-Agent": f"AI-Book-recommender-TFG/1.0 ({contact_email})",
            "Accept": "application/json",
        })

    def get_source_name(self) -> str:
        """Get the source identifier."""
        return "open_library"

    # ------------------------------------------------------------------ HTTP

    def _get_json(self, path: str, params: Optional[dict] = None, timeout: int = 15) -> Dict[str, Any]:
        """GET a JSON document with retries on transient errors (429, 5xx, network)."""
        max_retries, base_backoff_s = 3, 0.5
        retryable_statuses = {429, 500, 502, 503, 504}
        for attempt in range(max_retries + 1):
            try:
                response = self._session.get(self.BASE_URL + path, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 404:
                    return {}
                if status_code not in retryable_statuses or attempt == max_retries:
                    raise RuntimeError(f"Open Library request failed: {e}") from e
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Open Library request failed: {e}") from e
            sleep_s = base_backoff_s * (2 ** attempt) + random.uniform(0, 0.2)
            logger.warning("Open Library transient error; retrying in %.2fs (attempt %s)", sleep_s, attempt + 1)
            time.sleep(sleep_s)
        return {}

    # ------------------------------------------------------------- port API

    def search_books(self, query: str, max_results: int = 10, language: Optional[str] = None) -> List[Book]:
        """
        Search Open Library works.

        Args:
            query: Search query string
            max_results: Maximum number of books to return
            language: Optional ISO 639-1 code; only works available in that
                language are returned and the book gets that language

        Returns:
            List of Book entities (works without title or authors are skipped)

        Raises:
            ValueError: If query is empty or the language is not supported
            RuntimeError: If the API request fails
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {max_results}")
        marc = None
        if language:
            marc = ISO_TO_MARC.get(language.lower())
            if marc is None:
                raise ValueError(f"Unsupported language '{language}' (supported: {sorted(ISO_TO_MARC)})")

        q = f"{query.strip()} language:{marc}" if marc else query.strip()
        books: List[Book] = []
        page = 1
        while len(books) < max_results:
            data = self._get_json("/search.json", params={
                "q": q, "fields": SEARCH_FIELDS, "limit": min(PAGE_SIZE, max_results), "page": page,
            })
            docs = data.get("docs", [])
            if not docs:
                break
            for doc in docs:
                book = self._doc_to_book(doc, language=language.lower() if language else None)
                if book is not None:
                    books.append(book)
                if len(books) >= max_results:
                    break
            if len(docs) < min(PAGE_SIZE, max_results):
                break
            page += 1
        return books

    def get_book_by_id(self, external_id: str) -> Optional[Book]:
        """
        Fetch one work by its Open Library id (e.g. "OL6863683W").

        Returns:
            The Book, or None if the work does not exist or lacks title/authors
        """
        work = self._get_json(f"/works/{external_id}.json")
        if not work or not work.get("title"):
            return None
        authors = []
        for entry in work.get("authors", [])[:3]:
            author_key = (entry.get("author") or {}).get("key")
            if author_key:
                name = self._get_json(f"{author_key}.json").get("name")
                if name:
                    authors.append(name)
        if not authors:
            return None
        return Book(
            id=uuid4(),
            title=work["title"].strip(),
            authors=authors,
            description=self._description_of(work),
            categories=list(work.get("subjects", []))[:MAX_SUBJECTS],
            source=self.get_source_name(),
            source_id=external_id,
        )

    # -------------------------------------------------------------- mapping

    @staticmethod
    def _description_of(work: Dict[str, Any]) -> Optional[str]:
        """A work description is either a string or {"type": ..., "value": ...}."""
        description = work.get("description")
        if isinstance(description, dict):
            description = description.get("value")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None

    def _doc_to_book(self, doc: Dict[str, Any], language: Optional[str]) -> Optional[Book]:
        """Map one search result to a Book (None if title or authors are missing)."""
        title = (doc.get("title") or "").strip()
        authors = [a.strip() for a in doc.get("author_name", []) if a and a.strip()]
        key = doc.get("key", "")
        if not title or not authors or not key.startswith("/works/"):
            return None
        work_id = key.rsplit("/", 1)[-1]

        if language is None:
            mapped = [MARC_TO_ISO[code] for code in doc.get("language", []) if code in MARC_TO_ISO]
            language = mapped[0] if len(set(mapped)) == 1 else None  # ambiguous when several

        description = None
        if self._fetch_descriptions:
            description = self._description_of(self._get_json(f"/works/{work_id}.json"))
            time.sleep(self._delay)
        if description is None:
            first_sentence = doc.get("first_sentence")
            if isinstance(first_sentence, list) and first_sentence:
                description = first_sentence[0]

        year = doc.get("first_publish_year")
        isbns = doc.get("isbn", [])
        isbn13 = next((i for i in isbns if len(i) == 13), None)
        isbn10 = next((i for i in isbns if len(i) == 10), None)
        cover_id = doc.get("cover_i")
        metadata = BookMetadata(
            isbn=isbn10,
            isbn13=isbn13,
            publisher=(doc.get("publisher") or [None])[0],
            page_count=doc.get("number_of_pages_median"),
            thumbnail_url=self.COVERS_URL.format(cover_id=cover_id) if cover_id else None,
            preview_link=f"{self.BASE_URL}{key}",
        )
        return Book(
            id=uuid4(),
            title=title,
            authors=authors,
            description=description,
            language=language,
            categories=list(doc.get("subject", []))[:MAX_SUBJECTS],
            published_date=datetime(year, 1, 1) if isinstance(year, int) and 0 < year < 3000 else None,
            source=self.get_source_name(),
            source_id=work_id,
            metadata=metadata,
        )
