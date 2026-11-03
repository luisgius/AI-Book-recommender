"""
Ingestion Service for the Book Recommendation System.

This service orchestrates the ingestion pipeline:
1. Fetch books from external API (Google Books)
2. Normalize and deduplicate
3. Persist books to catalog (SQLite)
4. Return a summary of the operation

Note: Index building (BM25 + FAISS) is handled separately by the
script layer to maintain separation of concerns.
"""

import logging
from typing import Optional

from app.domain.entities import Book
from app.domain.value_objects import IngestionSummary
from app.domain.ports import (
    ExternalBooksProvider,
    BookCatalogRepository,
)

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates the ingestion pipeline for books.

    This service coordinates the external provider and catalog repository to:
    - Fetch books from external sources
    - Deduplicate (skip books already in catalog)
    - Persist new books to catalog
    - Track and report success/failure statistics
    """

    def __init__(
        self,
        books_provider: ExternalBooksProvider,
        catalog_repo: BookCatalogRepository,
    ) -> None:
        """
        Initialize the ingestion service with required dependencies.

        Args:
            books_provider: External API client (e.g., Google Books)
            catalog_repo: Repository for persisting books to DB
        """
        self._books_provider = books_provider
        self._catalog_repo = catalog_repo

    def ingest_books(
        self,
        query: str,
        max_results: int = 100,
        language: Optional[str] = None,
    ) -> IngestionSummary:
        """
        Main ingestion method: fetch, deduplicate, and persist books.

        Workflow:
        1. Fetch books from external API
        2. Check for duplicates (via source_id)
        3. Save new books to catalog
        4. Track success/failure statistics

        Args:
            query: Search query for external API
            max_results: Maximum number of books to fetch
            language: Optional language filter (ISO 639-1 code)

        Returns:
            IngestionSummary with statistics about the operation

        Raises:
            RuntimeError: If critical errors occur (e.g., API unreachable)
        """
        logger.info(f"Starting ingestion: query='{query}', max_results={max_results}")

        # Step 1: Fetch books from external API
        try:
            logger.info("Fetching books from external API...")
            books = self._books_provider.search_books(query, max_results, language)
            logger.info(f"Fetched {len(books)} books from API")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch books from external API: {e}") from e

        if not books:
            logger.warning("No books fetched from API")
            return IngestionSummary(
                n_fetched=0,
                n_inserted=0,
                n_skipped=0,
                n_errors=0,
                query=query,
                language=language,
            )

        # Step 2: Deduplicate and persist books
        n_inserted = 0
        n_skipped = 0
        n_errors = 0
        error_messages = []

        for book in books:
            try:
                if book.source == "google_books" and book.source_id is None:
                    n_errors += 1
                    error_msg = (
                        f"Invalid book from google_books without source_id: {book.title}"
                    )
                    logger.warning(error_msg)
                    error_messages.append(error_msg)
                    continue

                # Check if book already exists (by source + source_id)
                if book.source_id is not None:
                    existing = self._catalog_repo.get_by_source_id(book.source, book.source_id)
                    if existing:
                        logger.debug(
                            f"Skipping duplicate: {book.title} (source_id={book.source_id})"
                        )
                        n_skipped += 1
                        continue

                # Insert new book
                self._catalog_repo.save(book)
                n_inserted += 1
                logger.debug(f"Inserted: {book.title}")

            except Exception as e:
                n_errors += 1
                error_msg = f"Failed to insert {book.title}: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)

        logger.info(
            f"Ingestion complete: {n_inserted} inserted, {n_skipped} skipped, {n_errors} errors"
        )

        return IngestionSummary(
            n_fetched=len(books),
            n_inserted=n_inserted,
            n_skipped=n_skipped,
            n_errors=n_errors,
            query=query,
            language=language,
            errors=error_messages,
        )


