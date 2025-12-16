"""
Domain service for catalog ingestion and index rebuilding.

=============================================================================
TEACHING NOTES: Domain Services in Hexagonal Architecture
=============================================================================

A domain service orchestrates complex operations that:
1. Span multiple domain concepts (books, indices, external sources)
2. Don't naturally belong to a single entity
3. Need to coordinate between multiple ports

Key principle: This service depends ONLY on PORTS (interfaces), not on
concrete implementations. It doesn't know about SQLite, FAISS, or HTTP.
This makes it:
- Testable (inject fake implementations)
- Flexible (swap implementations without changing business logic)
- Clean (no infrastructure concerns in domain code)

=============================================================================
TEACHING NOTES: SQLite as Source of Truth
=============================================================================

The ingestion flow follows this pattern:

    External API --> SQLite (persist) --> Rebuild indices from SQLite

Why rebuild from SQLite (not just the new books)?
1. Indices must reflect the COMPLETE catalog state
2. New books may update existing entries (upsert)
3. SQLite is authoritative - indices are derived artifacts

This ensures consistency: indices always match SQLite exactly.

=============================================================================
"""

from typing import List, Optional

from app.domain.entities import Book
from app.domain.ports import (
    BookCatalogRepository,
    LexicalSearchRepository,
    EmbeddingsStore,
    ExternalBooksProvider,
)


class CatalogIngestionService:
    """
    Orchestrates book ingestion from external sources and index rebuilding.

    This service implements the complete ingestion pipeline:
    1. Fetch books from an external provider (e.g., Google Books)
    2. Persist them to the catalog repository (source of truth)
    3. Rebuild search indices from the full catalog

    The service depends only on domain ports, making it independent of
    infrastructure details like HTTP clients, databases, or index formats.

    Usage:
        service = CatalogIngestionService(
            catalog_repo=sqlite_repo,
            lexical_repo=bm25_repo,
            embeddings_store=faiss_store,
            external_provider=google_books_client,
        )
        new_books = service.ingest_and_reindex("machine learning", max_results=50)
    """

    def __init__(
        self,
        catalog_repo: BookCatalogRepository,
        lexical_repo: LexicalSearchRepository,
        embeddings_store: EmbeddingsStore,
        external_provider: ExternalBooksProvider,
    ) -> None:
        """
        Initialize the ingestion service with required dependencies.

        All dependencies are ports (interfaces), not concrete implementations.
        This follows the Dependency Inversion Principle: high-level modules
        (this service) depend on abstractions (ports), not details (adapters).

        Args:
            catalog_repo: Repository for persisting books (source of truth)
            lexical_repo: Repository for BM25 lexical search index
            embeddings_store: Store for vector embeddings and FAISS index
            external_provider: Provider for fetching books from external APIs
        """
        self._catalog_repo = catalog_repo
        self._lexical_repo = lexical_repo
        self._embeddings_store = embeddings_store
        self._external_provider = external_provider

    def ingest_and_reindex(
        self,
        query: str,
        max_results: int = 20,
        language: Optional[str] = None,
        *,
        persist_lexical_index_path: Optional[str] = None,
        persist_vector_index_dir: Optional[str] = None,
    ) -> List[Book]:
        """
        Fetch books from external source, persist to catalog, and rebuild indices.

        This is the main entry point for the ingestion pipeline. It performs:

        1. FETCH: Query external provider for books matching the search
        2. PERSIST: Save fetched books to catalog (upsert on source+source_id)
        3. LOAD: Retrieve FULL catalog from SQLite (source of truth)
        4. REBUILD LEXICAL: Build BM25 index from complete catalog
        5. REBUILD VECTOR: Generate embeddings and build FAISS index
        6. PERSIST INDICES: Optionally save indices to disk

        Why load full catalog before rebuilding?
        - Indices must reflect ALL books, not just new ones
        - Upserts may have updated existing books
        - SQLite is authoritative; indices are derived

        Args:
            query: Search query for external provider (e.g., "python programming")
            max_results: Maximum books to fetch from external provider
            language: Optional language filter (ISO 639-1 code, e.g., "en", "es")
            persist_lexical_index_path: If provided, save BM25 index to this file
            persist_vector_index_dir: If provided, save FAISS index to this directory

        Returns:
            List of newly fetched books (from external provider).
            Note: Returns fetched books, not the full catalog.

        Raises:
            ValueError: If query is empty or blank
            RuntimeError: If any step in the pipeline fails
        """
        # =====================================================================
        # Step 0: Validate input
        # =====================================================================
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        # =====================================================================
        # Step 1: FETCH books from external provider
        # =====================================================================
        # The external provider (e.g., Google Books) returns Book entities
        # with source="google_books" and source_id=<Google's volume ID>
        try:
            fetched_books = self._external_provider.search_books(
                query=query,
                max_results=max_results,
                language=language,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch books from external provider: {e}") from e

        # =====================================================================
        # Step 2: PERSIST fetched books to catalog (SQLite)
        # =====================================================================
        # save_many() performs upsert: if (source, source_id) exists, it updates
        # the row but preserves the existing UUID. This is critical for index
        # consistency - indices reference books by UUID.
        if fetched_books:
            try:
                self._catalog_repo.save_many(fetched_books)
            except Exception as e:
                raise RuntimeError(f"Failed to persist books to catalog: {e}") from e

        # =====================================================================
        # Step 3: LOAD full catalog from SQLite (source of truth)
        # =====================================================================
        # We rebuild indices from the COMPLETE catalog, not just new books.
        # This ensures indices are always consistent with SQLite.
        try:
            all_books = self._catalog_repo.get_all()
        except Exception as e:
            raise RuntimeError(f"Failed to load catalog for reindexing: {e}") from e

        # =====================================================================
        # Step 4: REBUILD lexical (BM25) index
        # =====================================================================
        # BM25 index is built from book searchable text (title + authors +
        # description + categories). The index is derived from SQLite data.
        try:
            self._lexical_repo.build_index(all_books)

            if persist_lexical_index_path:
                self._lexical_repo.save_index(persist_lexical_index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to rebuild lexical index: {e}") from e

        # =====================================================================
        # Step 5: REBUILD vector (FAISS) index
        # =====================================================================
        # Vector index requires:
        # a) Generate embeddings for each book's searchable text
        # b) Store embeddings associated with book UUIDs
        # c) Build the FAISS index for approximate nearest neighbor search
        if all_books:
            try:
                # Prepare texts and book IDs for batch processing
                texts = [book.get_searchable_text() for book in all_books]
                book_ids = [book.id for book in all_books]

                # Generate embeddings in batch (more efficient than one-by-one)
                embeddings = self._embeddings_store.generate_embeddings_batch(texts)

                # Store embeddings with their associated book UUIDs
                self._embeddings_store.store_embeddings_batch(book_ids, embeddings)

                # Build the FAISS index from stored embeddings
                self._embeddings_store.build_index()

                if persist_vector_index_dir:
                    self._embeddings_store.save_index(persist_vector_index_dir)

            except Exception as e:
                raise RuntimeError(f"Failed to rebuild vector index: {e}") from e

        # =====================================================================
        # Return the newly fetched books (not the full catalog)
        # =====================================================================
        # This allows callers to know what was imported in this run
        return fetched_books
