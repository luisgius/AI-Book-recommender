"""
Tests for CatalogIngestionService.

End-to-end pipeline tests using:
- SQLite in-memory (real repository)
- Fake external provider
- Fake/spy lexical and embeddings stores

Verifies the complete ingestion flow: fetch -> persist -> rebuild indices.
"""

import pytest
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from app.domain.entities import Book
from app.domain.utils.uuid7 import uuid7
from app.domain.services import CatalogIngestionService


# =============================================================================
# Fake implementations for testing
# =============================================================================


class FakeExternalBooksProvider:
    """Fake external provider that returns predefined books."""

    def __init__(self, books_to_return: Optional[List[Book]] = None):
        self._books = books_to_return or []
        self.search_calls: List[dict] = []

    def search_books(
        self,
        query: str,
        max_results: int = 10,
        language: Optional[str] = None,
    ) -> List[Book]:
        self.search_calls.append({
            "query": query,
            "max_results": max_results,
            "language": language,
        })
        return self._books

    def get_book_by_id(self, external_id: str) -> Optional[Book]:
        for book in self._books:
            if book.source_id == external_id:
                return book
        return None

    def get_source_name(self) -> str:
        return "fake_provider"


class FakeBookCatalogRepository:
    """Fake catalog repository with spy capabilities."""

    def __init__(self, initial_books: Optional[List[Book]] = None):
        self._books: dict[UUID, Book] = {}
        if initial_books:
            for book in initial_books:
                self._books[book.id] = book

        # Spy tracking
        self.save_calls: List[Book] = []
        self.save_many_calls: List[List[Book]] = []
        self.get_all_calls: int = 0

    def save(self, book: Book) -> None:
        self.save_calls.append(book)
        self._books[book.id] = book

    def save_many(self, books: List[Book]) -> None:
        self.save_many_calls.append(books)
        for book in books:
            self._books[book.id] = book

    def get_by_id(self, book_id: UUID) -> Optional[Book]:
        return self._books.get(book_id)

    def get_by_source_id(self, source: str, source_id: str) -> Optional[Book]:
        for book in self._books.values():
            if book.source == source and book.source_id == source_id:
                return book
        return None

    def get_all(self, limit: Optional[int] = None) -> List[Book]:
        self.get_all_calls += 1
        books = list(self._books.values())
        if limit:
            return books[:limit]
        return books

    def count(self) -> int:
        return len(self._books)

    def delete(self, book_id: UUID) -> bool:
        if book_id in self._books:
            del self._books[book_id]
            return True
        return False


class FakeLexicalSearchRepository:
    """Fake lexical search repository with spy capabilities."""

    def __init__(self):
        self._indexed_books: List[Book] = []
        self.build_index_calls: List[List[Book]] = []
        self.save_index_calls: List[str] = []

    def build_index(self, books: List[Book]) -> None:
        self.build_index_calls.append(books)
        self._indexed_books = books

    def add_to_index(self, book: Book) -> None:
        self._indexed_books.append(book)

    def search(self, query_text: str, max_results: int = 10, filters=None) -> List:
        return []

    def save_index(self, path: str) -> None:
        self.save_index_calls.append(path)

    def load_index(self, path: str) -> None:
        pass


class FakeEmbeddingsStore:
    """Fake embeddings store with spy capabilities."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._embeddings: dict[str, List[float]] = {}

        # Spy tracking
        self.generate_batch_calls: List[List[str]] = []
        self.store_batch_calls: List[tuple] = []
        self.build_index_calls: int = 0
        self.save_index_calls: List[str] = []

    def generate_embedding(self, text: str) -> List[float]:
        return [0.1] * self._dimension

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        self.generate_batch_calls.append(texts)
        return [[0.1] * self._dimension for _ in texts]

    def store_embedding(self, book_id: UUID, embedding: List[float]) -> None:
        self._embeddings[str(book_id)] = embedding

    def store_embeddings_batch(
        self, book_ids: List[UUID], embeddings: List[List[float]]
    ) -> None:
        self.store_batch_calls.append((book_ids, embeddings))
        for book_id, emb in zip(book_ids, embeddings):
            self._embeddings[str(book_id)] = emb

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self._embeddings.get(str(book_id))

    def build_index(self) -> None:
        self.build_index_calls += 1

    def save_index(self, path: str) -> None:
        self.save_index_calls.append(path)

    def load_index(self, path: str) -> None:
        pass

    def get_dimension(self) -> int:
        return self._dimension


# =============================================================================
# Test fixtures
# =============================================================================


def _make_book(
    title: str,
    source_id: str,
    authors: Optional[List[str]] = None,
    source: str = "fake_provider",
) -> Book:
    """Helper to create test Book entities."""
    return Book(
        id=uuid7(),
        title=title,
        authors=authors or ["Test Author"],
        description=f"Description for {title}",
        language="en",
        categories=["Test"],
        published_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source=source,
        source_id=source_id,
    )


# =============================================================================
# Tests: Input validation
# =============================================================================


class TestIngestAndReindexValidation:
    """Tests for input validation."""

    def test_empty_query_raises_value_error(self):
        """Empty query should raise ValueError."""
        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(),
        )

        with pytest.raises(ValueError, match="query cannot be empty"):
            service.ingest_and_reindex("")

    def test_whitespace_query_raises_value_error(self):
        """Whitespace-only query should raise ValueError."""
        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(),
        )

        with pytest.raises(ValueError, match="query cannot be empty"):
            service.ingest_and_reindex("   ")


# =============================================================================
# Tests: Pipeline flow
# =============================================================================


class TestIngestAndReindexPipeline:
    """Tests for the complete ingestion pipeline."""

    def test_calls_external_provider_with_correct_params(self):
        """Should call external provider with query, max_results, language."""
        provider = FakeExternalBooksProvider()
        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=provider,
        )

        service.ingest_and_reindex("python", max_results=15, language="en")

        assert len(provider.search_calls) == 1
        call = provider.search_calls[0]
        assert call["query"] == "python"
        assert call["max_results"] == 15
        assert call["language"] == "en"

    def test_saves_fetched_books_to_catalog(self):
        """Should save fetched books via save_many."""
        books = [
            _make_book("Book 1", "id1"),
            _make_book("Book 2", "id2"),
        ]
        catalog = FakeBookCatalogRepository()
        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(books),
        )

        service.ingest_and_reindex("test")

        assert len(catalog.save_many_calls) == 1
        assert len(catalog.save_many_calls[0]) == 2

    def test_does_not_call_save_many_when_no_books_fetched(self):
        """Should not call save_many when provider returns empty list."""
        catalog = FakeBookCatalogRepository()
        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider([]),  # Empty
        )

        service.ingest_and_reindex("test")

        assert len(catalog.save_many_calls) == 0

    def test_loads_full_catalog_for_reindexing(self):
        """Should call get_all to load full catalog for index rebuild."""
        # Pre-populate catalog with existing books
        existing = [_make_book("Existing", "existing1")]
        catalog = FakeBookCatalogRepository(initial_books=existing)
        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider([]),
        )

        service.ingest_and_reindex("test")

        assert catalog.get_all_calls == 1

    def test_rebuilds_lexical_index_from_full_catalog(self):
        """Should rebuild lexical index with all books from catalog."""
        existing = [_make_book("Existing", "existing1")]
        new_books = [_make_book("New Book", "new1")]
        catalog = FakeBookCatalogRepository(initial_books=existing)
        lexical = FakeLexicalSearchRepository()

        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=lexical,
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(new_books),
        )

        service.ingest_and_reindex("test")

        # Should have called build_index once
        assert len(lexical.build_index_calls) == 1
        # Should include both existing and new books (2 total after save)
        indexed_books = lexical.build_index_calls[0]
        assert len(indexed_books) == 2

    def test_generates_embeddings_for_all_catalog_books(self):
        """Should generate embeddings for all books in catalog."""
        existing = [_make_book("Existing", "existing1")]
        new_books = [_make_book("New Book", "new1")]
        catalog = FakeBookCatalogRepository(initial_books=existing)
        embeddings = FakeEmbeddingsStore()

        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=embeddings,
            external_provider=FakeExternalBooksProvider(new_books),
        )

        service.ingest_and_reindex("test")

        # Should have called generate_embeddings_batch once
        assert len(embeddings.generate_batch_calls) == 1
        # With 2 texts (one per book)
        assert len(embeddings.generate_batch_calls[0]) == 2

    def test_stores_embeddings_with_book_ids(self):
        """Should store embeddings associated with correct book IDs."""
        books = [
            _make_book("Book 1", "id1"),
            _make_book("Book 2", "id2"),
        ]
        catalog = FakeBookCatalogRepository()
        embeddings = FakeEmbeddingsStore()

        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=embeddings,
            external_provider=FakeExternalBooksProvider(books),
        )

        service.ingest_and_reindex("test")

        # Should have called store_embeddings_batch once
        assert len(embeddings.store_batch_calls) == 1
        stored_ids, stored_embs = embeddings.store_batch_calls[0]
        assert len(stored_ids) == 2
        assert len(stored_embs) == 2

    def test_builds_vector_index_after_storing_embeddings(self):
        """Should call build_index on embeddings store."""
        books = [_make_book("Book 1", "id1")]
        embeddings = FakeEmbeddingsStore()

        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=embeddings,
            external_provider=FakeExternalBooksProvider(books),
        )

        service.ingest_and_reindex("test")

        assert embeddings.build_index_calls == 1

    def test_returns_fetched_books_not_full_catalog(self):
        """Should return only the newly fetched books."""
        existing = [_make_book("Existing", "existing1")]
        new_books = [
            _make_book("New 1", "new1"),
            _make_book("New 2", "new2"),
        ]
        catalog = FakeBookCatalogRepository(initial_books=existing)

        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(new_books),
        )

        result = service.ingest_and_reindex("test")

        # Should return only the 2 new books, not the existing one
        assert len(result) == 2
        assert result[0].title == "New 1"
        assert result[1].title == "New 2"


# =============================================================================
# Tests: Index persistence
# =============================================================================


class TestIndexPersistence:
    """Tests for optional index persistence."""

    def test_saves_lexical_index_when_path_provided(self):
        """Should save lexical index when persist path is provided."""
        books = [_make_book("Book", "id1")]
        lexical = FakeLexicalSearchRepository()

        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=lexical,
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(books),
        )

        service.ingest_and_reindex(
            "test",
            persist_lexical_index_path="/tmp/lexical.pkl",
        )

        assert len(lexical.save_index_calls) == 1
        assert lexical.save_index_calls[0] == "/tmp/lexical.pkl"

    def test_does_not_save_lexical_index_when_no_path(self):
        """Should not save lexical index when no path provided."""
        books = [_make_book("Book", "id1")]
        lexical = FakeLexicalSearchRepository()

        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=lexical,
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(books),
        )

        service.ingest_and_reindex("test")

        assert len(lexical.save_index_calls) == 0

    def test_saves_vector_index_when_path_provided(self):
        """Should save vector index when persist path is provided."""
        books = [_make_book("Book", "id1")]
        embeddings = FakeEmbeddingsStore()

        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=embeddings,
            external_provider=FakeExternalBooksProvider(books),
        )

        service.ingest_and_reindex(
            "test",
            persist_vector_index_dir="/tmp/vector",
        )

        assert len(embeddings.save_index_calls) == 1
        assert embeddings.save_index_calls[0] == "/tmp/vector"


# =============================================================================
# Tests: Empty catalog handling
# =============================================================================


class TestEmptyCatalogHandling:
    """Tests for handling empty catalog scenarios."""

    def test_rebuilds_indices_even_when_no_books_fetched(self):
        """Should rebuild indices even if external provider returns nothing."""
        existing = [_make_book("Existing", "existing1")]
        catalog = FakeBookCatalogRepository(initial_books=existing)
        lexical = FakeLexicalSearchRepository()
        embeddings = FakeEmbeddingsStore()

        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=lexical,
            embeddings_store=embeddings,
            external_provider=FakeExternalBooksProvider([]),  # No new books
        )

        service.ingest_and_reindex("test")

        # Should still rebuild indices from existing catalog
        assert len(lexical.build_index_calls) == 1
        assert len(lexical.build_index_calls[0]) == 1  # 1 existing book
        assert embeddings.build_index_calls == 1

    def test_skips_vector_index_when_catalog_empty(self):
        """Should skip vector index build when catalog is completely empty."""
        embeddings = FakeEmbeddingsStore()

        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),  # Empty
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=embeddings,
            external_provider=FakeExternalBooksProvider([]),  # No new books
        )

        service.ingest_and_reindex("test")

        # Should not call embeddings methods when no books
        assert len(embeddings.generate_batch_calls) == 0
        assert len(embeddings.store_batch_calls) == 0
        assert embeddings.build_index_calls == 0


# =============================================================================
# Tests: Error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_wraps_provider_exception_in_runtime_error(self):
        """Should wrap external provider exceptions in RuntimeError."""

        class FailingProvider:
            def search_books(self, query, max_results=10, language=None):
                raise ConnectionError("Network error")

            def get_book_by_id(self, external_id):
                return None

            def get_source_name(self):
                return "failing"

        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FailingProvider(),
        )

        with pytest.raises(RuntimeError, match="Failed to fetch books"):
            service.ingest_and_reindex("test")

    def test_wraps_catalog_exception_in_runtime_error(self):
        """Should wrap catalog exceptions in RuntimeError."""

        class FailingCatalog(FakeBookCatalogRepository):
            def save_many(self, books):
                raise Exception("Database error")

        books = [_make_book("Book", "id1")]
        service = CatalogIngestionService(
            catalog_repo=FailingCatalog(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(books),
        )

        with pytest.raises(RuntimeError, match="Failed to persist books"):
            service.ingest_and_reindex("test")

    def test_wraps_lexical_exception_in_runtime_error(self):
        """Should wrap lexical index exceptions in RuntimeError."""

        class FailingLexical(FakeLexicalSearchRepository):
            def build_index(self, books):
                raise Exception("Index error")

        books = [_make_book("Book", "id1")]
        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FailingLexical(),
            embeddings_store=FakeEmbeddingsStore(),
            external_provider=FakeExternalBooksProvider(books),
        )

        with pytest.raises(RuntimeError, match="Failed to rebuild lexical index"):
            service.ingest_and_reindex("test")

    def test_wraps_embeddings_exception_in_runtime_error(self):
        """Should wrap embeddings exceptions in RuntimeError."""

        class FailingEmbeddings(FakeEmbeddingsStore):
            def generate_embeddings_batch(self, texts):
                raise Exception("Embedding error")

        books = [_make_book("Book", "id1")]
        service = CatalogIngestionService(
            catalog_repo=FakeBookCatalogRepository(),
            lexical_repo=FakeLexicalSearchRepository(),
            embeddings_store=FailingEmbeddings(),
            external_provider=FakeExternalBooksProvider(books),
        )

        with pytest.raises(RuntimeError, match="Failed to rebuild vector index"):
            service.ingest_and_reindex("test")


# =============================================================================
# Tests: Upsert with real SQLite (source of truth validation)
# =============================================================================


class TestUpsertWithRealSQLite:
    """
    Tests that validate SQLite as source of truth with real repository.
    
    These tests use SQLite in-memory to verify upsert semantics:
    - (source, source_id) uniqueness
    - UUID preservation on update
    - Data updates propagate to reindex
    """

    def test_upsert_preserves_uuid_and_updates_data(self):
        """
        Upsert scenario: provider returns book with same (source, source_id)
        but different UUID and updated title.
        
        Validates:
        1. count() does not increase (no duplicate)
        2. Original UUID is preserved
        3. Title is updated to new value
        4. Reindex uses the updated version
        """
        from app.infrastructure.db.sqlite_book_catalog_repository import (
            SqliteBookCatalogRepository,
        )

        # Use SQLite in-memory for isolation
        catalog = SqliteBookCatalogRepository("sqlite:///:memory:")
        lexical = FakeLexicalSearchRepository()
        embeddings = FakeEmbeddingsStore()

        # Step 1: Insert original book
        original_book = Book(
            id=uuid7(),
            title="Original Title",
            authors=["Author One"],
            description="Original description",
            language="en",
            categories=["Fiction"],
            published_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            source="google_books",
            source_id="vol123",  # Same source_id
        )
        catalog.save(original_book)
        original_uuid = original_book.id

        assert catalog.count() == 1

        # Step 2: Provider returns "updated" book with SAME (source, source_id)
        # but DIFFERENT UUID and updated title
        updated_book = Book(
            id=uuid7(),  # Different UUID (as if freshly fetched from API)
            title="Updated Title",  # Changed
            authors=["Author One", "Author Two"],  # Changed
            description="Updated description",  # Changed
            language="en",
            categories=["Fiction", "Drama"],  # Changed
            published_date=datetime(2024, 6, 15, tzinfo=timezone.utc),  # Changed
            source="google_books",
            source_id="vol123",  # SAME source_id
        )

        provider = FakeExternalBooksProvider([updated_book])
        service = CatalogIngestionService(
            catalog_repo=catalog,
            lexical_repo=lexical,
            embeddings_store=embeddings,
            external_provider=provider,
        )

        # Step 3: Run ingestion
        result = service.ingest_and_reindex("test query")

        # Verify: count() did NOT increase (upsert, not insert)
        assert catalog.count() == 1

        # Verify: Original UUID is preserved
        persisted = catalog.get_by_source_id("google_books", "vol123")
        assert persisted is not None
        assert persisted.id == original_uuid  # UUID preserved!

        # Verify: Data was updated
        assert persisted.title == "Updated Title"
        assert persisted.authors == ["Author One", "Author Two"]
        assert persisted.description == "Updated description"
        assert persisted.categories == ["Fiction", "Drama"]

        # Verify: Reindex was called with updated data
        assert len(lexical.build_index_calls) == 1
        indexed_books = lexical.build_index_calls[0]
        assert len(indexed_books) == 1
        assert indexed_books[0].title == "Updated Title"
        assert indexed_books[0].id == original_uuid

        # Verify: Embeddings were generated for updated book
        assert len(embeddings.store_batch_calls) == 1
        stored_ids, _ = embeddings.store_batch_calls[0]
        assert len(stored_ids) == 1
        assert stored_ids[0] == original_uuid
