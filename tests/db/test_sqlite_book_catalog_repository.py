"""
Integration tests for SqliteBookCatalogRepository.

These tests use a real in-memory SQLite database (no mocks) to verify
the repository correctly implements the BookCatalogRepository protocol.

Test categories:
1. Basic CRUD operations (save, get, delete)
2. Upsert semantics (deduplication by source+source_id)
3. UUID preservation on upsert
4. Edge cases (empty lists, None values, special characters)
5. Batch operations (save_many transactionality)
"""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from app.domain.utils.uuid7 import uuid7

from app.domain.entities import Book
from app.domain.value_objects import BookMetadata
from app.infrastructure.db.sqlite_book_catalog_repository import (
    SqliteBookCatalogRepository,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def repo() -> SqliteBookCatalogRepository:
    """
    Create a fresh in-memory SQLite repository for each test.

    Using :memory: ensures tests are isolated and fast.
    """
    return SqliteBookCatalogRepository("sqlite:///:memory:")


@pytest.fixture
def sample_book() -> Book:
    """Create a sample book for testing."""
    return Book(
        id=uuid7(),
        title="Don Quijote de la Mancha",
        authors=["Miguel de Cervantes"],
        description="La historia del ingenioso hidalgo.",
        language="es",
        categories=["Fiction", "Classic"],
        published_date=datetime(1605, 1, 16, tzinfo=timezone.utc),
        source="google_books",
        source_id="abc123",
        metadata=BookMetadata(
            isbn="1234567890",
            page_count=1000,
            average_rating=4.8,
        ),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_book_without_metadata() -> Book:
    """Create a sample book without optional metadata."""
    return Book(
        id=uuid7(),
        title="Simple Book",
        authors=["Author One"],
        source="open_library",
        source_id="ol12345",
    )


# -----------------------------------------------------------------------------
# Basic CRUD Tests
# -----------------------------------------------------------------------------


class TestSaveAndRetrieve:
    """Tests for save() and get_by_id() operations."""

    def test_save_and_get_by_id(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """A saved book should be retrievable by its UUID."""
        repo.save(sample_book)

        retrieved = repo.get_by_id(sample_book.id)

        assert retrieved is not None
        assert retrieved.id == sample_book.id
        assert retrieved.title == sample_book.title
        assert retrieved.authors == sample_book.authors
        assert retrieved.description == sample_book.description
        assert retrieved.language == sample_book.language
        assert retrieved.categories == sample_book.categories
        assert retrieved.source == sample_book.source
        assert retrieved.source_id == sample_book.source_id

    def test_get_by_id_not_found(self, repo: SqliteBookCatalogRepository) -> None:
        """Getting a non-existent UUID should return None."""
        result = repo.get_by_id(uuid7())
        assert result is None

    def test_save_preserves_metadata(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """BookMetadata should round-trip through save/get."""
        repo.save(sample_book)
        retrieved = repo.get_by_id(sample_book.id)

        assert retrieved is not None
        assert retrieved.metadata is not None
        assert retrieved.metadata.isbn == "1234567890"
        assert retrieved.metadata.page_count == 1000
        assert retrieved.metadata.average_rating == 4.8

    def test_save_without_metadata(
        self, repo: SqliteBookCatalogRepository, sample_book_without_metadata: Book
    ) -> None:
        """Books without metadata should save and load correctly."""
        repo.save(sample_book_without_metadata)
        retrieved = repo.get_by_id(sample_book_without_metadata.id)

        assert retrieved is not None
        assert retrieved.metadata is None

    def test_save_with_unicode_authors(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Authors with non-ASCII characters should be preserved."""
        book = Book(
            id=uuid7(),
            title="Test Book",
            authors=["José García Márquez", "Müller", "João Silva"],
            source="test",
            source_id="unicode_test",
        )
        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved is not None
        assert retrieved.authors == book.authors


class TestGetBySourceId:
    """Tests for get_by_source_id() operation."""

    def test_get_by_source_id_found(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """A saved book should be retrievable by source+source_id."""
        repo.save(sample_book)

        retrieved = repo.get_by_source_id(sample_book.source, sample_book.source_id)

        assert retrieved is not None
        assert retrieved.id == sample_book.id

    def test_get_by_source_id_not_found(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Non-existent source+source_id should return None."""
        result = repo.get_by_source_id("nonexistent", "xyz")
        assert result is None

    def test_get_by_source_id_different_source_same_id(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Same source_id from different sources should be distinct."""
        book1 = Book(
            id=uuid7(),
            title="Book from Google",
            authors=["Author A"],
            source="google_books",
            source_id="shared_id",
        )
        book2 = Book(
            id=uuid7(),
            title="Book from Open Library",
            authors=["Author B"],
            source="open_library",
            source_id="shared_id",
        )

        repo.save(book1)
        repo.save(book2)

        google_book = repo.get_by_source_id("google_books", "shared_id")
        ol_book = repo.get_by_source_id("open_library", "shared_id")

        assert google_book is not None
        assert ol_book is not None
        assert google_book.id != ol_book.id
        assert google_book.title == "Book from Google"
        assert ol_book.title == "Book from Open Library"


# -----------------------------------------------------------------------------
# Upsert & Deduplication Tests
# -----------------------------------------------------------------------------


class TestUpsertBehavior:
    """Tests for upsert semantics based on (source, source_id)."""

    def test_upsert_updates_existing_book(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        Saving a book with same (source, source_id) should update, not insert.
        """
        original = Book(
            id=uuid7(),
            title="Original Title",
            authors=["Original Author"],
            source="google_books",
            source_id="upsert_test",
        )
        repo.save(original)

        updated = Book(
            id=uuid7(),  # Different UUID!
            title="Updated Title",
            authors=["Updated Author", "Second Author"],
            source="google_books",
            source_id="upsert_test",  # Same source+source_id
            description="New description",
        )
        repo.save(updated)

        # Should have exactly 1 book
        assert repo.count() == 1

        # Retrieve by original UUID (should still work)
        retrieved = repo.get_by_id(original.id)
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        assert retrieved.authors == ["Updated Author", "Second Author"]
        assert retrieved.description == "New description"

    def test_upsert_preserves_original_uuid(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        On upsert, the original UUID should be preserved, not replaced.

        This is critical because other systems (BM25 index, vector index)
        may reference books by UUID.
        """
        original_uuid = uuid7()
        original = Book(
            id=original_uuid,
            title="Original",
            authors=["Author"],
            source="test_source",
            source_id="preserve_uuid_test",
        )
        repo.save(original)

        new_uuid = uuid7()
        updated = Book(
            id=new_uuid,  # Different UUID
            title="Updated",
            authors=["Author"],
            source="test_source",
            source_id="preserve_uuid_test",
        )
        repo.save(updated)

        # Book should still have original UUID
        retrieved = repo.get_by_source_id("test_source", "preserve_uuid_test")
        assert retrieved is not None
        assert retrieved.id == original_uuid
        assert retrieved.id != new_uuid

        # New UUID should not exist
        by_new_uuid = repo.get_by_id(new_uuid)
        assert by_new_uuid is None


# -----------------------------------------------------------------------------
# get_all and count Tests
# -----------------------------------------------------------------------------


class TestGetAllAndCount:
    """Tests for get_all() and count() operations."""

    def test_get_all_empty(self, repo: SqliteBookCatalogRepository) -> None:
        """Empty catalog should return empty list."""
        assert repo.get_all() == []
        assert repo.count() == 0

    def test_get_all_with_books(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """get_all should return all saved books."""
        book2 = Book(
            id=uuid7(),
            title="Second Book",
            authors=["Author Two"],
            source="test",
            source_id="second",
        )

        repo.save(sample_book)
        repo.save(book2)

        all_books = repo.get_all()
        assert len(all_books) == 2
        assert repo.count() == 2

    def test_get_all_with_limit(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """get_all(limit=N) should return at most N books in deterministic order."""
        for i in range(5):
            book = Book(
                id=uuid7(),
                title=f"Book {i}",
                authors=["Author"],
                source="test",
                source_id=f"book_{i}",
            )
            repo.save(book)

        # Verify limit is respected
        limited = repo.get_all(limit=3)
        assert len(limited) == 3
        assert repo.count() == 5

        # Verify deterministic ordering (by insertion order via id asc)
        all_books = repo.get_all()
        assert [b.title for b in all_books] == ["Book 0", "Book 1", "Book 2", "Book 3", "Book 4"]


# -----------------------------------------------------------------------------
# Delete Tests
# -----------------------------------------------------------------------------


class TestDelete:
    """Tests for delete() operation."""

    def test_delete_existing_book(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """Deleting an existing book should return True and remove it."""
        repo.save(sample_book)
        assert repo.count() == 1

        result = repo.delete(sample_book.id)

        assert result is True
        assert repo.count() == 0
        assert repo.get_by_id(sample_book.id) is None

    def test_delete_nonexistent_book(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Deleting a non-existent book should return False."""
        result = repo.delete(uuid7())
        assert result is False


# -----------------------------------------------------------------------------
# Batch Operations Tests
# -----------------------------------------------------------------------------


class TestSaveMany:
    """Tests for save_many() batch operation."""

    def test_save_many_basic(self, repo: SqliteBookCatalogRepository) -> None:
        """save_many should insert multiple books efficiently."""
        books = [
            Book(
                id=uuid7(),
                title=f"Book {i}",
                authors=[f"Author {i}"],
                source="batch_test",
                source_id=f"batch_{i}",
            )
            for i in range(10)
        ]

        repo.save_many(books)

        assert repo.count() == 10
        for book in books:
            retrieved = repo.get_by_id(book.id)
            assert retrieved is not None

    def test_save_many_empty_list(self, repo: SqliteBookCatalogRepository) -> None:
        """save_many with empty list should not error."""
        repo.save_many([])
        assert repo.count() == 0

    def test_save_many_with_upserts(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """save_many should handle mixed inserts and upserts."""
        # Insert initial book
        existing = Book(
            id=uuid7(),
            title="Existing Book",
            authors=["Existing Author"],
            source="batch_upsert",
            source_id="existing",
        )
        repo.save(existing)

        # Batch with one upsert and one insert
        books = [
            Book(
                id=uuid7(),  # Different UUID, but same source+source_id
                title="Updated Existing",
                authors=["Updated Author"],
                source="batch_upsert",
                source_id="existing",
            ),
            Book(
                id=uuid7(),
                title="New Book",
                authors=["New Author"],
                source="batch_upsert",
                source_id="new",
            ),
        ]

        repo.save_many(books)

        # Should have 2 books total (1 updated + 1 new)
        assert repo.count() == 2

        # Existing book should be updated but keep original UUID
        retrieved = repo.get_by_id(existing.id)
        assert retrieved is not None
        assert retrieved.title == "Updated Existing"


# -----------------------------------------------------------------------------
# Edge Cases & Data Integrity Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and data integrity."""

    def test_empty_categories_list(self, repo: SqliteBookCatalogRepository) -> None:
        """Empty categories list should be preserved."""
        book = Book(
            id=uuid7(),
            title="No Categories",
            authors=["Author"],
            categories=[],
            source="test",
            source_id="no_cats",
        )
        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved is not None
        assert retrieved.categories == []

    def test_none_description(self, repo: SqliteBookCatalogRepository) -> None:
        """None description should be preserved."""
        book = Book(
            id=uuid7(),
            title="No Description",
            authors=["Author"],
            description=None,
            source="test",
            source_id="no_desc",
        )
        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved is not None
        assert retrieved.description is None

    def test_datetime_with_timezone(self, repo: SqliteBookCatalogRepository) -> None:
        """Timezone-aware datetimes should round-trip correctly."""
        published = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        book = Book(
            id=uuid7(),
            title="Dated Book",
            authors=["Author"],
            published_date=published,
            source="test",
            source_id="dated",
        )
        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved is not None
        assert retrieved.published_date is not None
        assert retrieved.published_date.year == 2024
        assert retrieved.published_date.month == 6
        assert retrieved.published_date.day == 15

    def test_multiple_authors(self, repo: SqliteBookCatalogRepository) -> None:
        """Multiple authors should be preserved in order."""
        book = Book(
            id=uuid7(),
            title="Multi-Author Book",
            authors=["First Author", "Second Author", "Third Author"],
            source="test",
            source_id="multi_author",
        )
        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved is not None
        assert retrieved.authors == ["First Author", "Second Author", "Third Author"]

    def test_special_characters_in_title(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Titles with special characters should be preserved."""
        book = Book(
            id=uuid7(),
            title="Book: A 'Special' Title with \"Quotes\" & Symbols!",
            authors=["Author"],
            source="test",
            source_id="special_chars",
        )
        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved is not None
        assert retrieved.title == book.title

    def test_save_without_source_id_raises_error(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Saving a book without source_id should raise ValueError."""
        book = Book(
            id=uuid7(),
            title="Book Without Source ID",
            authors=["Author"],
            source="test",
            source_id=None,
        )

        with pytest.raises(ValueError, match="source_id is required"):
            repo.save(book)