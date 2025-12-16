"""
Integration tests for SqliteBookCatalogRepository.

=============================================================================
TEACHING NOTES: Why Integration Tests (Not Mocks)?
=============================================================================

These tests use a REAL SQLite database (in-memory) instead of mocking.
This is intentional because:

1. SQL behavior is hard to mock correctly (transactions, constraints, types)
2. We want to catch real bugs (invalid SQL, wrong column types, etc.)
3. SQLite in-memory is fast enough for tests

Using sqlite:///:memory: gives us:
- Fresh database for each test (isolation)
- No disk I/O (speed)
- Automatic cleanup (no temp files)

=============================================================================
Test Categories:
=============================================================================
1. Basic CRUD: save, get_by_id, delete
2. Upsert behavior: deduplication by (source, source_id)
3. UUID preservation: existing UUID not replaced on upsert
4. Collection operations: get_all, count, limit
5. Metadata round-trip: BookMetadata serialization
6. Edge cases: unicode, empty lists, None values

=============================================================================
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
    Create a fresh in-memory SQLite repository.

    Each test gets an isolated, empty database.
    The :memory: URL tells SQLite to use RAM instead of disk.
    """
    return SqliteBookCatalogRepository("sqlite:///:memory:")


@pytest.fixture
def sample_book() -> Book:
    """
    A fully-populated sample book for testing.

    Includes all fields: metadata, categories, dates, etc.
    """
    return Book(
        id=uuid7(),
        title="Don Quijote de la Mancha",
        authors=["Miguel de Cervantes"],
        description="La historia del ingenioso hidalgo.",
        language="es",
        categories=["Fiction", "Classic"],
        published_date=datetime(1605, 1, 16, tzinfo=timezone.utc),
        source="google_books",
        source_id="cervantes_001",
        metadata=BookMetadata(
            isbn="1234567890",
            isbn13="9781234567890",
            publisher="Editorial Castalia",
            page_count=1023,
            average_rating=4.8,
            ratings_count=5000,
            thumbnail_url="http://example.com/quijote.jpg",
            preview_link="http://example.com/preview/quijote",
        ),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def minimal_book() -> Book:
    """
    A minimal book with only required fields.

    Tests that optional fields (description, metadata, etc.) can be None.
    """
    return Book(
        id=uuid7(),
        title="Minimal Book",
        authors=["Unknown Author"],
        source="test_source",
        source_id="minimal_001",
    )


# -----------------------------------------------------------------------------
# Test: Basic CRUD Operations
# -----------------------------------------------------------------------------


class TestSaveAndGetById:
    """
    Tests for save() and get_by_id() - the fundamental operations.

    These form the foundation: if these don't work, nothing else will.
    """

    def test_save_and_get_by_id_roundtrip(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """
        GIVEN a Book entity
        WHEN saved to the repository and retrieved by ID
        THEN all fields should match the original

        This is the most important test: complete data round-trip.
        """
        # Act
        repo.save(sample_book)
        retrieved = repo.get_by_id(sample_book.id)

        # Assert - all fields preserved
        assert retrieved is not None
        assert retrieved.id == sample_book.id
        assert retrieved.title == sample_book.title
        assert retrieved.authors == sample_book.authors
        assert retrieved.description == sample_book.description
        assert retrieved.language == sample_book.language
        assert retrieved.categories == sample_book.categories
        assert retrieved.source == sample_book.source
        assert retrieved.source_id == sample_book.source_id

    def test_get_by_id_returns_none_for_unknown_id(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN an empty repository
        WHEN querying for a non-existent UUID
        THEN None is returned (no exception raised)
        """
        result = repo.get_by_id(uuid7())
        assert result is None

    def test_save_minimal_book(
        self, repo: SqliteBookCatalogRepository, minimal_book: Book
    ) -> None:
        """
        GIVEN a book with only required fields
        WHEN saved and retrieved
        THEN optional fields remain None/empty
        """
        repo.save(minimal_book)
        retrieved = repo.get_by_id(minimal_book.id)

        assert retrieved is not None
        assert retrieved.description is None
        assert retrieved.language is None
        assert retrieved.metadata is None
        assert retrieved.categories == []


class TestGetBySourceId:
    """
    Tests for get_by_source_id() - lookup by external identifier.

    This method is essential during ingestion to detect duplicates.
    """

    def test_get_by_source_id(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """
        GIVEN a saved book
        WHEN querying by its source and source_id
        THEN the book is found
        """
        repo.save(sample_book)

        retrieved = repo.get_by_source_id(sample_book.source, sample_book.source_id)

        assert retrieved is not None
        assert retrieved.id == sample_book.id
        assert retrieved.title == sample_book.title

    def test_get_by_source_id_not_found(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN an empty repository
        WHEN querying for non-existent source+source_id
        THEN None is returned
        """
        result = repo.get_by_source_id("nonexistent", "xyz123")
        assert result is None

    def test_same_source_id_different_sources(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN books with same source_id but different sources
        WHEN queried by each source+source_id pair
        THEN the correct book is returned for each

        This validates that (source, source_id) is a COMPOSITE key.
        """
        book_google = Book(
            id=uuid7(),
            title="Python Basics (Google)",
            authors=["Author A"],
            source="google_books",
            source_id="python123",  # Same source_id
        )
        book_openlibrary = Book(
            id=uuid7(),
            title="Python Basics (OpenLibrary)",
            authors=["Author B"],
            source="open_library",
            source_id="python123",  # Same source_id, different source
        )

        repo.save(book_google)
        repo.save(book_openlibrary)

        # Both should be stored as separate books
        assert repo.count() == 2

        # Each should be retrievable by its own source+source_id
        google_result = repo.get_by_source_id("google_books", "python123")
        ol_result = repo.get_by_source_id("open_library", "python123")

        assert google_result.title == "Python Basics (Google)"
        assert ol_result.title == "Python Basics (OpenLibrary)"


# -----------------------------------------------------------------------------
# Test: Upsert Behavior (Deduplication)
# -----------------------------------------------------------------------------


class TestUpsertBehavior:
    """
    Tests for upsert semantics: UPDATE if exists, INSERT if not.

    The deduplication key is (source, source_id).
    This prevents importing the same external book twice.
    """

    def test_save_upsert_preserves_existing_uuid_on_same_source_and_source_id(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN a book already saved
        WHEN saving another book with same (source, source_id) but different UUID
        THEN the original UUID is preserved, content is updated

        This is CRITICAL: indices reference books by UUID. If we replaced
        the UUID on re-import, all index references would break.
        """
        # Arrange - save original book
        original_uuid = uuid7()
        original = Book(
            id=original_uuid,
            title="Original Title",
            authors=["Original Author"],
            source="google_books",
            source_id="dup_test_001",
        )
        repo.save(original)

        # Act - save "updated" book with DIFFERENT UUID but same source+source_id
        new_uuid = uuid7()
        updated = Book(
            id=new_uuid,  # Different UUID!
            title="Updated Title",
            authors=["Updated Author"],
            source="google_books",
            source_id="dup_test_001",  # Same source+source_id
            description="New description added",
        )
        repo.save(updated)

        # Assert
        # 1. Only one book exists (upsert, not insert)
        assert repo.count() == 1

        # 2. UUID is preserved (original, not new)
        retrieved = repo.get_by_source_id("google_books", "dup_test_001")
        assert retrieved.id == original_uuid
        assert retrieved.id != new_uuid

        # 3. Content is updated
        assert retrieved.title == "Updated Title"
        assert retrieved.authors == ["Updated Author"]
        assert retrieved.description == "New description added"

        # 4. New UUID cannot be found
        assert repo.get_by_id(new_uuid) is None

    def test_upsert_updates_updated_at_timestamp(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN a saved book
        WHEN updated via upsert
        THEN updated_at timestamp changes, created_at stays the same
        """
        original = Book(
            id=uuid7(),
            title="Original",
            authors=["Author"],
            source="test",
            source_id="timestamp_test",
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        repo.save(original)

        # Update
        updated = Book(
            id=uuid7(),
            title="Updated",
            authors=["Author"],
            source="test",
            source_id="timestamp_test",
        )
        repo.save(updated)

        retrieved = repo.get_by_id(original.id)

        # created_at should be from the original
        assert retrieved.created_at.year == 2020

        # updated_at should be more recent (set during upsert)
        assert retrieved.updated_at > original.updated_at


# -----------------------------------------------------------------------------
# Test: Collection Operations
# -----------------------------------------------------------------------------


class TestCollectionOperations:
    """Tests for get_all() and count() operations."""

    def test_get_all_empty_repository(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Empty repository returns empty list."""
        assert repo.get_all() == []
        assert repo.count() == 0

    def test_get_all_respects_limit(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN 5 books in the repository
        WHEN calling get_all(limit=3)
        THEN exactly 3 books are returned
        """
        # Arrange - add 5 books
        for i in range(5):
            book = Book(
                id=uuid7(),
                title=f"Book {i}",
                authors=["Author"],
                source="test",
                source_id=f"limit_test_{i}",
            )
            repo.save(book)

        # Act
        limited = repo.get_all(limit=3)
        all_books = repo.get_all()

        # Assert
        assert len(limited) == 3
        assert len(all_books) == 5
        assert repo.count() == 5

    def test_count_increments_on_save(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Count increases with each new book, not on upserts."""
        assert repo.count() == 0

        book = Book(
            id=uuid7(),
            title="Book 1",
            authors=["Author"],
            source="test",
            source_id="count_test_1",
        )
        repo.save(book)
        assert repo.count() == 1

        # Upsert same book - count should NOT increase
        repo.save(book)
        assert repo.count() == 1

        # New book - count should increase
        book2 = Book(
            id=uuid7(),
            title="Book 2",
            authors=["Author"],
            source="test",
            source_id="count_test_2",
        )
        repo.save(book2)
        assert repo.count() == 2


# -----------------------------------------------------------------------------
# Test: Delete Operation
# -----------------------------------------------------------------------------


class TestDelete:
    """Tests for delete() operation."""

    def test_delete_returns_true_when_deleted_and_false_when_missing(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """
        GIVEN a saved book
        WHEN deleted
        THEN returns True and book is gone

        GIVEN a non-existent UUID
        WHEN deleted
        THEN returns False
        """
        repo.save(sample_book)
        assert repo.count() == 1

        # Delete existing - should return True
        result = repo.delete(sample_book.id)
        assert result is True
        assert repo.count() == 0
        assert repo.get_by_id(sample_book.id) is None

        # Delete non-existent - should return False
        result = repo.delete(uuid7())
        assert result is False

    def test_delete_is_idempotent(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """Deleting an already-deleted book returns False (no error)."""
        repo.save(sample_book)

        repo.delete(sample_book.id)  # First delete
        result = repo.delete(sample_book.id)  # Second delete

        assert result is False


# -----------------------------------------------------------------------------
# Test: Metadata Round-Trip
# -----------------------------------------------------------------------------


class TestMetadataRoundtrip:
    """Tests for BookMetadata JSON serialization."""

    def test_metadata_roundtrip(
        self, repo: SqliteBookCatalogRepository, sample_book: Book
    ) -> None:
        """
        GIVEN a book with full BookMetadata
        WHEN saved and retrieved
        THEN all metadata fields are preserved
        """
        repo.save(sample_book)
        retrieved = repo.get_by_id(sample_book.id)

        assert retrieved.metadata is not None
        original_meta = sample_book.metadata
        retrieved_meta = retrieved.metadata

        assert retrieved_meta.isbn == original_meta.isbn
        assert retrieved_meta.isbn13 == original_meta.isbn13
        assert retrieved_meta.publisher == original_meta.publisher
        assert retrieved_meta.page_count == original_meta.page_count
        assert retrieved_meta.average_rating == original_meta.average_rating
        assert retrieved_meta.ratings_count == original_meta.ratings_count
        assert retrieved_meta.thumbnail_url == original_meta.thumbnail_url
        assert retrieved_meta.preview_link == original_meta.preview_link

    def test_partial_metadata(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN a book with partial metadata (some fields None)
        WHEN saved and retrieved
        THEN present fields are preserved, None fields stay None
        """
        book = Book(
            id=uuid7(),
            title="Partial Metadata Book",
            authors=["Author"],
            source="test",
            source_id="partial_meta",
            metadata=BookMetadata(
                isbn="1111111111",
                page_count=200,
                # Other fields are None
            ),
        )

        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved.metadata is not None
        assert retrieved.metadata.isbn == "1111111111"
        assert retrieved.metadata.page_count == 200
        assert retrieved.metadata.publisher is None
        assert retrieved.metadata.average_rating is None

    def test_null_metadata(
        self, repo: SqliteBookCatalogRepository, minimal_book: Book
    ) -> None:
        """Book without metadata should have metadata=None after retrieval."""
        repo.save(minimal_book)
        retrieved = repo.get_by_id(minimal_book.id)

        assert retrieved.metadata is None


# -----------------------------------------------------------------------------
# Test: Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_unicode_in_authors_and_title(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN a book with non-ASCII characters (Spanish, accents, etc.)
        WHEN saved and retrieved
        THEN unicode is preserved correctly

        This tests that ensure_ascii=False works in JSON serialization.
        """
        book = Book(
            id=uuid7(),
            title="Cien anos de soledad",
            authors=["Gabriel Garcia Marquez", "Jose Martinez"],
            description="Novela sobre la familia Buendia.",
            source="test",
            source_id="unicode_test",
        )

        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved.title == book.title
        assert retrieved.authors == book.authors
        assert retrieved.description == book.description

    def test_empty_categories_list(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Empty categories list should remain empty (not None)."""
        book = Book(
            id=uuid7(),
            title="No Categories",
            authors=["Author"],
            categories=[],
            source="test",
            source_id="empty_cats",
        )

        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved.categories == []

    def test_many_authors(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Books with many authors should be stored correctly."""
        book = Book(
            id=uuid7(),
            title="Collaborative Work",
            authors=[f"Author {i}" for i in range(20)],
            source="test",
            source_id="many_authors",
        )

        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert len(retrieved.authors) == 20
        assert retrieved.authors[0] == "Author 0"
        assert retrieved.authors[19] == "Author 19"

    def test_datetime_timezone_handling(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN a book with timezone-aware datetimes
        WHEN saved and retrieved
        THEN the date/time values are preserved
        """
        published = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        book = Book(
            id=uuid7(),
            title="Timezone Test",
            authors=["Author"],
            published_date=published,
            source="test",
            source_id="tz_test",
        )

        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved.published_date is not None
        assert retrieved.published_date.year == 2024
        assert retrieved.published_date.month == 6
        assert retrieved.published_date.day == 15

    def test_special_characters_in_strings(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """Strings with quotes, backslashes, etc. should be handled."""
        book = Book(
            id=uuid7(),
            title='Book "With" Quotes & Special <Characters>',
            authors=["O'Brien", "Smith\\Jones"],
            description="Description with\nnewlines\tand\ttabs.",
            source="test",
            source_id="special_chars",
        )

        repo.save(book)
        retrieved = repo.get_by_id(book.id)

        assert retrieved.title == book.title
        assert retrieved.authors == book.authors
        assert retrieved.description == book.description


# -----------------------------------------------------------------------------
# Test: Batch Operations
# -----------------------------------------------------------------------------


class TestSaveMany:
    """Tests for save_many() batch operation."""

    def test_save_many_basic(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """save_many inserts multiple books in one transaction."""
        books = [
            Book(
                id=uuid7(),
                title=f"Batch Book {i}",
                authors=["Author"],
                source="batch_test",
                source_id=f"batch_{i}",
            )
            for i in range(10)
        ]

        repo.save_many(books)

        assert repo.count() == 10
        for book in books:
            assert repo.get_by_id(book.id) is not None

    def test_save_many_empty_list(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """save_many with empty list does nothing (no error)."""
        repo.save_many([])
        assert repo.count() == 0

    def test_save_many_with_upserts(
        self, repo: SqliteBookCatalogRepository
    ) -> None:
        """
        GIVEN an existing book
        WHEN save_many includes a book with same (source, source_id)
        THEN the existing book is updated, new books are inserted
        """
        # Pre-existing book
        existing = Book(
            id=uuid7(),
            title="Existing Book",
            authors=["Original Author"],
            source="batch_upsert",
            source_id="existing_001",
        )
        repo.save(existing)

        # Batch with one upsert + one new insert
        books = [
            Book(
                id=uuid7(),  # Different UUID
                title="Updated Existing",
                authors=["Updated Author"],
                source="batch_upsert",
                source_id="existing_001",  # Same - will upsert
            ),
            Book(
                id=uuid7(),
                title="New Book",
                authors=["New Author"],
                source="batch_upsert",
                source_id="new_001",  # New - will insert
            ),
        ]

        repo.save_many(books)

        # 2 books total (1 updated + 1 new)
        assert repo.count() == 2

        # Existing book updated but UUID preserved
        updated = repo.get_by_id(existing.id)
        assert updated.title == "Updated Existing"
