"""
Tests for SqliteBookCatalogRepository.

Validates the SQLite implementation of the BookCatalogRepository protocol,
including CRUD operations, constraint handling, and data serialization.

Test Pattern: AAA (Arrange-Act-Assert)
- Arrange: Set up test data and preconditions
- Act: Execute the operation being tested
- Assert: Verify the expected outcomes
"""
import pytest
from uuid import uuid4

from app.domain.entities import Book
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository


# Arrange (preparar datos prueba) Act (ejecutar accion a probar) Assert (verificar resultado es bueno)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def repo(tmp_path):
    """
    Create a repository with a temporary database for each test.
    
    Uses pytest's tmp_path fixture to ensure isolation between tests.
    The database is automatically cleaned up after each test.
    """
    db_path = tmp_path / "test_catalog.db"
    return SqliteBookCatalogRepository(db_path)


@pytest.fixture
def sample_book():
    """
    Create a sample book entity for testing.
    
    Returns a fully populated Book with all fields set,
    useful for testing serialization/deserialization.
    """
    return Book.create_new(
        title="El Quijote",
        authors=["Miguel de Cervantes"],
        description="Novela clásica española",
        language="es",
        categories=["Fiction", "Classic"],
        source="google_books",
        source_id="quijote_123"
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestRepositoryInitialization:
    """Tests for repository initialization and schema creation."""

    def test_creates_table_on_init(self, repo):
        """Repository should create the books table automatically on initialization."""
        # Assert: New repo should be empty but functional
        assert repo.count() == 0


# ============================================================================
# SAVE OPERATION TESTS
# ============================================================================

class TestSaveOperation:
    """Tests for the save() method."""

    def test_save_increments_count(self, repo, sample_book):
        """Saving a book should increment the total count by one."""
        # Arrange
        assert repo.count() == 0
        
        # Act
        repo.save(sample_book)
        
        # Assert
        assert repo.count() == 1

    def test_save_persists_all_fields(self, repo, sample_book):
        """Saved book should retain all its fields when retrieved."""
        # Act
        repo.save(sample_book)
        retrieved = repo.get_by_id(sample_book.id)
        
        # Assert: All fields should match
        assert retrieved is not None
        assert retrieved.id == sample_book.id
        assert retrieved.title == sample_book.title
        assert retrieved.authors == sample_book.authors
        assert retrieved.description == sample_book.description
        assert retrieved.language == sample_book.language
        assert retrieved.categories == sample_book.categories
        assert retrieved.source == sample_book.source
        assert retrieved.source_id == sample_book.source_id


# ============================================================================
# GET BY ID TESTS
# ============================================================================

class TestGetById:
    """Tests for the get_by_id() method."""

    def test_returns_book_when_exists(self, repo, sample_book):
        """Should return the book when it exists in the database."""
        # Arrange
        repo.save(sample_book)
        
        # Act
        retrieved = repo.get_by_id(sample_book.id)
        
        # Assert
        assert retrieved is not None
        assert retrieved.id == sample_book.id

    def test_returns_none_when_not_found(self, repo):
        """Should return None for non-existent book IDs."""
        # Arrange
        random_uuid = uuid4()
        
        # Act
        retrieved = repo.get_by_id(random_uuid)
        
        # Assert
        assert retrieved is None


# ============================================================================
# DELETE OPERATION TESTS
# ============================================================================

class TestDeleteOperation:
    """Tests for the delete() method."""

    def test_delete_existing_returns_true(self, repo, sample_book):
        """Deleting an existing book should return True and remove it."""
        # Arrange
        repo.save(sample_book)
        
        # Act
        result = repo.delete(sample_book.id)
        
        # Assert
        assert result is True
        assert repo.count() == 0
        assert repo.get_by_id(sample_book.id) is None

    def test_delete_nonexistent_returns_false(self, repo):
        """Deleting a non-existent book should return False."""
        # Arrange
        random_uuid = uuid4()
        
        # Act
        result = repo.delete(random_uuid)
        
        # Assert
        assert result is False


# ============================================================================
# GET ALL TESTS
# ============================================================================

class TestGetAll:
    """Tests for the get_all() method."""

    def test_returns_empty_list_when_no_books(self, repo):
        """Should return an empty list when the catalog is empty."""
        # Act
        result = repo.get_all()
        
        # Assert
        assert result == []

    def test_returns_all_saved_books(self, repo, sample_book):
        """Should return all books that have been saved."""
        # Arrange: Save 4 books total (1 sample + 3 generated)
        repo.save(sample_book)
        for i in range(3):
            book = Book.create_new(
                title=f"Book {i}",
                authors=["Author"],
                source="test",
                source_id=f"id_{i}"
            )
            repo.save(book)
        
        # Act
        result = repo.get_all()
        
        # Assert
        assert len(result) == 4
        assert all(isinstance(book, Book) for book in result)

    def test_get_all_with_limit(self, repo, sample_book):
        """Should return only the specified number of books when limit is set."""
        
        for i in range(3):
            book = Book.create_new(
                title=f"Book {i}",
                authors=["Author"],
                source="test",
                source_id=f"id_{i}"
            )
            repo.save(book)
        
        # Act
        result = repo.get_all(limit = 2)

        assert len(result) == 2
        assert all(isinstance(book, Book) for book in result)
        
        

    def test_get_all_ordered_by_created_at_desc(self, repo):
        """Should return books ordered by created_at in descending order (newest first)."""
        # Arrange: Create books with explicit timestamps to ensure different created_at
        from datetime import datetime, timedelta, UTC
        
        base_time = datetime.now(UTC)
        books = []
        for i in range(3):
            book = Book(
                id=uuid4(),
                title=f"Book {i}",
                authors=["Author"],
                source="test",
                source_id=f"id_{i}",
                created_at=base_time + timedelta(hours=i),  # Book 0 oldest, Book 2 newest
                updated_at=base_time + timedelta(hours=i),
            )
            books.append(book)
            repo.save(book)
        
        # Act
        result = repo.get_all()

        # Assert: DESC order means newest (Book 2) comes first
        assert result[0].title == "Book 2"  # Newest
        assert result[1].title == "Book 1"
        assert result[2].title == "Book 0"  # Oldest
        assert result[0].created_at > result[1].created_at > result[2].created_at  # DESC = primero es más nuevo


# ============================================================================
# SAVE MANY TESTS
# ============================================================================

class TestSaveMany:
    """Tests for the save_many() batch operation."""

    def test_save_many_empty_list(self, repo):
        """Saving an empty list should not fail and not change the count."""
        value_before = repo.count()
        books = []
        repo.save_many(books)
        value_after = repo.count()

        assert value_before == value_after



    def test_save_many_multiple_books(self, repo):
        """Should save multiple books in a single transaction."""
        # Arrange
        value_before = repo.count()
        books = [
            Book.create_new(title=f"Book {i}", authors=["Author"], source="test", source_id=f"id_{i}")
            for i in range(3)
        ]
        
        # Act
        repo.save_many(books)
        
        # Assert
        assert repo.count() == value_before + 3


    def test_save_many_duplicate_source_id_raises_error(self, repo, sample_book):
        """Should raise ValueError when batch contains duplicate (source, source_id) with different ID."""
        # Arrange: First save a book
        repo.save(sample_book)
        
        # Create a new book with SAME (source, source_id) but DIFFERENT UUID
        duplicate_book = Book.create_new(
            title="Different Book",
            authors=["Other Author"],
            source=sample_book.source,        # Same source
            source_id=sample_book.source_id,  # Same source_id
        )
        
        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError):
            repo.save_many([duplicate_book])

    def test_save_many_is_atomic(self, repo):
        """If one book fails validation, none should be saved (transaction rollback)."""
        # Arrange: Create a valid book and one that will cause a constraint violation
        repo.save(Book.create_new(title="Existing", authors=["A"], source="test", source_id="existing_id"))
        initial_count = repo.count()
        
        books = [
            Book.create_new(title="Valid Book", authors=["Author"], source="test", source_id="new_id"),
            Book.create_new(title="Duplicate", authors=["Author"], source="test", source_id="existing_id"),  # Will fail
        ]
        
        # Act: Try to save batch (should fail)
        try:
            repo.save_many(books)
        except ValueError:
            pass  # Expected
        
        # Assert: Count should remain unchanged (atomic = all or nothing)
        # Note: This test may fail if save_many doesn't rollback on error
        assert repo.count() == initial_count


# ============================================================================
# GET BY SOURCE ID TESTS
# ============================================================================

class TestGetBySourceId:
    """Tests for the get_by_source_id() method."""

    def test_returns_book_when_exists(self, repo, sample_book):
        """Should return the book when source and source_id match."""
        repo.save(sample_book)
        source_id = sample_book.source_id
        source = sample_book.source

        book = repo.get_by_source_id(source, source_id)

        assert book.id == sample_book.id

    def test_returns_none_when_not_found(self, repo):
        """Should return None when no book matches the source and source_id."""
        # Arrange: Use non-existent source/source_id
        source = "nonexistent_source"
        source_id = "nonexistent_id"

        # Act
        book = repo.get_by_source_id(source, source_id)

        # Assert
        assert book is None



# ============================================================================
# CONSTRAINT TESTS
# ============================================================================

class TestUniqueConstraints:
    """Tests for database constraint enforcement."""

    def test_save_updates_existing_book_same_id(self, repo, sample_book):
        """Saving a book with the same ID should update it, not create duplicate."""
        repo.save(sample_book)
        original_id = sample_book.id

        new_book = Book(
            id=original_id,  # Same ID
            title="Title updated",  # Different title
            authors=sample_book.authors,
            source=sample_book.source,
            source_id=sample_book.source_id,
            )
         
        repo.save(new_book)

        assert repo.count() ==1
        retrieved = repo.get_by_id(original_id)
        assert retrieved.title == "Title updated"
    


    def test_save_duplicate_source_id_different_id_raises_error(self, repo, sample_book):
        """Should raise ValueError when (source, source_id) exists with different UUID."""
        repo.save(sample_book)

        new_book = Book(
            id=uuid4(),  # Different ID
            title=sample_book.title,  
            authors=sample_book.authors,
            source=sample_book.source,
            source_id=sample_book.source_id,
            )
        
        with pytest.raises(ValueError):
            repo.save_many([new_book])
        




# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Tests for data serialization/deserialization to SQLite."""

    def test_save_and_retrieve_with_metadata(self, repo):
        """Book with full BookMetadata should serialize and deserialize correctly."""
        # Arrange
        from app.domain.value_objects import BookMetadata
        
        metadata = BookMetadata(
            isbn="1234567890",
            isbn13="1234567890123",
            publisher="Editorial Planeta",
            page_count=500,
            average_rating=4.5,
            ratings_count=1000,
            thumbnail_url="https://example.com/cover.jpg",
            preview_link="https://example.com/preview",
        )
        book = Book.create_new(
            title="Book with Metadata",
            authors=["Author"],
            source="test",
            source_id="meta_1",
            metadata=metadata,
        )
        
        # Act
        repo.save(book)
        retrieved = repo.get_by_id(book.id)
        
        # Assert
        assert retrieved.metadata is not None
        assert retrieved.metadata.isbn == "1234567890"
        assert retrieved.metadata.isbn13 == "1234567890123"
        assert retrieved.metadata.publisher == "Editorial Planeta"
        assert retrieved.metadata.page_count == 500
        assert retrieved.metadata.average_rating == 4.5
        assert retrieved.metadata.ratings_count == 1000
        assert retrieved.metadata.thumbnail_url == "https://example.com/cover.jpg"
        assert retrieved.metadata.preview_link == "https://example.com/preview"

    def test_save_and_retrieve_with_null_optional_fields(self, repo):
        """Book with None optional fields should save and load correctly."""
        # Arrange: Book with minimal required fields only
        book = Book.create_new(
            title="Minimal Book",
            authors=["Author"],
            source="test",
            # No description, language, categories, metadata, etc.
        )
        
        # Act
        repo.save(book)
        retrieved = repo.get_by_id(book.id)
        
        # Assert
        assert retrieved is not None
        assert retrieved.title == "Minimal Book"
        assert retrieved.description is None
        assert retrieved.language is None
        assert retrieved.metadata is None

    def test_authors_list_serialization(self, repo):
        """Authors list should be serialized to JSON and back."""
        # Arrange: Book with multiple authors
        book = Book.create_new(
            title="Collaborative Work",
            authors=["Author One", "Author Two", "Author Three"],
            source="test",
            source_id="multi_author",
        )
        
        # Act
        repo.save(book)
        retrieved = repo.get_by_id(book.id)
        
        # Assert
        assert retrieved.authors == ["Author One", "Author Two", "Author Three"]
        assert len(retrieved.authors) == 3

    def test_categories_list_serialization(self, repo):
        """Categories list should be serialized to JSON and back."""
        # Arrange: Book with multiple categories
        book = Book.create_new(
            title="Multi-genre Book",
            authors=["Author"],
            categories=["Fiction", "Mystery", "Thriller", "Bestseller"],
            source="test",
            source_id="multi_cat",
        )
        
        # Act
        repo.save(book)
        retrieved = repo.get_by_id(book.id)
        
        # Assert
        assert retrieved.categories == ["Fiction", "Mystery", "Thriller", "Bestseller"]
        assert len(retrieved.categories) == 4


# ============================================================================
# DATE PARSING TESTS
# ============================================================================

class TestDateParsing:
    """Tests for the _parse_date_safe() helper method."""

    def test_parse_full_iso_date(self, repo):
        """Should parse full ISO format like '2023-05-15T10:30:00'."""
        # Act
        result = repo._parse_date_safe("2023-05-15T10:30:00")
        
        # Assert
        assert result is not None
        assert result.year == 2023
        assert result.month == 5
        assert result.day == 15

    def test_parse_year_month_date(self, repo):
        """Should parse year-month format like '2023-05'."""
        # Act
        result = repo._parse_date_safe("2023-05")
        
        # Assert
        assert result is not None
        assert result.year == 2023
        assert result.month == 5

    def test_parse_year_only_date(self, repo):
        """Should parse year-only format like '2023'."""
        # Act
        result = repo._parse_date_safe("2023")
        
        # Assert
        assert result is not None
        assert result.year == 2023

    def test_parse_invalid_date_returns_none(self, repo):
        """Should return None for invalid date strings without crashing."""
        # Act & Assert: Various invalid formats should return None, not crash
        assert repo._parse_date_safe("not-a-date") is None
        assert repo._parse_date_safe("2023/05/15") is None
        assert repo._parse_date_safe("") is None
        assert repo._parse_date_safe(None) is None


# ============================================================================
# COUNT TESTS
# ============================================================================

class TestCount:
    """Tests for the count() method."""

    def test_count_empty_catalog(self, repo):
        """Should return 0 for an empty catalog."""
        # Assert
        assert repo.count() == 0

    def test_count_reflects_saves_and_deletes(self, repo, sample_book):
        """Count should accurately reflect saves and deletions."""
        # Assert initial state
        assert repo.count() == 0
        
        # Act: Save a book
        repo.save(sample_book)
        assert repo.count() == 1
        
        # Act: Save another book
        another_book = Book.create_new(title="Another", authors=["A"], source="test", source_id="x")
        repo.save(another_book)
        assert repo.count() == 2
        
        # Act: Delete one
        repo.delete(sample_book.id)
        assert repo.count() == 1
        
        # Act: Delete the other
        repo.delete(another_book.id)
        assert repo.count() == 0
