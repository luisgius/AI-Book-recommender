"""
SQLite implementation of BookCatalogRepository.

This adapter implements the BookCatalogRepository port using SQLite as the
persistence mechanism. It serves as the "source of truth" catalog for books.

Key design decisions:
- Uses SQLAlchemy 2.0 with synchronous SQLite driver
- Single 'books' table with denormalized structure (no separate authors/categories tables)
- UUID stored as TEXT because SQLite lacks native UUID type
- Lists (authors, categories) stored as JSON TEXT for simplicity
- Metadata stored as JSON TEXT (optional, may be NULL)
- Upsert semantics based on UNIQUE(source, source_id) constraint
- On conflict, existing UUID is preserved to maintain referential integrity

This aligns with TFG Section 4.2.5 and keeps the design deliberately simple
for an academic project while remaining production-like.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.engine import Engine

from app.domain.entities import Book
from app.domain.value_objects import BookMetadata
from app.domain.ports import BookCatalogRepository


# SQLAlchemy declarative base for ORM mappings
Base = declarative_base()


class BookRow(Base):
    """
    SQLAlchemy ORM model for the 'books' table.
    
    Schema rationale:
    - id: INTEGER PRIMARY KEY (SQLite's internal rowid alias for performance)
    - uuid: TEXT NOT NULL UNIQUE - domain UUID stored as string; needed because
      the domain layer uses UUID for identity, but SQLite lacks native UUID support
    - authors, categories: JSON arrays stored as TEXT with ensure_ascii=False
      to preserve non-ASCII characters (e.g., Spanish author names)
    - metadata_json: optional JSON blob for BookMetadata
    - UNIQUE(source, source_id): prevents duplicate imports from same external API
    """

    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(Text, nullable=False, unique=True, index=True)
    title = Column(Text, nullable=False)
    authors = Column(Text, nullable=False)  # JSON array
    description = Column(Text, nullable=True)
    language = Column(Text, nullable=True)
    categories = Column(Text, nullable=True)  # JSON array
    published_date = Column(Text, nullable=True)  # ISO8601 string
    source = Column(Text, nullable=False)
    source_id = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON blob for BookMetadata
    created_at = Column(Text, nullable=False)  # ISO8601 string
    updated_at = Column(Text, nullable=False)  # ISO8601 string

    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_source_source_id"),
    )


class SqliteBookCatalogRepository(BookCatalogRepository):
    """
    SQLite-backed implementation of BookCatalogRepository.

    This repository manages the canonical book catalog using SQLite. All CRUD
    operations go through this adapter, which handles:
    - UUID <-> TEXT conversion
    - List[str] <-> JSON serialization for authors/categories
    - datetime <-> ISO8601 TEXT conversion
    - BookMetadata <-> JSON serialization
    - Upsert logic with (source, source_id) deduplication

    Usage:
        repo = SqliteBookCatalogRepository("sqlite:///data/catalog.db")
        repo.save(book)
        found = repo.get_by_id(book.id)

    Thread safety: Each instance creates its own engine and session factory.
    For multi-threaded usage, create separate instances or use connection pooling.
    """
    def __init__(self, database_url: str = "sqlite:///data/catalog.db") -> None:
        """
        Initialize the repository with a SQLite database.

        Args:
            database_url: SQLAlchemy connection string. Defaults to a file-based
                         SQLite database in the data/ directory.
                         Use "sqlite:///:memory:" for in-memory testing.

        The constructor:
        1. Creates the SQLAlchemy engine
        2. Creates all tables if they don't exist
        3. Sets up a session factory for transactional operations
        """
        # Create engine with SQLite-specific settings
        # check_same_thread=False allows multi-threaded access (be careful with writes)
        self._engine: Engine = create_engine(
            database_url,
            echo=False, # Set to True for SQL Debugging
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )

        # Create tables if they don't exist
        Base.metadata.create_all(self._engine)

        # Session factory for creating sessions
        self._session_factory = sessionmaker(bind = self._engine)

    def save(self, book: Book) -> None:
        """
        Save or update a book in the catalog (upsert).

        Upsert strategy:
        1. Check if a row with same (source, source_id) exists
        2. If yes: update all fields EXCEPT uuid (preserve existing identity)
        3. If no: insert new row with the book's UUID

        Why preserve existing UUID on conflict?
        - Other systems (BM25 index, vector index) may reference books by UUID
        - Changing UUID on re-import would break those references
        - The domain treats UUID as immutable identity

        Args:
            book: The Book entity to persist

        Raises:
            ValueError: If book data violates constraints (e.g., empty title)
            RuntimeError: If a database error occurs
        """
        # Validate source_id before attempting database operations
        if not book.source_id:
            raise ValueError("book.source_id is required for persistence/deduplication")

        with self._session_factory() as session:
            try:
                self._upsert_book(session, book)
                session.commit()
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Failed to save book: {e}") from e

    def save_many(self, books: List[Book]) -> None:
        """
        Save multiple books in a single transaction.

        This is more efficient than calling save() repeatedly because:
        - Single transaction reduces SQLite lock overhead
        - Batch commit is faster than multiple commits

        All books are saved or none are (transactional).

        Args:
            books: List of Book entities to persist

        Raises:
            ValueError: If any book violates constraints
            RuntimeError: If a database error occurs (all changes rolled back)
        """
        if not books:
            return

        # Validate all books before attempting database operations
        for book in books:
            if not book.source_id:
                raise ValueError("book.source_id is required for persistence/deduplication")

        with self._session_factory() as session:
            try:
                for book in books:
                    self._upsert_book(session, book)
                session.commit()
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Failed to save books: {e}") from e


    def get_by_id(self, book_id: UUID) -> Optional[Book]:
        """
        Retrieve a book by its domain UUID.

        Args:
            book_id: The UUID of the book to retrieve

        Returns:
            The Book entity if found, None otherwise
        """
        with self._session_factory() as session:
            row = session.query(BookRow).filter(
                BookRow.uuid == str(book_id)
            ).first()

            if row is None:
                return None

            return self._row_to_book(row)

    
    def get_by_source_id(self, source: str, source_id: str) -> Optional[Book]:
        """
        Retrieve a book by its external source identifier.

        This is useful during ingestion to check if a book from an external
        API already exists in our catalog before inserting.

        Args:
            source: Source system name (e.g., 'google_books')
            source_id: ID in the external system

        Returns:
            The Book entity if found, None otherwise
        """
        with self._session_factory() as session:
            row = session.query(BookRow).filter(
                BookRow.source == source,
                BookRow.source_id == source_id,
            ).first()

            if row is None:
                return None

            return self._row_to_book(row)

    def get_all(self, limit: Optional[int] = None) -> List[Book]:
        """
        Retrieve all books from the catalog.

        For large catalogs, use the limit parameter to paginate.

        Args:
            limit: Optional maximum number of books to return

        Returns:
            List of all books, up to the specified limit
        """
        with self._session_factory() as session:
            query = session.query(BookRow).order_by(BookRow.id.asc())
            if limit is not None:
                query = query.limit(limit)

            rows = query.all()
            return [self._row_to_book(row) for row in rows]

    def count(self) -> int:
        """
        Get the total number of books in the catalog.

        Returns:
            Total book count
        """
        with self._session_factory() as session:
            return session.query(BookRow).count()

    def delete(self, book_id: UUID) -> bool:
        """
        Delete a book from the catalog.

        Note: This only removes from SQLite. Callers are responsible for
        cleaning up related data in BM25/vector indices if needed.

        Args:
            book_id: UUID of the book to delete

        Returns:
            True if the book was deleted, False if not found
        """
        with self._session_factory() as session:
            try:
                result = session.query(BookRow).filter(
                    BookRow.uuid == str(book_id)
                ).delete()
                session.commit()
                return result > 0
            except Exception as e:
                session.rollback()
                raise RuntimeError(f"Failed to delete book: {e}") from e

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _upsert_book(self, session: Session, book: Book) -> None:
        """
        Insert or update a book row (helper for save/save_many).

        On conflict with (source, source_id):
        - Updates all fields except uuid
        - Preserves the existing uuid to maintain identity consistency

        Args:
            session: Active SQLAlchemy session
            book: Book entity to upsert

        Note: source_id validation is done in save()/save_many() before calling this.
        """
        # Check if book with same (source, source_id) exists
        existing = session.query(BookRow).filter(
            BookRow.source == book.source,
            BookRow.source_id == book.source_id,
        ).first()

        if existing is not None:
            # UPDATE: preserve existing uuid, update other fields
            existing.title = book.title
            existing.authors = self._serialize_list(book.authors)
            existing.description = book.description
            existing.language = book.language
            existing.categories = self._serialize_list(book.categories)
            existing.published_date = self._serialize_datetime(book.published_date)
            existing.metadata_json = self._serialize_metadata(book.metadata)
            existing.updated_at = self._serialize_datetime(datetime.now(timezone.utc))
            # Note: uuid, source, source_id, created_at are NOT updated
        else:
            # INSERT: new row with book's uuid
            row = BookRow(
                uuid=str(book.id),
                title=book.title,
                authors=self._serialize_list(book.authors),
                description=book.description,
                language=book.language,
                categories=self._serialize_list(book.categories),
                published_date=self._serialize_datetime(book.published_date),
                source=book.source,
                source_id=book.source_id,
                metadata_json=self._serialize_metadata(book.metadata),
                created_at=self._serialize_datetime(book.created_at),
                updated_at=self._serialize_datetime(book.updated_at),
            )
            session.add(row)
    
    def _row_to_book(self, row: BookRow) -> Book:
        """
        Convert a SQLAlchemy BookRow to a domain Book entity.

        Handles:
        - TEXT uuid -> UUID
        - JSON TEXT -> List[str] for authors/categories
        - ISO8601 TEXT -> datetime (timezone-aware)
        - JSON TEXT -> BookMetadata (optional)
        """
        return Book(
            id=UUID(row.uuid),
            title=row.title,
            authors=self._deserialize_list(row.authors),
            description=row.description,
            language=row.language,
            categories=self._deserialize_list(row.categories) if row.categories else [],
            published_date=self._deserialize_datetime(row.published_date),
            source=row.source,
            source_id=row.source_id,
            metadata=self._deserialize_metadata(row.metadata_json),
            created_at=self._deserialize_datetime(row.created_at) or datetime.now(timezone.utc),
            updated_at=self._deserialize_datetime(row.updated_at) or datetime.now(timezone.utc),
        )

    @staticmethod
    def _serialize_list(items: List[str]) -> str:
        """Serialize a list of strings to JSON TEXT."""
        return json.dumps(items, ensure_ascii=False)

    @staticmethod
    def _deserialize_list(json_text: str) -> List[str]:
        """Deserialize JSON TEXT to a list of strings."""
        if not json_text:
            return []
        return json.loads(json_text)

    @staticmethod
    def _serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
        """
        Serialize datetime to ISO8601 string.

        Preserves timezone info if present. For naive datetimes,
        stores as-is (no timezone suffix).
        """
        if dt is None:
            return None
        return dt.isoformat()

    @staticmethod
    def _deserialize_datetime(iso_str: Optional[str]) -> Optional[datetime]:
        """
        Deserialize ISO8601 string to datetime.

        Handles both timezone-aware (2024-01-15T10:30:00+00:00)
        and naive (2024-01-15T10:30:00) formats.
        """
        if not iso_str:
            return None
        try:
            # Python 3.11+ handles ISO8601 with timezone well
            return datetime.fromisoformat(iso_str)
        except ValueError:
            # Fallback for edge cases
            return None

    @staticmethod
    def _serialize_metadata(metadata: Optional[BookMetadata]) -> Optional[str]:
        """
        Serialize BookMetadata to JSON TEXT.

        Only includes non-None fields to save space.
        """
        if metadata is None:
            return None

        data = {}
        if metadata.isbn is not None:
            data["isbn"] = metadata.isbn
        if metadata.isbn13 is not None:
            data["isbn13"] = metadata.isbn13
        if metadata.publisher is not None:
            data["publisher"] = metadata.publisher
        if metadata.page_count is not None:
            data["page_count"] = metadata.page_count
        if metadata.average_rating is not None:
            data["average_rating"] = metadata.average_rating
        if metadata.ratings_count is not None:
            data["ratings_count"] = metadata.ratings_count
        if metadata.thumbnail_url is not None:
            data["thumbnail_url"] = metadata.thumbnail_url
        if metadata.preview_link is not None:
            data["preview_link"] = metadata.preview_link

        if not data:
            return None

        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _deserialize_metadata(json_text: Optional[str]) -> Optional[BookMetadata]:
        """Deserialize JSON TEXT to BookMetadata."""
        if not json_text:
            return None

        try:
            data = json.loads(json_text)
            return BookMetadata(
                isbn=data.get("isbn"),
                isbn13=data.get("isbn13"),
                publisher=data.get("publisher"),
                page_count=data.get("page_count"),
                average_rating=data.get("average_rating"),
                ratings_count=data.get("ratings_count"),
                thumbnail_url=data.get("thumbnail_url"),
                preview_link=data.get("preview_link"),
            )
        except (json.JSONDecodeError, KeyError):
            return None