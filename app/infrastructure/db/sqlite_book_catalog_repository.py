"""
SQLite implementation of the BookCatalogRepository port.

This adapter persists Book entities to a SQLite database, handling
serialization/deserialization and enforcing the unique constraint on (source, source_id).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from app.domain.entities import Book
from app.domain.ports import BookCatalogRepository
from app.domain.value_objects import BookMetadata

class SqliteBookCatalogRepository(BookCatalogRepository):
    """
    The unique constraint on (source, source_id) is enforced for records where source_id is non-null.
    Books without source_id are treated as non-deduplicable at the catalog level (multiple entries may exist).
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the repository with a database path
        """
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row # Needed to access by name column and not a number
        return conn

    def _init_schema(self) -> None:
        """Create the books table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                description TEXT,
                language TEXT,
                categories TEXT,
                published_date TEXT,
                source TEXT NOT NULL,
                source_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source, source_id)
            )
        """)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_books_source_sourceid ON books(source, source_id)"
            )
        conn.commit()

    def _book_to_row(self, book: Book) -> dict:
        """Convert a Book entity to a database row dict."""
        # Handle metadata serialization
        metadata_json = None
        if book.metadata is not None:
            metadata_json = json.dumps({
                "isbn": book.metadata.isbn,
                "isbn13": book.metadata.isbn13,
                "publisher": book.metadata.publisher,
                "page_count": book.metadata.page_count,
                "average_rating": book.metadata.average_rating,
                "ratings_count": book.metadata.ratings_count,
                "thumbnail_url": book.metadata.thumbnail_url,
                "preview_link": book.metadata.preview_link,
            })

        return {
            "id": str(book.id),
            "title": book.title,
            "authors": json.dumps(book.authors),
            "description": book.description,
            "language": book.language,
            "categories": json.dumps(book.categories or []),
            "published_date": book.published_date.isoformat() if book.published_date else None,
            "source": book.source,
            "source_id": book.source_id,
            "metadata": metadata_json,
            "created_at": book.created_at.isoformat(),
            "updated_at": book.updated_at.isoformat(),
        }

    def _parse_date_safe(self, date_str: str) -> Optional[datetime]:
        """Parse a date string robustly, handling partial dates like '2023' or '2023-05'."""
        if not date_str:
            return None
        
        # Try full ISO format first
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            pass
        
        # Try year-month format: "2023-05"
        try:
            return datetime.strptime(date_str, "%Y-%m")
        except ValueError:
            pass
        
        # Try year-only format: "2023"
        try:
            return datetime.strptime(date_str, "%Y")
        except ValueError:
            pass
        
        return None

    def _row_to_book(self, row: sqlite3.Row) -> Book:
        """Convert a database row to a Book entity."""
        # Deserialize metadata if present
        metadata = None
        if row["metadata"]:
            meta_dict = json.loads(row["metadata"])
            metadata = BookMetadata(
                isbn=meta_dict.get("isbn"),
                isbn13=meta_dict.get("isbn13"),
                publisher=meta_dict.get("publisher"),
                page_count=meta_dict.get("page_count"),
                average_rating=meta_dict.get("average_rating"),
                ratings_count=meta_dict.get("ratings_count"),
                thumbnail_url=meta_dict.get("thumbnail_url"),
                preview_link=meta_dict.get("preview_link"),
            )

        # Deserialize published_date if present (handles partial dates)
        published_date = self._parse_date_safe(row["published_date"])

        cats = json.loads(row["categories"]) if row["categories"] else []
        categories: List[str] = cats or []

        return Book(
            id=UUID(row["id"]),
            title=row["title"],
            authors=json.loads(row["authors"]),
            description=row["description"],
            language=row["language"],
            categories=categories,
            published_date=published_date,
            source=row["source"],
            source_id=row["source_id"],
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


    def count(self) -> int:
        """Get the total number of books in the catalog."""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) as cnt FROM books").fetchone()
            return result["cnt"]

    def get_by_id(self, book_id: UUID) -> Optional[Book]:
        """Retrieve a book by its internal UUID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM books WHERE id = ?",
                (str(book_id),)
            ).fetchone()

            if row is None:
                return None

            return self._row_to_book(row)

    def get_by_source_id(self, source: str, source_id: str) -> Optional[Book]:
        """Retrieve a book by its external source identifier."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM books WHERE source = ? AND source_id = ?",
                (source, source_id)
            ).fetchone()

            if row is None:
                return None

            return self._row_to_book(row)

    def save(self, book: Book) -> None:
        """Save a book to the catalog."""
        row = self._book_to_row(book)

        try:
            with self._get_connection() as conn:
                if book.source_id is not None:
                    existing = conn.execute(
                        "SELECT id FROM books WHERE source = ? AND source_id = ?",
                        (book.source, book.source_id)
                    ).fetchone()

                    if existing and existing["id"] != str(book.id):
                        raise ValueError(
                            f"Book with source='{book.source}' and source_id='{book.source_id}' "
                            f"already exists with different ID"
                        )

                conn.execute("""
                    INSERT INTO books
                    (id, title, authors, description, language, categories,
                     published_date, source, source_id, metadata, created_at, updated_at)
                    VALUES
                    (:id, :title, :authors, :description, :language, :categories,
                     :published_date, :source, :source_id, :metadata, :created_at, :updated_at)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        authors=excluded.authors,
                        description=excluded.description,
                        language=excluded.language,
                        categories=excluded.categories,
                        published_date=excluded.published_date,
                        source=excluded.source,
                        source_id=excluded.source_id,
                        metadata=excluded.metadata,
                        updated_at=excluded.updated_at
                """, row)
                conn.commit()

        except sqlite3.IntegrityError as e:
            raise ValueError(f"Book violates catalog constraints: {e}") from e
        except sqlite3.Error as e:
            raise RuntimeError(f"Database error while saving book: {e}") from e  

    def delete(self, book_id: UUID) -> bool:
        """Delete a book from the catalog. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM books WHERE id = ?",
                (str(book_id),)
            )
            conn.commit()
            return cursor.rowcount > 0

    
    def get_all(self, limit: Optional[int] = None) -> List[Book]:
        """Retrieve all books from the catalog."""
        with self._get_connection() as conn:
            if limit is not None:
                rows = conn.execute(
                    "SELECT * FROM books ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM books ORDER BY created_at DESC"
                ).fetchall()
            
            return [self._row_to_book(row) for row in rows]

    def save_many(self, books: List[Book]) -> None:
        """Save multiple books in a single transaction for efficiency."""
        if not books:
            return

        rows = [self._book_to_row(book) for book in books]

        try:
            with self._get_connection() as conn:
                # Build a single query to check all duplicates at once (O(1) instead of O(n))
                books_with_source_id = [
                    (book.source, book.source_id, str(book.id))
                    for book in books if book.source_id is not None
                ]
                
                if books_with_source_id:
                    pairs = [(src, sid) for src, sid, _ in books_with_source_id]
                    where = " OR ".join(["(source=? AND source_id=?)"] * len(pairs))
                    params = [x for pair in pairs for x in pair]
                    existing_rows = conn.execute(
                        f"SELECT id, source, source_id FROM books WHERE {where}",
                        params,
                    ).fetchall()
                    
                    # Check for conflicts: same (source, source_id) but different id
                    existing_map = {(r["source"], r["source_id"]): r["id"] for r in existing_rows}
                    for source, source_id, book_id in books_with_source_id:
                        existing_id = existing_map.get((source, source_id))
                        if existing_id and existing_id != book_id:
                            raise ValueError(
                                f"Book with source='{source}' and source_id='{source_id}' "
                                f"already exists with different ID"
                            )

                conn.executemany("""
                    INSERT INTO books
                    (id, title, authors, description, language, categories,
                     published_date, source, source_id, metadata, created_at, updated_at)
                    VALUES
                    (:id, :title, :authors, :description, :language, :categories,
                     :published_date, :source, :source_id, :metadata, :created_at, :updated_at)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        authors=excluded.authors,
                        description=excluded.description,
                        language=excluded.language,
                        categories=excluded.categories,
                        published_date=excluded.published_date,
                        source=excluded.source,
                        source_id=excluded.source_id,
                        metadata=excluded.metadata,
                        updated_at=excluded.updated_at
                """, rows)
                conn.commit()

        except sqlite3.IntegrityError as e:
            raise ValueError(f"One or more books violate catalog constraints: {e}") from e
        except sqlite3.Error as e:
            raise RuntimeError(f"Database error while saving books: {e}") from e



