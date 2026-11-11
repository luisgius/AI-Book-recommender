"""
SQLite implementation of the BookCatalogRepository port.

This adapter persists Book entities to a SQLite database, handling
serialization/deserialization and enforcing the unique constraint on (source, source_id).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from uuid import UUID

from app.domain.entities import Book
from app.domain.ports import BookCatalogRepository
from app.domain.value_objects import BookDescription, BookMetadata, SourceRecord

class SqliteBookCatalogRepository(BookCatalogRepository):
    """
    The unique constraint on (source, source_id) is enforced for records where source_id is non-null.
    Books without source_id are treated as non-deduplicable at the catalog level (multiple entries may exist).
    """

    SAVE_CHUNK_SIZE = 400
    """Books per INSERT/duplicate-check round in save_many (keeps SQL variables bounded)"""

    LOOKUP_CHUNK_SIZE = 900
    """Ids per SELECT in get_many (below SQLite's historical 999-variable limit)"""

    def __init__(self, db_path: Path, compact_json: bool = False) -> None:
        """
        Initialize the repository with a database path

        Args:
            db_path: SQLite file (created with its schema if missing)
            compact_json: Store non-ASCII text in the metadata JSON as UTF-8
                instead of \\u escapes. Used by the multi-source catalog
                (multilingual descriptions); the default keeps the format of
                the original catalog.
        """
        self._db_path = db_path
        self._compact_json = compact_json
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

    @staticmethod
    def _metadata_to_dict(book: Book) -> Optional[dict]:
        """
        Serialise BookMetadata for the ``metadata`` JSON column.

        The eight original keys are always written, exactly as before. The
        multi-source fields are added only when they hold a value, so a book
        without them produces the same JSON as before they existed, and old
        rows (which lack them) load with their default values.

        The primary description's text is already in the ``description``
        column; inside ``descriptions`` it is stored with ``"text": null``
        and restored on load, to avoid keeping the same text twice.
        """
        meta = book.metadata
        if meta is None:
            return None
        data = {
            "isbn": meta.isbn,
            "isbn13": meta.isbn13,
            "publisher": meta.publisher,
            "page_count": meta.page_count,
            "average_rating": meta.average_rating,
            "ratings_count": meta.ratings_count,
            "thumbnail_url": meta.thumbnail_url,
            "preview_link": meta.preview_link,
        }
        if meta.work_language is not None:
            data["work_language"] = meta.work_language
        if meta.descriptions:
            data["descriptions"] = [
                {
                    "text": None if d.text == book.description else d.text,
                    "language": d.language,
                    "source": d.source,
                    "license": d.license,
                    "attribution_url": d.attribution_url,
                    "auto_generated": d.auto_generated,
                }
                for d in meta.descriptions
            ]
        if meta.titles:
            data["titles"] = [list(pair) for pair in meta.titles]
        for key in ("wikidata_id", "doi", "wikipedia_editions", "gutenberg_downloads"):
            if getattr(meta, key) is not None:
                data[key] = getattr(meta, key)
        for key in ("openlibrary_ids", "gutenberg_ids", "isbn13s"):
            if getattr(meta, key):
                data[key] = list(getattr(meta, key))
        if meta.provenance:
            data["provenance"] = [
                {"source": r.source, "record_id": r.record_id, "license": r.license, "url": r.url}
                for r in meta.provenance
            ]
        return data

    @staticmethod
    def _metadata_from_dict(meta_dict: dict, description: Optional[str]) -> BookMetadata:
        """Rebuild BookMetadata from the JSON column (old rows lack the new keys)."""
        descriptions = tuple(
            BookDescription(
                text=d["text"] if d.get("text") is not None else (description or ""),
                language=d.get("language"),
                source=d.get("source", "unknown"),
                license=d.get("license"),
                attribution_url=d.get("attribution_url"),
                auto_generated=bool(d.get("auto_generated", False)),
            )
            for d in meta_dict.get("descriptions", [])
            if d.get("text") is not None or description
        )
        return BookMetadata(
            isbn=meta_dict.get("isbn"),
            isbn13=meta_dict.get("isbn13"),
            publisher=meta_dict.get("publisher"),
            page_count=meta_dict.get("page_count"),
            average_rating=meta_dict.get("average_rating"),
            ratings_count=meta_dict.get("ratings_count"),
            thumbnail_url=meta_dict.get("thumbnail_url"),
            preview_link=meta_dict.get("preview_link"),
            work_language=meta_dict.get("work_language"),
            descriptions=descriptions,
            titles=tuple(tuple(pair) for pair in meta_dict.get("titles", [])),
            wikidata_id=meta_dict.get("wikidata_id"),
            openlibrary_ids=tuple(meta_dict.get("openlibrary_ids", [])),
            gutenberg_ids=tuple(meta_dict.get("gutenberg_ids", [])),
            isbn13s=tuple(meta_dict.get("isbn13s", [])),
            doi=meta_dict.get("doi"),
            wikipedia_editions=meta_dict.get("wikipedia_editions"),
            gutenberg_downloads=meta_dict.get("gutenberg_downloads"),
            provenance=tuple(
                SourceRecord(
                    source=r["source"],
                    record_id=r["record_id"],
                    license=r.get("license"),
                    url=r.get("url"),
                )
                for r in meta_dict.get("provenance", [])
            ),
        )

    def _book_to_row(self, book: Book) -> dict:
        """Convert a Book entity to a database row dict."""
        # Handle metadata serialization
        metadata_dict = self._metadata_to_dict(book)
        metadata_json = None
        if metadata_dict is not None:
            metadata_json = json.dumps(metadata_dict, ensure_ascii=not self._compact_json)

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
            metadata = self._metadata_from_dict(json.loads(row["metadata"]), row["description"])

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

    def get_many(self, book_ids: List[UUID]) -> Dict[UUID, Book]:
        """Retrieve several books by id (unknown ids are ignored)."""
        found: Dict[UUID, Book] = {}
        ids = [str(book_id) for book_id in dict.fromkeys(book_ids)]
        with self._get_connection() as conn:
            for start in range(0, len(ids), self.LOOKUP_CHUNK_SIZE):
                chunk = ids[start:start + self.LOOKUP_CHUNK_SIZE]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT * FROM books WHERE id IN ({placeholders})", chunk
                ).fetchall()
                for row in rows:
                    book = self._row_to_book(row)
                    found[book.id] = book
        return found

    def iter_all(self, batch_size: int = 1000) -> Iterator[Book]:
        """
        Stream every book ordered by id, one batch in memory at a time.

        Uses keyset pagination (WHERE id > last id), which stays fast on large
        tables because it walks the primary-key index.
        """
        last_id = ""
        while True:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM books WHERE id > ? ORDER BY id LIMIT ?",
                    (last_id, batch_size),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._row_to_book(row)
            last_id = rows[-1]["id"]

    def save_many(self, books: List[Book]) -> None:
        """
        Save multiple books efficiently.

        Large lists are written in chunks of SAVE_CHUNK_SIZE books, each in
        one transaction, so the duplicate check never exceeds SQLite's limit
        on query variables and bulk ingestion of hundreds of thousands of
        books keeps memory and transaction size bounded.
        """
        for start in range(0, len(books), self.SAVE_CHUNK_SIZE):
            self._save_chunk(books[start:start + self.SAVE_CHUNK_SIZE])

    def _save_chunk(self, books: List[Book]) -> None:
        """Save one chunk of books in a single transaction."""
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



