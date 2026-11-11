"""
Storage of the multi-source fields: backwards compatibility of the metadata
JSON, round trip of the new fields, get_many/iter_all, save_many batching and
the SQLite duplicate-key index.
"""

import json
import sqlite3
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.domain.entities import Book
from app.domain.value_objects import BookDescription, BookMetadata, SourceRecord
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from app.infrastructure.db.sqlite_catalog_build_state import SqliteCatalogBuildState, SqliteDuplicateKeyIndex

# A row exactly as stored by the original Google Books catalog (data/catalog.db).
OLD_METADATA_JSON = (
    '{"isbn": "1887902996", "isbn13": "9781887902991", "publisher": "Franklin, Beedle & Associates, Inc.", '
    '"page_count": 533, "average_rating": 4, "ratings_count": 18, "thumbnail_url": "http://books.google.com/x", '
    '"preview_link": "http://books.google.es/y"}'
)


@pytest.fixture
def repo(tmp_path):
    return SqliteBookCatalogRepository(tmp_path / "catalog.db")


def old_style_book() -> Book:
    return Book(
        id=uuid4(),
        title="Python Programming",
        authors=["John Zelle"],
        description="An introduction to computer science.",
        language="en",
        categories=["Computers"],
        published_date=datetime(2004, 1, 1),
        source="google_books",
        source_id="aJQILlLxRmAC",
        metadata=BookMetadata(
            isbn="1887902996", isbn13="9781887902991", publisher="Franklin, Beedle & Associates, Inc.",
            page_count=533, average_rating=4, ratings_count=18,
            thumbnail_url="http://books.google.com/x", preview_link="http://books.google.es/y",
        ),
    )


class TestBackwardsCompatibility:
    def test_old_row_loads_with_default_new_fields(self, repo, tmp_path):
        book_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(tmp_path / "catalog.db") as conn:
            conn.execute(
                "INSERT INTO books VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (book_id, "Python Programming", '["John Zelle"]', "Intro.", "en", '["Computers"]',
                 "2004", "google_books", "aJQILlLxRmAC", OLD_METADATA_JSON, now, now),
            )
        book = repo.get_by_id(UUID(book_id))
        assert book.metadata.isbn13 == "9781887902991"
        assert book.metadata.descriptions == () and book.metadata.provenance == ()
        assert book.metadata.wikidata_id is None and book.metadata.work_language is None

    def test_old_style_book_serialises_to_the_same_json_as_before(self, repo):
        row = repo._book_to_row(old_style_book())
        assert row["metadata"] == OLD_METADATA_JSON

    def test_searchable_text_of_an_old_style_book_is_unchanged(self):
        book = old_style_book()
        expected = " ".join([book.title, " ".join(book.authors), book.description, " ".join(book.categories)])
        assert book.get_searchable_text() == expected


class TestNewFields:
    def make_multilingual(self) -> Book:
        es = BookDescription("Texto en español " * 15, "es", "wikipedia", "CC BY-SA 4.0", "https://es.wikipedia.org/wiki/A")
        en = BookDescription("English text " * 20, "en", "gutenberg", None, "https://www.gutenberg.org/ebooks/1", True)
        return Book(
            id=uuid4(), title="Obra", authors=["Autora"], description=es.text, language="es",
            source="wikidata", source_id="Q1",
            metadata=BookMetadata(
                isbn13="9788437604947", work_language="es", descriptions=(es, en),
                titles=(("en", "Work"),), wikidata_id="Q1", openlibrary_ids=("OL1W",),
                gutenberg_ids=("1", "2"), isbn13s=("9788437604947",), doi="10.1/x",
                wikipedia_editions=12, gutenberg_downloads=345,
                provenance=(SourceRecord("wikidata", "Q1", "CC0 1.0", "https://www.wikidata.org/wiki/Q1"),),
            ),
        )

    def test_round_trip(self, repo):
        book = self.make_multilingual()
        repo.save(book)
        loaded = repo.get_by_id(book.id)
        assert loaded.metadata == book.metadata

    def test_primary_description_text_is_not_stored_twice(self, repo):
        book = self.make_multilingual()
        stored = json.loads(repo._book_to_row(book)["metadata"])
        assert stored["descriptions"][0]["text"] is None
        assert stored["descriptions"][1]["text"].startswith("English text")


class TestBulkAccess:
    def test_get_many_ignores_unknown_ids(self, repo):
        books = [old_style_book() for _ in range(3)]
        for i, book in enumerate(books):
            book.source_id = f"id{i}"
        repo.save_many(books)
        found = repo.get_many([books[0].id, books[2].id, uuid4()])
        assert set(found) == {books[0].id, books[2].id}

    def test_save_many_in_chunks_and_iter_all_in_id_order(self, repo):
        books = []
        for i in range(repo.SAVE_CHUNK_SIZE * 2 + 37):
            book = old_style_book()
            book.source_id = f"id{i}"
            books.append(book)
        repo.save_many(books)
        assert repo.count() == len(books)
        streamed = list(repo.iter_all(batch_size=100))
        assert [str(b.id) for b in streamed] == sorted(str(b.id) for b in books)


class TestBuildState:
    def test_key_index_keeps_registration_order_and_ignores_repeats(self, tmp_path):
        index = SqliteDuplicateKeyIndex(tmp_path / "keys.db")
        a, b = uuid4(), uuid4()
        index.add([("title:x|y", a), ("gutenberg:1", a)])
        index.add([("title:x|y", b), ("title:x|y", a)])
        assert index.lookup(["title:x|y", "gutenberg:1", "missing"]) == {"title:x|y": [a, b], "gutenberg:1": [a]}
        assert index.count() == 3

    def test_source_status_and_stats(self, tmp_path):
        state = SqliteCatalogBuildState(tmp_path / "catalog.db")
        assert state.status("doab") is None
        state.mark_started("doab", 2)
        assert state.status("doab") == "running"
        state.mark_done("doab", {"read": 5}, {"license": "CC0"})
        [record] = state.all_sources()
        assert record["status"] == "done" and record["stats"] == {"read": 5}
