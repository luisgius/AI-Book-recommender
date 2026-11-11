"""
Catalog facts the API needs at startup, without reading every book (V13).

BookCatalogRepository.get_categories: one aggregate query in SQLite and
Postgres (cached there), a default for other adapters, and the LLM client
wiring that uses it instead of get_all().
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.api.v1 import dependencies as deps
from app.domain.entities import Book
from app.domain.ports import BookCatalogRepository
from app.infrastructure.config.settings import LLMSettings
from app.infrastructure.db.postgres_book_catalog_repository import PostgresBookCatalogRepository
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository


def book(title, categories):
    return Book(id=uuid4(), title=title, authors=["A"], categories=categories, source="test", source_id=title,
                created_at=datetime(2025, 1, 1, tzinfo=UTC), updated_at=datetime(2025, 1, 1, tzinfo=UTC))


BOOKS = [book("a", ["Fiction", "Classic"]), book("b", ["Fiction"]), book("c", []), book("d", ["Ñandú", "Art"])]


def test_sqlite_categories_come_from_one_aggregate_query(tmp_path):
    repo = SqliteBookCatalogRepository(tmp_path / "catalog.db")
    repo.save_many(BOOKS)

    assert repo.get_categories() == ["Art", "Classic", "Fiction", "Ñandú"]
    assert repo.get_categories() == sorted({c for b in repo.get_all() for c in b.categories})
    assert SqliteBookCatalogRepository(tmp_path / "empty.db").get_categories() == []


class RecordingDatabase:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def query(self, sql, params=None, *, settings=None):
        self.calls.append(sql)
        return self.rows

    def execute(self, sql, params=None, *, settings=None):
        self.calls.append(sql)
        return 1


def test_postgres_categories_are_one_statement_cached_until_a_write():
    db = RecordingDatabase([{"category": "Fiction"}, {"category": "Art"}, {"category": None}])
    repo = PostgresBookCatalogRepository(db, "cat")

    assert repo.get_categories() == ["Art", "Fiction"]
    assert repo.get_categories() == ["Art", "Fiction"]
    assert len(db.calls) == 1 and "SELECT DISTINCT unnest(categories)" in db.calls[0]

    repo.delete(uuid4())
    repo.get_categories()
    assert len(db.calls) == 3  # the delete, then a fresh read


def test_port_default_walks_the_catalog():
    class OnlyIter(BookCatalogRepository):
        def iter_all(self, batch_size=1000):
            return iter(BOOKS)

    assert OnlyIter().get_categories() == ["Art", "Classic", "Fiction", "Ñandú"]


def test_llm_client_gets_the_categories_without_reading_every_book(monkeypatch):
    class Catalog:
        def get_categories(self):
            return ["Art", "Fiction"]

        def get_all(self, limit=None):
            raise AssertionError("the whole catalog must not be read")

    built = {}

    class FakeClient:
        def __init__(self, **kwargs):
            built.update(kwargs)

    deps.reset_dependencies()
    monkeypatch.setattr(deps, "_llm_settings", LLMSettings(provider="ollama"))
    monkeypatch.setattr(deps, "_catalog_repository", Catalog())
    monkeypatch.setattr("app.infrastructure.llm.langchain_llm_client.LangChainLLMClient", FakeClient)
    try:
        assert isinstance(deps.get_llm_client(), FakeClient)
        assert built["known_categories"] == ["Art", "Fiction"]
    finally:
        deps.reset_dependencies()
