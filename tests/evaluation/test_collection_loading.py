"""Tests for loading the test collection: query language labels and judgments with extra fields."""

import json
from typing import cast
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.domain.ports import BookCatalogRepository
from app.evaluation.evaluation_job import (
    DEFAULT_QUERIES_PATH,
    load_relevance_judgments,
    load_test_queries,
    resolve_judgments_to_uuids,
)
from app.evaluation.types import QUERY_LANGUAGES, TestQuery


def test_load_test_queries_reads_the_language(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([
        {"query_id": "q1", "text": "novela historica", "category": "fiction", "language": "es"},
        {"query_id": "q2", "text": "historical novels"},
    ]), encoding="utf-8")
    first, second = load_test_queries(str(path))
    assert first == TestQuery("q1", "novela historica", category="fiction", language="es")
    assert second.language is None and second.category is None


def test_test_query_rejects_unknown_languages():
    with pytest.raises(ValueError):
        TestQuery("q1", "roman historique", language="fr")


def test_every_bundled_query_has_a_language_label():
    queries = load_test_queries(DEFAULT_QUERIES_PATH)
    assert queries
    assert all(q.language in QUERY_LANGUAGES for q in queries)


class Catalog:
    """Catalog double: resolves (source, source_id) to a book."""

    def __init__(self, books):
        self.books = {(b.source, b.source_id): b for b in books}

    def get_by_source_id(self, source, source_id):
        return self.books.get((source, source_id))


def test_judgments_with_provenance_fields_load_and_resolve(tmp_path):
    books = [Book(id=uuid4(), title=f"B{i}", authors=["A"], source="google_books", source_id=f"s{i}")
             for i in range(2)]
    path = tmp_path / "judgments.json"
    path.write_text(json.dumps([
        {
            "query_id": "q1",
            "notes": "extra per-query field",
            "judgments": [
                {"source": "google_books", "source_id": "s0", "relevance": 3,
                 "judge": "llm", "model": "some-model", "prompt_version": "v1", "date": "2026-11-11"},
                {"source": "google_books", "source_id": "s1", "relevance": 0, "judge": "human"},
                {"source": "google_books", "source_id": "missing", "relevance": 2},
            ],
        }
    ]), encoding="utf-8")

    raw = load_relevance_judgments(str(path))
    resolved = resolve_judgments_to_uuids(raw, cast(BookCatalogRepository, Catalog(books)))

    assert raw["q1"][0]["model"] == "some-model"
    assert resolved["q1"].judgments == {books[0].id: 3, books[1].id: 0}
