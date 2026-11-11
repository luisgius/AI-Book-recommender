"""
API tests for the optional re-ranking stage: the rerank_score field of
POST /search results and the reranker block of GET /health.

A real SearchService runs on fake repositories and a fake reranker; no
model is loaded.
"""

from dataclasses import replace
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.api.v1 import schemas as api
from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService
from app.infrastructure.config.settings import LLMSettings
from app.main import app


def make_book(title: str) -> Book:
    return Book(id=uuid4(), title=title, authors=["Author"], language="en",
                categories=["Computers"], published_date=datetime(2020, 1, 1))


BOOKS = [make_book("Python"), make_book("Java"), make_book("Cooking")]


class Repo:
    def __init__(self, source: str):
        self._source = source

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [SearchResult(book=b, final_score=1.0 / (i + 1), rank=i + 1, source=self._source,
                             **{field: 1.0 / (i + 1)}) for i, b in enumerate(BOOKS[:max_results])]

    def is_ready(self) -> bool:
        return True


class Embeddings:
    def generate_query_embedding(self, text: str) -> List[float]:
        return [0.1, 0.2]

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return [0.1, 0.2]

    def is_ready(self) -> bool:
        return True


class ReversingReranker:
    """Puts the fused last result first; score = fused rank."""

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        ordered = sorted(results, key=lambda r: r.rank, reverse=True)
        return [replace(r, rank=i, rerank_score=float(r.rank)) for i, r in enumerate(ordered, start=1)]

    def get_model_name(self) -> str:
        return "fake-cross-encoder"

    def is_ready(self) -> bool:
        return True


@pytest.fixture
def client_with(monkeypatch):
    """Return a factory that builds a TestClient around a SearchService."""
    def build(reranker=None):
        kwargs = {"reranker": reranker} if reranker is not None else {}
        service = SearchService(Repo("lexical"), Repo("vector"), Embeddings(), **kwargs)
        app.dependency_overrides[deps.get_search_service] = lambda: service
        monkeypatch.setattr(deps, "_llm_settings", LLMSettings(provider="ollama"))
        return TestClient(app)
    yield build
    app.dependency_overrides.clear()


def test_schema_has_an_optional_rerank_score():
    field = api.SearchResult.model_fields["rerank_score"]
    assert field.is_required() is False
    assert field.default is None


def test_results_expose_rerank_and_fusion_scores(client_with):
    client = client_with(ReversingReranker())

    response = client.post("/api/v1/search", json={"text": "python", "max_results": 3})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["book"]["title"] for r in results] == ["Cooking", "Java", "Python"]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert [r["rerank_score"] for r in results] == [3.0, 2.0, 1.0]
    # final_score keeps the RRF fusion score (Cooking was fused last)
    assert results[0]["final_score"] < results[2]["final_score"]


def test_rerank_score_is_null_without_reranker(client_with):
    client = client_with()

    results = client.post("/api/v1/search", json={"text": "python"}).json()["results"]

    assert [r["book"]["title"] for r in results] == ["Python", "Java", "Cooking"]
    assert all(r["rerank_score"] is None for r in results)


def test_health_reports_the_reranker(client_with):
    client = client_with(ReversingReranker())

    body = client.get("/api/v1/health").json()

    assert body["reranker"] == {"configured": True, "model": "fake-cross-encoder", "ready": True}
    assert body["overall"] is True


def test_health_without_reranker(client_with):
    client = client_with()

    body = client.get("/api/v1/health").json()

    assert body["reranker"] == {"configured": False, "model": None, "ready": False}
