"""
API tests for POST /search and GET /health with the LLM path wired.

A real SearchService runs on fake repositories and a stub LLM client, so the
request -> domain -> response conversion is exercised end to end without
indexes, models or a running LLM.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.domain.entities import Book, Citation, Explanation, SearchResult
from app.domain.services import SearchService
from app.domain.value_objects import QueryIntent, SearchFilters
from app.infrastructure.config.settings import LLMSettings
from app.main import app


def make_book(title: str, language: str = "en", category: str = "Computers") -> Book:
    return Book(id=uuid4(), title=title, authors=["Author"], language=language,
                categories=[category], published_date=datetime(2020, 1, 1),
                description=f"A book about {title.lower()}")


class Repo:
    def __init__(self, books: List[Book], source: str):
        self._books, self._source = books, source

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        matching = [b for b in self._books if b.matches_filters(filters)][:max_results]
        field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [SearchResult(book=b, final_score=1.0 / (i + 1), rank=i + 1, source=self._source,
                             **{field: 1.0 / (i + 1)}) for i, b in enumerate(matching)]

    def is_ready(self) -> bool:
        return True


class Embeddings:
    def generate_embedding(self, text: str) -> List[float]:
        return [0.1, 0.2]

    generate_query_embedding = generate_embedding  # same vector for queries

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return [0.1, 0.2]

    def get_embeddings_batch(self, book_ids):
        return {i: v for i in book_ids if (v := self.get_embedding(i)) is not None}

    def is_ready(self) -> bool:
        return True


class StubLLM:
    def __init__(self, extracted: SearchFilters = SearchFilters()):
        self.extracted = extracted

    def extract_query_intent(self, query_text: str) -> QueryIntent:
        return QueryIntent(intent_type="recommendation", original_query=query_text,
                           reformulated_query="python programming", extracted_filters=self.extracted,
                           confidence=0.8, reasoning="asks for similar books",
                           query_variations=["python programming", "python"])

    def generate_grounded_explanation(self, query_text: str, book: Book) -> Explanation:
        return Explanation(
            book_id=book.id, query_text=query_text, text=f"{book.title} matches.", model="stub-model",
            citations=[Citation(book_id=book.id, chunk_id="title", snippet=book.title, relevance_score=0.9)],
        )

    def get_model_name(self) -> str:
        return "stub-model"


BOOKS = [make_book("Python"), make_book("Python avanzado", language="es"), make_book("Cooking")]


@pytest.fixture
def client_with(monkeypatch):
    """Return a factory that builds a TestClient around a SearchService."""
    def build(llm=None):
        service = SearchService(Repo(BOOKS, "lexical"), Repo(BOOKS, "vector"), Embeddings(),
                                llm_client=llm, max_explanations=2)
        app.dependency_overrides[deps.get_search_service] = lambda: service
        monkeypatch.setattr(deps, "_llm_settings", LLMSettings(provider="ollama"))
        return TestClient(app)
    yield build
    app.dependency_overrides.clear()


def test_explanations_are_returned_as_text_with_citations(client_with):
    client = client_with(StubLLM())

    response = client.post("/api/v1/search", json={"text": "python", "max_results": 3, "use_explanations": True})

    assert response.status_code == 200
    results = response.json()["results"]
    explanation = results[0]["explanation"]
    assert explanation["text"].endswith("matches.")
    assert explanation["grounded"] is True
    assert explanation["model"] == "stub-model"
    assert explanation["citations"][0]["field"] == "title"
    assert results[2]["explanation"] is None  # max_explanations=2


def test_use_understanding_returns_the_query_intent(client_with):
    client = client_with(StubLLM(extracted=SearchFilters(language="es")))

    response = client.post("/api/v1/search", json={"text": "libros como Python", "use_understanding": True})

    body = response.json()
    assert response.status_code == 200
    assert body["query_intent"]["intent_type"] == "recommendation"
    assert body["query_intent"]["extracted_filters"]["language"] == "es"
    assert body["query_intent"]["query_variations"] == ["python programming", "python"]
    assert [r["book"]["title"] for r in body["results"]] == ["Python avanzado"]


def test_explicit_request_filters_win_over_extracted_ones(client_with):
    client = client_with(StubLLM(extracted=SearchFilters(language="es")))

    response = client.post("/api/v1/search", json={
        "text": "libros como Python", "use_understanding": True, "filters": {"language": "en"},
    })

    titles = [r["book"]["title"] for r in response.json()["results"]]
    assert "Python avanzado" not in titles and "Python" in titles


def test_understanding_without_llm_degrades_to_default_intent(client_with):
    client = client_with(llm=None)

    response = client.post("/api/v1/search", json={"text": "python", "use_understanding": True})

    body = response.json()
    assert response.status_code == 200
    assert body["query_intent"]["confidence"] == 0.0
    assert body["results"]


def test_plain_search_has_no_query_intent(client_with):
    client = client_with(StubLLM())

    body = client.post("/api/v1/search", json={"text": "python"}).json()

    assert body["query_intent"] is None
    assert all(r["explanation"] is None for r in body["results"])


def test_health_reports_the_llm_separately(client_with):
    client = client_with(StubLLM())

    body = client.get("/api/v1/health").json()

    assert body["llm"] == {"configured": True, "provider": "ollama", "model": "qwen3.8:27b-mlx"}
    assert body["overall"] is True
