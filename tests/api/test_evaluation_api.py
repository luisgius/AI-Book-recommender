"""
Tests for POST /evaluate, the retrieval evaluation runner, and
GET /books/{book_id}/similar.
"""

import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.api.v1 import evaluation_endpoints
from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService
from app.evaluation.retrieval_evaluation import RETRIEVAL_MODES, run_retrieval_evaluation
from app.main import app


def make_book(title: str, source_id: str, language: str = "en") -> Book:
    return Book(id=uuid4(), title=title, authors=["Author"], language=language, categories=["Computers"],
                published_date=datetime(2020, 1, 1), source="google_books", source_id=source_id)


class Repo:
    """Returns its books in a fixed order (relevant book first)."""

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
    def __init__(self, known: Optional[set] = None):
        self._known = known

    def generate_embedding(self, text: str) -> List[float]:
        return [0.1, 0.2]

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        if self._known is not None and book_id not in self._known:
            return None
        return [0.1, 0.2]

    def is_ready(self) -> bool:
        return True


class Catalog:
    def __init__(self, books: List[Book]):
        self._by_source = {(b.source, b.source_id): b for b in books}

    def get_by_source_id(self, source: str, source_id: str) -> Optional[Book]:
        return self._by_source.get((source, source_id))


@pytest.fixture
def collection(tmp_path):
    """Two queries; the first ranked book is relevant for q1, nothing is relevant for q2."""
    books = [make_book("Relevant", "r1"), make_book("Other", "o1"), make_book("Third", "t1")]
    queries = tmp_path / "queries.json"
    judgments = tmp_path / "judgments.json"
    queries.write_text(json.dumps([{"query_id": "q1", "text": "python"}, {"query_id": "q2", "text": "cocina"}]))
    judgments.write_text(json.dumps([
        {"query_id": "q1", "judgments": [{"source": "google_books", "source_id": "r1", "relevance": 3},
                                          {"source": "google_books", "source_id": "o1", "relevance": 0}]},
        {"query_id": "q2", "judgments": [{"source": "google_books", "source_id": "o1", "relevance": 0}]},
    ]))
    return books, str(queries), str(judgments)


def run(books, queries, judgments, **kwargs):
    service = SearchService(Repo(books, "lexical"), Repo(books, "vector"), Embeddings())
    return run_retrieval_evaluation(
        search_service=service, lexical_repo=Repo(books, "lexical"), vector_repo=Repo(books, "vector"),
        embeddings_store=Embeddings(), catalog_repo=Catalog(books),
        queries_path=queries, judgments_path=judgments, **kwargs,
    )


class TestRetrievalEvaluation:
    def test_reports_every_requested_mode_with_standard_metrics(self, collection):
        report = run(*collection)

        assert set(report["modes"]) == set(RETRIEVAL_MODES)
        assert report["num_queries"] == 2
        assert report["num_queries_with_relevant"] == 1
        hybrid = report["modes"]["hybrid_rrf"]
        for metric in ("ndcg_at_10", "recall_at_10", "precision_at_5", "mrr", "ild_at_10"):
            assert metric in hybrid

    def test_metrics_match_hand_computed_values(self, collection):
        report = run(*collection, modes=["lexical_only"])
        lexical = report["modes"]["lexical_only"]

        # q1: relevant book at rank 1 -> MRR 1, P@5 1/5, nDCG@10 1; q2 has no relevant book -> 0
        assert lexical["mrr"] == pytest.approx(0.5)
        assert lexical["precision_at_5"] == pytest.approx(0.1)
        assert lexical["ndcg_at_10"] == pytest.approx(0.5)

    def test_unknown_mode_is_rejected(self, collection):
        with pytest.raises(ValueError):
            run(*collection, modes=["hybrid_rrf", "llm_magic"])


@pytest.fixture
def api_client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def override_retrieval_deps(books):
    service = SearchService(Repo(books, "lexical"), Repo(books, "vector"), Embeddings({b.id for b in books}))
    app.dependency_overrides[deps.get_search_service] = lambda: service
    app.dependency_overrides[deps.get_bm25_repository] = lambda: Repo(books, "lexical")
    app.dependency_overrides[deps.get_vector_repository] = lambda: Repo(books, "vector")
    app.dependency_overrides[deps.get_embeddings_store] = lambda: Embeddings()
    app.dependency_overrides[deps.get_catalog_repository] = lambda: Catalog(books)


class TestEvaluateEndpoint:
    def test_passes_options_and_returns_the_report(self, api_client, monkeypatch):
        books = [make_book("A", "a")]
        override_retrieval_deps(books)
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {"num_queries": 44, "num_queries_with_relevant": 40, "latency_ms": 12.0,
                    "modes": {"hybrid_rrf": {"ndcg_at_10": 0.54}}}

        monkeypatch.setattr(evaluation_endpoints, "run_retrieval_evaluation", fake_run)

        response = api_client.post("/api/v1/evaluate", json={"modes": ["hybrid_rrf"], "max_results": 50})

        assert response.status_code == 200
        assert response.json()["modes"]["hybrid_rrf"]["ndcg_at_10"] == 0.54
        assert captured["modes"] == ["hybrid_rrf"] and captured["max_results"] == 50

    def test_unknown_mode_is_a_validation_error(self, api_client):
        override_retrieval_deps([make_book("A", "a")])
        assert api_client.post("/api/v1/evaluate", json={"modes": ["nope"]}).status_code == 422

    def test_missing_evaluation_data_is_503(self, api_client, monkeypatch):
        override_retrieval_deps([make_book("A", "a")])

        def missing(**kwargs):
            raise FileNotFoundError("test_queries.json")

        monkeypatch.setattr(evaluation_endpoints, "run_retrieval_evaluation", missing)
        assert api_client.post("/api/v1/evaluate", json={}).status_code == 503


class TestSimilarBooksEndpoint:
    def test_returns_other_books_most_similar_first(self, api_client):
        books = [make_book("Source", "s"), make_book("Neighbour", "n"), make_book("Far", "f")]
        override_retrieval_deps(books)

        response = api_client.get(f"/api/v1/books/{books[0].id}/similar", params={"max_results": 5})

        body = response.json()
        assert response.status_code == 200
        assert body["book_id"] == str(books[0].id)
        titles = [r["book"]["title"] for r in body["results"]]
        assert "Source" not in titles and titles[0] == "Neighbour"

    def test_language_filter_is_applied(self, api_client):
        books = [make_book("Source", "s"), make_book("English", "e"), make_book("Spanish", "sp", language="es")]
        override_retrieval_deps(books)

        response = api_client.get(f"/api/v1/books/{books[0].id}/similar", params={"language": "es"})

        assert [r["book"]["title"] for r in response.json()["results"]] == ["Spanish"]

    def test_unknown_book_is_404(self, api_client):
        override_retrieval_deps([make_book("A", "a"), make_book("B", "b")])
        assert api_client.get(f"/api/v1/books/{uuid4()}/similar").status_code == 404
