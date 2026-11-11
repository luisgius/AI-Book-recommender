"""
SearchService when the lexical side returns fewer results than asked, or none (task V13).

Since V13 the BM25 adapter returns only books that contain a query term (the
Postgres BM25 always did), so a lexical list can be short or empty. Fusion,
re-ranking, MMR, the lexical-only degraded path, the understanding path and
the API response must all accept that.
"""

from typing import Dict, List, Optional
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dependencies as deps
from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService
from app.domain.value_objects import SearchQuery
from app.infrastructure.config.settings import LLMSettings


def make_book(title: str) -> Book:
    return Book(id=uuid4(), title=title, authors=["Author"], language="en", categories=["Fiction"],
                source="test", source_id=title)


BOOKS = [make_book(f"Book {i}") for i in range(12)]
VECTORS: Dict[UUID, List[float]] = {b.id: [1.0, i / 10.0] for i, b in enumerate(BOOKS)}


class Lexical:
    """Returns the given books (fewer than max_results is allowed), like the fixed BM25 adapter."""

    def __init__(self, books: List[Book]) -> None:
        self._books = books
        self.calls = 0

    def search(self, query_text: str, max_results: int = 10, filters=None) -> List[SearchResult]:
        self.calls += 1
        return [SearchResult(book=b, final_score=5.0 - i, rank=i + 1, source="lexical", lexical_score=5.0 - i)
                for i, b in enumerate(self._books[:max_results])]

    def is_ready(self) -> bool:
        return True


class Vector:
    def __init__(self, books: List[Book], ready: bool = True) -> None:
        self._books, self._ready = books, ready

    def search(self, query_embedding, max_results: int = 10, filters=None) -> List[SearchResult]:
        return [SearchResult(book=b, final_score=0.9 - i / 100, rank=i + 1, source="vector",
                             vector_score=0.9 - i / 100) for i, b in enumerate(self._books[:max_results])]

    def is_ready(self) -> bool:
        return self._ready


class Embeddings:
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls = 0

    def generate_query_embedding(self, text: str) -> List[float]:
        return [1.0, 0.0]

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        self.single_calls += 1
        return VECTORS.get(book_id)

    def get_embeddings_batch(self, book_ids: List[UUID]) -> Dict[UUID, List[float]]:
        self.batch_calls += 1
        return {i: VECTORS[i] for i in book_ids if i in VECTORS}

    def is_ready(self) -> bool:
        return True


class Reranker:
    def __init__(self) -> None:
        self.received: List[List[UUID]] = []

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        self.received.append([r.book.id for r in results])
        reordered = list(reversed(results))
        for i, r in enumerate(reordered):
            r.rerank_score = float(len(reordered) - i)
        return reordered

    def get_model_name(self) -> str:
        return "fake-reranker"

    def is_ready(self) -> bool:
        return True


def service(lexical_books, vector_books=BOOKS, *, vector_ready=True, reranker=None) -> SearchService:
    return SearchService(Lexical(lexical_books), Vector(vector_books, ready=vector_ready), Embeddings(),
                         reranker=reranker, rerank_depth=5)


def ids(items) -> List[UUID]:
    """Ids of search results or of books."""
    return [item.book.id if isinstance(item, SearchResult) else item.id for item in items]


# --- hybrid path -------------------------------------------------------------------------


def test_empty_lexical_list_fuses_to_the_vector_ranking():
    response = service([]).search_with_fallback(SearchQuery(text="no lexical match", max_results=5))

    assert not response.degraded
    assert ids(response.results) == ids(BOOKS[:5])
    assert [r.rank for r in response.results] == [1, 2, 3, 4, 5]
    assert all(r.lexical_score is None for r in response.results)
    assert response.metadata.candidates_lexical == 0


def test_short_lexical_list_still_boosts_the_books_it_contains():
    response = service([BOOKS[7]]).search_with_fallback(SearchQuery(text="one match", max_results=5))

    assert response.results[0].book.id == BOOKS[7].id  # in both lists
    assert response.results[0].lexical_score == 5.0
    assert len(response.results) == 5
    assert response.metadata.candidates_lexical == 1


def test_empty_lexical_list_with_reranker_and_mmr():
    reranker = Reranker()
    svc = service([], reranker=reranker)

    response = svc.search_with_fallback(SearchQuery(text="q", max_results=4, use_diversification=True))

    assert not response.degraded
    assert len(response.results) == 4
    assert reranker.received and len(reranker.received[0]) == 5  # rerank_depth
    assert all(r.rerank_score is not None for r in response.results)


def test_nothing_from_either_side_gives_an_empty_response():
    response = service([], vector_books=[]).search_with_fallback(
        SearchQuery(text="q", max_results=10, use_diversification=True))

    assert response.results == [] and not response.degraded


# --- lexical-only degraded path ----------------------------------------------------------


@pytest.mark.parametrize("diversify", [False, True])
def test_degraded_path_with_no_lexical_result_returns_an_empty_list(diversify):
    reranker = Reranker()
    svc = service([], vector_ready=False, reranker=reranker)

    response = svc.search_with_fallback(SearchQuery(text="q", max_results=10, use_diversification=diversify))

    assert response.degraded and response.search_mode == "lexical_only"
    assert response.results == []
    assert reranker.received == []  # nothing to re-rank
    assert response.metadata.candidates_lexical == 0


def test_degraded_path_with_fewer_lexical_results_than_asked():
    response = service(BOOKS[:3], vector_ready=False).search_with_fallback(
        SearchQuery(text="q", max_results=10, use_diversification=True))

    assert response.degraded
    assert sorted(ids(response.results), key=str) == sorted(ids(BOOKS[:3]), key=str)
    assert [r.rank for r in response.results] == [1, 2, 3]


def test_understanding_path_without_llm_and_without_lexical_results():
    response, intent = service([]).search_with_understanding("q", max_results=3)

    assert intent.intent_type == "exploratory"
    assert ids(response.results) == ids(BOOKS[:3])


def test_degraded_understanding_path_without_lexical_results():
    response, _ = service([], vector_ready=False).search_with_understanding("q", max_results=3)

    assert response.degraded and response.results == []


# --- MMR uses one batch lookup -----------------------------------------------------------


def test_mmr_reads_every_candidate_embedding_in_one_batch_call():
    svc = service(BOOKS[:6])
    embeddings = svc._embeddings_store

    svc.search_with_fallback(SearchQuery(text="q", max_results=5, use_diversification=True))

    assert embeddings.batch_calls == 1
    assert embeddings.single_calls == 0


def test_find_similar_books_with_mmr_makes_one_single_and_one_batch_lookup():
    svc = service([])
    embeddings = svc._embeddings_store

    results = svc.find_similar_books(BOOKS[0].id, max_results=4, use_diversification=True)

    assert len(results) == 4 and BOOKS[0].id not in ids(results)
    assert (embeddings.single_calls, embeddings.batch_calls) == (1, 1)


# --- API -----------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from app.main import app

    svc = service([], vector_ready=False)
    app.dependency_overrides[deps.get_search_service] = lambda: svc
    monkeypatch.setattr(deps, "_llm_settings", LLMSettings(provider="none"))
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_answers_an_empty_degraded_search_with_200(client):
    response = client.post("/api/v1/search", json={"text": "nothing matches", "include_metadata": True})

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["degraded"] is True
