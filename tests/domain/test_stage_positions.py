"""
Per-stage positions of a search result (lexical_rank, vector_rank, rerank_rank).

They are filled where the lists are fused and re-ranked (SearchService), so
every adapter pair gets them: the adapters only number their own lists 1..n.
"""

from typing import List

from app.domain.entities import Book, SearchResult
from app.domain.services import SearchService
from app.domain.value_objects import SearchQuery
from tests.domain.test_reranking import FakeEmbeddings, FakeReranker, StubLLM, make_books


class ListRepo:
    """A repository that returns a fixed list of books, numbered 1..n."""

    def __init__(self, books: List[Book], source: str, ready: bool = True):
        self._books = books
        self._source = source
        self._ready = ready

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        if not self._ready:
            raise RuntimeError("index not loaded")
        field = "lexical_score" if self._source == "lexical" else "vector_score"
        return [
            SearchResult(book=b, final_score=1.0 / i, rank=i, source=self._source, **{field: 1.0 / i})
            for i, b in enumerate(self._books[:max_results], start=1)
        ]

    def is_ready(self) -> bool:
        return self._ready


def service_with(lexical: List[Book], vector: List[Book], all_books: List[Book], **kwargs) -> SearchService:
    return SearchService(
        lexical_search=ListRepo(lexical, "lexical"),
        vector_search=ListRepo(vector, "vector", ready=kwargs.pop("vector_ready", True)),
        embeddings_store=FakeEmbeddings(all_books),
        **kwargs,
    )


def by_title(results: List[SearchResult]) -> dict:
    return {r.book.title: r for r in results}


def test_fusion_keeps_the_position_of_each_book_in_each_list():
    books = make_books(6)
    a, b, c, d = books[:4]
    service = service_with(lexical=[a, b, c], vector=[c, d, a], all_books=books)

    results = by_title(service.search(SearchQuery(text="q", max_results=10)))

    assert (results[a.title].lexical_rank, results[a.title].vector_rank) == (1, 3)
    assert (results[b.title].lexical_rank, results[b.title].vector_rank) == (2, None)
    assert (results[c.title].lexical_rank, results[c.title].vector_rank) == (3, 1)
    assert (results[d.title].lexical_rank, results[d.title].vector_rank) == (None, 2)
    assert all(r.rerank_rank is None for r in results.values())  # no reranker configured


def test_the_reranker_position_is_set_on_the_head_only():
    books = make_books(10)
    service = service_with(lexical=books, vector=books, all_books=books,
                           reranker=FakeReranker(), rerank_depth=4)

    results = service.search(SearchQuery(text="q", max_results=10))

    assert [r.rerank_rank for r in results[:4]] == [1, 2, 3, 4]
    assert all(r.rerank_rank is None for r in results[4:])
    # The fake reverses the head: the book first after re-ranking was fourth in both lists
    assert (results[0].lexical_rank, results[0].vector_rank) == (4, 4)


def test_a_failed_reranker_leaves_no_rerank_position():
    books = make_books(5)
    service = service_with(lexical=books, vector=books, all_books=books, reranker=FakeReranker(fail=True))

    results = service.search(SearchQuery(text="q", max_results=5))

    assert all(r.rerank_rank is None for r in results)


def test_lexical_only_fallback_has_keyword_positions_and_no_semantic_ones():
    books = make_books(5)
    service = service_with(lexical=books[::-1], vector=books, all_books=books, vector_ready=False)

    response = service.search_with_fallback(SearchQuery(text="q", max_results=5))

    assert response.search_mode == "lexical_only"
    assert [r.lexical_rank for r in response.results] == [1, 2, 3, 4, 5]
    assert all(r.vector_rank is None for r in response.results)


def test_multi_query_fusion_keeps_the_best_position_among_the_variations():
    books = make_books(8)
    service = service_with(lexical=books, vector=books[::-1], all_books=books, llm_client=StubLLM())

    response, _intent = service.search_with_understanding("q", max_results=8)

    for result in response.results:
        assert result.lexical_rank is not None and result.vector_rank is not None
        assert result.lexical_rank >= 1 and result.vector_rank >= 1
    first = by_title(response.results)[books[0].title]
    assert (first.lexical_rank, first.vector_rank) == (1, 8)


def test_the_api_exposes_the_positions():
    from app.api.v1.converters import domain_search_result_to_api

    book = make_books(1)[0]
    api = domain_search_result_to_api(SearchResult(
        book=book, final_score=0.1, rank=1, source="hybrid", lexical_rank=3, vector_rank=None, rerank_rank=1,
    ))
    assert (api.lexical_rank, api.vector_rank, api.rerank_rank) == (3, None, 1)
