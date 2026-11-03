"""
Tests for graceful degradation (RNF-08) and health checks (RNF-06) in SearchService.

These tests verify:
1. Search falls back to lexical-only when vector search is unavailable
2. SearchResponse includes degradation metadata
3. Health status correctly reports component availability
4. Latency tracking works correctly
"""

import pytest
from uuid import uuid4, UUID
from typing import List, Optional, Dict

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchQuery, SearchFilters, SearchResponse
from app.domain.services import SearchService


# =============================================================================
# Fake implementations for testing
# =============================================================================


class FakeLexicalSearchRepository:
    """Fake lexical search with controllable is_ready status."""

    def __init__(self, results: List[SearchResult] = None, ready: bool = True):
        self._results = results or []
        self._ready = ready

    def search(self, query_text: str, max_results: int = 10, filters=None):
        if not self._ready:
            raise RuntimeError("Lexical search not ready")
        return self._results[:max_results]

    def build_index(self, books):
        pass

    def add_to_index(self, book):
        pass

    def save_index(self, path):
        pass

    def load_index(self, path):
        pass

    def is_ready(self) -> bool:
        return self._ready


class FakeVectorSearchRepository:
    """Fake vector search with controllable is_ready status."""

    def __init__(self, results: List[SearchResult] = None, ready: bool = True):
        self._results = results or []
        self._ready = ready

    def search(self, query_embedding, max_results: int = 10, filters=None):
        if not self._ready:
            raise RuntimeError("Vector search not ready")
        return self._results[:max_results]

    def is_ready(self) -> bool:
        return self._ready


class FakeEmbeddingsStore:
    """Fake embeddings store with controllable is_ready status."""

    def __init__(self, embeddings: Dict[UUID, List[float]] = None, ready: bool = True):
        self._embeddings = embeddings or {}
        self._ready = ready

    def generate_embedding(self, text: str) -> List[float]:
        if not self._ready:
            raise RuntimeError("Embeddings store not ready")
        return [0.1] * 384

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self._embeddings.get(book_id)

    def is_ready(self) -> bool:
        return self._ready

    def get_dimension(self) -> int:
        return 384


# =============================================================================
# Helper functions
# =============================================================================


def create_book(title: str, book_id: UUID = None) -> Book:
    """Create a test book with minimal required fields."""
    return Book(
        id=book_id or uuid4(),
        title=title,
        authors=["Test Author"],
        source="test",
        source_id=f"test-{title.lower().replace(' ', '-')}",
    )


def create_search_result(book: Book, score: float, rank: int, source: str = "lexical") -> SearchResult:
    """Create a search result for testing."""
    return SearchResult(
        book=book,
        final_score=score,
        rank=rank,
        source=source,
        lexical_score=score if source == "lexical" else None,
        vector_score=score if source == "vector" else None,
    )


# =============================================================================
# Tests: Health Status
# =============================================================================


class TestHealthStatus:
    """Tests for get_health_status functionality (RNF-06)."""

    def test_health_status_all_ready(self):
        """Health status should show all components ready when available."""
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(ready=True),
            vector_search=FakeVectorSearchRepository(ready=True),
            embeddings_store=FakeEmbeddingsStore(ready=True),
        )

        status = service.get_health_status()

        assert status["lexical_search"] is True
        assert status["vector_search"] is True
        assert status["embeddings_store"] is True
        assert status["overall"] is True

    def test_health_status_vector_not_ready(self):
        """Health status should reflect when vector search is unavailable."""
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(ready=True),
            vector_search=FakeVectorSearchRepository(ready=False),
            embeddings_store=FakeEmbeddingsStore(ready=True),
        )

        status = service.get_health_status()

        assert status["lexical_search"] is True
        assert status["vector_search"] is False
        assert status["embeddings_store"] is True
        assert status["overall"] is False

    def test_health_status_embeddings_not_ready(self):
        """Health status should reflect when embeddings store is unavailable."""
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(ready=True),
            vector_search=FakeVectorSearchRepository(ready=True),
            embeddings_store=FakeEmbeddingsStore(ready=False),
        )

        status = service.get_health_status()

        assert status["lexical_search"] is True
        assert status["vector_search"] is True
        assert status["embeddings_store"] is False
        assert status["overall"] is False

    def test_health_status_all_not_ready(self):
        """Health status should reflect when all components are unavailable."""
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(ready=False),
            vector_search=FakeVectorSearchRepository(ready=False),
            embeddings_store=FakeEmbeddingsStore(ready=False),
        )

        status = service.get_health_status()

        assert status["lexical_search"] is False
        assert status["vector_search"] is False
        assert status["embeddings_store"] is False
        assert status["overall"] is False


# =============================================================================
# Tests: Graceful Degradation
# =============================================================================


class TestGracefulDegradation:
    """Tests for search_with_fallback graceful degradation (RNF-08)."""

    def test_hybrid_search_when_all_ready(self):
        """Should use hybrid search when all components are available."""
        book = create_book("Test Book")
        lexical_results = [create_search_result(book, 0.8, 1, "lexical")]
        vector_results = [create_search_result(book, 0.9, 1, "vector")]

        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(lexical_results, ready=True),
            vector_search=FakeVectorSearchRepository(vector_results, ready=True),
            embeddings_store=FakeEmbeddingsStore(ready=True),
        )

        query = SearchQuery(text="test query")
        response = service.search_with_fallback(query)

        assert response.degraded is False
        assert response.degradation_reason is None
        assert response.search_mode == "hybrid"
        assert response.latency_ms is not None
        assert response.latency_ms >= 0

    def test_fallback_to_lexical_when_vector_not_ready(self):
        """Should fall back to lexical search when vector is unavailable."""
        book = create_book("Test Book")
        lexical_results = [create_search_result(book, 0.8, 1, "lexical")]

        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(lexical_results, ready=True),
            vector_search=FakeVectorSearchRepository([], ready=False),
            embeddings_store=FakeEmbeddingsStore(ready=True),
        )

        query = SearchQuery(text="test query")
        response = service.search_with_fallback(query)

        assert response.degraded is True
        assert "unavailable" in response.degradation_reason.lower()
        assert response.search_mode == "lexical_only"
        assert len(response.results) == 1

    def test_fallback_to_lexical_when_embeddings_not_ready(self):
        """Should fall back to lexical search when embeddings store is unavailable."""
        book = create_book("Test Book")
        lexical_results = [create_search_result(book, 0.8, 1, "lexical")]

        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(lexical_results, ready=True),
            vector_search=FakeVectorSearchRepository([], ready=True),
            embeddings_store=FakeEmbeddingsStore(ready=False),
        )

        query = SearchQuery(text="test query")
        response = service.search_with_fallback(query)

        assert response.degraded is True
        assert response.search_mode == "lexical_only"

    def test_raises_when_both_unavailable(self):
        """Should raise error when both lexical and vector are unavailable."""
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository([], ready=False),
            vector_search=FakeVectorSearchRepository([], ready=False),
            embeddings_store=FakeEmbeddingsStore(ready=False),
        )

        query = SearchQuery(text="test query")

        with pytest.raises(RuntimeError, match="unavailable"):
            service.search_with_fallback(query)

    def test_response_includes_latency(self):
        """SearchResponse should include latency measurement."""
        book = create_book("Test Book")
        lexical_results = [create_search_result(book, 0.8, 1, "lexical")]

        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(lexical_results, ready=True),
            vector_search=FakeVectorSearchRepository([], ready=False),
            embeddings_store=FakeEmbeddingsStore(ready=True),
        )

        query = SearchQuery(text="test query")
        response = service.search_with_fallback(query)

        assert response.latency_ms is not None
        assert isinstance(response.latency_ms, float)
        assert response.latency_ms >= 0


# =============================================================================
# Tests: SearchResponse Value Object
# =============================================================================


class TestSearchResponse:
    """Tests for SearchResponse value object validation."""

    def test_valid_response(self):
        """Should create a valid SearchResponse."""
        response = SearchResponse(
            results=[],
            degraded=False,
            search_mode="hybrid",
        )
        assert response.degraded is False

    def test_degraded_requires_reason(self):
        """Degraded response must include degradation_reason."""
        with pytest.raises(ValueError, match="degradation_reason"):
            SearchResponse(
                results=[],
                degraded=True,
                degradation_reason=None,  # Missing required reason
            )

    def test_degraded_with_reason_is_valid(self):
        """Degraded response with reason should be valid."""
        response = SearchResponse(
            results=[],
            degraded=True,
            degradation_reason="Vector index unavailable",
            search_mode="lexical_only",
        )
        assert response.degraded is True
        assert response.degradation_reason == "Vector index unavailable"

    def test_invalid_search_mode_rejected(self):
        """Should reject invalid search modes."""
        with pytest.raises(ValueError, match="search_mode"):
            SearchResponse(
                results=[],
                search_mode="invalid_mode",
            )

    def test_valid_search_modes(self):
        """Should accept all valid search modes."""
        for mode in ["hybrid", "lexical_only", "vector_only"]:
            response = SearchResponse(results=[], search_mode=mode)
            assert response.search_mode == mode
