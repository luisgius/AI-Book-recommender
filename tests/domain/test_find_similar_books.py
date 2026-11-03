"""
Tests for find_similar_books (Item-to-Item recommendation) in SearchService.

These tests verify:
1. Similar books are found using vector search
2. Source book is excluded from results
3. Filters are applied correctly
4. MMR diversification works with similar books
5. Error handling for missing embeddings
"""

import pytest
from uuid import uuid4, UUID
from typing import List, Optional, Dict

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchFilters
from app.domain.services import SearchService


# =============================================================================
# Fake implementations for testing
# =============================================================================


class FakeLexicalSearchRepository:
    """Fake lexical search (not used in find_similar_books)."""

    def search(self, query_text: str, max_results: int = 10, filters=None):
        return []

    def build_index(self, books):
        pass

    def add_to_index(self, book):
        pass

    def save_index(self, path):
        pass

    def load_index(self, path):
        pass


class FakeVectorSearchRepository:
    """Fake vector search that returns predefined results."""

    def __init__(self, results: List[SearchResult] = None):
        self._results = results or []

    def search(
        self,
        query_embedding: List[float],
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        return self._results[:max_results]


class FakeEmbeddingsStore:
    """Fake embeddings store with controllable embeddings."""

    def __init__(self, embeddings: Dict[UUID, List[float]] = None):
        self._embeddings = embeddings or {}

    def generate_embedding(self, text: str) -> List[float]:
        return [0.1] * 384

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self._embeddings.get(book_id)

    def get_dimension(self) -> int:
        return 384


# =============================================================================
# Helper functions
# =============================================================================


def create_book(title: str, book_id: UUID = None, language: str = None) -> Book:
    """Create a test book with minimal required fields."""
    return Book(
        id=book_id or uuid4(),
        title=title,
        authors=["Test Author"],
        source="test",
        source_id=f"test-{title.lower().replace(' ', '-')}",
        language=language,
    )


def create_search_result(book: Book, score: float, rank: int) -> SearchResult:
    """Create a search result for testing."""
    return SearchResult(
        book=book,
        final_score=score,
        rank=rank,
        source="vector",
        vector_score=score,
    )


# =============================================================================
# Tests
# =============================================================================


class TestFindSimilarBooks:
    """Tests for find_similar_books functionality."""

    def test_finds_similar_books_using_vector_search(self):
        """Should find similar books using the source book's embedding."""
        # Create source book and similar books
        source_book = create_book("Python Programming")
        similar1 = create_book("Python Cookbook")
        similar2 = create_book("Learning Python")

        # Setup embeddings
        embeddings = {
            source_book.id: [1.0, 0.0, 0.0],
            similar1.id: [0.9, 0.1, 0.0],
            similar2.id: [0.8, 0.2, 0.0],
        }

        # Setup vector search results (includes source book which should be filtered)
        vector_results = [
            create_search_result(source_book, score=1.0, rank=1),  # Will be excluded
            create_search_result(similar1, score=0.9, rank=2),
            create_search_result(similar2, score=0.8, rank=3),
        ]

        # Create service
        embeddings_store = FakeEmbeddingsStore(embeddings)
        vector_repo = FakeVectorSearchRepository(vector_results)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        # Find similar books
        results = service.find_similar_books(
            book_id=source_book.id,
            max_results=5,
        )

        # Should have 2 results (source book excluded)
        assert len(results) == 2
        assert results[0].book.id == similar1.id
        assert results[1].book.id == similar2.id

    def test_excludes_source_book_from_results(self):
        """Source book should never appear in similar books results."""
        source_book = create_book("Source Book")
        other_book = create_book("Other Book")

        embeddings = {
            source_book.id: [1.0, 0.0],
            other_book.id: [0.5, 0.5],
        }

        # Vector search returns source book as most similar
        vector_results = [
            create_search_result(source_book, score=1.0, rank=1),
            create_search_result(other_book, score=0.5, rank=2),
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        vector_repo = FakeVectorSearchRepository(vector_results)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        results = service.find_similar_books(book_id=source_book.id, max_results=5)

        # Source book should NOT be in results
        result_ids = [r.book.id for r in results]
        assert source_book.id not in result_ids
        assert other_book.id in result_ids

    def test_raises_error_for_missing_embedding(self):
        """Should raise ValueError if source book has no embedding."""
        source_book = create_book("Book Without Embedding")

        # Empty embeddings store
        embeddings_store = FakeEmbeddingsStore({})
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=FakeVectorSearchRepository([]),
            embeddings_store=embeddings_store,
        )

        with pytest.raises(ValueError, match="No embedding found"):
            service.find_similar_books(book_id=source_book.id, max_results=5)

    def test_applies_filters_to_similar_books(self):
        """Filters should be applied to similar books results."""
        source_book = create_book("Source", language="en")
        book_en = create_book("English Book", language="en")
        book_es = create_book("Spanish Book", language="es")

        embeddings = {
            source_book.id: [1.0, 0.0],
            book_en.id: [0.9, 0.1],
            book_es.id: [0.8, 0.2],
        }

        vector_results = [
            create_search_result(book_en, score=0.9, rank=1),
            create_search_result(book_es, score=0.8, rank=2),
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        vector_repo = FakeVectorSearchRepository(vector_results)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        # Filter to only English books
        results = service.find_similar_books(
            book_id=source_book.id,
            max_results=5,
            filters=SearchFilters(language="en"),
        )

        # Only English book should be returned
        assert len(results) == 1
        assert results[0].book.language == "en"

    def test_reassigns_ranks_starting_from_one(self):
        """Results should have ranks starting from 1."""
        source_book = create_book("Source")
        book1 = create_book("Book 1")
        book2 = create_book("Book 2")

        embeddings = {
            source_book.id: [1.0, 0.0],
            book1.id: [0.9, 0.1],
            book2.id: [0.8, 0.2],
        }

        vector_results = [
            create_search_result(book1, score=0.9, rank=5),  # Original rank 5
            create_search_result(book2, score=0.8, rank=10),  # Original rank 10
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        vector_repo = FakeVectorSearchRepository(vector_results)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        results = service.find_similar_books(book_id=source_book.id, max_results=5)

        # Ranks should be reassigned to 1, 2
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_respects_max_results(self):
        """Should return at most max_results books."""
        source_book = create_book("Source")
        books = [create_book(f"Book {i}") for i in range(10)]

        embeddings = {source_book.id: [1.0, 0.0]}
        for book in books:
            embeddings[book.id] = [0.5, 0.5]

        vector_results = [
            create_search_result(book, score=0.5, rank=i + 1)
            for i, book in enumerate(books)
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        vector_repo = FakeVectorSearchRepository(vector_results)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        results = service.find_similar_books(book_id=source_book.id, max_results=3)

        assert len(results) == 3

    def test_returns_empty_list_when_no_similar_books(self):
        """Should return empty list if no similar books found."""
        source_book = create_book("Lonely Book")

        embeddings = {source_book.id: [1.0, 0.0]}

        # Only source book in results (will be filtered out)
        vector_results = [
            create_search_result(source_book, score=1.0, rank=1),
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        vector_repo = FakeVectorSearchRepository(vector_results)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        results = service.find_similar_books(book_id=source_book.id, max_results=5)

        assert len(results) == 0

    def test_can_apply_mmr_diversification(self):
        """Should support MMR diversification for similar books."""
        source_book = create_book("Source")
        book1 = create_book("Similar 1")
        book2 = create_book("Similar 2")
        book3 = create_book("Different Topic")

        # book1 and book2 are similar, book3 is different
        embeddings = {
            source_book.id: [1.0, 0.0, 0.0],
            book1.id: [0.95, 0.05, 0.0],
            book2.id: [0.9, 0.1, 0.0],
            book3.id: [0.0, 0.0, 1.0],
        }

        vector_results = [
            create_search_result(book1, score=0.95, rank=1),
            create_search_result(book2, score=0.9, rank=2),
            create_search_result(book3, score=0.7, rank=3),
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        vector_repo = FakeVectorSearchRepository(vector_results)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        results = service.find_similar_books(
            book_id=source_book.id,
            max_results=3,
            use_diversification=True,
            diversity_lambda=0.5,
        )

        # Should return results (diversification applied)
        assert len(results) == 3
        # All results should have valid ranks
        assert all(r.rank > 0 for r in results)
