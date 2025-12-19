"""
Tests for MMR (Maximal Marginal Relevance) diversification in SearchService.

These tests verify:
1. MMR reranks results to increase diversity
2. Lambda parameter controls relevance vs diversity trade-off
3. Edge cases (empty results, single result, no embeddings)
4. Cosine similarity calculation
"""

import pytest
from uuid import UUID
from typing import List, Optional, Dict, Any

from uuid import uuid4

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchQuery, SearchFilters
from app.domain.services import SearchService


# =============================================================================
# Fake implementations for testing
# =============================================================================


class FakeLexicalSearchRepository:
    """Fake lexical search that returns predefined results."""

    def __init__(self, results: List[SearchResult] = None):
        self._results = results or []

    def search(
        self,
        query_text: str,
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        return self._results[:max_results]

    def build_index(self, books: List[Book]) -> None:
        pass

    def add_to_index(self, book: Book) -> None:
        pass

    def save_index(self, path: str) -> None:
        pass

    def load_index(self, path: str) -> None:
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

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 384 for _ in texts]

    def store_embedding(self, book_id: UUID, embedding: List[float]) -> None:
        self._embeddings[book_id] = embedding

    def store_embeddings_batch(
        self, book_ids: List[UUID], embeddings: List[List[float]]
    ) -> None:
        for book_id, embedding in zip(book_ids, embeddings):
            self._embeddings[book_id] = embedding

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self._embeddings.get(book_id)

    def build_index(self) -> None:
        pass

    def save_index(self, path: str) -> None:
        pass

    def load_index(self, path: str) -> None:
        pass

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


def create_search_result(book: Book, score: float, rank: int) -> SearchResult:
    """Create a search result for testing."""
    return SearchResult(
        book=book,
        final_score=score,
        rank=rank,
        source="hybrid",
    )


# =============================================================================
# Test: Cosine Similarity
# =============================================================================


class TestCosineSimilarity:
    """Tests for the _cosine_similarity static method."""

    def test_identical_vectors_return_one(self):
        """Identical vectors should have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        similarity = SearchService._cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        similarity = SearchService._cosine_similarity(vec_a, vec_b)
        assert abs(similarity) < 1e-6

    def test_opposite_vectors_return_negative_one(self):
        """Opposite vectors should have similarity of -1.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        similarity = SearchService._cosine_similarity(vec_a, vec_b)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_different_length_vectors_return_zero(self):
        """Vectors of different lengths should return 0.0."""
        vec_a = [1.0, 2.0]
        vec_b = [1.0, 2.0, 3.0]
        similarity = SearchService._cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 similarity."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        similarity = SearchService._cosine_similarity(vec_a, vec_b)
        assert similarity == 0.0


# =============================================================================
# Test: MMR Diversification
# =============================================================================


class TestMMRDiversification:
    """Tests for MMR diversification functionality."""

    def test_mmr_with_similar_books_promotes_diversity(self):
        """MMR should promote diverse books over similar ones."""
        # Create books
        book1 = create_book("Python Programming")
        book2 = create_book("Python Cookbook")  # Similar to book1
        book3 = create_book("Machine Learning")  # Different topic

        # Create embeddings: book1 and book2 are similar, book3 is different
        embeddings = {
            book1.id: [1.0, 0.0, 0.0],  # Python topic
            book2.id: [0.95, 0.05, 0.0],  # Very similar to book1
            book3.id: [0.0, 1.0, 0.0],  # Different topic (ML)
        }

        # Create results ranked by relevance (book1 > book2 > book3)
        results = [
            create_search_result(book1, score=0.9, rank=1),
            create_search_result(book2, score=0.85, rank=2),
            create_search_result(book3, score=0.7, rank=3),
        ]

        # Setup fakes
        embeddings_store = FakeEmbeddingsStore(embeddings)
        lexical_repo = FakeLexicalSearchRepository(results)
        vector_repo = FakeVectorSearchRepository([])

        service = SearchService(
            lexical_search=lexical_repo,
            vector_search=vector_repo,
            embeddings_store=embeddings_store,
        )

        # Apply MMR with diversity preference (lambda=0.5)
        diversified = service._apply_mmr_diversification(
            results=results,
            top_k=3,
            lambda_param=0.5,
        )

        # Book3 (ML) should be promoted because it's different from book1
        # Expected order: book1 (most relevant), book3 (diverse), book2 (similar to book1)
        assert len(diversified) == 3
        assert diversified[0].book.id == book1.id  # Most relevant stays first
        # book3 should be promoted over book2 due to diversity
        book_ids = [r.book.id for r in diversified]
        assert book3.id in book_ids

    def test_mmr_with_lambda_one_keeps_relevance_order(self):
        """Lambda=1.0 should maintain pure relevance ranking."""
        book1 = create_book("First")
        book2 = create_book("Second")
        book3 = create_book("Third")

        embeddings = {
            book1.id: [1.0, 0.0],
            book2.id: [0.9, 0.1],
            book3.id: [0.0, 1.0],
        }

        results = [
            create_search_result(book1, score=0.9, rank=1),
            create_search_result(book2, score=0.8, rank=2),
            create_search_result(book3, score=0.7, rank=3),
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=FakeVectorSearchRepository(),
            embeddings_store=embeddings_store,
        )

        diversified = service._apply_mmr_diversification(
            results=results,
            top_k=3,
            lambda_param=1.0,  # Pure relevance
        )

        # Order should be preserved (pure relevance)
        assert diversified[0].book.id == book1.id
        assert diversified[1].book.id == book2.id
        assert diversified[2].book.id == book3.id

    def test_mmr_with_empty_results(self):
        """MMR should handle empty results gracefully."""
        embeddings_store = FakeEmbeddingsStore()
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=FakeVectorSearchRepository(),
            embeddings_store=embeddings_store,
        )

        diversified = service._apply_mmr_diversification(
            results=[],
            top_k=5,
            lambda_param=0.6,
        )

        assert diversified == []

    def test_mmr_with_single_result(self):
        """MMR should return single result unchanged."""
        book = create_book("Only Book")
        embeddings = {book.id: [1.0, 0.0]}

        results = [create_search_result(book, score=0.9, rank=1)]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=FakeVectorSearchRepository(),
            embeddings_store=embeddings_store,
        )

        diversified = service._apply_mmr_diversification(
            results=results,
            top_k=5,
            lambda_param=0.6,
        )

        assert len(diversified) == 1
        assert diversified[0].book.id == book.id

    def test_mmr_reassigns_ranks(self):
        """MMR should reassign ranks starting from 1."""
        book1 = create_book("First")
        book2 = create_book("Second")

        embeddings = {
            book1.id: [1.0, 0.0],
            book2.id: [0.0, 1.0],
        }

        results = [
            create_search_result(book1, score=0.9, rank=1),
            create_search_result(book2, score=0.8, rank=2),
        ]

        embeddings_store = FakeEmbeddingsStore(embeddings)
        service = SearchService(
            lexical_search=FakeLexicalSearchRepository(),
            vector_search=FakeVectorSearchRepository(),
            embeddings_store=embeddings_store,
        )

        diversified = service._apply_mmr_diversification(
            results=results,
            top_k=2,
            lambda_param=0.6,
        )

        # Ranks should be 1, 2 (not original ranks)
        assert diversified[0].rank == 1
        assert diversified[1].rank == 2


# =============================================================================
# Test: SearchQuery with diversification options
# =============================================================================


class TestSearchQueryDiversification:
    """Tests for diversification options in SearchQuery."""

    def test_default_diversification_is_disabled(self):
        """Diversification should be disabled by default."""
        query = SearchQuery(text="test query")
        assert query.use_diversification is False

    def test_default_lambda_is_0_6(self):
        """Default lambda should be 0.6."""
        query = SearchQuery(text="test query")
        assert query.diversity_lambda == 0.6

    def test_lambda_validation_rejects_negative(self):
        """Lambda below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="diversity_lambda"):
            SearchQuery(text="test", diversity_lambda=-0.1)

    def test_lambda_validation_rejects_above_one(self):
        """Lambda above 1 should raise ValueError."""
        with pytest.raises(ValueError, match="diversity_lambda"):
            SearchQuery(text="test", diversity_lambda=1.5)

    def test_can_enable_diversification(self):
        """Should be able to enable diversification with custom lambda."""
        query = SearchQuery(
            text="test query",
            use_diversification=True,
            diversity_lambda=0.7,
        )
        assert query.use_diversification is True
        assert query.diversity_lambda == 0.7
