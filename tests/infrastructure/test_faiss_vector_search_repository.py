"""
Tests for FaissVectorSearchRepository.

These tests verify the FAISS-based vector search repository works correctly:
- Returns empty list if index not built / ntotal == 0
- Dimension mismatch raises ValueError
- Ranks are consecutive from 1
- source == "vector", lexical_score is None, vector_score == final_score
- Filters work (language, category, year range)
- Out-of-sync mapping triggers RuntimeError

Test approach:
- Unit tests use a mock FaissIndexProvider to isolate the repository logic
- Integration tests use the real EmbeddingsStoreFaiss for end-to-end verification

The mock approach allows testing edge cases (empty index, out-of-sync mapping)
without needing to construct invalid FAISS state.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any
from uuid import uuid4, UUID

import numpy as np

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import SearchFilters
from app.infrastructure.search import FaissIndexProvider
from app.infrastructure.search.faiss_vector_search_repository import (
    FaissVectorSearchRepository,
    OVER_FETCH_FACTOR,
)


# -----------------------------------------------------------------------------
# Mock FaissIndexProvider for unit tests
# -----------------------------------------------------------------------------


@dataclass
class MockFaissIndex:
    """
    Mock FAISS index that simulates IndexFlatL2 behavior.

    This mock allows testing the repository without a real FAISS index,
    enabling controlled testing of edge cases.
    """

    ntotal: int
    dimension: int
    # Pre-computed search results: list of (distance, index) pairs
    search_results: List[tuple[float, int]]

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate FAISS search.

        Returns distances and indices as numpy arrays with shape (1, k).
        """
        # Return up to k results, pad with -1 if fewer available
        results = self.search_results[:k]

        distances = []
        indices = []

        for dist, idx in results:
            distances.append(dist)
            indices.append(idx)

        # Pad with -1 and high distance if fewer results
        while len(indices) < k:
            indices.append(-1)
            distances.append(float("inf"))

        return (
            np.array([distances], dtype=np.float32),
            np.array([indices], dtype=np.int64),
        )


class MockFaissIndexProvider:
    """
    Mock implementation of FaissIndexProvider protocol for testing.

    Allows precise control over:
    - Index existence and size
    - ID mapping
    - Search results (distances and indices)
    """

    def __init__(
        self,
        dimension: int = 384,
        id_mapping: Optional[List[str]] = None,
        search_results: Optional[List[tuple[float, int]]] = None,
        index_exists: bool = True,
    ):
        self._dimension = dimension
        self._id_mapping = id_mapping or []
        self._search_results = search_results or []
        self._index_exists = index_exists

        # Create mock index
        if index_exists and self._id_mapping:
            self._index = MockFaissIndex(
                ntotal=len(self._id_mapping),
                dimension=dimension,
                search_results=self._search_results,
            )
        else:
            self._index = None

    def get_index(self) -> Optional[Any]:
        return self._index

    def get_id_mapping(self) -> List[str]:
        return self._id_mapping

    def get_dimension(self) -> int:
        return self._dimension

    def set_out_of_sync(self, extra_vectors: int = 1) -> None:
        """
        Make the index out of sync with the mapping.

        This simulates corruption where the FAISS index has more/fewer
        vectors than the id_mapping.
        """
        if self._index is not None:
            self._index.ntotal = len(self._id_mapping) + extra_vectors


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def create_test_book(
    book_id: Optional[UUID] = None,
    title: str = "Test Book",
    authors: Optional[List[str]] = None,
    language: Optional[str] = None,
    categories: Optional[List[str]] = None,
    published_year: Optional[int] = None,
) -> Book:
    """Helper to create test Book instances."""
    published_date = None
    if published_year:
        published_date = datetime(published_year, 1, 1)

    return Book(
        id=book_id or uuid4(),
        title=title,
        authors=authors or ["Test Author"],
        description=f"Description of {title}",
        language=language,
        categories=categories or [],
        published_date=published_date,
        source="test",
        source_id=str(book_id or uuid4()),
    )


@pytest.fixture
def sample_books() -> List[Book]:
    """Create a sample set of books for testing."""
    return [
        create_test_book(
            book_id=UUID("00000000-0000-0000-0000-000000000001"),
            title="Mystery in London",
            language="en",
            categories=["Mystery", "Fiction"],
            published_year=2020,
        ),
        create_test_book(
            book_id=UUID("00000000-0000-0000-0000-000000000002"),
            title="Space Adventure",
            language="en",
            categories=["Science Fiction"],
            published_year=2021,
        ),
        create_test_book(
            book_id=UUID("00000000-0000-0000-0000-000000000003"),
            title="Cocina Espanola",
            language="es",
            categories=["Cooking", "Non-Fiction"],
            published_year=2019,
        ),
        create_test_book(
            book_id=UUID("00000000-0000-0000-0000-000000000004"),
            title="Romance Novel",
            language="en",
            categories=["Romance"],
            published_year=2018,
        ),
        create_test_book(
            book_id=UUID("00000000-0000-0000-0000-000000000005"),
            title="History Book",
            language="en",
            categories=["History", "Non-Fiction"],
            published_year=2015,
        ),
    ]


@pytest.fixture
def mock_provider_with_books(sample_books: List[Book]) -> tuple[MockFaissIndexProvider, List[Book]]:
    """
    Create a mock provider with search results matching sample_books.

    Search results are ordered by distance (ascending), simulating
    what FAISS would return for a query.
    """
    # ID mapping matches book order (sorted by UUID for reproducibility)
    id_mapping = [str(book.id) for book in sorted(sample_books, key=lambda b: str(b.id))]

    # Search results: (distance, faiss_index) ordered by distance
    # Lower distance = more similar. We'll return books in a specific order.
    search_results = [
        (0.5, 0),   # First book in mapping (distance 0.5)
        (1.0, 1),   # Second book (distance 1.0)
        (1.5, 2),   # Third book (distance 1.5)
        (2.0, 3),   # Fourth book (distance 2.0)
        (2.5, 4),   # Fifth book (distance 2.5)
    ]

    provider = MockFaissIndexProvider(
        dimension=384,
        id_mapping=id_mapping,
        search_results=search_results,
    )

    return provider, sample_books


# -----------------------------------------------------------------------------
# Test: Empty Index Handling
# -----------------------------------------------------------------------------


class TestEmptyIndex:
    """Tests for empty or non-existent index handling."""

    def test_returns_empty_list_when_index_not_built(self, sample_books: List[Book]):
        """Should return empty list when index is None."""
        provider = MockFaissIndexProvider(index_exists=False)
        repo = FaissVectorSearchRepository(provider, sample_books)

        query_embedding = [0.1] * 384
        results = repo.search(query_embedding, max_results=10)

        assert results == []

    def test_returns_empty_list_when_index_empty(self, sample_books: List[Book]):
        """Should return empty list when index.ntotal == 0."""
        provider = MockFaissIndexProvider(
            id_mapping=[],  # Empty mapping
            search_results=[],
        )
        repo = FaissVectorSearchRepository(provider, sample_books)

        query_embedding = [0.1] * 384
        results = repo.search(query_embedding, max_results=10)

        assert results == []


# -----------------------------------------------------------------------------
# Test: Dimension Validation
# -----------------------------------------------------------------------------


class TestDimensionValidation:
    """Tests for query embedding dimension validation."""

    def test_wrong_dimension_raises_value_error(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Should raise ValueError when query dimension doesn't match."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        # Wrong dimension (100 instead of 384)
        wrong_embedding = [0.1] * 100

        with pytest.raises(ValueError, match="dimension mismatch"):
            repo.search(wrong_embedding, max_results=10)

    def test_correct_dimension_succeeds(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Should not raise when query dimension matches."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        correct_embedding = [0.1] * 384

        # Should not raise
        results = repo.search(correct_embedding, max_results=10)
        assert isinstance(results, list)


# -----------------------------------------------------------------------------
# Test: SearchResult Structure
# -----------------------------------------------------------------------------


class TestSearchResultStructure:
    """Tests for correct SearchResult field values."""

    def test_source_is_vector(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """All results should have source == 'vector'."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=5)

        assert all(r.source == "vector" for r in results)

    def test_lexical_score_is_none(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """All results should have lexical_score == None."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=5)

        assert all(r.lexical_score is None for r in results)

    def test_vector_score_equals_final_score(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """vector_score should equal final_score for all results."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=5)

        for result in results:
            assert result.vector_score == result.final_score

    def test_book_is_fully_hydrated(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Results should contain complete Book entities, not just IDs."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=5)

        for result in results:
            assert isinstance(result.book, Book)
            assert result.book.title is not None
            assert result.book.authors is not None
            assert len(result.book.authors) > 0


# -----------------------------------------------------------------------------
# Test: Rank Ordering
# -----------------------------------------------------------------------------


class TestRankOrdering:
    """Tests for correct rank assignment."""

    def test_ranks_are_consecutive_from_one(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Ranks should be 1, 2, 3, ... (1-indexed, consecutive)."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=5)

        expected_ranks = list(range(1, len(results) + 1))
        actual_ranks = [r.rank for r in results]

        assert actual_ranks == expected_ranks

    def test_results_sorted_by_similarity_descending(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Results should be sorted by final_score descending (highest first)."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=5)

        scores = [r.final_score for r in results]

        # Check descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i+1} ({scores[i]}) should be >= "
                f"score at rank {i+2} ({scores[i+1]})"
            )

    def test_similarity_score_semantics(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """
        Similarity scores should follow the formula: 1 / (1 + distance).

        This means:
        - Lower distance -> higher similarity
        - distance=0 -> similarity=1
        - Scores are in range (0, 1]
        """
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=5)

        # All scores should be positive and <= 1
        for result in results:
            assert 0 < result.final_score <= 1, (
                f"Score {result.final_score} should be in (0, 1]"
            )

        # First result should have highest score (lowest distance)
        # Our mock has distances 0.5, 1.0, 1.5, 2.0, 2.5
        # Expected scores: 1/(1+0.5)=0.667, 1/(1+1.0)=0.5, etc.
        expected_first_score = 1.0 / (1.0 + 0.5)  # ~0.667
        assert abs(results[0].final_score - expected_first_score) < 0.001


# -----------------------------------------------------------------------------
# Test: Filtering
# -----------------------------------------------------------------------------


class TestFiltering:
    """Tests for filter application."""

    def test_language_filter(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Language filter should only return books with matching language."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        filters = SearchFilters(language="es")
        results = repo.search([0.1] * 384, max_results=10, filters=filters)

        assert len(results) > 0
        assert all(r.book.language == "es" for r in results)

    def test_language_filter_case_insensitive(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Language filter should be case-insensitive."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        filters_upper = SearchFilters(language="EN")
        filters_lower = SearchFilters(language="en")

        results_upper = repo.search([0.1] * 384, max_results=10, filters=filters_upper)
        results_lower = repo.search([0.1] * 384, max_results=10, filters=filters_lower)

        # Should return same books
        upper_ids = {r.book.id for r in results_upper}
        lower_ids = {r.book.id for r in results_lower}

        assert upper_ids == lower_ids

    def test_category_filter(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Category filter should match substring in any category."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        filters = SearchFilters(category="Fiction")
        results = repo.search([0.1] * 384, max_results=10, filters=filters)

        assert len(results) > 0
        for result in results:
            categories_lower = [c.lower() for c in result.book.categories]
            assert any("fiction" in c for c in categories_lower)

    def test_min_year_filter(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """min_year filter should exclude books published before the year."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        filters = SearchFilters(min_year=2020)
        results = repo.search([0.1] * 384, max_results=10, filters=filters)

        assert len(results) > 0
        for result in results:
            year = result.book.get_published_year()
            assert year is not None
            assert year >= 2020

    def test_max_year_filter(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """max_year filter should exclude books published after the year."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        filters = SearchFilters(max_year=2018)
        results = repo.search([0.1] * 384, max_results=10, filters=filters)

        assert len(results) > 0
        for result in results:
            year = result.book.get_published_year()
            assert year is not None
            assert year <= 2018

    def test_combined_filters(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Multiple filters should be applied as AND conditions."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        # English books from 2019 onwards
        filters = SearchFilters(language="en", min_year=2019)
        results = repo.search([0.1] * 384, max_results=10, filters=filters)

        for result in results:
            assert result.book.language.lower() == "en"
            assert result.book.get_published_year() >= 2019

    def test_empty_filters_returns_all(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Empty SearchFilters should not filter out any results."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results_no_filter = repo.search([0.1] * 384, max_results=10)
        results_empty_filter = repo.search([0.1] * 384, max_results=10, filters=SearchFilters())

        assert len(results_no_filter) == len(results_empty_filter)

    def test_filter_excludes_books_without_metadata(
        self, sample_books: List[Book]
    ):
        """Books without the filtered field should be excluded."""
        # Create a book without language
        book_no_lang = create_test_book(
            book_id=UUID("00000000-0000-0000-0000-000000000099"),
            title="No Language Book",
            language=None,
        )
        books = sample_books + [book_no_lang]

        id_mapping = [str(book.id) for book in sorted(books, key=lambda b: str(b.id))]
        search_results = [(float(i), i) for i in range(len(books))]

        provider = MockFaissIndexProvider(
            id_mapping=id_mapping,
            search_results=search_results,
        )
        repo = FaissVectorSearchRepository(provider, books)

        filters = SearchFilters(language="en")
        results = repo.search([0.1] * 384, max_results=10, filters=filters)

        result_ids = {r.book.id for r in results}
        assert book_no_lang.id not in result_ids


# -----------------------------------------------------------------------------
# Test: Index Sync Validation
# -----------------------------------------------------------------------------


class TestIndexSyncValidation:
    """Tests for index/mapping consistency validation."""

    def test_out_of_sync_raises_runtime_error(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Should raise RuntimeError when id_mapping size != index.ntotal."""
        provider, books = mock_provider_with_books

        # Make the index out of sync
        provider.set_out_of_sync(extra_vectors=2)

        repo = FaissVectorSearchRepository(provider, books)

        with pytest.raises(RuntimeError, match="out of sync"):
            repo.search([0.1] * 384, max_results=10)

    def test_sync_validation_error_message_is_helpful(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Error message should include mapping and index sizes."""
        provider, books = mock_provider_with_books
        provider.set_out_of_sync(extra_vectors=3)

        repo = FaissVectorSearchRepository(provider, books)

        with pytest.raises(RuntimeError) as exc_info:
            repo.search([0.1] * 384, max_results=10)

        error_msg = str(exc_info.value)
        assert "id_mapping" in error_msg
        assert "FAISS index" in error_msg
        assert "Rebuild" in error_msg


# -----------------------------------------------------------------------------
# Test: Over-fetch Behavior
# -----------------------------------------------------------------------------


class TestOverFetchBehavior:
    """Tests for the over-fetch strategy used for filtering."""

    def test_max_results_respected(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Should return at most max_results items."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=2)

        assert len(results) <= 2

    def test_over_fetch_compensates_for_filtering(self, sample_books: List[Book]):
        """
        Over-fetching should allow getting max_results even after filtering.

        If we request 3 results and filter removes 2 of the first 3,
        we should still get 3 results (from the over-fetched candidates).
        """
        # Create provider where most results will be filtered out
        # Only "en" books pass the filter
        id_mapping = [str(book.id) for book in sorted(sample_books, key=lambda b: str(b.id))]

        # All 5 books as search results
        search_results = [(float(i) * 0.5, i) for i in range(len(sample_books))]

        provider = MockFaissIndexProvider(
            id_mapping=id_mapping,
            search_results=search_results,
        )
        repo = FaissVectorSearchRepository(provider, sample_books)

        # Filter for English - should get 4 books (all except Spanish cookbook)
        filters = SearchFilters(language="en")
        results = repo.search([0.1] * 384, max_results=3, filters=filters)

        assert len(results) == 3
        assert all(r.book.language == "en" for r in results)


# -----------------------------------------------------------------------------
# Test: update_books Helper
# -----------------------------------------------------------------------------


class TestUpdateBooks:
    """Tests for the update_books helper method."""

    def test_update_books_replaces_mapping(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """update_books should replace the internal book mapping."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        # Create new books
        new_books = [
            create_test_book(
                book_id=UUID("00000000-0000-0000-0000-000000000001"),
                title="Updated Title",
            ),
        ]

        repo.update_books(new_books)

        # The mapping should now have only the new books
        assert len(repo._book_map) == 1


# -----------------------------------------------------------------------------
# Test: Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_book_in_index_but_not_in_mapping_is_skipped(self, sample_books: List[Book]):
        """
        If a book_id is in FAISS but not in book_map, it should be skipped
        with a warning (not crash).
        """
        # Only include some books in the repository
        partial_books = sample_books[:2]

        # But the index has all 5 books
        id_mapping = [str(book.id) for book in sorted(sample_books, key=lambda b: str(b.id))]
        search_results = [(float(i), i) for i in range(len(sample_books))]

        provider = MockFaissIndexProvider(
            id_mapping=id_mapping,
            search_results=search_results,
        )

        # Repository only knows about partial_books
        repo = FaissVectorSearchRepository(provider, partial_books)

        # Should not crash, but return fewer results
        results = repo.search([0.1] * 384, max_results=10)

        # Only books that are in both index and book_map should appear
        result_ids = {r.book.id for r in results}
        partial_ids = {b.id for b in partial_books}

        assert result_ids.issubset(partial_ids)

    def test_handles_faiss_returning_minus_one_indices(self, sample_books: List[Book]):
        """
        FAISS returns -1 for indices when fewer than k results exist.
        These should be gracefully ignored.
        """
        # Only 2 books in index, but we'll request 10
        small_books = sample_books[:2]
        id_mapping = [str(book.id) for book in sorted(small_books, key=lambda b: str(b.id))]

        # Only 2 real results
        search_results = [
            (0.5, 0),
            (1.0, 1),
        ]

        provider = MockFaissIndexProvider(
            id_mapping=id_mapping,
            search_results=search_results,
        )
        repo = FaissVectorSearchRepository(provider, small_books)

        # Request more than available
        results = repo.search([0.1] * 384, max_results=10)

        # Should only get 2 results (no crashes from -1 indices)
        assert len(results) == 2

    def test_zero_max_results(
        self, mock_provider_with_books: tuple[MockFaissIndexProvider, List[Book]]
    ):
        """Requesting 0 results should return empty list."""
        provider, books = mock_provider_with_books
        repo = FaissVectorSearchRepository(provider, books)

        results = repo.search([0.1] * 384, max_results=0)

        assert results == []


# -----------------------------------------------------------------------------
# Integration Test with Real EmbeddingsStoreFaiss
# -----------------------------------------------------------------------------


class TestIntegrationWithRealStore:
    """
    Integration tests using the real EmbeddingsStoreFaiss.

    These tests verify end-to-end functionality but are slower
    due to actual embedding generation and FAISS operations.
    """

    @pytest.fixture
    def real_store_and_books(self, sample_books: List[Book]):
        """Create a real EmbeddingsStoreFaiss with indexed books."""
        from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss

        store = EmbeddingsStoreFaiss()

        # Generate and store embeddings for each book
        for book in sample_books:
            text = book.get_searchable_text()
            embedding = store.generate_embedding(text)
            store.store_embedding(book.id, embedding)

        store.build_index()

        return store, sample_books

    def test_real_vector_search(
        self, real_store_and_books: tuple[Any, List[Book]]
    ):
        """End-to-end vector search with real embeddings."""
        store, books = real_store_and_books
        repo = FaissVectorSearchRepository(store, books)

        # Search for mystery-related content
        query_embedding = store.generate_embedding("detective crime investigation mystery")
        results = repo.search(query_embedding, max_results=3)

        assert len(results) == 3
        assert all(r.source == "vector" for r in results)
        assert all(r.lexical_score is None for r in results)
        assert all(r.vector_score == r.final_score for r in results)

        # Ranks should be 1, 2, 3
        assert [r.rank for r in results] == [1, 2, 3]

    def test_semantic_similarity_ranking(
        self, real_store_and_books: tuple[Any, List[Book]]
    ):
        """
        Semantically similar books should rank higher.

        A query about space/sci-fi should rank the "Space Adventure" book
        higher than the "Cocina Espanola" (Spanish Cookbook).
        """
        store, books = real_store_and_books
        repo = FaissVectorSearchRepository(store, books)

        # Search for space/sci-fi content
        query_embedding = store.generate_embedding("space exploration science fiction galaxy")
        results = repo.search(query_embedding, max_results=5)

        # Find ranks of specific books
        space_book = next((r for r in results if "Space" in r.book.title), None)
        cookbook = next((r for r in results if "Cocina" in r.book.title), None)

        assert space_book is not None, "Space Adventure should be in results"

        if cookbook is not None:
            # Space book should rank higher (lower rank number)
            assert space_book.rank < cookbook.rank, (
                f"Space book (rank {space_book.rank}) should rank higher than "
                f"cookbook (rank {cookbook.rank}) for sci-fi query"
            )

    def test_filters_work_with_real_store(
        self, real_store_and_books: tuple[Any, List[Book]]
    ):
        """Filters should work correctly with real embeddings."""
        store, books = real_store_and_books
        repo = FaissVectorSearchRepository(store, books)

        query_embedding = store.generate_embedding("interesting book to read")

        # Filter for Spanish books
        filters = SearchFilters(language="es")
        results = repo.search(query_embedding, max_results=10, filters=filters)

        assert len(results) > 0
        assert all(r.book.language == "es" for r in results)
