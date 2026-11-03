"""
Integration tests for BM25SearchRepository.

These are real integration tests (no mocks) that verify the BM25 search
functionality with actual Book entities and the rank-bm25 library.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.domain.entities import Book
from app.domain.value_objects import SearchFilters
from app.infrastructure.search.bm25_search_repository import BM25SearchRepository


@pytest.fixture
def sample_books():
    """Create a small collection of books for testing."""
    return [
        Book(
            id=uuid4(),
            title="Introduction to Machine Learning",
            authors=["John Smith"],
            description="A comprehensive guide to machine learning algorithms and techniques.",
            language="en",
            categories=["Computer Science", "AI"],
            published_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            source="test",
            source_id="test_ml_001",
        ),
        Book(
            id=uuid4(),
            title="Deep Learning Fundamentals",
            authors=["Jane Doe"],
            description="Learn the fundamentals of deep learning and neural networks.",
            language="en",
            categories=["Computer Science", "AI", "Deep Learning"],
            published_date=datetime(2021, 6, 15, tzinfo=timezone.utc),
            source="test",
            source_id="test_dl_002",
        ),
        Book(
            id=uuid4(),
            title="Python Programming for Beginners",
            authors=["Alice Johnson"],
            description="Start your programming journey with Python. No prior experience required.",
            language="en",
            categories=["Programming", "Python"],
            published_date=datetime(2019, 3, 10, tzinfo=timezone.utc),
            source="test",
            source_id="test_py_003",
        ),
        Book(
            id=uuid4(),
            title="Historia de España",
            authors=["Carlos García"],
            description="Un recorrido por la historia de España desde sus orígenes.",
            language="es",
            categories=["History", "Spain"],
            published_date=datetime(2018, 11, 20, tzinfo=timezone.utc),
            source="test",
            source_id="test_es_004",
        ),
        Book(
            id=uuid4(),
            title="Advanced Neural Networks",
            authors=["Bob Wilson"],
            description="Explore advanced topics in neural networks and deep learning architectures.",
            language="en",
            categories=["Computer Science", "AI", "Deep Learning"],
            published_date=datetime(2022, 8, 5, tzinfo=timezone.utc),
            source="test",
            source_id="test_nn_005",
        ),
    ]
@pytest.fixture
def bm25_repo(sample_books):
    """Create a BM25 repository with indexed books."""
    repo = BM25SearchRepository()
    repo.build_index(sample_books)
    return repo


class TestBM25SearchRepository:
    """Test suite for BM25SearchRepository."""

    def test_build_index_creates_valid_index(self, sample_books):
        """Test that build_index successfully creates an index."""
        repo = BM25SearchRepository()
        repo.build_index(sample_books)

        assert repo._index is not None
        assert len(repo._books) == 5
        assert len(repo._book_map) == 5
        assert len(repo._tokenized_corpus) == 5

    def test_build_index_with_empty_list(self):
        """Test building index with empty book list."""
        repo = BM25SearchRepository()
        repo.build_index([])

        assert repo._index is None
        assert len(repo._books) == 0
        assert len(repo._book_map) == 0

    def test_search_returns_non_empty_results(self, bm25_repo):
        """Test that search returns results for a relevant query."""
        results = bm25_repo.search("machine learning", max_results=10)

        assert len(results) > 0
        assert all(isinstance(result.book, Book) for result in results)

    def test_search_result_structure(self, bm25_repo):
        """Test that SearchResult entities have correct structure."""
        results = bm25_repo.search("deep learning neural networks", max_results=5)

        assert len(results) > 0

        for result in results:
            # Check that book is a complete Book entity
            assert isinstance(result.book, Book)
            assert result.book.id is not None
            assert result.book.title is not None

            # Check scores
            assert result.lexical_score is not None
            assert result.lexical_score >= 0
            assert result.final_score == result.lexical_score
            assert result.vector_score is None

            # Check metadata
            assert result.source == "lexical"
            assert result.rank >= 1

    def test_search_ranking_is_consecutive(self, bm25_repo):
        """Test that rank values are consecutive starting from 1."""
        results = bm25_repo.search("python programming", max_results=5)

        assert len(results) > 0

        ranks = [result.rank for result in results]
        expected_ranks = list(range(1, len(results) + 1))
        assert ranks == expected_ranks

    def test_search_respects_max_results(self, bm25_repo):
        """Test that search returns at most max_results."""
        results = bm25_repo.search("learning", max_results=2)

        assert len(results) <= 2

    def test_search_with_empty_query_raises_error(self, bm25_repo):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            bm25_repo.search("", max_results=10)

        with pytest.raises(ValueError, match="query_text cannot be empty"):
            bm25_repo.search("   ", max_results=10)

    def test_search_empty_index_returns_empty_list(self):
        """Test search on empty index returns empty results."""
        repo = BM25SearchRepository()
        results = repo.search("any query", max_results=10)

        assert results == []

    def test_search_with_language_filter(self, bm25_repo):
        """Test filtering by language."""
        filters = SearchFilters(language="es")
        results = bm25_repo.search("historia", max_results=10, filters=filters)

        assert len(results) > 0
        assert all(result.book.language == "es" for result in results)

    def test_search_with_category_filter(self, bm25_repo):
        """Test filtering by category (substring match)."""
        filters = SearchFilters(category="Deep Learning")
        results = bm25_repo.search("neural", max_results=10, filters=filters)

        assert len(results) > 0
        for result in results:
            assert any("deep learning" in cat.lower() for cat in result.book.categories)

    def test_search_with_year_range_filter(self, bm25_repo):
        """Test filtering by publication year range."""
        filters = SearchFilters(min_year=2020, max_year=2022)
        results = bm25_repo.search("learning", max_results=10, filters=filters)

        assert len(results) > 0
        for result in results:
            year = result.book.get_published_year()
            assert year is not None
            assert 2020 <= year <= 2022

    def test_search_with_combined_filters(self, bm25_repo):
        """Test combining multiple filters."""
        filters = SearchFilters(language="en", category="AI", min_year=2020)
        results = bm25_repo.search("learning", max_results=10, filters=filters)

        for result in results:
            assert result.book.language == "en"
            assert any("ai" in cat.lower() for cat in result.book.categories)
            year = result.book.get_published_year()
            assert year is not None
            assert year >= 2020

    def test_add_to_index_adds_new_book(self, bm25_repo):
        """Test adding a single book to existing index."""
        new_book = Book(
            id=uuid4(),
            title="Quantum Computing Basics",
            authors=["David Lee"],
            description="Introduction to quantum computing principles.",
            language="en",
            categories=["Quantum", "Computer Science"],
            published_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            source="test",
            source_id="test_qc_006",
        )

        bm25_repo.add_to_index(new_book)

        # Verify we can search for the new book (behavioral test)
        results = bm25_repo.search("quantum computing", max_results=5)
        book_ids = [r.book.id for r in results]
        assert new_book.id in book_ids

    def test_add_to_index_skips_duplicate(self, bm25_repo, sample_books):
        """Test that adding a duplicate book is handled gracefully."""
        existing_book = sample_books[0]

        # Search before adding duplicate
        results_before = bm25_repo.search("machine learning", max_results=10)
        count_before = len(results_before)

        bm25_repo.add_to_index(existing_book)

        # Search after - should have same count (behavioral test)
        results_after = bm25_repo.search("machine learning", max_results=10)
        count_after = len(results_after)

        assert count_after == count_before

    def test_save_and_load_index(self, bm25_repo, tmp_path):
        """Test persisting and loading index from disk."""
        index_path = tmp_path / "test_index.pkl"

        # Get results from original index
        original_results = bm25_repo.search("machine learning", max_results=5)
        original_count = len(original_results)

        # Save index
        bm25_repo.save_index(str(index_path))
        assert index_path.exists()

        # Create new repo and load index
        new_repo = BM25SearchRepository()
        new_repo.load_index(str(index_path))

        # Verify search works with loaded index (behavioral test)
        loaded_results = new_repo.search("machine learning", max_results=5)
        assert len(loaded_results) == original_count

        # Verify same books are returned (by ID)
        original_ids = {r.book.id for r in original_results}
        loaded_ids = {r.book.id for r in loaded_results}
        assert original_ids == loaded_ids

    def test_load_index_with_invalid_path_raises_error(self, tmp_path):
        """Test that loading from non-existent path raises IOError."""
        repo = BM25SearchRepository()

        # Use tmp_path for portability across OS
        nonexistent_path = tmp_path / "does_not_exist.pkl"

        with pytest.raises(IOError, match="Index file not found"):
            repo.load_index(str(nonexistent_path))

    def test_tokenization(self):
        """Test the tokenization logic."""
        text = "Hello World, This is a TEST!"
        tokens = BM25SearchRepository._tokenize(text)

        assert tokens == ["hello", "world,", "this", "is", "a", "test!"]

    def test_matches_filters_with_all_filters(self, sample_books):
        """Test filter matching with all filter types."""
        book = sample_books[1]  # Deep Learning Fundamentals
        filters = SearchFilters(
            language="en", category="Deep Learning", min_year=2021, max_year=2023
        )

        assert BM25SearchRepository._matches_filters(book, filters)

    def test_matches_filters_rejects_wrong_language(self, sample_books):
        """Test that wrong language is rejected."""
        book = sample_books[0]  # English book
        filters = SearchFilters(language="es")

        assert not BM25SearchRepository._matches_filters(book, filters)

    def test_matches_filters_rejects_wrong_category(self, sample_books):
        """Test that wrong category is rejected."""
        book = sample_books[2]  # Python Programming
        filters = SearchFilters(category="History")

        assert not BM25SearchRepository._matches_filters(book, filters)

    def test_matches_filters_rejects_out_of_year_range(self, sample_books):
        """Test that books outside year range are rejected."""
        book = sample_books[3]  # Published 2018
        filters = SearchFilters(min_year=2020)

        assert not BM25SearchRepository._matches_filters(book, filters)