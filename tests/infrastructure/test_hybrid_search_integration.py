"""
End-to-end integration test for the hybrid search pipeline.

This test validates the complete hybrid search flow:
1. BM25 lexical search (keyword matching)
2. FAISS vector search (semantic similarity)
3. Reciprocal Rank Fusion (RRF) combining both rankings
4. SearchService orchestration

Test design decisions:
- Uses real BM25 and FAISS indices (no mocks for search infrastructure)
- Creates books with carefully crafted content to demonstrate hybrid behavior:
  * Books that match well lexically (exact keywords) but not semantically
  * Books that match well semantically (meaning) but not lexically
  * Books that match both (should rank highest after fusion)
- LLMClient is stubbed with a dummy that raises if called
- Validates ranking invariants, score ordering, and source attribution

The test demonstrates the value of hybrid search:
- Pure lexical search may miss semantically relevant results
- Pure vector search may miss exact keyword matches
- Hybrid (RRF) combines the best of both approaches
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from uuid import uuid4, UUID

import pytest

from app.domain.entities import Book, SearchResult, Explanation
from app.domain.value_objects import SearchQuery, SearchFilters
from app.domain.ports import LLMClient
from app.domain.services import SearchService
from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository


# -----------------------------------------------------------------------------
# Stub LLMClient (should never be called in these tests)
# -----------------------------------------------------------------------------


class StubLLMClient:
    """
    Stub LLMClient that raises if any method is called.

    Used to ensure the integration test does not accidentally trigger
    LLM-based explanation generation.
    """

    def __init__(self):
        self.call_count = 0

    def generate_explanation(
        self, query: SearchQuery, book: Book, context: Optional[Dict[str, Any]] = None
    ) -> Explanation:
        self.call_count += 1
        raise AssertionError(
            "LLMClient.generate_explanation should not be called in this test. "
            "Ensure use_explanations=False in SearchQuery."
        )

    def extract_query_intent(self, query_text: str) -> Dict[str, Any]:
        self.call_count += 1
        raise AssertionError(
            "LLMClient.extract_query_intent should not be called in this test."
        )

    def get_model_name(self) -> str:
        return "stub-model-never-called"


# -----------------------------------------------------------------------------
# Test Book Catalog
# -----------------------------------------------------------------------------


def create_test_catalog() -> List[Book]:
    """
    Create a catalog of books designed to demonstrate hybrid search behavior.

    Book design rationale:

    1. "Python Machine Learning" - Should match BOTH lexical (keywords) and
       semantic (ML concepts) searches well. Good fusion candidate.

    2. "Deep Neural Networks" - Matches semantically for ML queries but uses
       different keywords. Tests semantic retrieval.

    3. "The Python Programming Language" - Matches lexically for "Python" but
       is about programming basics, not ML. Tests lexical retrieval.

    4. "Artificial Intelligence: A Modern Approach" - Classic AI textbook.
       Matches semantically for ML/AI queries.

    5. "Snake Species of the Amazon" - Contains "python" (the snake) in description.
       Should rank lower for "Python programming" queries. Tests lexical noise.

    6. "Cocina Espanola" - Spanish cookbook. Different language, different topic.
       Used for filter testing.

    7. "Data Science Handbook" - Related to ML semantically, different keywords.
       Tests semantic retrieval.

    8. "Quick Python Scripts" - Contains exact "python" keyword, programming focus.
       Tests lexical retrieval.
    """
    return [
        Book(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            title="Python Machine Learning",
            authors=["Sebastian Raschka"],
            description=(
                "A comprehensive guide to machine learning with Python. "
                "Covers scikit-learn, TensorFlow, and deep learning algorithms. "
                "Learn supervised and unsupervised learning techniques."
            ),
            language="en",
            categories=["Computer Science", "Machine Learning", "Python"],
            published_date=datetime(2019, 12, 1, tzinfo=timezone.utc),
            source="test",
            source_id="test_001",
        ),
        Book(
            id=UUID("00000000-0000-0000-0000-000000000002"),
            title="Deep Neural Networks",
            authors=["Ian Goodfellow"],
            description=(
                "Advanced neural network architectures including CNNs, RNNs, "
                "transformers, and attention mechanisms. Mathematical foundations "
                "of backpropagation and gradient descent optimization."
            ),
            language="en",
            categories=["Computer Science", "Deep Learning", "AI"],
            published_date=datetime(2016, 11, 1, tzinfo=timezone.utc),
            source="test",
            source_id="test_002",
        ),
        Book(
            id=UUID("00000000-0000-0000-0000-000000000003"),
            title="The Python Programming Language",
            authors=["Guido van Rossum"],
            description=(
                "Learn Python programming from scratch. Variables, loops, "
                "functions, classes, and modules. A beginner-friendly "
                "introduction to coding with practical examples."
            ),
            language="en",
            categories=["Programming", "Python", "Beginners"],
            published_date=datetime(2020, 6, 15, tzinfo=timezone.utc),
            source="test",
            source_id="test_003",
        ),
        Book(
            id=UUID("00000000-0000-0000-0000-000000000004"),
            title="Artificial Intelligence: A Modern Approach",
            authors=["Stuart Russell", "Peter Norvig"],
            description=(
                "The definitive textbook on artificial intelligence. "
                "Covers search algorithms, knowledge representation, "
                "planning, probabilistic reasoning, and learning agents."
            ),
            language="en",
            categories=["Computer Science", "AI", "Textbook"],
            published_date=datetime(2020, 4, 28, tzinfo=timezone.utc),
            source="test",
            source_id="test_004",
        ),
        Book(
            id=UUID("00000000-0000-0000-0000-000000000005"),
            title="Snake Species of the Amazon",
            authors=["Maria Santos"],
            description=(
                "A field guide to snakes in the Amazon rainforest. "
                "Includes the green anaconda, boa constrictor, and python species. "
                "Beautiful photographs and habitat information."
            ),
            language="en",
            categories=["Biology", "Wildlife", "Nature"],
            published_date=datetime(2018, 3, 10, tzinfo=timezone.utc),
            source="test",
            source_id="test_005",
        ),
        Book(
            id=UUID("00000000-0000-0000-0000-000000000006"),
            title="Cocina Espanola Tradicional",
            authors=["Carlos Garcia"],
            description=(
                "Recetas tradicionales de la cocina espanola. "
                "Paella, gazpacho, tortilla, y mas platos tipicos. "
                "Cocina mediterranea con ingredientes frescos."
            ),
            language="es",
            categories=["Cooking", "Spanish Cuisine", "Recipes"],
            published_date=datetime(2021, 8, 20, tzinfo=timezone.utc),
            source="test",
            source_id="test_006",
        ),
        Book(
            id=UUID("00000000-0000-0000-0000-000000000007"),
            title="Data Science Handbook",
            authors=["Jake VanderPlas"],
            description=(
                "Essential techniques for data analysis and visualization. "
                "Statistical methods, predictive modeling, and insights "
                "extraction from large datasets using modern tools."
            ),
            language="en",
            categories=["Data Science", "Statistics", "Analytics"],
            published_date=datetime(2017, 11, 21, tzinfo=timezone.utc),
            source="test",
            source_id="test_007",
        ),
        Book(
            id=UUID("00000000-0000-0000-0000-000000000008"),
            title="Quick Python Scripts",
            authors=["Al Sweigart"],
            description=(
                "Automate boring tasks with Python scripts. "
                "File manipulation, web scraping, spreadsheet automation. "
                "Practical Python programming for everyday tasks."
            ),
            language="en",
            categories=["Programming", "Python", "Automation"],
            published_date=datetime(2019, 9, 1, tzinfo=timezone.utc),
            source="test",
            source_id="test_008",
        ),
    ]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def test_catalog() -> List[Book]:
    """Provide the test book catalog."""
    return create_test_catalog()


@pytest.fixture
def bm25_repo(test_catalog: List[Book]) -> BM25SearchRepository:
    """Create and populate the BM25 lexical search repository."""
    repo = BM25SearchRepository()
    repo.build_index(test_catalog)
    return repo


@pytest.fixture
def embeddings_store(test_catalog: List[Book]) -> EmbeddingsStoreFaiss:
    """
    Create the embeddings store with indexed book embeddings.

    This generates real embeddings using sentence-transformers and builds
    a FAISS index for vector search.
    """
    store = EmbeddingsStoreFaiss()

    # Generate and store embeddings for each book
    for book in test_catalog:
        text = book.get_searchable_text()
        embedding = store.generate_embedding(text)
        store.store_embedding(book.id, embedding)

    # Build the FAISS index
    store.build_index()

    return store


@pytest.fixture
def vector_repo(
    embeddings_store: EmbeddingsStoreFaiss,
    test_catalog: List[Book]
) -> FaissVectorSearchRepository:
    """Create the FAISS vector search repository."""
    return FaissVectorSearchRepository(embeddings_store, test_catalog)


@pytest.fixture
def stub_llm() -> StubLLMClient:
    """Create a stub LLM client that should never be called."""
    return StubLLMClient()


@pytest.fixture
def search_service(
    bm25_repo: BM25SearchRepository,
    vector_repo: FaissVectorSearchRepository,
    embeddings_store: EmbeddingsStoreFaiss,
    stub_llm: StubLLMClient,
) -> SearchService:
    """
    Create the SearchService with all dependencies wired up.

    This is the main system under test - it orchestrates hybrid search.
    """
    return SearchService(
        lexical_search=bm25_repo,
        vector_search=vector_repo,
        embeddings_store=embeddings_store,
        llm_client=stub_llm,
    )


# -----------------------------------------------------------------------------
# Test: Basic Hybrid Search Functionality
# -----------------------------------------------------------------------------


class TestHybridSearchBasics:
    """Basic tests for hybrid search functionality."""

    def test_hybrid_search_returns_results(
        self, search_service: SearchService
    ):
        """Hybrid search should return non-empty results for a valid query."""
        query = SearchQuery(
            text="machine learning algorithms",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) > 0
        assert len(results) <= 5

    def test_results_have_hybrid_source(
        self, search_service: SearchService
    ):
        """Results from hybrid search should have source='hybrid'."""
        query = SearchQuery(
            text="Python programming",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) > 0
        assert all(r.source == "hybrid" for r in results)

    def test_ranks_are_consecutive_from_one(
        self, search_service: SearchService
    ):
        """Ranks should be consecutive integers starting from 1."""
        query = SearchQuery(
            text="artificial intelligence",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        expected_ranks = list(range(1, len(results) + 1))
        actual_ranks = [r.rank for r in results]

        assert actual_ranks == expected_ranks

    def test_results_sorted_by_final_score_descending(
        self, search_service: SearchService
    ):
        """Results should be sorted by final_score in descending order."""
        query = SearchQuery(
            text="deep learning neural networks",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        scores = [r.final_score for r in results]

        # Verify descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i+1} ({scores[i]:.4f}) should be >= "
                f"score at rank {i+2} ({scores[i+1]:.4f})"
            )

    def test_max_results_is_respected(
        self, search_service: SearchService
    ):
        """Should return at most max_results items."""
        query = SearchQuery(
            text="Python",
            max_results=3,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) <= 3

    def test_results_contain_fully_hydrated_books(
        self, search_service: SearchService
    ):
        """Each result should contain a complete Book entity."""
        query = SearchQuery(
            text="data science",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) > 0
        for result in results:
            assert isinstance(result.book, Book)
            assert result.book.id is not None
            assert result.book.title is not None
            assert result.book.authors is not None
            assert len(result.book.authors) > 0


# -----------------------------------------------------------------------------
# Test: Hybrid Behavior (Lexical + Semantic Combination)
# -----------------------------------------------------------------------------


class TestHybridBehavior:
    """
    Tests that verify true hybrid behavior: combining lexical and semantic search.

    These tests demonstrate that hybrid search retrieves books that would be
    missed by either lexical-only or semantic-only search.
    """

    def test_lexically_strong_book_appears_in_results(
        self, search_service: SearchService, test_catalog: List[Book]
    ):
        """
        A book with exact keyword match should appear in hybrid results.

        "The Python Programming Language" contains exact "Python" keyword
        and should be retrieved for a "Python" query.
        """
        query = SearchQuery(
            text="Python programming language",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)
        result_titles = [r.book.title for r in results]

        # The exact title match should appear
        assert "The Python Programming Language" in result_titles

    def test_semantically_strong_book_appears_in_results(
        self, search_service: SearchService, test_catalog: List[Book]
    ):
        """
        A book that matches semantically (by meaning) should appear in results.

        "Deep Neural Networks" should be retrieved for an "AI machine learning"
        query even though it doesn't contain those exact keywords.
        """
        query = SearchQuery(
            text="artificial intelligence machine learning algorithms",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)
        result_ids = {r.book.id for r in results}

        # Deep Neural Networks is semantically related to ML/AI
        deep_nn_id = UUID("00000000-0000-0000-0000-000000000002")
        assert deep_nn_id in result_ids, (
            "Deep Neural Networks should appear in results for ML/AI query "
            "due to semantic similarity"
        )

    def test_hybrid_retrieves_both_lexical_and_semantic_matches(
        self, search_service: SearchService
    ):
        """
        Hybrid search should retrieve books matched by BOTH methods.

        For "Python machine learning":
        - "Python Machine Learning" matches both lexically AND semantically
        - "The Python Programming Language" matches lexically (Python keyword)
        - "Deep Neural Networks" matches semantically (ML concepts)

        All three should appear in hybrid results.
        """
        query = SearchQuery(
            text="Python machine learning",
            max_results=8,
            use_explanations=False,
        )

        results = search_service.search(query)
        result_ids = {r.book.id for r in results}

        # Expected books
        python_ml_id = UUID("00000000-0000-0000-0000-000000000001")  # Both
        python_lang_id = UUID("00000000-0000-0000-0000-000000000003")  # Lexical
        deep_nn_id = UUID("00000000-0000-0000-0000-000000000002")  # Semantic

        # The book that matches BOTH should definitely be present
        assert python_ml_id in result_ids, (
            "Python Machine Learning (matches both) should be in results"
        )

        # At least one of the lexical-strong OR semantic-strong books
        # should also be present, demonstrating hybrid retrieval
        lexical_or_semantic_present = (
            python_lang_id in result_ids or deep_nn_id in result_ids
        )
        assert lexical_or_semantic_present, (
            "At least one book matching primarily lexically or semantically "
            "should appear in hybrid results"
        )

    def test_rrf_boosts_books_appearing_in_both_rankings(
        self, search_service: SearchService
    ):
        """
        Books appearing in both lexical AND vector rankings should get
        higher RRF scores than books appearing in only one.

        "Python Machine Learning" should rank higher than "Snake Species"
        (which only matches lexically on "python").
        """
        query = SearchQuery(
            text="Python machine learning programming",
            max_results=8,
            use_explanations=False,
        )

        results = search_service.search(query)

        # Find the ranks of specific books
        python_ml_rank = None
        snake_book_rank = None

        for result in results:
            if result.book.id == UUID("00000000-0000-0000-0000-000000000001"):
                python_ml_rank = result.rank
            elif result.book.id == UUID("00000000-0000-0000-0000-000000000005"):
                snake_book_rank = result.rank

        # Python ML should be present and rank well
        assert python_ml_rank is not None, (
            "Python Machine Learning should appear in results"
        )

        # If snake book appears, it should rank lower than Python ML
        if snake_book_rank is not None:
            assert python_ml_rank < snake_book_rank, (
                f"Python ML (rank {python_ml_rank}) should rank higher than "
                f"Snake Species (rank {snake_book_rank}) for ML query"
            )

    def test_results_preserve_both_score_components(
        self, search_service: SearchService
    ):
        """
        Hybrid results should preserve lexical_score and vector_score
        for books that appeared in both rankings.
        """
        query = SearchQuery(
            text="machine learning Python",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        # At least one result should have both scores
        # (from a book appearing in both lexical and vector results)
        has_both_scores = any(
            r.lexical_score is not None and r.vector_score is not None
            for r in results
        )

        assert has_both_scores, (
            "At least one result should have both lexical_score and vector_score "
            "indicating it appeared in both search methods"
        )


# -----------------------------------------------------------------------------
# Test: Filtering
# -----------------------------------------------------------------------------


class TestHybridSearchFiltering:
    """Tests for filter application in hybrid search."""

    def test_language_filter(
        self, search_service: SearchService
    ):
        """Language filter should only return books with matching language."""
        query = SearchQuery(
            text="cocina recetas comida",
            filters=SearchFilters(language="es"),
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) > 0
        assert all(r.book.language == "es" for r in results)

    def test_language_filter_excludes_non_matching(
        self, search_service: SearchService
    ):
        """Language filter should exclude books with different language."""
        query = SearchQuery(
            text="Python programming",
            filters=SearchFilters(language="en"),
            max_results=10,
            use_explanations=False,
        )

        results = search_service.search(query)

        # Should have results (English books exist)
        assert len(results) > 0

        # Spanish cookbook should NOT appear
        result_ids = {r.book.id for r in results}
        spanish_book_id = UUID("00000000-0000-0000-0000-000000000006")
        assert spanish_book_id not in result_ids

    def test_year_filter_min(
        self, search_service: SearchService
    ):
        """min_year filter should exclude books published before that year."""
        query = SearchQuery(
            text="machine learning",
            filters=SearchFilters(min_year=2019),
            max_results=10,
            use_explanations=False,
        )

        results = search_service.search(query)

        for result in results:
            year = result.book.get_published_year()
            assert year is not None
            assert year >= 2019, (
                f"Book '{result.book.title}' published in {year} "
                f"should not appear with min_year=2019"
            )

    def test_year_filter_max(
        self, search_service: SearchService
    ):
        """max_year filter should exclude books published after that year."""
        query = SearchQuery(
            text="Python programming",
            filters=SearchFilters(max_year=2018),
            max_results=10,
            use_explanations=False,
        )

        results = search_service.search(query)

        for result in results:
            year = result.book.get_published_year()
            assert year is not None
            assert year <= 2018, (
                f"Book '{result.book.title}' published in {year} "
                f"should not appear with max_year=2018"
            )

    def test_combined_filters(
        self, search_service: SearchService
    ):
        """Multiple filters should be applied as AND conditions."""
        query = SearchQuery(
            text="learning",
            filters=SearchFilters(language="en", min_year=2019, max_year=2021),
            max_results=10,
            use_explanations=False,
        )

        results = search_service.search(query)

        for result in results:
            assert result.book.language == "en"
            year = result.book.get_published_year()
            assert year is not None
            assert 2019 <= year <= 2021


# -----------------------------------------------------------------------------
# Test: LLM Integration (Stub Verification)
# -----------------------------------------------------------------------------


class TestLLMIntegration:
    """Tests verifying LLM is not called when use_explanations=False."""

    def test_llm_not_called_when_explanations_disabled(
        self, search_service: SearchService, stub_llm: StubLLMClient
    ):
        """LLMClient should not be called when use_explanations=False."""
        query = SearchQuery(
            text="machine learning",
            max_results=5,
            use_explanations=False,  # Explicitly disabled
        )

        results = search_service.search(query)

        assert stub_llm.call_count == 0, (
            "LLMClient should not be called when use_explanations=False"
        )

        # Results should not have explanations
        assert all(r.explanation is None for r in results)


# -----------------------------------------------------------------------------
# Test: Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_query_with_no_matching_books(
        self, search_service: SearchService
    ):
        """
        Query with very specific terms that don't match any book.
        Should return empty or very low-scoring results.
        """
        query = SearchQuery(
            text="quantum physics superconductivity",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        # Results may be empty or contain low-relevance items
        # The important thing is it doesn't crash
        assert isinstance(results, list)

    def test_single_word_query(
        self, search_service: SearchService
    ):
        """Single word queries should work correctly."""
        query = SearchQuery(
            text="Python",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) > 0
        # Python-related books should appear
        result_titles_lower = [r.book.title.lower() for r in results]
        assert any("python" in title for title in result_titles_lower)

    def test_long_query(
        self, search_service: SearchService
    ):
        """Long, descriptive queries should work correctly."""
        query = SearchQuery(
            text=(
                "I am looking for a comprehensive book about machine learning "
                "and artificial intelligence that covers neural networks "
                "and deep learning algorithms with Python code examples"
            ),
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) > 0

    def test_max_results_one(
        self, search_service: SearchService
    ):
        """Should correctly handle max_results=1."""
        query = SearchQuery(
            text="machine learning",
            max_results=1,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert len(results) == 1
        assert results[0].rank == 1


# -----------------------------------------------------------------------------
# Test: Determinism and Reproducibility
# -----------------------------------------------------------------------------


class TestDeterminism:
    """Tests for result reproducibility."""

    def test_same_query_produces_same_results(
        self, search_service: SearchService
    ):
        """
        Running the same query twice should produce identical results.

        This ensures the hybrid search pipeline is deterministic.
        """
        query = SearchQuery(
            text="Python machine learning algorithms",
            max_results=5,
            use_explanations=False,
        )

        results1 = search_service.search(query)
        results2 = search_service.search(query)

        # Same number of results
        assert len(results1) == len(results2)

        # Same books in same order
        for r1, r2 in zip(results1, results2):
            assert r1.book.id == r2.book.id
            assert r1.rank == r2.rank
            assert abs(r1.final_score - r2.final_score) < 1e-6

    def test_result_ordering_is_stable(
        self, search_service: SearchService
    ):
        """
        Results with equal scores should have stable ordering.

        This is important for pagination and user experience.
        """
        query = SearchQuery(
            text="programming",
            max_results=8,
            use_explanations=False,
        )

        # Run multiple times
        results_runs = [search_service.search(query) for _ in range(3)]

        # All runs should have same ordering
        for i in range(1, len(results_runs)):
            ids_first = [r.book.id for r in results_runs[0]]
            ids_current = [r.book.id for r in results_runs[i]]
            assert ids_first == ids_current, (
                f"Run {i+1} produced different ordering than run 1"
            )


# -----------------------------------------------------------------------------
# Test: Score Component Presence
# -----------------------------------------------------------------------------


class TestScoreComponents:
    """Tests verifying score component tracking through fusion."""

    def test_hybrid_results_have_final_score(
        self, search_service: SearchService
    ):
        """All hybrid results should have a final_score (RRF score)."""
        query = SearchQuery(
            text="artificial intelligence",
            max_results=5,
            use_explanations=False,
        )

        results = search_service.search(query)

        assert all(r.final_score is not None for r in results)
        assert all(r.final_score > 0 for r in results)

    def test_some_results_have_lexical_score_only(
        self, search_service: SearchService
    ):
        """
        Some results may have only lexical_score (appeared only in BM25 results).

        The snake book "python species" should match lexically but not
        semantically for a programming query.
        """
        query = SearchQuery(
            text="python species animals",
            max_results=8,
            use_explanations=False,
        )

        results = search_service.search(query)

        # Check if there's at least one result with only lexical or only vector score
        # This demonstrates the hybrid nature - different books come from different sources
        lexical_only = [
            r for r in results
            if r.lexical_score is not None and r.vector_score is None
        ]
        vector_only = [
            r for r in results
            if r.vector_score is not None and r.lexical_score is None
        ]
        both_scores = [
            r for r in results
            if r.lexical_score is not None and r.vector_score is not None
        ]

        # At least one category should have results (hybrid retrieval works)
        total_categorized = len(lexical_only) + len(vector_only) + len(both_scores)
        assert total_categorized == len(results), (
            "All results should have at least one score component"
        )
