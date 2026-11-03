"""
Tests for SearchService explanation generation integration.

These tests verify that SearchService correctly integrates with LLMClient
to generate explanations for search results when requested.
"""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4
from datetime import datetime, UTC

from app.domain.entities import Book, SearchResult, Explanation, Citation
from app.domain.value_objects import SearchQuery, SearchFilters
from app.domain.services import SearchService


# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture
def sample_books():
    """Create a list of sample books for testing."""
    return [
        Book(
            id=uuid4(),
            title=f"Test Book {i}",
            authors=[f"Author {i}"],
            description=f"Description for book {i}",
            language="en",
            categories=["Fiction"],
            published_date="2020-01-01",
            source="test",
            source_id=f"test-{i}",
            metadata={}
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_search_results(sample_books):
    """Create sample search results."""
    return [
        SearchResult(
            book=book,
            final_score=1.0 - (i * 0.1),
            rank=i + 1,
            source="hybrid",
            lexical_score=0.5,
            vector_score=0.5,
        )
        for i, book in enumerate(sample_books)
    ]


@pytest.fixture
def mock_lexical_search():
    """Mock LexicalSearchRepository."""
    mock = Mock()
    mock.search = Mock(return_value=[])
    mock.is_ready = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_vector_search():
    """Mock VectorSearchRepository."""
    mock = Mock()
    mock.search = Mock(return_value=[])
    mock.is_ready = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_embeddings_store():
    """Mock EmbeddingsStore."""
    mock = Mock()
    mock.generate_embedding = Mock(return_value=[0.1] * 384)  # Fake embedding
    mock.get_embedding = Mock(return_value=[0.1] * 384)
    mock.is_ready = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_llm_client():
    """
    Mock LLMClient that returns a valid Explanation with citations.
    """
    mock = Mock()

    def generate_explanation(query_text: str, book: Book) -> Explanation:
        """Mock explanation generation."""
        return Explanation(
            book_id=book.id,
            query_text=query_text,
            text=f"This book '{book.title}' is relevant to your query.",
            citations=[
                Citation(
                    book_id=book.id,
                    chunk_id="title",
                    snippet=book.title[:20],
                    relevance_score=0.9
                )
            ],
            model="mock-model",
            created_at=datetime.now(UTC)
        )

    mock.generate_grounded_explanation = Mock(side_effect=generate_explanation)
    return mock


@pytest.fixture
def search_service_with_llm(
    mock_lexical_search,
    mock_vector_search,
    mock_embeddings_store,
    mock_llm_client
):
    """Create SearchService with mocked LLMClient."""
    return SearchService(
        lexical_search=mock_lexical_search,
        vector_search=mock_vector_search,
        embeddings_store=mock_embeddings_store,
        llm_client=mock_llm_client,
    )


@pytest.fixture
def search_service_without_llm(
    mock_lexical_search,
    mock_vector_search,
    mock_embeddings_store
):
    """Create SearchService WITHOUT LLMClient (llm_client=None)."""
    return SearchService(
        lexical_search=mock_lexical_search,
        vector_search=mock_vector_search,
        embeddings_store=mock_embeddings_store,
        llm_client=None,  # No LLM client
    )


# ================================================================================
# HELPER: Setup mocks to return specific results
# ================================================================================

def setup_search_mocks_with_results(
    mock_lexical_search,
    mock_vector_search,
    sample_search_results
):
    """Configure mocks to return sample results."""
    # Return first 5 results from lexical, last 5 from vector (for RRF fusion testing)
    mock_lexical_search.search.return_value = sample_search_results[:5]
    mock_vector_search.search.return_value = sample_search_results[5:]


# ================================================================================
# TEST: use_explanations=True with LLMClient available
# ================================================================================

def test_search_with_explanations_enabled_calls_llm_for_top5(
    search_service_with_llm,
    mock_lexical_search,
    mock_vector_search,
    mock_llm_client,
    sample_search_results
):
    """
    Test that when use_explanations=True and llm_client is available,
    generate_grounded_explanation is called exactly min(5, len(results)) times.
    """
    # Setup: Return 10 results
    setup_search_mocks_with_results(
        mock_lexical_search,
        mock_vector_search,
        sample_search_results
    )

    query = SearchQuery(
        text="test query",
        use_explanations=True,  # Enable explanations
        max_results=10
    )

    # Execute
    results = search_service_with_llm.search(query)

    # Verify: LLM should be called exactly 5 times (top-5)
    assert mock_llm_client.generate_grounded_explanation.call_count == 5

    # Verify: First 5 results should have explanations
    for i in range(5):
        assert results[i].explanation is not None
        assert isinstance(results[i].explanation, Explanation)
        assert results[i].explanation.book_id == results[i].book.id

    # Verify: Results 6-10 should NOT have explanations (only top-5 get explanations)
    for i in range(5, len(results)):
        assert results[i].explanation is None


def test_search_with_explanations_enabled_fewer_than_5_results(
    search_service_with_llm,
    mock_lexical_search,
    mock_vector_search,
    mock_llm_client,
    sample_search_results
):
    """
    Test that when use_explanations=True but results < 5,
    LLM is called len(results) times (not 5).
    """
    # Setup: Return only 3 results
    three_results = sample_search_results[:3]
    setup_search_mocks_with_results(
        mock_lexical_search,
        mock_vector_search,
        three_results
    )

    query = SearchQuery(
        text="test query",
        use_explanations=True,
        max_results=3
    )

    # Execute
    results = search_service_with_llm.search(query)

    # Verify: LLM called exactly 3 times (not 5)
    assert mock_llm_client.generate_grounded_explanation.call_count == 3

    # All 3 results should have explanations
    for result in results:
        assert result.explanation is not None


# ================================================================================
# TEST: use_explanations=False does NOT call LLM
# ================================================================================

def test_search_with_explanations_disabled_does_not_call_llm(
    search_service_with_llm,
    mock_lexical_search,
    mock_vector_search,
    mock_llm_client,
    sample_search_results
):
    """
    Test that when use_explanations=False, LLM is NOT called.
    """
    setup_search_mocks_with_results(
        mock_lexical_search,
        mock_vector_search,
        sample_search_results
    )

    query = SearchQuery(
        text="test query",
        use_explanations=False,  # Disable explanations
        max_results=10
    )

    # Execute
    results = search_service_with_llm.search(query)

    # Verify: LLM should NOT be called
    assert mock_llm_client.generate_grounded_explanation.call_count == 0

    # Verify: No results have explanations
    for result in results:
        assert result.explanation is None


# ================================================================================
# TEST: llm_client=None does NOT crash
# ================================================================================

def test_search_with_explanations_enabled_but_no_llm_client(
    search_service_without_llm,
    mock_lexical_search,
    mock_vector_search,
    sample_search_results
):
    """
    Test that when use_explanations=True but llm_client=None,
    the search does NOT crash and returns results without explanations.
    """
    setup_search_mocks_with_results(
        mock_lexical_search,
        mock_vector_search,
        sample_search_results
    )

    query = SearchQuery(
        text="test query",
        use_explanations=True,  # Request explanations
        max_results=10
    )

    # Execute - should NOT crash
    results = search_service_without_llm.search(query)

    # Verify: Results returned successfully
    assert len(results) > 0

    # Verify: No results have explanations (llm_client was None)
    for result in results:
        assert result.explanation is None


# ================================================================================
# TEST: LLM failure handling (one explanation fails, others succeed)
# ================================================================================

def test_search_with_explanations_partial_llm_failure(
    search_service_with_llm,
    mock_lexical_search,
    mock_vector_search,
    mock_llm_client,
    sample_search_results
):
    """
    Test that when LLM fails for ONE result, the search continues
    and other results still get explanations.
    """
    setup_search_mocks_with_results(
        mock_lexical_search,
        mock_vector_search,
        sample_search_results
    )

    # Setup: Make LLM fail on the 3rd call
    call_count = [0]

    def generate_with_failure(query_text: str, book: Book) -> Explanation:
        call_count[0] += 1
        if call_count[0] == 3:
            raise RuntimeError("Simulated LLM failure")

        return Explanation(
            book_id=book.id,
            query_text=query_text,
            text=f"Explanation for {book.title}",
            citations=[],
            model="mock-model",
            created_at=datetime.now(UTC)
        )

    mock_llm_client.generate_grounded_explanation = Mock(side_effect=generate_with_failure)

    query = SearchQuery(
        text="test query",
        use_explanations=True,
        max_results=10
    )

    # Execute - should NOT crash despite one failure
    results = search_service_with_llm.search(query)

    # Verify: LLM was called 5 times (top-5)
    assert mock_llm_client.generate_grounded_explanation.call_count == 5

    # Verify: 4 results have explanations (3rd failed)
    explanations_count = sum(1 for r in results[:5] if r.explanation is not None)
    assert explanations_count == 4


# ================================================================================
# TEST: Explanation object structure
# ================================================================================

def test_search_explanation_has_correct_structure(
    search_service_with_llm,
    mock_lexical_search,
    mock_vector_search,
    mock_llm_client,
    sample_search_results
):
    """
    Test that explanations have the correct structure (Explanation entity with citations).
    """
    setup_search_mocks_with_results(
        mock_lexical_search,
        mock_vector_search,
        sample_search_results
    )

    query = SearchQuery(
        text="test query",
        use_explanations=True,
        max_results=10
    )

    results = search_service_with_llm.search(query)

    # Check first result's explanation structure
    first_explanation = results[0].explanation

    assert first_explanation is not None
    assert isinstance(first_explanation, Explanation)
    assert first_explanation.book_id == results[0].book.id
    assert first_explanation.query_text == "test query"
    assert len(first_explanation.citations) > 0  # Mock returns 1 citation
    assert isinstance(first_explanation.citations[0], Citation)
    assert first_explanation.model == "mock-model"


# ================================================================================
# TEST: Integration with search_with_fallback()
# ================================================================================

def test_search_with_fallback_generates_explanations(
    search_service_with_llm,
    mock_lexical_search,
    mock_vector_search,
    mock_llm_client,
    sample_search_results
):
    """
    Test that search_with_fallback() also generates explanations when enabled.
    """
    setup_search_mocks_with_results(
        mock_lexical_search,
        mock_vector_search,
        sample_search_results
    )

    query = SearchQuery(
        text="test query",
        use_explanations=True,
        max_results=10
    )

    # Execute using search_with_fallback
    response = search_service_with_llm.search_with_fallback(query)

    # Verify: LLM was called for top-5
    assert mock_llm_client.generate_grounded_explanation.call_count == 5

    # Verify: First 5 results have explanations
    for i in range(5):
        assert response.results[i].explanation is not None
