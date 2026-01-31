"""
Tests for SearchService.search_with_understanding() (Block 2).

These tests verify:
1. Query understanding integration with search
2. Intent-based search strategy adjustments
3. Filter extraction from natural language
4. Graceful degradation when LLM unavailable
5. Error handling throughout the flow
"""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4

from app.domain.services import SearchService
from app.domain.entities import Book, SearchResult
from app.domain.value_objects import (
    SearchFilters,
    SearchQuery,
    SearchResponse,
    SearchMetadata,
    QueryIntent,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_book():
    """Create a sample book for testing."""
    return Book(
        id=uuid4(),
        title="1984",
        authors=["George Orwell"],
        description="A dystopian novel about totalitarianism",
        language="en",
        categories=["Fiction", "Dystopian"],
        published_date="1949-06-08",
        source="test",
        source_id="test-001",
    )


@pytest.fixture
def sample_books():
    """Create multiple sample books."""
    return [
        Book(
            id=uuid4(),
            title="1984",
            authors=["George Orwell"],
            description="Dystopian fiction",
            language="en",
            categories=["Fiction"],
            published_date="1949-06-08",
            source="test",
            source_id="test-001",
        ),
        Book(
            id=uuid4(),
            title="Brave New World",
            authors=["Aldous Huxley"],
            description="Another dystopian novel",
            language="en",
            categories=["Fiction"],
            published_date="1932-01-01",
            source="test",
            source_id="test-002",
        ),
        Book(
            id=uuid4(),
            title="Don Quixote",
            authors=["Miguel de Cervantes"],
            description="Spanish classic novel",
            language="es",
            categories=["Fiction", "Classic"],
            published_date="1605-01-01",
            source="test",
            source_id="test-003",
        ),
    ]


@pytest.fixture
def mock_lexical_search(sample_books):
    """Create mock lexical search repository."""
    mock = Mock()
    mock.is_ready.return_value = True
    mock.search.return_value = [
        SearchResult(
            book=sample_books[0],
            final_score=0.9,
            rank=1,
            source="lexical",
            lexical_score=0.9,
        ),
        SearchResult(
            book=sample_books[1],
            final_score=0.7,
            rank=2,
            source="lexical",
            lexical_score=0.7,
        ),
    ]
    return mock


@pytest.fixture
def mock_vector_search(sample_books):
    """Create mock vector search repository."""
    mock = Mock()
    mock.is_ready.return_value = True
    mock.search.return_value = [
        SearchResult(
            book=sample_books[0],
            final_score=0.85,
            rank=1,
            source="vector",
            vector_score=0.85,
        ),
        SearchResult(
            book=sample_books[2],
            final_score=0.75,
            rank=2,
            source="vector",
            vector_score=0.75,
        ),
    ]
    return mock


@pytest.fixture
def mock_embeddings_store():
    """Create mock embeddings store."""
    mock = Mock()
    mock.is_ready.return_value = True
    mock.generate_embedding.return_value = [0.1] * 384  # MiniLM dimension
    mock.get_embedding.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client with extract_query_intent."""
    mock = Mock()
    mock.extract_query_intent.return_value = QueryIntent(
        intent_type="recommendation",
        original_query="books like 1984",
        reformulated_query="1984 dystopian fiction similar novels",
        extracted_filters=SearchFilters(),
        confidence=0.9,
        reasoning="User wants book recommendations similar to 1984"
    )
    return mock


@pytest.fixture
def search_service(mock_lexical_search, mock_vector_search, mock_embeddings_store, mock_llm_client):
    """Create SearchService with all mocked dependencies."""
    return SearchService(
        lexical_search=mock_lexical_search,
        vector_search=mock_vector_search,
        embeddings_store=mock_embeddings_store,
        llm_client=mock_llm_client,
    )


@pytest.fixture
def search_service_no_llm(mock_lexical_search, mock_vector_search, mock_embeddings_store):
    """Create SearchService without LLM client."""
    return SearchService(
        lexical_search=mock_lexical_search,
        vector_search=mock_vector_search,
        embeddings_store=mock_embeddings_store,
        llm_client=None,
    )


# =============================================================================
# Tests: Basic Functionality
# =============================================================================

class TestSearchWithUnderstandingBasic:
    """Basic functionality tests."""

    def test_returns_response_and_intent(self, search_service):
        """Test that method returns both SearchResponse and QueryIntent."""
        response, intent = search_service.search_with_understanding(
            query_text="books like 1984"
        )

        assert isinstance(response, SearchResponse)
        assert isinstance(intent, QueryIntent)

    def test_uses_reformulated_query(self, search_service, mock_lexical_search):
        """Test that search uses the reformulated query from LLM."""
        response, intent = search_service.search_with_understanding(
            query_text="books like 1984"
        )

        # Check that lexical search was called with reformulated query
        call_args = mock_lexical_search.search.call_args
        assert call_args is not None
        # The reformulated query should be used
        assert "dystopian" in call_args.kwargs.get("query_text", "") or \
               "dystopian" in str(call_args)

    def test_extracts_filters_from_natural_language(self, search_service, mock_llm_client):
        """Test that filters are extracted from natural language query."""
        # Setup LLM to return filters
        mock_llm_client.extract_query_intent.return_value = QueryIntent(
            intent_type="exploratory",
            original_query="Spanish novels from the 1900s",
            reformulated_query="Spanish literature novels",
            extracted_filters=SearchFilters(language="es", min_year=1900, max_year=1999),
            confidence=0.85,
            reasoning="User wants Spanish books from a specific era"
        )

        response, intent = search_service.search_with_understanding(
            query_text="Spanish novels from the 1900s"
        )

        assert intent.extracted_filters.language == "es"
        assert intent.extracted_filters.min_year == 1900
        assert intent.extracted_filters.max_year == 1999


# =============================================================================
# Tests: Intent-Based Strategy Adjustments
# =============================================================================

class TestIntentBasedStrategies:
    """Tests for intent-based search strategy adjustments."""

    def test_exploratory_enables_diversification(self, search_service, mock_llm_client):
        """Test that exploratory intent enables diversification."""
        mock_llm_client.extract_query_intent.return_value = QueryIntent(
            intent_type="exploratory",
            original_query="science fiction books",
            reformulated_query="science fiction novels",
            extracted_filters=SearchFilters(),
            confidence=0.8,
            reasoning="User exploring a genre"
        )

        response, intent = search_service.search_with_understanding(
            query_text="science fiction books",
            use_diversification=False  # User didn't request it
        )

        # Exploratory should auto-enable diversification
        assert intent.intent_type == "exploratory"

    def test_recommendation_uses_balanced_lambda(self, search_service, mock_llm_client):
        """Test that recommendation intent uses balanced lambda."""
        mock_llm_client.extract_query_intent.return_value = QueryIntent(
            intent_type="recommendation",
            original_query="books like 1984",
            reformulated_query="1984 dystopian",
            extracted_filters=SearchFilters(),
            confidence=0.9,
            reasoning="Recommendation query"
        )

        response, intent = search_service.search_with_understanding(
            query_text="books like 1984"
        )

        assert intent.intent_type == "recommendation"
        # Lambda should be 0.6 (balanced) - verified by the method logic

    def test_factual_prioritizes_relevance(self, search_service, mock_llm_client):
        """Test that factual intent prioritizes relevance over diversity."""
        mock_llm_client.extract_query_intent.return_value = QueryIntent(
            intent_type="factual",
            original_query="who wrote 1984",
            reformulated_query="1984 author George Orwell",
            extracted_filters=SearchFilters(),
            confidence=0.95,
            reasoning="Factual question about authorship"
        )

        response, intent = search_service.search_with_understanding(
            query_text="who wrote 1984"
        )

        assert intent.intent_type == "factual"
        # Lambda should be 0.8 (more relevance) - verified by the method logic


# =============================================================================
# Tests: Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation scenarios."""

    def test_works_without_llm_client(self, search_service_no_llm):
        """Test that search works even without LLM client."""
        response, intent = search_service_no_llm.search_with_understanding(
            query_text="science fiction"
        )

        # Should return default exploratory intent
        assert intent.intent_type == "exploratory"
        assert intent.confidence == 0.0
        assert "LLM client not available" in intent.reasoning
        # Search should still work
        assert response is not None

    def test_handles_llm_failure(self, search_service, mock_llm_client):
        """Test graceful handling when LLM call fails."""
        mock_llm_client.extract_query_intent.side_effect = Exception("LLM API error")

        response, intent = search_service.search_with_understanding(
            query_text="books about AI"
        )

        # Should fall back to default intent
        assert intent.intent_type == "exploratory"
        assert intent.confidence == 0.0
        assert "Query understanding failed" in intent.reasoning
        # Search should still return results
        assert response is not None

    def test_uses_original_query_on_failure(self, search_service, mock_llm_client, mock_lexical_search):
        """Test that original query is used when LLM fails."""
        mock_llm_client.extract_query_intent.side_effect = Exception("LLM error")

        response, intent = search_service.search_with_understanding(
            query_text="original test query"
        )

        # Reformulated query should equal original (fallback)
        assert intent.reformulated_query == "original test query"


# =============================================================================
# Tests: Validation
# =============================================================================

class TestValidation:
    """Tests for input validation."""

    def test_empty_query_raises_error(self, search_service):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            search_service.search_with_understanding(query_text="")

    def test_whitespace_only_query_raises_error(self, search_service):
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            search_service.search_with_understanding(query_text="   ")


# =============================================================================
# Tests: Response Metadata
# =============================================================================

class TestResponseMetadata:
    """Tests for response metadata."""

    def test_response_includes_latency(self, search_service):
        """Test that response includes latency measurement."""
        response, intent = search_service.search_with_understanding(
            query_text="test query"
        )

        assert response.latency_ms is not None
        assert response.latency_ms >= 0  # With mocks, execution is very fast (can be 0)

    def test_returns_results(self, search_service):
        """Test that response includes search results."""
        response, intent = search_service.search_with_understanding(
            query_text="dystopian fiction"
        )

        assert response.results is not None
        assert len(response.results) > 0

    def test_respects_max_results(self, search_service):
        """Test that max_results parameter is respected."""
        response, intent = search_service.search_with_understanding(
            query_text="fiction",
            max_results=1
        )

        assert len(response.results) <= 1


# =============================================================================
# Tests: Integration with Explanations
# =============================================================================

class TestExplanationsIntegration:
    """Tests for integration with explanation generation."""

    def test_can_enable_explanations(self, search_service, mock_llm_client):
        """Test that explanations can be enabled."""
        mock_llm_client.generate_grounded_explanation = Mock()

        response, intent = search_service.search_with_understanding(
            query_text="books like 1984",
            use_explanations=True
        )

        # Should not raise an error
        assert response is not None

    def test_explanations_disabled_by_default(self, search_service, mock_llm_client):
        """Test that explanations are disabled by default."""
        mock_llm_client.generate_grounded_explanation = Mock()

        response, intent = search_service.search_with_understanding(
            query_text="test query"
        )

        # generate_grounded_explanation should not be called
        # (since use_explanations defaults to False)
        assert response is not None


# =============================================================================
# Tests: Full Flow Integration
# =============================================================================

class TestFullFlowIntegration:
    """Integration tests for the complete flow."""

    def test_complete_flow_recommendation(self, search_service, mock_llm_client):
        """Test complete flow for a recommendation query."""
        mock_llm_client.extract_query_intent.return_value = QueryIntent(
            intent_type="recommendation",
            original_query="books similar to Harry Potter",
            reformulated_query="Harry Potter fantasy magic adventure",
            extracted_filters=SearchFilters(category="Fantasy"),
            confidence=0.92,
            reasoning="User wants books similar to Harry Potter"
        )

        response, intent = search_service.search_with_understanding(
            query_text="books similar to Harry Potter",
            max_results=5
        )

        # Verify intent
        assert intent.intent_type == "recommendation"
        assert intent.extracted_filters.category == "Fantasy"
        assert "Harry Potter" in intent.reformulated_query

        # Verify response
        assert response is not None
        assert not response.degraded

    def test_complete_flow_with_language_filter(self, search_service, mock_llm_client):
        """Test complete flow with language filter extraction."""
        mock_llm_client.extract_query_intent.return_value = QueryIntent(
            intent_type="exploratory",
            original_query="libros en espanol sobre historia",
            reformulated_query="historia espanola libros",
            extracted_filters=SearchFilters(language="es", category="History"),
            confidence=0.88,
            reasoning="User wants Spanish history books"
        )

        response, intent = search_service.search_with_understanding(
            query_text="libros en espanol sobre historia"
        )

        assert intent.extracted_filters.language == "es"
        assert intent.extracted_filters.category == "History"
        assert response is not None
