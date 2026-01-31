"""
Tests for the LangGraph query understanding flow (Block 2).

These tests verify:
1. Individual node functions with mocked LLM
2. Routing logic based on intent
3. Graph assembly and compilation
4. Error handling and graceful degradation
"""

import pytest
from unittest.mock import Mock, MagicMock

from app.infrastructure.llm.graphs.query_understanding import (
    QueryUnderstandingState,
    parse_intent_node,
    extract_filters_node,
    reformulate_query_node,
    handle_recommendation_node,
    handle_factual_node,
    handle_exploratory_node,
    route_by_intent,
    create_query_understanding_graph,
)
from app.infrastructure.llm.schemas_query import (
    QueryIntentLLM,
    ExtractedFiltersLLM,
    ReformulatedQueryLLM,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM that supports with_structured_output."""
    llm = Mock()
    return llm


@pytest.fixture
def sample_intent_result():
    """Sample QueryIntentLLM result."""
    return QueryIntentLLM(
        intent_type="recommendation",
        confidence=0.9,
        reasoning="User wants similar books"
    )


@pytest.fixture
def sample_filters_result():
    """Sample ExtractedFiltersLLM result."""
    return ExtractedFiltersLLM(
        language="es",
        category="Fiction",
        min_year=2000,
        max_year=2020,
        author_hint="Cervantes"
    )


@pytest.fixture
def sample_reformulated_result():
    """Sample ReformulatedQueryLLM result."""
    return ReformulatedQueryLLM(
        optimized_query="Don Quixote Spanish literature",
        keywords=["Don Quixote", "Spanish", "literature", "classic"],
        removed_noise=["I want", "please"]
    )


@pytest.fixture
def base_state():
    """Base state for testing nodes."""
    return QueryUnderstandingState(
        original_query="books like Don Quixote in Spanish",
        intent=None,
        filters=None,
        reformulated=None,
        final_query=None,
        error=None
    )


# =============================================================================
# Tests: parse_intent_node
# =============================================================================

class TestParseIntentNode:
    """Tests for the intent parsing node."""

    def test_parse_intent_success(self, mock_llm, sample_intent_result, base_state):
        """Test successful intent parsing."""
        # Setup mock
        mock_structured = Mock()
        mock_structured.invoke.return_value = sample_intent_result
        mock_llm.with_structured_output.return_value = mock_structured

        # Execute
        result = parse_intent_node(base_state, mock_llm)

        # Assert
        assert "intent" in result
        assert result["intent"] == sample_intent_result
        assert result["intent"].intent_type == "recommendation"
        assert result["intent"].confidence == 0.9

    def test_parse_intent_failure_graceful_degradation(self, mock_llm, base_state):
        """Test graceful degradation when intent parsing fails."""
        # Setup mock to raise exception
        mock_llm.with_structured_output.side_effect = Exception("LLM error")

        # Execute
        result = parse_intent_node(base_state, mock_llm)

        # Assert graceful degradation
        assert result["intent"] is None
        assert "error" in result
        assert "Intent parsing failed" in result["error"]


# =============================================================================
# Tests: extract_filters_node
# =============================================================================

class TestExtractFiltersNode:
    """Tests for the filter extraction node."""

    def test_extract_filters_success(self, mock_llm, sample_filters_result, base_state):
        """Test successful filter extraction."""
        mock_structured = Mock()
        mock_structured.invoke.return_value = sample_filters_result
        mock_llm.with_structured_output.return_value = mock_structured

        result = extract_filters_node(base_state, mock_llm)

        assert "filters" in result
        assert result["filters"].language == "es"
        assert result["filters"].category == "Fiction"
        assert result["filters"].min_year == 2000

    def test_extract_filters_failure_graceful_degradation(self, mock_llm, base_state):
        """Test graceful degradation when filter extraction fails."""
        mock_llm.with_structured_output.side_effect = Exception("LLM error")

        result = extract_filters_node(base_state, mock_llm)

        assert result["filters"] is None
        assert "error" in result
        assert "Filter extraction failed" in result["error"]


# =============================================================================
# Tests: reformulate_query_node
# =============================================================================

class TestReformulateQueryNode:
    """Tests for the query reformulation node."""

    def test_reformulate_query_success(self, mock_llm, sample_reformulated_result, base_state):
        """Test successful query reformulation."""
        mock_structured = Mock()
        mock_structured.invoke.return_value = sample_reformulated_result
        mock_llm.with_structured_output.return_value = mock_structured

        result = reformulate_query_node(base_state, mock_llm)

        assert "reformulated" in result
        assert result["reformulated"].optimized_query == "Don Quixote Spanish literature"
        assert "Don Quixote" in result["reformulated"].keywords

    def test_reformulate_query_failure_graceful_degradation(self, mock_llm, base_state):
        """Test graceful degradation when reformulation fails."""
        mock_llm.with_structured_output.side_effect = Exception("LLM error")

        result = reformulate_query_node(base_state, mock_llm)

        assert result["reformulated"] is None
        assert "error" in result
        assert "Query reformulation failed" in result["error"]


# =============================================================================
# Tests: Intent Handler Nodes
# =============================================================================

class TestHandleRecommendationNode:
    """Tests for the recommendation intent handler."""

    def test_handle_recommendation_with_reformulated(self, sample_reformulated_result):
        """Test recommendation handler uses reformulated query."""
        state = QueryUnderstandingState(
            original_query="original query",
            intent=None,
            filters=None,
            reformulated=sample_reformulated_result,
            final_query=None,
            error=None
        )

        result = handle_recommendation_node(state)

        assert result["final_query"] == "Don Quixote Spanish literature"

    def test_handle_recommendation_fallback_to_original(self):
        """Test recommendation handler falls back to original when no reformulation."""
        state = QueryUnderstandingState(
            original_query="original query",
            intent=None,
            filters=None,
            reformulated=None,
            final_query=None,
            error=None
        )

        result = handle_recommendation_node(state)

        assert result["final_query"] == "original query"


class TestHandleFactualNode:
    """Tests for the factual intent handler."""

    def test_handle_factual_with_reformulated(self, sample_reformulated_result):
        """Test factual handler uses reformulated query."""
        state = QueryUnderstandingState(
            original_query="who wrote Don Quixote",
            intent=None,
            filters=None,
            reformulated=sample_reformulated_result,
            final_query=None,
            error=None
        )

        result = handle_factual_node(state)

        assert result["final_query"] == "Don Quixote Spanish literature"

    def test_handle_factual_fallback_to_original(self):
        """Test factual handler falls back to original when no reformulation."""
        state = QueryUnderstandingState(
            original_query="who wrote Don Quixote",
            intent=None,
            filters=None,
            reformulated=None,
            final_query=None,
            error=None
        )

        result = handle_factual_node(state)

        assert result["final_query"] == "who wrote Don Quixote"


class TestHandleExploratoryNode:
    """Tests for the exploratory intent handler."""

    def test_handle_exploratory_expands_with_keywords(self, sample_reformulated_result):
        """Test exploratory handler expands query with keywords."""
        state = QueryUnderstandingState(
            original_query="Spanish literature",
            intent=None,
            filters=None,
            reformulated=sample_reformulated_result,
            final_query=None,
            error=None
        )

        result = handle_exploratory_node(state)

        # Should combine optimized query with top keywords
        assert "Don Quixote Spanish literature" in result["final_query"]
        assert "Don Quixote" in result["final_query"]

    def test_handle_exploratory_no_keywords(self):
        """Test exploratory handler without keywords uses optimized query."""
        reformulated = ReformulatedQueryLLM(
            optimized_query="optimized query",
            keywords=[],
            removed_noise=[]
        )
        state = QueryUnderstandingState(
            original_query="original",
            intent=None,
            filters=None,
            reformulated=reformulated,
            final_query=None,
            error=None
        )

        result = handle_exploratory_node(state)

        assert result["final_query"] == "optimized query"

    def test_handle_exploratory_fallback_to_original(self):
        """Test exploratory handler falls back to original when no reformulation."""
        state = QueryUnderstandingState(
            original_query="explore science fiction",
            intent=None,
            filters=None,
            reformulated=None,
            final_query=None,
            error=None
        )

        result = handle_exploratory_node(state)

        assert result["final_query"] == "explore science fiction"


# =============================================================================
# Tests: Routing
# =============================================================================

class TestRouteByIntent:
    """Tests for the conditional routing function."""

    def test_route_recommendation(self):
        """Test routing to recommendation handler."""
        intent = QueryIntentLLM(
            intent_type="recommendation",
            confidence=0.9,
            reasoning="User wants similar books"
        )
        state = QueryUnderstandingState(
            original_query="query",
            intent=intent,
            filters=None,
            reformulated=None,
            final_query=None,
            error=None
        )

        result = route_by_intent(state)

        assert result == "recommendation"

    def test_route_factual(self):
        """Test routing to factual handler."""
        intent = QueryIntentLLM(
            intent_type="factual",
            confidence=0.85,
            reasoning="User asking factual question"
        )
        state = QueryUnderstandingState(
            original_query="query",
            intent=intent,
            filters=None,
            reformulated=None,
            final_query=None,
            error=None
        )

        result = route_by_intent(state)

        assert result == "factual"

    def test_route_exploratory(self):
        """Test routing to exploratory handler."""
        intent = QueryIntentLLM(
            intent_type="exploratory",
            confidence=0.7,
            reasoning="User exploring a topic"
        )
        state = QueryUnderstandingState(
            original_query="query",
            intent=intent,
            filters=None,
            reformulated=None,
            final_query=None,
            error=None
        )

        result = route_by_intent(state)

        assert result == "exploratory"

    def test_route_fallback_when_intent_none(self):
        """Test fallback to exploratory when intent is None."""
        state = QueryUnderstandingState(
            original_query="query",
            intent=None,  # Intent parsing failed
            filters=None,
            reformulated=None,
            final_query=None,
            error="Intent parsing failed"
        )

        result = route_by_intent(state)

        assert result == "exploratory"


# =============================================================================
# Tests: Graph Builder
# =============================================================================

class TestCreateQueryUnderstandingGraph:
    """Tests for the graph builder function."""

    def test_graph_compiles_successfully(self, mock_llm):
        """Test that the graph compiles without errors."""
        graph = create_query_understanding_graph(mock_llm)

        # Graph should be a compiled StateGraph
        assert graph is not None
        assert hasattr(graph, 'invoke')

    def test_graph_full_flow_success(self, mock_llm, sample_intent_result,
                                      sample_filters_result, sample_reformulated_result):
        """Test complete flow through the graph with mocked LLM."""
        # Setup mock to return different results for different calls
        mock_structured_intent = Mock()
        mock_structured_intent.invoke.return_value = sample_intent_result

        mock_structured_filters = Mock()
        mock_structured_filters.invoke.return_value = sample_filters_result

        mock_structured_reformulated = Mock()
        mock_structured_reformulated.invoke.return_value = sample_reformulated_result

        # Return different mocks based on the schema
        def mock_with_structured_output(schema):
            if schema == QueryIntentLLM:
                return mock_structured_intent
            elif schema == ExtractedFiltersLLM:
                return mock_structured_filters
            elif schema == ReformulatedQueryLLM:
                return mock_structured_reformulated

        mock_llm.with_structured_output.side_effect = mock_with_structured_output

        # Create and invoke graph
        graph = create_query_understanding_graph(mock_llm)
        result = graph.invoke({
            "original_query": "Spanish novels like Don Quixote",
            "intent": None,
            "filters": None,
            "reformulated": None,
            "final_query": None,
            "error": None
        })

        # Verify results
        assert result["intent"] == sample_intent_result
        assert result["filters"] == sample_filters_result
        assert result["reformulated"] == sample_reformulated_result
        assert result["final_query"] is not None

    def test_graph_handles_partial_failure(self, mock_llm, sample_intent_result):
        """Test that graph continues even if some nodes fail."""
        # Only intent succeeds, filters and reformulation fail
        mock_structured_intent = Mock()
        mock_structured_intent.invoke.return_value = sample_intent_result

        call_count = [0]

        def mock_with_structured_output(schema):
            call_count[0] += 1
            if schema == QueryIntentLLM:
                return mock_structured_intent
            else:
                raise Exception("Simulated failure")

        mock_llm.with_structured_output.side_effect = mock_with_structured_output

        graph = create_query_understanding_graph(mock_llm)
        result = graph.invoke({
            "original_query": "test query",
            "intent": None,
            "filters": None,
            "reformulated": None,
            "final_query": None,
            "error": None
        })

        # Intent should succeed
        assert result["intent"] == sample_intent_result
        # Filters and reformulated should be None (failed gracefully)
        assert result["filters"] is None
        assert result["reformulated"] is None
        # Final query should fall back to original
        assert result["final_query"] == "test query"
