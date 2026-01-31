"""
Tests for LangChainLLMClient.

This module tests the LangChain implementation of the LLMClient port,
including all three main methods:
1. generate_grounded_explanation() - RAG explanation with citations
2. extract_query_intent() - Query understanding via LangGraph
3. judge_explanation() - LLM-as-Judge evaluation

All LLM calls are mocked to avoid actual API usage and ensure deterministic tests.

Note: These tests mock LangChain imports to avoid version compatibility issues
and API key requirements.
"""

import pytest
import sys
from uuid import uuid4
from datetime import datetime, UTC
from unittest.mock import Mock, MagicMock, patch

from app.domain.entities import Book, Explanation, Citation
from app.domain.value_objects import SearchFilters, QueryIntent
from app.infrastructure.llm.schemas import GroundedExplanationLLM, CitationLLM
from app.infrastructure.llm.schemas_judge import ExplanationJudgmentLLM, JudgmentDimension
from app.infrastructure.llm.schemas_query import QueryIntentLLM, ExtractedFiltersLLM, ReformulatedQueryLLM


# ================================================================================
# MOCK SETUP: Mock LangChain imports before importing the client module
# ================================================================================

# Create mock modules for LangChain dependencies
mock_langchain_openai = MagicMock()
mock_langchain_anthropic = MagicMock()
mock_langchain_chat_models = MagicMock()

# Patch sys.modules to use mocks
sys.modules['langchain_openai'] = mock_langchain_openai
sys.modules['langchain_anthropic'] = mock_langchain_anthropic
sys.modules['langchain.chat_models'] = mock_langchain_chat_models

# Now we can safely import the module
import app.infrastructure.llm.langchain_llm_client as llm_client_module


# ================================================================================
# HELPER: Create a mocked LangChainLLMClient instance
# ================================================================================

def create_mock_client(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.0,
    chain_mock=None,
    graph_mock=None,
    judge_mock=None
):
    """
    Create a LangChainLLMClient with mocked internal components.

    This avoids LLM API calls by mocking the chain, graph, and judge.
    """
    with patch.object(llm_client_module, 'load_dotenv'):
        with patch.object(llm_client_module, 'init_chat_model') as mock_init:
            with patch.object(llm_client_module, 'create_grounded_explanation_chain') as mock_chain_factory:
                with patch.object(llm_client_module, 'create_query_understanding_graph') as mock_graph_factory:
                    with patch.object(llm_client_module, 'create_llm_judge_chain') as mock_judge_factory:
                        mock_init.return_value = Mock()
                        mock_chain_factory.return_value = chain_mock or Mock()
                        mock_graph_factory.return_value = graph_mock or Mock()
                        mock_judge_factory.return_value = judge_mock or Mock()

                        return llm_client_module.LangChainLLMClient(
                            provider=provider,
                            model_name=model_name,
                            temperature=temperature
                        )


# ================================================================================
# TEST: Initialization
# ================================================================================

class TestLangChainLLMClientInitialization:
    """Tests for LangChainLLMClient initialization."""

    def test_initialization_default_parameters(self):
        """Test that client initializes with default parameters."""
        client = create_mock_client()

        assert client.provider == "openai"
        assert client.model_name == "gpt-4o-mini"
        assert client.temperature == 0.0

    def test_initialization_custom_parameters(self):
        """Test that client initializes with custom parameters."""
        client = create_mock_client(
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            temperature=0.5
        )

        assert client.provider == "anthropic"
        assert client.model_name == "claude-3-haiku-20240307"
        assert client.temperature == 0.5

    def test_initialization_fails_without_api_key(self):
        """Test that initialization fails when API key is missing."""
        with patch.object(llm_client_module, 'load_dotenv'):
            with patch.object(llm_client_module, 'init_chat_model') as mock_init:
                mock_init.side_effect = Exception("API key not found")

                with pytest.raises(RuntimeError) as excinfo:
                    llm_client_module.LangChainLLMClient()

                assert "Could not initialize LLM" in str(excinfo.value)
                assert "OPENAI_API_KEY" in str(excinfo.value)


# ================================================================================
# TEST: generate_grounded_explanation()
# ================================================================================

class TestGenerateGroundedExplanation:
    """Tests for generate_grounded_explanation method."""

    def test_generate_explanation_success(
        self, sample_book, sample_query_text, mock_grounded_explanation_llm
    ):
        """Test successful explanation generation with valid citations."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_grounded_explanation_llm

        with patch.object(llm_client_module, 'validate_grounded_explanation') as mock_validate:
            expected_explanation = Explanation(
                book_id=sample_book.id,
                query_text=sample_query_text,
                text="The title 'Pragmatic Programmer' matches the query.",
                citations=[
                    Citation(
                        book_id=sample_book.id,
                        chunk_id="title",
                        snippet="Pragmatic Programmer",
                        relevance_score=0.9
                    )
                ],
                model="gpt-4o-mini"
            )
            mock_validate.return_value = expected_explanation

            client = create_mock_client(chain_mock=mock_chain)
            result = client.generate_grounded_explanation(sample_query_text, sample_book)

            assert isinstance(result, Explanation)
            assert result.book_id == sample_book.id
            assert result.query_text == sample_query_text
            assert len(result.citations) > 0
            mock_chain.invoke.assert_called_once()

    def test_generate_explanation_empty_query_raises_error(self, sample_book):
        """Test that empty query raises ValueError."""
        client = create_mock_client()

        with pytest.raises(ValueError) as excinfo:
            client.generate_grounded_explanation("", sample_book)

        assert "query_text cannot be empty" in str(excinfo.value)

    def test_generate_explanation_whitespace_query_raises_error(self, sample_book):
        """Test that whitespace-only query raises ValueError."""
        client = create_mock_client()

        with pytest.raises(ValueError) as excinfo:
            client.generate_grounded_explanation("   ", sample_book)

        assert "query_text cannot be empty" in str(excinfo.value)

    def test_generate_explanation_none_book_raises_error(self, sample_query_text):
        """Test that None book raises ValueError."""
        client = create_mock_client()

        with pytest.raises(ValueError) as excinfo:
            client.generate_grounded_explanation(sample_query_text, None)

        assert "book cannot be None" in str(excinfo.value)

    def test_generate_explanation_chain_failure_returns_fallback(
        self, sample_book, sample_query_text
    ):
        """Test that chain failure returns a fallback explanation (graceful degradation)."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("LLM API error")

        client = create_mock_client(chain_mock=mock_chain)
        result = client.generate_grounded_explanation(sample_query_text, sample_book)

        # Should return fallback explanation, not raise
        assert isinstance(result, Explanation)
        assert result.book_id == sample_book.id
        assert "Unable to generate explanation" in result.text
        assert len(result.citations) == 0


# ================================================================================
# TEST: extract_query_intent()
# ================================================================================

class TestExtractQueryIntent:
    """Tests for extract_query_intent method."""

    def test_extract_intent_success(
        self, sample_query_text, mock_query_intent_llm,
        mock_extracted_filters, mock_reformulated_query
    ):
        """Test successful query intent extraction."""
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            "original_query": sample_query_text,
            "intent": mock_query_intent_llm,
            "filters": mock_extracted_filters,
            "reformulated": mock_reformulated_query,
            "final_query": "software development programming best practices",
            "error": None
        }

        client = create_mock_client(graph_mock=mock_graph)
        result = client.extract_query_intent(sample_query_text)

        assert isinstance(result, QueryIntent)
        assert result.intent_type == "recommendation"
        assert result.original_query == sample_query_text
        assert result.confidence == 0.9
        assert result.extracted_filters.language == "en"
        assert result.extracted_filters.category == "Programming"
        mock_graph.invoke.assert_called_once()

    def test_extract_intent_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        client = create_mock_client()

        with pytest.raises(ValueError) as excinfo:
            client.extract_query_intent("")

        assert "query_text cannot be empty" in str(excinfo.value)

    def test_extract_intent_graph_failure_returns_fallback(self, sample_query_text):
        """Test that graph failure returns fallback intent (graceful degradation)."""
        mock_graph = Mock()
        mock_graph.invoke.side_effect = Exception("Graph execution failed")

        client = create_mock_client(graph_mock=mock_graph)
        result = client.extract_query_intent(sample_query_text)

        # Should return fallback intent
        assert isinstance(result, QueryIntent)
        assert result.intent_type == "exploratory"
        assert result.original_query == sample_query_text
        assert result.reformulated_query == sample_query_text
        assert result.confidence == 0.0
        assert "fallback" in result.reasoning.lower()

    def test_extract_intent_partial_result_handles_missing_fields(self, sample_query_text):
        """Test that partial results (some nodes failed) are handled gracefully."""
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            "original_query": sample_query_text,
            "intent": None,
            "filters": None,
            "reformulated": None,
            "final_query": None,
            "error": "Intent extraction failed"
        }

        client = create_mock_client(graph_mock=mock_graph)
        result = client.extract_query_intent(sample_query_text)

        # Should use defaults for missing fields
        assert isinstance(result, QueryIntent)
        assert result.intent_type == "exploratory"
        assert result.original_query == sample_query_text
        assert result.reformulated_query == sample_query_text
        assert result.confidence == 0.5
        assert result.extracted_filters.is_empty()


# ================================================================================
# TEST: judge_explanation()
# ================================================================================

class TestJudgeExplanation:
    """Tests for judge_explanation method."""

    def test_judge_explanation_success(
        self, sample_book, sample_query_text, sample_explanation, mock_judgment
    ):
        """Test successful explanation judgment."""
        mock_judge = Mock()
        mock_judge.invoke.return_value = mock_judgment

        client = create_mock_client(judge_mock=mock_judge)
        result = client.judge_explanation(sample_query_text, sample_book, sample_explanation)

        assert isinstance(result, ExplanationJudgmentLLM)
        assert result.groundedness.score == 5
        assert result.clarity.score == 4
        assert result.relevance.score == 5
        mock_judge.invoke.assert_called_once()

    def test_judge_explanation_empty_query_raises_error(
        self, sample_book, sample_explanation
    ):
        """Test that empty query raises ValueError."""
        client = create_mock_client()

        with pytest.raises(ValueError) as excinfo:
            client.judge_explanation("", sample_book, sample_explanation)

        assert "query_text cannot be empty" in str(excinfo.value)

    def test_judge_explanation_none_book_raises_error(
        self, sample_query_text, sample_explanation
    ):
        """Test that None book raises ValueError."""
        client = create_mock_client()

        with pytest.raises(ValueError) as excinfo:
            client.judge_explanation(sample_query_text, None, sample_explanation)

        assert "book cannot be None" in str(excinfo.value)

    def test_judge_explanation_none_explanation_raises_error(
        self, sample_book, sample_query_text
    ):
        """Test that None explanation raises ValueError."""
        client = create_mock_client()

        with pytest.raises(ValueError) as excinfo:
            client.judge_explanation(sample_query_text, sample_book, None)

        assert "Explanation cannot be None" in str(excinfo.value)

    def test_judge_explanation_chain_failure_raises_runtime_error(
        self, sample_book, sample_query_text, sample_explanation
    ):
        """Test that judge chain failure raises RuntimeError (no graceful degradation for judge)."""
        mock_judge = Mock()
        mock_judge.invoke.side_effect = Exception("Judge API error")

        client = create_mock_client(judge_mock=mock_judge)

        with pytest.raises(RuntimeError) as excinfo:
            client.judge_explanation(sample_query_text, sample_book, sample_explanation)

        assert "LLM judge call failed" in str(excinfo.value)


# ================================================================================
# TEST: get_model_name()
# ================================================================================

class TestGetModelName:
    """Tests for get_model_name method."""

    def test_get_model_name_default(self):
        """Test that get_model_name returns the default model name."""
        client = create_mock_client()
        assert client.get_model_name() == "gpt-4o-mini"

    def test_get_model_name_custom(self):
        """Test that get_model_name returns custom model name."""
        client = create_mock_client(model_name="claude-3-opus-20240229")
        assert client.get_model_name() == "claude-3-opus-20240229"


# ================================================================================
# TEST: Internal conversion methods
# ================================================================================

class TestInternalMethods:
    """Tests for internal helper methods."""

    def test_convert_graph_result_to_query_intent_complete(
        self, mock_query_intent_llm, mock_extracted_filters, mock_reformulated_query
    ):
        """Test conversion from graph result to QueryIntent with all fields."""
        client = create_mock_client()

        result_dict = {
            "original_query": "books about Python",
            "intent": mock_query_intent_llm,
            "filters": mock_extracted_filters,
            "reformulated": mock_reformulated_query,
            "final_query": "Python programming books tutorials",
            "error": None
        }

        query_intent = client._convert_graph_result_to_query_intent(result_dict)

        assert query_intent.intent_type == "recommendation"
        assert query_intent.original_query == "books about Python"
        assert query_intent.reformulated_query == "Python programming books tutorials"
        assert query_intent.confidence == 0.9
        assert query_intent.extracted_filters.language == "en"

    def test_create_fallback_query_intent(self):
        """Test creation of fallback QueryIntent."""
        client = create_mock_client()

        fallback = client._create_fallback_query_intent("test query")

        assert fallback.intent_type == "exploratory"
        assert fallback.original_query == "test query"
        assert fallback.reformulated_query == "test query"
        assert fallback.confidence == 0.0
        assert fallback.extracted_filters.is_empty()
        assert "fallback" in fallback.reasoning.lower()

    def test_convert_graph_result_with_missing_intent(self):
        """Test conversion when intent is missing."""
        client = create_mock_client()

        result_dict = {
            "original_query": "test query",
            "intent": None,
            "filters": None,
            "reformulated": None,
            "final_query": None,
            "error": None
        }

        query_intent = client._convert_graph_result_to_query_intent(result_dict)

        assert query_intent.intent_type == "exploratory"
        assert query_intent.confidence == 0.5
        assert query_intent.reformulated_query == "test query"

    def test_convert_graph_result_with_partial_filters(self, mock_query_intent_llm):
        """Test conversion with partial filter extraction."""
        client = create_mock_client()

        # Only language extracted, no category
        partial_filters = ExtractedFiltersLLM(
            language="es",
            category=None,
            min_year=2020,
            max_year=None
        )

        result_dict = {
            "original_query": "libros de ciencia ficcion recientes",
            "intent": mock_query_intent_llm,
            "filters": partial_filters,
            "reformulated": None,
            "final_query": "ciencia ficcion libros",
            "error": None
        }

        query_intent = client._convert_graph_result_to_query_intent(result_dict)

        assert query_intent.extracted_filters.language == "es"
        assert query_intent.extracted_filters.category is None
        assert query_intent.extracted_filters.min_year == 2020


# ================================================================================
# TEST: Retry logic
# ================================================================================

class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    def test_retry_succeeds_on_second_attempt(
        self, sample_book, sample_query_text, mock_grounded_explanation_llm
    ):
        """Test that retry succeeds when second attempt works."""
        mock_chain = Mock()
        # First call fails, second succeeds
        mock_chain.invoke.side_effect = [
            Exception("Transient API error"),
            mock_grounded_explanation_llm
        ]

        with patch.object(llm_client_module, 'validate_grounded_explanation') as mock_validate:
            expected_explanation = Explanation(
                book_id=sample_book.id,
                query_text=sample_query_text,
                text="Retry succeeded",
                citations=[],
                model="gpt-4o-mini"
            )
            mock_validate.return_value = expected_explanation

            # Use minimal delays for fast tests
            client = create_mock_client(
                chain_mock=mock_chain,
            )
            client.max_retries = 2
            client.base_delay = 0.01
            client.max_delay = 0.1

            result = client.generate_grounded_explanation(sample_query_text, sample_book)

            assert isinstance(result, Explanation)
            assert result.text == "Retry succeeded"
            assert mock_chain.invoke.call_count == 2

    def test_retry_exhausted_returns_fallback_for_explanation(
        self, sample_book, sample_query_text
    ):
        """Test that exhausted retries return fallback explanation."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Persistent API error")

        client = create_mock_client(chain_mock=mock_chain)
        client.max_retries = 2
        client.base_delay = 0.01
        client.max_delay = 0.1

        result = client.generate_grounded_explanation(sample_query_text, sample_book)

        # Should return fallback explanation
        assert isinstance(result, Explanation)
        assert "Unable to generate explanation" in result.text
        assert len(result.citations) == 0
        # 3 attempts total (1 initial + 2 retries)
        assert mock_chain.invoke.call_count == 3

    def test_retry_exhausted_raises_for_judge(
        self, sample_book, sample_query_text, sample_explanation
    ):
        """Test that exhausted retries raise RuntimeError for judge."""
        mock_judge = Mock()
        mock_judge.invoke.side_effect = Exception("Persistent API error")

        client = create_mock_client(judge_mock=mock_judge)
        client.max_retries = 2
        client.base_delay = 0.01
        client.max_delay = 0.1

        with pytest.raises(RuntimeError) as excinfo:
            client.judge_explanation(sample_query_text, sample_book, sample_explanation)

        assert "LLM judge call failed" in str(excinfo.value)
        # 3 attempts total (1 initial + 2 retries)
        assert mock_judge.invoke.call_count == 3

    def test_retry_configuration_is_customizable(self):
        """Test that retry configuration can be customized."""
        client = create_mock_client()
        client.max_retries = 5
        client.base_delay = 2.0
        client.max_delay = 120.0

        assert client.max_retries == 5
        assert client.base_delay == 2.0
        assert client.max_delay == 120.0

    def test_retry_succeeds_on_third_attempt_for_graph(
        self, sample_query_text, mock_query_intent_llm,
        mock_extracted_filters, mock_reformulated_query
    ):
        """Test that query understanding retries work."""
        mock_graph = Mock()
        # First two calls fail, third succeeds
        mock_graph.invoke.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            {
                "original_query": sample_query_text,
                "intent": mock_query_intent_llm,
                "filters": mock_extracted_filters,
                "reformulated": mock_reformulated_query,
                "final_query": "optimized query",
                "error": None
            }
        ]

        client = create_mock_client(graph_mock=mock_graph)
        client.max_retries = 3
        client.base_delay = 0.01
        client.max_delay = 0.1

        result = client.extract_query_intent(sample_query_text)

        assert isinstance(result, QueryIntent)
        assert result.intent_type == "recommendation"
        assert mock_graph.invoke.call_count == 3
