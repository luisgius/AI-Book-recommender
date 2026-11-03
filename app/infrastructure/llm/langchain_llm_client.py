"""
LangChain implementation of the LLMClient port.
"""

import logging
import time
import random
from datetime import datetime, UTC
from typing import Callable, TypeVar

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import init_chat_model

from app.domain.ports import LLMClient
from app.domain.entities import Book, Explanation
from app.domain.value_objects import QueryIntent, SearchFilters

from .chains import create_grounded_explanation_chain, create_llm_judge_chain
from .guardrails import validate_grounded_explanation
from .schemas import GroundedExplanationLLM
from .schemas_judge import ExplanationJudgmentLLM
from .graphs.query_understanding import create_query_understanding_graph

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Type variable for generic retry function
T = TypeVar('T')

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds


class LangChainLLMClient(LLMClient):
    """
    LangChain-based implementation of the LLMClient port.

    This adapter implements the LLMClient protocol and orchestrates:
    1. LLM invocation via LangChain chains with retry logic
    2. Guardrail validation to prevent hallucinations
    3. Conversion from infrastructure schemas to domain entities
    4. Error handling and logging

    Retry behavior:
    - Uses exponential backoff with jitter for transient failures
    - Configurable max_retries, base_delay, and max_delay
    - Logs retry attempts for observability

    Protocol compliance:
    - generate_grounded_explanation() -> Explanation (domain entity)
    - extract_query_intent() -> QueryIntent (domain value object)
    - judge_explanation() -> ExplanationJudgmentLLM (infrastructure schema)
    - get_model_name() -> str
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        """
        Initialize the LLM client.

        The API key is loaded from environment variables:
        - OPENAI_API_KEY for OpenAI models
        - ANTHROPIC_API_KEY for Anthropic models

        Make sure to set the appropriate environment variable before initializing.

        Args:
            provider: LLM provider ("openai" or "anthropic")
            model_name: Model identifier (e.g., "gpt-4o-mini", "claude-3-haiku-20240307")
            temperature: Sampling temperature (0.0 = deterministic, recommended for evaluation)
            max_retries: Maximum number of retry attempts for transient failures (default: 3)
            base_delay: Initial delay in seconds for exponential backoff (default: 1.0)
            max_delay: Maximum delay in seconds between retries (default: 60.0)

        Raises:
            RuntimeError: If the LLM cannot be initialized (e.g., missing API key)
        """
        load_dotenv()

        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Log initialization (without exposing API key)
        logger.info(f"Initializing {provider} LLM with model {model_name} (temp={temperature})")

        try:
            self.model = init_chat_model(
                model=self.model_name,
                model_provider=self.provider,
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f"Failed to initialize {provider} model {model_name}: {e}")
            raise RuntimeError(
                f"Could not initialize LLM. Make sure the {provider.upper()}_API_KEY "
                f"environment variable is set. Error: {e}"
            )

        self.chain = create_grounded_explanation_chain(self.model)

        # Initialize the query understanding graph (Block 2)
        self.query_understanding_graph = create_query_understanding_graph(self.model)

        # Initialize the judge chain (Block 3)
        self.judge_chain = create_llm_judge_chain(self.model)
        
        logger.info("LLM client initialized successfully")

    def _invoke_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str,
    ) -> T:
        """
        Execute an operation with exponential backoff retry.

        This method handles transient failures (rate limits, network issues, API errors)
        by retrying with exponentially increasing delays plus random jitter.

        Args:
            operation: A callable that performs the LLM operation
            operation_name: Name of the operation for logging (e.g., "explanation generation")

        Returns:
            The result of the operation if successful

        Raises:
            Exception: The last exception if all retries are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return operation()

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        self.max_delay
                    )

                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {self.max_retries + 1} attempts. "
                        f"Last error: {e}"
                    )

        # Re-raise the last exception after all retries exhausted
        raise last_exception

    def generate_grounded_explanation(
        self, query_text: str, book: Book
    ) -> Explanation:
        """
        Generate a grounded explanation with citations.
        
        This method:
        1. Validates inputs
        2. Prepares chain inputs from book fields
        3. Invokes the LangChain chain
        4. Applies guardrails to validate citations
        5. Returns domain Explanation entity
        
        Args:
            query_text: User's search query
            book: Book entity to explain
            
        Returns:
            Explanation entity (may have empty citations if grounding failed)
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If LLM call fails after retries
        """
        # 1. Validate inputs
        if not query_text or not query_text.strip():
            logger.error("Cannot generate explanation: query_text is empty")
            raise ValueError("query_text cannot be empty")

        if book is None:
            logger.error("Cannot generate explanation: book is None")
            raise ValueError("book cannot be None")

        if not book.title:
            logger.warning(f"Book {book.id} has no title, using 'Untitled'")

        # 2. Prepare inputs for the chain
        inputs = {
            "query_text": query_text,
            "book_id": str(book.id),
            "title": book.title or "Untitled",
            "authors": ", ".join(book.authors) if book.authors else "Unknown authors",
            "categories": ", ".join(book.categories) if book.categories else "Uncategorized",
            "description": book.description or "No description available"
        }

        logger.debug(f"Generating explanation for book {book.id} (query: '{query_text[:50]}...')")

        # 3. Invoke the chain with retry logic and apply guardrails
        try:
            # Use retry wrapper for the LLM call
            result = self._invoke_with_retry(
                operation=lambda: self.chain.invoke(inputs),
                operation_name=f"Explanation generation for book {book.id}"
            )

            explanation = validate_grounded_explanation(
                explanation_llm=result,
                book=book,
                query_text=query_text,
                model_name=self.model_name
            )
            logger.info(f"Generated explanation for book {book.id} with {len(explanation.citations)} citations")

            return explanation

        except Exception as e:
            logger.error(f"Failed to generate explanation for book {book.id} after retries: {e}")

            return Explanation(
                book_id=book.id,
                query_text=query_text,
                text="Unable to generate explanation. Please try again.",
                citations=[],
                model=self.model_name,
                created_at=datetime.now(UTC)
            )

    
    def get_model_name(self) -> str:
        """Get the model identifier."""
        return self.model_name
    
    def extract_query_intent(self, query_text: str) -> QueryIntent:
        """
        Extract query intent using the LangGraph query understanding flow.

        This method:
        1. Validates input
        2. Runs the LangGraph flow (parse_intent → extract_filters → reformulate)
        3. Converts graph output to domain QueryIntent value object
        4. Handles errors with graceful degradation

        Args:
            query_text: The raw user query

        Returns:
            QueryIntent value object with intent, filters, and reformulated query

        Raises:
            ValueError: If query_text is empty
        """
        # 1. Validate input
        if not query_text or not query_text.strip():
            logger.error("Cannot extract intent: query_text is empty")
            raise ValueError("query_text cannot be empty")

        logger.debug(f"Extracting intent for query: '{query_text[:50]}...'")

        try:
            # 2. Run the LangGraph flow with retry logic
            result = self._invoke_with_retry(
                operation=lambda: self.query_understanding_graph.invoke({
                    "original_query": query_text,
                    "intent": None,
                    "filters": None,
                    "reformulated": None,
                    "final_query": None,
                    "query_variations": [],
                    "error": None
                }),
                operation_name="Query understanding"
            )

            # 3. Convert graph output to domain QueryIntent
            query_intent = self._convert_graph_result_to_query_intent(result)

            logger.info(
                f"Extracted intent: {query_intent.intent_type} "
                f"(confidence: {query_intent.confidence:.2f})"
            )

            return query_intent

        except Exception as e:
            logger.error(f"Query understanding flow failed after retries: {e}")

            # Graceful degradation: return default intent
            return self._create_fallback_query_intent(query_text)

    def _convert_graph_result_to_query_intent(self, result: dict) -> QueryIntent:
        """
        Convert LangGraph result to domain QueryIntent.

        Maps infrastructure Pydantic schemas to domain value objects.

        Args:
            result: The final state from the LangGraph flow

        Returns:
            QueryIntent domain value object
        """
        # Extract intent type (default: exploratory)
        if result.get("intent") is not None:
            intent_type = result["intent"].intent_type
            confidence = result["intent"].confidence
            reasoning = result["intent"].reasoning
        else:
            intent_type = "exploratory"
            confidence = 0.5
            reasoning = "Default classification (intent extraction failed)"

        # Extract filters and convert to domain SearchFilters
        if result.get("filters") is not None:
            filters = result["filters"]
            extracted_filters = SearchFilters(
                language=filters.language,
                category=filters.category,
                min_year=filters.min_year,
                max_year=filters.max_year
            )
        else:
            extracted_filters = SearchFilters()

        # Get reformulated query (fallback to original)
        reformulated_query = result.get("final_query") or result["original_query"]

        # Get query variations for multi-query retrieval
        query_variations = result.get("query_variations", [])

        return QueryIntent(
            intent_type=intent_type,
            original_query=result["original_query"],
            reformulated_query=reformulated_query,
            extracted_filters=extracted_filters,
            confidence=confidence,
            reasoning=reasoning,
            query_variations=query_variations,
        )

    def _create_fallback_query_intent(self, query_text: str) -> QueryIntent:
        """
        Create a fallback QueryIntent when the flow completely fails.

        Uses safe defaults:
        - intent_type: "exploratory" (most general)
        - reformulated_query: same as original
        - empty filters
        - low confidence

        Args:
            query_text: Original query text

        Returns:
            QueryIntent with fallback values
        """
        logger.warning(f"Using fallback intent for query: '{query_text[:50]}...'")

        return QueryIntent(
            intent_type="exploratory",
            original_query=query_text,
            reformulated_query=query_text,
            extracted_filters=SearchFilters(),
            confidence=0.0,
            reasoning="Fallback classification (query understanding flow failed completely)"
        )

    def judge_explanation(self, query_text:str, book:Book, explanation:Explanation) -> ExplanationJudgmentLLM:
        """
        Evaluate explanation quality using LLM-as-Judge.
        
        This method:
        1. Validates inputs
        2. Prepares chain inputs (query, book title, explanation text, citations)
        3. Invokes the judge chain
        4. Returns structured judgment with scores for 3 dimensions
        
        Args:
            query_text: User's search query
            book: Book entity that was explained
            explanation: The Explanation entity to evaluate
            
        Returns:
            ExplanationJudgmentLLM with groundedness, clarity, relevance scores
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If LLM judge call fails
        """
        # 1. Validate inputs
        if not query_text or not query_text.strip():
            logger.error("Cannot generate explanation: query_text is empty")
            raise ValueError("query_text cannot be empty")

        if book is None:
            logger.error("Cannot generate explanation: book is None")
            raise ValueError("book cannot be None")

        if not book.title:
            logger.warning(f"Book {book.id} has no title, using 'Untitled'")

        if explanation is None:
            logger.error("Cannot generate judgement: Explanation is None")
            raise ValueError("Explanation cannot be None")

        # 2. Prepare inputs
        inputs = {
            "query_text": query_text,
            "book_title": book.title or "Untitled",
            "explanation_text": explanation.text,
            "citations": explanation.citations
        }


        logger.debug(
            f"Judging explanation for book {book.id} "
            f"(query: '{query_text[:50]}...', citations: {len(explanation.citations)})"
        )

        # 3. Invoke the judge chain with retry logic
        try:
            judgment = self._invoke_with_retry(
                operation=lambda: self.judge_chain.invoke(inputs),
                operation_name=f"LLM judge for book {book.id}"
            )

            logger.info(
                f"Judged explanation for book {book.id}: "
                f"G={judgment.groundedness.score}, "
                f"C={judgment.clarity.score}, "
                f"R={judgment.relevance.score}"
            )
            return judgment

        except Exception as e:
            logger.error(f"Failed to judge explanation for book {book.id} after retries: {e}")
            raise RuntimeError(
                f"LLM judge call failed for book {book.id}. "
                f"This may indicate an API issue or invalid input. Error: {e}"
            )