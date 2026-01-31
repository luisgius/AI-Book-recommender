"""
LLM Judge Service for evaluating explanation quality.

This service orchestrates:
1. LLM-as-judge evaluation (groundedness, clarity, relevance)
2. Deterministic citation metrics (precision, recall)
3. Combining results into structured judgment records
"""

import logging
from typing import Optional

from app.domain.entities import Book, Explanation
from app.domain.ports import LLMClient
from app.evaluation.evaluation_service import EvaluationService
from app.evaluation.types import ExplanationJudgmentResult


logger = logging.getLogger(__name__)


class LLMJudgeService:
    """
    Service for judging explanation quality.
    
    Combines LLM-as-judge scores with deterministic citation metrics.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        eval_service: EvaluationService
    ):
        """
        Initialize the judge service.
        
        Args:
            llm_client: Client for LLM-as-judge calls
            eval_service: Service for computing citation metrics
        """
        self.llm_client = llm_client
        self.eval_service = eval_service
        logger.info("LLM Judge Service initialized")


    def judge_explanation(
    self,
    query_id: str,
    query_text: str,
    book: Book,
    explanation: Explanation
    ) -> ExplanationJudgmentResult:
        """
        Judge a single explanation.
        
        Combines LLM-as-judge evaluation with deterministic citation metrics.
        If LLM call fails, returns partial result with only citation metrics.
        
        Args:
            query_id: Identifier for the query (for result tracking)
            query_text: The user's search query as plain text
            book: The book entity that was explained
            explanation: The Explanation entity to evaluate
            
        Returns:
            ExplanationJudgmentResult with:
            - LLM scores (groundedness, clarity, relevance) if successful
            - Citation metrics (precision, recall) always computed
            - None for LLM scores if judge call fails
            
        Example:
            >>> result = service.judge_explanation(
            ...     "q01", "AI books", book, explanation
            ... )
            >>> print(result.groundedness_score)  # May be None if LLM failed
            4.0
            >>> print(result.citation_precision)  # Always present
            0.85
        """

        # Step 1: Compute citation metrics (deterministic, always succeed)
        logger.debug(
            f"Computing citation metrics for query={query_id}, book={book.id}"
        )

        citation_precision = self.eval_service.compute_citation_precision(explanation, book)
        citation_recall = self.eval_service.compute_citation_recall(explanation, book)

        logger.debug(
            f"Citation metrics: precision={citation_precision:.2f}, recall={citation_recall:.2f}"
        )

        # Step 2: Try to get LLM judgment (may fail due to API issues)
        try:
            logger.debug(f"Calling LLM judge for query={query_id}, book={book.id}")
            
            judgment = self.llm_client.judge_explanation(query_text, book, explanation)
            
            # Extract scores from judgment
            groundedness_score = judgment.groundedness.score
            groundedness_reasoning = judgment.groundedness.reasoning
            clarity_score = judgment.clarity.score
            clarity_reasoning = judgment.clarity.reasoning
            relevance_score = judgment.relevance.score
            relevance_reasoning = judgment.relevance.reasoning
            
            logger.info(
                f"LLM judge succeeded: G={groundedness_score}, C={clarity_score}, R={relevance_score}"
            )

        except Exception as e:
            # LLM call failed - log and set scores to None
            logger.warning(
                f"LLM judge failed for query={query_id}, book={book.id}: {e}"
            )
            logger.warning("Returning partial result with only citation metrics")
            
            # All LLM scores are None
            groundedness_score = None
            groundedness_reasoning = None
            clarity_score = None
            clarity_reasoning = None
            relevance_score = None
            relevance_reasoning = None

        # Step 3: Combine everything into result
        return ExplanationJudgmentResult(
            query_id=query_id,
            book_id=book.id,
            # Citation metrics (always present) - required fields first
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            # LLM scores (may be None if call failed) - optional fields
            groundedness_score=groundedness_score,
            groundedness_reasoning=groundedness_reasoning,
            clarity_score=clarity_score,
            clarity_reasoning=clarity_reasoning,
            relevance_score=relevance_score,
            relevance_reasoning=relevance_reasoning,
        )
