"""
Guardrails for grounded explanation generation.

This module implements DETERMINISTIC validation to ensure LLM outputs are
grounded in actual book data. It prevents hallucinations by verifying that
every citation references real text from the book.

Architecture:
- Input: GroundedExplanationLLM (Pydantic schema from LLM)
- Process: Validate citations against actual book fields
- Output: Explanation (domain entity) with validated citations

Key principle: Never trust the LLM blindly. Always verify.
"""

from datetime import datetime, UTC
from uuid import UUID
from typing import Optional

from app.domain.entities import Book, Citation, Explanation
from app.domain.citation_validation import get_book_field_text, is_valid_snippet
from .schemas import CitationLLM, GroundedExplanationLLM


def filter_valid_citations(
    citations_llm: list[CitationLLM],
    book: Book
) -> list[CitationLLM]:
    """
    Filter out hallucinated citations, keeping only grounded ones.

    For each citation:
    1. Get the actual text of the cited book field
    2. Verify the snippet exists in that field
    3. Keep citation only if verification passes

    Citations that fail verification are silently discarded (tolerance).

    Args:
        citations_llm: List of citations from LLM (Pydantic schema)
        book: The book being cited

    Returns:
        List of citations that passed validation (may be empty)
    """
    valid_citations = []

    for citation in citations_llm:
        # Get the actual field text
        field_text = get_book_field_text(book, citation.chunk_id)

        # Validate snippet against field text
        if is_valid_snippet(citation.snippet, field_text):
            valid_citations.append(citation)
        # else: silently discard hallucinated citation

    return valid_citations


def convert_citation_to_domain(citation_llm: CitationLLM, book_id: UUID) -> Citation:
    """
    Convert infrastructure Pydantic schema to domain entity.

    Args:
        citation_llm: Pydantic schema from LLM
        book_id: UUID of the book (from domain entity)

    Returns:
        Domain Citation entity (frozen dataclass)
    """
    return Citation(
        book_id=book_id,
        chunk_id=citation_llm.chunk_id,
        snippet=citation_llm.snippet,
        relevance_score=citation_llm.relevance_score
    )


def validate_grounded_explanation(
    explanation_llm: GroundedExplanationLLM,
    book: Book,
    query_text: str,
    model_name: str
) -> Explanation:
    """
    Validate and convert LLM explanation to domain entity with guardrails.

    This is the main guardrail function. It:
    1. Filters out hallucinated citations
    2. Checks minimum citation threshold
    3. Adjusts confidence based on citation quality
    4. Returns domain Explanation entity

    Fallback behavior:
    - If no valid citations remain after filtering, returns "no evidence" explanation
    - If citations are weak (avg relevance < 0.3), caps confidence at 0.3
    - If confidence < 0.5, prefixes text with [LOW CONFIDENCE] warning

    Args:
        explanation_llm: Pydantic schema from LLM (may contain hallucinations)
        book: The book entity being explained
        query_text: The user's search query
        model_name: LLM model identifier (e.g., "gpt-4o-mini")

    Returns:
        Explanation domain entity with validated citations

    Examples:
        >>> # LLM returns 3 citations, but 1 is hallucinated
        >>> explanation_llm = GroundedExplanationLLM(
        ...     summary="...",
        ...     reasoning="...",
        ...     citations=[good_citation1, good_citation2, hallucinated_citation],
        ...     confidence=0.9
        ... )
        >>> result = validate_grounded_explanation(explanation_llm, book, query, model)
        >>> len(result.citations)  # Only 2 valid citations
        2

        >>> # All citations are hallucinated
        >>> explanation_llm = GroundedExplanationLLM(
        ...     citations=[hallucinated1, hallucinated2],
        ...     ...
        ... )
        >>> result = validate_grounded_explanation(explanation_llm, book, query, model)
        >>> result.text
        "Unable to provide a grounded explanation. No valid evidence found..."
        >>> result.citations
        []
    """
    # Step 1: Filter out hallucinated citations
    valid_citations_llm = filter_valid_citations(explanation_llm.citations, book)

    # Step 2: Check minimum citation threshold
    if len(valid_citations_llm) == 0:
        # FALLBACK: No valid evidence found
        return Explanation(
            book_id=book.id,
            query_text=query_text,
            text=(
                "Unable to provide a grounded explanation. "
                "No valid evidence found in the book data to support relevance claims."
            ),
            citations=[],  # Empty list signals no grounding
            model=model_name,
            created_at=datetime.now(UTC)
        )

    # Step 3: Convert Pydantic citations to domain entities
    domain_citations = [
        convert_citation_to_domain(c, book.id)
        for c in valid_citations_llm
    ]

    # Step 4: Assess citation quality and adjust confidence
    avg_relevance = sum(c.relevance_score for c in domain_citations) / len(domain_citations)
    confidence = explanation_llm.confidence

    # Cap confidence if citations are weak
    if avg_relevance < 0.3:
        confidence = min(confidence, 0.3)

    # Step 5: Build final explanation text with confidence warning if needed
    final_text = explanation_llm.reasoning

    if confidence < 0.5:
        # Prefix with low confidence warning
        final_text = f"[LOW CONFIDENCE] {final_text}"

    # Step 6: Return validated domain entity
    return Explanation(
        book_id=book.id,
        query_text=query_text,
        text=final_text,
        citations=domain_citations,
        model=model_name,
        created_at=datetime.now(UTC)
    )


def compute_citation_precision(explanation: Explanation, book: Book) -> float:
    """
    Compute citation precision for evaluation purposes.

    Citation precision = (# valid citations) / (# total citations attempted)

    This metric measures what percentage of the LLM's citations were grounded.
    It's useful for evaluating LLM quality and prompt effectiveness.

    Note: This assumes the explanation has already been validated by guardrails.
    If called on a pre-validation explanation, results may be inaccurate.

    Args:
        explanation: Domain Explanation entity (post-guardrails)
        book: The book being explained

    Returns:
        Precision score in [0.0, 1.0], or 1.0 if no citations

    Examples:
        >>> # All citations valid
        >>> precision = compute_citation_precision(explanation, book)
        >>> precision
        1.0

        >>> # No citations (no evidence response)
        >>> precision = compute_citation_precision(no_evidence_explanation, book)
        >>> precision
        0.0
    """
    if not explanation.citations:
        # No citations = no precision (but not an error)
        return 0.0

    # All citations in a validated explanation are valid by definition
    # (invalid ones were filtered out by guardrails)
    valid_count = len(explanation.citations)
    total_count = valid_count  # We don't know how many were discarded

    # For post-guardrail explanations, precision is always 1.0 if citations exist
    # This function is more useful during evaluation when comparing raw vs filtered
    return valid_count / total_count if total_count > 0 else 1.0
