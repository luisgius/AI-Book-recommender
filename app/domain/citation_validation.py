"""
Domain layer utilities for citation validation.

This module provides deterministic validation functions to ensure citations
reference actual content from book entities. These are pure domain functions
with no infrastructure dependencies.

Used by:
- app/evaluation/evaluation_service.py (citation precision/recall)
- app/infrastructure/llm/guardrails.py (LLM output validation)
"""

from app.domain.entities import Book


def get_book_field_text(book: Book, chunk_id: str) -> str:
    """
    Get the text content of a book field by chunk_id.

    This is the source of truth for validation. The snippet MUST be
    a substring of the text returned by this function.

    Args:
        book: The book entity
        chunk_id: One of "title", "description", "categories", "authors"

    Returns:
        The text content of that field, or empty string if field is None/invalid
    """
    if chunk_id == "title":
        return book.title

    elif chunk_id == "description":
        # description can be None
        return book.description or ""

    elif chunk_id == "categories":
        # Join categories into a single searchable string
        return ", ".join(book.categories)

    elif chunk_id == "authors":
        # Join authors into a single searchable string
        return ", ".join(book.authors)

    else:
        # Invalid chunk_id (should not happen if Pydantic validated correctly)
        return ""


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Normalization rules:
    - Convert to lowercase (case-insensitive matching)
    - Collapse multiple whitespaces into single space
    - Strip leading/trailing whitespace

    This allows fuzzy matching while still being deterministic.

    Args:
        text: Raw text string

    Returns:
        Normalized text
    """
    # Split on whitespace and rejoin with single spaces
    # This handles tabs, newlines, multiple spaces, etc.
    return " ".join(text.lower().split())


def is_valid_snippet(snippet: str, field_text: str) -> bool:
    """
    Deterministic validation: check if snippet is a substring of field_text.

    This is the core anti-hallucination check. The snippet MUST exist
    verbatim (modulo normalization) in the book field.

    Args:
        snippet: The text the LLM claims is from the book
        field_text: The actual text from the book field

    Returns:
        True if snippet is a valid substring, False otherwise

    Examples:
        >>> is_valid_snippet("Neural Networks", "Introduction to Neural Networks and Deep Learning")
        True

        >>> is_valid_snippet("Quantum Computing", "Introduction to Neural Networks")
        False  # Hallucination!
    """
    if not snippet or not field_text:
        return False

    # Check for whitespace-only strings BEFORE normalization
    # (normalization would turn "   " into "", which is substring of everything)
    if not snippet.strip():
        return False

    # Normalize both texts for comparison
    snippet_norm = normalize_text(snippet)
    field_norm = normalize_text(field_text)

    # After normalization, empty string check (defensive programming)
    if not snippet_norm:
        return False

    # Check substring containment
    return snippet_norm in field_norm
