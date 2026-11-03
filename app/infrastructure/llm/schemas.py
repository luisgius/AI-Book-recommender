"""
Pydantic schemas for LLM structured outputs.

These schemas are used ONLY in the infrastructure layer to parse and validate
LLM responses. They are NOT the same as domain entities.

Architecture separation:
- Domain entities (app/domain/entities.py): Pure Python dataclasses, no validation framework
- Infrastructure schemas (this file): Pydantic BaseModels for LLM I/O validation

The LangChainLLMClient converts from these Pydantic schemas to domain entities.
"""

from pydantic import BaseModel, Field
from typing import Literal


class CitationLLM(BaseModel):
    """
    Pydantic schema for a citation returned by the LLM.

    This is the INFRASTRUCTURE representation used for parsing LLM JSON output.
    It will be converted to domain.entities.Citation after validation.

    Key differences from domain Citation:
    - book_id: str (LLM returns strings, will be converted to UUID later)
    - chunk_id: str (will be validated against Literal values)
    - No frozen=True (Pydantic handles immutability differently)
    - Includes Field() validators for LLM output requirements
    """

    book_id: str = Field(
        description="UUID of the book being cited (as string)"
    )

    chunk_id: Literal["title", "description", "categories", "authors"] = Field(
        description="Which book field is being cited. MUST be one of: title, description, categories, authors"
    )

    snippet: str = Field(
        min_length=1,
        max_length=200,
        description="Exact verbatim text from the cited book field (max 200 chars). MUST be a literal substring."
    )

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant this citation is to supporting the explanation (0.0 to 1.0)"
    )

    class Config:
        """Pydantic configuration."""
        # Allow JSON schema generation for LLM structured output
        json_schema_extra = {
            "example": {
                "book_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_id": "title",
                "snippet": "Artificial Intelligence: A Modern Approach",
                "relevance_score": 0.9
            }
        }


class GroundedExplanationLLM(BaseModel):
    """
    Pydantic schema for a grounded explanation returned by the LLM.

    This is the INFRASTRUCTURE representation used for parsing LLM structured output.
    It will be converted to domain.entities.Explanation after validation and guardrails.

    The LLM MUST provide:
    - summary: One-sentence high-level explanation
    - reasoning: Detailed explanation with citation markers
    - citations: List of at least 1 citation (enforced by min_length=1)
    - confidence: Self-assessed confidence score

    Guardrails (applied AFTER parsing this schema):
    - Validate that each citation.snippet actually exists in the book field
    - Filter out hallucinated citations
    - If no valid citations remain after filtering, return "no evidence" response
    """

    summary: str = Field(
        min_length=10,
        max_length=200,
        description="One-sentence summary of why this book is relevant to the query"
    )

    reasoning: str = Field(
        min_length=20,
        max_length=500,
        description=(
            "Detailed explanation of relevance. Use [CITE: chunk_id] markers to reference citations. "
            "Every claim MUST be supported by a citation."
        )
    )

    citations: list[CitationLLM] = Field(
        min_length=1,
        description=(
            "List of citations supporting the explanation. MUST include at least one citation. "
            "Each citation must reference a specific book field with a verbatim snippet."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Self-assessed confidence in this explanation (0.0 to 1.0). "
            "Lower if citations are weak or evidence is indirect."
        )
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "summary": "This book covers AI fundamentals and modern techniques.",
                "reasoning": (
                    "The book is highly relevant because [CITE: title] it explicitly focuses on "
                    "artificial intelligence, and [CITE: description] the description mentions "
                    "machine learning and neural networks."
                ),
                "citations": [
                    {
                        "book_id": "123e4567-e89b-12d3-a456-426614174000",
                        "chunk_id": "title",
                        "snippet": "Artificial Intelligence: A Modern Approach",
                        "relevance_score": 0.95
                    },
                    {
                        "book_id": "123e4567-e89b-12d3-a456-426614174000",
                        "chunk_id": "description",
                        "snippet": "comprehensive introduction to machine learning and neural networks",
                        "relevance_score": 0.85
                    }
                ],
                "confidence": 0.9
            }
        }


# Type alias for code readability
ChunkFieldStr = Literal["title", "description", "categories", "authors"]


