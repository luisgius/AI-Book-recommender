"""
Pydantic schemas for query understanding LLM outputs.

These schemas are used ONLY in the infrastructure layer to parse and validate
LLM responses from the query understanding LangGraph flow. They are NOT the same
as domain value objects.

Architecture separation:
- Domain value objects (app/domain/value_objects.py): Pure Python dataclasses, no Pydantic
- Infrastructure schemas (this file): Pydantic BaseModels for LLM I/O validation

The LangChainLLMClient converts from these Pydantic schemas to domain value objects.
"""

from pydantic import BaseModel, Field
from typing import Literal


class QueryIntentLLM(BaseModel):
    """
    Pydantic schema for query intent classification returned by the LLM.

    This is the INFRASTRUCTURE representation used for parsing LLM JSON output
    from the "parse_intent" node in the LangGraph flow.

    It will be combined with other schemas and converted to domain.value_objects.QueryIntent
    after the full flow completes.
    """

    intent_type: Literal["recommendation", "factual", "exploratory"] = Field(
        description=(
            "The classified intent type. MUST be exactly one of: "
            "recommendation (find similar books), "
            "factual (answer a specific question), "
            "exploratory (discover new topics)"
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the classification (0.0 to 1.0). "
            "Use lower confidence if the query is ambiguous or could fit multiple intents."
        )
    )

    reasoning: str = Field(
        min_length=10,
        max_length=200,
        description=(
            "Brief explanation of WHY you classified the query this way. "
            "What keywords or patterns led to this classification?"
        )
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "intent_type": "recommendation",
                "confidence": 0.9,
                "reasoning": "User explicitly asks for 'books like 1984', indicating a recommendation intent based on similarity."
            }
        }


class ExtractedFiltersLLM(BaseModel):
    """
    Pydantic schema for filters extracted from natural language.

    This is the INFRASTRUCTURE representation used for parsing LLM JSON output
    from the "extract_filters" node in the LangGraph flow.

    The LLM should extract filters from the query text when present.
    All fields are optional - only include them if explicitly mentioned.

    Note: Language codes and categories will be normalized later in the conversion layer.
    """

    language: str | None = Field(
        default=None,
        description=(
            "ISO 639-1 language code if the user specifies a language. "
            "Examples: 'en' for English, 'es' for Spanish, 'fr' for French. "
            "ONLY include if the user explicitly mentions a language."
        )
    )

    category: str | None = Field(
        default=None,
        description=(
            "Book category/genre if mentioned. Examples: 'Fiction', 'Science Fiction', "
            "'History', 'Biography'. Use standard category names, not abbreviations."
        )
    )

    min_year: int | None = Field(
        default=None,
        description=(
            "Minimum publication year if the user specifies a time range. "
            "Examples: 'recent books' → 2020, 'from the 90s' → 1990, 'last 5 years' → 2020"
        )
    )

    max_year: int | None = Field(
        default=None,
        description=(
            "Maximum publication year if the user specifies a time range. "
            "Examples: 'from the 90s' → 1999, 'before 2000' → 1999"
        )
    )

    author_hint: str | None = Field(
        default=None,
        description=(
            "Author name if the user mentions a specific author. "
            "Examples: 'books by Asimov', 'Stephen King novels'. "
            "This is a hint, not a strict filter (we don't have author filtering yet)."
        )
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "language": "es",
                "category": "Fiction",
                "min_year": 2020,
                "max_year": None,
                "author_hint": None
            }
        }


class ReformulatedQueryLLM(BaseModel):
    """
    Pydantic schema for query reformulation returned by the LLM.

    This is the INFRASTRUCTURE representation used for parsing LLM JSON output
    from the "reformulate_query" node in the LangGraph flow.

    The LLM should:
    1. Clean the query (remove noise words, fix typos)
    2. Expand it with synonyms or related terms for better retrieval
    3. Extract key keywords that represent the core search intent
    """

    optimized_query: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "The optimized query text for better retrieval. "
            "Remove filler words, fix typos, expand with synonyms. "
            "Focus on the core search intent."
        )
    )

    keywords: list[str] = Field(
        max_length=10,
        description=(
            "List of key search terms extracted from the query (max 10). "
            "These should be the most important words for retrieval."
        )
    )

    removed_noise: list[str] = Field(
        default_factory=list,
        description=(
            "List of noise words/phrases removed during reformulation. "
            "Examples: 'I want', 'please', 'can you', 'looking for'. "
            "This is optional, for debugging/transparency."
        )
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "optimized_query": "science fiction artificial intelligence ethics",
                "keywords": ["science fiction", "AI", "artificial intelligence", "ethics", "technology"],
                "removed_noise": ["I'm looking for", "books about", "please"]
            }
        }


# Type alias for code readability
IntentType = Literal["recommendation", "factual", "exploratory"]
