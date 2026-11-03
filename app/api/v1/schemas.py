"""
"""



from pydantic import BaseModel, Field, AwareDatetime
from datetime import datetime, timezone
from uuid import UUID
import uuid
from typing import Literal







class SearchFilters(BaseModel):
    """
    Filters that can be applied to a search query.
    """
    language: str | None = None
    category: str | None = None
    min_year: int | None = None
    max_year: int | None = None



# reequest body de post/search
class SearchRequest(BaseModel):
    """
    Request body for POST /search endpoint.
    """
    text: str = Field(description="The search query text")
    filters: SearchFilters = Field(default_factory=SearchFilters)
    max_results: int | None = Field(default=None, ge=1, le=100, description="Max results (1-100)")
    use_explanations: bool | None = Field(default=None, description="Generate LLM explanations")
    use_diversification: bool | None = Field(default=None, description="Apply MMR diversification")
    diversity_lambda: float | None = Field(default=None, ge=0.0, le=1.0, description="Diversity trade-off")
    include_metadata: bool | None = Field(default=None, description="Include debug metadata in response")



# response body of post search

class BookMetadata(BaseModel):
    isbn: str | None = None
    page_count: int | None = None
    publisher: str | None = None

class Book(BaseModel):
    """
    API representation of a Book entity.

    Maps from the domain Book entity for API responses.
    """

    id: UUID = Field(description="Unique identifier for this book in our system")
    title: str = Field(description="Book title")
    authors: list[str] = Field(description="List of author names")
    description: str | None = Field(default=None, description="Book description/summary")
    language: str | None = Field(
        default=None,
        min_length=2,
        max_length=10,
        description="ISO 639-1 language code (e.g., 'es', 'en')"
    )
    categories: list[str] = Field(default_factory=list, description="List of categories/genres")
    published_date: datetime | None = Field(default=None, description="Publication date")
    source: str = Field(default="unknown", description="External source (e.g., 'google_books')")
    source_id: str | None = Field(default=None, description="ID in the external source system")
    metadata: BookMetadata | None = Field(default=None, description="Additional optional metadata")
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this book was added to our catalog"
    )
    updated_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this book was last updated"
    )


class SearchResult(BaseModel):
    """
    Represents a single search result linking a book to its relevance scores.
    """
    book: Book = Field(description="The book that matched the query")
    
    final_score: float = Field(
        description="Final relevance score used for ranking (higher is better)."
    )
    
    rank: int = Field(description="Position in the result list (1-indexed)")

    source: Literal['lexical', 'vector', 'hybrid'] = Field(
        description="Which search method produced this result."
    )

    lexical_score: float | None = Field(
        default=None, 
        description="BM25 relevance score from lexical search."
    )
    
    vector_score: float | None = Field(
        default=None,
        description="Cosine similarity score from vector search."
    )
    
    explanation: str | None = Field(
        default=None,
        description="Optional natural language explanation of relevance."
    )

class SearchMetadata(BaseModel):
    """
    Debug/telemetry metadata for search operations.

    Provides transparency about how the search was executed.
    """
    fusion_method: Literal['rrf', 'none'] = Field(description="Fusion method used")
    rrf_k: int | None = Field(default=None, description="RRF k parameter (typically 60)")
    diversification_enabled: bool = Field(default=False, description="Whether MMR was applied")
    candidates_lexical: int = Field(default=0, ge=0, description="Candidates from BM25")
    candidates_vector: int = Field(default=0, ge=0, description="Candidates from vector search")


class SearchResponse(BaseModel):
    """
    Response wrapper for search operations with degradation metadata (RNF-08).
    """
    results: list[SearchResult] = Field(description="List of SearchResult entities")
    degraded: bool = Field(default=False, description="True if search was in degraded mode")
    degradation_reason: str | None = Field(
        default=None,
        description="Human-readable explanation of why degradation occurred"
    )
    search_mode: Literal['hybrid', 'lexical_only', 'vector_only'] = Field(
        default="hybrid",
        description="The search mode used"
    )
    latency_ms: float | None = Field(default=None, description="Search execution time in ms")
    metadata: SearchMetadata | None = Field(
        default=None,
        description="Optional debug/telemetry metadata (include_metadata=true)"
    )