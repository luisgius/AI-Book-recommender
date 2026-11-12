"""
Pydantic models for the v1 HTTP API (request and response bodies).

These are boundary types: converters.py maps them to and from the domain
entities and value objects.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, Field


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
    use_understanding: bool | None = Field(
        default=None,
        description=(
            "Interpret the query with the LLM first (intent, filters, reformulation, "
            "multi-query retrieval). Explicit filters still take precedence."
        ),
    )



# response body of post search

class BookMetadata(BaseModel):
    isbn: str | None = None
    page_count: int | None = None
    publisher: str | None = None


class DescriptionSource(BaseModel):
    """Origin and licence of the description shown for a book (attribution, V17)."""
    source: str = Field(description="Source of the text (e.g. 'wikipedia', 'gutenberg', 'doab')")
    language: str | None = Field(default=None, description="ISO 639-1 code of the text")
    license: str | None = Field(default=None, description="Licence of the text (e.g. 'CC BY-SA 4.0')")
    attribution_url: str | None = Field(
        default=None, description="Page to cite for the text (e.g. the Wikipedia article)"
    )
    auto_generated: bool = Field(default=False, description="True when the source generated the text automatically")


class ProvenanceRecord(BaseModel):
    """One source record that contributed to a book (e.g. its Wikidata item)."""
    source: str
    record_id: str
    license: str | None = None
    url: str | None = None

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
    description_source: DescriptionSource | None = Field(
        default=None,
        description="Source, licence and attribution link of the description (multi-source catalog)",
    )
    provenance: list[ProvenanceRecord] = Field(
        default_factory=list, description="Source records the book was built from, with their licences"
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this book was added to our catalog"
    )
    updated_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this book was last updated"
    )


class Citation(BaseModel):
    """A verified piece of evidence from the book data that supports an explanation."""
    field: Literal['title', 'description', 'categories', 'authors'] = Field(
        description="Book field the snippet was taken from"
    )
    snippet: str = Field(description="Verbatim text from that field")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance of the evidence (0-1)")


class Explanation(BaseModel):
    """Grounded LLM explanation of why a book matches the query (RAG)."""
    text: str = Field(description="Natural language explanation")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations that passed verification against the book data",
    )
    grounded: bool = Field(description="True if at least one citation was verified")
    model: str = Field(description="LLM that generated the explanation")


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
    
    explanation: Explanation | None = Field(
        default=None,
        description="Optional grounded explanation of relevance (use_explanations=true)."
    )

    rerank_score: float | None = Field(
        default=None,
        description=(
            "Cross-encoder relevance score (raw model output, higher is better) when "
            "the optional re-ranking stage scored this result; final_score keeps the "
            "fusion score. Null without a reranker or beyond the re-ranking depth."
        ),
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
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Milliseconds per stage (lexical, embedding, vector, rerank, mmr, explanations)",
    )


class QueryIntent(BaseModel):
    """How the LLM understood the query (returned when use_understanding=true)."""
    intent_type: Literal['recommendation', 'factual', 'exploratory']
    original_query: str
    reformulated_query: str
    extracted_filters: SearchFilters
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    query_variations: list[str] = Field(default_factory=list)


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
    query_intent: QueryIntent | None = Field(
        default=None,
        description="Query understanding result (use_understanding=true)"
    )
    notices: list[str] = Field(
        default_factory=list,
        description=(
            "Codes of steps skipped for this request: 'search_budget_spent' (the day's budget of "
            "model-backed searches is spent: lexical search without re-ranking), "
            "'llm_budget_spent' (the day's LLM budget is spent: no understanding or explanations) and "
            "'llm_unavailable' (understanding or explanations were asked for but this server has no "
            "LLM configured, e.g. no OPENROUTER_API_KEY: plain search)"
        ),
    )

# ---------------------------------------------------------------------------
# Evaluation and item-to-item recommendation
# ---------------------------------------------------------------------------

EvaluationMode = Literal['lexical_only', 'vector_only', 'hybrid_rrf', 'hybrid_rrf_mmr']


class EvaluationRequest(BaseModel):
    """Request body for POST /evaluate."""
    modes: list[EvaluationMode] = Field(
        default_factory=lambda: ['lexical_only', 'vector_only', 'hybrid_rrf', 'hybrid_rrf_mmr'],
        min_length=1,
        description="Retrieval modes to evaluate",
    )
    max_results: int = Field(default=100, ge=10, le=100, description="Ranked list depth per query")
    mmr_lambda: float = Field(default=0.6, ge=0.0, le=1.0, description="MMR trade-off for hybrid_rrf_mmr")


class EvaluationResponse(BaseModel):
    """Aggregate IR metrics per retrieval mode on the bundled test collection."""
    num_queries: int = Field(description="Test queries evaluated")
    num_queries_with_relevant: int = Field(
        description="Queries with at least one judged-relevant book (the others score 0)"
    )
    latency_ms: float = Field(description="Time to run the evaluation")
    modes: dict[str, dict[str, float]] = Field(
        description="Per mode: ndcg_at_k, recall_at_k, precision_at_k, mrr, ild_at_k, ild_coverage_at_k"
    )


class SimilarBooksResponse(BaseModel):
    """Books similar to a given book (item-to-item vector similarity)."""
    book_id: UUID = Field(description="The book the recommendations are for")
    results: list[SearchResult] = Field(description="Similar books, most similar first")


# ---------------------------------------------------------------------------
# Agentic search (Phase 4)
# ---------------------------------------------------------------------------


class AgentAskRequest(BaseModel):
    """Request body for POST /agent/ask."""
    question: str = Field(min_length=3, max_length=500, description="Book request in natural language")


class AgentStep(BaseModel):
    """One tool call made by the agent."""
    tool: str
    arguments: dict
    result_count: int = Field(ge=0, description="Books returned by the tool")


class AgentRecommendation(BaseModel):
    """A recommended book (always one returned by a tool) and the agent's reason."""
    book: Book
    reason: str


class AgentAskResponse(BaseModel):
    """Answer of the agentic search with its verified recommendations and trace."""
    question: str
    answer: str
    recommendations: list[AgentRecommendation]
    steps: list[AgentStep] = Field(description="Tool calls in order (ReAct trace)")
    model: str
    dropped_recommendations: int = Field(
        ge=0, description="Recommendations removed because no tool had returned that book"
    )
    reflections: int = Field(ge=0, description="Re-searches after a failed verification")
    latency_ms: float
