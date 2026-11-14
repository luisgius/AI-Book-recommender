"""
Value objects for the domain layer.

Value objects are immutable objects that represent descriptive aspects
of the domain with no conceptual identity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Literal, Tuple


class RetrievalStrategy(Enum):
    """
    Retrieval strategy that controls the balance between lexical and vector search.

    The strategy determines candidate pool sizes for each retrieval method before
    RRF fusion. Since RRF is rank-based, we influence the final ranking by adjusting
    how many candidates come from each source.

    Mapping from intent:
    - recommendation -> VECTOR_HEAVY (similarity is semantic)
    - factual -> LEXICAL_HEAVY (exact keyword matches matter)
    - exploratory -> BALANCED (cast a wide net)
    """
    LEXICAL_HEAVY = "lexical_heavy"
    VECTOR_HEAVY = "vector_heavy"
    BALANCED = "balanced"


# Strategy-to-pool-size mapping.
# Factors are multiplied by max_results to get candidate counts.
# Example with max_results=10:
#   LEXICAL_HEAVY -> BM25 gets 40 candidates, FAISS gets 10
#   VECTOR_HEAVY  -> BM25 gets 10, FAISS gets 40
#   BALANCED      -> both get 20 (same as current behavior)
STRATEGY_POOL_SIZES = {
    RetrievalStrategy.LEXICAL_HEAVY: {"bm25_factor": 4, "vector_factor": 1},
    RetrievalStrategy.VECTOR_HEAVY: {"bm25_factor": 1, "vector_factor": 4},
    RetrievalStrategy.BALANCED: {"bm25_factor": 2, "vector_factor": 2},
}

# Maps intent type to retrieval strategy
INTENT_TO_STRATEGY = {
    "recommendation": RetrievalStrategy.VECTOR_HEAVY,
    "factual": RetrievalStrategy.LEXICAL_HEAVY,
    "exploratory": RetrievalStrategy.BALANCED,
}

# Candidate stage of SearchService: where the candidates that are filtered,
# re-ranked and cut to the top k come from.
# - CANDIDATES_HYBRID: BM25 and vector search fused with RRF (k=60, equal
#   weights), the default (DEFAULT_CANDIDATES below).
# - CANDIDATES_VECTOR: vector search only. Keyword search (BM25) runs only as
#   the fallback, when the embedding or the vector search fails, or when the
#   day's budget of model-backed searches is spent. Selectable by
#   configuration (SEARCH_CANDIDATES=vector in the application).
# In both modes the reranker, when configured, orders the candidates.
# The strategy of an understood query (STRATEGY_POOL_SIZES) only applies to
# the hybrid mode: without a keyword list there is no balance to set.
CANDIDATES_VECTOR = "vector"
CANDIDATES_HYBRID = "hybrid"
CANDIDATE_MODES = (CANDIDATES_VECTOR, CANDIDATES_HYBRID)

# The one default candidate mode (V30). SearchService takes it as the default
# of its constructor and the application's settings (SEARCH_CANDIDATES unset)
# take it from here, so the domain and the application cannot disagree.
# Why hybrid: on descriptive queries of the deployed catalog (V14,
# data/evaluation/CATALOG_EVALUATION.md) the BM25 candidates cost 0.029
# nDCG@10 under the reranker (LLM judges), but on queries that name an author
# (V29, data/evaluation/known_item/KNOWN_ITEM.md, ground truth taken from the
# catalog, no judge) they add 0.067 (Holm p < 0.001), and on titles they make
# no difference. The author weighed both and kept the hybrid fusion.
DEFAULT_CANDIDATES = CANDIDATES_HYBRID


@dataclass(frozen=True)
class SearchFilters:
    """
    Filters that can be applied to a search query.

    All filters are optional. When a filter is None, it means "no restriction".
    """

    language: Optional[str] = None
    """ISO 639-1 language code (e.g., 'es', 'en', 'fr')"""

    category: Optional[str] = None
    """Book category/genre (e.g., 'Fiction', 'Science', 'History')"""

    min_year: Optional[int] = None
    """Minimum publication year (inclusive)"""

    max_year: Optional[int] = None
    """Maximum publication year (inclusive)"""

    def __post_init__(self) -> None:
        """Validate filter constraints."""
        if self.min_year is not None and self.max_year is not None:
            if self.min_year > self.max_year:
                raise ValueError(
                    f"min_year ({self.min_year}) cannot be greater than "
                    f"max_year ({self.max_year})"
                )

        if self.language is not None and len(self.language) != 2:
            raise ValueError(
                f"language must be a 2-letter ISO 639-1 code, got '{self.language}'"
            )

    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return all(
            getattr(self, field_name) is None
            for field_name in ["language", "category", "min_year", "max_year"]
        )

    def overridden_by(self, explicit: Optional["SearchFilters"]) -> "SearchFilters":
        """
        Combine these filters with explicit ones; explicit values win.

        Used when the LLM extracts filters from the query text and the user
        also set filters in the request: what the user chose explicitly takes
        precedence, field by field. If the combination is contradictory
        (min_year > max_year), the explicit filters are returned unchanged.

        Args:
            explicit: Filters set explicitly by the user (may be None)

        Returns:
            A new SearchFilters instance
        """
        if explicit is None or explicit.is_empty():
            return self

        def pick(field_name: str):
            value = getattr(explicit, field_name)
            return value if value is not None else getattr(self, field_name)

        try:
            return SearchFilters(
                language=pick("language"),
                category=pick("category"),
                min_year=pick("min_year"),
                max_year=pick("max_year"),
            )
        except ValueError:
            return explicit


@dataclass(frozen=True)
class SearchQuery:
    """
    Represents a user's search query with optional filters.

    This is the input to the search service.
    """

    text: str
    """The raw search query text from the user"""

    filters: SearchFilters = field(default_factory=SearchFilters)
    """Optional filters to refine the search"""

    max_results: int = 10
    """Maximum number of results to return"""

    use_explanations: bool = False
    """Whether to generate LLM explanations for results"""

    use_diversification: bool = False
    """Whether to apply MMR diversification to reduce redundancy in results"""

    diversity_lambda: float = 0.6
    """Trade-off between relevance and diversity (0.0 = max diversity, 1.0 = max relevance)"""

    def __post_init__(self) -> None:
        """Validate query constraints."""
        if not self.text or not self.text.strip():
            raise ValueError("Search query text cannot be empty")

        if self.max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {self.max_results}")

        if self.max_results > 100:
            raise ValueError(f"max_results cannot exceed 100, got {self.max_results}")

        if not (0.0 <= self.diversity_lambda <= 1.0):
            raise ValueError(
                f"diversity_lambda must be between 0.0 and 1.0, got {self.diversity_lambda}"
            )


@dataclass(frozen=True)
class BookDescription:
    """
    One description of a work in one language, with its origin.

    A work of the multi-source catalog can have several descriptions (for
    example the intro of its English and Spanish Wikipedia articles). Each one
    keeps the source, licence and attribution URL it came with, because some
    texts (Wikipedia, CC BY-SA 4.0) must be attributed per record.
    """

    text: str
    """Plain-text description"""

    language: Optional[str] = None
    """ISO 639-1 code of the text; None until the ingestion detects it"""

    source: str = "unknown"
    """Source of the text (e.g. 'wikipedia', 'gutenberg', 'doab', 'open_library')"""

    license: Optional[str] = None
    """Licence of the text (e.g. 'CC BY-SA 4.0'), when the source states one"""

    attribution_url: Optional[str] = None
    """Page to cite for this text (e.g. the Wikipedia article URL)"""

    auto_generated: bool = False
    """True when the source says the text was generated automatically"""

    def __post_init__(self) -> None:
        """Validate description data."""
        if not self.text or not self.text.strip():
            raise ValueError("Description text cannot be empty")
        if self.language is not None and len(self.language) != 2:
            raise ValueError(
                f"language must be a 2-letter ISO 639-1 code, got '{self.language}'"
            )


@dataclass(frozen=True)
class SourceRecord:
    """
    Provenance of a catalog record: one source record that contributed to it.

    A merged work lists one SourceRecord per contributing record, so the
    catalog can always say where each piece of data came from and under
    which licence.
    """

    source: str
    """Source name (e.g. 'wikidata', 'wikipedia', 'gutenberg', 'doab', 'open_library')"""

    record_id: str
    """Identifier of the record in that source (e.g. 'Q480', 'en:Don Quixote')"""

    license: Optional[str] = None
    """Licence that applies to the data taken from this record"""

    url: Optional[str] = None
    """Public page of the record (also the attribution URL when one is needed)"""


@dataclass(frozen=True)
class SourceLicense:
    """
    Licence and attribution terms of a bulk data source, as verified at download.

    Bulk sources report this so the catalog report can list, per source, what
    may be done with the data and how it must be attributed.
    """

    source: str
    """Source name used in the catalog (e.g. 'doab')"""

    name: str
    """Human-readable name (e.g. 'Directory of Open Access Books')"""

    license: str
    """Licence of the metadata (e.g. 'CC0 1.0')"""

    terms_url: str
    """Page where the licence or terms were checked"""

    attribution: str
    """What must be shown to attribute the data"""

    notes: str = ""
    """Restrictions or remarks (e.g. per-record licences, access rules)"""


@dataclass(frozen=True)
class BookMetadata:
    """
    Additional metadata for a book.

    This is optional information that may not be available for all books.

    The fields after ``preview_link`` were added for the multi-source catalog
    (Wikidata/Wikipedia, Project Gutenberg, DOAB, Open Library). They all
    default to empty values, so books created before them (the original
    Google Books catalog) are unchanged.
    """

    isbn: Optional[str] = None
    """International Standard Book Number"""

    isbn13: Optional[str] = None
    """13-digit ISBN"""

    publisher: Optional[str] = None
    """Publisher name"""

    page_count: Optional[int] = None
    """Number of pages"""

    average_rating: Optional[float] = None
    """Average user rating (0.0 to 5.0)"""

    ratings_count: Optional[int] = None
    """Number of ratings"""

    thumbnail_url: Optional[str] = None
    """URL to book cover image"""

    preview_link: Optional[str] = None
    """URL to preview/info page"""

    # --- Multi-source catalog fields -------------------------------------

    work_language: Optional[str] = None
    """Language of the work as stated by a source (ISO 639-1); None if unknown"""

    descriptions: Tuple[BookDescription, ...] = ()
    """Every description of the work (at most one per language), primary first"""

    titles: Tuple[Tuple[str, str], ...] = ()
    """Titles of the work in other languages, as (language, title) pairs"""

    wikidata_id: Optional[str] = None
    """Wikidata item of the work (e.g. 'Q480')"""

    openlibrary_ids: Tuple[str, ...] = ()
    """Open Library work ids (e.g. 'OL45804W')"""

    gutenberg_ids: Tuple[str, ...] = ()
    """Project Gutenberg e-book numbers (e.g. '2000')"""

    isbn13s: Tuple[str, ...] = ()
    """Every known ISBN-13 of the work (isbn13 keeps the first one)"""

    doi: Optional[str] = None
    """DOI of the book (open access books)"""

    wikipedia_editions: Optional[int] = None
    """Popularity signal: number of Wikipedia language editions with an article"""

    gutenberg_downloads: Optional[int] = None
    """Popularity signal: Project Gutenberg downloads in the last 30 days"""

    provenance: Tuple[SourceRecord, ...] = ()
    """Source records that contributed to this work"""

    def __post_init__(self) -> None:
        """Validate metadata constraints."""
        if self.average_rating is not None:
            if not (0.0 <= self.average_rating <= 5.0):
                raise ValueError(
                    f"average_rating must be between 0.0 and 5.0, "
                    f"got {self.average_rating}"
                )

        if self.page_count is not None and self.page_count < 0:
            raise ValueError(f"page_count cannot be negative, got {self.page_count}")

        if self.ratings_count is not None and self.ratings_count < 0:
            raise ValueError(
                f"ratings_count cannot be negative, got {self.ratings_count}"
            )

        if self.work_language is not None and len(self.work_language) != 2:
            raise ValueError(
                f"work_language must be a 2-letter ISO 639-1 code, got '{self.work_language}'"
            )

    def get_description(self, language: str) -> Optional[BookDescription]:
        """Return the description in the given language, if there is one."""
        for description in self.descriptions:
            if description.language == language:
                return description
        return None

    def description_languages(self) -> List[str]:
        """Languages that have a description, in stored order (primary first)."""
        return [d.language for d in self.descriptions if d.language is not None]

    def sources(self) -> List[str]:
        """Distinct names of the sources that contributed, in provenance order."""
        seen: List[str] = []
        for record in self.provenance:
            if record.source not in seen:
                seen.append(record.source)
        return seen


@dataclass(frozen=True)
class SearchMetadata:
    """
    Debug/telemetry metadata for search operations.

    This value object captures internal details about how a search was executed,
    useful for debugging, evaluation, and transparency in the TFG.
    """

    fusion_method: str
    """The fusion method used: 'rrf' (Reciprocal Rank Fusion) or 'none'"""

    rrf_k: Optional[int] = None
    """The k parameter used in RRF formula (typically 60), None if fusion_method='none'"""

    diversification_enabled: bool = False
    """Whether MMR diversification was applied"""

    candidates_lexical: int = 0
    """Number of candidates retrieved from lexical (BM25) search"""

    candidates_vector: int = 0
    """Number of candidates retrieved from vector search"""

    timings_ms: Dict[str, float] = field(default_factory=dict)
    """Milliseconds spent in each stage (lexical, embedding, vector, fusion,
    rerank, mmr, explanations), so latency can be split by stage (V17)"""

    def __post_init__(self) -> None:
        """Validate metadata constraints."""
        valid_fusion_methods = {"rrf", "none"}
        if self.fusion_method not in valid_fusion_methods:
            raise ValueError(
                f"fusion_method must be one of {valid_fusion_methods}, "
                f"got '{self.fusion_method}'"
            )

        if self.fusion_method == "rrf" and self.rrf_k is None:
            raise ValueError("rrf_k is required when fusion_method='rrf'")

        if self.candidates_lexical < 0:
            raise ValueError(f"candidates_lexical cannot be negative, got {self.candidates_lexical}")

        if self.candidates_vector < 0:
            raise ValueError(f"candidates_vector cannot be negative, got {self.candidates_vector}")


@dataclass(frozen=True)
class SearchResponse:
    """
    Response wrapper for search operations with degradation metadata (RNF-08).

    This value object wraps search results and includes metadata about
    the search execution, particularly for graceful degradation scenarios.
    """

    results: list
    """List of SearchResult entities"""

    degraded: bool = False
    """True if the search was performed in degraded mode (e.g., FAISS unavailable)"""

    degradation_reason: Optional[str] = None
    """Human-readable explanation of why degradation occurred"""

    search_mode: str = "hybrid"
    """The search mode used: 'hybrid', 'lexical_only', or 'vector_only'"""

    latency_ms: Optional[float] = None
    """Search execution time in milliseconds"""

    metadata: Optional[SearchMetadata] = None
    """Optional debug/telemetry metadata about the search execution"""

    notices: Tuple[str, ...] = ()
    """Short messages for the user about features that were skipped (for
    example, explanations refused because the day's LLM budget is spent)"""

    def __post_init__(self) -> None:
        """Validate response constraints."""
        if self.degraded and self.degradation_reason is None:
            raise ValueError("degradation_reason is required when degraded=True")

        valid_modes = {"hybrid", "lexical_only", "vector_only"}
        if self.search_mode not in valid_modes:
            raise ValueError(
                f"search_mode must be one of {valid_modes}, got '{self.search_mode}'"
            )


@dataclass(frozen=True)
class IngestionSummary:
    """
    Summary of an ingestion operation.

    This value object captures the outcome of ingesting books from
    an external provider into the catalog repository.
    """

    n_fetched: int
    """Number of books fetched from the external provider"""

    n_inserted: int
    """Number of books successfully inserted into the catalog"""

    n_skipped: int
    """Number of books skipped (e.g., duplicates)"""

    n_errors: int
    """Number of books that failed to insert due to errors"""

    query: str
    """The query used to fetch books"""

    language: Optional[str] = None
    """Language filter applied (ISO 639-1 code)"""

    errors: list[str] = field(default_factory=list)
    """List of error messages (optional, for debugging)"""

    def __post_init__(self) -> None:
        """Validate summary constraints."""
        if self.n_fetched < 0:
            raise ValueError(f"n_fetched cannot be negative, got {self.n_fetched}")
        if self.n_inserted < 0:
            raise ValueError(f"n_inserted cannot be negative, got {self.n_inserted}")
        if self.n_skipped < 0:
            raise ValueError(f"n_skipped cannot be negative, got {self.n_skipped}")
        if self.n_errors < 0:
            raise ValueError(f"n_errors cannot be negative, got {self.n_errors}")

        # Invariant: fetched = inserted + skipped + errors
        expected_fetched = self.n_inserted + self.n_skipped + self.n_errors
        if self.n_fetched != expected_fetched:
            raise ValueError(
                f"Invariant violated: n_fetched ({self.n_fetched}) must equal "
                f"n_inserted + n_skipped + n_errors ({expected_fetched})"
            )

@dataclass(frozen=True)
class QueryIntent:
    """
    Value object representing the understood intent of a user query.

    This is the result of the query understanding LangGraph flow (Block 2),
    containing the classified intent, extracted filters, and optimized query.

    The intent classification helps the system choose the best search strategy:
    - recommendation: Focus on similarity, use vector search heavily
    - factual: Direct lookup, prioritize exact matches
    - exploratory: Diverse results, balance precision and discovery
    """

    intent_type: Literal["recommendation", "factual", "exploratory"]
    """The classified intent type"""

    original_query: str
    """The raw query text as entered by the user"""

    reformulated_query: str
    """Optimized query for better retrieval (cleaned, expanded, or focused)"""

    extracted_filters: SearchFilters
    """Filters extracted from natural language (language, year, category)"""

    confidence: float
    """LLM's confidence in the classification (0.0 to 1.0)"""

    reasoning: str
    """Explanation of why the LLM classified the query this way"""

    query_variations: list = field(default_factory=list)
    """Alternative query formulations for multi-query retrieval.

    Each variation is searched independently (in parallel), then results are
    fused with RRF across all variations. Books appearing in multiple variation
    rankings get boosted scores.

    If empty, only reformulated_query is used (single-query mode).
    The reformulated_query is always included as the primary variation
    by the SearchService - these are ADDITIONAL variations.
    """

    def __post_init__(self) -> None:
        """Validate query intent constraints."""
        if not self.original_query or not self.original_query.strip():
            raise ValueError("original_query cannot be empty")

        if not self.reformulated_query or not self.reformulated_query.strip():
            raise ValueError("reformulated_query cannot be empty")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        if not self.reasoning or not self.reasoning.strip():
            raise ValueError("reasoning cannot be empty")

        # Note: intent_type is validated by Literal type hint at runtime
        # extracted_filters is validated by SearchFilters.__post_init__

# =============================================================================
# Conversation with the assistant (short clarification before recommending)
# =============================================================================
#
# The server keeps no conversation: the client sends the whole history with
# every request. These limits bound what one request can make the LLM read.

MAX_CONVERSATION_MESSAGES = 10
"""Messages kept from a conversation (the most recent ones); older ones are dropped.
Ten messages are five exchanges, enough for two clarifying questions and a
follow-up, while keeping the routing prompt short."""

MAX_MESSAGE_CHARS = 500
"""Characters kept from each message (the same limit as an agent question)."""

MAX_CLARIFYING_TURNS = 2
"""Clarifying questions the assistant may ask in one conversation; after them it searches."""

ROUTE_CLARIFY = "clarify"
ROUTE_SEARCH = "search"
ROUTE_AGENT = "agent"
ROUTE_OUT_OF_SCOPE = "out_of_scope"
ASSISTANT_ROUTES = (ROUTE_CLARIFY, ROUTE_SEARCH, ROUTE_AGENT, ROUTE_OUT_OF_SCOPE)

ChatRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """
    One turn of a conversation with the assistant, as the client sends it back.

    route is set on assistant turns: the route of the reply that produced the
    turn (the client echoes it). It is what lets a stateless server count the
    clarifying questions already asked.
    """

    role: ChatRole
    content: str
    route: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the role and the route."""
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{self.role}'")
        if self.route is not None and self.route not in ASSISTANT_ROUTES:
            raise ValueError(f"route must be one of {ASSISTANT_ROUTES}, got '{self.route}'")


def bound_conversation(
    messages: List[ChatMessage],
    max_messages: int = MAX_CONVERSATION_MESSAGES,
    max_chars: int = MAX_MESSAGE_CHARS,
) -> List[ChatMessage]:
    """
    The part of a conversation the assistant reads.

    Keeps the last max_messages messages, cuts each one to max_chars
    characters and drops empty ones.

    Raises:
        ValueError: If no user message is left, or the last message is not
            the user's (there is nothing to reply to)
    """
    kept = [
        ChatMessage(role=m.role, content=m.content.strip()[:max_chars], route=m.route)
        for m in messages
        if m.content and m.content.strip()
    ][-max_messages:]
    if not kept or kept[-1].role != "user":
        raise ValueError("the conversation must end with a user message")
    return kept


def count_clarifications(messages: List[ChatMessage]) -> int:
    """
    Clarifying questions the assistant already asked in this conversation.

    Counted over every message the client sent (before bounding), so dropping
    old turns cannot reopen the allowance. An assistant turn without a route
    counts as a clarification: a client that does not echo routes gets fewer
    questions, never more.
    """
    return sum(
        1 for m in messages
        if m.role == "assistant" and (m.route is None or m.route == ROUTE_CLARIFY)
    )


def last_user_message(messages: List[ChatMessage]) -> str:
    """Text of the last user message ("" if there is none)."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""
