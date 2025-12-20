"""
Value objects for the domain layer.

Value objects are immutable objects that represent descriptive aspects
of the domain with no conceptual identity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


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
class BookMetadata:
    """
    Additional metadata for a book.

    This is optional information that may not be available for all books.
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
