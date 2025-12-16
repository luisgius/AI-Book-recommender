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

    def __post_init__(self) -> None:
        """Validate query constraints."""
        if not self.text or not self.text.strip():
            raise ValueError("Search query text cannot be empty")

        if self.max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {self.max_results}")

        if self.max_results > 100:
            raise ValueError(f"max_results cannot exceed 100, got {self.max_results}")


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
