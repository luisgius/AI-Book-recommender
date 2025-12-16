"""
Domain entities for the book recommendation system.

Entities are objects with a unique identity that runs through time and
different representations. They are the core building blocks of the domain.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional, List
from uuid import UUID, uuid4

from .value_objects import BookMetadata


@dataclass
class Book:
    """
    Represents a book in the catalog.

    This is the central entity of the domain. A book has a unique identity
    and contains all the information needed for search, recommendation,
    and display.
    """

    id: UUID
    """Unique identifier for this book in our system"""

    title: str
    """Book title"""

    authors: List[str]
    """List of author names"""

    description: Optional[str] = None
    """Book description/summary"""

    language: Optional[str] = None
    """ISO 639-1 language code (e.g., 'es', 'en')"""

    categories: List[str] = field(default_factory=list)
    """List of categories/genres"""

    published_date: Optional[datetime] = None
    """Publication date (may be partial, e.g., year only)"""

    source: str = "unknown"
    """External source of this book (e.g., 'google_books', 'open_library')"""

    source_id: Optional[str] = None
    """ID in the external source system"""

    metadata: Optional[BookMetadata] = None
    """Additional optional metadata"""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When this book was added to our catalog"""

    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When this book was last updated"""

    def __post_init__(self) -> None:
        """Validate book data."""
        if not self.title or not self.title.strip():
            raise ValueError("Book title cannot be empty")

        if not self.authors:
            raise ValueError("Book must have at least one author")

        if self.language and len(self.language) != 2:
            raise ValueError(
                f"language must be a 2-letter ISO 639-1 code, got '{self.language}'"
            )

    def __eq__(self, other: object) -> bool:
        """Two books are equal if they have the same ID."""
        if not isinstance(other, Book):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on book ID."""
        return hash(self.id)

    def get_published_year(self) -> Optional[int]:
        """Extract the publication year if available."""
        if self.published_date:
            return self.published_date.year
        return None

    def has_description(self) -> bool:
        """Check if book has a non-empty description."""
        return bool(self.description and self.description.strip())

    def get_searchable_text(self) -> str:
        """
        Get all searchable text for this book concatenated.

        This is used for building search indices.
        """
        parts = [
            self.title,
            " ".join(self.authors),
        ]

        if self.description:
            parts.append(self.description)

        if self.categories:
            parts.append(" ".join(self.categories))

        return " ".join(parts)

    @staticmethod
    def create_new(
        title: str,
        authors: List[str],
        source: str = "unknown",
        **kwargs,
    ) -> "Book":
        """
        Factory method to create a new book with auto-generated ID.

        Args:
            title: Book title
            authors: List of author names
            source: Source system identifier
            **kwargs: Additional book attributes

        Returns:
            A new Book instance with generated UUID
        """
        return Book(
            id=uuid4(),
            title=title,
            authors=authors,
            source=source,
            **kwargs,
        )


@dataclass
class SearchResult:
    """
    Represents a single search result linking a book to its relevance scores.

    This entity represents the output of a search operation. It associates
    a book with metadata about why/how it matched the query.

    For hybrid search, this entity tracks both component scores (lexical and vector)
    as well as the final combined score used for ranking.
    """

    book: Book
    """The book that matched the query"""

    final_score: float
    """
    Final relevance score used for ranking (higher is better).
    For hybrid search, this is the RRF combined score.
    For single-method search, this equals lexical_score or vector_score.
    """

    rank: int
    """Position in the result list (1-indexed)"""

    source: str
    """
    Which search method produced this result.
    Values: 'lexical' (BM25), 'vector' (embeddings), 'hybrid' (RRF fusion)
    """

    lexical_score: Optional[float] = None
    """
    BM25 relevance score from lexical search (if available).
    Higher values indicate better keyword match.
    """

    vector_score: Optional[float] = None
    """
    Cosine similarity score from vector search (if available).
    Typically in range [-1, 1] or [0, 1] depending on normalization.
    Higher values indicate better semantic similarity.
    """

    explanation: Optional[str] = None
    """Optional natural language explanation of relevance (generated by LLM)"""

    def __post_init__(self) -> None:
        """Validate search result data."""
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")

        if self.source not in ["lexical", "vector", "hybrid"]:
            raise ValueError(
                f"source must be 'lexical', 'vector', or 'hybrid', got '{self.source}'"
            )

    def __eq__(self, other: object) -> bool:
        """Two results are equal if they refer to the same book with same rank."""
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.book.id == other.book.id and self.rank == other.rank

    def has_explanation(self) -> bool:
        """Check if this result has an explanation."""
        return bool(self.explanation and self.explanation.strip())

    def has_lexical_score(self) -> bool:
        """Check if this result has a lexical (BM25) score."""
        return self.lexical_score is not None

    def has_vector_score(self) -> bool:
        """Check if this result has a vector (semantic) score."""
        return self.vector_score is not None

    def is_hybrid(self) -> bool:
        """Check if this result comes from hybrid fusion."""
        return self.source == "hybrid"


@dataclass
class Explanation:
    """
    Represents an LLM-generated explanation for why a book is relevant.

    This entity encapsulates the output of the explanation service.
    """

    book_id: UUID
    """ID of the book being explained"""

    query_text: str
    """The original search query"""

    text: str
    """The generated explanation"""

    model: str = "unknown"
    """Which LLM model generated this explanation"""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When this explanation was generated"""

    def __post_init__(self) -> None:
        """Validate explanation data."""
        if not self.text or not self.text.strip():
            raise ValueError("Explanation text cannot be empty")

        if not self.query_text or not self.query_text.strip():
            raise ValueError("Query text cannot be empty")

    def get_short_summary(self, max_length: int = 100) -> str:
        """
        Get a shortened version of the explanation.

        Args:
            max_length: Maximum character length

        Returns:
            Truncated explanation with ellipsis if needed
        """
        if len(self.text) <= max_length:
            return self.text

        return self.text[: max_length - 3].strip() + "..."
