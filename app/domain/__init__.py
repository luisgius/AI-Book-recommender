"""
Domain layer - Core business logic and entities.

This layer contains the business entities, value objects, and defines
the ports (interfaces) that the infrastructure layer must implement.

It has NO dependencies on external frameworks, databases, or APIs.
"""

from .entities import Book, SearchResult, Explanation
from .value_objects import SearchQuery, SearchFilters, BookMetadata

__all__ = [
    # Entities
    "Book",
    "SearchResult",
    "Explanation",
    # Value Objects
    "SearchQuery",
    "SearchFilters",
    "BookMetadata",
]
