"""
Domain services package.

Services orchestrate domain logic that doesn't naturally belong to a single
entity. They coordinate between entities and ports to implement use cases.

Following Hexagonal Architecture principles, services depend only on domain
entities, value objects, and port protocols (never on concrete implementations).
"""

from .search_service import SearchService
from .catalog_ingestion_service import CatalogIngestionService

__all__ = [
    "SearchService",
    "CatalogIngestionService",
]
