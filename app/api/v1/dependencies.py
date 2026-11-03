"""
FastAPI dependencies for dependency injection.

This module provides singleton instances of repositories and services
for use with FastAPI's Depends() system.

Note: We use module-level singletons instead of @lru_cache with Depends()
parameters, which is an antipattern that can cause unexpected behavior.
"""

import os
from pathlib import Path
from typing import Optional

from app.domain.ports import (
    BookCatalogRepository,
    LexicalSearchRepository,
    VectorSearchRepository,
    EmbeddingsStore,
)
from app.domain.services import SearchService
from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository
from app.infrastructure.search.bm25_search_repository import BM25SearchRepository
from app.infrastructure.search.embeddings_store_faiss import EmbeddingsStoreFaiss
from app.infrastructure.search.faiss_vector_search_repository import FaissVectorSearchRepository

# Configuration from environment
DB_PATH = Path(os.getenv("DB_PATH", "data/catalog.db"))
INDEXES_DIR = Path(os.getenv("INDEXES_DIR", "data/indexes"))

# Module-level singletons (initialized lazily)
_catalog_repository: Optional[BookCatalogRepository] = None
_bm25_repository: Optional[LexicalSearchRepository] = None
_embeddings_store: Optional[EmbeddingsStore] = None
_vector_repository: Optional[VectorSearchRepository] = None
_search_service: Optional[SearchService] = None


def get_catalog_repository() -> BookCatalogRepository:
    """Provide a singleton instance of the catalog repository."""
    global _catalog_repository
    if _catalog_repository is None:
        _catalog_repository = SqliteBookCatalogRepository(DB_PATH)
    return _catalog_repository


def get_bm25_repository() -> LexicalSearchRepository:
    """Provide a singleton instance of the BM25 repository."""
    global _bm25_repository
    if _bm25_repository is None:
        bm25_path = INDEXES_DIR / "bm25_index.pkl"
        repo = BM25SearchRepository()
        if bm25_path.exists():
            repo.load_index(str(bm25_path))
        _bm25_repository = repo
    return _bm25_repository


def get_embeddings_store() -> EmbeddingsStore:
    """Provide a singleton instance of the embeddings store."""
    global _embeddings_store
    if _embeddings_store is None:
        faiss_path = INDEXES_DIR / "faiss_index"
        store = EmbeddingsStoreFaiss()
        if faiss_path.exists():
            store.load_index(str(faiss_path))
        _embeddings_store = store
    return _embeddings_store


def get_vector_repository() -> VectorSearchRepository:
    """Provide a singleton instance of the vector search repository."""
    global _vector_repository
    if _vector_repository is None:
        catalog_repo = get_catalog_repository()
        embeddings_store = get_embeddings_store()
        books = catalog_repo.get_all()
        _vector_repository = FaissVectorSearchRepository(embeddings_store, books)
    return _vector_repository


def get_search_service() -> SearchService:
    """Provide the Search Service with all dependencies wired."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService(
            lexical_search=get_bm25_repository(),
            vector_search=get_vector_repository(),
            embeddings_store=get_embeddings_store(),
        )
    return _search_service


def reset_dependencies() -> None:
    """
    Reset all singletons. Useful for testing.

    This allows tests to inject mock dependencies by resetting
    the module state between test cases.
    """
    global _catalog_repository, _bm25_repository, _embeddings_store
    global _vector_repository, _search_service

    _catalog_repository = None
    _bm25_repository = None
    _embeddings_store = None
    _vector_repository = None
    _search_service = None
