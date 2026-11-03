"""
API endpoints for book search operations.

This module defines the FastAPI routes for searching books and retrieving
book details. It handles HTTP concerns and delegates to domain services.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID

from app.domain.ports import BookCatalogRepository
from app.domain.services import SearchService
from app.api.v1 import schemas as api
from app.api.v1.converters import (
    api_request_to_domain,
    domain_response_to_api,
    domain_book_to_api,
)
from app.api.v1.dependencies import get_search_service, get_catalog_repository

router = APIRouter()


@router.post("/search", response_model=api.SearchResponse)
def search_books(
    request: api.SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> api.SearchResponse:
    """
    Search for books using hybrid search (lexical + vector).

    This endpoint performs a hybrid search combining BM25 lexical search
    with semantic vector search, using Reciprocal Rank Fusion (RRF) to
    merge results.

    Args:
        request: Search request with query text, filters, and options

    Returns:
        SearchResponse with ranked results and optional metadata
    """
    try:
        include_metadata = bool(request.include_metadata)

        domain_query = api_request_to_domain(request)
        domain_response = service.search_with_fallback(domain_query)

        return domain_response_to_api(
            domain_response,
            include_metadata=include_metadata,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.get("/books/{book_id}", response_model=api.Book)
def get_book_by_id(
    book_id: UUID,
    catalog_repo: BookCatalogRepository = Depends(get_catalog_repository),
) -> api.Book:
    """
    Get a book by its unique identifier.

    Args:
        book_id: UUID of the book to retrieve

    Returns:
        Book details

    Raises:
        404: Book not found
    """
    book = catalog_repo.get_by_id(book_id)
    if book is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with id '{book_id}' not found",
        )

    return domain_book_to_api(book)


@router.get("/health")
def health_check(
    service: SearchService = Depends(get_search_service),
) -> dict:
    """
    Check system health and component readiness (RNF-06).

    Returns the status of all search components:
    - lexical_search: BM25 index availability
    - vector_search: FAISS index availability
    - embeddings_store: Embedding model availability
    - overall: True only if all components are ready
    """
    health_status = service.get_health_status()

    return {
        "status": "ok" if health_status["overall"] else "degraded",
        "components": {
            "lexical_search": health_status["lexical_search"],
            "vector_search": health_status["vector_search"],
            "embeddings_store": health_status["embeddings_store"],
        },
        "overall": health_status["overall"],
    }
