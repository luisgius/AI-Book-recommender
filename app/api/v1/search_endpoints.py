"""
API endpoints for book search operations.

This module defines the FastAPI routes for searching books and retrieving
book details. It handles HTTP concerns and delegates to domain services.
"""

import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.domain.ports import BookCatalogRepository
from app.domain.services import SearchService
from app.api.v1 import schemas as api
from app.api.v1.converters import (
    api_request_to_domain,
    domain_response_to_api,
    domain_book_to_api,
)
from app.api.rate_limiting import LLM, SEARCH
from app.api.v1.dependencies import (
    database_health,
    get_budget_settings,
    get_rest_rate_limits,
    get_search_service,
    get_catalog_repository,
    get_llm_settings,
    get_search_profile_settings,
)

router = APIRouter()

# When this process started (health reports the uptime, which tells whether a
# request was served by a freshly started serverless instance).
PROCESS_STARTED = time.time()


@router.post("/search", response_model=api.SearchResponse)
def search_books(
    request: api.SearchRequest,
    http_request: Request,
    service: SearchService = Depends(get_search_service),
) -> api.SearchResponse:
    """
    Search for books.

    The candidates come from BM25 and vector search fused with Reciprocal
    Rank Fusion (SEARCH_CANDIDATES=hybrid, the default) or from the vector
    search alone (SEARCH_CANDIDATES=vector); the reranker, when configured,
    then orders them. If the embedding or the vector search fails, or the daily
    budget of model-backed searches is spent, BM25 alone answers and the
    response says so (degraded, search_mode "lexical_only").

    With use_understanding=true the LLM first interprets the query (intent,
    filters, reformulation) and the search runs over several query
    variations (RAG retrieval step). With use_explanations=true the top
    results get grounded explanations (RAG generation step). Without a
    configured LLM both options degrade to the plain search.

    Args:
        request: Search request with query text, filters, and options

    Returns:
        SearchResponse with ranked results, optional metadata and, when the
        LLM path is used, the understood query intent

    A public deployment limits searches per client (SEARCH_RATE_LIMIT_PER_MINUTE)
    and, for searches that use the LLM, LLM requests per client
    (LLM_RATE_LIMIT_PER_MINUTE): above them the answer is 429. The daily budget
    may degrade the search; `notices` then says which step was skipped.
    """
    limits = get_rest_rate_limits()
    limits.enforce(SEARCH, http_request)
    if request.use_understanding or request.use_explanations:
        limits.enforce(LLM, http_request)
    try:
        include_metadata = bool(request.include_metadata)

        domain_query = api_request_to_domain(request)
        query_intent = None
        if request.use_understanding:
            domain_response, query_intent = service.search_with_understanding(
                query_text=domain_query.text,
                max_results=domain_query.max_results,
                use_explanations=domain_query.use_explanations,
                use_diversification=domain_query.use_diversification,
                diversity_lambda=domain_query.diversity_lambda,
                filters=domain_query.filters,
            )
        else:
            domain_response = service.search_with_fallback(domain_query)

        return domain_response_to_api(
            domain_response,
            include_metadata=include_metadata,
            query_intent=query_intent,
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

    The LLM and the reranker are reported separately because they are
    optional: without the LLM the search still works, only understanding and
    explanations are skipped; without the reranker (or if it fails) the
    fused order is returned. The reranker model loads on the first search,
    so "ready" only turns false after a failed load.

    "profile" names the adapters in use: "local" (SQLite, BM25, FAISS) or
    "cloud" (Postgres schema, lexical ranking and vector search mode), and
    the candidate mode ("candidates": "vector" or "hybrid"), which the web
    page reads to describe the pipeline this server runs.
    "database" (cloud profile) says whether Postgres answers a SELECT 1 now;
    "budget" lists the daily limits of a public deployment (None: no limit);
    "uptime_s" is the age of this process.
    """
    health_status = service.get_health_status()
    llm_settings = get_llm_settings()
    database = database_health()
    budget = get_budget_settings()

    overall = health_status["overall"] and (database is None or database["reachable"])
    return {
        "status": "ok" if overall else "degraded",
        "profile": get_search_profile_settings().describe(),
        "database": database,
        "budget": budget.limits() if budget.enabled else None,
        "uptime_s": round(time.time() - PROCESS_STARTED, 1),
        "components": {
            "lexical_search": health_status["lexical_search"],
            "vector_search": health_status["vector_search"],
            "embeddings_store": health_status["embeddings_store"],
        },
        "llm": {
            "configured": service.has_llm(),
            "provider": llm_settings.provider,
            "model": llm_settings.model if llm_settings.enabled else None,
        },
        "reranker": {
            "configured": service.has_reranker(),
            "model": service.get_reranker_model_name(),
            "ready": health_status["reranker"],
        },
        "overall": overall,
    }
