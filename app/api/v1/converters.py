"""
Converters between domain entities/value objects and API schemas.

This module centralizes all conversion logic between the domain layer
and the API layer, maintaining clean separation of concerns.
"""

from dataclasses import asdict

from app.domain import entities as domain
from app.domain import value_objects as domain_vo
from app.api.v1 import schemas as api


def domain_book_to_api(book: domain.Book) -> api.Book:
    """
    Convert a domain Book entity to an API Book model.

    Args:
        book: Domain Book entity

    Returns:
        API Book model
    """
    book_dict = asdict(book)
    return api.Book(**book_dict)


def domain_search_result_to_api(result: domain.SearchResult) -> api.SearchResult:
    """
    Convert a domain SearchResult entity to an API SearchResult model.

    Args:
        result: Domain SearchResult entity

    Returns:
        API SearchResult model
    """
    return api.SearchResult(
        book=domain_book_to_api(result.book),
        final_score=result.final_score,
        rank=result.rank,
        source=result.source,
        lexical_score=result.lexical_score,
        vector_score=result.vector_score,
        explanation=result.explanation,
    )


def domain_metadata_to_api(metadata: domain_vo.SearchMetadata) -> api.SearchMetadata:
    """
    Convert a domain SearchMetadata value object to an API SearchMetadata model.

    Args:
        metadata: Domain SearchMetadata value object

    Returns:
        API SearchMetadata model
    """
    return api.SearchMetadata(
        fusion_method=metadata.fusion_method,
        rrf_k=metadata.rrf_k,
        diversification_enabled=metadata.diversification_enabled,
        candidates_lexical=metadata.candidates_lexical,
        candidates_vector=metadata.candidates_vector,
    )


def domain_response_to_api(
    response: domain_vo.SearchResponse,
    *,
    include_metadata: bool = False,
) -> api.SearchResponse:
    """
    Convert a domain SearchResponse value object to an API SearchResponse model.

    Args:
        response: Domain SearchResponse value object
        include_metadata: Whether to include debug metadata in the response

    Returns:
        API SearchResponse model
    """
    api_results = [domain_search_result_to_api(r) for r in response.results]

    api_metadata = None
    if include_metadata and response.metadata is not None:
        api_metadata = domain_metadata_to_api(response.metadata)

    return api.SearchResponse(
        results=api_results,
        degraded=response.degraded,
        degradation_reason=response.degradation_reason,
        search_mode=response.search_mode,
        latency_ms=response.latency_ms,
        metadata=api_metadata,
    )


def api_filters_to_domain(filters: api.SearchFilters) -> domain_vo.SearchFilters:
    """
    Convert API SearchFilters to domain SearchFilters value object.

    Args:
        filters: API SearchFilters model

    Returns:
        Domain SearchFilters value object
    """
    return domain_vo.SearchFilters(
        language=filters.language,
        category=filters.category,
        min_year=filters.min_year,
        max_year=filters.max_year,
    )


def api_request_to_domain(request: api.SearchRequest) -> domain_vo.SearchQuery:
    """
    Convert API SearchRequest to domain SearchQuery value object.

    Args:
        request: API SearchRequest model

    Returns:
        Domain SearchQuery value object
    """
    domain_filters = (
        api_filters_to_domain(request.filters)
        if request.filters
        else domain_vo.SearchFilters()
    )

    return domain_vo.SearchQuery(
        text=request.text,
        filters=domain_filters,
        max_results=request.max_results if request.max_results is not None else 10,
        use_explanations=request.use_explanations if request.use_explanations is not None else False,
        use_diversification=request.use_diversification if request.use_diversification is not None else False,
        diversity_lambda=request.diversity_lambda if request.diversity_lambda is not None else 0.6,
    )
