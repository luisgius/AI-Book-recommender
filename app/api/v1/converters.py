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
    metadata = book.metadata
    descriptions = getattr(metadata, "descriptions", ()) or ()
    primary = descriptions[0] if descriptions else None
    book_dict["description_source"] = (
        api.DescriptionSource(
            source=primary.source,
            language=primary.language,
            license=primary.license,
            attribution_url=primary.attribution_url,
            auto_generated=primary.auto_generated,
        )
        if primary is not None
        else None
    )
    book_dict["provenance"] = [
        api.ProvenanceRecord(source=r.source, record_id=r.record_id, license=r.license, url=r.url)
        for r in getattr(metadata, "provenance", ()) or ()
    ]
    return api.Book(**book_dict)


def domain_explanation_to_api(explanation: domain.Explanation) -> api.Explanation:
    """
    Convert a domain Explanation (with its verified citations) to the API model.

    Args:
        explanation: Domain Explanation entity

    Returns:
        API Explanation model
    """
    return api.Explanation(
        text=explanation.text,
        citations=[
            api.Citation(
                field=citation.chunk_id,
                snippet=citation.snippet,
                relevance_score=citation.relevance_score,
            )
            for citation in explanation.citations
        ],
        grounded=explanation.is_grounded(),
        model=explanation.model,
    )


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
        explanation=(
            domain_explanation_to_api(result.explanation)
            if result.explanation is not None
            else None
        ),
        rerank_score=result.rerank_score,
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
        timings_ms=dict(metadata.timings_ms),
    )


def domain_query_intent_to_api(intent: domain_vo.QueryIntent) -> api.QueryIntent:
    """
    Convert a domain QueryIntent to the API model.

    Args:
        intent: Domain QueryIntent value object

    Returns:
        API QueryIntent model
    """
    filters = intent.extracted_filters
    return api.QueryIntent(
        intent_type=intent.intent_type,
        original_query=intent.original_query,
        reformulated_query=intent.reformulated_query,
        extracted_filters=api.SearchFilters(
            language=filters.language,
            category=filters.category,
            min_year=filters.min_year,
            max_year=filters.max_year,
        ),
        confidence=intent.confidence,
        reasoning=intent.reasoning,
        query_variations=list(intent.query_variations),
    )


def domain_response_to_api(
    response: domain_vo.SearchResponse,
    *,
    include_metadata: bool = False,
    query_intent: domain_vo.QueryIntent | None = None,
) -> api.SearchResponse:
    """
    Convert a domain SearchResponse value object to an API SearchResponse model.

    Args:
        response: Domain SearchResponse value object
        include_metadata: Whether to include debug metadata in the response
        query_intent: Query understanding result, if the LLM path was used

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
        query_intent=(
            domain_query_intent_to_api(query_intent) if query_intent is not None else None
        ),
        notices=list(response.notices),
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


def api_messages_to_domain(messages: list[api.ChatMessage]) -> list[domain_vo.ChatMessage]:
    """Convert the conversation of an assistant request to domain ChatMessages."""
    return [domain_vo.ChatMessage(role=m.role, content=m.content, route=m.route) for m in messages]


def domain_reply_to_api(reply: domain.AssistantReply, latency_ms: float) -> api.AssistantChatResponse:
    """
    Convert a domain AssistantReply to the API response.

    Args:
        reply: Domain AssistantReply
        latency_ms: Time the endpoint took
    """
    response = reply.search_response
    filters = reply.filters
    usage = reply.usage or {}
    return api.AssistantChatResponse(
        route=reply.route,
        message=reply.message,
        options=list(reply.options),
        search_query=reply.search_query,
        filters=(
            api.SearchFilters(language=filters.language, category=filters.category,
                              min_year=filters.min_year, max_year=filters.max_year)
            if filters is not None else None
        ),
        results=[domain_search_result_to_api(r) for r in response.results] if response is not None else [],
        degraded=response.degraded if response is not None else False,
        recommendations=[
            api.AssistantRecommendation(
                book=domain_book_to_api(r.book),
                reason=r.reason,
                citations=[
                    api.Citation(field=c.chunk_id, snippet=c.snippet, relevance_score=c.relevance_score)
                    for c in r.citations
                ],
            )
            for r in reply.recommendations
        ],
        steps=[
            api.AgentStep(tool=s.tool, arguments=s.arguments, result_count=s.result_count)
            for s in (reply.agent_answer.steps if reply.agent_answer is not None else [])
        ],
        notices=list(reply.notices),
        forced_route=reply.forced_route,
        clarifications=reply.clarifications,
        timings_ms=dict(reply.timings_ms),
        model=reply.model,
        usage=api.AssistantUsage(
            llm_calls=usage.get("llm_calls", 0),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=usage.get("cost_usd"),
        ),
        latency_ms=latency_ms,
    )
