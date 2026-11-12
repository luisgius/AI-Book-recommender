"""
API endpoints for evaluation and item-to-item recommendation.

- POST /evaluate runs the retrieval evaluation on the bundled test collection.
- GET /books/{book_id}/similar returns books similar to a given book (RF-02).

Both only orchestrate: the work is done by app.evaluation and SearchService.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.api.v1 import schemas as api
from app.api.v1.converters import domain_search_result_to_api
from app.api.rate_limiting import SEARCH
from app.api.v1.dependencies import (
    get_bm25_repository,
    get_catalog_repository,
    get_embeddings_store,
    get_evaluate_settings,
    get_rest_rate_limits,
    get_search_service,
    get_vector_repository,
)
from app.domain.ports import (
    BookCatalogRepository,
    EmbeddingsStore,
    LexicalSearchRepository,
    VectorSearchRepository,
)
from app.domain.services import SearchService
from app.domain.value_objects import SearchFilters

router = APIRouter()


def require_evaluate_enabled() -> None:
    """
    404 when POST /evaluate is switched off (EVALUATE_ENDPOINT_ENABLED=false).

    Checked before any other dependency, so a disabled endpoint builds
    nothing and calls no model.
    """
    if not get_evaluate_settings().enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


@router.post("/evaluate", response_model=api.EvaluationResponse, dependencies=[Depends(require_evaluate_enabled)])
def evaluate(
    request: api.EvaluationRequest,
    service: SearchService = Depends(get_search_service),
    lexical_repo: LexicalSearchRepository = Depends(get_bm25_repository),
    vector_repo: VectorSearchRepository = Depends(get_vector_repository),
    embeddings_store: EmbeddingsStore = Depends(get_embeddings_store),
    catalog_repo: BookCatalogRepository = Depends(get_catalog_repository),
) -> api.EvaluationResponse:
    """
    Evaluate retrieval quality on the bundled test collection.

    Runs the requested retrieval modes (BM25, FAISS, hybrid RRF, hybrid RRF +
    MMR) over the test queries and returns nDCG@k, Recall@k, Precision@k,
    MRR and ILD@k per mode. No LLM is called; LLM-based evaluation (query
    understanding, LLM-as-judge) runs from the CLI evaluation job.

    A public deployment switches it off (EVALUATE_ENDPOINT_ENABLED=false: 404),
    because one call runs every test query.
    """
    # Imported here: the evaluation code is not needed to serve searches, and
    # importing it at startup would load the local adapters in every profile.
    from app.evaluation.retrieval_evaluation import run_retrieval_evaluation

    try:
        report = run_retrieval_evaluation(
            search_service=service,
            lexical_repo=lexical_repo,
            vector_repo=vector_repo,
            embeddings_store=embeddings_store,
            catalog_repo=catalog_repo,
            modes=request.modes,
            max_results=request.max_results,
            mmr_lambda=request.mmr_lambda,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Evaluation data not available: {e}",
        )
    return api.EvaluationResponse(**report)


@router.get("/books/{book_id}/similar", response_model=api.SimilarBooksResponse)
def similar_books(
    book_id: UUID,
    http_request: Request,
    max_results: int = Query(default=10, ge=1, le=50),
    language: str | None = Query(default=None, min_length=2, max_length=2),
    use_diversification: bool = False,
    service: SearchService = Depends(get_search_service),
) -> api.SimilarBooksResponse:
    """
    Recommend books similar to a given book (item-to-item, vector similarity).

    No model API is called (the book's stored vector is the query); it
    counts toward the per-client search limit of a public deployment.

    Raises:
        404: The book is unknown or has no embedding in the vector index
        429: The client is over SEARCH_RATE_LIMIT_PER_MINUTE
    """
    get_rest_rate_limits().enforce(SEARCH, http_request)
    try:
        results = service.find_similar_books(
            book_id=book_id,
            max_results=max_results,
            filters=SearchFilters(language=language) if language else None,
            use_diversification=use_diversification,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return api.SimilarBooksResponse(
        book_id=book_id,
        results=[domain_search_result_to_api(r) for r in results],
    )
