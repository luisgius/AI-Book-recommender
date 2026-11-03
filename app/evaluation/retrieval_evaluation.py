"""
Retrieval evaluation runner used by POST /api/v1/evaluate.

Runs the retrieval modes of the evaluation job over the bundled test
collection (test_queries.json + relevance_judgments.json) with services that
are already loaded, and returns aggregate IR metrics per mode.

It reuses the evaluation job's own building blocks (query/judgment loading,
UUID resolution, per-mode search and evaluate_mode), so the API and the CLI
compute exactly the same numbers. LLM-based evaluation (query understanding,
LLM-as-judge) stays in the CLI job because it takes minutes per run:
    python -m app.evaluation.evaluation_job --help
"""

import time
from typing import Any, Dict, List, Sequence

from app.domain.ports import (
    BookCatalogRepository,
    EmbeddingsStore,
    LexicalSearchRepository,
    VectorSearchRepository,
)
from app.domain.services import SearchService
from app.evaluation.evaluation_job import (
    DEFAULT_JUDGMENTS_PATH,
    DEFAULT_QUERIES_PATH,
    evaluate_mode,
    load_relevance_judgments,
    load_test_queries,
    resolve_judgments_to_uuids,
    run_hybrid_search,
    run_lexical_search,
    run_vector_search,
)
from app.evaluation.evaluation_service import EvaluationService

RETRIEVAL_MODES = ("lexical_only", "vector_only", "hybrid_rrf", "hybrid_rrf_mmr")


def run_retrieval_evaluation(
    *,
    search_service: SearchService,
    lexical_repo: LexicalSearchRepository,
    vector_repo: VectorSearchRepository,
    embeddings_store: EmbeddingsStore,
    catalog_repo: BookCatalogRepository,
    modes: Sequence[str] = RETRIEVAL_MODES,
    max_results: int = 100,
    mmr_lambda: float = 0.6,
    mmr_top_k: int = 20,
    queries_path: str = DEFAULT_QUERIES_PATH,
    judgments_path: str = DEFAULT_JUDGMENTS_PATH,
) -> Dict[str, Any]:
    """
    Evaluate the requested retrieval modes on the test collection.

    Args:
        search_service: Hybrid search service (no LLM calls are made)
        lexical_repo: BM25 repository (lexical_only mode)
        vector_repo: FAISS repository (vector_only mode)
        embeddings_store: Embeddings store (query embeddings and MMR)
        catalog_repo: Catalog used to resolve judgments to book UUIDs
        modes: Subset of RETRIEVAL_MODES to run
        max_results: Ranked list depth per query (metrics use cut-offs <= 100)
        mmr_lambda: Relevance/diversity trade-off for hybrid_rrf_mmr
        mmr_top_k: How many top results MMR re-ranks
        queries_path: Test queries JSON
        judgments_path: Relevance judgments JSON

    Returns:
        Dict with num_queries, num_queries_with_relevant, latency_ms and, per
        mode, the aggregate metrics (nDCG@k, Recall@k, Precision@k, MRR, ILD@k).

    Raises:
        ValueError: If an unknown mode is requested
    """
    unknown = [m for m in modes if m not in RETRIEVAL_MODES]
    if unknown:
        raise ValueError(f"Unknown evaluation modes {unknown}; choose from {list(RETRIEVAL_MODES)}")

    start = time.time()
    queries = load_test_queries(queries_path)
    judgments = resolve_judgments_to_uuids(load_relevance_judgments(judgments_path), catalog_repo)
    eval_service = EvaluationService()

    def search(mode: str, query_text: str) -> List:
        if mode == "lexical_only":
            return run_lexical_search(query_text, lexical_repo, max_results=max_results)
        if mode == "vector_only":
            return run_vector_search(query_text, vector_repo, embeddings_store, max_results=max_results)
        return run_hybrid_search(
            query_text,
            search_service,
            embeddings_store,
            max_results=max_results,
            use_diversification=(mode == "hybrid_rrf_mmr"),
            diversity_lambda=mmr_lambda,
            mmr_top_k=mmr_top_k,
        )

    per_mode: Dict[str, Dict[str, float]] = {}
    num_with_relevant = 0
    for mode in modes:
        results_by_query = {q.query_id: search(mode, q.text) for q in queries}
        metrics = evaluate_mode(mode, results_by_query, judgments, eval_service)
        per_mode[mode] = {
            name: value
            for name, value in metrics.items()
            if isinstance(value, float) and not name.startswith("n_")
        }
        num_with_relevant = metrics.get("num_queries", 0) - metrics.get("n_queries_no_relevant", 0)

    return {
        "num_queries": len(queries),
        "num_queries_with_relevant": num_with_relevant,
        "latency_ms": (time.time() - start) * 1000,
        "modes": per_mode,
    }
