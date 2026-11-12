"""
Domain services for the book recommendation system.

Services orchestrate domain logic that doesn't naturally belong to a single
entity. They coordinate between entities and ports to implement use cases.

Following Hexagonal Architecture principles, services depend only on domain
entities, value objects, and port protocols (never on concrete implementations).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import List, Optional, Dict
from uuid import UUID
import logging
import time

from .entities import Book, SearchResult, Explanation
from .value_objects import (
    SearchQuery, SearchFilters, SearchResponse, SearchMetadata, QueryIntent,
    RetrievalStrategy, STRATEGY_POOL_SIZES, INTENT_TO_STRATEGY,
)
from .ports import (
    BUDGET_LLM_REQUEST,
    BUDGET_MODEL_SEARCH,
    LexicalSearchRepository,
    VectorSearchRepository,
    EmbeddingsStore,
    LLMClient,
    Reranker,
    UsageBudget,
)

logger = logging.getLogger(__name__)

# Default number of fused candidates sent to the optional reranker.
DEFAULT_RERANK_DEPTH = 30

# Explanations generated at the same time (one LLM call each). The calls only
# wait on the network, so threads overlap them; the bound keeps a request from
# opening more connections to the provider than results it explains.
DEFAULT_EXPLANATION_WORKERS = 3

# Notices returned in SearchResponse.notices when a daily budget is spent (V17).
NOTICE_SEARCH_BUDGET = "search_budget_spent"
NOTICE_LLM_BUDGET = "llm_budget_spent"
SEARCH_BUDGET_REASON = (
    "The day's budget of model-backed searches is spent: lexical search only, "
    "without re-ranking, until 00:00 UTC"
)
LLM_BUDGET_REASON = "The day's budget of LLM requests is spent until 00:00 UTC"

# Notice returned when a request asks for query understanding or explanations
# and no LLM is configured: the search runs without them.
NOTICE_LLM_UNAVAILABLE = "llm_unavailable"


def _elapsed_ms(started: float) -> float:
    """Milliseconds since `started` (a time.perf_counter() value), rounded."""
    return round((time.perf_counter() - started) * 1000, 1)


class SearchService:
    """
    Orchestrates hybrid search combining lexical (BM25) and semantic (vector) search.

    This service implements the core search use case:
    1. Execute both BM25 and vector searches sequentially
    2. Fuse results using Reciprocal Rank Fusion (RRF)
    3. Apply filters to final results
    4. Optionally re-rank the top `rerank_depth` results with a Reranker
       (cross-encoder), always against the user's original query text
    5. Optionally diversify (MMR) and cut to max_results
    6. Optionally generate explanations via LLM

    The service is technology-agnostic and depends only on port protocols.
    """

    def __init__(
        self,
        lexical_search: LexicalSearchRepository,
        vector_search: VectorSearchRepository,
        embeddings_store: EmbeddingsStore,
        llm_client: LLMClient | None = None,
        max_explanations: int = 5,
        reranker: Reranker | None = None,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
        usage_budget: UsageBudget | None = None,
        explanation_workers: int = DEFAULT_EXPLANATION_WORKERS,
    ) -> None:
        """
        Initialize the search service with required dependencies.

        Args:
            lexical_search: Repository for BM25-based keyword search
            vector_search: Repository for semantic vector search
            embeddings_store: Store for generating query embeddings
            llm_client: Optional client for generating explanations (RAG)
            max_explanations: How many top results get an LLM explanation
                when explanations are requested (each one is an LLM call)
            reranker: Optional third stage that re-orders the top fused
                results. None (default) keeps the fused order.
            rerank_depth: How many top fused results the reranker scores.
                Retrieval then supplies at least this many candidates, even
                when fewer results are requested.
            usage_budget: Optional daily allowance of paid model calls (V17).
                Each search takes one BUDGET_MODEL_SEARCH unit (the query
                embedding and the re-ranking are API calls in the cloud
                profile); without one it runs lexical-only and is not
                re-ranked. Each request that uses the LLM takes one
                BUDGET_LLM_REQUEST unit; without one the LLM steps are
                skipped and a notice is returned. None (default): no limit.
            explanation_workers: How many explanations are generated at the
                same time (1 generates them one after another)
        """
        if explanation_workers < 1:
            raise ValueError(f"explanation_workers must be >= 1, got {explanation_workers}")
        if max_explanations < 0:
            raise ValueError(f"max_explanations must be >= 0, got {max_explanations}")
        if rerank_depth < 1:
            raise ValueError(f"rerank_depth must be >= 1, got {rerank_depth}")

        self._lexical_search = lexical_search
        self._vector_search = vector_search
        self._embeddings_store = embeddings_store
        self._llm_client = llm_client
        self._max_explanations = max_explanations
        self._explanation_workers = explanation_workers
        self._reranker = reranker
        self._rerank_depth = rerank_depth
        self._usage_budget = usage_budget

    def _spend(self, kind: str) -> bool:
        """
        One unit of the daily budget of `kind`: True without a budget or when granted.

        A budget that fails is treated as spent (fail closed): a paid call
        is never made without a counted unit.
        """
        if self._usage_budget is None:
            return True
        try:
            return bool(self._usage_budget.try_consume(kind))
        except Exception as e:  # noqa: BLE001 - the search must go on without the paid step
            logger.warning(f"Usage budget unavailable, treating '{kind}' as spent: {e}")
            return False

    def has_llm(self) -> bool:
        """True if an LLM client is wired (query understanding and explanations available)."""
        return self._llm_client is not None

    def has_reranker(self) -> bool:
        """True if a reranker is wired (third retrieval stage enabled)."""
        return self._reranker is not None

    def get_reranker_model_name(self) -> Optional[str]:
        """Model name of the wired reranker, or None without one."""
        return self._reranker.get_model_name() if self._reranker is not None else None

    def get_health_status(self) -> Dict[str, bool]:
        """
        Check the health status of all search components (RNF-06).

        The reranker is reported but does not count toward "overall": it is
        optional and, if it fails, the search returns the fused order.

        Returns:
            Dictionary with component names and their ready status:
            {
                "lexical_search": True/False,
                "vector_search": True/False,
                "embeddings_store": True/False,
                "reranker": True/False (False when none is configured),
                "overall": True/False
            }
        """
        lexical_ready = self._lexical_search.is_ready()
        vector_ready = self._vector_search.is_ready()
        embeddings_ready = self._embeddings_store.is_ready()
        reranker_ready = self._reranker is not None and self._reranker.is_ready()

        return {
            "lexical_search": lexical_ready,
            "vector_search": vector_ready,
            "embeddings_store": embeddings_ready,
            "reranker": reranker_ready,
            "overall": lexical_ready and vector_ready and embeddings_ready,
        }

    def search_with_fallback(
        self,
        query: SearchQuery,
        *,
        rerank_query_text: Optional[str] = None,
    ) -> SearchResponse:
        """
        Execute search with graceful degradation (RNF-08).

        If vector search fails or is unavailable, automatically falls back
        to lexical-only search and returns a degraded response with metadata.

        Args:
            query: The search query
            rerank_query_text: Text the reranker compares the books with.
                Defaults to query.text; the understanding path passes the
                user's original wording because query.text holds the LLM
                reformulation.

        Returns:
            SearchResponse with results and degradation metadata

        With a usage budget (V17), the search takes one model-search unit
        (when none is left it runs lexical-only without re-ranking, degraded
        with SEARCH_BUDGET_REASON) and a request for explanations takes one
        LLM unit (when none is left the results come without explanations
        and NOTICE_LLM_BUDGET is returned). Explanations asked for without a
        configured LLM are skipped with NOTICE_LLM_UNAVAILABLE.
        """
        notices: List[str] = []
        if query.use_explanations and self._llm_client is None:
            notices.append(NOTICE_LLM_UNAVAILABLE)
        if query.use_explanations and self._llm_client is not None and not self._spend(BUDGET_LLM_REQUEST):
            query = replace(query, use_explanations=False)
            notices.append(NOTICE_LLM_BUDGET)
        use_models = self._spend(BUDGET_MODEL_SEARCH)
        if not use_models:
            notices.append(NOTICE_SEARCH_BUDGET)
        response = self._search_with_fallback(query, rerank_query_text, use_models=use_models)
        return replace(response, notices=tuple(notices))

    def _search_with_fallback(
        self,
        query: SearchQuery,
        rerank_query_text: Optional[str] = None,
        *,
        use_models: bool = True,
    ) -> SearchResponse:
        """
        The search of search_with_fallback, after the budget was checked.

        use_models=False runs lexical-only without re-ranking (no model API
        call), the degraded mode of a spent daily budget.
        """
        start_time = time.time()
        degraded = False
        degradation_reason = None
        search_mode = "hybrid"
        metadata: Optional[SearchMetadata] = None
        rerank_text = rerank_query_text or query.text

        # Check if vector search is available
        vector_available = self._vector_search.is_ready() and self._embeddings_store.is_ready()

        if not use_models:
            degraded = True
            degradation_reason = SEARCH_BUDGET_REASON
            search_mode = "lexical_only"
            try:
                results, meta = self._search_lexical_only_with_debug(query, rerank_text, use_reranker=False)
            except Exception as e:
                logger.error(f"Lexical search failed: {e}")
                raise RuntimeError("Lexical search unavailable") from e
            metadata = SearchMetadata(
                fusion_method="none",
                rrf_k=None,
                diversification_enabled=bool(query.use_diversification),
                candidates_lexical=meta.get("candidates_lexical", 0),
                candidates_vector=0,
                timings_ms=meta.get("timings_ms", {}),
            )
        elif not vector_available:
            # Graceful degradation: use lexical search only
            logger.warning("Vector search unavailable, degrading to lexical-only search")
            degraded = True
            degradation_reason = "Vector index (FAISS) unavailable - using lexical search only"
            search_mode = "lexical_only"

            try:
                results, meta = self._search_lexical_only_with_debug(query, rerank_text)
                metadata = SearchMetadata(
                    fusion_method="none",
                    rrf_k=None,
                    diversification_enabled=bool(query.use_diversification),
                    candidates_lexical=meta.get("candidates_lexical", 0),
                    candidates_vector=0,
                    timings_ms=meta.get("timings_ms", {}),
                )
            except Exception as e:
                logger.error(f"Lexical search also failed: {e}")
                raise RuntimeError("Both vector and lexical search unavailable") from e
        else:
            # Try hybrid search, fallback to lexical if vector fails
            try:
                results, meta = self._search_hybrid_with_debug(query, rerank_text)
                metadata = SearchMetadata(
                    fusion_method="rrf",
                    rrf_k=meta.get("rrf_k", 60),
                    diversification_enabled=bool(query.use_diversification),
                    candidates_lexical=meta.get("candidates_lexical", 0),
                    candidates_vector=meta.get("candidates_vector", 0),
                    timings_ms=meta.get("timings_ms", {}),
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to lexical: {e}")
                degraded = True
                degradation_reason = f"Vector search failed ({str(e)}) - using lexical search only"
                search_mode = "lexical_only"
                results, meta = self._search_lexical_only_with_debug(query, rerank_text)
                metadata = SearchMetadata(
                    fusion_method="none",
                    rrf_k=None,
                    diversification_enabled=bool(query.use_diversification),
                    candidates_lexical=meta.get("candidates_lexical", 0),
                    candidates_vector=0,
                    timings_ms=meta.get("timings_ms", {}),
                )

        latency_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            degraded=degraded,
            degradation_reason=degradation_reason,
            search_mode=search_mode,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def search_with_understanding(
        self,
        query_text: str,
        max_results: int = 10,
        use_explanations: bool = False,
        use_diversification: bool = False,
        diversity_lambda: float = 0.6,
        filters: Optional[SearchFilters] = None,
    ) -> tuple[SearchResponse, QueryIntent]:
        """
        Execute search with LLM-powered query understanding (Block 2).

        This method implements intelligent search by:
        1. Using LLM to understand query intent (recommendation, factual, exploratory)
        2. Extracting filters from natural language (language, category, year)
        3. Reformulating the query for better retrieval
        4. Adjusting search strategy based on intent type

        Intent-based search strategies:
        - recommendation: Higher weight on vector search (semantic similarity)
        - factual: Higher weight on lexical search (exact keyword matches)
        - exploratory: Balanced hybrid + increased diversification

        Args:
            query_text: Raw user query in natural language
            max_results: Maximum number of results to return
            use_explanations: Whether to generate LLM explanations for results
            use_diversification: Whether to apply MMR diversification
            diversity_lambda: Trade-off between relevance and diversity
            filters: Filters set explicitly by the user. They are combined
                with the filters the LLM extracted; explicit values win.

        Returns:
            Tuple of (SearchResponse, QueryIntent):
            - SearchResponse: Results with metadata
            - QueryIntent: The understood intent (for debugging/display)

        Raises:
            ValueError: If query_text is empty
            RuntimeError: If both query understanding and search fail

        Example:
            >>> response, intent = search_service.search_with_understanding(
            ...     query_text="Spanish novels like Don Quixote from the 1900s"
            ... )
            >>> print(f"Intent: {intent.intent_type}")  # "recommendation"
            >>> print(f"Filters: {intent.extracted_filters}")  # language=es, min_year=1900
            >>> print(f"Reformulated: {intent.reformulated_query}")  # "Don Quixote Spanish literature"
        """
        start_time = time.time()

        # 1. Validate input
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")

        logger.info(f"Executing search with understanding for: '{query_text[:50]}...'")

        # Daily budget (V17): one LLM unit for the whole request (understanding
        # and explanations), one model-search unit for the retrieval.
        notices: List[str] = []
        if self._llm_client is None:
            notices.append(NOTICE_LLM_UNAVAILABLE)
        llm_allowed = self._llm_client is not None and self._spend(BUDGET_LLM_REQUEST)
        if self._llm_client is not None and not llm_allowed:
            notices.append(NOTICE_LLM_BUDGET)
            use_explanations = False
        use_models = self._spend(BUDGET_MODEL_SEARCH)
        if not use_models:
            notices.append(NOTICE_SEARCH_BUDGET)

        # 2. Extract query intent using LLM
        if not llm_allowed:
            reason = (
                "LLM client not available - using default exploratory intent"
                if self._llm_client is None
                else f"{LLM_BUDGET_REASON} - using default exploratory intent"
            )
            logger.warning(f"Search without understanding: {reason}")
            # Fallback: create a basic QueryIntent with defaults
            query_intent = QueryIntent(
                intent_type="exploratory",
                original_query=query_text,
                reformulated_query=query_text,
                extracted_filters=SearchFilters(),
                confidence=0.0,
                reasoning=reason,
            )
        else:
            try:
                query_intent = self._llm_client.extract_query_intent(query_text)
                logger.info(
                    f"Query understood - Intent: {query_intent.intent_type} "
                    f"(confidence: {query_intent.confidence:.2f})"
                )
            except Exception as e:
                logger.error(f"Query understanding failed: {e}")
                # Fallback to basic intent
                query_intent = QueryIntent(
                    intent_type="exploratory",
                    original_query=query_text,
                    reformulated_query=query_text,
                    extracted_filters=SearchFilters(),
                    confidence=0.0,
                    reasoning=f"Query understanding failed: {str(e)}"
                )

        # 3. Map intent to retrieval strategy
        strategy = INTENT_TO_STRATEGY.get(
            query_intent.intent_type, RetrievalStrategy.BALANCED
        )
        logger.info(f"Intent '{query_intent.intent_type}' -> strategy '{strategy.value}'")

        # 4. Adjust diversification based on intent
        adjusted_diversification = use_diversification
        adjusted_lambda = diversity_lambda

        if query_intent.intent_type == "exploratory":
            adjusted_diversification = True
            adjusted_lambda = 0.5
        elif query_intent.intent_type == "recommendation":
            adjusted_lambda = 0.6
        elif query_intent.intent_type == "factual":
            adjusted_lambda = 0.8

        # 5. Build variations list.
        #    reformulated_query is always the primary. Additional variations
        #    from LangGraph are appended if they're different.
        variations = [query_intent.reformulated_query]
        for v in query_intent.query_variations:
            if v not in variations:
                variations.append(v)

        # 6. Build a SearchQuery (used for filters/max_results/explanations)
        search_query = SearchQuery(
            text=query_intent.reformulated_query,
            filters=query_intent.extracted_filters.overridden_by(filters),
            max_results=max_results,
            use_explanations=use_explanations,
            use_diversification=adjusted_diversification,
            diversity_lambda=adjusted_lambda,
        )

        # 7. Execute multi-query search with strategy routing
        #    Check vector availability first for graceful degradation
        vector_available = (
            self._vector_search.is_ready() and self._embeddings_store.is_ready()
        )

        if vector_available and use_models:
            try:
                results, meta = self._multi_query_search(
                    variations=variations,
                    strategy=strategy,
                    query=search_query,
                )

                # Apply filters to the full fused pool (repositories already
                # filtered, this keeps the guarantee for every path)
                results = self._apply_filters(results, search_query.filters)

                # Optional third stage: re-rank against the user's own words,
                # not the LLM reformulation or the query variations
                results = self._rerank(query_text, results)

                # Select the final top-k: MMR chooses among the whole pool,
                # otherwise keep the best-ranked results
                if adjusted_diversification and len(results) > 1:
                    results = self._apply_mmr_diversification(
                        results=results,
                        top_k=max_results,
                        lambda_param=adjusted_lambda,
                    )
                else:
                    results = results[:max_results]
                    for i, r in enumerate(results, start=1):
                        r.rank = i

                # Explanations are generated against the user's original wording
                if use_explanations and self._llm_client is not None:
                    results = self._add_explanations(query_text, results)

                search_time_ms = (time.time() - start_time) * 1000
                metadata = SearchMetadata(
                    fusion_method="rrf",
                    rrf_k=60,
                    diversification_enabled=adjusted_diversification,
                    candidates_lexical=meta.get("candidates_lexical", 0),
                    candidates_vector=meta.get("candidates_vector", 0),
                    timings_ms=meta.get("timings_ms", {}),
                )
                response = SearchResponse(
                    results=results,
                    degraded=False,
                    search_mode="hybrid",
                    latency_ms=search_time_ms,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Multi-query search failed, falling back: {e}")
                response = self._fallback_with_original_explanations(search_query, query_text)
        else:
            # Graceful degradation: vector unavailable (or no model-search
            # budget left), use the lexical fallback
            logger.warning("Vector search unavailable or budget spent, using fallback for understanding search")
            response = self._fallback_with_original_explanations(search_query, query_text, use_models=use_models)
        response = replace(response, notices=tuple(notices))

        # 8. Log timing
        total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Search with understanding completed in {total_time_ms:.1f}ms "
            f"({len(variations)} variations, strategy={strategy.value})"
        )

        return response, query_intent

    def _fallback_with_original_explanations(
        self, search_query: SearchQuery, original_query_text: str, use_models: bool = True
    ) -> SearchResponse:
        """
        Run the fallback search for the understanding path.

        The fallback searches with the reformulated query, but re-ranking and
        explanations must answer the user's original wording: the original
        text is passed to the reranker, and explanations are generated here
        instead of inside the fallback search. The budget was already checked
        by the caller, so the fallback does not take another unit.
        """
        response = self._search_with_fallback(
            replace(search_query, use_explanations=False),
            original_query_text,
            use_models=use_models,
        )
        if search_query.use_explanations and self._llm_client is not None:
            self._add_explanations(original_query_text, response.results)
        return response

    def _search_lexical_only_with_debug(
        self, query: SearchQuery, rerank_text: Optional[str] = None, use_reranker: bool = True
    ) -> tuple[List[SearchResult], Dict]:
        timings: Dict[str, float] = {}
        results = self._search_lexical_only(query, rerank_text, use_reranker=use_reranker, timings=timings)
        return results, {"candidates_lexical": len(results), "timings_ms": timings}

    def _search_hybrid_with_debug(
        self, query: SearchQuery, rerank_text: Optional[str] = None
    ) -> tuple[list[SearchResult], Dict]:
        """
        Hybrid search pipeline: retrieve -> fuse (RRF) -> filter ->
        optional re-rank -> optional MMR -> top-k -> optional explanations.

        Args:
            query: The search query
            rerank_text: Text for the reranker (defaults to query.text)
        """
        logger.info(f"Executing hybrid search for query: '{query.text}'")

        candidate_limit = self._retrieval_size(query.max_results) * 2
        timings: Dict[str, float] = {}

        logger.debug("Executing lexical (BM25) search")
        started = time.perf_counter()
        lexical_results = self._lexical_search.search(
            query_text=query.text,
            max_results=candidate_limit,
            filters=query.filters
        )
        timings["lexical"] = _elapsed_ms(started)

        logger.debug("Generating query embedding and executing vector search")
        started = time.perf_counter()
        query_embedding = self._embeddings_store.generate_query_embedding(query.text)
        timings["embedding"] = _elapsed_ms(started)
        started = time.perf_counter()
        vector_results = self._vector_search.search(
            query_embedding=query_embedding,
            max_results=candidate_limit,
            filters=query.filters,
        )
        timings["vector"] = _elapsed_ms(started)

        logger.debug(
            f"Retrieved {len(lexical_results)} lexical results, "
            f"{len(vector_results)} vector results"
        )

        rrf_k = 60
        fused_results = self._fuse_results_rrf(
            lexical_results=lexical_results,
            vector_results=vector_results,
            k=rrf_k,
        )

        logger.debug(f"Fused into {len(fused_results)} unique results")

        filtered_results = self._apply_filters(fused_results, query.filters)

        started = time.perf_counter()
        filtered_results = self._rerank(rerank_text or query.text, filtered_results)
        if self._reranker is not None:
            timings["rerank"] = _elapsed_ms(started)

        if query.use_diversification:
            logger.debug(f"Applying MMR diversification with lambda={query.diversity_lambda}")
            started = time.perf_counter()
            final_results = self._apply_mmr_diversification(
                results=filtered_results,
                top_k=query.max_results,
                lambda_param=query.diversity_lambda,
            )
            timings["mmr"] = _elapsed_ms(started)
        else:
            final_results = filtered_results[: query.max_results]
            for i, result in enumerate(final_results, start=1):
                result.rank = i

        logger.info(f"Returning {len(final_results)} results")

        if query.use_explanations and self._llm_client is not None:
            logger.debug("Generating explanations for top results")
            started = time.perf_counter()
            final_results = self._add_explanations(query.text, final_results)
            timings["explanations"] = _elapsed_ms(started)

        return final_results, {
            "timings_ms": timings,
            "fusion_method": "rrf",
            "rrf_k": rrf_k,
            "candidates_lexical": len(lexical_results),
            "candidates_vector": len(vector_results),
        }

    def _search_lexical_only(
        self,
        query: SearchQuery,
        rerank_text: Optional[str] = None,
        *,
        use_reranker: bool = True,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[SearchResult]:
        """
        Execute lexical-only search (fallback mode).

        Used when vector search is unavailable for graceful degradation.
        The reranker does not depend on the vector index, so it still runs,
        unless use_reranker is False (the daily model budget is spent).

        Args:
            query: The search query
            rerank_text: Text for the reranker (defaults to query.text)
            use_reranker: Re-rank the results when a reranker is configured
            timings: Filled with the milliseconds of each stage, if given

        Returns:
            List of SearchResult from BM25 search only
        """
        logger.info(f"Executing lexical-only search for query: '{query.text}'")
        timings = timings if timings is not None else {}
        rerank = use_reranker and self._reranker is not None

        started = time.perf_counter()
        results = self._lexical_search.search(
            query_text=query.text,
            max_results=self._retrieval_size(query.max_results) if rerank else query.max_results,
            filters=query.filters,
        )
        timings["lexical"] = _elapsed_ms(started)

        if rerank:
            started = time.perf_counter()
            results = self._rerank(rerank_text or query.text, results)
            timings["rerank"] = _elapsed_ms(started)

        # Apply diversification if requested (uses embeddings if available)
        if query.use_diversification and len(results) > 1:
            try:
                if self._embeddings_store.is_ready():
                    results = self._apply_mmr_diversification(
                        results=results,
                        top_k=query.max_results,
                        lambda_param=query.diversity_lambda,
                    )
            except Exception as e:
                logger.warning(f"MMR diversification failed in degraded mode: {e}")

        # Cut to max_results (retrieval fetched rerank_depth when re-ranking)
        results = results[: query.max_results]

        # Re-assign ranks
        for i, result in enumerate(results, start=1):
            result.rank = i

        # Degraded mode still honours explanation requests (RAG over BM25 results)
        if query.use_explanations and self._llm_client is not None:
            started = time.perf_counter()
            results = self._add_explanations(query.text, results)
            timings["explanations"] = _elapsed_ms(started)

        return results

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Execute a hybrid search combining lexical and semantic approaches.

        This method implements the following algorithm:

        1. **Sequential retrieval:**
           - Execute BM25 search over book text (title, authors, description, categories)
           - Generate query embedding and execute vector similarity search
           - Both searches retrieve up to `query.max_results * 2` candidates
             (with a reranker, as if max_results were at least rerank_depth)

        2. **Fusion via Reciprocal Rank Fusion (RRF):**
           - RRF is a rank-based fusion method that combines rankings without
             needing to normalize scores from different systems
           - Formula: RRF_score(book) = Σ(1 / (k + rank_i)) for each system i
           - We use k=60 (standard value from literature)
           - Books appearing in both rankings get boosted scores
           - Final ranking is determined by RRF score (descending)

        3. **Post-processing:**
           - Remove duplicates (same book from both systems)
           - Apply filters from SearchQuery (language, category, year range)
           - Optionally re-rank the top `rerank_depth` results with the
             reranker (cross-encoder) against query.text
           - Limit results to query.max_results (or select them with MMR)
           - Re-assign ranks (1-indexed)

        4. **Optional explanation generation:**
           - If query.use_explanations is True and llm_client is available,
             generate natural language explanations for top results using RAG

        Args:
            query: The search query with text, filters, and parameters

        Returns:
            List of SearchResult entities, ranked by hybrid relevance score

        Raises:
            ValueError: If query is invalid
            RuntimeError: If search execution fails

        Example:
            >>> query = SearchQuery(
            ...     text="science fiction space opera",
            ...     filters=SearchFilters(language="en", min_year=2000),
            ...     max_results=10,
            ...     use_explanations=True
            ... )
            >>> results = search_service.search(query)
            >>> for result in results:
            ...     print(f"{result.rank}. {result.book.title} (score: {result.final_score:.3f})")
        """

        results, _debug = self._search_hybrid_with_debug(query)
        return results

    def find_similar_books(
        self,
        book_id: UUID,
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
        use_diversification: bool = False,
        diversity_lambda: float = 0.6,
    ) -> List[SearchResult]:
        """
        Find books similar to a given book using pure vector search (Item-to-Item).

        This implements the RF-02 requirement for item-to-item recommendations.
        Unlike the hybrid search() method, this uses only semantic similarity
        based on the source book's embedding vector.

        Algorithm:
        1. Retrieve the source book's embedding from EmbeddingsStore
        2. Perform vector search to find nearest neighbors
        3. Exclude the source book from results
        4. Optionally apply filters and MMR diversification

        Args:
            book_id: UUID of the source book to find similar books for
            max_results: Maximum number of similar books to return
            filters: Optional filters (language, category, year range)
            use_diversification: Whether to apply MMR diversification
            diversity_lambda: Trade-off between similarity and diversity

        Returns:
            List of SearchResult entities, ranked by vector similarity (descending)

        Raises:
            ValueError: If book_id is not found or has no embedding
        """
        logger.info(f"Finding similar books for book_id: {book_id}")

        # Step 1: Get the source book's embedding
        source_embedding = self._embeddings_store.get_embedding(book_id)
        if source_embedding is None:
            raise ValueError(f"No embedding found for book_id: {book_id}")

        # Step 2: Perform vector search (retrieve extra to account for filtering)
        candidate_limit = max_results * 2 + 1  # +1 to exclude source book

        vector_results = self._vector_search.search(
            query_embedding=source_embedding,
            max_results=candidate_limit,
            filters=filters,
        )

        logger.debug(f"Retrieved {len(vector_results)} vector results")

        # Step 3: Exclude the source book from results
        filtered_results = [r for r in vector_results if r.book.id != book_id]

        logger.debug(f"After excluding source book: {len(filtered_results)} results")

        # Step 4: Apply additional filters if provided
        if filters is not None and not filters.is_empty():
            filtered_results = self._apply_filters(filtered_results, filters)

        # Step 5: Apply diversification or limit results
        if use_diversification and len(filtered_results) > 1:
            logger.debug(f"Applying MMR diversification with lambda={diversity_lambda}")
            final_results = self._apply_mmr_diversification(
                results=filtered_results,
                top_k=max_results,
                lambda_param=diversity_lambda,
            )
        else:
            final_results = filtered_results[:max_results]
            # Re-assign ranks
            for i, result in enumerate(final_results, start=1):
                result.rank = i

        logger.info(f"Returning {len(final_results)} similar books")
        return final_results

    # ==========================================================================
    # Multi-Query Retrieval with Strategy Router
    # ==========================================================================

    def _search_hybrid_with_strategy(
        self,
        variation_text: str,
        strategy: RetrievalStrategy,
        filters: SearchFilters,
        max_results: int,
    ) -> List[SearchResult]:
        """
        Execute a single hybrid search with strategy-controlled pool sizes.

        The strategy determines how many candidates each retrieval method
        contributes before RRF fusion. This biases the final ranking toward
        the method that best suits the query intent.

        Args:
            variation_text: The query text for this variation
            strategy: Controls BM25/vector candidate ratio
            filters: Filters to apply
            max_results: Used to compute pool sizes via strategy factors

        Returns:
            Ranked list of SearchResult (fused via RRF)
        """
        pool_sizes = STRATEGY_POOL_SIZES[strategy]
        bm25_top_k = max_results * pool_sizes["bm25_factor"]
        vector_top_k = max_results * pool_sizes["vector_factor"]

        logger.debug(
            f"Strategy {strategy.value}: BM25 top_k={bm25_top_k}, "
            f"vector top_k={vector_top_k} for '{variation_text[:40]}...'"
        )

        # Lexical search
        lexical_results = self._lexical_search.search(
            query_text=variation_text,
            max_results=bm25_top_k,
            filters=filters,
        )

        # Vector search
        query_embedding = self._embeddings_store.generate_query_embedding(variation_text)
        vector_results = self._vector_search.search(
            query_embedding=query_embedding,
            max_results=vector_top_k,
            filters=filters,
        )

        # RRF fusion within this variation
        fused = self._fuse_results_rrf(lexical_results, vector_results, k=60)

        return fused

    def _fuse_multi_query_results(
        self,
        all_results: List[List[SearchResult]],
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Fuse results from multiple query variations using RRF.

        Each variation produced its own ranked list. We treat each as a
        separate "ranking system" and fuse with RRF - the same algorithm
        used for BM25+FAISS fusion, applied at a higher level.

        Books appearing in multiple variation rankings get boosted scores,
        which naturally surfaces results that are relevant from multiple
        perspectives. Each fused result keeps the best lexical and vector
        component scores seen across variations, for transparency.

        Args:
            all_results: List of ranked results, one per query variation,
                in variation order (the order breaks RRF ties)
            max_results: Optional cut-off. Callers that still filter or
                diversify must pass None so they work on the whole pool.

        Returns:
            Fused and deduplicated list of SearchResult
        """
        rrf_scores: Dict[UUID, float] = {}
        books_map: Dict[UUID, Book] = {}
        lexical_scores: Dict[UUID, float] = {}
        vector_scores: Dict[UUID, float] = {}
        k = 60

        for variation_results in all_results:
            for result in variation_results:
                book_id = result.book.id
                rrf_scores[book_id] = (
                    rrf_scores.get(book_id, 0.0) + (1.0 / (k + result.rank))
                )
                books_map[book_id] = result.book
                if result.lexical_score is not None:
                    lexical_scores[book_id] = max(
                        result.lexical_score, lexical_scores.get(book_id, float("-inf"))
                    )
                if result.vector_score is not None:
                    vector_scores[book_id] = max(
                        result.vector_score, vector_scores.get(book_id, float("-inf"))
                    )

        # sorted() is stable: ties keep first-seen order, which follows variation order
        sorted_ids = sorted(rrf_scores, key=lambda bid: rrf_scores[bid], reverse=True)
        if max_results is not None:
            sorted_ids = sorted_ids[:max_results]

        return [
            SearchResult(
                book=books_map[bid],
                final_score=rrf_scores[bid],
                rank=i + 1,
                source="hybrid",
                lexical_score=lexical_scores.get(bid),
                vector_score=vector_scores.get(bid),
            )
            for i, bid in enumerate(sorted_ids)
        ]

    def _multi_query_search(
        self,
        variations: List[str],
        strategy: RetrievalStrategy,
        query: SearchQuery,
    ) -> tuple[List[SearchResult], Dict]:
        """
        Run hybrid search for each query variation IN PARALLEL, then fuse.

        Uses ThreadPoolExecutor because:
        - FAISS (C extension) and embeddings (PyTorch) release the GIL
        - Each variation is fully independent (no shared mutable state)
        - 2-3 threads is lightweight; no global pool needed

        Results are fused in variation order (not completion order), so the
        ranking is deterministic. The fused pool is returned whole: filtering,
        diversification and the final top-k cut happen afterwards.

        If a variation fails, results from other variations are still used
        (graceful degradation). If ALL fail, falls back to single balanced search.

        Args:
            variations: List of query text variations (2-3 typically)
            strategy: Retrieval strategy controlling pool sizes
            query: Original SearchQuery (for filters and max_results)

        Returns:
            Tuple of (fused results, debug metadata dict). The metadata counts
            the unique books contributed by lexical and by vector search.
        """
        logger.info(
            f"Multi-query search: {len(variations)} variations, "
            f"strategy={strategy.value}"
        )

        results_by_index: Dict[int, List[SearchResult]] = {}

        # Pool sizes scale with this; a reranker needs at least rerank_depth
        retrieval_size = self._retrieval_size(query.max_results)

        # Run all variations in parallel
        with ThreadPoolExecutor(max_workers=len(variations)) as executor:
            future_to_index = {
                executor.submit(
                    self._search_hybrid_with_strategy,
                    variation_text=variation,
                    strategy=strategy,
                    filters=query.filters,
                    max_results=retrieval_size,
                ): index
                for index, variation in enumerate(variations)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                variation = variations[index]
                try:
                    results = future.result()
                    results_by_index[index] = results
                    logger.debug(
                        f"Variation '{variation[:30]}...' returned {len(results)} results"
                    )
                except Exception as e:
                    logger.warning(f"Variation '{variation[:30]}...' failed: {e}")

        all_variation_results = [results_by_index[i] for i in sorted(results_by_index)]

        if not all_variation_results:
            # All variations failed - fall back to single balanced search
            logger.warning("All variations failed, falling back to single balanced search")
            fallback = self._search_hybrid_with_strategy(
                variation_text=variations[0],
                strategy=RetrievalStrategy.BALANCED,
                filters=query.filters,
                max_results=retrieval_size,
            )
            all_variation_results = [fallback]

        # Fuse results across all variations (whole pool, no cut-off here)
        fused = self._fuse_multi_query_results(all_variation_results)

        metadata = {
            "fusion_method": "rrf",
            "rrf_k": 60,
            "strategy": strategy.value,
            "variations_attempted": len(variations),
            "variations_succeeded": len(all_variation_results),
            "candidates_lexical": sum(1 for r in fused if r.lexical_score is not None),
            "candidates_vector": sum(1 for r in fused if r.vector_score is not None),
        }

        return fused, metadata

    def _fuse_results_rrf(
        self,
        lexical_results: List[SearchResult],
        vector_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Fuse results from multiple search systems using Reciprocal Rank Fusion.

        RRF is a simple yet effective rank-based fusion method that doesn't
        require score normalization. It assigns each book a final_score based on its
        rank in each result list.

        Formula:
            RRF_score(book) = Σ(1 / (k + rank_i))

        Where:
        - k is a constant (typically 60) that reduces the impact of high ranks
        - rank_i is the rank of the book in system i (1-indexed)
        - The sum is over all systems where the book appears

        Books appearing in multiple systems get higher final_score (boosting effect).

        Args:
            lexical_results: Results from BM25 search
            vector_results: Results from vector search
            k: Constant for RRF formula (default: 60)

        Returns:
            List of SearchResult entities, ranked by RRF final_score (descending)
        """

        # Build mappings for fusion
        rrf_scores: Dict[UUID, float] = {}
        books_map: Dict[UUID, Book] = {}
        lexical_scores_map: Dict[UUID, float] = {}
        vector_scores_map: Dict[UUID, float] = {}

        # Process lexical results
        for result in lexical_results:
            book_id = result.book.id
            rrf_scores[book_id] = rrf_scores.get(book_id, 0.0) + (1.0 / (k + result.rank))
            books_map[book_id] = result.book
            # Preserve original lexical score
            if result.has_lexical_score():
                lexical_scores_map[book_id] = result.lexical_score
            elif result.final_score is not None:
                lexical_scores_map[book_id] = result.final_score

        # Process vector results
        for result in vector_results:
            book_id = result.book.id
            rrf_scores[book_id] = rrf_scores.get(book_id, 0.0) + (1.0 / (k + result.rank))
            books_map[book_id] = result.book
            # Preserve original vector score
            if result.has_vector_score():
                vector_scores_map[book_id] = result.vector_score
            elif result.final_score is not None:
                vector_scores_map[book_id] = result.final_score

        # Sort books by RRF score (descending)
        sorted_book_ids = sorted(
            rrf_scores.keys(),
            key=lambda book_id: rrf_scores[book_id],
            reverse=True,
        )

        # Construct fused results with all score information
        fused_results = [
            SearchResult(
                book=books_map[book_id],
                final_score=rrf_scores[book_id],
                rank=i + 1,
                source="hybrid",
                lexical_score=lexical_scores_map.get(book_id),
                vector_score=vector_scores_map.get(book_id),
            )
            for i, book_id in enumerate(sorted_book_ids)
        ]

        return fused_results

    def _apply_filters(
        self,
        results: List[SearchResult],
        filters: SearchFilters,
    ) -> List[SearchResult]:
        """
        Apply SearchFilters to a list of results.

        Note: Filters may already be partially applied by search repositories,
        but we ensure they are fully applied here for consistency.

        Args:
            results: List of search results
            filters: Filters to apply (guaranteed non-None by SearchQuery)

        Returns:
            Filtered list of results
        """

        if filters.is_empty():
            return results

        # Book.matches_filters is the single filter definition shared with the
        # repositories, so this pass can never discard what they accepted
        return [result for result in results if result.book.matches_filters(filters)]

    # ==========================================================================
    # Optional re-ranking stage (cross-encoder)
    # ==========================================================================

    def _retrieval_size(self, max_results: int) -> int:
        """
        How many results the retrieval stages work for.

        Without a reranker this is max_results, as before. With one, the
        retrieval runs as if at least `rerank_depth` results were requested,
        so the reranker always has enough candidates to choose from.
        """
        if self._reranker is None:
            return max_results
        return max(max_results, self._rerank_depth)

    def _rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Re-order the top `rerank_depth` results with the reranker.

        Runs after fusion and filtering, before MMR and the final cut. The
        results beyond the depth keep their fused order after the re-ranked
        head. If no reranker is configured, the list is returned unchanged;
        if the reranker fails, a warning is logged and the fused order is
        returned (graceful degradation, the search never fails because of
        this stage).

        Args:
            query_text: The user's original query text
            results: Fused and filtered results, best first

        Returns:
            Results with the head re-ordered, rerank_score set on the head
            and ranks renumbered 1..n over the whole list
        """
        if self._reranker is None or not results:
            return results

        head = results[: self._rerank_depth]
        tail = results[self._rerank_depth :]

        try:
            reranked_head = self._reranker.rerank(query_text, head)
            # The port promises the same results back; anything else is a bug
            # in the adapter and is treated like a failure
            if len(reranked_head) != len(head) or (
                {r.book.id for r in reranked_head} != {r.book.id for r in head}
            ):
                raise ValueError("reranker must return exactly the results it received")
        except Exception as e:
            logger.warning(f"Re-ranking failed, keeping the fused order: {e}")
            return results

        reranked = list(reranked_head) + tail
        for i, result in enumerate(reranked, start=1):
            result.rank = i
        return reranked

    def _add_explanations(
        self,
        query_text: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Generate LLM-based explanations for search results (RAG pattern).

        This method implements the generation step of RAG:
        - Context (books) has already been retrieved
        - For each result, generate an explanation via LLM

        Only the top `max_explanations` results are explained, because each
        explanation is one LLM call. The calls are independent, so up to
        `explanation_workers` of them run at the same time; each explanation
        is written to its own result, so the order of the results does not
        change. A failed explanation leaves its result without one and does
        not affect the others. The budget is not touched here: the caller
        already took one LLM unit for the whole request.

        Args:
            query_text: The user's query the explanations must answer
            results: List of search results

        Returns:
            Results with explanation field populated
        """
        if self._llm_client is None:
            logger.warning("LLM Client not available, skipping explanations")
            return results

        llm_client = self._llm_client
        to_explain = results[: self._max_explanations]

        def explain(result: SearchResult) -> None:
            try:
                result.explanation = llm_client.generate_grounded_explanation(
                    query_text=query_text,
                    book=result.book,
                )
            except Exception as e:
                logger.error(f"Failed to generate explanation for book {result.book.id}: {e}")
                # Continue with no explanation for this result

        if len(to_explain) <= 1 or self._explanation_workers == 1:
            for result in to_explain:
                explain(result)
            return results

        workers = min(self._explanation_workers, len(to_explain))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="explanation") as executor:
            list(executor.map(explain, to_explain))  # waits for all; explain() never raises
        return results

    def _apply_mmr_diversification(
        self,
        results: List[SearchResult],
        top_k: int,
        lambda_param: float = 0.6,
    ) -> List[SearchResult]:
        """
        Rerank results using Maximal Marginal Relevance (MMR) to increase diversity.

        MMR iteratively selects documents that are both relevant to the query AND
        different from already-selected documents.

        Formula:
            MMR(d) = lambda * Relevance(d) - (1-lambda) * max(Similarity(d, d_selected))

        Where:
        - lambda: trade-off between relevance (1.0) and diversity (0.0)
        - Relevance(d): the RRF score from hybrid search, or the reranker
          score (rerank_score) when the results were re-ranked
        - Similarity: cosine similarity between book embeddings

        After re-ranking only the head of the list has a rerank_score, and
        it is not on the same scale as the fused scores of the tail. MMR then
        diversifies the re-ranked head; the tail only fills the slots left
        when top_k is larger than the head, in its fused order.

        Args:
            results: List of search results (already ranked by RRF or the reranker)
            top_k: Number of results to select
            lambda_param: Trade-off parameter (default: 0.6, slight preference for relevance)

        Returns:
            Reranked list of top_k results optimized for diversity
        """
        if len(results) <= 1:
            return results

        reranked_head = [r for r in results if r.has_rerank_score()]
        if reranked_head and len(reranked_head) < len(results):
            tail = [r for r in results if not r.has_rerank_score()]
            selected = self._apply_mmr_diversification(
                reranked_head, min(top_k, len(reranked_head)), lambda_param
            )
            selected = selected + tail[: max(0, top_k - len(selected))]
            for i, result in enumerate(selected, start=1):
                result.rank = i
            return selected

        def relevance_of(result: SearchResult) -> float:
            # Without re-ranking rerank_score is None and this is final_score
            return result.rerank_score if result.has_rerank_score() else result.final_score

        # Embeddings of all candidate books in one lookup (one database round
        # trip in the cloud profile instead of one per book)
        book_embeddings: Dict[UUID, List[float]] = self._embeddings_store.get_embeddings_batch(
            [result.book.id for result in results]
        )

        # If no embeddings available, return original results
        if not book_embeddings:
            logger.warning("No embeddings available for MMR diversification, skipping")
            return results[:top_k]

        # Normalize relevance scores to [0, 1] for fair comparison with similarity
        max_score = max(relevance_of(r) for r in results) if results else 1.0
        min_score = min(relevance_of(r) for r in results) if results else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        def normalize_score(score: float) -> float:
            return (score - min_score) / score_range

        # Greedy MMR selection.
        # max_similarity[book_id] holds the highest similarity between that
        # candidate and any book selected so far. It is updated once per
        # selection (k * n similarity computations) instead of being recomputed
        # against every selected book at every step (k^2 * n), which made the
        # evaluation with k=100 take minutes per query. Same formula, same
        # tie-breaking (first candidate with the strictly highest score).
        selected: List[SearchResult] = []
        candidates = list(results)
        max_similarity: Dict[UUID, float] = {}

        while len(selected) < top_k and candidates:
            best_mmr_score = float('-inf')
            best_idx = 0

            for i, candidate in enumerate(candidates):
                # Skip if no embedding
                if candidate.book.id not in book_embeddings:
                    continue

                relevance = normalize_score(relevance_of(candidate))
                similarity = max_similarity.get(candidate.book.id, 0.0)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * similarity

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            chosen = candidates.pop(best_idx)
            selected.append(chosen)

            # Update the running maximum similarity of the remaining candidates
            chosen_embedding = book_embeddings.get(chosen.book.id)
            if chosen_embedding is not None:
                for candidate in candidates:
                    embedding = book_embeddings.get(candidate.book.id)
                    if embedding is None:
                        continue
                    similarity = self._cosine_similarity(embedding, chosen_embedding)
                    if similarity > max_similarity.get(candidate.book.id, float('-inf')):
                        max_similarity[candidate.book.id] = similarity

        # Reassign ranks (1-indexed)
        for i, result in enumerate(selected, start=1):
            result.rank = i

        logger.debug(f"MMR diversification selected {len(selected)} results")
        return selected

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector
            vec_b: Second vector

        Returns:
            Cosine similarity in range [-1, 1], typically [0, 1] for embeddings
        """
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)