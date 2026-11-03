"""
Domain services for the book recommendation system.

Services orchestrate domain logic that doesn't naturally belong to a single
entity. They coordinate between entities and ports to implement use cases.

Following Hexagonal Architecture principles, services depend only on domain
entities, value objects, and port protocols (never on concrete implementations).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    LexicalSearchRepository,
    VectorSearchRepository,
    EmbeddingsStore,
    LLMClient,
)

logger = logging.getLogger(__name__)

class SearchService:
    """
    Orchestrates hybrid search combining lexical (BM25) and semantic (vector) search.

    This service implements the core search use case:
    1. Execute both BM25 and vector searches sequentially
    2. Fuse results using Reciprocal Rank Fusion (RRF)
    3. Apply filters to final results
    4. Optionally generate explanations via LLM

    The service is technology-agnostic and depends only on port protocols.
    """

    def __init__(
        self,
        lexical_search: LexicalSearchRepository,
        vector_search: VectorSearchRepository,
        embeddings_store: EmbeddingsStore,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the search service with required dependencies.

        Args:
            lexical_search: Repository for BM25-based keyword search
            vector_search: Repository for semantic vector search
            embeddings_store: Store for generating query embeddings
            llm_client: Optional client for generating explanations (RAG)
        """
        self._lexical_search = lexical_search
        self._vector_search = vector_search
        self._embeddings_store = embeddings_store
        self._llm_client = llm_client

    def get_health_status(self) -> Dict[str, bool]:
        """
        Check the health status of all search components (RNF-06).

        Returns:
            Dictionary with component names and their ready status:
            {
                "lexical_search": True/False,
                "vector_search": True/False,
                "embeddings_store": True/False,
                "overall": True/False
            }
        """
        lexical_ready = self._lexical_search.is_ready()
        vector_ready = self._vector_search.is_ready()
        embeddings_ready = self._embeddings_store.is_ready()

        return {
            "lexical_search": lexical_ready,
            "vector_search": vector_ready,
            "embeddings_store": embeddings_ready,
            "overall": lexical_ready and vector_ready and embeddings_ready,
        }

    def search_with_fallback(self, query: SearchQuery) -> SearchResponse:
        """
        Execute search with graceful degradation (RNF-08).

        If vector search fails or is unavailable, automatically falls back
        to lexical-only search and returns a degraded response with metadata.

        Args:
            query: The search query

        Returns:
            SearchResponse with results and degradation metadata
        """
        start_time = time.time()
        degraded = False
        degradation_reason = None
        search_mode = "hybrid"
        metadata: Optional[SearchMetadata] = None

        # Check if vector search is available
        vector_available = self._vector_search.is_ready() and self._embeddings_store.is_ready()

        if not vector_available:
            # Graceful degradation: use lexical search only
            logger.warning("Vector search unavailable, degrading to lexical-only search")
            degraded = True
            degradation_reason = "Vector index (FAISS) unavailable - using lexical search only"
            search_mode = "lexical_only"

            try:
                results, meta = self._search_lexical_only_with_debug(query)
                metadata = SearchMetadata(
                    fusion_method="none",
                    rrf_k=None,
                    diversification_enabled=bool(query.use_diversification),
                    candidates_lexical=meta.get("candidates_lexical", 0),
                    candidates_vector=0,
                )
            except Exception as e:
                logger.error(f"Lexical search also failed: {e}")
                raise RuntimeError("Both vector and lexical search unavailable") from e
        else:
            # Try hybrid search, fallback to lexical if vector fails
            try:
                results, meta = self._search_hybrid_with_debug(query)
                metadata = SearchMetadata(
                    fusion_method="rrf",
                    rrf_k=meta.get("rrf_k", 60),
                    diversification_enabled=bool(query.use_diversification),
                    candidates_lexical=meta.get("candidates_lexical", 0),
                    candidates_vector=meta.get("candidates_vector", 0),
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to lexical: {e}")
                degraded = True
                degradation_reason = f"Vector search failed ({str(e)}) - using lexical search only"
                search_mode = "lexical_only"
                results, meta = self._search_lexical_only_with_debug(query)
                metadata = SearchMetadata(
                    fusion_method="none",
                    rrf_k=None,
                    diversification_enabled=bool(query.use_diversification),
                    candidates_lexical=meta.get("candidates_lexical", 0),
                    candidates_vector=0,
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

        # 2. Extract query intent using LLM
        if self._llm_client is None:
            logger.warning("LLM client not available, using basic search without understanding")
            # Fallback: create a basic QueryIntent with defaults
            query_intent = QueryIntent(
                intent_type="exploratory",
                original_query=query_text,
                reformulated_query=query_text,
                extracted_filters=SearchFilters(),
                confidence=0.0,
                reasoning="LLM client not available - using default exploratory intent"
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
            filters=query_intent.extracted_filters,
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

        if vector_available:
            try:
                results, meta = self._multi_query_search(
                    variations=variations,
                    strategy=strategy,
                    query=search_query,
                )

                # Apply filters (may already be partially applied by repos)
                results = self._apply_filters(results, search_query.filters)

                # Apply diversification if requested
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

                # Generate explanations if requested
                if use_explanations and self._llm_client is not None:
                    results = self._add_explanations(query_text, results)

                search_time_ms = (time.time() - start_time) * 1000
                metadata = SearchMetadata(
                    fusion_method="rrf",
                    rrf_k=60,
                    diversification_enabled=adjusted_diversification,
                    candidates_lexical=meta.get("variations_succeeded", 0),
                    candidates_vector=meta.get("variations_succeeded", 0),
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
                response = self.search_with_fallback(search_query)
        else:
            # Graceful degradation: vector unavailable, use lexical fallback
            logger.warning("Vector search unavailable, using fallback for understanding search")
            response = self.search_with_fallback(search_query)

        # 8. Log timing
        total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Search with understanding completed in {total_time_ms:.1f}ms "
            f"({len(variations)} variations, strategy={strategy.value})"
        )

        return response, query_intent

    def _search_lexical_only_with_debug(self, query: SearchQuery) -> tuple[List[SearchResult], Dict]:
        results = self._search_lexical_only(query)
        return results, {"candidates_lexical": len(results)}

    def _search_hybrid_with_debug(self, query: SearchQuery) -> tuple[list[SearchResult], Dict]:
        logger.info(f"Executing hybrid search for query: '{query.text}'")

        candidate_limit = query.max_results * 2

        logger.debug("Executing lexical (BM25) search")
        lexical_results = self._lexical_search.search(
            query_text=query.text,
            max_results=candidate_limit,
            filters=query.filters
        )

        logger.debug("Generating query embedding and executing vector search")
        query_embedding = self._embeddings_store.generate_embedding(query.text)
        vector_results = self._vector_search.search(
            query_embedding=query_embedding,
            max_results=candidate_limit,
            filters=query.filters,
        )

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

        if query.use_diversification:
            logger.debug(f"Applying MMR diversification with lambda={query.diversity_lambda}")
            final_results = self._apply_mmr_diversification(
                results=filtered_results,
                top_k=query.max_results,
                lambda_param=query.diversity_lambda,
            )
        else:
            final_results = filtered_results[: query.max_results]
            for i, result in enumerate(final_results, start=1):
                result.rank = i

        logger.info(f"Returning {len(final_results)} results")

        if query.use_explanations and self._llm_client is not None:
            logger.debug("Generating explanations for top results")
            final_results = self._add_explanations(query.text, final_results)

        return final_results, {
            "fusion_method": "rrf",
            "rrf_k": rrf_k,
            "candidates_lexical": len(lexical_results),
            "candidates_vector": len(vector_results),
        }

    def _search_lexical_only(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute lexical-only search (fallback mode).

        Used when vector search is unavailable for graceful degradation.

        Args:
            query: The search query

        Returns:
            List of SearchResult from BM25 search only
        """
        logger.info(f"Executing lexical-only search for query: '{query.text}'")

        results = self._lexical_search.search(
            query_text=query.text,
            max_results=query.max_results,
            filters=query.filters,
        )

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

        # Re-assign ranks
        for i, result in enumerate(results, start=1):
            result.rank = i

        return results

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Execute a hybrid search combining lexical and semantic approaches.

        This method implements the following algorithm:

        1. **Sequential retrieval:**
           - Execute BM25 search over book text (title, authors, description, categories)
           - Generate query embedding and execute vector similarity search
           - Both searches retrieve up to `query.max_results * 2` candidates

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
           - Limit results to query.max_results
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

        logger.info(f"Executing hybrid search for query: '{query.text}'")

        # Step 1: Sequential retrieval
        # Retrieve more candidates than needed to improve fusion quality
        candidate_limit = query.max_results * 2

        logger.debug("Executing lexical (BM25) search")
        lexical_results = self._lexical_search.search(
            query_text=query.text,
            max_results=candidate_limit,
            filters=query.filters
        )

        logger.debug("Generating query embedding and executing vector search")
        query_embedding = self._embeddings_store.generate_embedding(query.text)
        vector_results = self._vector_search.search(
            query_embedding=query_embedding,
            max_results=candidate_limit,
            filters=query.filters,
        )

        logger.debug(
            f"Retrieved {len(lexical_results)} lexical results, "
            f"{len(vector_results)} vector results"
        )

        # Step 2: Fusion via Reciprocal Rank Fusion
        fused_results = self._fuse_results_rrf(
            lexical_results=lexical_results,
            vector_results=vector_results,
            k=60,
        )

        logger.debug(f"Fused into {len(fused_results)} unique results")

        # Step 3: Post-processing
        # Apply filters (may already be applied by repositories, but ensure here)
        filtered_results = self._apply_filters(fused_results, query.filters)

        # Step 4: Optional MMR diversification
        if query.use_diversification:
            logger.debug(f"Applying MMR diversification with lambda={query.diversity_lambda}")
            final_results = self._apply_mmr_diversification(
                results=filtered_results,
                top_k=query.max_results,
                lambda_param=query.diversity_lambda,
            )
        else:
            # Limit to requested number of results
            final_results = filtered_results[: query.max_results]
            # Re-assign ranks
            for i, result in enumerate(final_results, start=1):
                result.rank = i

        logger.info(f"Returning {len(final_results)} results")

        # Step 5: Optional explanation generation
        if query.use_explanations and self._llm_client is not None:
            logger.debug("Generating explanations for top results")
            final_results = self._add_explanations(query.text, final_results)

        return final_results

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
        query_embedding = self._embeddings_store.generate_embedding(variation_text)
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
        max_results: int,
    ) -> List[SearchResult]:
        """
        Fuse results from multiple query variations using RRF.

        Each variation produced its own ranked list. We treat each as a
        separate "ranking system" and fuse with RRF - the same algorithm
        used for BM25+FAISS fusion, applied at a higher level.

        Books appearing in multiple variation rankings get boosted scores,
        which naturally surfaces results that are relevant from multiple
        perspectives.

        Args:
            all_results: List of ranked results, one per query variation
            max_results: Maximum number of final results

        Returns:
            Fused and deduplicated list of SearchResult
        """
        rrf_scores: Dict[UUID, float] = {}
        books_map: Dict[UUID, Book] = {}
        k = 60

        for variation_results in all_results:
            for result in variation_results:
                book_id = result.book.id
                rrf_scores[book_id] = (
                    rrf_scores.get(book_id, 0.0) + (1.0 / (k + result.rank))
                )
                books_map[book_id] = result.book

        sorted_ids = sorted(rrf_scores, key=lambda bid: rrf_scores[bid], reverse=True)

        return [
            SearchResult(
                book=books_map[bid],
                final_score=rrf_scores[bid],
                rank=i + 1,
                source="hybrid",
            )
            for i, bid in enumerate(sorted_ids[:max_results])
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

        If a variation fails, results from other variations are still used
        (graceful degradation). If ALL fail, falls back to single balanced search.

        Args:
            variations: List of query text variations (2-3 typically)
            strategy: Retrieval strategy controlling pool sizes
            query: Original SearchQuery (for filters and max_results)

        Returns:
            Tuple of (fused results, debug metadata dict)
        """
        logger.info(
            f"Multi-query search: {len(variations)} variations, "
            f"strategy={strategy.value}"
        )

        all_variation_results: List[List[SearchResult]] = []

        # Run all variations in parallel
        with ThreadPoolExecutor(max_workers=len(variations)) as executor:
            future_to_variation = {
                executor.submit(
                    self._search_hybrid_with_strategy,
                    variation_text=variation,
                    strategy=strategy,
                    filters=query.filters,
                    max_results=query.max_results,
                ): variation
                for variation in variations
            }

            for future in as_completed(future_to_variation):
                variation = future_to_variation[future]
                try:
                    results = future.result()
                    all_variation_results.append(results)
                    logger.debug(
                        f"Variation '{variation[:30]}...' returned {len(results)} results"
                    )
                except Exception as e:
                    logger.warning(f"Variation '{variation[:30]}...' failed: {e}")

        if not all_variation_results:
            # All variations failed - fall back to single balanced search
            logger.warning("All variations failed, falling back to single balanced search")
            fallback = self._search_hybrid_with_strategy(
                variation_text=variations[0],
                strategy=RetrievalStrategy.BALANCED,
                filters=query.filters,
                max_results=query.max_results,
            )
            all_variation_results = [fallback]

        # Fuse results across all variations
        fused = self._fuse_multi_query_results(all_variation_results, query.max_results)

        metadata = {
            "fusion_method": "rrf",
            "rrf_k": 60,
            "strategy": strategy.value,
            "variations_attempted": len(variations),
            "variations_succeeded": len(all_variation_results),
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

        # SearchQuery.filters always provides a SearchFilters instance (never None)
        # due to default_factory, but check is_empty() for efficiency
        if filters.is_empty():
            return results

        filtered = []
        for result in results:
            book = result.book

            # Language filter
            if filters.language is not None:
                if book.language != filters.language:
                    continue

            # Category filter
            if filters.category is not None:
                if filters.category not in book.categories:
                    continue

            # Year range filter
            pub_year = book.get_published_year()
            if pub_year is not None:
                if filters.min_year is not None and pub_year < filters.min_year:
                    continue
                if filters.max_year is not None and pub_year > filters.max_year:
                    continue

            filtered.append(result)

        return filtered

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

        Args:
            query: The original search query
            results: List of search results

        Returns:
            Results with explanation field populated
        """
        if self._llm_client is None:
            logger.warning("LLM Client not available, skipping explanations")
            return results

        for result in results[:5]:
            try:
                explanation = self._llm_client.generate_grounded_explanation(
                    query_text=query_text,
                    book=result.book,
                )
                result.explanation = explanation
            except Exception as e:
                logger.error(f"Failed to generate explanation for book {result.book.id}: {e}")
                # Continue with no explanation for this result

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
        - Relevance(d): the RRF score from hybrid search
        - Similarity: cosine similarity between book embeddings

        Args:
            results: List of search results (already ranked by RRF)
            top_k: Number of results to select
            lambda_param: Trade-off parameter (default: 0.6, slight preference for relevance)

        Returns:
            Reranked list of top_k results optimized for diversity
        """
        if len(results) <= 1:
            return results

        # Get embeddings for all candidate books
        book_embeddings: Dict[UUID, List[float]] = {}
        for result in results:
            embedding = self._embeddings_store.get_embedding(result.book.id)
            if embedding is not None:
                book_embeddings[result.book.id] = embedding

        # If no embeddings available, return original results
        if not book_embeddings:
            logger.warning("No embeddings available for MMR diversification, skipping")
            return results[:top_k]

        # Normalize RRF scores to [0, 1] for fair comparison with similarity
        max_score = max(r.final_score for r in results) if results else 1.0
        min_score = min(r.final_score for r in results) if results else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        def normalize_score(score: float) -> float:
            return (score - min_score) / score_range

        # Greedy MMR selection
        selected: List[SearchResult] = []
        candidates = list(results)

        while len(selected) < top_k and candidates:
            best_mmr_score = float('-inf')
            best_idx = 0

            for i, candidate in enumerate(candidates):
                # Skip if no embedding
                if candidate.book.id not in book_embeddings:
                    continue

                # Relevance term (normalized RRF score)
                relevance = normalize_score(candidate.final_score)

                # Diversity term: max similarity to any already-selected document
                if selected:
                    max_similarity = max(
                        self._cosine_similarity(
                            book_embeddings[candidate.book.id],
                            book_embeddings[s.book.id]
                        )
                        for s in selected
                        if s.book.id in book_embeddings
                    ) if any(s.book.id in book_embeddings for s in selected) else 0.0
                else:
                    max_similarity = 0.0

                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            # Add best candidate to selected
            selected.append(candidates.pop(best_idx))

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