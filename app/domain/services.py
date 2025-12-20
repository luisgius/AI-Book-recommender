"""
Domain services for the book recommendation system.

Services orchestrate domain logic that doesn't naturally belong to a single
entity. They coordinate between entities and ports to implement use cases.

Following Hexagonal Architecture principles, services depend only on domain
entities, value objects, and port protocols (never on concrete implementations).
"""

from typing import List, Optional, Dict
from uuid import UUID
import logging
import time

from .entities import Book, SearchResult, Explanation
from .value_objects import SearchQuery, SearchFilters, SearchResponse, SearchMetadata
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
        llm_client: Optional[LLMClient] = None,
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
            final_results = self._add_explanations(query, final_results)

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
            final_results = self._add_explanations(query, final_results)

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
        query: SearchQuery,
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

        for result in results:
            try:
                explanation = self._llm_client.generate_explanation(
                    query=query,
                    book=result.book,
                    context={
                        "rank": result.rank,
                        "final_score": result.final_score,
                        "lexical_score": result.lexical_score,
                        "vector_score": result.vector_score,
                    },
                )
                result.explanation = explanation.text
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