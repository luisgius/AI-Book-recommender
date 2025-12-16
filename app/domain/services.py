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

from .entities import Book, SearchResult, Explanation
from .value_objects import SearchQuery, SearchFilters
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

        # Limit to requested number of results
        final_results = filtered_results[: query.max_results]

        # Re-assign ranks
        for i, result in enumerate(final_results, start=1):
            result.rank = i

        logger.info(f"Returning {len(final_results)} results")

        # Step 4: Optional explanation generation
        if query.use_explanations and self._llm_client is not None:
            logger.debug("Generating explanations for top results")
            final_results = self._add_explanations(query, final_results)

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