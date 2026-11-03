"""
Tests for multi-query retrieval and strategy router in SearchService.

These tests verify:
1. _search_hybrid_with_strategy() uses strategy-controlled pool sizes
2. _fuse_multi_query_results() correctly fuses N ranked lists with RRF
3. _multi_query_search() runs variations in parallel and degrades gracefully
4. search_with_understanding() integrates strategy + variations end-to-end

Test strategy:
- Uses fakes (not mocks) for deterministic, implementation-decoupled tests
- Fakes record call arguments so we can verify pool sizes and query text
- Each test targets ONE behavior to isolate failures
"""

import pytest
from uuid import uuid4, UUID
from typing import List, Optional, Dict
from unittest.mock import Mock

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import (
    SearchQuery, SearchFilters, QueryIntent,
    RetrievalStrategy, STRATEGY_POOL_SIZES, INTENT_TO_STRATEGY,
)
from app.domain.services import SearchService


# =============================================================================
# Fakes: record call arguments for assertion
# =============================================================================


class SpyLexicalSearchRepository:
    """
    Fake lexical search that records every call's arguments.

    Returns a fixed set of results based on book catalog.
    The spy pattern lets us verify WHAT arguments were passed
    (e.g., max_results reflecting strategy pool sizes) without
    coupling to implementation details.
    """

    def __init__(self, books: List[Book]):
        self._books = books
        self.calls: List[Dict] = []

    def search(
        self,
        query_text: str,
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        self.calls.append({
            "query_text": query_text,
            "max_results": max_results,
            "filters": filters,
        })
        # Return all books as results, limited by max_results
        return [
            SearchResult(
                book=book,
                final_score=1.0 / (i + 1),  # Descending scores
                rank=i + 1,
                source="lexical",
                lexical_score=1.0 / (i + 1),
            )
            for i, book in enumerate(self._books[:max_results])
        ]

    def is_ready(self) -> bool:
        return True

    def build_index(self, books):
        pass

    def add_to_index(self, book):
        pass

    def save_index(self, path):
        pass

    def load_index(self, path):
        pass


class SpyVectorSearchRepository:
    """
    Fake vector search that records every call's arguments.

    Returns books in reverse order compared to lexical, so RRF
    fusion produces meaningful blended rankings.
    """

    def __init__(self, books: List[Book]):
        self._books = list(reversed(books))  # Reverse order
        self.calls: List[Dict] = []

    def search(
        self,
        query_embedding: List[float],
        max_results: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        self.calls.append({
            "query_embedding": query_embedding,
            "max_results": max_results,
            "filters": filters,
        })
        return [
            SearchResult(
                book=book,
                final_score=1.0 / (i + 1),
                rank=i + 1,
                source="vector",
                vector_score=1.0 / (i + 1),
            )
            for i, book in enumerate(self._books[:max_results])
        ]

    def is_ready(self) -> bool:
        return True


class FakeEmbeddingsStore:
    """Fake embeddings store that returns deterministic embeddings."""

    def __init__(self):
        self._embeddings: Dict[UUID, List[float]] = {}

    def generate_embedding(self, text: str) -> List[float]:
        return [0.1] * 384

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return self._embeddings.get(book_id, [0.1] * 384)

    def store_embedding(self, book_id: UUID, embedding: List[float]) -> None:
        self._embeddings[book_id] = embedding

    def is_ready(self) -> bool:
        return True

    def get_dimension(self) -> int:
        return 384


class FailingVectorSearchRepository:
    """Vector search that raises an exception on every search call."""

    def search(self, query_embedding, max_results=10, filters=None):
        raise RuntimeError("Vector search exploded")

    def is_ready(self) -> bool:
        return True


class FailingEmbeddingsStore:
    """Embeddings store that raises an exception."""

    def generate_embedding(self, text: str):
        raise RuntimeError("Embeddings exploded")

    def get_embedding(self, book_id):
        return None

    def is_ready(self) -> bool:
        return True

    def get_dimension(self) -> int:
        return 384


# =============================================================================
# Helpers
# =============================================================================


def create_book(title: str, book_id: UUID = None) -> Book:
    """Create a test book with minimal required fields."""
    return Book(
        id=book_id or uuid4(),
        title=title,
        authors=["Test Author"],
        source="test",
        source_id=f"test-{title.lower().replace(' ', '-')}",
    )


def create_books(n: int) -> List[Book]:
    """Create n test books with unique IDs."""
    return [create_book(f"Book {i+1}") for i in range(n)]


def create_service(books: List[Book], llm_client=None) -> tuple:
    """
    Create a SearchService with spy repositories.

    Returns (service, lexical_spy, vector_spy) so tests can
    inspect call arguments.
    """
    lexical = SpyLexicalSearchRepository(books)
    vector = SpyVectorSearchRepository(books)
    embeddings = FakeEmbeddingsStore()

    service = SearchService(
        lexical_search=lexical,
        vector_search=vector,
        embeddings_store=embeddings,
        llm_client=llm_client,
    )

    return service, lexical, vector


# =============================================================================
# Tests: _search_hybrid_with_strategy
# =============================================================================


class TestSearchHybridWithStrategy:
    """Tests for strategy-controlled pool sizes in hybrid search."""

    def test_vector_heavy_gives_more_vector_candidates(self):
        """VECTOR_HEAVY should request more vector candidates than BM25."""
        books = create_books(5)
        service, lexical_spy, vector_spy = create_service(books)

        service._search_hybrid_with_strategy(
            variation_text="test query",
            strategy=RetrievalStrategy.VECTOR_HEAVY,
            filters=SearchFilters(),
            max_results=10,
        )

        # With VECTOR_HEAVY: bm25_factor=1, vector_factor=4
        # So bm25_top_k = 10*1 = 10, vector_top_k = 10*4 = 40
        assert lexical_spy.calls[0]["max_results"] == 10
        assert vector_spy.calls[0]["max_results"] == 40

    def test_lexical_heavy_gives_more_bm25_candidates(self):
        """LEXICAL_HEAVY should request more BM25 candidates than vector."""
        books = create_books(5)
        service, lexical_spy, vector_spy = create_service(books)

        service._search_hybrid_with_strategy(
            variation_text="who wrote 1984",
            strategy=RetrievalStrategy.LEXICAL_HEAVY,
            filters=SearchFilters(),
            max_results=10,
        )

        # With LEXICAL_HEAVY: bm25_factor=4, vector_factor=1
        assert lexical_spy.calls[0]["max_results"] == 40
        assert vector_spy.calls[0]["max_results"] == 10

    def test_balanced_gives_equal_candidates(self):
        """BALANCED should request equal candidates from both sources."""
        books = create_books(5)
        service, lexical_spy, vector_spy = create_service(books)

        service._search_hybrid_with_strategy(
            variation_text="science fiction",
            strategy=RetrievalStrategy.BALANCED,
            filters=SearchFilters(),
            max_results=10,
        )

        # With BALANCED: bm25_factor=2, vector_factor=2
        assert lexical_spy.calls[0]["max_results"] == 20
        assert vector_spy.calls[0]["max_results"] == 20

    def test_pool_sizes_scale_with_max_results(self):
        """Pool sizes should be factors * max_results, scaling linearly."""
        books = create_books(5)
        service, lexical_spy, vector_spy = create_service(books)

        # Use max_results=5 instead of 10
        service._search_hybrid_with_strategy(
            variation_text="test",
            strategy=RetrievalStrategy.VECTOR_HEAVY,
            filters=SearchFilters(),
            max_results=5,
        )

        # bm25_factor=1 * 5 = 5, vector_factor=4 * 5 = 20
        assert lexical_spy.calls[0]["max_results"] == 5
        assert vector_spy.calls[0]["max_results"] == 20

    def test_passes_query_text_to_lexical(self):
        """The variation text should be forwarded to lexical search."""
        books = create_books(3)
        service, lexical_spy, _ = create_service(books)

        service._search_hybrid_with_strategy(
            variation_text="dystopian fiction totalitarianism",
            strategy=RetrievalStrategy.BALANCED,
            filters=SearchFilters(),
            max_results=10,
        )

        assert lexical_spy.calls[0]["query_text"] == "dystopian fiction totalitarianism"

    def test_passes_filters_to_both_repos(self):
        """Filters should be forwarded to both lexical and vector repos."""
        books = create_books(3)
        service, lexical_spy, vector_spy = create_service(books)
        filters = SearchFilters(language="es")

        service._search_hybrid_with_strategy(
            variation_text="test",
            strategy=RetrievalStrategy.BALANCED,
            filters=filters,
            max_results=10,
        )

        assert lexical_spy.calls[0]["filters"] == filters
        assert vector_spy.calls[0]["filters"] == filters

    def test_returns_rrf_fused_results(self):
        """Results should be RRF-fused (source='hybrid')."""
        books = create_books(3)
        service, _, _ = create_service(books)

        results = service._search_hybrid_with_strategy(
            variation_text="test",
            strategy=RetrievalStrategy.BALANCED,
            filters=SearchFilters(),
            max_results=10,
        )

        assert len(results) > 0
        for r in results:
            assert r.source == "hybrid"


# =============================================================================
# Tests: _fuse_multi_query_results
# =============================================================================


class TestFuseMultiQueryResults:
    """Tests for second-level RRF fusion across query variations."""

    def test_single_variation_preserves_ranking(self):
        """With one variation, fusion should preserve the original ranking."""
        books = create_books(3)
        service, _, _ = create_service(books)

        variation_results = [
            SearchResult(book=books[0], final_score=0.5, rank=1, source="hybrid"),
            SearchResult(book=books[1], final_score=0.3, rank=2, source="hybrid"),
            SearchResult(book=books[2], final_score=0.1, rank=3, source="hybrid"),
        ]

        fused = service._fuse_multi_query_results([variation_results], max_results=10)

        assert len(fused) == 3
        assert fused[0].book.id == books[0].id
        assert fused[1].book.id == books[1].id
        assert fused[2].book.id == books[2].id

    def test_book_in_multiple_variations_gets_boosted(self):
        """A book appearing in two variation rankings should get a higher RRF score."""
        book_a = create_book("Book A")
        book_b = create_book("Book B")
        book_c = create_book("Book C")
        service, _, _ = create_service([book_a, book_b, book_c])

        # Variation 1: A rank 1, B rank 2
        variation_1 = [
            SearchResult(book=book_a, final_score=0.5, rank=1, source="hybrid"),
            SearchResult(book=book_b, final_score=0.3, rank=2, source="hybrid"),
        ]

        # Variation 2: C rank 1, A rank 2
        # Book A appears in BOTH variations -> boosted
        variation_2 = [
            SearchResult(book=book_c, final_score=0.5, rank=1, source="hybrid"),
            SearchResult(book=book_a, final_score=0.3, rank=2, source="hybrid"),
        ]

        fused = service._fuse_multi_query_results(
            [variation_1, variation_2], max_results=10
        )

        # Book A should be ranked first because it appears in both variations
        # RRF for A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        # RRF for C: 1/(60+1) = 0.01639
        # RRF for B: 1/(60+2) = 0.01613
        assert fused[0].book.id == book_a.id

    def test_max_results_limits_output(self):
        """Fusion should respect the max_results parameter."""
        books = create_books(5)
        service, _, _ = create_service(books)

        variation = [
            SearchResult(book=b, final_score=0.5, rank=i+1, source="hybrid")
            for i, b in enumerate(books)
        ]

        fused = service._fuse_multi_query_results([variation], max_results=2)

        assert len(fused) == 2

    def test_ranks_are_reassigned_starting_from_one(self):
        """Fused results should have ranks 1, 2, 3, ..."""
        books = create_books(3)
        service, _, _ = create_service(books)

        variation = [
            SearchResult(book=b, final_score=0.5, rank=i+1, source="hybrid")
            for i, b in enumerate(books)
        ]

        fused = service._fuse_multi_query_results([variation], max_results=10)

        for i, result in enumerate(fused):
            assert result.rank == i + 1

    def test_empty_variations_returns_empty(self):
        """Fusing zero variation lists should return empty."""
        service, _, _ = create_service([])

        fused = service._fuse_multi_query_results([], max_results=10)

        assert fused == []

    def test_deduplicates_across_variations(self):
        """Same book in different variations should appear once in output."""
        book = create_book("Shared Book")
        service, _, _ = create_service([book])

        variation_1 = [
            SearchResult(book=book, final_score=0.5, rank=1, source="hybrid"),
        ]
        variation_2 = [
            SearchResult(book=book, final_score=0.4, rank=1, source="hybrid"),
        ]

        fused = service._fuse_multi_query_results(
            [variation_1, variation_2], max_results=10
        )

        assert len(fused) == 1
        assert fused[0].book.id == book.id


# =============================================================================
# Tests: _multi_query_search (parallel execution)
# =============================================================================


class TestMultiQuerySearch:
    """Tests for parallel multi-query search execution."""

    def test_searches_each_variation(self):
        """Each variation should trigger a separate hybrid search."""
        books = create_books(3)
        service, lexical_spy, vector_spy = create_service(books)

        query = SearchQuery(text="test query")
        variations = ["variation A", "variation B"]

        service._multi_query_search(
            variations=variations,
            strategy=RetrievalStrategy.BALANCED,
            query=query,
        )

        # Each variation should produce one lexical + one vector call
        assert len(lexical_spy.calls) == 2
        assert len(vector_spy.calls) == 2

        # Verify both variation texts were used
        queried_texts = {call["query_text"] for call in lexical_spy.calls}
        assert "variation A" in queried_texts
        assert "variation B" in queried_texts

    def test_returns_fused_results(self):
        """Multi-query search should return fused results from all variations."""
        books = create_books(5)
        service, _, _ = create_service(books)

        query = SearchQuery(text="test query")

        results, metadata = service._multi_query_search(
            variations=["query 1", "query 2"],
            strategy=RetrievalStrategy.BALANCED,
            query=query,
        )

        assert len(results) > 0
        # Results should be from hybrid source (RRF fused)
        for r in results:
            assert r.source == "hybrid"

    def test_metadata_tracks_variation_counts(self):
        """Metadata should report how many variations were attempted and succeeded."""
        books = create_books(3)
        service, _, _ = create_service(books)

        query = SearchQuery(text="test")

        _, metadata = service._multi_query_search(
            variations=["v1", "v2", "v3"],
            strategy=RetrievalStrategy.BALANCED,
            query=query,
        )

        assert metadata["variations_attempted"] == 3
        assert metadata["variations_succeeded"] == 3
        assert metadata["strategy"] == "balanced"

    def test_single_variation_works(self):
        """Multi-query search with one variation should work normally."""
        books = create_books(3)
        service, lexical_spy, _ = create_service(books)

        query = SearchQuery(text="test")

        results, metadata = service._multi_query_search(
            variations=["only variation"],
            strategy=RetrievalStrategy.VECTOR_HEAVY,
            query=query,
        )

        assert len(results) > 0
        assert metadata["variations_succeeded"] == 1
        assert lexical_spy.calls[0]["query_text"] == "only variation"

    def test_applies_strategy_to_all_variations(self):
        """All variations should use the same retrieval strategy."""
        books = create_books(3)
        service, lexical_spy, vector_spy = create_service(books)

        query = SearchQuery(text="test", max_results=10)

        service._multi_query_search(
            variations=["v1", "v2"],
            strategy=RetrievalStrategy.LEXICAL_HEAVY,
            query=query,
        )

        # LEXICAL_HEAVY: bm25_factor=4, vector_factor=1
        # All calls should use 10*4=40 for lexical and 10*1=10 for vector
        for call in lexical_spy.calls:
            assert call["max_results"] == 40
        for call in vector_spy.calls:
            assert call["max_results"] == 10


# =============================================================================
# Tests: Graceful degradation in multi-query search
# =============================================================================


class TestMultiQueryGracefulDegradation:
    """Tests for graceful degradation when variations fail."""

    def test_partial_failure_uses_successful_variations(self):
        """If some variations fail, results from successful ones are still used."""
        books = create_books(3)
        call_count = 0

        class PartiallyFailingLexical(SpyLexicalSearchRepository):
            """Fails on second call, succeeds on first."""

            def search(self, query_text, max_results=10, filters=None):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Transient failure")
                return super().search(query_text, max_results, filters)

        lexical = PartiallyFailingLexical(books)
        vector = SpyVectorSearchRepository(books)
        embeddings = FakeEmbeddingsStore()

        service = SearchService(
            lexical_search=lexical,
            vector_search=vector,
            embeddings_store=embeddings,
        )

        query = SearchQuery(text="test")

        results, metadata = service._multi_query_search(
            variations=["v1", "v2"],
            strategy=RetrievalStrategy.BALANCED,
            query=query,
        )

        # Should still return results from the successful variation
        assert len(results) > 0
        # One variation succeeded (the other failed at lexical stage)
        assert metadata["variations_succeeded"] >= 1


# =============================================================================
# Tests: search_with_understanding integration with multi-query
# =============================================================================


class TestSearchWithUnderstandingMultiQuery:
    """Tests for multi-query integration in search_with_understanding."""

    def test_builds_variations_from_intent(self):
        """Variations from LangGraph should be passed to multi-query search."""
        books = create_books(5)

        mock_llm = Mock()
        mock_llm.extract_query_intent.return_value = QueryIntent(
            intent_type="recommendation",
            original_query="books like 1984",
            reformulated_query="1984 dystopian fiction",
            extracted_filters=SearchFilters(),
            confidence=0.9,
            reasoning="Recommendation query",
            query_variations=["1984 dystopian fiction", "dystopian totalitarianism"],
        )

        service, lexical_spy, _ = create_service(books, llm_client=mock_llm)

        response, intent = service.search_with_understanding(
            query_text="books like 1984"
        )

        # The reformulated query is always the primary variation.
        # Additional variations from LangGraph are appended (deduplicated).
        # So we expect 2 unique variations:
        #   "1984 dystopian fiction" (primary = reformulated)
        #   "dystopian totalitarianism" (from LangGraph)
        # Each variation triggers a lexical + vector call
        queried_texts = {call["query_text"] for call in lexical_spy.calls}
        assert "1984 dystopian fiction" in queried_texts
        assert "dystopian totalitarianism" in queried_texts

    def test_deduplicates_variations(self):
        """If LangGraph returns the same text as reformulated, don't search twice."""
        books = create_books(3)

        mock_llm = Mock()
        mock_llm.extract_query_intent.return_value = QueryIntent(
            intent_type="factual",
            original_query="who wrote 1984",
            reformulated_query="1984 author George Orwell",
            extracted_filters=SearchFilters(),
            confidence=0.95,
            reasoning="Factual query",
            # This variation is identical to reformulated_query
            query_variations=["1984 author George Orwell", "who wrote 1984"],
        )

        service, lexical_spy, _ = create_service(books, llm_client=mock_llm)

        service.search_with_understanding(query_text="who wrote 1984")

        # Should only search unique variations:
        # "1984 author George Orwell" (primary, deduplicated from variations)
        # "who wrote 1984" (additional variation)
        queried_texts = [call["query_text"] for call in lexical_spy.calls]
        assert queried_texts.count("1984 author George Orwell") == 1

    def test_recommendation_uses_vector_heavy_strategy(self):
        """Recommendation intent should apply VECTOR_HEAVY pool sizes."""
        books = create_books(3)

        mock_llm = Mock()
        mock_llm.extract_query_intent.return_value = QueryIntent(
            intent_type="recommendation",
            original_query="books like 1984",
            reformulated_query="1984 dystopian",
            extracted_filters=SearchFilters(),
            confidence=0.9,
            reasoning="Recommendation",
        )

        service, lexical_spy, vector_spy = create_service(books, llm_client=mock_llm)

        service.search_with_understanding(
            query_text="books like 1984",
            max_results=10,
        )

        # VECTOR_HEAVY: bm25_factor=1, vector_factor=4
        # lexical gets 10*1=10, vector gets 10*4=40
        assert lexical_spy.calls[0]["max_results"] == 10
        assert vector_spy.calls[0]["max_results"] == 40

    def test_factual_uses_lexical_heavy_strategy(self):
        """Factual intent should apply LEXICAL_HEAVY pool sizes."""
        books = create_books(3)

        mock_llm = Mock()
        mock_llm.extract_query_intent.return_value = QueryIntent(
            intent_type="factual",
            original_query="who wrote 1984",
            reformulated_query="1984 author",
            extracted_filters=SearchFilters(),
            confidence=0.95,
            reasoning="Factual",
        )

        service, lexical_spy, vector_spy = create_service(books, llm_client=mock_llm)

        service.search_with_understanding(
            query_text="who wrote 1984",
            max_results=10,
        )

        # LEXICAL_HEAVY: bm25_factor=4, vector_factor=1
        assert lexical_spy.calls[0]["max_results"] == 40
        assert vector_spy.calls[0]["max_results"] == 10

    def test_exploratory_uses_balanced_strategy(self):
        """Exploratory intent should apply BALANCED pool sizes."""
        books = create_books(3)

        mock_llm = Mock()
        mock_llm.extract_query_intent.return_value = QueryIntent(
            intent_type="exploratory",
            original_query="science fiction AI",
            reformulated_query="science fiction artificial intelligence",
            extracted_filters=SearchFilters(),
            confidence=0.8,
            reasoning="Exploratory",
        )

        service, lexical_spy, vector_spy = create_service(books, llm_client=mock_llm)

        service.search_with_understanding(
            query_text="science fiction AI",
            max_results=10,
        )

        # BALANCED: bm25_factor=2, vector_factor=2
        assert lexical_spy.calls[0]["max_results"] == 20
        assert vector_spy.calls[0]["max_results"] == 20

    def test_no_llm_falls_back_to_balanced(self):
        """Without LLM client, should use BALANCED strategy (exploratory default)."""
        books = create_books(3)

        service, lexical_spy, vector_spy = create_service(books, llm_client=None)

        response, intent = service.search_with_understanding(
            query_text="some query",
            max_results=10,
        )

        # Default intent is "exploratory" -> BALANCED
        assert intent.intent_type == "exploratory"
        assert lexical_spy.calls[0]["max_results"] == 20
        assert vector_spy.calls[0]["max_results"] == 20

    def test_returns_search_response_and_query_intent(self):
        """Method should return a (SearchResponse, QueryIntent) tuple."""
        books = create_books(3)

        mock_llm = Mock()
        mock_llm.extract_query_intent.return_value = QueryIntent(
            intent_type="recommendation",
            original_query="test",
            reformulated_query="test reformulated",
            extracted_filters=SearchFilters(),
            confidence=0.9,
            reasoning="Test",
        )

        service, _, _ = create_service(books, llm_client=mock_llm)

        response, intent = service.search_with_understanding(query_text="test")

        from app.domain.value_objects import SearchResponse
        assert isinstance(response, SearchResponse)
        assert isinstance(intent, QueryIntent)
        assert response.results is not None
        assert not response.degraded
