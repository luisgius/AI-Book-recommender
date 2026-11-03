"""
Tests for domain value objects.
"""

import pytest

from app.domain.value_objects import (
    SearchFilters, SearchQuery, BookMetadata, QueryIntent,
    RetrievalStrategy, STRATEGY_POOL_SIZES, INTENT_TO_STRATEGY,
)


class TestSearchFilters:
    """Tests for the SearchFilters value object."""

    def test_create_empty_filters(self):
        """Test creating filters with no restrictions."""
        filters = SearchFilters()

        assert filters.language is None
        assert filters.category is None
        assert filters.min_year is None
        assert filters.max_year is None
        assert filters.is_empty() is True

    def test_create_filters_with_language(self):
        """Test creating filters with language restriction."""
        filters = SearchFilters(language="es")

        assert filters.language == "es"
        assert filters.is_empty() is False

    def test_create_filters_with_year_range(self):
        """Test creating filters with year range."""
        filters = SearchFilters(min_year=2000, max_year=2020)

        assert filters.min_year == 2000
        assert filters.max_year == 2020

    def test_filters_validation_invalid_year_range(self):
        """Test that min_year > max_year raises ValueError."""
        with pytest.raises(ValueError, match="min_year.*cannot be greater"):
            SearchFilters(min_year=2020, max_year=2000)

    def test_filters_validation_invalid_language(self):
        """Test that invalid language code raises ValueError."""
        with pytest.raises(ValueError, match="2-letter ISO 639-1 code"):
            SearchFilters(language="spanish")

    def test_filters_immutability(self):
        """Test that filters are immutable (frozen dataclass)."""
        filters = SearchFilters(language="en")

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            filters.language = "es"


class TestSearchQuery:
    """Tests for the SearchQuery value object."""

    def test_create_query_with_text_only(self):
        """Test creating a query with just text."""
        query = SearchQuery(text="python programming")

        assert query.text == "python programming"
        assert query.filters.is_empty()
        assert query.max_results == 10
        assert query.use_explanations is False

    def test_create_query_with_filters(self):
        """Test creating a query with filters."""
        filters = SearchFilters(language="en", category="Programming")
        query = SearchQuery(
            text="clean code",
            filters=filters,
            max_results=20,
            use_explanations=True,
        )

        assert query.text == "clean code"
        assert query.filters.language == "en"
        assert query.filters.category == "Programming"
        assert query.max_results == 20
        assert query.use_explanations is True

    def test_query_validation_empty_text(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SearchQuery(text="")

    def test_query_validation_whitespace_only_text(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SearchQuery(text="   ")

    def test_query_validation_max_results_too_low(self):
        """Test that max_results < 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            SearchQuery(text="query", max_results=0)

    def test_query_validation_max_results_too_high(self):
        """Test that max_results > 100 raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed 100"):
            SearchQuery(text="query", max_results=101)

    def test_query_immutability(self):
        """Test that queries are immutable."""
        query = SearchQuery(text="test")

        with pytest.raises(Exception):
            query.text = "changed"


class TestBookMetadata:
    """Tests for the BookMetadata value object."""

    def test_create_empty_metadata(self):
        """Test creating metadata with no data."""
        metadata = BookMetadata()

        assert metadata.isbn is None
        assert metadata.publisher is None
        assert metadata.page_count is None
        assert metadata.average_rating is None

    def test_create_metadata_with_all_fields(self):
        """Test creating metadata with all fields."""
        metadata = BookMetadata(
            isbn="0132350882",
            isbn13="9780132350884",
            publisher="Prentice Hall",
            page_count=464,
            average_rating=4.5,
            ratings_count=1500,
            thumbnail_url="http://example.com/cover.jpg",
            preview_link="http://example.com/preview",
        )

        assert metadata.isbn == "0132350882"
        assert metadata.publisher == "Prentice Hall"
        assert metadata.page_count == 464
        assert metadata.average_rating == 4.5
        assert metadata.ratings_count == 1500

    def test_metadata_validation_invalid_rating_too_low(self):
        """Test that rating < 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 5.0"):
            BookMetadata(average_rating=-1.0)

    def test_metadata_validation_invalid_rating_too_high(self):
        """Test that rating > 5 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 5.0"):
            BookMetadata(average_rating=6.0)

    def test_metadata_validation_negative_page_count(self):
        """Test that negative page count raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            BookMetadata(page_count=-10)

    def test_metadata_validation_negative_ratings_count(self):
        """Test that negative ratings count raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            BookMetadata(ratings_count=-5)

    def test_metadata_immutability(self):
        """Test that metadata is immutable."""
        metadata = BookMetadata(isbn="123456")

        with pytest.raises(Exception):
            metadata.isbn = "654321"


class TestQueryIntent:
    """Tests for the QueryIntent value object (Block 2)."""

    def test_create_recommendation_intent(self):
        """Test creating a recommendation intent."""
        intent = QueryIntent(
            intent_type="recommendation",
            original_query="books like 1984",
            reformulated_query="1984 dystopian fiction similar",
            extracted_filters=SearchFilters(),
            confidence=0.95,
            reasoning="User wants similar books to 1984"
        )

        assert intent.intent_type == "recommendation"
        assert intent.original_query == "books like 1984"
        assert intent.reformulated_query == "1984 dystopian fiction similar"
        assert intent.confidence == 0.95

    def test_create_factual_intent(self):
        """Test creating a factual intent."""
        intent = QueryIntent(
            intent_type="factual",
            original_query="who wrote Don Quixote",
            reformulated_query="Don Quixote author",
            extracted_filters=SearchFilters(),
            confidence=0.9,
            reasoning="User asking a specific question about authorship"
        )

        assert intent.intent_type == "factual"

    def test_create_exploratory_intent(self):
        """Test creating an exploratory intent."""
        intent = QueryIntent(
            intent_type="exploratory",
            original_query="science fiction about AI",
            reformulated_query="science fiction artificial intelligence",
            extracted_filters=SearchFilters(category="Science Fiction"),
            confidence=0.8,
            reasoning="User exploring a topic/genre"
        )

        assert intent.intent_type == "exploratory"
        assert intent.extracted_filters.category == "Science Fiction"

    def test_intent_with_extracted_filters(self):
        """Test intent with filters extracted from natural language."""
        filters = SearchFilters(language="es", min_year=1900, max_year=1999)
        intent = QueryIntent(
            intent_type="recommendation",
            original_query="Spanish novels from the 1900s like Don Quixote",
            reformulated_query="Don Quixote Spanish literature novels",
            extracted_filters=filters,
            confidence=0.85,
            reasoning="Recommendation query with language and year filters"
        )

        assert intent.extracted_filters.language == "es"
        assert intent.extracted_filters.min_year == 1900
        assert intent.extracted_filters.max_year == 1999

    def test_intent_validation_empty_original_query(self):
        """Test that empty original_query raises ValueError."""
        with pytest.raises(ValueError, match="original_query cannot be empty"):
            QueryIntent(
                intent_type="exploratory",
                original_query="",
                reformulated_query="some query",
                extracted_filters=SearchFilters(),
                confidence=0.5,
                reasoning="Test"
            )

    def test_intent_validation_whitespace_original_query(self):
        """Test that whitespace-only original_query raises ValueError."""
        with pytest.raises(ValueError, match="original_query cannot be empty"):
            QueryIntent(
                intent_type="exploratory",
                original_query="   ",
                reformulated_query="some query",
                extracted_filters=SearchFilters(),
                confidence=0.5,
                reasoning="Test"
            )

    def test_intent_validation_empty_reformulated_query(self):
        """Test that empty reformulated_query raises ValueError."""
        with pytest.raises(ValueError, match="reformulated_query cannot be empty"):
            QueryIntent(
                intent_type="exploratory",
                original_query="valid query",
                reformulated_query="",
                extracted_filters=SearchFilters(),
                confidence=0.5,
                reasoning="Test"
            )

    def test_intent_validation_empty_reasoning(self):
        """Test that empty reasoning raises ValueError."""
        with pytest.raises(ValueError, match="reasoning cannot be empty"):
            QueryIntent(
                intent_type="exploratory",
                original_query="valid query",
                reformulated_query="valid reformulation",
                extracted_filters=SearchFilters(),
                confidence=0.5,
                reasoning=""
            )

    def test_intent_validation_confidence_too_low(self):
        """Test that confidence < 0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            QueryIntent(
                intent_type="exploratory",
                original_query="valid query",
                reformulated_query="valid reformulation",
                extracted_filters=SearchFilters(),
                confidence=-0.1,
                reasoning="Test reasoning"
            )

    def test_intent_validation_confidence_too_high(self):
        """Test that confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            QueryIntent(
                intent_type="exploratory",
                original_query="valid query",
                reformulated_query="valid reformulation",
                extracted_filters=SearchFilters(),
                confidence=1.5,
                reasoning="Test reasoning"
            )

    def test_intent_confidence_boundaries(self):
        """Test that confidence at boundaries (0.0 and 1.0) is valid."""
        intent_zero = QueryIntent(
            intent_type="exploratory",
            original_query="query",
            reformulated_query="query",
            extracted_filters=SearchFilters(),
            confidence=0.0,
            reasoning="Fallback - no confidence"
        )
        assert intent_zero.confidence == 0.0

        intent_one = QueryIntent(
            intent_type="recommendation",
            original_query="query",
            reformulated_query="query",
            extracted_filters=SearchFilters(),
            confidence=1.0,
            reasoning="Maximum confidence"
        )
        assert intent_one.confidence == 1.0

    def test_intent_immutability(self):
        """Test that QueryIntent is immutable (frozen dataclass)."""
        intent = QueryIntent(
            intent_type="exploratory",
            original_query="test query",
            reformulated_query="test query",
            extracted_filters=SearchFilters(),
            confidence=0.5,
            reasoning="Test"
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            intent.intent_type = "factual"

    def test_intent_invalid_type_not_enforced_by_dataclass(self):
        """Test that invalid intent_type is accepted at runtime (Literal not enforced)."""
        # Note: Literal types are not enforced at runtime in Python
        # This test documents the behavior - type checking happens at static analysis
        intent = QueryIntent(
            intent_type="invalid_type",  # type: ignore
            original_query="query",
            reformulated_query="query",
            extracted_filters=SearchFilters(),
            confidence=0.5,
            reasoning="Test"
        )
        assert intent.intent_type == "invalid_type"

    def test_query_variations_defaults_to_empty_list(self):
        """Test that query_variations defaults to empty list."""
        intent = QueryIntent(
            intent_type="exploratory",
            original_query="test query",
            reformulated_query="test query",
            extracted_filters=SearchFilters(),
            confidence=0.5,
            reasoning="Test"
        )
        assert intent.query_variations == []

    def test_query_variations_can_be_set(self):
        """Test that query_variations accepts a list of strings."""
        variations = ["variation 1", "variation 2"]
        intent = QueryIntent(
            intent_type="recommendation",
            original_query="books like 1984",
            reformulated_query="1984 dystopian",
            extracted_filters=SearchFilters(),
            confidence=0.9,
            reasoning="Test",
            query_variations=variations,
        )
        assert intent.query_variations == ["variation 1", "variation 2"]

    def test_query_variations_does_not_share_default(self):
        """Test that each instance gets its own list (no mutable default trap)."""
        intent_a = QueryIntent(
            intent_type="exploratory",
            original_query="query a",
            reformulated_query="query a",
            extracted_filters=SearchFilters(),
            confidence=0.5,
            reasoning="A"
        )
        intent_b = QueryIntent(
            intent_type="exploratory",
            original_query="query b",
            reformulated_query="query b",
            extracted_filters=SearchFilters(),
            confidence=0.5,
            reasoning="B"
        )
        # They should be equal but NOT the same object
        assert intent_a.query_variations is not intent_b.query_variations


# =============================================================================
# Tests: RetrievalStrategy and Mappings
# =============================================================================


class TestRetrievalStrategy:
    """Tests for RetrievalStrategy enum and associated mappings."""

    def test_enum_has_three_values(self):
        """RetrievalStrategy should have exactly 3 members."""
        assert len(RetrievalStrategy) == 3

    def test_lexical_heavy_value(self):
        """LEXICAL_HEAVY should have string value 'lexical_heavy'."""
        assert RetrievalStrategy.LEXICAL_HEAVY.value == "lexical_heavy"

    def test_vector_heavy_value(self):
        """VECTOR_HEAVY should have string value 'vector_heavy'."""
        assert RetrievalStrategy.VECTOR_HEAVY.value == "vector_heavy"

    def test_balanced_value(self):
        """BALANCED should have string value 'balanced'."""
        assert RetrievalStrategy.BALANCED.value == "balanced"

    def test_strategy_pool_sizes_covers_all_strategies(self):
        """Every strategy must have a pool size entry."""
        for strategy in RetrievalStrategy:
            assert strategy in STRATEGY_POOL_SIZES, (
                f"Missing pool size mapping for {strategy}"
            )

    def test_pool_sizes_have_both_factors(self):
        """Each pool size entry must define bm25_factor and vector_factor."""
        for strategy, sizes in STRATEGY_POOL_SIZES.items():
            assert "bm25_factor" in sizes, f"Missing bm25_factor for {strategy}"
            assert "vector_factor" in sizes, f"Missing vector_factor for {strategy}"

    def test_lexical_heavy_favors_bm25(self):
        """LEXICAL_HEAVY should have bm25_factor > vector_factor."""
        sizes = STRATEGY_POOL_SIZES[RetrievalStrategy.LEXICAL_HEAVY]
        assert sizes["bm25_factor"] > sizes["vector_factor"]

    def test_vector_heavy_favors_vector(self):
        """VECTOR_HEAVY should have vector_factor > bm25_factor."""
        sizes = STRATEGY_POOL_SIZES[RetrievalStrategy.VECTOR_HEAVY]
        assert sizes["vector_factor"] > sizes["bm25_factor"]

    def test_balanced_has_equal_factors(self):
        """BALANCED should have equal bm25_factor and vector_factor."""
        sizes = STRATEGY_POOL_SIZES[RetrievalStrategy.BALANCED]
        assert sizes["bm25_factor"] == sizes["vector_factor"]

    def test_all_factors_are_positive(self):
        """All pool size factors must be > 0."""
        for strategy, sizes in STRATEGY_POOL_SIZES.items():
            assert sizes["bm25_factor"] > 0, f"Non-positive bm25_factor for {strategy}"
            assert sizes["vector_factor"] > 0, f"Non-positive vector_factor for {strategy}"


class TestIntentToStrategy:
    """Tests for the INTENT_TO_STRATEGY mapping."""

    def test_recommendation_maps_to_vector_heavy(self):
        """Recommendation intent should use VECTOR_HEAVY strategy."""
        assert INTENT_TO_STRATEGY["recommendation"] == RetrievalStrategy.VECTOR_HEAVY

    def test_factual_maps_to_lexical_heavy(self):
        """Factual intent should use LEXICAL_HEAVY strategy."""
        assert INTENT_TO_STRATEGY["factual"] == RetrievalStrategy.LEXICAL_HEAVY

    def test_exploratory_maps_to_balanced(self):
        """Exploratory intent should use BALANCED strategy."""
        assert INTENT_TO_STRATEGY["exploratory"] == RetrievalStrategy.BALANCED

    def test_all_intent_types_are_mapped(self):
        """All three intent types must be present in the mapping."""
        expected_intents = {"recommendation", "factual", "exploratory"}
        assert set(INTENT_TO_STRATEGY.keys()) == expected_intents
