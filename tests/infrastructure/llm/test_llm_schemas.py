"""
Tests for the JSON schemas sent to LLM providers (schemas*.py and the
agent's recommendation list): only standard keywords, soft length limits on
free text, hard numeric ranges.
"""

import json

import pytest
from pydantic import ValidationError

from app.infrastructure.llm.agents.search_agent import RecommendationListLLM
from app.infrastructure.llm.schemas import CitationLLM, GroundedExplanationLLM
from app.infrastructure.llm.schemas_judge import ExplanationJudgmentLLM, JudgmentDimension, RelevanceGradeLLM
from app.infrastructure.llm.schemas_query import ExtractedFiltersLLM, QueryIntentLLM, ReformulatedQueryLLM

LLM_SCHEMAS = [
    QueryIntentLLM, ExtractedFiltersLLM, ReformulatedQueryLLM, GroundedExplanationLLM,
    ExplanationJudgmentLLM, RelevanceGradeLLM, RecommendationListLLM,
]


def citation(**overrides) -> dict:
    fields = dict(book_id="b1", chunk_id="title", snippet="Dune", relevance_score=0.9)
    fields.update(overrides)
    return fields


@pytest.mark.parametrize("schema", LLM_SCHEMAS, ids=lambda s: s.__name__)
def test_schema_has_no_example_payload(schema):
    # Strict structured-output APIs can reject "example" as an unknown keyword.
    assert '"example"' not in json.dumps(schema.model_json_schema())


class TestSoftLengthLimits:
    def test_schema_still_states_the_limits(self):
        properties = QueryIntentLLM.model_json_schema()["properties"]
        assert (properties["reasoning"]["minLength"], properties["reasoning"]["maxLength"]) == (10, 200)
        assert ReformulatedQueryLLM.model_json_schema()["properties"]["keywords"]["maxItems"] == 10

    def test_long_free_text_is_cut_not_rejected(self):
        intent = QueryIntentLLM(intent_type="exploratory", confidence=0.5, reasoning="r" * 450)
        assert intent.reasoning == "r" * 200
        explanation = GroundedExplanationLLM(summary="s" * 300, reasoning="x" * 900,
                                             citations=[citation()], confidence=0.5)
        assert (len(explanation.summary), len(explanation.reasoning)) == (200, 500)
        assert len(JudgmentDimension(score=3, reasoning="j" * 250).reasoning) == 200
        assert len(ReformulatedQueryLLM(optimized_query="q" * 400, keywords=[]).optimized_query) == 300

    def test_cut_snippet_is_still_a_prefix_of_the_original(self):
        text = "A Victorian mystery novel about a cursed diamond. " * 6
        assert text.startswith(CitationLLM(**citation(snippet=text)).snippet)

    def test_extra_keywords_are_dropped(self):
        query = ReformulatedQueryLLM(optimized_query="q", keywords=[f"k{i}" for i in range(14)])
        assert query.keywords == [f"k{i}" for i in range(10)]

    def test_minimum_lengths_stay_hard(self):
        with pytest.raises(ValidationError):
            QueryIntentLLM(intent_type="factual", confidence=0.5, reasoning="short")
        with pytest.raises(ValidationError):
            GroundedExplanationLLM(summary="tiny", reasoning="x" * 30, citations=[citation()], confidence=0.5)
        with pytest.raises(ValidationError):
            GroundedExplanationLLM(summary="s" * 20, reasoning="x" * 30, citations=[], confidence=0.5)


class TestHardRanges:
    @pytest.mark.parametrize("confidence", [-0.1, 1.5])
    def test_confidence_out_of_range(self, confidence):
        with pytest.raises(ValidationError):
            QueryIntentLLM(intent_type="factual", confidence=confidence, reasoning="A factual question.")

    def test_relevance_score_out_of_range(self):
        with pytest.raises(ValidationError):
            CitationLLM(**citation(relevance_score=1.2))

    @pytest.mark.parametrize("score", [0, 6])
    def test_judge_score_out_of_range(self, score):
        with pytest.raises(ValidationError):
            JudgmentDimension(score=score, reasoning="ok")

    def test_unknown_intent_type(self):
        with pytest.raises(ValidationError):
            QueryIntentLLM(intent_type="chat", confidence=0.5, reasoning="Not a known intent.")
