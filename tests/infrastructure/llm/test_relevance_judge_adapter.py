"""
Tests for the LLM relevance judge adapter, its prompt, schema and settings
(app/infrastructure/llm/relevance_judge.py, prompts_judge.py,
schemas_judge.py, JudgeSettings in config/settings.py).

No LLM is called: the chat model is a fake, and init_chat_model is patched
where the model factory is tested.
"""

import inspect
import re
from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

import app.infrastructure.llm.langchain_llm_client as llm_client_module
from app.domain.entities import Book
from app.evaluation.types import RelevanceJudgment
from app.infrastructure.config.settings import JUDGE_MAX_TOKENS, JudgeSettings
from app.infrastructure.llm.prompts_judge import (
    MISSING_DESCRIPTION,
    RELEVANCE_JUDGE_PROMPT_VERSION,
    RELEVANCE_SCALE,
    build_relevance_judge_prompt,
    get_judge_prompt_metadata,
)
from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge, build_book_prompt
from app.infrastructure.llm.schemas_judge import RelevanceGradeLLM


def make_book(**overrides) -> Book:
    fields = dict(
        id=uuid4(),
        title="Cocina rumana tradicional",
        authors=["Ana Popescu", "Ion Ionescu"],
        description="Recetas de sarmale, mici y papanasi.",
        language="es",
        categories=["Cooking"],
        published_date=datetime(2019, 5, 1),
        source="google_books",
        source_id="abc123",
    )
    fields.update(overrides)
    return Book(**fields)


class FakeChatModel:
    """Records the messages and returns a fixed structured answer."""

    def __init__(self, answer: RelevanceGradeLLM):
        self.answer = answer
        self.schema = None
        self.messages = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        return self.answer


# =============================================================================
# Prompt
# =============================================================================


class TestRelevanceJudgePrompt:
    def test_contains_the_query_and_every_book_field(self):
        prompt = build_book_prompt("recetas cocina rumana", make_book())
        for text in ('Query: "recetas cocina rumana"', "Title: Cocina rumana tradicional",
                     "Authors: Ana Popescu, Ion Ionescu", "Categories: Cooking", "Year: 2019",
                     "Language: Spanish", "Description: Recetas de sarmale, mici y papanasi."):
            assert text in prompt

    @pytest.mark.parametrize("description", [None, "", "   "])
    def test_missing_description_is_said_explicitly(self, description):
        prompt = build_book_prompt("historia", make_book(description=description))
        assert f"Description: {MISSING_DESCRIPTION}" in prompt
        assert "If the description is missing" in prompt

    def test_other_missing_fields_are_named(self):
        prompt = build_book_prompt("python", make_book(categories=[], published_date=None, language=None))
        assert "Categories: none" in prompt
        assert "Year: unknown" in prompt
        assert "Language: unknown" in prompt

    def test_english_query_and_book(self):
        prompt = build_book_prompt("machine learning python", make_book(title="ML", language="en"))
        assert 'Query: "machine learning python"' in prompt and "Language: English" in prompt
        assert "in Spanish or in English" in prompt

    def test_scale_is_the_one_of_the_existing_judgments(self):
        # RelevanceJudgment documents the scale as "- 3: Perfectly relevant".
        documented = {int(grade): label.strip()
                      for grade, label in re.findall(r"- (\d): (.+)", RelevanceJudgment.__doc__)}
        assert RELEVANCE_SCALE == documented
        prompt = build_book_prompt("q", make_book())
        for grade, label in RELEVANCE_SCALE.items():
            assert f"{grade} = {label}" in prompt

    def test_the_prompt_cannot_receive_an_existing_grade(self):
        # The prompt is a function of the query and the book fields only.
        parameters = set(inspect.signature(build_relevance_judge_prompt).parameters)
        assert parameters == {"query_text", "title", "authors", "categories", "year", "language", "description"}
        prompt = build_book_prompt("recetas cocina rumana", make_book()).lower()
        for word in ("original", "existing", "previous", "assessor", "annotat", "rank", "score"):
            assert word not in prompt

    def test_no_role_play(self):
        assert not build_book_prompt("q", make_book()).lower().startswith("you are")

    def test_version_is_recorded(self):
        assert RELEVANCE_JUDGE_PROMPT_VERSION == "v1.0"
        assert get_judge_prompt_metadata()["relevance_judge"] == RELEVANCE_JUDGE_PROMPT_VERSION


# =============================================================================
# Schema and adapter
# =============================================================================


class TestRelevanceGradeSchema:
    def test_grade_must_be_0_to_3(self):
        assert RelevanceGradeLLM(rationale="Fits.", grade=3).grade == 3
        for grade in (-1, 4):
            with pytest.raises(ValidationError):
                RelevanceGradeLLM(rationale="x", grade=grade)

    def test_rationale_is_generated_before_the_grade(self):
        assert list(RelevanceGradeLLM.model_fields) == ["rationale", "grade"]


class TestLangChainRelevanceJudge:
    def test_judge_sends_the_prompt_and_returns_grade_and_rationale(self):
        model = FakeChatModel(RelevanceGradeLLM(rationale="  A Romanian\n cookbook.  ", grade=3))
        judge = LangChainRelevanceJudge(model, "fake:1b")
        book = make_book()

        assert judge.judge("recetas cocina rumana", book) == (3, "A Romanian cookbook.")
        assert model.schema is RelevanceGradeLLM
        assert model.messages == [[{"role": "user", "content": build_book_prompt("recetas cocina rumana", book)}]]
        assert (judge.model_name, judge.prompt_version) == ("fake:1b", RELEVANCE_JUDGE_PROMPT_VERSION)

    def test_from_settings_builds_a_deterministic_capped_model(self):
        settings = JudgeSettings(provider="ollama", model="qwen-judge", base_url="http://127.0.0.1:11434",
                                 num_ctx=8192, timeout_s=60.0)
        with patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            judge = LangChainRelevanceJudge.from_settings(settings)

        kwargs = init_model.call_args.kwargs
        assert kwargs["model"] == "qwen-judge" and kwargs["model_provider"] == "ollama"
        assert kwargs["temperature"] == 0.0
        assert kwargs["reasoning"] is False
        assert kwargs["num_predict"] == JUDGE_MAX_TOKENS
        assert kwargs["base_url"] == "http://127.0.0.1:11434"
        assert kwargs["client_kwargs"] == {"timeout": 60.0}
        assert judge.model_name == "qwen-judge"

    def test_hosted_provider_gets_max_tokens(self):
        settings = JudgeSettings(provider="openai", model="gpt-x", base_url=None, num_ctx=None)
        with patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            LangChainRelevanceJudge.from_settings(settings)
        kwargs = init_model.call_args.kwargs
        assert kwargs["max_tokens"] == JUDGE_MAX_TOKENS and "num_predict" not in kwargs
        assert "reasoning" not in kwargs


# =============================================================================
# Settings
# =============================================================================


class TestJudgeSettings:
    def test_defaults_fall_back_to_the_system_llm(self):
        settings = JudgeSettings.from_env({})
        assert (settings.provider, settings.model) == ("ollama", "qwen3.8:27b-mlx")
        assert settings.base_url == "http://127.0.0.1:11434" and settings.num_ctx == 8192
        assert settings.temperature == 0.0 and settings.reasoning is False
        assert settings.same_as_system_llm

    def test_each_value_falls_back_to_its_llm_variable(self):
        settings = JudgeSettings.from_env({"LLM_MODEL": "m1", "LLM_BASE_URL": "http://h:1", "LLM_TIMEOUT_S": "30"})
        assert (settings.model, settings.base_url, settings.timeout_s) == ("m1", "http://h:1", 30.0)

    def test_judge_variables_override(self):
        settings = JudgeSettings.from_env({"JUDGE_MODEL": "other:7b", "JUDGE_BASE_URL": "http://judge:2"})
        assert (settings.provider, settings.model, settings.base_url) == ("ollama", "other:7b", "http://judge:2")
        assert not settings.same_as_system_llm
        assert settings.system_llm == "ollama/qwen3.8:27b-mlx"

    def test_another_provider_uses_its_own_defaults(self):
        settings = JudgeSettings.from_env({"JUDGE_PROVIDER": "openai"})
        assert (settings.provider, settings.model, settings.base_url, settings.num_ctx) == (
            "openai", "gpt-6-luna", None, None,
        )

    def test_needs_an_llm(self):
        with pytest.raises(ValueError, match="needs an LLM"):
            JudgeSettings.from_env({"LLM_PROVIDER": "none"})
        with pytest.raises(ValueError):
            JudgeSettings.from_env({"JUDGE_PROVIDER": "other"})
