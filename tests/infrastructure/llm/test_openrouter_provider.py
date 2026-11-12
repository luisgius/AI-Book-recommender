"""
Tests for the OpenRouter provider and the explicit structured-output method:
settings parsing and fallbacks, the missing-key error, the options sent to
the OpenAI-compatible client, and the method chosen per chat-model class.

No network: chat models are built but never invoked, or init_chat_model is
patched.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import app.infrastructure.llm.langchain_llm_client as llm_client_module
from app.infrastructure.config.settings import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_OPENROUTER_URL,
    JudgeSettings,
    LLMSettings,
    openrouter_api_key,
    parse_reasoning,
)
from app.infrastructure.llm.langchain_llm_client import build_chat_model, openrouter_reasoning
from app.infrastructure.llm.schemas_query import QueryIntentLLM
from app.infrastructure.llm.structured_output import structured_output_method, with_structured_output
from app.infrastructure.llm.usage import LLMUsageRecorder


@pytest.fixture
def openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def no_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


# =============================================================================
# Settings
# =============================================================================


class TestOpenRouterSettings:
    def test_defaults(self):
        settings = LLMSettings.from_env({"LLM_PROVIDER": "openrouter"})
        assert (settings.provider, settings.model) == ("openrouter", "deepseek/deepseek-v4.1-flash")
        assert settings.base_url == DEFAULT_OPENROUTER_URL
        assert settings.reasoning is False  # thinking off, as for Ollama
        assert settings.num_ctx is None
        assert settings.enabled

    def test_explicit_values(self):
        settings = LLMSettings.from_env({
            "LLM_PROVIDER": "OpenRouter", "LLM_MODEL": "deepseek/deepseek-v4.1-flash",
            "LLM_BASE_URL": "http://proxy:1/v1", "LLM_REASONING": "low",
        })
        assert (settings.provider, settings.model, settings.base_url) == (
            "openrouter", "deepseek/deepseek-v4.1-flash", "http://proxy:1/v1",
        )
        assert settings.reasoning == "low"

    def test_ollama_defaults_are_unchanged(self):
        settings = LLMSettings.from_env({"LLM_PROVIDER": "ollama"})
        assert (settings.provider, settings.model, settings.base_url) == (
            "ollama", "qwen3.8:27b-mlx", DEFAULT_OLLAMA_URL,
        )
        assert settings.reasoning is False

    def test_openai_default_model_is_not_the_retired_one(self):
        assert LLMSettings.from_env({"LLM_PROVIDER": "openai"}).model == "gpt-6-luna"

    @pytest.mark.parametrize("raw, expected", [
        ("true", True), ("False", False), ("default", None), ("", None), ("low", "low"), (" MAX ", "max"),
    ])
    def test_reasoning_values(self, raw, expected):
        assert parse_reasoning(raw) == expected

    def test_unknown_effort_given_directly_is_rejected(self):
        with pytest.raises(ValueError, match="LLM_REASONING"):
            LLMSettings(provider="openrouter", reasoning="extreme")


class TestOpenRouterJudgeSettings:
    def test_default_judge_is_another_family_with_low_effort(self):
        settings = JudgeSettings.from_env({"LLM_PROVIDER": "openrouter"})
        assert (settings.provider, settings.model) == ("openrouter", "z-ai/glm-5.3-flash")
        assert settings.reasoning == "low"  # its reasoning cannot be switched off
        assert settings.base_url == DEFAULT_OPENROUTER_URL and settings.num_ctx is None
        assert settings.system_llm == "openrouter/deepseek/deepseek-v4.1-flash"
        assert not settings.same_as_system_llm

    def test_openrouter_judge_for_a_local_system_llm(self):
        settings = JudgeSettings.from_env({"JUDGE_PROVIDER": "openrouter", "LLM_PROVIDER": "ollama"})
        assert (settings.model, settings.base_url, settings.num_ctx) == (
            "z-ai/glm-5.3-flash", DEFAULT_OPENROUTER_URL, None,
        )
        assert settings.system_llm == "ollama/qwen3.8:27b-mlx"

    def test_explicit_judge_model_keeps_thinking_off(self):
        settings = JudgeSettings.from_env({"JUDGE_PROVIDER": "openrouter", "JUDGE_MODEL": "qwen/qwen3.8-flash"})
        assert (settings.model, settings.reasoning) == ("qwen/qwen3.8-flash", False)

    def test_judge_reasoning_variable_wins(self):
        settings = JudgeSettings.from_env({"JUDGE_PROVIDER": "openrouter", "JUDGE_REASONING": "default"})
        assert settings.reasoning is None
        settings = JudgeSettings.from_env({
            "JUDGE_PROVIDER": "openrouter", "JUDGE_MODEL": "google/gemini-3.8-flash", "JUDGE_REASONING": "low",
        })
        assert settings.reasoning == "low"

    def test_judge_base_url_falls_back_to_the_system_url_on_the_same_provider(self):
        settings = JudgeSettings.from_env({"LLM_PROVIDER": "openrouter", "LLM_BASE_URL": "http://proxy:1/v1"})
        assert settings.base_url == "http://proxy:1/v1"

    def test_ollama_judge_is_unchanged(self):
        settings = JudgeSettings.from_env({"LLM_PROVIDER": "ollama"})
        assert (settings.model, settings.reasoning) == ("qwen3.8:27b-mlx", False)


# =============================================================================
# API key
# =============================================================================


class TestApiKey:
    def test_key_is_read_from_the_environment(self):
        assert openrouter_api_key({"OPENROUTER_API_KEY": " k-1 "}) == "k-1"

    @pytest.mark.parametrize("env", [{}, {"OPENROUTER_API_KEY": "  "}])
    def test_missing_key_names_the_variable(self, env):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            openrouter_api_key(env)

    def test_building_a_model_without_key_fails_clearly(self, no_openrouter_key):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            build_chat_model("openrouter", "qwen/qwen3.8-flash")

    def test_llm_client_without_key_fails_clearly(self, no_openrouter_key):
        with patch.object(llm_client_module, "load_dotenv"):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                llm_client_module.LangChainLLMClient(provider="openrouter", model_name="qwen/qwen3.8-flash")

    def test_key_is_not_in_the_settings(self, openrouter_key):
        settings = LLMSettings.from_env({"LLM_PROVIDER": "openrouter"})
        assert "test-key" not in repr(settings)


# =============================================================================
# Chat model construction
# =============================================================================


class TestOpenRouterChatModel:
    def test_options_reach_the_openai_compatible_client(self, openrouter_key):
        with patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            build_chat_model("openrouter", "qwen/qwen3.8-flash", reasoning=False, max_tokens=1024, timeout_s=60)
        kwargs = init_model.call_args.kwargs
        assert (kwargs["model"], kwargs["model_provider"]) == ("qwen/qwen3.8-flash", "openai")
        assert kwargs["base_url"] == DEFAULT_OPENROUTER_URL
        assert kwargs["api_key"] == "test-key"
        assert (kwargs["max_tokens"], kwargs["timeout"]) == (1024, 60)
        assert kwargs["extra_body"] == {"provider": {"require_parameters": True, "ignore": ["novita"]},
                                        "reasoning": {"enabled": False}}
        assert "num_ctx" not in kwargs and "num_predict" not in kwargs and "default_headers" not in kwargs

    @pytest.mark.parametrize("reasoning", [True, "low", None])
    def test_providers_are_excluded_only_when_thinking_is_off(self, openrouter_key, reasoning):
        # Novita keeps thinking with reasoning off (V25), so it is excluded
        # then and only then: with thinking on it is a valid provider.
        with patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            build_chat_model("openrouter", "m", reasoning=reasoning)
        assert init_model.call_args.kwargs["extra_body"]["provider"] == {"require_parameters": True}

    def test_custom_base_url(self, openrouter_key):
        with patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            build_chat_model("openrouter", "m", base_url="http://proxy:1/v1")
        assert init_model.call_args.kwargs["base_url"] == "http://proxy:1/v1"

    @pytest.mark.parametrize("reasoning, field", [
        (False, {"enabled": False}), (True, {"enabled": True}), ("low", {"effort": "low"}), (None, None),
    ])
    def test_reasoning_field(self, reasoning, field):
        assert openrouter_reasoning(reasoning) == field

    def test_model_default_reasoning_sends_no_field(self, openrouter_key):
        with patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            build_chat_model("openrouter", "m", reasoning=None)
        assert "reasoning" not in init_model.call_args.kwargs["extra_body"]

    def test_real_client_sends_no_identifying_headers(self, openrouter_key):
        recorder = LLMUsageRecorder()
        model = build_chat_model("openrouter", "qwen/qwen3.8-flash", reasoning=False, usage_recorder=recorder)
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == DEFAULT_OPENROUTER_URL
        assert model.openai_api_key.get_secret_value() == "test-key"
        assert model.default_headers is None
        headers = {name.lower() for name in model.root_client.default_headers}
        assert not headers & {"http-referer", "x-title"}
        assert model.callbacks == [recorder]

    def test_ollama_gets_the_recorder_and_keeps_its_options(self):
        recorder = LLMUsageRecorder()
        with patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            build_chat_model("ollama", "qwen3.8:27b-mlx", base_url=DEFAULT_OLLAMA_URL, num_ctx=8192,
                             reasoning=False, max_tokens=1024, usage_recorder=recorder)
        kwargs = init_model.call_args.kwargs
        assert kwargs["model_provider"] == "ollama" and kwargs["callbacks"] == [recorder]
        assert (kwargs["num_ctx"], kwargs["reasoning"], kwargs["num_predict"]) == (8192, False, 1024)
        assert "api_key" not in kwargs and "extra_body" not in kwargs

    def test_llm_client_attaches_its_recorder(self, openrouter_key):
        with patch.object(llm_client_module, "load_dotenv"), \
             patch.object(llm_client_module, "init_chat_model") as init_model:
            init_model.return_value = Mock()
            client = llm_client_module.LangChainLLMClient(provider="openrouter", model_name="qwen/qwen3.8-flash")
        assert init_model.call_args.kwargs["callbacks"] == [client.usage_recorder]


# =============================================================================
# Structured-output method
# =============================================================================


class TestStructuredOutputMethod:
    def test_method_per_client_class(self):
        assert structured_output_method(ChatOllama(model="m")) == "json_schema"
        assert structured_output_method(ChatOpenAI(model="m", api_key="k")) == "json_schema"
        assert structured_output_method(Mock()) is None

    def test_known_client_gets_the_method_explicitly(self):
        calls = []
        ChatOllamaDouble = type("ChatOllama", (), {
            "with_structured_output": lambda self, schema, **kwargs: calls.append((schema, kwargs)),
        })
        with_structured_output(ChatOllamaDouble(), QueryIntentLLM)
        assert calls == [(QueryIntentLLM, {"method": "json_schema"})]

    def test_unknown_object_keeps_its_default(self):
        llm = Mock()
        with_structured_output(llm, QueryIntentLLM)
        llm.with_structured_output.assert_called_once_with(QueryIntentLLM)

    def test_openai_compatible_client_sends_the_schema_as_response_format(self):
        structured = with_structured_output(ChatOpenAI(model="m", api_key="k"), QueryIntentLLM)
        assert structured.first.kwargs["response_format"] is QueryIntentLLM

    def test_ollama_sends_the_schema_as_format(self):
        structured = with_structured_output(ChatOllama(model="m"), QueryIntentLLM)
        assert structured.first.kwargs["format"] == QueryIntentLLM.model_json_schema()
