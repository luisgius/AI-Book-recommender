"""
Tests for the LLM adapter wiring: settings, model construction, retry policy,
validation of LLM-extracted filters and query expansion without duplicates.

No LLM is called: init_chat_model and the chain/graph builders are patched.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

import app.infrastructure.llm.langchain_llm_client as llm_client_module
from app.domain.entities import Book
from app.domain.value_objects import SearchFilters
from app.infrastructure.config.settings import DEFAULT_OLLAMA_URL, LLMSettings
from app.infrastructure.llm.graphs.query_understanding import (
    append_new_terms,
    handle_exploratory_node,
)


def build_client(**kwargs):
    """Build a LangChainLLMClient with the model factory and builders patched."""
    with patch.object(llm_client_module, "load_dotenv"), \
         patch.object(llm_client_module, "init_chat_model") as init_model, \
         patch.object(llm_client_module, "create_grounded_explanation_chain"), \
         patch.object(llm_client_module, "create_query_understanding_graph"), \
         patch.object(llm_client_module, "create_llm_judge_chain"):
        init_model.return_value = Mock()
        client = llm_client_module.LangChainLLMClient(**kwargs)
    return client, init_model


# =============================================================================
# Settings
# =============================================================================


class TestLLMSettings:
    def test_defaults_target_local_ollama(self):
        settings = LLMSettings.from_env({})
        assert settings.provider == "ollama"
        assert settings.base_url == DEFAULT_OLLAMA_URL
        assert settings.reasoning is False
        assert settings.num_ctx == 8192
        assert settings.max_explanations == 3
        assert (settings.max_tokens, settings.timeout_s) == (1024, 180.0)
        assert settings.enabled

    def test_hosted_provider_has_no_ollama_options(self):
        settings = LLMSettings.from_env({"LLM_PROVIDER": "openai"})
        assert settings.model == "gpt-4o-mini"
        assert settings.base_url is None
        assert settings.num_ctx is None
        assert settings.reasoning is None

    def test_none_disables_the_llm(self):
        assert not LLMSettings.from_env({"LLM_PROVIDER": "none"}).enabled

    def test_explicit_values_are_read(self):
        settings = LLMSettings.from_env({
            "LLM_MODEL": "other:7b", "LLM_TEMPERATURE": "0.3", "LLM_NUM_CTX": "4096",
            "LLM_REASONING": "default", "LLM_MAX_RETRIES": "0", "LLM_MAX_EXPLANATIONS": "5",
        })
        assert (settings.model, settings.temperature, settings.num_ctx) == ("other:7b", 0.3, 4096)
        assert settings.reasoning is None
        assert (settings.max_retries, settings.max_explanations) == (0, 5)

    def test_category_filter_switch(self):
        assert LLMSettings.from_env({}).apply_extracted_category is False
        assert LLMSettings.from_env({"LLM_APPLY_CATEGORY_FILTER": "true"}).apply_extracted_category is True

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError):
            LLMSettings.from_env({"LLM_PROVIDER": "mystery"})


# =============================================================================
# Model construction
# =============================================================================


def test_ollama_options_reach_the_model_factory():
    _, init_model = build_client(
        provider="ollama", model_name="qwen3.8:27b-mlx", base_url="http://127.0.0.1:11434",
        num_ctx=8192, reasoning=False,
    )
    kwargs = init_model.call_args.kwargs
    assert kwargs["model"] == "qwen3.8:27b-mlx"
    assert "num_predict" not in kwargs  # no cap unless configured
    assert kwargs["model_provider"] == "ollama"
    assert kwargs["base_url"] == "http://127.0.0.1:11434"
    assert kwargs["num_ctx"] == 8192
    assert kwargs["reasoning"] is False


def test_output_cap_and_timeout_use_each_provider_parameter_names():
    _, ollama = build_client(provider="ollama", model_name="m", max_tokens=1024, timeout_s=60)
    assert ollama.call_args.kwargs["num_predict"] == 1024
    assert ollama.call_args.kwargs["client_kwargs"] == {"timeout": 60}
    _, openai = build_client(provider="openai", model_name="m", max_tokens=1024, timeout_s=60)
    assert openai.call_args.kwargs["max_tokens"] == 1024
    assert openai.call_args.kwargs["timeout"] == 60


def test_ollama_only_options_are_not_sent_to_other_providers():
    _, init_model = build_client(provider="openai", model_name="gpt-4o-mini", num_ctx=8192, reasoning=False)
    kwargs = init_model.call_args.kwargs
    assert "num_ctx" not in kwargs and "reasoning" not in kwargs


# =============================================================================
# Retry policy
# =============================================================================


class TestRetryPolicy:
    @pytest.mark.parametrize("error", [
        ConnectionError("refused"), ValueError("bad output"), TimeoutError("slow"),
    ])
    def test_non_retryable_errors_fail_fast(self, error):
        client, _ = build_client(max_retries=3)
        operation = Mock(side_effect=error)
        with patch.object(llm_client_module.time, "sleep") as sleep:
            with pytest.raises(type(error)):
                client._invoke_with_retry(operation, "op")
        assert operation.call_count == 1
        sleep.assert_not_called()

    def test_transient_errors_are_retried(self):
        client, _ = build_client(max_retries=2)
        operation = Mock(side_effect=[RuntimeError("503"), RuntimeError("503"), "ok"])
        with patch.object(llm_client_module.time, "sleep"):
            assert client._invoke_with_retry(operation, "op") == "ok"
        assert operation.call_count == 3

    def test_unreachable_llm_degrades_to_fallback_explanation(self):
        client, _ = build_client(max_retries=3)
        client.chain = Mock(invoke=Mock(side_effect=ConnectionError("Ollama is not running")))
        book = Book(id=uuid4(), title="T", authors=["A"])
        with patch.object(llm_client_module.time, "sleep") as sleep:
            explanation = client.generate_grounded_explanation("query", book)
        assert explanation.citations == []
        assert client.chain.invoke.call_count == 1
        sleep.assert_not_called()


# =============================================================================
# Validation of LLM-extracted filters
# =============================================================================


def extracted(**values):
    base = {"language": None, "category": None, "min_year": None, "max_year": None}
    base.update(values)
    return SimpleNamespace(**base)


class TestSanitizeFilters:
    def setup_method(self):
        self.client, _ = build_client(
            known_categories=["Spanish fiction", "Computers", "Cooking"], apply_extracted_category=True
        )

    def test_language_is_normalised_or_dropped(self):
        query = "novelas en español"
        assert self.client._sanitize_filters(extracted(language="ES"), query).language == "es"
        assert self.client._sanitize_filters(extracted(language="spa"), query).language is None
        assert self.client._sanitize_filters(extracted(language="Spanish"), query).language is None

    @pytest.mark.parametrize("query", [
        "libros de python en español", "novelas españolas", "Spanish novels", "libros en castellano",
    ])
    def test_language_kept_when_the_query_names_it(self, query):
        assert self.client._sanitize_filters(extracted(language="es"), query).language == "es"

    @pytest.mark.parametrize("query", [
        "libros para aprender machine learning con python", "recetas de cocina rumana",
    ])
    def test_language_dropped_when_only_inferred_from_the_query_wording(self, query):
        assert self.client._sanitize_filters(extracted(language="es"), query).language is None

    def test_category_must_be_able_to_match_the_catalog(self):
        assert self.client._sanitize_filters(extracted(category="fiction")).category == "fiction"
        assert self.client._sanitize_filters(extracted(category="History")).category is None

    def test_category_is_a_soft_signal_by_default(self):
        client, _ = build_client(known_categories=["Spanish fiction"])
        assert client._sanitize_filters(extracted(category="fiction")).category is None

    def test_category_is_kept_when_catalog_is_unknown(self):
        client, _ = build_client(apply_extracted_category=True)
        assert client._sanitize_filters(extracted(category="History")).category == "History"

    def test_years_must_be_plausible_and_ordered(self):
        assert self.client._sanitize_filters(extracted(min_year=1950)).min_year == 1950
        assert self.client._sanitize_filters(extracted(min_year=50)).min_year is None
        inverted = self.client._sanitize_filters(extracted(min_year=2000, max_year=1990))
        assert (inverted.min_year, inverted.max_year) == (None, None)

    def test_result_is_a_valid_domain_value_object(self):
        filters = self.client._sanitize_filters(extracted(language="spa", category="History", min_year=3000))
        assert filters == SearchFilters()


# =============================================================================
# Query expansion without duplicated terms
# =============================================================================


class TestAppendNewTerms:
    def test_skips_words_already_in_the_query(self):
        assert append_new_terms("novelas guerra civil", ["guerra civil", "España", "novelas"]) == \
            "novelas guerra civil España"

    def test_returns_base_query_when_nothing_is_new(self):
        assert append_new_terms("python machine learning", ["Python", "learning"]) == "python machine learning"

    def test_exploratory_node_does_not_repeat_the_query(self):
        reformulated = SimpleNamespace(
            optimized_query="novelas españolas guerra civil",
            keywords=["novelas españolas guerra civil", "literatura histórica"],
        )
        state = {"original_query": "q", "reformulated": reformulated}
        final = handle_exploratory_node(state)["final_query"]
        words = final.lower().split()
        assert len(words) == len(set(words))
        assert final == "novelas españolas guerra civil literatura histórica"


# =============================================================================
# Explanation prompt
# =============================================================================


def test_explanation_prompt_asks_for_the_query_language_and_verbatim_snippets():
    from app.infrastructure.llm.prompts import (
        GROUNDED_EXPLANATION_PROMPT_VERSION,
        build_grounded_explanation_prompt,
    )

    prompt = build_grounded_explanation_prompt(
        query_text="novelas sobre la guerra civil", book_id="b1", title="T", authors="A",
        categories="Fiction", description="D",
    )

    assert GROUNDED_EXPLANATION_PROMPT_VERSION == "v1.1"
    assert "SAME LANGUAGE as the User Query" in prompt
    assert "never translated" in prompt
    assert 'User Query: "novelas sobre la guerra civil"' in prompt
