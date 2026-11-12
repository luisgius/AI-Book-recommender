"""
The composition root in a fresh clone: no .env, no OPENROUTER_API_KEY, no LLM_PROVIDER.

The LLM must be off (not the local Ollama default of earlier versions), so the
API serves plain search and says why understanding, explanations and the
agent are unavailable. No model, index or network is touched.
"""

import pytest

from app.api.v1 import dependencies as deps


@pytest.fixture
def fresh_environment(monkeypatch):
    """No LLM variable in the environment and no .env file read."""
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(deps, "load_dotenv", lambda *args, **kwargs: False)
    deps.reset_dependencies()
    yield
    deps.reset_dependencies()


def test_without_key_or_provider_the_llm_is_off(fresh_environment):
    settings = deps.get_llm_settings()

    assert settings.provider == "none" and not settings.enabled
    assert deps.get_llm_client() is None
    assert deps.get_search_agent() is None


def test_an_openrouter_key_alone_turns_the_llm_on(fresh_environment, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")

    settings = deps.get_llm_settings()

    assert (settings.provider, settings.model) == ("openrouter", "deepseek/deepseek-v4.1-flash")
