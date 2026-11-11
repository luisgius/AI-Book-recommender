"""
Structured output with an explicit method per chat-model client.

LangChain's with_structured_output() can steer a model in several ways, and
the default differs per client class and library version. This module names
the method used for each client the project builds, so a library upgrade
cannot silently change how the model is asked for JSON:

- ChatOllama ("ollama"): "json_schema". The schema goes in Ollama's `format`
  field and constrains decoding.
- ChatOpenAI ("openai" and "openrouter"): "json_schema". The schema goes in
  `response_format` in strict mode (the OpenAI SDK marks every field required
  and forbids extra keys). Through OpenRouter this was checked on Qwen3.8
  Flash, MiniMax M3.1 Flash, DeepSeek V4.1 Flash and Gemini 3.8 Flash.
- ChatAnthropic ("anthropic"): "function_calling", the long-standing method
  of that client (forced tool use).

Strict modes do not enforce every JSON Schema keyword (string lengths are
accepted but not enforced by some providers), so the Pydantic schema still
validates every answer.

Any other object (test doubles) is called without a method, so it keeps its
own default.
"""

from typing import Any, Optional

# Structured-output method per LangChain chat-model class.
STRUCTURED_OUTPUT_METHODS = {
    "ChatOllama": "json_schema",
    "ChatOpenAI": "json_schema",
    "ChatAnthropic": "function_calling",
}


def structured_output_method(llm: Any) -> Optional[str]:
    """The method used for this chat model, or None for an unknown class."""
    for cls in type(llm).__mro__:
        if cls.__name__ in STRUCTURED_OUTPUT_METHODS:
            return STRUCTURED_OUTPUT_METHODS[cls.__name__]
    return None


def with_structured_output(llm: Any, schema: Any, *, include_raw: bool = False) -> Any:
    """
    llm.with_structured_output(schema) with the method of llm's client class.

    Args:
        llm: LangChain chat model
        schema: Pydantic model the answer must follow
        include_raw: Also return the model's message (LangChain's
            include_raw: a dict with "raw", "parsed" and "parsing_error"),
            e.g. to read the token usage of the call

    Returns:
        A runnable that returns an instance of schema (or the include_raw dict)
    """
    method = structured_output_method(llm)
    extra = {"include_raw": True} if include_raw else {}
    if method is None:
        return llm.with_structured_output(schema, **extra)
    return llm.with_structured_output(schema, method=method, **extra)
