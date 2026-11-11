"""
Runtime configuration for the LLM and embedding adapters, read from
environment variables.

Kept as small frozen dataclasses (no framework) so the composition root can
build the adapters from them and tests can construct them directly.

LLM environment variables (all optional):
    LLM_PROVIDER          "ollama" (default), "openrouter", "openai", "anthropic"
                          or "none"
    LLM_MODEL             model id; default depends on the provider
    LLM_BASE_URL          server URL (default http://127.0.0.1:11434 for Ollama,
                          https://openrouter.ai/api/v1 for OpenRouter)
    LLM_TEMPERATURE       sampling temperature (default 0.0, reproducible runs)
    LLM_NUM_CTX           context window for Ollama (default 8192)
    LLM_REASONING         thinking mode: "false" (default for Ollama and
                          OpenRouter: no thinking tokens), "true", "default"
                          (let the model decide) or an effort level ("low",
                          "medium", "high", ...) for models whose thinking
                          cannot be switched off
    LLM_MAX_RETRIES       retries for transient errors (default 1)
    LLM_MAX_EXPLANATIONS  top results explained per search (default 3)
    LLM_APPLY_CATEGORY_FILTER  "true" to apply the category inferred by the
                          LLM as a hard filter (default "false": soft signal)
    LLM_MAX_TOKENS        cap on generated tokens per call (default 1024)
    LLM_TIMEOUT_S         HTTP timeout per call in seconds (default 180)
    OPENROUTER_API_KEY    API key, required by the "openrouter" provider (read
                          only when the chat model is built, never stored here)

Relevance judge environment variables (all optional; evaluation only):
    JUDGE_PROVIDER        provider of the LLM that grades (query, book) pairs;
                          default LLM_PROVIDER
    JUDGE_MODEL           judge model id; default z-ai/glm-5.3-flash on
                          OpenRouter, otherwise LLM_MODEL (or the provider's
                          default model when JUDGE_PROVIDER differs)
    JUDGE_BASE_URL        judge server URL; default LLM_BASE_URL (or the
                          provider's default when JUDGE_PROVIDER differs)
    JUDGE_REASONING       judge thinking mode, same values as LLM_REASONING;
                          default "false" ("low" for the default OpenRouter
                          judge, whose thinking cannot be switched off)

Cost estimate of recorded LLM calls (optional, both or neither):
    LLM_PRICE_INPUT_PER_MTOK   USD per million input tokens
    LLM_PRICE_OUTPUT_PER_MTOK  USD per million output tokens

Embedding environment variables (all optional):
    EMBEDDING_MODEL       sentence-transformers model (default all-MiniLM-L6-v2).
                          The FAISS index must be built with the same model.
    EMBEDDING_DEVICE      "cpu", "mps" or "cuda" (default: picked automatically)

Re-ranking environment variables (all optional):
    RERANKER_MODEL        cross-encoder model for the third retrieval stage,
                          e.g. BAAI/bge-reranker-v2-m3. Empty, unset or "none"
                          (default) disables re-ranking.
    RERANK_DEPTH          top fused results the reranker scores (default 30)
    RERANKER_DEVICE       "cpu", "mps" or "cuda" (default: picked automatically)
"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Union

# Ollama listens on IPv4; "localhost" can resolve to ::1 first on macOS.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# OpenRouter serves many model families behind the OpenAI chat-completions API.
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODELS = {
    "ollama": "qwen3.8:27b-mlx",
    "openrouter": "qwen/qwen3.8-flash",
    "openai": "gpt-6-luna",  # gpt-4o-mini-2024-07-18 is retired on 2027-01-14
    "anthropic": "claude-haiku-5-5",
}

# Server URL per provider when LLM_BASE_URL / JUDGE_BASE_URL is not set
# (a provider not listed uses its client library's default).
DEFAULT_BASE_URLS = {
    "ollama": DEFAULT_OLLAMA_URL,
    "openrouter": DEFAULT_OPENROUTER_URL,
}

SUPPORTED_PROVIDERS = ("ollama", "openrouter", "openai", "anthropic", "none")

# Providers whose thinking mode is off unless LLM_REASONING says otherwise:
# thinking tokens make structured output slower and costlier, and the
# default models of both providers can switch it off.
THINKING_OFF_BY_DEFAULT = ("ollama", "openrouter")

# Effort levels accepted by LLM_REASONING / JUDGE_REASONING besides
# true/false/default. Some OpenRouter models (e.g. MiniMax M3.1 Flash,
# Gemini 3.8 Flash) reject "false" because their reasoning is mandatory;
# for them the lowest effort level is the closest to "off".
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

# Thinking mode: True/False switch it on/off, an effort level sets its
# effort, None keeps the model default.
Reasoning = Union[bool, str, None]

# Default judge per provider when JUDGE_MODEL is not set. On OpenRouter it is
# GLM-5.3 Flash (Z.ai), another vendor than the default system LLM (Qwen3.8
# Flash, Alibaba): the literature recommends a judge from a different family
# than the system it judges. Both were chosen on the project's own tasks
# (data/evaluation/LLM_MODEL_SELECTION.md, 2026-11-11). Other providers fall
# back to the system model.
DEFAULT_JUDGE_MODELS = {"openrouter": "z-ai/glm-5.3-flash"}

# Thinking mode of those default judges when JUDGE_REASONING is not set.
# GLM-5.3 Flash has mandatory reasoning on OpenRouter ("off" is rejected with
# HTTP 400), so it runs at effort "low", the level it was measured with.
DEFAULT_JUDGE_REASONING = {"openrouter": "low"}

# Environment variable holding the OpenRouter API key.
OPENROUTER_API_KEY_VAR = "OPENROUTER_API_KEY"

# Default embedding model: the one the shipped index (data/indexes) was built with.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Torch device families accepted for EMBEDDING_DEVICE ("cuda:1" is also valid).
SUPPORTED_EMBEDDING_DEVICES = ("cpu", "mps", "cuda")


def parse_reasoning(raw: str) -> Reasoning:
    """
    Parse an LLM_REASONING / JUDGE_REASONING value.

    "true"/"false" give a bool, an effort level ("low", ...) is kept as text,
    anything else ("default") gives None: the model decides.
    """
    value = raw.strip().lower()
    if value in ("true", "false"):
        return value == "true"
    if value in REASONING_EFFORTS:
        return value
    return None


def _check_reasoning(reasoning: Reasoning, variable: str) -> None:
    if isinstance(reasoning, str) and reasoning not in REASONING_EFFORTS:
        raise ValueError(
            f"{variable} must be true, false, default or one of {REASONING_EFFORTS}, got '{reasoning}'"
        )


def openrouter_api_key(env: Optional[Mapping[str, str]] = None) -> str:
    """
    The OpenRouter API key, read from OPENROUTER_API_KEY.

    It is read only when an OpenRouter chat model is built and never stored
    in the settings dataclasses, so settings can be printed or written to an
    evaluation report without leaking it.

    Raises:
        ValueError: If the variable is unset or empty
    """
    env = os.environ if env is None else env
    key = env.get(OPENROUTER_API_KEY_VAR, "").strip()
    if not key:
        raise ValueError(
            f"{OPENROUTER_API_KEY_VAR} is not set: the 'openrouter' provider needs an OpenRouter "
            f"API key. Set {OPENROUTER_API_KEY_VAR} in the environment or in .env."
        )
    return key


@dataclass(frozen=True)
class LLMSettings:
    """Settings needed to build the LLM client."""

    provider: str = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    base_url: Optional[str] = DEFAULT_OLLAMA_URL
    temperature: float = 0.0
    num_ctx: Optional[int] = 8192
    reasoning: Reasoning = False
    max_retries: int = 1
    max_explanations: int = 3
    apply_extracted_category: bool = False
    max_tokens: Optional[int] = 1024
    timeout_s: Optional[float] = 180.0

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {SUPPORTED_PROVIDERS}, got '{self.provider}'"
            )
        if self.max_retries < 0:
            raise ValueError(f"LLM_MAX_RETRIES must be >= 0, got {self.max_retries}")
        if self.max_explanations < 0:
            raise ValueError(
                f"LLM_MAX_EXPLANATIONS must be >= 0, got {self.max_explanations}"
            )
        _check_reasoning(self.reasoning, "LLM_REASONING")

    @property
    def enabled(self) -> bool:
        """False when the LLM is switched off (LLM_PROVIDER=none)."""
        return self.provider != "none"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "LLMSettings":
        """Build settings from environment variables (see module docstring)."""
        env = os.environ if env is None else env
        provider = env.get("LLM_PROVIDER", "ollama").strip().lower()
        is_ollama = provider == "ollama"

        reasoning_raw = env.get(
            "LLM_REASONING", "false" if provider in THINKING_OFF_BY_DEFAULT else "default"
        )

        num_ctx_raw = env.get("LLM_NUM_CTX")
        return cls(
            provider=provider,
            model=env.get("LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
            base_url=env.get("LLM_BASE_URL", DEFAULT_BASE_URLS.get(provider)),
            temperature=float(env.get("LLM_TEMPERATURE", "0.0")),
            num_ctx=int(num_ctx_raw) if num_ctx_raw else (8192 if is_ollama else None),
            reasoning=parse_reasoning(reasoning_raw),
            max_retries=int(env.get("LLM_MAX_RETRIES", "1")),
            max_explanations=int(env.get("LLM_MAX_EXPLANATIONS", "3")),
            apply_extracted_category=env.get("LLM_APPLY_CATEGORY_FILTER", "false").strip().lower() == "true",
            max_tokens=int(env.get("LLM_MAX_TOKENS", "1024")),
            timeout_s=float(env.get("LLM_TIMEOUT_S", "180")),
        )


# A one-sentence rationale and a grade fit well under this cap.
JUDGE_MAX_TOKENS = 256


@dataclass(frozen=True)
class JudgeSettings:
    """
    Settings of the LLM relevance judge (app/evaluation/relevance_judge.py).

    Separate from LLMSettings so the judge can be another model than the one
    the search system uses: the literature recommends a judge from a
    different model family than the systems it judges. Temperature (0) and
    the token cap are fixed, not read from the environment, so every
    judgment is made the same way. Thinking is off unless JUDGE_REASONING
    says otherwise; models whose thinking cannot be switched off (such as
    the default OpenRouter judge) run at their lowest effort instead.
    """

    provider: str = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    base_url: Optional[str] = DEFAULT_OLLAMA_URL
    num_ctx: Optional[int] = 8192
    timeout_s: Optional[float] = 180.0
    temperature: float = 0.0
    reasoning: Reasoning = False
    max_tokens: int = JUDGE_MAX_TOKENS
    system_llm: str = ""  # "provider/model" of the system LLM, for provenance notes

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS or self.provider == "none":
            raise ValueError(
                "The relevance judge needs an LLM: JUDGE_PROVIDER (or LLM_PROVIDER) must be one of "
                f"{tuple(p for p in SUPPORTED_PROVIDERS if p != 'none')}, got '{self.provider}'"
            )
        if not self.model.strip():
            raise ValueError("JUDGE_MODEL must not be empty")
        _check_reasoning(self.reasoning, "JUDGE_REASONING")

    @property
    def same_as_system_llm(self) -> bool:
        """True when the judge is the model the search system itself uses."""
        return f"{self.provider}/{self.model}" == self.system_llm

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "JudgeSettings":
        """
        Build the judge settings from JUDGE_* variables, falling back to LLM_*.

        Without JUDGE_MODEL the judge is the provider's default judge when it
        has one (OpenRouter: a model of another family than the default
        system LLM), otherwise the system model. When JUDGE_PROVIDER names
        another provider than LLM_PROVIDER, the model and URL fall back to
        that provider's defaults instead, since a model id of one provider
        means nothing to another.
        """
        env = os.environ if env is None else env
        system = LLMSettings.from_env(env)
        provider = env.get("JUDGE_PROVIDER", "").strip().lower() or system.provider
        same_provider = provider == system.provider
        is_ollama = provider == "ollama"

        model = env.get("JUDGE_MODEL", "").strip()
        if not model:
            if provider in DEFAULT_JUDGE_MODELS:
                model = DEFAULT_JUDGE_MODELS[provider]
            else:
                model = system.model if same_provider else DEFAULT_MODELS.get(provider, "")

        reasoning_raw = env.get("JUDGE_REASONING", "").strip()
        if reasoning_raw:
            reasoning = parse_reasoning(reasoning_raw)
        elif model == DEFAULT_JUDGE_MODELS.get(provider):
            reasoning = DEFAULT_JUDGE_REASONING[provider]
        else:
            reasoning = False

        return cls(
            provider=provider,
            model=model,
            base_url=env.get("JUDGE_BASE_URL", "").strip()
            or (system.base_url if same_provider else DEFAULT_BASE_URLS.get(provider)),
            num_ctx=system.num_ctx if same_provider else (8192 if is_ollama else None),
            timeout_s=system.timeout_s,
            reasoning=reasoning,
            system_llm=f"{system.provider}/{system.model}",
        )


@dataclass(frozen=True)
class TokenPrices:
    """
    Prices of one model in USD per million tokens, used only to estimate the
    cost of recorded LLM calls (app/infrastructure/llm/usage.py).

    No price is hardcoded: prices change often and differ per provider, so
    they are given for each run (LLM_PRICE_INPUT_PER_MTOK and
    LLM_PRICE_OUTPUT_PER_MTOK, or constructed directly).
    """

    input_per_mtok: float
    output_per_mtok: float

    def __post_init__(self) -> None:
        if self.input_per_mtok < 0 or self.output_per_mtok < 0:
            raise ValueError("Token prices must be >= 0")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimated cost in USD of the given token counts."""
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["TokenPrices"]:
        """
        Prices from LLM_PRICE_INPUT_PER_MTOK and LLM_PRICE_OUTPUT_PER_MTOK.

        Returns:
            TokenPrices, or None when neither variable is set

        Raises:
            ValueError: If only one of the two variables is set
        """
        env = os.environ if env is None else env
        raw_input = env.get("LLM_PRICE_INPUT_PER_MTOK", "").strip()
        raw_output = env.get("LLM_PRICE_OUTPUT_PER_MTOK", "").strip()
        if not raw_input and not raw_output:
            return None
        if not (raw_input and raw_output):
            raise ValueError(
                "Set both LLM_PRICE_INPUT_PER_MTOK and LLM_PRICE_OUTPUT_PER_MTOK "
                "(USD per million tokens), or neither"
            )
        return cls(input_per_mtok=float(raw_input), output_per_mtok=float(raw_output))


@dataclass(frozen=True)
class EmbeddingSettings:
    """Settings needed to build the embeddings store (vector side)."""

    model: str = DEFAULT_EMBEDDING_MODEL
    device: Optional[str] = None  # None: sentence-transformers picks the device

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty")
        if self.device is not None and self.device.split(":")[0] not in SUPPORTED_EMBEDDING_DEVICES:
            raise ValueError(
                f"EMBEDDING_DEVICE must be one of {SUPPORTED_EMBEDDING_DEVICES} "
                f"(optionally with an index, e.g. cuda:0), got '{self.device}'"
            )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "EmbeddingSettings":
        """Build settings from environment variables (see module docstring)."""
        env = os.environ if env is None else env
        model = env.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
        device = env.get("EMBEDDING_DEVICE", "").strip().lower() or None
        return cls(model=model, device=device)


# Top fused results scored by the reranker when RERANK_DEPTH is not set.
DEFAULT_RERANK_DEPTH = 30


@dataclass(frozen=True)
class RerankerSettings:
    """Settings of the optional cross-encoder reranker (third retrieval stage)."""

    model: Optional[str] = None  # None: re-ranking disabled (default)
    depth: int = DEFAULT_RERANK_DEPTH
    device: Optional[str] = None  # None: sentence-transformers picks the device

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ValueError("RERANKER_MODEL must be a model name or unset")
        if self.depth < 1:
            raise ValueError(f"RERANK_DEPTH must be >= 1, got {self.depth}")
        if self.device is not None and self.device.split(":")[0] not in SUPPORTED_EMBEDDING_DEVICES:
            raise ValueError(
                f"RERANKER_DEVICE must be one of {SUPPORTED_EMBEDDING_DEVICES} "
                f"(optionally with an index, e.g. cuda:0), got '{self.device}'"
            )

    @property
    def enabled(self) -> bool:
        """True when a reranker model is configured."""
        return self.model is not None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RerankerSettings":
        """Build settings from environment variables (see module docstring)."""
        env = os.environ if env is None else env
        model = env.get("RERANKER_MODEL", "").strip()
        depth = env.get("RERANK_DEPTH", "").strip()
        return cls(
            model=None if model.lower() in ("", "none") else model,
            depth=int(depth) if depth else DEFAULT_RERANK_DEPTH,
            device=env.get("RERANKER_DEVICE", "").strip().lower() or None,
        )


# --- MCP server: read-only catalog tools for AI assistants -------------------

# Host headers and browser origins the MCP endpoint accepts by default: this
# machine only (protection against DNS rebinding). A public deployment lists
# its own hostname in MCP_ALLOWED_HOSTS.
DEFAULT_MCP_ALLOWED_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*")
DEFAULT_MCP_ALLOWED_ORIGINS = ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*")

# Requests per minute accepted from one client address (0 disables the limit).
DEFAULT_MCP_RATE_LIMIT_PER_MINUTE = 60

# Largest JSON payload one tool may return, in bytes (UTF-8, compact JSON).
DEFAULT_MCP_MAX_RESPONSE_BYTES = 40_000


def _split_list(raw: str) -> tuple[str, ...]:
    """Split a comma-separated environment value into its non-empty items."""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class MCPSettings:
    """
    Settings of the MCP endpoint (/mcp) that exposes the catalog to AI assistants.

    Environment variables (all optional):
        MCP_ENABLED                "false" switches the endpoint off (default "true")
        MCP_ALLOWED_HOSTS          comma-separated Host header values accepted;
                                   "host:*" accepts any port. "*" turns the Host
                                   and Origin checks off (only behind a proxy that
                                   controls the Host header). Default: localhost.
        MCP_ALLOWED_ORIGINS        comma-separated browser origins accepted
                                   (requests without an Origin header always pass).
                                   Default: http://localhost and http://127.0.0.1.
        MCP_RATE_LIMIT_PER_MINUTE  requests per minute per client address (default 60,
                                   0 disables the limit)
        MCP_MAX_RESPONSE_BYTES     largest JSON payload of one tool result (default 40000)
    """

    enabled: bool = True
    allowed_hosts: tuple[str, ...] = DEFAULT_MCP_ALLOWED_HOSTS
    allowed_origins: tuple[str, ...] = DEFAULT_MCP_ALLOWED_ORIGINS
    rate_limit_per_minute: int = DEFAULT_MCP_RATE_LIMIT_PER_MINUTE
    max_response_bytes: int = DEFAULT_MCP_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        if self.rate_limit_per_minute < 0:
            raise ValueError(
                f"MCP_RATE_LIMIT_PER_MINUTE must be >= 0, got {self.rate_limit_per_minute}"
            )
        if self.max_response_bytes < 2_000:
            raise ValueError(
                f"MCP_MAX_RESPONSE_BYTES must be at least 2000, got {self.max_response_bytes}"
            )

    @property
    def host_check_enabled(self) -> bool:
        """False when MCP_ALLOWED_HOSTS is "*" (Host and Origin checks off)."""
        return "*" not in self.allowed_hosts

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "MCPSettings":
        """Build settings from environment variables (see class docstring)."""
        env = os.environ if env is None else env
        enabled = env.get("MCP_ENABLED", "true").strip().lower()
        if enabled not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            raise ValueError(f"MCP_ENABLED must be true or false, got '{enabled}'")
        rate_limit = env.get("MCP_RATE_LIMIT_PER_MINUTE", "").strip()
        max_bytes = env.get("MCP_MAX_RESPONSE_BYTES", "").strip()
        return cls(
            enabled=enabled in ("true", "1", "yes", "on"),
            allowed_hosts=_split_list(env.get("MCP_ALLOWED_HOSTS", "")) or DEFAULT_MCP_ALLOWED_HOSTS,
            allowed_origins=_split_list(env.get("MCP_ALLOWED_ORIGINS", "")) or DEFAULT_MCP_ALLOWED_ORIGINS,
            rate_limit_per_minute=int(rate_limit) if rate_limit else DEFAULT_MCP_RATE_LIMIT_PER_MINUTE,
            max_response_bytes=int(max_bytes) if max_bytes else DEFAULT_MCP_MAX_RESPONSE_BYTES,
        )
