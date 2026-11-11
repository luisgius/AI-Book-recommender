"""
Runtime configuration for the LLM and embedding adapters, read from
environment variables.

Kept as small frozen dataclasses (no framework) so the composition root can
build the adapters from them and tests can construct them directly.

LLM environment variables (all optional):
    LLM_PROVIDER          "ollama" (default), "openai", "anthropic" or "none"
    LLM_MODEL             model id; default depends on the provider
    LLM_BASE_URL          server URL (default http://127.0.0.1:11434 for Ollama)
    LLM_TEMPERATURE       sampling temperature (default 0.0, reproducible runs)
    LLM_NUM_CTX           context window for Ollama (default 8192)
    LLM_REASONING         "false" (default for Ollama: no thinking tokens),
                          "true", or "default" (let the model decide)
    LLM_MAX_RETRIES       retries for transient errors (default 1)
    LLM_MAX_EXPLANATIONS  top results explained per search (default 3)
    LLM_APPLY_CATEGORY_FILTER  "true" to apply the category inferred by the
                          LLM as a hard filter (default "false": soft signal)
    LLM_MAX_TOKENS        cap on generated tokens per call (default 1024)
    LLM_TIMEOUT_S         HTTP timeout per call in seconds (default 180)

Relevance judge environment variables (all optional; evaluation only):
    JUDGE_PROVIDER        provider of the LLM that grades (query, book) pairs;
                          default LLM_PROVIDER
    JUDGE_MODEL           judge model id; default LLM_MODEL (or the provider's
                          default model when JUDGE_PROVIDER differs)
    JUDGE_BASE_URL        judge server URL; default LLM_BASE_URL (or the
                          provider's default when JUDGE_PROVIDER differs)

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
from typing import Mapping, Optional

# Ollama listens on IPv4; "localhost" can resolve to ::1 first on macOS.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

DEFAULT_MODELS = {
    "ollama": "qwen3.8:27b-mlx",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "none")

# Default embedding model: the one the shipped index (data/indexes) was built with.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Torch device families accepted for EMBEDDING_DEVICE ("cuda:1" is also valid).
SUPPORTED_EMBEDDING_DEVICES = ("cpu", "mps", "cuda")


@dataclass(frozen=True)
class LLMSettings:
    """Settings needed to build the LLM client."""

    provider: str = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    base_url: Optional[str] = DEFAULT_OLLAMA_URL
    temperature: float = 0.0
    num_ctx: Optional[int] = 8192
    reasoning: Optional[bool] = False
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

        reasoning_raw = env.get("LLM_REASONING", "false" if is_ollama else "default")
        reasoning = {"true": True, "false": False}.get(reasoning_raw.strip().lower())

        num_ctx_raw = env.get("LLM_NUM_CTX")
        return cls(
            provider=provider,
            model=env.get("LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
            base_url=env.get("LLM_BASE_URL", DEFAULT_OLLAMA_URL if is_ollama else None),
            temperature=float(env.get("LLM_TEMPERATURE", "0.0")),
            num_ctx=int(num_ctx_raw) if num_ctx_raw else (8192 if is_ollama else None),
            reasoning=reasoning,
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
    different model family than the systems it judges. Temperature (0),
    thinking mode (off) and the token cap are fixed, not read from the
    environment, so every judgment is made the same way.
    """

    provider: str = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    base_url: Optional[str] = DEFAULT_OLLAMA_URL
    num_ctx: Optional[int] = 8192
    timeout_s: Optional[float] = 180.0
    temperature: float = 0.0
    reasoning: bool = False
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

    @property
    def same_as_system_llm(self) -> bool:
        """True when the judge is the model the search system itself uses."""
        return f"{self.provider}/{self.model}" == self.system_llm

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "JudgeSettings":
        """
        Build the judge settings from JUDGE_* variables, falling back to LLM_*.

        When JUDGE_PROVIDER names another provider than LLM_PROVIDER, the model
        and URL fall back to that provider's defaults instead, since a model
        id of one provider means nothing to another.
        """
        env = os.environ if env is None else env
        system = LLMSettings.from_env(env)
        provider = env.get("JUDGE_PROVIDER", "").strip().lower() or system.provider
        same_provider = provider == system.provider
        is_ollama = provider == "ollama"
        return cls(
            provider=provider,
            model=env.get("JUDGE_MODEL", "").strip()
            or (system.model if same_provider else DEFAULT_MODELS.get(provider, "")),
            base_url=env.get("JUDGE_BASE_URL", "").strip()
            or (system.base_url if same_provider else (DEFAULT_OLLAMA_URL if is_ollama else None)),
            num_ctx=system.num_ctx if same_provider else (8192 if is_ollama else None),
            timeout_s=system.timeout_s,
            system_llm=f"{system.provider}/{system.model}",
        )


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
