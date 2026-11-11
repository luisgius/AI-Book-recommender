"""
Reranker port implemented by OpenRouter's rerank API (third retrieval stage).

The cloud profile cannot load a cross-encoder (no model in the process), so
the re-ranking model runs at the provider: one HTTP request per search sends
the user's query and the texts of the fused candidates and gets one
relevance score per candidate back.

Request (POST https://openrouter.ai/api/v1/rerank, checked against
OpenRouter's API reference on 2026-11-11):
    {"model": ..., "query": ..., "documents": [texts],
     "provider": {"order": [p], "allow_fallbacks": false} (optional)}
Response: {"results": [{"index": i, "relevance_score": s, "document": {...}}, ...]
           (sorted best first), "model": ..., "provider": ..., "usage": {"total_tokens", "cost"}}

Same contract as CrossEncoderReranker:
- The text of each book is Book.get_searchable_text() (title, authors,
  description, categories: the text BM25 indexes), cut to its first
  max_document_chars characters (default 2,000, about 500 tokens, close to the
  512-token pair limit of the local cross-encoder). The cut bounds the cost,
  which is per token.
- New SearchResult objects are returned (dataclasses.replace), sorted by
  score with ties kept in the fused order, rerank_score and rank set and the
  fusion scores kept; the input list is never modified.
- Any failure raises: an HTTP error after the retries of openrouter_http.py,
  or a response that does not score every candidate exactly once.
  SearchService catches it, logs a warning and keeps the fused order, so a
  search never fails because of this stage.
- is_ready() is False after a non-transient failure (wrong key, no credits,
  unknown model) until a later call succeeds; there is no model to load, so
  every search still tries the request.
- Every request is recorded in an optional UsageRecorder under the
  operation "rerank" (latency, tokens, reported cost, errors).

This module imports no model library (no torch).
"""

import logging
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional

import httpx

from app.domain.entities import Book, SearchResult
from app.domain.ports import Reranker
from app.infrastructure.config.settings import DEFAULT_OPENROUTER_URL, RerankerSettings
from app.infrastructure.usage_recording import RERANK_OPERATION, UsageRecorder
from app.infrastructure.search.openrouter_http import (
    DEFAULT_TIMEOUT_S,
    OpenRouterError,
    OpenRouterHttp,
)

logger = logging.getLogger(__name__)

# Characters of each book's searchable text sent to the reranker.
DEFAULT_MAX_DOCUMENT_CHARS = 2000

# One retry less than the encoder: a slow rerank delays the whole search.
DEFAULT_RERANK_RETRIES = 2


def read_scores(body: Mapping[str, Any], expected: int) -> List[float]:
    """
    Score of each document, in input order, from a rerank response.

    Raises:
        ValueError: If a document has no score, two scores, or an index outside 0..n-1
    """
    results = body.get("results")
    if not isinstance(results, list):
        raise ValueError("Rerank response has no results list")
    scores: Dict[int, float] = {}
    for item in results:
        index = int(item["index"])
        if index in scores or not 0 <= index < expected:
            raise ValueError(f"Rerank response has a repeated or unknown index {index}")
        scores[index] = float(item["relevance_score"])
    if len(scores) != expected:
        raise ValueError(f"Rerank response scores {len(scores)} of {expected} documents")
    return [scores[i] for i in range(expected)]


class ApiReranker(Reranker):
    """
    Reranker backed by a rerank model served by OpenRouter (see module docstring).

    Example:
        >>> reranker = ApiReranker("qwen/qwen3-reranker-8b", upstream_provider="primeintellect")
        >>> reranked = reranker.rerank("novelas de misterio", fused_results[:30])
    """

    def __init__(
        self,
        model: str,
        *,
        upstream_provider: Optional[str] = None,
        max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
        usage_recorder: Optional[UsageRecorder] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_RERANK_RETRIES,
        transport: Optional[httpx.BaseTransport] = None,
        http: Optional[OpenRouterHttp] = None,
    ) -> None:
        """
        Args:
            model: OpenRouter id of a rerank model (e.g. qwen/qwen3-reranker-8b)
            upstream_provider: Provider slug to pin (e.g. "primeintellect");
                None lets OpenRouter route each request
            max_document_chars: Characters of each book text sent
            usage_recorder: Records every request under the operation "rerank"
            api_key, base_url, timeout_s, max_retries, transport: HTTP options
                (see OpenRouterHttp); ignored when `http` is given
            http: An already built client (tests and evaluation)
        """
        if not model or not model.strip():
            raise ValueError("model must not be empty")
        if max_document_chars < 1:
            raise ValueError(f"max_document_chars must be >= 1, got {max_document_chars}")
        self._model = model.strip()
        self._upstream = upstream_provider.lower() if upstream_provider else None
        self._max_document_chars = max_document_chars
        self._recorder = usage_recorder
        self._http = http or OpenRouterHttp(
            api_key=api_key, base_url=base_url, timeout_s=timeout_s,
            max_retries=max_retries, transport=transport,
        )
        self._last_error: Optional[BaseException] = None

    @classmethod
    def from_settings(
        cls, settings: RerankerSettings, usage_recorder: Optional[UsageRecorder] = None
    ) -> "ApiReranker":
        """Build the reranker of RERANKER_PROVIDER=openrouter (reads OPENROUTER_API_KEY)."""
        if settings.provider != "openrouter" or settings.model is None:
            raise ValueError("ApiReranker needs RERANKER_PROVIDER=openrouter and a RERANKER_MODEL")
        return cls(settings.model, upstream_provider=settings.openrouter_provider,
                   usage_recorder=usage_recorder)

    def document_text(self, book: Book) -> str:
        """Text of a book as sent to the reranker: its searchable text, cut."""
        return book.get_searchable_text()[: self._max_document_chars]

    def request_body(self, query_text: str, books: List[Book]) -> Dict[str, Any]:
        """JSON body of the rerank request for these books."""
        body: Dict[str, Any] = {
            "model": self._model,
            "query": query_text,
            "documents": [self.document_text(book) for book in books],
        }
        if self._upstream:
            body["provider"] = {"order": [self._upstream], "allow_fallbacks": False}
        return body

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Score every (query, book) pair in one request and sort by score.

        Args:
            query_text: The user's original query text
            results: Candidates to re-order

        Returns:
            New SearchResult objects, best first, with rerank_score and rank
            set; the fusion scores (final, lexical, vector) are kept

        Raises:
            OpenRouterError: If the request fails (after its retries)
            ValueError: If the response does not score every candidate once
        """
        if not results:
            return []
        body = self.request_body(query_text, [result.book for result in results])
        try:
            response = self._http.post("rerank", body)
            scores = read_scores(response.body, len(results))
        except (OpenRouterError, ValueError, KeyError, TypeError) as error:
            self._last_error = error
            self._record(getattr(error, "latency_s", None) or 0.0, None, None, type(error).__name__)
            raise
        self._last_error = None
        usage = response.body.get("usage") or {}
        self._record(response.latency_s, usage.get("total_tokens"), usage.get("cost"), None)

        # sorted() is stable: equal scores keep the fused order
        order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        return [
            replace(results[i], rank=position, rerank_score=scores[i])
            for position, i in enumerate(order, start=1)
        ]

    def _record(self, latency_s: float, tokens: Optional[int], cost: Any, error: Optional[str]) -> None:
        if self._recorder is not None:
            self._recorder.record_call(
                RERANK_OPERATION, self._model, latency_s, tokens, 0,
                cost_usd=float(cost) if isinstance(cost, (int, float)) else None, error=error,
            )

    def get_model_name(self) -> str:
        """OpenRouter id of the rerank model."""
        return self._model

    def is_ready(self) -> bool:
        """False after a non-transient failure (key, credits, model) until a call succeeds."""
        error = self._last_error
        return error is None or bool(getattr(error, "transient", False))

    @property
    def upstream_provider(self) -> Optional[str]:
        """Pinned provider slug, or None (OpenRouter's routing)."""
        return self._upstream
