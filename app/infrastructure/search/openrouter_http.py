"""
Minimal HTTP client for OpenRouter's non-chat endpoints (embeddings, rerank).

Shared by OpenRouterEmbeddingEncoder and ApiReranker. Chat completions go
through LangChain instead (app/infrastructure/llm/); these two endpoints are
plain JSON requests, so a few lines of httpx (already a dependency) are
simpler than another client library.

Behaviour:
- POST JSON to https://openrouter.ai/api/v1/<path> with the API key as a
  bearer token. No identifying header is sent (no HTTP-Referer, no X-Title).
- One timeout per attempt (timeout_s), for the whole attempt: the body is
  read in chunks and the attempt is abandoned (a transient timeout) once it
  has lasted timeout_s. A per-read timeout alone is not enough: while an
  upstream provider is busy, OpenRouter keeps a non-streaming request open by
  sending keep-alive whitespace, so every read succeeds and a request could
  wait indefinitely (seen on 2026-11-11 with DeepInfra "Model busy", V13).
- Transient failures are retried up to max_retries times, waiting
  backoff_s * 2^(attempt-1) seconds, or the Retry-After header when the
  server sends one (at most MAX_WAIT_S): timeouts, connection errors, and
  HTTP 408, 425, 429, 500, 502, 503, 504, 524 and 529 (OpenRouter's
  "provider overloaded"). Other errors (400 bad request, 401 key, 402
  credits, 403, 404 model) are raised at once.
- OpenRouter can answer 200 with an {"error": ...} body; that is treated as
  the error it carries.
- The latency returned is the wall clock of the whole call, retries and
  waits included. Error messages never contain the key.

This module imports no model library, so the cloud profile can use it in a
process without torch.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import httpx

from app.infrastructure.config.settings import DEFAULT_OPENROUTER_URL, openrouter_api_key

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying (the request may succeed a moment later).
TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 524, 529})

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 1.0
MAX_WAIT_S = 30.0  # longest wait between attempts, also for Retry-After


class OpenRouterError(RuntimeError):
    """
    A request to OpenRouter failed (after its retries, if it was transient).

    Attributes:
        status: HTTP status (or the error code of a 200 body), None for a
            timeout or connection error
        transient: True if the failure was of a kind worth retrying
        latency_s: Wall clock spent on the call before giving up
    """

    def __init__(self, message: str, status: Optional[int] = None, transient: bool = False,
                 latency_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.status = status
        self.transient = transient
        self.latency_s = latency_s


@dataclass(frozen=True)
class ApiResponse:
    """A successful response: the JSON body, the call's wall clock and attempts made."""

    body: Dict[str, Any]
    latency_s: float
    attempts: int


def _error_message(payload: Any) -> str:
    """Short message of an OpenRouter error body (never the request)."""
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            return str(error.get("message", ""))[:300]
        if error is not None:
            return str(error)[:300]
    return ""


class OpenRouterHttp:
    """Posts JSON to OpenRouter with retries on transient errors (see module docstring)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        Args:
            api_key: OpenRouter key; None reads OPENROUTER_API_KEY (error if unset)
            base_url: API root (default https://openrouter.ai/api/v1)
            timeout_s: Timeout of one attempt, in seconds
            max_retries: Retries after the first attempt for transient errors
            backoff_s: First wait between attempts (doubles each retry)
            transport: httpx transport (tests pass a fake one; None = network)
            sleep: Wait function (tests pass one that does not wait)
        """
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        key = api_key if api_key is not None else openrouter_api_key()
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {key}"},
        )

    def _wait_s(self, attempt: int, response: Optional[httpx.Response]) -> float:
        """Seconds to wait before the next attempt (Retry-After if the server sent one)."""
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), MAX_WAIT_S)
            except ValueError:
                pass
        return min(self._backoff_s * (2 ** (attempt - 1)), MAX_WAIT_S)

    def post(self, path: str, body: Mapping[str, Any]) -> ApiResponse:
        """
        POST `body` as JSON to `path` (relative to the API root).

        Returns:
            The parsed JSON body, the wall clock of the call and the attempts made

        Raises:
            OpenRouterError: On a non-transient error, or when the retries are spent
        """
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            response: Optional[httpx.Response] = None
            try:
                response = self._post_once(path, body)
            except (httpx.TimeoutException, httpx.TransportError) as error:
                failure = OpenRouterError(f"POST {path}: {type(error).__name__}", transient=True)
            else:
                failure = self._check(path, response)
                if failure is None:
                    return ApiResponse(response.json(), time.perf_counter() - started, attempt)
            failure.latency_s = time.perf_counter() - started
            if not failure.transient or attempt > self._max_retries:
                if attempt > 1:
                    failure.args = (f"{failure.args[0]} (after {attempt} attempts)",)
                raise failure
            wait = self._wait_s(attempt, response)
            logger.warning("OpenRouter %s failed (%s); retrying in %.1f s", path, failure, wait)
            self._sleep(wait)

    def _post_once(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        """
        One attempt, abandoned once it has lasted timeout_s (see the module docstring).

        Raises:
            httpx.ReadTimeout: When the whole response has not arrived in time
        """
        deadline = time.perf_counter() + self._timeout_s
        with self._client.stream("POST", path, json=dict(body)) as streamed:
            if streamed.is_stream_consumed:  # already read into memory (in-memory test transports)
                return streamed
            chunks = []
            # Raw (still compressed) bytes: the Response built below decodes them
            # with the response's own Content-Encoding header.
            for chunk in streamed.iter_raw():
                chunks.append(chunk)
                if time.perf_counter() > deadline:
                    raise httpx.ReadTimeout(f"no complete response within {self._timeout_s:.0f} s",
                                            request=streamed.request)
            return httpx.Response(streamed.status_code, headers=streamed.headers, content=b"".join(chunks),
                                  request=streamed.request)

    @staticmethod
    def _check(path: str, response: httpx.Response) -> Optional[OpenRouterError]:
        """None if the response is a usable 200; otherwise the error it represents."""
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code != 200:
            status = response.status_code
            return OpenRouterError(f"POST {path}: HTTP {status} {_error_message(payload)}".rstrip(),
                                   status=status, transient=status in TRANSIENT_STATUS)
        if not isinstance(payload, dict):
            return OpenRouterError(f"POST {path}: the response is not a JSON object", status=200)
        if "error" in payload:
            code = (payload.get("error") or {}).get("code") if isinstance(payload.get("error"), Mapping) else None
            status = code if isinstance(code, int) else None
            return OpenRouterError(f"POST {path}: error in a 200 response: {_error_message(payload)}",
                                   status=status, transient=status in TRANSIENT_STATUS)
        return None

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()
