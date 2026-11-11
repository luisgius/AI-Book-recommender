"""
In-process rate limit for the public MCP endpoint.

A sliding window per client address: at most `max_requests` requests in any
`window_seconds`. The state lives in this process only, so with several
workers or replicas each one counts on its own, and behind a reverse proxy the
client address is the proxy's unless uvicorn runs with --proxy-headers. A
shared limit (Redis, or the platform's own limiter) is the next step if that
matters.
"""

import math
import time
from collections import deque
from typing import Callable, Deque, Dict

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Above this many tracked addresses, idle ones are forgotten (bounds memory)
MAX_TRACKED_CLIENTS = 10_000

# JSON-RPC "server error" range; the HTTP status (429) is what clients act on
RATE_LIMIT_ERROR_CODE = -32000


class SlidingWindowRateLimiter:
    """Counts requests per key and refuses those above the limit."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            max_requests: Requests allowed per key in any window
            window_seconds: Length of the sliding window
            clock: Time source in seconds (injectable for tests)
        """
        if max_requests < 1:
            raise ValueError(f"max_requests must be >= 1, got {max_requests}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}

    def check(self, key: str) -> float:
        """
        Record one request from `key` if it is allowed.

        Returns:
            0.0 when the request is allowed, otherwise the seconds until the
            key may send its next request.
        """
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window_seconds:
            hits.popleft()

        if len(hits) >= self.max_requests:
            return hits[0] + self.window_seconds - now

        hits.append(now)
        if len(self._hits) > MAX_TRACKED_CLIENTS:
            self._forget_idle(now)
        return 0.0

    def _forget_idle(self, now: float) -> None:
        """Drop keys with no request inside the current window."""
        idle = [k for k, hits in self._hits.items() if not hits or hits[-1] <= now - self.window_seconds]
        for key in idle:
            del self._hits[key]


class RateLimitMiddleware:
    """ASGI wrapper that answers 429 Too Many Requests above the limit."""

    def __init__(self, app: ASGIApp, limiter: SlidingWindowRateLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        key = client[0] if client else "unknown"
        retry_after = self.limiter.check(key)
        if retry_after > 0:
            seconds = max(1, math.ceil(retry_after))
            response = JSONResponse(
                status_code=429,
                headers={"Retry-After": str(seconds)},
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": RATE_LIMIT_ERROR_CODE,
                        "message": (
                            f"Rate limit exceeded: at most {self.limiter.max_requests} requests per "
                            f"{self.limiter.window_seconds:g} seconds per client. Retry in {seconds} s."
                        ),
                    },
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
