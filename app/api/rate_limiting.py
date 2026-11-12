"""
Per-client request limits of the REST endpoints (V17).

The same sliding-window limiter as the MCP endpoint (app/api/mcp/rate_limit.py)
applied to the endpoints that cost money or wake the database:

- "search": POST /api/v1/search and GET /api/v1/books/{id}/similar
  (SEARCH_RATE_LIMIT_PER_MINUTE per client)
- "llm": searches with query understanding or explanations and
  POST /api/v1/agent/ask (LLM_RATE_LIMIT_PER_MINUTE per client), on top of
  the search limit

Above a limit the endpoint answers 429 with a Retry-After header.

What these limits cannot do: the counters live in the memory of one process.
On a serverless platform each instance counts on its own (and an instance
that scales to zero forgets its counts), so a client that reaches several
instances gets several allowances, and many clients together are not limited
at all. The hard ceiling on spending is the daily budget (UsageBudget), which
is counted in the database.

The client is identified by the connection's address or, behind a proxy that
sets it and cannot be bypassed, by a header (CLIENT_IP_HEADER; on Vercel
x-real-ip, which the platform overwrites so a client cannot forge it).
"""

import math
import time
from typing import Callable, Mapping, Optional, Tuple

from fastapi import HTTPException, Request, status

from app.api.mcp.rate_limit import SlidingWindowRateLimiter
from app.infrastructure.config.settings import RateLimitSettings

SEARCH = "search"
LLM = "llm"
WINDOW_SECONDS = 60.0


def client_address(
    headers: Mapping[str, str], client: Optional[Tuple[str, int]], header: Optional[str] = None
) -> str:
    """
    The address requests are counted under.

    Args:
        headers: Request headers (case-insensitive mapping)
        client: (host, port) of the connection, if known
        header: Trusted header with the client address (first value of a
            comma-separated list), or None to use the connection's address
    """
    if header:
        value = headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return client[0] if client else "unknown"


class RestRateLimits:
    """
    The two per-client limiters of the REST API, built from RateLimitSettings.

    A limit of 0 disables that limiter (the local default).
    """

    def __init__(self, settings: RateLimitSettings, clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self._limiters = {
            SEARCH: SlidingWindowRateLimiter(settings.search_per_minute, WINDOW_SECONDS, clock)
            if settings.search_per_minute else None,
            LLM: SlidingWindowRateLimiter(settings.llm_per_minute, WINDOW_SECONDS, clock)
            if settings.llm_per_minute else None,
        }

    def enforce(self, kind: str, request: Request) -> None:
        """
        Count one request of `kind` from this client.

        Raises:
            HTTPException: 429 with Retry-After when the client is over the limit
        """
        limiter = self._limiters[kind]
        if limiter is None:
            return
        key = client_address(request.headers, request.client, self.settings.client_ip_header)
        retry_after = limiter.check(key)
        if retry_after > 0:
            seconds = max(1, math.ceil(retry_after))
            what = "searches" if kind == SEARCH else "LLM requests"
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Too many {what} from this client: at most {limiter.max_requests} per minute. "
                    f"Retry in {seconds} s."
                ),
                headers={"Retry-After": str(seconds)},
            )
