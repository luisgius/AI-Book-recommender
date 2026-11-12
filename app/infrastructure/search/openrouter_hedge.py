"""
Hedged requests to OpenRouter: the same request to several providers, first answer wins.

Why (data/evaluation/catalog50k/investigation/FINDINGS.md, question 2): the
deployed query embedding (qwen/qwen3-embedding-8b pinned to DeepInfra) failed
in 51 of 504 calls of the V14 evaluation, all read timeouts during slow
episodes of DeepInfra; a retry on the same provider fails with it, while
Nebius fails independently. The measured remedy, used here:
- send the request to the primary provider (DeepInfra) at 0 s;
- if no answer has arrived after `after_s` (1.0 s), or every request sent so
  far has failed, send the same request to the second provider (Nebius);
- send one more request to the primary at 2.5 s;
- keep the 5 s deadline for the whole call.
In the calm rounds 2 and 3 of the investigation this failed 0 of 540 calls
(the single-provider policy 3 of 540), at the same median latency and about
0.17 extra requests per search.

How a call runs (HedgedPoster.post_first):
- Legs start at their scheduled offset, or at once when every leg already
  started has failed (an HTTP error does not wait for the timer).
- The first leg with a usable 200 answer wins. The others are cancelled at
  once: their connections are closed, so a late answer is never read. Each
  leg makes a single attempt; the later legs are the retries.
- When every leg has failed, or the deadline passes, HedgeFailed is raised.
- Every leg is reported (HedgedResponse.legs / HedgeFailed.legs) with its
  outcome, so the caller can record what was really sent and paid for.

The legs are coroutines on httpx's async client, run on one event loop in a
background thread that the poster owns; post_first() itself is a blocking
call, like OpenRouterHttp.post(), so it can be used from the synchronous
domain code (and from several threads at once). Cancelling an asyncio task is
the only portable way to abandon an in-flight httpx request; a thread blocked
in a synchronous read cannot be interrupted.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

from app.infrastructure.config.settings import DEFAULT_OPENROUTER_URL, openrouter_api_key
from app.infrastructure.search.openrouter_http import OpenRouterError, OpenRouterHttp

logger = logging.getLogger(__name__)

# Outcome of a leg that lost the race while its request was in flight.
CANCELLED = "cancelled"
# Outcome of a leg that never started (the race ended first).
NOT_SENT = "not_sent"


@dataclass(frozen=True)
class HedgeLeg:
    """One request of a hedged call: when it starts (seconds after the call) and its body."""

    start_s: float
    body: Mapping[str, Any]
    label: str  # for the report, e.g. the provider slug


@dataclass
class LegOutcome:
    """
    What happened to one leg.

    error is None for the winner, CANCELLED for a leg stopped in flight,
    NOT_SENT for a leg that never started, otherwise the failure. Only legs
    with sent=True reached OpenRouter.
    """

    label: str
    sent: bool = False
    started_s: Optional[float] = None  # seconds after the call started
    latency_s: Optional[float] = None  # wall clock of the leg itself
    body: Optional[Dict[str, Any]] = None  # the JSON answer of the winner
    error: Optional[str] = None


@dataclass(frozen=True)
class HedgedResponse:
    """The winning answer, the call's wall clock and every leg's outcome (in leg order)."""

    body: Dict[str, Any]
    latency_s: float
    winner: str
    legs: List[LegOutcome] = field(default_factory=list)


class HedgeFailed(OpenRouterError):
    """Every leg failed, or the deadline passed first. legs: every leg's outcome."""

    def __init__(self, message: str, legs: List[LegOutcome], latency_s: float) -> None:
        super().__init__(message, transient=True, latency_s=latency_s)
        self.legs = legs


class _LoopThread:
    """An asyncio event loop running in a daemon thread, started on first use."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="openrouter-hedge", daemon=True).start()
                self._loop = loop
            return self._loop

    def run(self, coroutine, timeout_s: float):
        """Run a coroutine on the loop and wait for its result (blocking)."""
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop()).result(timeout=timeout_s)

    def stop(self) -> None:
        with self._lock:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop = None


class HedgedPoster:
    """Posts the same JSON request as several legs and returns the first good answer (module docstring)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_URL,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        """
        Args:
            api_key: OpenRouter key; None reads OPENROUTER_API_KEY (error if unset)
            base_url: API root (default https://openrouter.ai/api/v1)
            transport: httpx async transport (tests pass a fake one; None = network)
        """
        self._key = api_key if api_key is not None else openrouter_api_key()
        self._base_url = base_url.rstrip("/") + "/"
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None  # created on the loop, on first use
        self._runner = _LoopThread()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # No identifying header: only the key (no HTTP-Referer, no X-Title)
            self._client = httpx.AsyncClient(
                base_url=self._base_url, transport=self._transport,
                headers={"Authorization": f"Bearer {self._key}"},
            )
        return self._client

    def post_first(self, path: str, legs: Sequence[HedgeLeg], *, deadline_s: float) -> HedgedResponse:
        """
        Race the legs; return the first usable answer (see the module docstring).

        Args:
            path: Endpoint path, e.g. "embeddings"
            legs: The requests, each with its start offset (sorted here)
            deadline_s: Most seconds for the whole call

        Raises:
            HedgeFailed: Every leg failed or the deadline passed
            ValueError: No legs, or a leg starting after the deadline
        """
        if not legs:
            raise ValueError("a hedged call needs at least one leg")
        if deadline_s <= 0:
            raise ValueError(f"deadline_s must be > 0, got {deadline_s}")
        ordered = sorted(legs, key=lambda leg: leg.start_s)
        if ordered[0].start_s != 0:
            raise ValueError("the first leg must start at 0 s")
        # The extra second only guards against a loop that never answers
        return self._runner.run(self._race(path, ordered, deadline_s), timeout_s=deadline_s + 1.0)

    async def _send(self, path: str, leg: HedgeLeg, timeout_s: float) -> Dict[str, Any]:
        """One attempt of one leg: the JSON body of a usable 200 answer, or OpenRouterError."""
        try:
            response = await self._get_client().post(path, json=dict(leg.body), timeout=timeout_s)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise OpenRouterError(f"POST {path}: {type(error).__name__}", transient=True) from None
        failure = OpenRouterHttp._check(path, response)
        if failure is not None:
            raise failure
        return response.json()

    async def _race(self, path: str, legs: List[HedgeLeg], deadline_s: float) -> HedgedResponse:
        clock = time.perf_counter
        started = clock()
        outcomes = [LegOutcome(label=leg.label) for leg in legs]
        waiting = list(range(len(legs)))  # legs not started yet, in start order
        running: Dict[asyncio.Task, int] = {}
        leg_started: Dict[int, float] = {}

        def start(index: int) -> None:
            now = clock()
            outcomes[index].sent = True
            outcomes[index].started_s = round(now - started, 4)
            leg_started[index] = now
            remaining = max(0.01, deadline_s - (now - started))
            running[asyncio.ensure_future(self._send(path, legs[index], remaining))] = index

        async def stop_running(reason: str) -> None:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            for task, index in running.items():
                outcomes[index].error = reason
                outcomes[index].latency_s = round(clock() - leg_started[index], 4)
            running.clear()

        try:
            while True:
                elapsed = clock() - started
                # Start every leg whose time has come, and the next one at once
                # when nothing is in flight (every leg sent so far has failed)
                while waiting and (legs[waiting[0]].start_s <= elapsed or not running):
                    start(waiting.pop(0))
                    elapsed = clock() - started
                left = deadline_s - elapsed
                if left <= 0:
                    await stop_running(CANCELLED)
                    break
                wait = left if not waiting else min(left, legs[waiting[0]].start_s - elapsed)
                done, _ = await asyncio.wait(set(running), timeout=max(wait, 0.0),
                                             return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index = running.pop(task)
                    outcomes[index].latency_s = round(clock() - leg_started[index], 4)
                    error = task.exception()
                    if error is None:
                        outcomes[index].body = task.result()
                        await stop_running(CANCELLED)
                        for other in waiting:
                            outcomes[other].error = NOT_SENT
                        return HedgedResponse(task.result(), clock() - started, legs[index].label, outcomes)
                    outcomes[index].error = str(error)
                    logger.warning("Hedged %s leg '%s' failed: %s", path, legs[index].label, error)
                if not running and not waiting:
                    break
        finally:
            if running:  # cancelled from outside (e.g. the caller gave up)
                await stop_running(CANCELLED)
        for index in waiting:
            outcomes[index].error = NOT_SENT
        elapsed = clock() - started
        summary = "; ".join(f"{o.label}: {o.error}" for o in outcomes if o.sent)
        if elapsed >= deadline_s:
            message = f"POST {path}: no answer within the {deadline_s:g} s deadline ({summary})"
        else:
            message = f"POST {path}: every hedged request failed ({summary})"
        raise HedgeFailed(message, outcomes, elapsed)

    def close(self) -> None:
        """Close the connection pool and stop the loop thread."""
        if self._client is not None:
            try:
                self._runner.run(self._client.aclose(), timeout_s=5.0)
            except Exception:  # noqa: BLE001 - closing must not fail the caller
                pass
            self._client = None
        self._runner.stop()
