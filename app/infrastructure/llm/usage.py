"""
Usage recording of LLM calls: operation, model, latency and tokens.

LLMUsageRecorder is a LangChain callback handler. build_chat_model attaches it
to the chat model it builds, so every call made through that model is
recorded, whichever chain, LangGraph graph or agent makes it; no value has to
be threaded through the chains.

The operation of a call (query_understanding, explanation, relevance_judge,
agent, ...) is a label in the LangChain run metadata. Callers set it with
operation_config() when they invoke a chain or graph; LangChain passes run
metadata down to the chat model calls inside, including those made from
LangGraph nodes, which also add the node name (recorded as `step`). Calls
without a label get the recorder's default operation.

Token counts are the ones the provider reports (usage_metadata of the
response: Ollama's prompt_eval_count/eval_count, the OpenAI-compatible
`usage` block). When a provider reports none, the counts are recorded as
None, never estimated. Output tokens include thinking tokens when the
provider counts them as output. The cost is recorded only when the provider
reports it (OpenRouter's `usage.cost`, in USD, the price of the endpoint
that served the call); otherwise it is None.

The recorder only appends a small record per call (no logging, no I/O), so
it costs nothing noticeable when nobody reads it. Evaluation code reads
`recorder.records` or `recorder.summary(prices)`.
"""

import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.infrastructure.config.settings import TokenPrices

# Run-metadata key that carries the operation label.
OPERATION_METADATA_KEY = "llm_operation"

# Operation of calls made without a label.
UNLABELLED = "unlabelled"

# Key of the summary row that aggregates every operation.
TOTAL = "total"


def operation_config(operation: str) -> Dict[str, Any]:
    """
    LangChain run config that labels the LLM calls of one operation.

    Usage: chain.invoke(inputs, config=operation_config("explanation")).
    """
    return {"metadata": {OPERATION_METADATA_KEY: operation}}


@dataclass(frozen=True)
class LLMCallRecord:
    """One chat-model call."""

    operation: str
    model: str
    latency_s: float  # wall clock from request to response, network included
    input_tokens: Optional[int]  # None: the provider did not report it
    output_tokens: Optional[int]  # None: the provider did not report it
    step: Optional[str] = None  # LangGraph node that made the call, if any
    error: Optional[str] = None  # exception type when the call failed
    cost_usd: Optional[float] = None  # cost the provider reported; None: not reported


@dataclass(frozen=True)
class _PendingCall:
    started: float
    operation: str
    model: str
    step: Optional[str]


def reported_tokens(response: LLMResult) -> Tuple[Optional[int], Optional[int]]:
    """
    Input and output tokens reported by the provider for one call.

    Reads the message usage_metadata (filled by ChatOllama and ChatOpenAI),
    then the OpenAI-style llm_output["token_usage"].

    Returns:
        (input_tokens, output_tokens); (None, None) when nothing is reported
    """
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                return usage.get("input_tokens"), usage.get("output_tokens")
    token_usage = (response.llm_output or {}).get("token_usage") or {}
    if token_usage:
        return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")
    return None, None


def reported_cost(response: LLMResult) -> Optional[float]:
    """
    Cost in USD the provider reported for one call, if any.

    OpenRouter adds `cost` to the OpenAI-style `usage` block, which
    LangChain's OpenAI client keeps in llm_output["token_usage"] and in the
    message's response_metadata["token_usage"]. Other providers report no
    cost, so the result is None for them.
    """
    usage_blocks = [(response.llm_output or {}).get("token_usage") or {}]
    for generations in response.generations:
        for generation in generations:
            metadata = getattr(getattr(generation, "message", None), "response_metadata", None) or {}
            usage_blocks.append(metadata.get("token_usage") or {})
    for block in usage_blocks:
        if isinstance(block.get("cost"), (int, float)):
            return float(block["cost"])
    return None


class LLMUsageRecorder(BaseCallbackHandler):
    """
    Records operation, model, latency and tokens of every call of a chat model.

    One recorder belongs to one chat model (one LLM client or one judge), so
    its records share a model and one TokenPrices can price all of them.
    Thread-safe: calls may finish on several threads.
    """

    # Run the hooks in the caller's thread, also under asyncio, so the
    # latency is measured where the call happens.
    run_inline = True

    def __init__(self, default_operation: str = UNLABELLED) -> None:
        """
        Args:
            default_operation: Operation of calls whose run metadata has no
                label (e.g. "relevance_judge" for a model used for one task)
        """
        super().__init__()
        self.default_operation = default_operation
        self._records: List[LLMCallRecord] = []
        self._pending: Dict[UUID, _PendingCall] = {}
        self._lock = threading.Lock()

    # -- LangChain callback hooks ----------------------------------------------

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Start the clock and remember the operation, model and graph node."""
        metadata = metadata or {}
        pending = _PendingCall(
            started=time.perf_counter(),
            operation=metadata.get(OPERATION_METADATA_KEY) or self.default_operation,
            model=metadata.get("ls_model_name") or "unknown",
            step=metadata.get("langgraph_node"),
        )
        with self._lock:
            self._pending[run_id] = pending

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Record a finished call with the tokens (and cost) the provider reported."""
        input_tokens, output_tokens = reported_tokens(response)
        self._finish(run_id, input_tokens, output_tokens, error=None, cost_usd=reported_cost(response))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Record a failed call (no tokens), so latency statistics include it."""
        self._finish(run_id, None, None, error=type(error).__name__)

    def _finish(
        self,
        run_id: UUID,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        error: Optional[str],
        cost_usd: Optional[float] = None,
    ) -> None:
        ended = time.perf_counter()
        with self._lock:
            pending = self._pending.pop(run_id, None)
            if pending is None:  # a run this recorder did not see start
                return
            self._records.append(LLMCallRecord(
                operation=pending.operation,
                model=pending.model,
                latency_s=ended - pending.started,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                step=pending.step,
                error=error,
                cost_usd=cost_usd,
            ))

    # -- reading ---------------------------------------------------------------

    @property
    def records(self) -> List[LLMCallRecord]:
        """A copy of the records, in completion order."""
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        """Forget the records, e.g. between two measured phases of a run."""
        with self._lock:
            self._records.clear()

    def summary(self, prices: Optional[TokenPrices] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calls, tokens and latency per operation, plus a "total" row.

        Args:
            prices: Prices of the recorder's model; without them the
                estimated cost is None

        Returns:
            {operation: row}, where a row has calls, failed_calls,
            calls_without_usage, input_tokens, output_tokens (sums of the
            reported counts; None when no call reported any), total_latency_s,
            mean_latency_s, median_latency_s, estimated_cost_usd (from the
            reported counts only; calls_without_usage says how many calls
            it misses) and reported_cost_usd (sum of the costs the provider
            reported; None when no call reported one).
        """
        records = self.records
        operations = sorted({record.operation for record in records})
        rows = {op: _summarize([r for r in records if r.operation == op], prices) for op in operations}
        rows[TOTAL] = _summarize(records, prices)
        return rows


def _sum_reported(values: List[Optional[float]]) -> Optional[float]:
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def _summarize(records: List[LLMCallRecord], prices: Optional[TokenPrices]) -> Dict[str, Any]:
    latencies = [record.latency_s for record in records]
    input_tokens = _sum_reported([record.input_tokens for record in records])
    output_tokens = _sum_reported([record.output_tokens for record in records])
    cost = None
    if prices is not None and (input_tokens is not None or output_tokens is not None):
        cost = prices.cost(input_tokens or 0, output_tokens or 0)
    return {
        "calls": len(records),
        "failed_calls": sum(1 for record in records if record.error is not None),
        "calls_without_usage": sum(
            1 for record in records if record.input_tokens is None or record.output_tokens is None
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_latency_s": sum(latencies),
        "mean_latency_s": statistics.fmean(latencies) if latencies else None,
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "estimated_cost_usd": cost,
        "reported_cost_usd": _sum_reported([record.cost_usd for record in records]),
    }
