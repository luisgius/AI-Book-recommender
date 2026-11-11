"""
Usage records of model calls: operation, model, latency, tokens and reported cost.

UsageRecorder is the plain recorder, with no LangChain dependency, so the API
embedding encoder and the API reranker of the cloud profile can record their
HTTP calls in a process that does not import LangChain. The LangChain
callback handler that records chat-model calls (LLMUsageRecorder in
app/infrastructure/llm/usage.py) extends it, so every recorder is read the
same way: `records` or `summary(prices)`.

The recorder only appends a small record per call (no logging, no I/O), so it
costs nothing noticeable when nobody reads it. It is thread-safe: calls may
finish on several threads.
"""

import statistics
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.infrastructure.config.settings import TokenPrices

# Operation of calls made without a label.
UNLABELLED = "unlabelled"

# Key of the summary row that aggregates every operation.
TOTAL = "total"

# Operations of calls that are not chat-model calls (recorded with
# UsageRecorder.record_call by the API encoder and the API reranker).
EMBEDDING_OPERATION = "embedding"
RERANK_OPERATION = "rerank"


@dataclass(frozen=True)
class LLMCallRecord:
    """One model call (chat, embedding or rerank)."""

    operation: str
    model: str
    latency_s: float  # wall clock from request to response, network included
    input_tokens: Optional[int]  # None: the provider did not report it
    output_tokens: Optional[int]  # None: the provider did not report it
    step: Optional[str] = None  # LangGraph node that made the call, if any
    error: Optional[str] = None  # exception type when the call failed
    cost_usd: Optional[float] = None  # cost the provider reported; None: not reported


class UsageRecorder:
    """Records operation, model, latency, tokens and reported cost of model calls."""

    def __init__(self, default_operation: str = UNLABELLED) -> None:
        """
        Args:
            default_operation: Operation of calls recorded without a label
                (used by the LangChain subclass)
        """
        self.default_operation = default_operation
        self._records: List[LLMCallRecord] = []
        self._lock = threading.Lock()

    def record_call(
        self,
        operation: str,
        model: str,
        latency_s: float,
        input_tokens: Optional[int],
        output_tokens: Optional[int] = 0,
        cost_usd: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record one call that does not go through a LangChain chat model.

        Used for embedding and rerank requests (operations EMBEDDING_OPERATION
        and RERANK_OPERATION), which have input tokens but generate none, so
        their output tokens are 0 rather than unknown.

        Args:
            operation: Operation label of the call
            model: Model id
            latency_s: Wall clock of the call, retries included
            input_tokens: Tokens the provider reported (None: not reported)
            output_tokens: Generated tokens (0 for embedding and rerank calls)
            cost_usd: Cost the provider reported (None: not reported)
            error: Exception type when the call failed
        """
        self._append(LLMCallRecord(
            operation=operation,
            model=model,
            latency_s=latency_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            cost_usd=cost_usd,
        ))

    def _append(self, record: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(record)

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
        return summarize_records(self.records, prices)


def summarize_records(records: List[LLMCallRecord], prices: Optional[TokenPrices] = None) -> Dict[str, Dict[str, Any]]:
    """The summary of UsageRecorder.summary for any list of records (one row per operation plus "total")."""
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
