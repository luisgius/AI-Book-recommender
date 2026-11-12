"""
Running spend ledger of the evaluation, kept on disk.

Every paid OpenRouter call made by the evaluation is appended to a JSON Lines
file with the cost OpenRouter reported for it, the stage and the operation.
The file survives interruptions, so a resumed run continues the same total.
Before a paid call, `check()` raises SpendCapReached once the total reaches
the cap (5 USD, protocol section 12): the run stops and the report says so.

LedgerRecorder plugs the ledger into the project's usage recorders (the API
encoder, the API reranker and the LangChain chat models all write to a
UsageRecorder), so the cost recorded is the one the provider reported.
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from app.infrastructure.llm.usage import LLMUsageRecorder
from app.infrastructure.usage_recording import LLMCallRecord, UsageRecorder

LEDGER_FILE = "spend_ledger.jsonl"


class SpendCapReached(RuntimeError):
    """The recorded spend reached the cap: no further paid call is made."""


class SpendLedger:
    """Append-only ledger of paid calls (thread-safe; each entry is flushed and synced)."""

    def __init__(self, path: Path | str, cap_usd: float) -> None:
        self.path = Path(path)
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:  # a partial last line of an interrupted run
                        continue

    def add(self, stage: str, operation: str, model: str, cost_usd: Optional[float], **extra: Any) -> None:
        """Append one call (cost None: the provider reported none)."""
        entry = {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "stage": stage,
            "operation": operation,
            "model": model,
            "cost_usd": cost_usd,
            **extra,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._entries.append(entry)

    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def total(self) -> float:
        return sum(float(e.get("cost_usd") or 0.0) for e in self.entries() if e.get("operation") != "key_usage")

    def check(self) -> None:
        """Raise SpendCapReached when the recorded spend reached the cap."""
        total = self.total()
        if total >= self.cap_usd:
            raise SpendCapReached(f"Recorded spend {total:.4f} USD reached the cap of {self.cap_usd:.2f} USD")


def summarize_ledger(entries: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Calls, calls without a reported cost and cost per stage and per model, plus the total."""
    by_stage: Dict[str, Dict[str, Any]] = {}
    by_model: Dict[str, Dict[str, Any]] = {}
    key_usage = [e for e in entries if e.get("operation") == "key_usage"]
    for entry in entries:
        if entry.get("operation") == "key_usage":
            continue
        for table, key in ((by_stage, entry.get("stage", "?")), (by_model, entry.get("model", "?"))):
            row = table.setdefault(key, {"calls": 0, "failed_calls": 0, "calls_without_cost": 0, "cost_usd": 0.0})
            row["calls"] += 1
            if entry.get("error"):
                row["failed_calls"] += 1
            if entry.get("cost_usd") is None:
                row["calls_without_cost"] += 1
            else:
                row["cost_usd"] += float(entry["cost_usd"])
    total = sum(row["cost_usd"] for row in by_stage.values())
    return {"by_stage": by_stage, "by_model": by_model, "total_usd": total,
            "calls": sum(row["calls"] for row in by_stage.values()), "key_usage_snapshots": key_usage}


class _LedgerMixin:
    """Forwards every recorded call to the ledger (the provider-reported cost)."""

    ledger: SpendLedger
    stage: str

    def _append(self, record: LLMCallRecord) -> None:  # type: ignore[override]
        super()._append(record)  # type: ignore[misc]
        self.ledger.add(self.stage, record.operation, record.model, record.cost_usd,
                        input_tokens=record.input_tokens, output_tokens=record.output_tokens,
                        latency_s=round(record.latency_s, 4), error=record.error)


class LedgerRecorder(_LedgerMixin, UsageRecorder):
    """UsageRecorder (API encoder and reranker) that also writes the ledger."""

    def __init__(self, ledger: SpendLedger, stage: str) -> None:
        UsageRecorder.__init__(self)
        self.ledger = ledger
        self.stage = stage


class LedgerLLMRecorder(_LedgerMixin, LLMUsageRecorder):
    """LLMUsageRecorder (LangChain chat models) that also writes the ledger."""

    def __init__(self, ledger: SpendLedger, stage: str, default_operation: str) -> None:
        LLMUsageRecorder.__init__(self, default_operation=default_operation)
        self.ledger = ledger
        self.stage = stage
