"""
In-memory implementation of the UsageBudget port (V17).

For a single process (the local profile, a demo on one machine) and for
tests. The count lives in this process only, so it must not guard a
serverless deployment, where every instance would have its own allowance:
there PostgresUsageBudget (app/infrastructure/db/postgres_usage_budget.py)
counts in the database that all instances share.
"""

import threading
from datetime import UTC, date, datetime
from typing import Callable, Dict, Mapping, Tuple

from app.domain.ports import BUDGET_LLM_REQUEST, BUDGET_MODEL_SEARCH, UsageBudget

KNOWN_KINDS = (BUDGET_MODEL_SEARCH, BUDGET_LLM_REQUEST)


def utc_today() -> date:
    """Today's date in UTC (the day the allowance belongs to)."""
    return datetime.now(UTC).date()


class InMemoryUsageBudget(UsageBudget):
    """
    Daily allowance per kind, counted in this process.

    Args:
        limits: Units per UTC day for each kind; a kind without a limit is
            always granted, a limit of 0 always refuses
        today: Clock returning the current UTC date (injectable for tests)
    """

    def __init__(self, limits: Mapping[str, int], today: Callable[[], date] = utc_today) -> None:
        unknown = set(limits) - set(KNOWN_KINDS)
        if unknown:
            raise ValueError(f"Unknown budget kinds {sorted(unknown)}; known: {KNOWN_KINDS}")
        if any(limit < 0 for limit in limits.values()):
            raise ValueError(f"Budget limits must be >= 0, got {dict(limits)}")
        self._limits: Dict[str, int] = dict(limits)
        self._today = today
        self._used: Dict[Tuple[date, str], int] = {}
        self._lock = threading.Lock()

    def try_consume(self, kind: str) -> bool:
        """Take one unit of `kind` today; False when the day's allowance is spent."""
        if kind not in KNOWN_KINDS:
            raise ValueError(f"Unknown budget kind {kind!r}; known: {KNOWN_KINDS}")
        limit = self._limits.get(kind)
        if limit is None:
            return True
        key = (self._today(), kind)
        with self._lock:
            used = self._used.get(key, 0)
            if used >= limit:
                return False
            self._used = {k: v for k, v in self._used.items() if k[0] == key[0]}  # forget past days
            self._used[key] = used + 1
            return True

    def limits(self) -> Dict[str, int]:
        """Units per UTC day for each limited kind."""
        return dict(self._limits)

    def used_today(self) -> Dict[str, int]:
        """Units granted so far today, per kind."""
        today = self._today()
        with self._lock:
            return {kind: used for (day, kind), used in self._used.items() if day == today}
