"""
Postgres implementation of the UsageBudget port: a daily allowance of paid
model calls shared by every instance of a stateless deployment (V17).

One small table counts, per UTC day and kind, the units granted:

    <schema>.daily_usage (day date, kind text, used integer, PRIMARY KEY (day, kind))

Taking a unit is ONE statement, so ONE round trip and atomic:

    INSERT INTO daily_usage (day, kind, used) VALUES (<today UTC>, <kind>, 1)
    ON CONFLICT (day, kind) DO UPDATE SET used = daily_usage.used + 1
        WHERE daily_usage.used < <limit>
    RETURNING used

The first unit of the day inserts the row; later ones increment it while it
is below the limit. The conflicting row is locked during the update, so two
instances that ask at the same moment are counted one after the other and
the limit is never passed. When the row is at the limit the WHERE clause
fails, nothing is returned and the unit is refused. The day comes from the
database clock, so every instance agrees on when the day changes (00:00 UTC).

The table is created on the first call of each process, in the same message
as the first increment (still one round trip). The schema is separate from
the catalog's, so reloading or dropping a catalog does not reset the count.

Failures (database unreachable, timeout) propagate; the search service then
treats the unit as refused (fail closed). PostgresDatabase retries a call
once when its connection dropped, so in that rare case a unit can be counted
twice: the error is on the side of spending less.
"""

import logging
import re
from typing import Dict, Mapping

from app.domain.ports import BUDGET_LLM_REQUEST, BUDGET_MODEL_SEARCH, UsageBudget
from app.infrastructure.config.settings import SCHEMA_NAME_PATTERN
from app.infrastructure.db.postgres_connection import PostgresDatabase

logger = logging.getLogger(__name__)

KNOWN_KINDS = (BUDGET_MODEL_SEARCH, BUDGET_LLM_REQUEST)


def create_table_sql(schema: str) -> str:
    """DDL of the counter table (idempotent)."""
    return (
        f"CREATE SCHEMA IF NOT EXISTS {schema}; "
        f"CREATE TABLE IF NOT EXISTS {schema}.daily_usage ("
        "day date NOT NULL, kind text NOT NULL, used integer NOT NULL, "
        "PRIMARY KEY (day, kind)); "
    )


def consume_sql(schema: str) -> str:
    """The atomic increment (see the module docstring); parameters kind and limit."""
    return (
        f"INSERT INTO {schema}.daily_usage AS u (day, kind, used) "
        "VALUES ((now() AT TIME ZONE 'UTC')::date, %(kind)s, 1) "
        "ON CONFLICT (day, kind) DO UPDATE SET used = u.used + 1 "
        "WHERE u.used < %(limit)s "
        "RETURNING used"
    )


class PostgresUsageBudget(UsageBudget):
    """
    Daily allowance of paid model calls, counted in Postgres.

    Args:
        db: Database (the pooled connection string of the API)
        schema: Schema of the counter table (a plain lower-case identifier)
        limits: Units per UTC day for each kind (BUDGET_MODEL_SEARCH,
            BUDGET_LLM_REQUEST). A kind without a limit is always granted
            and costs no database call; a limit of 0 always refuses.

    Example:
        >>> budget = PostgresUsageBudget(db, "usage_budget", {"model_search": 400, "llm_request": 60})
        >>> budget.try_consume("model_search")
        True
    """

    def __init__(self, db: PostgresDatabase, schema: str, limits: Mapping[str, int]) -> None:
        if not re.match(SCHEMA_NAME_PATTERN, schema) or schema.startswith("pg_"):
            raise ValueError(f"Invalid budget schema name {schema!r}: use lower-case letters, digits and _")
        unknown = set(limits) - set(KNOWN_KINDS)
        if unknown:
            raise ValueError(f"Unknown budget kinds {sorted(unknown)}; known: {KNOWN_KINDS}")
        if any(limit < 0 for limit in limits.values()):
            raise ValueError(f"Budget limits must be >= 0, got {dict(limits)}")
        self._db = db
        self._schema = schema
        self._limits: Dict[str, int] = dict(limits)
        self._table_ready = False

    def try_consume(self, kind: str) -> bool:
        """Take one unit of `kind` today (one round trip); False when the day's allowance is spent."""
        if kind not in KNOWN_KINDS:
            raise ValueError(f"Unknown budget kind {kind!r}; known: {KNOWN_KINDS}")
        limit = self._limits.get(kind)
        if limit is None:
            return True
        if limit <= 0:
            return False
        sql = consume_sql(self._schema)
        if not self._table_ready:
            sql = create_table_sql(self._schema) + sql
        rows = self._db.query(sql, {"kind": kind, "limit": limit})
        self._table_ready = True
        granted = bool(rows)
        if not granted:
            logger.warning("Daily budget of '%s' spent (%d units)", kind, limit)
        return granted

    def limits(self) -> Dict[str, int]:
        """Units per UTC day for each limited kind."""
        return dict(self._limits)

    def used_today(self) -> Dict[str, int]:
        """Units granted so far today, per kind (for reports and tests)."""
        rows = self._db.query(
            create_table_sql(self._schema)
            + f"SELECT kind, used FROM {self._schema}.daily_usage "
            "WHERE day = (now() AT TIME ZONE 'UTC')::date"
        )
        self._table_ready = True
        return {row["kind"]: int(row["used"]) for row in rows}
