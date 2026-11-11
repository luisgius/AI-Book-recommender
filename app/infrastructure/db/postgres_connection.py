"""
Connections to Postgres for the cloud search profile.

One small class, PostgresDatabase, used by every Postgres adapter and by the
loader. It is written for a serverless-friendly endpoint such as Neon's
pooled connection string (PgBouncer in transaction mode):

- No session state. Every call sends ONE message with the simple query
  protocol: optional `SET LOCAL ...;` statements followed by the statement.
  A multi-statement message runs as one implicit transaction, so the
  settings apply to that statement only and never leak to the next client
  of the pooled server connection. It is also a single network round trip.
- No server-side prepared statements: parameters are bound on the client
  (psycopg ClientCursor) and prepare_threshold is disabled.
- A tiny pool (psycopg_pool) with min_size=0: connections are opened on
  demand, reused while busy (a new TLS connection to Neon costs ~350 ms) and
  closed after max_idle seconds, so an idle process holds no connection and
  the compute can scale to zero.
- Timeouts: connect timeout, a statement_timeout per call, and a bounded wait
  for a free pooled connection. A call that fails because its connection
  died (e.g. the compute was suspended) is retried once on a fresh one.

Connection strings are never logged; errors are redacted (redact()).
"""

import logging
import re
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

DEFAULT_POOL_MAX_SIZE = 4
DEFAULT_MAX_IDLE_S = 240.0  # below Neon's 5-minute scale-to-zero timeout
DEFAULT_CONNECT_TIMEOUT_S = 10
DEFAULT_STATEMENT_TIMEOUT_MS = 15000
DEFAULT_POOL_WAIT_S = 15.0

_URL_PATTERN = re.compile(r"postgres(?:ql)?://\S+", re.IGNORECASE)
_SECRET_PATTERN = re.compile(r"(password|user)\s*=\s*\S+", re.IGNORECASE)
_SETTING_NAME = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?$")


def redact(text: object) -> str:
    """Remove connection strings and credentials from an error message."""
    cleaned = _URL_PATTERN.sub("<connection string redacted>", str(text))
    return _SECRET_PATTERN.sub(r"\1=<redacted>", cleaned)


def settings_prefix(settings: Mapping[str, object]) -> str:
    """
    `SET LOCAL name = 'value';` statements for transaction-local settings.

    Names must be plain (dotted) identifiers; values are quoted as literals,
    so neither can inject SQL.

    Raises:
        ValueError: If a setting name is not a plain identifier
    """
    parts = []
    for name, value in settings.items():
        if not _SETTING_NAME.match(name):
            raise ValueError(f"Invalid setting name: {name!r}")
        literal = str(value).replace("'", "''")
        parts.append(f"SET LOCAL {name} = '{literal}';")
    return " ".join(parts) + (" " if parts else "")


class PostgresDatabase:
    """
    A Postgres database reached through a connection string.

    Typical use:
        db = PostgresDatabase(database_url())
        rows = db.query("SELECT id, title FROM catalog.works WHERE id = %(id)s", {"id": book_id})
        db.execute("UPDATE ... ; UPDATE ...", params)   # one transaction

    Args:
        url: Connection string (pooled for the application, direct for loads)
        pool_max_size: Most connections open at the same time
        max_idle_s: Idle connections are closed after this many seconds
        connect_timeout_s: Timeout of a new connection (covers a cold start)
        statement_timeout_ms: statement_timeout of every call; None disables
            it (bulk loads and index builds)
        pool_wait_s: Most time to wait for a free connection
    """

    def __init__(
        self,
        url: str,
        *,
        pool_max_size: int = DEFAULT_POOL_MAX_SIZE,
        max_idle_s: float = DEFAULT_MAX_IDLE_S,
        connect_timeout_s: int = DEFAULT_CONNECT_TIMEOUT_S,
        statement_timeout_ms: Optional[int] = DEFAULT_STATEMENT_TIMEOUT_MS,
        pool_wait_s: float = DEFAULT_POOL_WAIT_S,
    ) -> None:
        if not url or not url.strip():
            raise ValueError("A Postgres connection string is required")
        if pool_max_size < 1:
            raise ValueError(f"pool_max_size must be >= 1, got {pool_max_size}")
        self._statement_timeout_ms = statement_timeout_ms
        self._pool = ConnectionPool(
            url,
            min_size=0,
            max_size=pool_max_size,
            max_idle=max_idle_s,
            timeout=pool_wait_s,
            open=False,
            name="search-db",
            kwargs={
                "autocommit": True,
                "prepare_threshold": None,  # no server-side prepared statements
                "connect_timeout": connect_timeout_s,
                "cursor_factory": psycopg.ClientCursor,  # client-side binding
            },
        )
        self._open_lock = threading.Lock()
        self._opened = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        sql: str,
        params: Optional[Mapping[str, Any] | Sequence[Any]] = None,
        *,
        settings: Optional[Mapping[str, object]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run one statement and return its rows as dicts, in one round trip.

        Args:
            sql: The statement (psycopg placeholders: %s or %(name)s)
            params: Its parameters
            settings: Transaction-local settings for this statement only
                (e.g. {"enable_indexscan": "off"})

        Raises:
            RuntimeError: If the database cannot be reached or the statement fails
        """
        return self._run(sql, params, settings or {}, fetch=True)

    def execute(
        self,
        sql: str,
        params: Optional[Mapping[str, Any] | Sequence[Any]] = None,
        *,
        settings: Optional[Mapping[str, object]] = None,
    ) -> int:
        """
        Run one or more statements (separated by ';') as one transaction.

        Returns:
            Rows affected by the last statement (-1 if not applicable)

        Raises:
            RuntimeError: If the database cannot be reached or a statement fails
            ValueError: If a statement violates a constraint
        """
        return self._run(sql, params, settings or {}, fetch=False)

    def execute_outside_transaction(self, sql: str) -> None:
        """
        Run a single statement that cannot run inside a transaction block
        (VACUUM), without statement_timeout or settings.
        """
        with self._connection() as conn:
            conn.execute(sql)

    def close(self) -> None:
        """Close every pooled connection."""
        if self._opened:
            self._pool.close()
            self._opened = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        if not self._opened:
            with self._open_lock:
                if not self._opened:
                    self._pool.open(wait=False)
                    self._opened = True
        with self._pool.connection() as conn:
            yield conn

    def _script(self, sql: str, settings: Mapping[str, object]) -> str:
        all_settings: Dict[str, object] = {}
        if self._statement_timeout_ms is not None:
            all_settings["statement_timeout"] = f"{int(self._statement_timeout_ms)}ms"
        all_settings.update(settings)
        return settings_prefix(all_settings) + sql

    def _run(self, sql: str, params: Any, settings: Mapping[str, object], *, fetch: bool) -> Any:
        script = self._script(sql, settings)
        for attempt in (1, 2):
            try:
                with self._connection() as conn:
                    cur = conn.cursor(row_factory=dict_row)
                    cur.execute(script, params)
                    if not fetch:
                        while cur.nextset():  # move to the last statement's result
                            pass
                        return cur.rowcount
                    # Skip the results of the SET LOCAL statements.
                    while cur.description is None and cur.nextset():
                        pass
                    return cur.fetchall() if cur.description is not None else []
            except psycopg.errors.IntegrityError as error:
                raise ValueError(f"Database constraint violated: {redact(error)}") from error
            except psycopg.errors.QueryCanceled as error:
                raise RuntimeError(f"Database statement timed out: {redact(error)}") from error
            except psycopg.OperationalError as error:
                # The connection was lost (compute suspended, network): one retry
                # on a fresh connection. Searches are reads and writes are
                # idempotent upserts, so a retry cannot apply anything twice.
                if attempt == 1:
                    logger.warning("Postgres connection failed, retrying once: %s", redact(error))
                    continue
                raise RuntimeError(f"Postgres unavailable: {redact(error)}") from error
            except psycopg.Error as error:
                raise RuntimeError(f"Postgres error: {redact(error)}") from error
        raise AssertionError("unreachable")  # pragma: no cover
