"""
SQLite storage for the state of a multi-source catalog build.

Two small pieces, kept outside the ``books`` table:

- ``SqliteDuplicateKeyIndex`` implements the DuplicateKeyIndex port: a table
  of (match key, book id) pairs used to find the work an incoming record
  must be merged into. It lives in its own file next to the catalog
  (``match_keys.db``) because it is only needed while building.
- ``SqliteCatalogBuildState`` records, inside the catalog database, which
  sources have been ingested and their ingestion counters, so the ``ingest``
  step can resume (finished sources are skipped) and the ``report`` step can
  show what was read and rejected per source.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from app.domain.ports import DuplicateKeyIndex


class SqliteDuplicateKeyIndex(DuplicateKeyIndex):
    """DuplicateKeyIndex backed by one SQLite table with a composite primary key."""

    LOOKUP_CHUNK_SIZE = 900

    def __init__(self, db_path: Path) -> None:
        """
        Args:
            db_path: SQLite file for the index (created if missing)
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS match_keys ("
            " key TEXT NOT NULL, book_id TEXT NOT NULL, seq INTEGER NOT NULL,"
            " PRIMARY KEY (key, book_id)) WITHOUT ROWID"
        )
        self._conn.commit()
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM match_keys").fetchone()
        self._seq = int(row[0])

    def lookup(self, keys: List[str]) -> Dict[str, List[UUID]]:
        """Ids registered under each key, in registration order."""
        found: Dict[str, List[Tuple[int, UUID]]] = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), self.LOOKUP_CHUNK_SIZE):
            chunk = unique[start:start + self.LOOKUP_CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk))
            for key, book_id, seq in self._conn.execute(
                f"SELECT key, book_id, seq FROM match_keys WHERE key IN ({placeholders})", chunk
            ):
                found.setdefault(key, []).append((seq, UUID(book_id)))
        return {key: [book_id for _, book_id in sorted(pairs)] for key, pairs in found.items()}

    def add(self, entries: List[Tuple[str, UUID]]) -> None:
        """Register (key, id) pairs; pairs already present are left unchanged."""
        rows = []
        for key, book_id in entries:
            self._seq += 1
            rows.append((key, str(book_id), self._seq))
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO match_keys (key, book_id, seq) VALUES (?, ?, ?)", rows
            )

    def count(self) -> int:
        """Number of (key, id) pairs."""
        return int(self._conn.execute("SELECT COUNT(*) FROM match_keys").fetchone()[0])

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()


class SqliteCatalogBuildState:
    """Per-source ingestion status and counters, stored in the catalog database."""

    def __init__(self, db_path: Path) -> None:
        """
        Args:
            db_path: The catalog SQLite file
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS catalog_build_sources ("
                " source TEXT PRIMARY KEY, position INTEGER NOT NULL, status TEXT NOT NULL,"
                " stats TEXT, license TEXT, started_at TEXT, finished_at TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def status(self, source: str) -> Optional[str]:
        """'running', 'done' or None if the source was never ingested."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM catalog_build_sources WHERE source = ?", (source,)
            ).fetchone()
        return row["status"] if row else None

    def mark_started(self, source: str, position: int) -> None:
        """Record that ingestion of a source began (position = ingestion order)."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO catalog_build_sources (source, position, status, started_at)"
                " VALUES (?, ?, 'running', ?)"
                " ON CONFLICT(source) DO UPDATE SET status='running', position=excluded.position,"
                " started_at=excluded.started_at, finished_at=NULL",
                (source, position, now),
            )

    def mark_done(self, source: str, stats: Dict, license_info: Dict) -> None:
        """Store the final counters and licence of a source."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE catalog_build_sources SET status='done', stats=?, license=?, finished_at=?"
                " WHERE source = ?",
                (json.dumps(stats), json.dumps(license_info), now, source),
            )

    def all_sources(self) -> List[Dict]:
        """Every recorded source with its status, counters and licence, in ingestion order."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM catalog_build_sources ORDER BY position"
            ).fetchall()
        return [
            {
                "source": row["source"],
                "status": row["status"],
                "stats": json.loads(row["stats"]) if row["stats"] else None,
                "license": json.loads(row["license"]) if row["license"] else None,
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
            }
            for row in rows
        ]
