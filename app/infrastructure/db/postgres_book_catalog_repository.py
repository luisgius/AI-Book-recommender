"""
Postgres implementation of the BookCatalogRepository port (cloud profile).

Books are stored in <schema>.works; their lexical documents (one per
description language) in <schema>.work_texts. Saving a book keeps both in
step and resets its embedding only when the text it was computed from
changed, so reloading an unchanged catalog writes nothing and re-embeds
nothing (idempotent, resumable loads).

Semantics follow SqliteBookCatalogRepository: (source, source_id) is unique
when source_id is set (a clash with another id raises ValueError), get_all()
orders by created_at descending and iter_all() by id.
"""

import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import UUID

from app.domain.entities import Book
from app.domain.ports import BookCatalogRepository
from app.infrastructure.db.postgres_connection import PostgresDatabase
from app.infrastructure.db.postgres_rows import (
    SIMPLE_CONFIG,
    WORK_COLUMNS,
    WORK_INSERT_COLUMNS,
    book_to_work_params,
    lexical_header_sql,
    normalize,
    row_to_book,
    text_rows,
    vector_literal,
)
from app.infrastructure.db.postgres_schema import text_config_names, validate_schema_name

logger = logging.getLogger(__name__)


class PostgresBookCatalogRepository(BookCatalogRepository):
    """
    Catalog of books in Postgres.

    Args:
        db: Database (pooled connection string for the API, direct for loads)
        schema: Catalog schema, created beforehand by postgres_schema.apply_schema
        store_text_bodies: Keep each description's text in work_texts.body
            after its tsvector is computed (default). False stores NULL
            there: the searches read only the tsvector, and the texts stay
            in works (description and metadata), so a deployed catalog
            saves the space of a second copy (V17).
    """

    SAVE_CHUNK_SIZE = 500
    """Books per write transaction in save_many"""

    def __init__(self, db: PostgresDatabase, schema: str, *, store_text_bodies: bool = True) -> None:
        self._db = db
        self._schema = validate_schema_name(schema)
        self._store_text_bodies = store_text_bodies
        self._configs: Optional[Set[str]] = None
        self._categories: Optional[List[str]] = None  # cached by get_categories, reset by writes

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _select(self, where: str, params: Dict[str, Any], tail: str = "") -> List[Book]:
        rows = self._db.query(f"SELECT {WORK_COLUMNS} FROM {self._schema}.works w WHERE {where} {tail}", params)
        return [row_to_book(row) for row in rows]

    def get_by_id(self, book_id: UUID) -> Optional[Book]:
        books = self._select("w.id = %(id)s", {"id": book_id})
        return books[0] if books else None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[Book]:
        books = self._select("w.source = %(source)s AND w.source_id = %(sid)s", {"source": source, "sid": source_id})
        return books[0] if books else None

    def get_all(self, limit: Optional[int] = None) -> List[Book]:
        tail = "ORDER BY w.created_at DESC, w.id" + (" LIMIT %(limit)s" if limit is not None else "")
        return self._select("TRUE", {"limit": limit} if limit is not None else {}, tail)

    def get_many(self, book_ids: List[UUID]) -> Dict[UUID, Book]:
        ids = list(dict.fromkeys(book_ids))
        if not ids:
            return {}
        return {book.id: book for book in self._select("w.id = ANY(%(ids)s::uuid[])", {"ids": ids})}

    def iter_all(self, batch_size: int = 1000) -> Iterator[Book]:
        """Stream every book ordered by id with keyset pagination (one batch in memory)."""
        last: Optional[UUID] = None
        while True:
            if last is None:
                batch = self._select("TRUE", {"n": batch_size}, "ORDER BY w.id LIMIT %(n)s")
            else:
                batch = self._select("w.id > %(last)s", {"last": last, "n": batch_size}, "ORDER BY w.id LIMIT %(n)s")
            if not batch:
                return
            yield from batch
            last = batch[-1].id

    def all_ids(self) -> List[UUID]:
        """Every work id (infrastructure helper for the loader's pruning step)."""
        return [row["id"] for row in self._db.query(f"SELECT id FROM {self._schema}.works ORDER BY id")]

    def count(self) -> int:
        return int(self._db.query(f"SELECT count(*) AS n FROM {self._schema}.works")[0]["n"])

    def get_categories(self) -> List[str]:
        """
        Every distinct category, sorted: one aggregate statement, cached.

        The API reads it once at startup (LLM filter validation), so no book
        is loaded. The cache is dropped when this repository writes; changes
        made by another process (the loader) are seen after a restart.
        """
        if self._categories is None:
            rows = self._db.query(
                f"SELECT DISTINCT unnest(categories) AS category FROM {self._schema}.works"
            )
            self._categories = sorted(str(row["category"]) for row in rows if row["category"] is not None)
        return list(self._categories)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def delete(self, book_id: UUID) -> bool:
        """Delete a book (its texts go with it: ON DELETE CASCADE)."""
        self._categories = None
        return self._db.execute(f"DELETE FROM {self._schema}.works WHERE id = %(id)s", {"id": book_id}) > 0

    def delete_many(self, book_ids: List[UUID]) -> int:
        """Delete several books; returns how many existed (loader helper)."""
        if not book_ids:
            return 0
        self._categories = None
        return self._db.execute(f"DELETE FROM {self._schema}.works WHERE id = ANY(%(ids)s::uuid[])", {"ids": book_ids})

    def save(self, book: Book) -> None:
        self.save_many([book])

    def save_many(self, books: List[Book]) -> None:
        """Save books in transactions of SAVE_CHUNK_SIZE (see upsert_many)."""
        self.upsert_many(books)

    def upsert_many(self, books: List[Book], embeddings: Optional[Mapping[UUID, Sequence[float]]] = None) -> int:
        """
        Insert new books and update changed ones; unchanged books are skipped.

        Each chunk is one transaction: the works rows, their work_texts rows
        with their tsvectors, and the catalog facts the lexical search reads
        (configurations in use, most texts per work). Every row is written
        once, in its final form (no insert-then-update), so a load leaves no
        dead rows behind.

        Args:
            books: Books to save
            embeddings: Optional document vectors by book id (loader helper,
                V18): a new work is inserted with its vector, and a work whose
                text changed gets the new one, instead of a NULL embedding
                filled by a second write. A work whose text is unchanged keeps
                its stored vector.

        Returns:
            Number of books inserted or changed

        Raises:
            ValueError: If (source, source_id) belongs to another id
            RuntimeError: On a database error
        """
        changed = 0
        for start in range(0, len(books), self.SAVE_CHUNK_SIZE):
            changed += self._upsert_chunk(books[start:start + self.SAVE_CHUNK_SIZE], embeddings or {})
        return changed

    def _upsert_chunk(self, books: List[Book], embeddings: Mapping[UUID, Sequence[float]]) -> int:
        books = list({book.id: book for book in books}.values())  # last one wins, like an upsert
        if not books:
            return 0
        params = [book_to_work_params(book) for book in books]
        stored = {
            row["id"]: row["content_hash"]
            for row in self._db.query(
                f"SELECT id, content_hash FROM {self._schema}.works WHERE id = ANY(%(ids)s::uuid[])",
                {"ids": [p["id"] for p in params]},
            )
        }
        todo = [(book, p) for book, p in zip(books, params) if stored.get(p["id"]) != p["content_hash"]]
        if not todo:
            return 0
        sql, values = self._write_script([book for book, _ in todo], [p for _, p in todo], embeddings)
        self._categories = None
        self._db.execute(sql, values)
        return len(todo)

    def _available_configs(self) -> Set[str]:
        if self._configs is None:
            self._configs = text_config_names(self._db, self._schema)
            if SIMPLE_CONFIG not in self._configs:
                raise RuntimeError(
                    f"Schema '{self._schema}' has no text-search configurations: apply the schema first "
                    f"(python -m scripts.load_postgres --schema {self._schema} --migrate-only)"
                )
        return self._configs

    def _write_script(
        self,
        books: List[Book],
        params: List[Dict[str, Any]],
        embeddings: Optional[Mapping[UUID, Sequence[float]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """The statements of one write transaction and their parameters."""
        s = self._schema
        embeddings = embeddings or {}
        values: Dict[str, Any] = {}
        work_rows: List[str] = []
        for i, p in enumerate(params):
            placeholders = []
            for column in WORK_INSERT_COLUMNS:
                values[f"{column}_{i}"] = p[column]
                placeholders.append(f"%({column}_{i})s")
            vector = embeddings.get(p["id"])
            values[f"embedding_{i}"] = vector_literal(normalize(vector)) if vector is not None else None
            placeholders.append(f"%(embedding_{i})s::halfvec")
            work_rows.append("(" + ", ".join(placeholders) + ")")

        texts: List[str] = []
        configs_used: Set[str] = set()
        max_texts = 1
        for i, book in enumerate(books):
            rows = text_rows(book, s, self._available_configs())
            max_texts = max(max_texts, len(rows))
            for j, row in enumerate(rows):
                values[f"tw_{i}_{j}"] = book.id
                values[f"tl_{i}_{j}"] = row.language
                values[f"tc_{i}_{j}"] = row.ts_config
                values[f"tb_{i}_{j}"] = row.body
                texts.append(f"(%(tw_{i}_{j})s::uuid, %(tl_{i}_{j})s::text, "
                             f"%(tc_{i}_{j})s::text, %(tb_{i}_{j})s::text)")
                configs_used.add(row.ts_config)
        values["ids"] = [p["id"] for p in params]
        values["configs_used"] = sorted(configs_used | {f"{s}.{SIMPLE_CONFIG}"})
        values["max_texts"] = max_texts

        # created_at keeps its first value, as in the SQLite catalog. A work
        # whose input text is unchanged keeps its stored embedding (or takes
        # the given one if it had none); a changed text takes the given vector,
        # or NULL for the loader to fill. Each text row is inserted with its
        # tsvector already computed (title, authors and categories of the
        # works row written just above, plus its description); without stored
        # bodies the text is not written at all.
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in WORK_INSERT_COLUMNS if c not in ("id", "created_at"))
        body = "v.body" if self._store_text_bodies else "NULL"
        sql = f"""
INSERT INTO {s}.works AS cur ({", ".join(WORK_INSERT_COLUMNS)}, embedding)
VALUES {", ".join(work_rows)}
ON CONFLICT (id) DO UPDATE SET {updates},
  embedding = CASE WHEN cur.text_hash = EXCLUDED.text_hash THEN coalesce(cur.embedding, EXCLUDED.embedding)
                   ELSE EXCLUDED.embedding END;
DELETE FROM {s}.work_texts WHERE work_id = ANY(%(ids)s::uuid[]);
INSERT INTO {s}.work_texts (work_id, language, ts_config, body, tsv)
SELECT v.work_id, v.language, v.ts_config, {body},
       to_tsvector('{s}.{SIMPLE_CONFIG}'::regconfig, {lexical_header_sql("w")})
       || to_tsvector(v.ts_config::regconfig, coalesce(v.body, ''))
FROM (VALUES {", ".join(texts)}) AS v(work_id, language, ts_config, body)
JOIN {s}.works w ON w.id = v.work_id;
INSERT INTO {s}.catalog_info AS info (key, value)
VALUES ('text_configs', to_jsonb(%(configs_used)s::text[])), ('max_texts_per_work', to_jsonb(%(max_texts)s::int))
ON CONFLICT (key) DO UPDATE SET value = CASE
  WHEN info.key = 'text_configs' THEN (
    SELECT to_jsonb(array_agg(DISTINCT name ORDER BY name)) FROM (
      SELECT jsonb_array_elements_text(info.value) AS name
      UNION SELECT jsonb_array_elements_text(EXCLUDED.value)) AS names)
  ELSE to_jsonb(GREATEST((info.value)::int, (EXCLUDED.value)::int))
END;
"""
        return sql, values
