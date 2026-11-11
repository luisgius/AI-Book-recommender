"""
EmbeddingsStore of the cloud profile: the model encodes, Postgres stores.

Texts are encoded by the configured encoder: OpenRouterEmbeddingEncoder (the
cloud default, no model in the process) or a local sentence-transformers
model through EmbeddingsStoreFaiss, used only as an encoder (no FAISS index
is loaded). Anything with the generate_* methods, get_dimension and
get_model_name works. Document vectors are stored in
<schema>.works.embedding and the ANN index is built there; this store reads
and writes them (get_embeddings_batch answers many ids in one statement),
while PgVectorSearchRepository searches them.
"""

import logging
from typing import Dict, List, Optional, Union
from uuid import UUID

from app.domain.ports import EmbeddingsStore
from app.infrastructure.db.postgres_connection import PostgresDatabase
from app.infrastructure.db.postgres_rows import normalize, parse_vector, vector_literal
from app.infrastructure.db.postgres_schema import build_vector_index, read_catalog_info, validate_schema_name
from app.infrastructure.search.text_encoding import TextEncoder

logger = logging.getLogger(__name__)


class PostgresEmbeddingsStore(EmbeddingsStore):
    """
    Args:
        encoder: Encoder of queries and documents: a TextEncoder such as
            OpenRouterEmbeddingEncoder, or a store whose generate_* methods
            encode (its own index, if any, is not used)
        db: Database
        schema: Catalog schema
        vector_index: ANN index type built by build_index ("lakebase_ann" or "hnsw")
        batch_size: Vectors per UPDATE statement in store_embeddings_batch
    """

    def __init__(
        self,
        encoder: Union[TextEncoder, EmbeddingsStore],
        db: PostgresDatabase,
        schema: str,
        vector_index: str = "lakebase_ann",
        batch_size: int = 100,
    ) -> None:
        self._encoder = encoder
        self._db = db
        self._schema = validate_schema_name(schema)
        self._vector_index = vector_index
        self._batch_size = batch_size
        self._has_vectors = False

    # Encoding: delegated -------------------------------------------------

    def generate_embedding(self, text: str) -> List[float]:
        return self._encoder.generate_embedding(text)

    def generate_query_embedding(self, text: str) -> List[float]:
        return self._encoder.generate_query_embedding(text)

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return self._encoder.generate_embeddings_batch(texts)

    def get_dimension(self) -> int:
        return self._encoder.get_dimension()

    def get_model_name(self) -> str:
        """Model of the encoder (recorded with the vectors by the loader)."""
        getter = getattr(self._encoder, "get_model_name", None)
        return getter() if callable(getter) else "unknown"

    # Storage: Postgres ----------------------------------------------------

    def store_embedding(self, book_id: UUID, embedding: List[float]) -> None:
        self.store_embeddings_batch([book_id], [embedding])

    def store_embeddings_batch(self, book_ids: List[UUID], embeddings: List[List[float]]) -> None:
        """
        Write unit-length vectors to works.embedding (unknown ids are ignored).

        Raises:
            ValueError: On a length or dimension mismatch
        """
        if len(book_ids) != len(embeddings):
            raise ValueError(f"Mismatched lengths: {len(book_ids)} book_ids vs {len(embeddings)} embeddings")
        dimension = self.get_dimension()
        for start in range(0, len(book_ids), self._batch_size):
            ids = book_ids[start:start + self._batch_size]
            vectors = embeddings[start:start + self._batch_size]
            params = {}
            rows = []
            for i, (book_id, vector) in enumerate(zip(ids, vectors)):
                if len(vector) != dimension:
                    raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {len(vector)}")
                params[f"id{i}"] = book_id
                params[f"v{i}"] = vector_literal(normalize(vector))
                rows.append(f"(%(id{i})s::uuid, %(v{i})s::halfvec)")
            self._db.execute(
                f"UPDATE {self._schema}.works w SET embedding = v.embedding "
                f"FROM (VALUES {', '.join(rows)}) AS v(id, embedding) WHERE w.id = v.id",
                params,
            )

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        """The stored (unit-length, half-precision) vector of a work, or None."""
        rows = self._db.query(
            f"SELECT embedding::text AS e FROM {self._schema}.works WHERE id = %(id)s AND embedding IS NOT NULL",
            {"id": book_id},
        )
        return parse_vector(rows[0]["e"]) if rows else None

    def get_embeddings_batch(self, book_ids: List[UUID]) -> Dict[UUID, List[float]]:
        """Stored vectors of several works in ONE statement (one round trip); ids without one are left out."""
        ids = list(dict.fromkeys(book_ids))
        if not ids:
            return {}
        rows = self._db.query(
            f"SELECT id, embedding::text AS e FROM {self._schema}.works "
            f"WHERE id = ANY(%(ids)s::uuid[]) AND embedding IS NOT NULL",
            {"ids": ids},
        )
        return {row["id"]: parse_vector(row["e"]) for row in rows}

    def build_index(self) -> None:
        """(Re)build the ANN index over the stored vectors."""
        build_vector_index(self._db, self._schema, self._vector_index)

    def save_index(self, path: str) -> None:
        raise NotImplementedError("The Postgres vector index is stored in the database, not in a file")

    def load_index(self, path: str) -> None:
        raise NotImplementedError("The Postgres vector index is stored in the database, not in a file")

    def is_ready(self) -> bool:
        """True when the encoder works and the schema holds vectors (checked once)."""
        try:
            self.get_dimension()
            if not self._has_vectors:
                self._has_vectors = bool(read_catalog_info(self._db, self._schema).get("works_with_embedding"))
            return self._has_vectors
        except Exception as error:  # noqa: BLE001 - health check must not raise
            logger.warning("Postgres embeddings store not ready: %s", error)
            return False
