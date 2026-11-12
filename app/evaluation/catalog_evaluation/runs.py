"""
Stage 2: frozen query embeddings, the E2 local index and the run files.

Run files: one JSON Lines file per system under <data>/runs/, one line per
query, written as soon as the query is done (resumable per system and
query): {"query_id", "ids": [100 work ids, best first], "scores": [...],
"timings_ms": {stage: ms}, "error": null | text, "extra": {...}}.

The query embedding is frozen (protocol section 2): each query is encoded
once per model and the stored vector is reused by every system, so the
contrasts are not mixed with the provider's run-to-run noise.

Works are hydrated from the local copy of the catalog (the loader's SQLite
source), which `parity` verifies to hold the same ids, the same embedded
text (text_hash) and the same content (content_hash) as the Neon schema.
"""

import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.evaluation.catalog_evaluation.design import RRF_K

Run = Dict[str, List[str]]  # query_id -> ranked work ids


# =============================================================================
# Pure helpers
# =============================================================================


def rrf_fuse(lexical: Sequence[str], vector: Sequence[str], k: int = RRF_K) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion of two ranked id lists, exactly as
    SearchService._fuse_results_rrf: score = sum 1 / (k + rank), books
    inserted lexical first, then the vector-only ones, and a stable sort by
    score (ties keep that insertion order).
    """
    scores: Dict[str, float] = {}
    for rank, book_id in enumerate(lexical, start=1):
        scores[book_id] = scores.get(book_id, 0.0) + 1.0 / (k + rank)
    for rank, book_id in enumerate(vector, start=1):
        scores[book_id] = scores.get(book_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda book_id: scores[book_id], reverse=True)
    return [(book_id, scores[book_id]) for book_id in ordered]


def apply_head_order(ids: Sequence[str], head_order: Sequence[int], depth: int) -> List[str]:
    """The list with its first len(head_order) items re-ordered (indexes into the head), tail unchanged."""
    head = list(ids[:depth])
    if sorted(head_order) != list(range(len(head))):
        raise ValueError("head_order must be a permutation of the head")
    return [head[i] for i in head_order] + list(ids[depth:])


def encode_vector(vector: Sequence[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def decode_vector(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32)


def model_slug(model: str) -> str:
    return model.replace("/", "__")


# =============================================================================
# JSON Lines stores (append, flush, sync; a partial last line is ignored)
# =============================================================================


class JsonlStore:
    """Append-only JSON Lines records keyed by one field, thread-safe."""

    def __init__(self, path: Path, key: str = "query_id") -> None:
        self.path = path
        self.key = key
        self._lock = threading.Lock()
        self.records: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.records[record[key]] = record

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.records.get(key)

    def add(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records[record[self.key]] = dict(record)


def runs_dir(data_dir: Path) -> Path:
    return data_dir / "runs"


def run_store(data_dir: Path, system: str) -> JsonlStore:
    return JsonlStore(runs_dir(data_dir) / f"{system}.jsonl")


def load_runs(data_dir: Path, systems: Iterable[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{system: {query_id: record}} from the run files."""
    return {system: run_store(data_dir, system).records for system in systems}


def embedding_store(data_dir: Path, model: str) -> JsonlStore:
    return JsonlStore(data_dir / "query_embeddings" / f"{model_slug(model)}.jsonl")


# =============================================================================
# The local catalog copy
# =============================================================================


class LocalCatalog:
    """Books of the loader's SQLite source, hydrated by id (read-only)."""

    def __init__(self, path: Path) -> None:
        from scripts.load_postgres import ReadOnlySqliteCatalog

        self._repo = ReadOnlySqliteCatalog(path)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def books(self, ids: Sequence[str]) -> List[Any]:
        from uuid import UUID

        missing = [i for i in ids if i not in self._cache]
        if missing:
            found = self._repo.get_many([UUID(i) for i in missing])
            with self._lock:
                for book_id, book in found.items():
                    self._cache[str(book_id)] = book
        return [self._cache[i] for i in ids]

    def iter_all(self):
        return self._repo.iter_all()


def check_parity(db: Any, schema: str, catalog: LocalCatalog) -> Dict[str, Any]:
    """Same ids, text_hash and content_hash in the local copy and in the Neon schema (no vectors read)."""
    from app.infrastructure.db.postgres_rows import content_hash, text_hash

    local = {str(b.id): (text_hash(b), content_hash(b)) for b in catalog.iter_all()}
    rows = db.query(f"SELECT id::text AS id, text_hash, content_hash FROM {schema}.works")
    remote = {r["id"]: (r["text_hash"], r["content_hash"]) for r in rows}
    common = set(local) & set(remote)
    return {
        "local_works": len(local), "neon_works": len(remote), "same_ids": set(local) == set(remote),
        "text_hash_equal": sum(local[i][0] == remote[i][0] for i in common),
        "content_hash_equal": sum(local[i][1] == remote[i][1] for i in common),
    }
