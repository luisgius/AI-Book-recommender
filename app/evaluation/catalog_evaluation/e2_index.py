"""
E2: a local exact index of baai/bge-m3 vectors for the 50,000 works.

The per-work text is the loader's: Book.get_searchable_text() of the same
SQLite source (scripts/load_postgres.py encodes exactly that), and `parity`
checks that text_hash, the hash of that text, is equal to the one Neon
stores for every work. Vectors are computed through OpenRouter in shards of
SHARD_SIZE works (works sorted by id), each saved as float32 .npy with its
id list before the next starts, under data/catalog_deploy/bge_m3/ (git
ignored, about 200 MB). Search is a brute-force cosine over the normalised
matrix (exact), ties broken by work id.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

SHARD_SIZE = 2000
EMBED_WORKERS = 2
BACKOFF_S = (10, 30, 60, 120, 240)  # waits before re-sending a batch the provider refused (HTTP 429)


def shard_paths(directory: Path, number: int) -> Tuple[Path, Path]:
    return directory / f"shard_{number:03d}.npy", directory / f"shard_{number:03d}.ids.json"


def build_vectors(books: Sequence[Any], encoder: Any, directory: Path, *, check: Callable[[], None],
                  log: Callable[[str], None] = print) -> int:
    """Encode every work not yet stored; returns the number of shards written in this run."""
    directory.mkdir(parents=True, exist_ok=True)
    ordered = sorted(books, key=lambda b: str(b.id))
    written = 0
    for number, start in enumerate(range(0, len(ordered), SHARD_SIZE)):
        vectors_path, ids_path = shard_paths(directory, number)
        if vectors_path.exists() and ids_path.exists():
            continue
        check()
        chunk = ordered[start:start + SHARD_SIZE]
        texts = [book.get_searchable_text() for book in chunk]
        batches = encoder.batches(texts)

        def encode(batch: List[str]) -> List[List[float]]:
            for wait in BACKOFF_S + (None,):
                try:
                    return encoder.generate_embeddings_batch(batch)
                except Exception as error:  # noqa: BLE001 - provider overload: wait and re-send
                    if wait is None:
                        raise
                    log(f"E2 batch failed ({type(error).__name__}); retrying in {wait} s")
                    time.sleep(wait)
            raise RuntimeError("unreachable")

        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            parts = list(pool.map(encode, batches))
        vectors = np.asarray([v for part in parts for v in part], dtype=np.float32)
        if vectors.shape[0] != len(chunk):
            raise RuntimeError(f"shard {number}: {vectors.shape[0]} vectors for {len(chunk)} works")
        np.save(vectors_path.with_suffix(".tmp.npy"), vectors)
        vectors_path.with_suffix(".tmp.npy").rename(vectors_path)
        ids_path.write_text(json.dumps([str(b.id) for b in chunk]), encoding="utf-8")
        written += 1
        log(f"E2 shard {number}: {len(chunk)} works")
    return written


class ExactIndex:
    """Exact cosine search over normalised vectors (numpy)."""

    def __init__(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError("one id per vector")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.ids = list(ids)
        self.matrix = (vectors / norms).astype(np.float32)

    @classmethod
    def load(cls, directory: Path) -> "ExactIndex":
        ids: List[str] = []
        parts = []
        number = 0
        while True:
            vectors_path, ids_path = shard_paths(directory, number)
            if not vectors_path.exists():
                break
            parts.append(np.load(vectors_path))
            ids += json.loads(ids_path.read_text(encoding="utf-8"))
            number += 1
        if not parts:
            raise FileNotFoundError(f"no E2 shards in {directory}")
        return cls(ids, np.vstack(parts))

    def search(self, query: Sequence[float], k: int) -> List[Tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.matrix @ q
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k] if k < len(self.ids) else np.arange(len(self.ids))
        ranked = sorted(top, key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in ranked]
