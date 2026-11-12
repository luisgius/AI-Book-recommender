"""
Start the API in the cloud configuration in THIS process, with no network, and
report which modules it imported (used by tests/api/test_cloud_profile_imports.py
in a fresh subprocess; not a test module itself).

    python -m tests.api.cloud_profile_driver           # report the imported modules
    python -m tests.api.cloud_profile_driver --block   # the model libraries cannot be imported at all

The caller sets the environment (SEARCH_PROFILE=cloud, EMBEDDING_PROVIDER and
RERANKER_PROVIDER=openrouter, a dummy OPENROUTER_API_KEY, DATABASE_URL to an
unreachable address). The Postgres database is replaced by an in-memory fake
that answers the statements of the cloud adapters, and OpenRouter's HTTP
client by canned embeddings and rerank responses, so every endpoint runs its
real code path: /api/v1/health, /api/v1/search (with MMR), /api/v1/books/{id},
/api/v1/books/{id}/similar, /ui/, /mcp and /api/v1/usage.

Prints one JSON object: the status of each request, the forbidden modules
found in sys.modules and the top-level packages imported.
"""

import importlib.abc
import json
import sys
from datetime import UTC, datetime
from uuid import UUID, uuid5, NAMESPACE_URL

# Packages the cloud profile must never import (heavy model code).
FORBIDDEN = ("torch", "faiss", "sentence_transformers", "transformers", "rank_bm25")


class _BlockModelLibraries(importlib.abc.MetaPathFinder):
    """Makes the forbidden packages unimportable, as in an environment without them."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ModuleNotFoundError(f"No module named '{fullname}' (blocked by the cloud profile test)")
        return None


SCHEMA = "catalog"
DIM = 1024  # the cloud default: qwen/qwen3-embedding-8b at 1024 dimensions, pinned to deepinfra
IDENTITY = "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024"


def _work(i: int) -> dict:
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    return {
        "id": uuid5(NAMESPACE_URL, f"book-{i}"), "source": "test", "source_id": f"b{i}",
        "title": f"Book {i}", "authors": ["Author"], "description": f"A book about topic {i}.",
        "language": "en", "categories": ["Fiction" if i % 2 else "History"],
        "published_date": datetime(2000 + i, 1, 1), "metadata": None, "created_at": stamp, "updated_at": stamp,
    }


WORKS = [_work(i) for i in range(6)]
VECTORS = {w["id"]: [1.0, i / 10.0] + [0.0] * (DIM - 2) for i, w in enumerate(WORKS)}
INFO = {
    "text_configs": [f"{SCHEMA}.simple_unaccent", f"{SCHEMA}.english_unaccent"], "max_texts_per_work": 1,
    "lexical_index": "lakebase_bm25", "vector_index": "lakebase_ann", "embedding_model": IDENTITY,
    "embedding_dimension": DIM, "works_with_embedding": len(WORKS),
}


def _literal(vector) -> str:
    return "[" + ",".join(str(v) for v in vector) + "]"


class FakeDatabase:
    """Answers the statements of the cloud adapters from WORKS (one call = one round trip)."""

    def __init__(self) -> None:
        self.calls = 0

    def query(self, sql, params=None, *, settings=None):
        self.calls += 1
        params = params or {}
        if "IS NOT NULL AS found" in sql:
            return [{"found": True}]
        if f"FROM {SCHEMA}.catalog_info" in sql:
            return [{"key": k, "value": v} for k, v in INFO.items()]
        if "unnest(categories)" in sql:
            return [{"category": c} for c in sorted({c for w in WORKS for c in w["categories"]})]
        if "embedding::text AS e" in sql:
            ids = params.get("ids") or [params.get("id")]
            return [{"id": i, "e": _literal(VECTORS[UUID(str(i))])} for i in ids if UUID(str(i)) in VECTORS]
        if "to_bm25query" in sql:
            return [{**w, "score": 2.0 - i / 10} for i, w in enumerate(WORKS[:3])]
        if "<=>" in sql:
            return [{**w, "distance": i / 100} for i, w in enumerate(WORKS)][: int(params.get("k", 10))]
        if "w.id = %(id)s" in sql:
            return [w for w in WORKS if w["id"] == UUID(str(params["id"]))]
        raise AssertionError(f"unexpected statement: {sql[:120]}")

    def ping(self) -> bool:
        self.calls += 1
        return True

    def execute(self, sql, params=None, *, settings=None):
        raise AssertionError("the API must not write")

    def close(self) -> None:
        pass


def _fake_post(self, path, body, **call_options):
    """OpenRouterHttp.post without network: embeddings and rerank answers."""
    from app.infrastructure.search.openrouter_http import ApiResponse

    # The cloud profile gives the calls a user waits for short deadlines (V17):
    # per call for the query embedding, per client (plus the deadline) for the reranker.
    if path == "embeddings":
        assert call_options == {"timeout_s": 3.0, "max_retries": 1, "deadline_s": 5.0}, call_options
    else:
        assert call_options == {"deadline_s": 5.0}, call_options
        assert (self._timeout_s, self._max_retries) == (3.0, 1)

    if path == "embeddings":
        assert body.get("dimensions") == DIM and body["provider"]["order"] == ["deepinfra"]
        data = [{"index": i, "embedding": [1.0, 0.05] + [0.0] * (DIM - 2)} for i, _ in enumerate(body["input"])]
        return ApiResponse(body={"data": data, "usage": {"prompt_tokens": 7, "cost": 1e-7}}, latency_s=0.01, attempts=1)
    if path == "rerank":
        assert body["model"] == "qwen/qwen3-reranker-8b" and body["provider"]["order"] == ["primeintellect"]
        n = len(body["documents"])
        results = [{"index": i, "relevance_score": 1.0 - i / (n + 1)} for i in reversed(range(n))]
        return ApiResponse(body={"results": results, "usage": {"total_tokens": 50, "cost": 1e-5}},
                           latency_s=0.02, attempts=1)
    raise AssertionError(f"unexpected OpenRouter path {path}")


def _fake_post_first(self, path, legs, *, deadline_s):
    """HedgedPoster.post_first without network: the first leg (DeepInfra) answers at once."""
    from app.infrastructure.search.openrouter_hedge import NOT_SENT, HedgedResponse, LegOutcome

    # The cloud defaults: DeepInfra, Nebius after 1 s, DeepInfra again at 2.5 s, 5 s deadline,
    # every request restricted to Zero Data Retention endpoints
    assert path == "embeddings" and deadline_s == 5.0
    assert [(leg.start_s, leg.label) for leg in legs] == [(0.0, "deepinfra"), (1.0, "nebius"), (2.5, "deepinfra")]
    assert all(leg.body["provider"]["zdr"] is True for leg in legs)
    body = legs[0].body
    assert body.get("dimensions") == DIM and body["provider"]["order"] == ["deepinfra"]
    data = [{"index": i, "embedding": [1.0, 0.05] + [0.0] * (DIM - 2)} for i, _ in enumerate(body["input"])]
    answer = {"data": data, "usage": {"prompt_tokens": 7, "cost": 1e-7}}
    outcomes = [LegOutcome(label="deepinfra", sent=True, started_s=0.0, latency_s=0.01, body=answer)]
    outcomes += [LegOutcome(label=leg.label, error=NOT_SENT) for leg in legs[1:]]
    return HedgedResponse(answer, 0.01, "deepinfra", outcomes)


def _block_network() -> None:
    """Any connection to an address other than this machine fails (the fakes must answer everything)."""
    import socket

    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        if host not in (None, "localhost", "127.0.0.1", "::1", "testserver"):
            raise OSError(f"network blocked in the cloud profile driver: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo


def main(argv) -> int:
    _block_network()
    if "--block" in argv:
        sys.meta_path.insert(0, _BlockModelLibraries())

    from fastapi.testclient import TestClient

    import app.main
    from app.api.v1 import dependencies as deps
    from app.infrastructure.search.openrouter_http import OpenRouterHttp

    from app.infrastructure.search.openrouter_hedge import HedgedPoster

    OpenRouterHttp.post = _fake_post
    HedgedPoster.post_first = _fake_post_first
    deps.load_dotenv = lambda *args, **kwargs: False  # the caller's environment only, never the repository's .env
    deps._postgres_database = FakeDatabase()
    client = TestClient(app.main.app)
    book_id = str(WORKS[0]["id"])
    statuses = {}
    with client:
        statuses["health"] = client.get("/api/v1/health").json()
        search = client.post("/api/v1/search", json={"text": "topic", "max_results": 3, "use_diversification": True})
        statuses["search"] = [search.status_code, len(search.json().get("results", []))]
        statuses["book"] = client.get(f"/api/v1/books/{book_id}").status_code
        similar = client.get(f"/api/v1/books/{book_id}/similar?max_results=3&use_diversification=true")
        statuses["similar"] = [similar.status_code, len(similar.json().get("results", []))]
        statuses["ui"] = client.get("/ui/").status_code
        mcp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
            headers={"Accept": "application/json, text/event-stream", "Host": "127.0.0.1:8000"})
        statuses["mcp"] = mcp.status_code
        statuses["usage"] = client.get("/api/v1/usage").json()
        statuses["assistant"] = type(deps.get_assistant()).__name__  # built, not called (no LLM call)
    print(json.dumps({
        "statuses": statuses,
        "forbidden_imported": sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN),
        "top_level_modules": sorted({m.split(".")[0] for m in sys.modules if not m.startswith("_")}),
        "round_trips": deps._postgres_database.calls,
    }, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
