"""
End-to-end test of the MCP endpoint against the real catalog.

The real FastAPI application (app.main) is started in-process (its lifespan
runs the MCP transport) and a real MCP client from the SDK talks to /mcp over
HTTP through httpx2's ASGI transport: no port, no subprocess. The tools run
through the production wiring (app/api/v1/dependencies.py) on the shipped
177-book catalog (data/catalog.db) and indexes (data/indexes, embedding
model all-MiniLM-L6-v2, loaded from the local cache).

The LLM client of that wiring is replaced by a spy that records any use:
the MCP tools must never call the LLM. The reranker is left off.

Run with: python -m pytest tests/integration/test_mcp_endpoint.py -q
"""

from pathlib import Path
from uuid import UUID

import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from app.api.v1 import dependencies as deps
from app.infrastructure.config.settings import EmbeddingSettings, LLMSettings, RerankerSettings, SearchProfileSettings
from app.main import app, mcp_endpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DB = REPO_ROOT / "data" / "catalog.db"
INDEXES_DIR = REPO_ROOT / "data" / "indexes"
BASE_URL = "http://127.0.0.1:8000"  # Host header accepted by the default allowlist

pytestmark = pytest.mark.skipif(
    not (CATALOG_DB.exists() and (INDEXES_DIR / "faiss_index").exists()),
    reason="shipped catalog or indexes not present",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class SpyLLMClient:
    """Records every attribute used; the MCP tools must never use the LLM."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"an MCP tool reached the LLM client ({name})")
        return record


@pytest.fixture
def real_wiring(monkeypatch):
    """
    Production dependency wiring on the shipped catalog, with a spy LLM.

    Settings are fixed here so a local .env (LLM provider, embedding model,
    reranker) cannot change what the test measures.
    """
    deps.reset_dependencies()
    spy = SpyLLMClient()
    monkeypatch.setattr(deps, "DB_PATH", CATALOG_DB)
    monkeypatch.setattr(deps, "INDEXES_DIR", INDEXES_DIR)
    monkeypatch.setattr(deps, "_profile_settings", SearchProfileSettings())  # local, vector candidates
    monkeypatch.setattr(deps, "_embedding_settings", EmbeddingSettings())
    monkeypatch.setattr(deps, "_llm_settings", LLMSettings(provider="none"))
    monkeypatch.setattr(deps, "_llm_client", spy)
    monkeypatch.setattr(deps, "_llm_initialized", True)
    monkeypatch.setattr(deps, "_reranker_settings", RerankerSettings())
    monkeypatch.setattr(deps, "_reranker", None)
    monkeypatch.setattr(deps, "_reranker_initialized", True)
    yield spy
    deps.reset_dependencies()


@pytest.mark.anyio
@pytest.mark.parametrize("mode, protocol", [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
async def test_mcp_client_lists_and_calls_every_tool_over_http(real_wiring, mode, protocol):
    assert mcp_endpoint is not None, "MCP is disabled in this environment (MCP_ENABLED=false)"
    catalog = deps.get_catalog_repository()

    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE_URL,
                                      timeout=120) as http:
            async with Client(streamable_http_client(f"{BASE_URL}/mcp", http_client=http), mode=mode) as client:
                assert client.protocol_version == protocol
                assert client.server_info.name == "ai-book-recommender"

                tools = {t.name for t in (await client.list_tools()).tools}
                assert tools == {"search_books", "get_book", "similar_books"}

                search = await client.call_tool(
                    "search_books", {"query": "introduction to machine learning with python", "max_results": 5})
                assert not search.is_error
                hits = search.structured_content["results"]
                assert 1 <= len(hits) <= 5
                # The default candidate mode: vector search, BM25 only as the fallback
                assert search.structured_content["search_mode"] == "vector_only"
                for hit in hits:
                    assert hit["retrieved_by"] == "vector" and hit["keyword_score"] is None
                    assert catalog.get_by_id(UUID(hit["book_id"])) is not None  # only catalog books
                book_id = hits[0]["book_id"]

                spanish = await client.call_tool(
                    "search_books", {"query": "novela histórica", "language": "es", "max_results": 5})
                assert spanish.structured_content["results"]
                assert {hit["language"] for hit in spanish.structured_content["results"]} == {"es"}

                record = await client.call_tool("get_book", {"book_id": book_id})
                assert not record.is_error
                assert record.structured_content["book_id"] == book_id
                assert record.structured_content["title"] == hits[0]["title"]

                similar = await client.call_tool("similar_books", {"book_id": book_id, "max_results": 3})
                assert not similar.is_error
                similar_ids = [hit["book_id"] for hit in similar.structured_content["results"]]
                assert len(similar_ids) == 3 and book_id not in similar_ids

                missing = await client.call_tool("get_book", {"book_id": "00000000-0000-0000-0000-000000000000"})
                assert missing.is_error and "No book with id" in missing.content[0].text

    assert real_wiring.calls == []
