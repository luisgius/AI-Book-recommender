#!/usr/bin/env python3
"""
MCP demo client: list the server's tools and call each one once.

It starts nothing: it connects to an API that is already running and shows
what an AI assistant sees through MCP (the tool list and one result per
tool), with the latency of every call measured on this side.

Usage:
    uvicorn app.main:app --port 8000          # in another terminal
    python -m scripts.mcp_demo --url http://127.0.0.1:8000/mcp

Args:
    --url: MCP endpoint (default http://127.0.0.1:8000/mcp)
    --query: Query for search_books (default "introduction to machine learning with python")
    --max-results: Results per search and similarity call (default 3)
    --mode: "auto" (protocol 2026-07-28, falls back to the handshake on
            older servers) or "legacy" (2025-11-25 handshake)

Only the MCP SDK is used (no import from app/), so the same script works
against a deployed server.
"""

import argparse
import sys
import time
from typing import Any, Awaitable, Callable, Tuple

import anyio
from mcp import Client
from mcp.types import CallToolResult, TextContent

DEFAULT_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_QUERY = "introduction to machine learning with python"


async def timed(call: Callable[[], Awaitable[Any]]) -> Tuple[Any, float]:
    """Run an async call and return its result with the elapsed milliseconds."""
    start = time.perf_counter()
    result = await call()
    return result, (time.perf_counter() - start) * 1000


def shorten(text: str, width: int) -> str:
    """One line of at most `width` characters."""
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 3] + "..."


def error_of(result: CallToolResult) -> str:
    """Text of a tool error (what the model would read)."""
    return " ".join(block.text for block in result.content if isinstance(block, TextContent))


def print_books(hits: list, score_key: str) -> None:
    """One line per book: rank, title, first author, year, language, score."""
    for hit in hits:
        author = hit["authors"][0] if hit["authors"] else "?"
        score = hit.get(score_key)
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        print(f"    {hit['rank']:>2}. {shorten(hit['title'], 60):<60} {shorten(author, 22):<22} "
              f"{hit['year'] or '----'} {hit['language'] or '--'}  {score_key}={score_text}")


async def run_demo(url: str, query: str, max_results: int, mode: str) -> int:
    """Connect, list the tools and call each one; return a process exit code."""
    connect_start = time.perf_counter()
    async with Client(url, mode=mode) as client:
        connect_ms = (time.perf_counter() - connect_start) * 1000
        info = client.server_info
        server = f"{info.name} {info.version}" if info else "unknown server"
        print(f"Connected to {url}")
        print(f"  server: {server} | protocol: {client.protocol_version} | {connect_ms:.0f} ms")

        listed, ms = await timed(client.list_tools)
        print(f"\ntools/list ({ms:.0f} ms)")
        for tool in listed.tools:
            schema = tool.input_schema.get("properties", {})
            required = set(tool.input_schema.get("required", []))
            args = ", ".join(f"{name}{'' if name in required else '?'}" for name in schema)
            read_only = tool.annotations.read_only_hint if tool.annotations else None
            print(f"  - {tool.name}({args})  read-only={read_only}")
            print(f"      {shorten(tool.description or '', 110)}")

        search, ms = await timed(lambda: client.call_tool(
            "search_books", {"query": query, "max_results": max_results}))
        print(f"\nsearch_books(query={query!r}, max_results={max_results})  ({ms:.0f} ms)")
        if search.is_error:
            print(f"  error: {error_of(search)}")
            return 1
        body = search.structured_content
        print(f"  mode={body['search_mode']} reranked={body['reranked']} results={len(body['results'])}")
        print_books(body["results"], "score")
        if not body["results"]:
            print("  no results: nothing to pass to get_book and similar_books")
            return 1
        book_id = body["results"][0]["book_id"]

        record, ms = await timed(lambda: client.call_tool("get_book", {"book_id": book_id}))
        print(f"\nget_book(book_id={book_id})  ({ms:.0f} ms)")
        if record.is_error:
            print(f"  error: {error_of(record)}")
            return 1
        book = record.structured_content
        print(f"  {book['title']} | {', '.join(book['authors'])} | {book['published_date'] or 'no date'} "
              f"| {book['language']} | {book['source']}:{book['source_id']}")
        print(f"  publisher={book['publisher']} isbn13={book['isbn13']} pages={book['page_count']}")
        print(f"  description ({len(book['description'] or '')} chars): {shorten(book['description'] or '', 110)}")

        similar, ms = await timed(lambda: client.call_tool(
            "similar_books", {"book_id": book_id, "max_results": max_results}))
        print(f"\nsimilar_books(book_id={book_id}, max_results={max_results})  ({ms:.0f} ms)")
        if similar.is_error:
            print(f"  error: {error_of(similar)}")
            return 1
        print_books(similar.structured_content["results"], "similarity")

        missing, ms = await timed(lambda: client.call_tool(
            "get_book", {"book_id": "00000000-0000-0000-0000-000000000000"}))
        print(f"\nget_book(unknown id)  ({ms:.0f} ms)  is_error={missing.is_error}")
        print(f"  {shorten(error_of(missing), 120)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="List the MCP tools of a running server and call each once.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"MCP endpoint (default {DEFAULT_URL})")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="query for search_books")
    parser.add_argument("--max-results", type=int, default=3, help="results per call (1-20, default 3)")
    parser.add_argument("--mode", choices=["auto", "legacy"], default="auto",
                        help="protocol negotiation: auto (2026-07-28) or legacy (2025-11-25)")
    args = parser.parse_args()
    try:
        sys.exit(anyio.run(run_demo, args.url, args.query, args.max_results, args.mode))
    except Exception as e:  # connection refused, 421/429 from the server, ...
        print(f"Could not complete the demo against {args.url}: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
