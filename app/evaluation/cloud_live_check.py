"""
Live check of the cloud profile (task V13): the API, as it would be deployed,
against a Neon schema loaded with the cloud embedding model.

The API runs in another process in the cloud configuration
(SEARCH_PROFILE=cloud: Postgres BM25 and vector search on Neon, query
embeddings and re-ranking by OpenRouter with the defaults chosen in
data/evaluation/API_MODELS.md, RRF in the unchanged SearchService) with
USAGE_ENDPOINT_ENABLED=true. This module talks to it over HTTP only:

1. Every endpoint answers: /api/v1/health, /api/v1/search, /api/v1/books/{id},
   /api/v1/books/{id}/similar (with and without MMR), /ui/, /mcp and
   /api/v1/usage.
2. The 44 test queries go through POST /api/v1/search (10 results, no LLM
   step), one at a time after one warm-up query. Per query: the client's
   wall clock, the server's search time (latency_ms of the response), and
   from the usage endpoint read before and after each search the time of the
   embedding call, of the rerank call and their reported cost. Database
   search = server time - embedding - re-ranking (it also holds the RRF
   fusion and the filters, well under a millisecond).
3. nDCG@10 of the cloud stack on the V13 judgments, next to the local
   reference hybrid_norm_e5_bge (rankings of results_bm25_padding.json), as
   a descriptive paired difference. The (query, book) pairs of the cloud
   top 10 still unjudged are graded by the configured judge (cached) and
   added to relevance_judgments_extended_v13.json.

Everything here is descriptive: no contrast was declared in advance.

CLI (from the repository root, the API already running):

    python -m app.evaluation.cloud_live_check run --base-url http://127.0.0.1:8765
    python -m app.evaluation.cloud_live_check report   # tables from the saved rankings, no API

Outputs: data/evaluation/results_cloud_live_check.json and
data/evaluation/CLOUD_LIVE_CHECK.md.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

import httpx

from app.evaluation.bm25_padding import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_QUERIES_PATH,
    DEFAULT_V13_JUDGMENTS,
    list_lengths,
    not_relevant_share,
)
from app.evaluation.comparison import PRIMARY_METRIC, _git, _sha256, compare_pairs, compare_systems
from app.evaluation.retrieval_ablation import DEPTH, latency_summary, require_judged, resolve_grades

DEFAULT_PADDING_JSON = "data/evaluation/results_bm25_padding.json"
DEFAULT_OUTPUT_JSON = "data/evaluation/results_cloud_live_check.json"
DEFAULT_OUTPUT_MD = "data/evaluation/CLOUD_LIVE_CHECK.md"

CLOUD = "cloud_stack"
REFERENCE = "hybrid_norm_e5_bge"
LOCAL_SYSTEMS = ("vec_e5", "hybrid_norm_e5_nopad", "hybrid_norm_e5_bge", "hybrid_norm_e5_bge_nopad")


# =============================================================================
# Talking to the running API
# =============================================================================


def _api_totals(usage: Mapping[str, Any]) -> Dict[str, float]:
    """Cumulative latency (s), calls and reported cost (USD) of the embedding and rerank calls."""
    api = usage.get("api") or {}
    out: Dict[str, float] = {}
    for operation in ("embedding", "rerank"):
        row = api.get(operation) or {}
        out[f"{operation}_s"] = float(row.get("total_latency_s") or 0.0)
        out[f"{operation}_calls"] = float(row.get("calls") or 0)
        out[f"{operation}_cost"] = float(row.get("reported_cost_usd") or 0.0)
    return out


def check_endpoints(client: httpx.Client, book_id: str) -> Dict[str, Any]:
    """Status of every endpoint (and the health payload)."""
    out: Dict[str, Any] = {"health": client.get("/api/v1/health").json()}
    out["ui"] = client.get("/ui/").status_code
    out["book"] = client.get(f"/api/v1/books/{book_id}").status_code
    for name, params in (("similar", {"max_results": 10}),
                         ("similar_mmr", {"max_results": 10, "use_diversification": "true"})):
        response = client.get(f"/api/v1/books/{book_id}/similar", params=params)
        out[name] = {"status": response.status_code, "results": len(response.json().get("results", []))}
    mmr = client.post("/api/v1/search", json={"text": "novela historica", "max_results": 10,
                                               "use_diversification": True})
    out["search_mmr"] = {"status": mmr.status_code, "results": len(mmr.json().get("results", [])),
                         "degraded": mmr.json().get("degraded")}
    mcp = client.post("/mcp", headers={"Accept": "application/json, text/event-stream"}, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "v13-live-check", "version": "1"}}})
    out["mcp_initialize"] = mcp.status_code
    out["usage"] = client.get("/api/v1/usage").status_code
    return out


def run_queries(client: httpx.Client, texts: Mapping[str, str], warmup_text: str,
                log: Callable[[str], None]) -> Dict[str, Any]:
    """The test queries through POST /api/v1/search, with the per-call split from the usage endpoint."""

    def one(text: str) -> Dict[str, Any]:
        before = _api_totals(client.get("/api/v1/usage").json())
        started = time.perf_counter()
        response = client.post("/api/v1/search", json={"text": text, "max_results": DEPTH})
        wall_ms = (time.perf_counter() - started) * 1000
        after = _api_totals(client.get("/api/v1/usage").json())
        response.raise_for_status()
        body = response.json()
        embed_ms = (after["embedding_s"] - before["embedding_s"]) * 1000
        rerank_ms = (after["rerank_s"] - before["rerank_s"]) * 1000
        server_ms = float(body.get("latency_ms") or 0.0)
        return {
            "ids": [r["book"]["id"] for r in body["results"]],
            "degraded": body.get("degraded"),
            "wall_ms": wall_ms,
            "server_ms": server_ms,
            "embedding_ms": embed_ms,
            "rerank_ms": rerank_ms,
            "database_ms": server_ms - embed_ms - rerank_ms,
            "embedding_calls": after["embedding_calls"] - before["embedding_calls"],
            "rerank_calls": after["rerank_calls"] - before["rerank_calls"],
            "cost_usd": (after["embedding_cost"] - before["embedding_cost"]) + (after["rerank_cost"] - before["rerank_cost"]),
        }

    warmup = one(warmup_text)
    per_query: Dict[str, Dict[str, Any]] = {}
    for qid, text in texts.items():
        per_query[qid] = one(text)
        log(f"{qid}: {per_query[qid]['wall_ms']:.0f} ms, {len(per_query[qid]['ids'])} results")
    return {"warmup": warmup, "per_query": per_query}


def run_live(args: argparse.Namespace, log: Callable[[str], None]) -> Dict[str, Any]:
    """`run`: endpoints, the 44 queries, the judge for the holes, the merged V13 judgments."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.evaluation.relevance_judge import JudgeTask, JudgmentCache, find_unjudged, merge_judgments
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    started = datetime.now(UTC)
    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    texts = {q.query_id: q.text for q in queries}
    with httpx.Client(base_url=args.base_url, timeout=args.timeout_s) as client:
        usage_at_start = client.get("/api/v1/usage").json()
        searches = run_queries(client, texts, queries[0].text, log)
        endpoints = check_endpoints(client, searches["per_query"][queries[0].query_id]["ids"][0])
        usage_at_end = client.get("/api/v1/usage").json()

    run = {qid: [UUID(b) for b in data["ids"]] for qid, data in searches["per_query"].items()}
    base_raw = load_relevance_judgments(args.judgments)
    current = resolve_grades(base_raw, books)
    holes = find_unjudged({CLOUD: run}, current, DEPTH)
    judging: Dict[str, Any] = {"holes_at_start": len(holes)}
    new_judgments = []
    if holes and args.no_judge:
        raise SystemExit(f"{len(holes)} top-10 pairs have no judgment; run without --no-judge to grade them")
    if holes:
        from app.evaluation.judge_sensitivity import judge_in_parallel
        from app.evaluation.relevance_judge import _build_judge
        from app.infrastructure.llm.relevance_judge import LangChainRelevanceJudge

        settings, judge = _build_judge()
        judges = [judge] + [LangChainRelevanceJudge.from_settings(settings) for _ in range(args.workers - 1)]
        cache = JudgmentCache.for_judge(args.cache_dir, judge.model_name, judge.prompt_version)
        books_by_id = {book.id: book for book in books}
        tasks = [JudgeTask(qid, texts[qid], books_by_id[book_id]) for qid, book_id in holes]
        result = judge_in_parallel(tasks, judges, cache)
        if result.remaining:
            raise SystemExit(f"{result.remaining} pairs still without a judgment; run again")
        new_judgments = [cache.get(task.key) for task in tasks]
        cost = [j.usage_recorder.summary()["total"]["reported_cost_usd"] for j in judges if j.usage_recorder]
        judging.update({
            "judge_model": judge.model_name, "judged_in_this_run": result.judged, "already_cached": result.cached,
            "grade_distribution": [sum(1 for j in new_judgments if j.grade == g) for g in range(4)],
            "reported_cost_usd": sum(c for c in cost if c is not None),
        })
        merged = merge_judgments(base_raw, new_judgments)
        Path(args.judgments).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log(f"Added {len(new_judgments)} judgments to {args.judgments}")

    return {
        "date": started.isoformat(timespec="seconds"),
        "command": "python -m app.evaluation.cloud_live_check " + " ".join(sys.argv[1:]),
        "git_commit": _git("rev-parse", "HEAD"),
        "base_url": args.base_url,
        "schema": endpoints["health"].get("profile", {}).get("schema"),
        "endpoints": endpoints,
        "searches": searches,
        "usage_at_start": usage_at_start,
        "usage_at_end": usage_at_end,
        "judging": judging,
        "runs": {CLOUD: {q: [str(b) for b in ids] for q, ids in run.items()}},
    }


# =============================================================================
# Report
# =============================================================================


def build_report(saved: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """nDCG@10 of the cloud stack and of the local references on the V13 judgments, latency and cost."""
    from app.evaluation.evaluation_job import load_relevance_judgments, load_test_queries
    from app.infrastructure.db.sqlite_book_catalog_repository import SqliteBookCatalogRepository

    books = SqliteBookCatalogRepository(Path(args.db_path)).get_all()
    queries = load_test_queries(args.queries_path)
    judgments = resolve_grades(load_relevance_judgments(args.judgments), books)
    local = json.loads(Path(args.padding_json).read_text(encoding="utf-8"))["runs"]
    runs = {CLOUD: saved["runs"][CLOUD], **{name: local[name] for name in LOCAL_SYSTEMS if name in local}}
    runs = {s: {q: [UUID(b) for b in ids] for q, ids in run.items()} for s, run in runs.items()}
    require_judged(runs, judgments, DEPTH)
    report = compare_systems(runs, judgments, queries, None, n_resamples=args.n_resamples, seed=args.seed)
    lengths = list_lengths(runs, judgments, report["query_ids"])
    for name, data in report["systems"].items():
        data["list_lengths"] = lengths[name]
        data["not_relevant_share_of_returned"] = not_relevant_share(runs[name], judgments)
    pairs = [("d1", CLOUD, REFERENCE), ("d2", CLOUD, "hybrid_norm_e5_bge_nopad"), ("d3", CLOUD, "vec_e5")]
    descriptive = {}
    for name, system, reference in pairs:
        if system in runs and reference in runs:
            one = compare_pairs(report, [(name, system, reference)], n_resamples=args.n_resamples, seed=args.seed)[name]
            one.pop("p_value_holm", None)
            descriptive[f"{system} vs {reference}"] = one
    per_query = saved["searches"]["per_query"].values()
    latency = {key: latency_summary([q[key] for q in per_query])
               for key in ("wall_ms", "server_ms", "embedding_ms", "database_ms", "rerank_ms")}
    costs = [q["cost_usd"] for q in per_query]
    analysis = {
        "num_queries": report["num_queries"], "languages": report["languages"],
        "systems": report["systems"], "descriptive": descriptive, "latency": latency,
        "cost_per_search_usd": {"mean": sum(costs) / len(costs), "total": sum(costs), "searches": len(costs)},
        "calls_per_search": {
            "embedding": sum(q["embedding_calls"] for q in per_query) / len(costs),
            "rerank": sum(q["rerank_calls"] for q in per_query) / len(costs),
        },
        "degraded_searches": sum(1 for q in per_query if q["degraded"]),
    }
    return {**saved, "analysis": analysis, "judgments_sha256": _sha256(args.judgments)}


def _f(value: Optional[float], signed: bool = False, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def build_markdown(report: Mapping[str, Any]) -> str:
    a = report["analysis"]
    e = report["endpoints"]
    health = e["health"]
    out: List[str] = []
    w = out.append
    w("# Live check of the cloud profile (V13)\n")
    w(f"Run of {report['date']} (git `{str(report.get('git_commit'))[:7]}`): the API started locally in the cloud "
      f"configuration against the Neon schema `{report.get('schema')}` (the 177-book collection, documents embedded "
      f"by the cloud encoder), reached over HTTP at {report['base_url']}. Descriptive: no contrast was declared in "
      "advance.\n")
    w("## Configuration reported by /api/v1/health\n")
    w(f"Profile: `{json.dumps(health.get('profile'))}`; components ready: `{json.dumps(health.get('components'))}`; "
      f"reranker: `{json.dumps(health.get('reranker'))}`; LLM: `{json.dumps(health.get('llm'))}` (no LLM call is "
      "made: the searches use neither understanding nor explanations).\n")
    w("## Every endpoint answers\n")
    rows = [["GET /api/v1/health", health.get("status", "")], ["GET /ui/", str(e["ui"])],
            ["GET /api/v1/books/{id}", str(e["book"])],
            ["GET /api/v1/books/{id}/similar", f"{e['similar']['status']} ({e['similar']['results']} results)"],
            ["GET /api/v1/books/{id}/similar?use_diversification=true",
             f"{e['similar_mmr']['status']} ({e['similar_mmr']['results']} results)"],
            ["POST /api/v1/search (use_diversification)", f"{e['search_mmr']['status']} ({e['search_mmr']['results']} results, "
             f"degraded {e['search_mmr']['degraded']})"],
            ["POST /mcp (initialize)", str(e["mcp_initialize"])], ["GET /api/v1/usage", str(e["usage"])]]
    w(_table(["Endpoint", "Answer"], rows) + "\n")

    w("## Latency of POST /api/v1/search over the test queries\n")
    w(f"{a['num_queries']} queries, one at a time after one warm-up query, 10 results, from this machine to Neon "
      "(Frankfurt) and OpenRouter. Server = `latency_ms` of the response; embedding and re-ranking = the calls "
      "recorded by the usage endpoint; database = server - embedding - re-ranking (BM25 and vector statements, "
      "RRF and filters).\n")
    names = {"wall_ms": "Client wall clock", "server_ms": "Server search", "embedding_ms": "Query embedding (API)",
             "database_ms": "Database search (Postgres BM25 + vector, fusion)", "rerank_ms": "Re-ranking (API)"}
    rows = [[names[k], _f(v["median_ms"], digits=0), _f(v["p95_ms"], digits=0)] for k, v in a["latency"].items()]
    w(_table(["Part", "Median ms", "p95 ms"], rows) + "\n")
    warm = report["searches"]["warmup"]
    w(f"Warm-up query (first search of the process: connection to Neon, catalog facts): {warm['wall_ms']:.0f} ms "
      f"wall clock ({warm['embedding_ms']:.0f} embedding, {warm['database_ms']:.0f} database, "
      f"{warm['rerank_ms']:.0f} re-ranking). Degraded searches: {a['degraded_searches']}.\n")

    w("## Cost per search (reported by the providers, from the usage endpoint)\n")
    c = a["cost_per_search_usd"]
    w(f"Mean {c['mean']:.6f} USD per search ({c['total']:.4f} USD over {c['searches']} searches); "
      f"{a['calls_per_search']['embedding']:.1f} embedding and {a['calls_per_search']['rerank']:.1f} rerank calls "
      "per search.\n")

    w("## nDCG@10 on the V13 judgments (descriptive)\n")
    j = report.get("judging", {})
    w(f"`{CLOUD}`: API query embedding (cloud default), Postgres BM25 (`lakebase_text`) and vector search "
      "(`halfvec`, the ANN index of the schema), RRF, API reranker over the top 30. Local references from "
      "`results_bm25_padding.json`. Holes of the cloud top 10 graded by the judge: "
      f"{j.get('holes_at_start', 0)}"
      + (f" (grades 0/1/2/3: {' / '.join(str(g) for g in j['grade_distribution'])}; "
         f"{_f(j.get('reported_cost_usd'), digits=4)} USD)" if j.get("grade_distribution") else "") + ".\n")
    rows = []
    for name, data in a["systems"].items():
        m, lengths = data["metrics"], data["list_lengths"]
        langs = data["by_language"]
        rows.append([f"`{name}`", f"{_f(m[PRIMARY_METRIC]['mean'])} [{_f(m[PRIMARY_METRIC]['ci_low'])}, "
                     f"{_f(m[PRIMARY_METRIC]['ci_high'])}]", _f(m["recall_at_10"]["mean"]), _f(m["mrr"]["mean"]),
                     f"{lengths['mean']:.1f}", _f(lengths["judged_share_of_returned"]),
                     *[_f(langs[lang][PRIMARY_METRIC]["mean"]) for lang in sorted(langs)]])
    langs = sorted(next(iter(a["systems"].values()))["by_language"])
    w(_table(["System", "nDCG@10 [95% CI]", "Recall@10", "MRR", "Results per query", "Judged share of returned",
              *[f"nDCG@10 {lang}" for lang in langs]], rows) + "\n")
    rows = [[k, _f(d["mean_system"]), _f(d["mean_reference"]),
             f"{_f(d['mean_difference'], True)} [{_f(d['ci_low'], True)}, {_f(d['ci_high'], True)}]", _f(d["p_value"])]
            for k, d in a["descriptive"].items()]
    w(_table(["Pair", "nDCG@10 system", "nDCG@10 reference", "Difference [95% CI]", "p (unadjusted)"], rows) + "\n")

    w("## Limits\n")
    w("- One run from one machine and network; hosted latency depends on the providers' load at that minute.")
    w("- 44 queries and 177 books; the added grades come from an LLM judge.")
    w("- The cloud stack differs from the local reference in three places at once (embedding model, BM25 "
      "engine, reranker), so the difference is not attributable to one of them.\n")
    w("## Reproduce\n")
    w("```\nSEARCH_PROFILE=cloud python -m scripts.load_postgres --from data/catalog.db --schema eval177_api\n"
      "SEARCH_PROFILE=cloud DATABASE_SCHEMA=eval177_api USAGE_ENDPOINT_ENABLED=true uvicorn app.main:app --port 8765\n"
      "python -m app.evaluation.cloud_live_check run --base-url http://127.0.0.1:8765\n"
      "python -m app.evaluation.cloud_live_check report   # no API, no judge\n```\n")
    return "\n".join(out)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db-path", default=DEFAULT_DB_PATH)
        p.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH)
        p.add_argument("--judgments", default=DEFAULT_V13_JUDGMENTS)
        p.add_argument("--padding-json", default=DEFAULT_PADDING_JSON)
        p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
        p.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
        p.add_argument("--n-resamples", type=int, default=10000)
        p.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("run", help="query the running API, grade the holes, write the report")
    common(run)
    run.add_argument("--base-url", required=True)
    run.add_argument("--timeout-s", type=float, default=120.0)
    run.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--no-judge", action="store_true")
    report = sub.add_parser("report", help="recompute the tables from the saved rankings")
    common(report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    output = Path(args.output_json)
    if args.command == "run":
        saved = run_live(args, log)
        output.write_text(json.dumps(saved, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        if not output.exists():
            log("No saved run yet: run `cloud_live_check run` first")
            return 1
        saved = json.loads(output.read_text(encoding="utf-8"))
        saved.pop("analysis", None)
    report = build_report(saved, args)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(build_markdown(report), encoding="utf-8")
    log(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
