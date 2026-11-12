#!/usr/bin/env python3
"""
Check a public deployment of the cloud profile from this machine (V17).

    python -m scripts.deployment_check run  --base-url https://<host> [--out data/evaluation/results_deployment_check.json]
    python -m scripts.deployment_check cold --base-url https://<host>   # after 10+ minutes without traffic
    python -m scripts.deployment_check report                          # Markdown from the JSON, no network

run: health; the web UI files; the test queries (Spanish, English and four
other languages, exact titles and authors, queries with no good answer in
the catalog) three times after one warm-up search, timed on this side and by
the backend (latency_ms and the stage split timings_ms); one book and its
similar books; one search with LLM explanations; that the costly or internal
endpoints are off; and the per-client rate limit, triggered with
GET /books/{id}/similar (no model call). Searches are spaced so this client
stays under the deployment's search limit while measuring.

cold: one search as the first request after the function and the database
have been idle, then /health (its uptime tells whether a new instance
served the search).

Every request is an ordinary public request: it spends the deployment's
daily budget like any visitor (about 45 model-backed searches and one LLM
request for run).
"""

import argparse
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_OUT = Path("data/evaluation/results_deployment_check.json")
DEFAULT_REPORT = Path("data/evaluation/DEPLOYMENT_CHECK.md")

# (query, language, kind): kind "topic", "exact" (title or author) or "no_answer"
QUERIES = [
    ("novelas sobre la guerra civil española", "es", "topic"),
    ("libros de filosofía para principiantes", "es", "topic"),
    ("Cien años de soledad", "es", "exact"),
    ("science fiction novels about artificial intelligence", "en", "topic"),
    ("history of the Roman Empire", "en", "topic"),
    ("Jane Austen", "en", "exact"),
    ("romans policiers qui se déroulent à Paris", "fr", "topic"),
    ("Romane über den Ersten Weltkrieg", "de", "topic"),
    ("romanzi storici ambientati a Firenze", "it", "topic"),
    ("romances sobre o mar e os pescadores", "pt", "topic"),
    ("manual de reparación del motor de un Seat Ibiza de 2004", "es", "no_answer"),
    ("how to configure a Kubernetes ingress controller", "en", "no_answer"),
]
ROUNDS = 3
SEARCH_SPACING_S = 3.2  # at most ~19 searches a minute from this client (limit 20)


def percentile(values: List[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile (None for no values)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))  # nearest rank: ceil(p * n)
    return ordered[min(rank, len(ordered)) - 1]


def summary(values: List[float]) -> Dict[str, Optional[float]]:
    """Median and p95 (milliseconds, rounded)."""
    if not values:
        return {"n": 0, "median": None, "p95": None}
    return {"n": len(values), "median": round(statistics.median(values), 1),
            "p95": round(percentile(values, 0.95), 1)}


def search(client: httpx.Client, text: str, **options: Any) -> Dict[str, Any]:
    """One POST /search with metadata; what the report needs from it."""
    started = time.perf_counter()
    try:
        response = client.post("/api/v1/search", json={"text": text, "max_results": 10, "include_metadata": True,
                                                        **options})
    except httpx.TransportError as error:  # network failure on the client side: recorded, the run goes on
        return {"status": None, "wall_ms": round((time.perf_counter() - started) * 1000, 1),
                "detail": f"client error: {type(error).__name__}: {error}"}
    wall_ms = (time.perf_counter() - started) * 1000
    record: Dict[str, Any] = {"status": response.status_code, "wall_ms": round(wall_ms, 1),
                              "vercel_id": response.headers.get("x-vercel-id")}
    if response.status_code != 200:
        record["detail"] = response.text[:300]
        return record
    body = response.json()
    record.update({
        "server_ms": round(body.get("latency_ms") or 0.0, 1),
        "search_mode": body.get("search_mode"),
        "degraded": body.get("degraded"),
        "degradation_reason": body.get("degradation_reason"),
        "notices": body.get("notices"),
        "timings_ms": (body.get("metadata") or {}).get("timings_ms", {}),
        "reranked": any(r.get("rerank_score") is not None for r in body.get("results", [])),
        "top": [{"id": r["book"]["id"], "title": r["book"]["title"], "authors": r["book"]["authors"][:2],
                 "language": r["book"]["language"], "source": r["book"]["source"],
                 "description_source": r["book"].get("description_source"),
                 "explanation": (r.get("explanation") or {}).get("text"),
                 "grounded": (r.get("explanation") or {}).get("grounded")}
                for r in body.get("results", [])[:3]],
        "results": len(body.get("results", [])),
    })
    return record


def run(base_url: str) -> Dict[str, Any]:
    """Every check except the cold start (see the module docstring)."""
    out: Dict[str, Any] = {"base_url": base_url, "started_at": datetime.now(UTC).isoformat(timespec="seconds")}
    with httpx.Client(base_url=base_url, timeout=60, headers={"User-Agent": "deployment-check"}) as client:
        health = client.get("/api/v1/health")
        out["health"] = {"status": health.status_code, "body": health.json(),
                         "vercel_id": health.headers.get("x-vercel-id")}

        ui = {}
        for path in ("/ui/", "/ui/app.js", "/ui/styles.css", "/"):
            r = client.get(path, follow_redirects=True)
            ui[path] = {"status": r.status_code, "bytes": len(r.content), "cache": r.headers.get("x-vercel-cache"),
                        "has_search_form": 'id="search-form"' in r.text if path == "/ui/" else None}
        out["ui"] = ui

        out["warmup"] = search(client, "warm-up: novels about the sea")
        time.sleep(SEARCH_SPACING_S)
        rounds: List[List[Dict[str, Any]]] = []
        for round_number in range(ROUNDS):
            records = []
            for text, language, kind in QUERIES:
                record = search(client, text)
                record.update({"query": text, "language": language, "kind": kind, "round": round_number + 1})
                records.append(record)
                print(f"round {round_number + 1} {record['status']} {record['wall_ms']:.0f} ms "
                      f"{record.get('search_mode')} {text}", flush=True)
                time.sleep(SEARCH_SPACING_S)
            rounds.append(records)
        out["rounds"] = rounds

        first = next((r for r in rounds[0] if r["kind"] == "exact" and r.get("top")), None)
        if first:
            book_id = first["top"][0]["id"]
            book = client.get(f"/api/v1/books/{book_id}")
            similar = client.get(f"/api/v1/books/{book_id}/similar", params={"max_results": 5})
            out["book"] = {"id": book_id, "status": book.status_code,
                           "title": book.json().get("title") if book.status_code == 200 else None,
                           "description_source": book.json().get("description_source") if book.status_code == 200 else None}
            out["similar"] = {"status": similar.status_code,
                              "titles": [r["book"]["title"] for r in similar.json().get("results", [])]
                              if similar.status_code == 200 else []}
        time.sleep(SEARCH_SPACING_S)

        out["explanations"] = search(client, "novelas de misterio ambientadas en Londres", use_explanations=True,
                                     max_results=3)
        time.sleep(SEARCH_SPACING_S)

        out["disabled"] = {
            "POST /api/v1/evaluate": client.post("/api/v1/evaluate", json={}).status_code,
            "GET /api/v1/usage": client.get("/api/v1/usage").status_code,
        }

        # Per-client limit: /similar counts toward the search limit and calls no model.
        # A minute of quiet first, so the searches above have left the window.
        time.sleep(61)
        if first:
            statuses, retry_after = [], None
            for _ in range(25):
                r = client.get(f"/api/v1/books/{first['top'][0]['id']}/similar", params={"max_results": 1})
                statuses.append(r.status_code)
                if r.status_code == 429 and retry_after is None:
                    retry_after = r.headers.get("retry-after")
                    out["rate_limit_detail"] = r.json().get("detail")
            out["rate_limit"] = {"requests": len(statuses), "ok": statuses.count(200),
                                 "limited": statuses.count(429),
                                 "first_429_at": statuses.index(429) + 1 if 429 in statuses else None,
                                 "retry_after_s": retry_after}
    out["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return out


def cold(base_url: str) -> Dict[str, Any]:
    """The first search after an idle period, then the instance's uptime."""
    with httpx.Client(base_url=base_url, timeout=150, headers={"User-Agent": "deployment-check"}) as client:
        record = search(client, "novelas sobre la guerra civil española")
        health = client.get("/api/v1/health").json()
    record.update({"at": datetime.now(UTC).isoformat(timespec="seconds"),
                   "instance_uptime_s_after": health.get("uptime_s"),
                   "database_after": health.get("database")})
    return record


# ----------------------------------------------------------------- report


def latency(results: Dict[str, Any]) -> Dict[str, Any]:
    """Warm latency over every round: client wall clock, server and stage split."""
    records = [r for rnd in results.get("rounds", []) for r in rnd if r.get("status") == 200]
    stages: Dict[str, List[float]] = {}
    for r in records:
        for stage, ms in (r.get("timings_ms") or {}).items():
            stages.setdefault(stage, []).append(ms)
    return {
        "client_wall": summary([r["wall_ms"] for r in records]),
        "server": summary([r["server_ms"] for r in records]),
        "stages": {stage: summary(values) for stage, values in sorted(stages.items())},
        "degraded": sum(1 for r in records if r.get("degraded")),
        "searches": len(records),
        "not_ok": sum(1 for rnd in results.get("rounds", []) for r in rnd if r.get("status") != 200),
    }


def render(results: Dict[str, Any], deployment: Dict[str, Any]) -> str:
    """The Markdown report (deployment facts come from the JSON's "deployment" entry)."""
    lat = latency(results)
    health = results["health"]["body"]
    lines = [
        "# Deployment check of the cloud profile (V17)", "",
        f"Run of {results['started_at']} against {results['base_url']} from the measuring machine "
        f"(Spain; requests enter Vercel's network at `{(results['health'].get('vercel_id') or '?').split('::')[0]}` "
        f"and run in `{(results['health'].get('vercel_id') or '?::?').split('::')[1]}`).", "",
        "## Deployment", "",
    ]
    for key, value in deployment.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## Health", "", "```json", json.dumps(health, indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Web UI", "", "| Path | Status | Bytes | Vercel cache |", "|---|---|---|---|"]
    for path, r in results["ui"].items():
        lines.append(f"| `{path}` | {r['status']} | {r['bytes']} | {r['cache'] or '-'} |")
    lines += ["", "## Queries (round 1)", "",
              "| Query | Lang. | Kind | Mode | Reranked | Top 3 (language) |", "|---|---|---|---|---|---|"]
    for r in results["rounds"][0]:
        top = "; ".join(f"{t['title']} ({t['language']})" for t in r.get("top", [])) or r.get("detail", "")
        lines.append(f"| {r['query']} | {r['language']} | {r['kind']} | {r.get('search_mode')}"
                     f"{' (degraded)' if r.get('degraded') else ''} | {r.get('reranked')} | {top} |")
    lines += ["", f"## Warm latency ({lat['searches']} searches: {len(QUERIES)} queries x {ROUNDS} rounds, "
              f"after one warm-up search; {lat['degraded']} degraded, {lat['not_ok']} not 200)", "",
              "| Part | n | Median ms | p95 ms |", "|---|---|---|---|",
              f"| Client wall clock | {lat['client_wall']['n']} | {lat['client_wall']['median']} | {lat['client_wall']['p95']} |",
              f"| Server (latency_ms) | {lat['server']['n']} | {lat['server']['median']} | {lat['server']['p95']} |"]
    for stage, s in lat["stages"].items():
        lines.append(f"| Stage `{stage}` | {s['n']} | {s['median']} | {s['p95']} |")
    degraded = [r for rnd in results["rounds"] for r in rnd if r.get("degraded")]
    if degraded:
        lines += ["", "Degraded searches:", ""]
        lines += [f"- round {r['round']}, \"{r['query']}\": {r.get('degradation_reason')}" for r in degraded]
    if results.get("cold_runs"):
        lines += ["", "## Cold start", "",
                  "The first search after the function and the database had been idle (Neon suspends its compute "
                  "after 5 minutes), then /health:", "",
                  "| At (UTC) | Idle min | Client wall ms | Server ms | Mode | Stages (ms) | Instance uptime at /health (s) |",
                  "|---|---|---|---|---|---|---|"]
        for c in results["cold_runs"]:
            mode = f"{c.get('search_mode')}{' (degraded)' if c.get('degraded') else ''}"
            lines.append(f"| {c['at']} | {c.get('idle_minutes', '?')} | {c['wall_ms']} | {c.get('server_ms')} | {mode} | "
                         f"{json.dumps(c.get('timings_ms'))} | {c.get('instance_uptime_s_after')} |")
        for c in results["cold_runs"]:
            if c.get("notes"):
                lines += ["", f"{c['at']}: " + " ".join(c["notes"])]
    lines += ["", "## Book, similar books, explanations", ""]
    if results.get("book"):
        lines.append(f"- `GET /api/v1/books/{results['book']['id']}`: {results['book']['status']} "
                     f"({results['book']['title']}; description source "
                     f"{json.dumps(results['book']['description_source'], ensure_ascii=False)})")
        lines.append(f"- `GET /api/v1/books/{{id}}/similar`: {results['similar']['status']} "
                     f"({'; '.join(results['similar']['titles'])})")
    e = results.get("explanations", {})
    lines.append(f"- Search with explanations: {e.get('status')}, {e.get('wall_ms')} ms, mode {e.get('search_mode')}, "
                 f"stages {json.dumps(e.get('timings_ms'))}")
    for t in e.get("top", []):
        lines.append(f"  - {t['title']}: grounded={t['grounded']}: {t['explanation']}")
    lines += ["", "## Disabled endpoints and rate limit", ""]
    for name, status in results["disabled"].items():
        lines.append(f"- `{name}`: {status}")
    if results.get("rate_limit"):
        rl = results["rate_limit"]
        lines.append(f"- {rl['requests']} quick `GET /books/{{id}}/similar` from this client: {rl['ok']} answered 200, "
                     f"{rl['limited']} answered 429 (first at request {rl['first_429_at']}, Retry-After "
                     f"{rl['retry_after_s']} s): \"{results.get('rate_limit_detail')}\"")
    for extra in results.get("extra_sections", []):
        lines += ["", f"## {extra['title']}", "", *extra["lines"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["run", "cold", "report"])
    parser.add_argument("--base-url", default=None, help="deployment URL, e.g. https://<host>")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results JSON")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown report")
    parser.add_argument("--idle-minutes", type=float, default=None, help="cold: minutes without traffic before it")
    args = parser.parse_args(argv)

    if args.step == "report":
        results = json.loads(args.out.read_text(encoding="utf-8"))
        args.report.write_text(render(results, results.get("deployment", {})), encoding="utf-8")
        print(f"Report written to {args.report}")
        return 0
    if not args.base_url:
        parser.error("--base-url is required for run and cold")
    base_url = args.base_url.rstrip("/")
    existing = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    if args.step == "run":
        existing.update(run(base_url))
    else:
        record = cold(base_url)
        record["idle_minutes"] = args.idle_minutes
        existing.setdefault("cold_runs", []).append(record)
    args.out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
