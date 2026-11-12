"""The deployment check's statistics and report (V17), from saved results: no network."""

import json

from scripts import deployment_check as dc


def _record(query, kind, wall, server, degraded=False):
    return {"query": query, "language": "es", "kind": kind, "round": 1, "status": 200, "wall_ms": wall,
            "server_ms": server, "search_mode": "lexical_only" if degraded else "hybrid", "degraded": degraded,
            "degradation_reason": "Vector search failed" if degraded else None, "notices": [],
            "timings_ms": {"lexical": 30.0, "embedding": 400.0} if not degraded else {"lexical": 30.0},
            "reranked": True, "results": 10,
            "top": [{"id": "x", "title": "Cien años de soledad", "authors": ["G. García Márquez"], "language": "es",
                     "source": "wikidata", "description_source": None, "explanation": None, "grounded": None}]}


RESULTS = {
    "base_url": "https://example.vercel.app", "started_at": "2026-11-11T21:00:00+00:00",
    "health": {"status": 200, "vercel_id": "arn1::fra1::abc", "body": {"status": "ok"}},
    "ui": {"/ui/": {"status": 200, "bytes": 100, "cache": "HIT", "has_search_form": True}},
    "rounds": [[_record("Cien años de soledad", "exact", 900.0, 800.0),
                _record("manual de un motor", "no_answer", 1500.0, 1400.0, degraded=True)]],
    "book": {"id": "x", "status": 200, "title": "Cien años de soledad", "description_source": None},
    "similar": {"status": 200, "titles": ["El otoño del patriarca"]},
    "explanations": {"status": 200, "wall_ms": 5000.0, "search_mode": "hybrid", "timings_ms": {}, "top": []},
    "disabled": {"POST /api/v1/evaluate": 404, "GET /api/v1/usage": 404},
    "rate_limit": {"requests": 25, "ok": 20, "limited": 5, "first_429_at": 21, "retry_after_s": "42"},
    "rate_limit_detail": "Too many searches from this client",
    "cold_runs": [{"at": "2026-11-11T21:20:00+00:00", "idle_minutes": 12, "wall_ms": 4000.0, "server_ms": 3000.0,
                   "search_mode": "hybrid", "timings_ms": {"lexical": 1500.0}, "instance_uptime_s_after": 4.0,
                   "notes": ["The function started in 2.2 s."]}],
}


def test_percentiles_and_summaries():
    assert dc.percentile([], 0.95) is None
    assert dc.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert dc.percentile(list(map(float, range(1, 101))), 0.95) == 95.0
    assert dc.summary([3.0, 1.0, 2.0]) == {"n": 3, "median": 2.0, "p95": 3.0}


def test_latency_pools_every_round_and_splits_by_stage():
    lat = dc.latency(RESULTS)

    assert lat["searches"] == 2 and lat["degraded"] == 1 and lat["not_ok"] == 0
    assert lat["client_wall"]["median"] == 1200.0
    assert lat["stages"]["lexical"]["n"] == 2 and lat["stages"]["embedding"]["n"] == 1


def test_report_has_every_section(tmp_path):
    out = tmp_path / "results.json"
    report = tmp_path / "REPORT.md"
    out.write_text(json.dumps({**RESULTS, "deployment": {"Vercel project": "ai-book-recommender"}}))

    assert dc.main(["report", "--out", str(out), "--report", str(report)]) == 0
    text = report.read_text()

    for heading in ("## Deployment", "## Health", "## Web UI", "## Queries (round 1)", "## Warm latency",
                    "## Cold start", "## Book, similar books, explanations", "## Disabled endpoints and rate limit"):
        assert heading in text
    assert "run in `fra1`" in text and "`POST /api/v1/evaluate`: 404" in text and "first at request 21" in text
    assert "| 2026-11-11T21:20:00+00:00 | 12 | 4000.0 | 3000.0 | hybrid |" in text
    assert "The function started in 2.2 s." in text


def test_report_sets_warm_latency_before_and_after_side_by_side(tmp_path):
    """V18: the latency of the run before the catalog was enlarged is kept next to the new one."""
    before = {"label": "30,000 works", "run_at": "2026-11-11T22:38:37+00:00", "latency": dc.latency(RESULTS)}
    after = {**RESULTS, "catalog_label": "50,000 works", "previous_latency": [before], "catalog_works": 50000,
             "cold_runs": [{**RESULTS["cold_runs"][0], "catalog_works": 30000}],
             "cold_note": "No new cold start was measured.", "deployment": {}}
    out, report = tmp_path / "results.json", tmp_path / "REPORT.md"
    out.write_text(json.dumps(after))

    assert dc.main(["report", "--out", str(out), "--report", str(report)]) == 0
    text = report.read_text()

    assert "## Warm latency before and after the enlargement" in text
    assert "| Part | 30,000 works: median ms | p95 ms | 50,000 works: median ms | p95 ms |" in text
    assert "| Client wall clock | 1200.0 | 1500.0 | 1200.0 | 1500.0 |" in text
    assert "| Degraded of searches | 1 of 2 | - | 1 of 2 | - |" in text
    assert "| 4.0 | 30000 |" in text and "No new cold start was measured." in text
