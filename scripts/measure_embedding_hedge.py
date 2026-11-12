"""
Measure the hedged query embedding against the single-provider policy, with real API calls.

Two arms, built from the application's own settings (EmbeddingSettings and
ApiQuerySettings of SEARCH_PROFILE=cloud), so each runs exactly the code a
search runs:
- single: the policy deployed before the hedge: DeepInfra pinned, one attempt
  of 3 s plus one retry, 5 s deadline (EMBEDDING_HEDGE_PROVIDER=none,
  EMBEDDING_OPENROUTER_ZDR=false);
- hedged: the new cloud default: DeepInfra, Nebius after 1.0 s, DeepInfra again
  at 2.5 s, 5 s deadline, first answer wins and the others are cancelled,
  Zero Data Retention endpoints only.
The query cache is off in both arms (every call reaches the API).

The texts are the queries of the deployed-catalog evaluation
(data/evaluation/catalog50k/queries/queries.json): the first --queries of them
in a seeded shuffle, --repetitions times each, in each arm. All calls are made
one after another in one shuffled order that interleaves the arms call by
call, so both arms see the same provider conditions.

Spend: the cost OpenRouter reports in each answer is summed as the run goes
and the run stops before --max-spend; the key's own usage counter
(GET /api/v1/key) is read before and after, so requests that were cancelled
and still billed would show up in the difference. The key is read from
--env-file into this process only and is never printed or written.

    python -m scripts.measure_embedding_hedge --env-file /path/to/.env \\
        --output /tmp/hedge_calls.jsonl --summary /tmp/hedge_summary.json
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

QUERIES = Path("data/evaluation/catalog50k/queries/queries.json")
KEY_URL = "https://openrouter.ai/api/v1/key"


def load_key(env_file: Path) -> None:
    """Put OPENROUTER_API_KEY from env_file into this process's environment (never printed)."""
    from dotenv import dotenv_values

    value = dotenv_values(env_file).get("OPENROUTER_API_KEY")
    if not value:
        sys.exit(f"OPENROUTER_API_KEY not found in {env_file}")
    os.environ["OPENROUTER_API_KEY"] = value


def key_usage() -> Optional[float]:
    """The key's total usage in USD as OpenRouter counts it (None if unavailable)."""
    try:
        response = httpx.get(KEY_URL, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                             timeout=15)
        return float(response.json()["data"]["usage"])
    except Exception:  # noqa: BLE001 - the measurement goes on without it
        return None


def probe_zdr(providers: List[str]) -> Dict[str, Any]:
    """
    One request per provider, pinned without fallbacks and with provider.zdr=true.

    OpenRouter answers 404 "No endpoints found" when the restriction leaves no
    eligible endpoint, so a 200 confirms the provider is ZDR-eligible for this model.
    """
    from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
    from app.infrastructure.search.openrouter_http import OpenRouterError

    results = {}
    for provider in providers:
        encoder = OpenRouterEmbeddingEncoder("qwen/qwen3-embedding-8b", dimensions=1024, upstream_provider=provider,
                                             zdr=True, max_retries=0, timeout_s=20)
        try:
            encoder.generate_query_embedding("zdr eligibility probe")
            results[provider] = {"ok": True, "body_provider": encoder.request_body(["x"])["provider"]}
        except OpenRouterError as error:
            results[provider] = {"ok": False, "status": error.status, "error": str(error)[:200]}
    return results


def build_arms() -> Dict[str, Any]:
    from app.infrastructure.config.settings import ApiQuerySettings, EmbeddingSettings
    from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder
    from app.infrastructure.usage_recording import UsageRecorder

    query_settings = ApiQuerySettings.from_env({"SEARCH_PROFILE": "cloud"})
    environments = {
        "single": {"SEARCH_PROFILE": "cloud", "EMBEDDING_HEDGE_PROVIDER": "none",
                   "EMBEDDING_OPENROUTER_ZDR": "false", "EMBEDDING_QUERY_CACHE_SIZE": "0"},
        "hedged": {"SEARCH_PROFILE": "cloud", "EMBEDDING_QUERY_CACHE_SIZE": "0"},
    }
    arms = {}
    for name, env in environments.items():
        recorder = UsageRecorder()
        encoder = OpenRouterEmbeddingEncoder.from_settings(EmbeddingSettings.from_env(env), recorder, query_settings)
        arms[name] = {"encoder": encoder, "recorder": recorder, "last": {}}
        if encoder.query_hedge is not None:
            poster = encoder._hedged  # noqa: SLF001 - observe which request won, nothing else changes
            original = poster.post_first
            last = arms[name]["last"]

            def observed(path, legs, *, deadline_s, _original=original, _last=last):
                _last.clear()
                try:
                    response = _original(path, legs, deadline_s=deadline_s)
                except Exception as error:  # noqa: BLE001
                    _last["legs"] = getattr(error, "legs", [])
                    raise
                _last["legs"], _last["winner_index"] = response.legs, next(
                    i for i, leg in enumerate(response.legs) if leg.sent and leg.error is None)
                return response

            poster.post_first = observed
    return arms


def vector_hash(vector: List[float]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(vector)}f", *vector)).hexdigest()[:16]


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(calls: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    mine = [c for c in calls if c["arm"] == arm]
    ok = [c for c in mine if c["error"] is None]
    wall = [c["wall_s"] for c in mine]
    by_query: Dict[str, set] = {}
    for c in ok:
        by_query.setdefault(c["query_id"], set()).add(c["vector"])
    summary = {
        "calls": len(mine),
        "failures": len(mine) - len(ok),
        "failure_errors": sorted({c["error"] for c in mine if c["error"]}),
        "wall_s_median": round(statistics.median(wall), 3) if wall else None,
        "wall_s_p95": round(percentile(wall, 0.95), 3) if wall else None,
        "wall_s_p99": round(percentile(wall, 0.99), 3) if wall else None,
        "wall_s_max": round(max(wall), 3) if wall else None,
        "requests_sent": sum(c["requests"] for c in mine),
        "requests_per_search": round(sum(c["requests"] for c in mine) / len(mine), 3) if mine else None,
        "cancelled_requests": sum(c["cancelled"] for c in mine),
        "reported_cost_usd": sum(c["cost_usd"] for c in mine),
        "calls_without_reported_cost": sum(1 for c in mine if c["error"] is None and c["cost_usd"] == 0),
        "queries_with_one_vector": sum(1 for v in by_query.values() if len(v) == 1),
        "queries": len(by_query),
    }
    if arm == "hedged":
        winners: Dict[str, int] = {}
        for c in ok:
            winners[c["winner"]] = winners.get(c["winner"], 0) + 1
        summary["answered_by"] = winners
        summary["extra_requests_per_search"] = round(summary["requests_per_search"] - 1, 3) if mine else None
    else:
        summary["extra_requests_per_search"] = round(summary["requests_per_search"] - 1, 3) if mine else None
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, required=True, help="file with OPENROUTER_API_KEY")
    parser.add_argument("--output", type=Path, required=True, help="JSON Lines file, one line per call")
    parser.add_argument("--summary", type=Path, required=True, help="JSON summary")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-spend", type=float, default=0.30, help="stop before this reported spend (USD)")
    parser.add_argument("--settle-s", type=float, default=0.0,
                        help="seconds to wait before reading the key's usage counter again")
    args = parser.parse_args(argv)

    load_key(args.env_file)
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    rng = random.Random(args.seed)
    rng.shuffle(queries)
    queries = queries[: args.queries]
    arms = build_arms()
    plan = [(arm, q, rep) for arm in arms for q in queries for rep in range(args.repetitions)]
    rng.shuffle(plan)

    usage_before = key_usage()
    zdr = probe_zdr(["deepinfra", "nebius"])
    print(f"ZDR probe: {json.dumps(zdr)}", file=sys.stderr)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    calls: List[Dict[str, Any]] = []
    spent = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for number, (arm, query, rep) in enumerate(plan, start=1):
            if spent >= args.max_spend * 0.9:
                print(f"Stopping: reported spend {spent:.6f} USD is near the cap", file=sys.stderr)
                break
            state = arms[arm]
            recorder, before = state["recorder"], len(state["recorder"].records)
            error, vector = None, None
            t0 = time.perf_counter()
            try:
                vector = state["encoder"].generate_query_embedding(query["text"])
            except Exception as e:  # noqa: BLE001 - a failure is a result
                error = f"{type(e).__name__}: {str(e)[:200]}"
            wall = time.perf_counter() - t0
            records = recorder.records[before:]
            cost = sum(r.cost_usd or 0.0 for r in records)
            spent += cost
            legs = state["last"].get("legs", [])
            winner = None
            if arm == "hedged" and error is None:
                index = state["last"]["winner_index"]
                winner = f"{legs[index].label}#{index + 1}"  # deepinfra#1, nebius#2 or deepinfra#3
            call = {
                "n": number, "time": time.strftime("%H:%M:%S"), "arm": arm, "query_id": query["query_id"],
                "rep": rep, "wall_s": round(wall, 4), "error": error,
                # requests that reached OpenRouter: the encoder records one per request sent
                "requests": len(records), "cancelled": sum(1 for r in records if r.error == "cancelled"),
                "cost_usd": cost, "winner": winner,
                "legs": [{"label": l.label, "sent": l.sent, "started_s": l.started_s, "latency_s": l.latency_s,
                          "error": None if l.error is None else str(l.error)[:120]} for l in legs],
                "vector": vector_hash(vector) if vector is not None else None,
            }
            calls.append(call)
            out.write(json.dumps(call, ensure_ascii=False) + "\n")
            out.flush()
            if number % 50 == 0:
                print(f"{number}/{len(plan)} calls, reported spend {spent:.6f} USD", file=sys.stderr)
    if args.settle_s > 0:
        time.sleep(args.settle_s)
    usage_after = key_usage()

    summary = {
        "started_at": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "queries": len(queries), "repetitions": args.repetitions, "seed": args.seed,
        "arms": {arm: summarize(calls, arm) for arm in arms},
        "zdr_probe": zdr,
        "reported_cost_usd_total": spent,
        "key_usage_delta_usd": (usage_after - usage_before) if usage_before is not None and usage_after is not None
        else None,
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for state in arms.values():
        state["encoder"]._http.close()  # noqa: SLF001
        if state["encoder"].query_hedge is not None:
            state["encoder"]._hedged.close()  # noqa: SLF001
    return 0


if __name__ == "__main__":
    sys.exit(main())
