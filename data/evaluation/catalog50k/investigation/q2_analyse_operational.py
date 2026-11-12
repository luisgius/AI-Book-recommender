"""Q2: what the V14 operational failures were, when, and for which queries; why vectors changed."""
import json, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, ".")
from app.evaluation.catalog_evaluation.runs import decode_vector
qs = {x["query_id"]: x for x in json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]}
for m in ("qwen__qwen3-embedding-8b", "baai__bge-m3"):
    recs = [json.loads(l) for l in open(f"data/evaluation/catalog50k/operational/{m}.jsonl")]
    print("=====", m, len(recs))
    errs = [r for r in recs if r["error"]]
    print("error kinds:", Counter(r["error"][:110] for r in errs).most_common(8))
    print("failed latency ms:", sorted(round(r["latency_ms"]) for r in errs)[:5], "...", sorted(round(r["latency_ms"]) for r in errs)[-5:])
    print("by repetition:", Counter(r["repetition"] for r in errs))
    # time: minute buckets
    mins = Counter(r["date"][11:16] for r in recs); emins = Counter(r["date"][11:16] for r in errs)
    print("first/last call:", recs[0]["date"], recs[-1]["date"])
    print("errors per 5-min:", sorted(Counter(r["date"][11:15] + "x" for r in errs).items()))
    print("calls per 5-min:", sorted(Counter(r["date"][11:15] + "x" for r in recs).items()))
    L = lambda r: len(qs[r["query_id"]]["text"])
    ok = [r for r in recs if not r["error"]]
    print("query chars: failed median %.0f, ok median %.0f" % (np.median([L(r) for r in errs]), np.median([L(r) for r in ok])))
    print("errors by language:", sorted(Counter(qs[r["query_id"]]["language"] for r in errs).items()))
    lat = np.array([r["latency_ms"] for r in ok]); print("ok latency median %.0f p95 %.0f p99 %.0f max %.0f; >3000ms ok: %d" % (np.median(lat), np.percentile(lat, 95), np.percentile(lat, 99), lat.max(), (lat > 3000).sum()))
    # vector variation: cluster distinct vectors per query
    byq = defaultdict(list)
    for r in ok:
        byq[r["query_id"]].append(decode_vector(r["vector"]))
    mins_cos, ndistinct = [], Counter()
    for q, vs in byq.items():
        distinct = {v.tobytes() for v in vs}; ndistinct[len(distinct)] += 1
        if len(vs) > 1:
            c = min(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for i, a in enumerate(vs) for b in vs[i + 1:])
            mins_cos.append(c)
    mc = np.array(mins_cos)
    print("distinct vectors per query:", dict(ndistinct))
    print("min pairwise cosine among repetitions: quantiles", np.round(np.quantile(mc, [0, .05, .25, .5, .75, 1]), 6))
    print("share of queries with a pair below 0.999:", float((mc < 0.999).mean()), "below 0.99:", float((mc < 0.99).mean()))
    t10 = defaultdict(list)
    for r in ok: t10[r["query_id"]].append(tuple(r["top10"]))
