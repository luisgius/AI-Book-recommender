"""
Q2: why does the same text give different qwen vectors? Same texts sent repeatedly to each provider
(pinned, no fallbacks, 30 s timeout), sequentially. Distinct vectors are grouped into variants
(cosine > 0.99999 = same variant); we record which variant each call got, its latency, and how the
variants of different providers relate. Also: the works' stored Neon vectors (halfvec, DeepInfra, at
load time) against fresh document vectors from each provider.
"""
import json, sys, time
from pathlib import Path
import numpy as np
INV = Path(__file__).resolve().parent; sys.path.insert(0, str(INV)); sys.path.insert(0, ".")
import orclient
from dotenv import load_dotenv; load_dotenv(".env")
from app.infrastructure.search.text_encoding import QWEN3_QUERY_INSTRUCTION

M = "qwen/qwen3-embedding-8b"
qs = {x["query_id"]: x for x in json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]}
texts = [QWEN3_QUERY_INSTRUCTION + qs[q]["text"] for q in ("q02", "b-en-6", "b-zh-3", "c04-de")]
REPS = 15
providers = sys.argv[1:] or ["deepinfra", "nebius", "siliconflow"]
calls = []
for rep in range(REPS):
    for p in providers:
        for ti, t in enumerate(texts):
            orclient.check()
            r = orclient.embed([t], M, provider={"order": [p], "allow_fallbacks": False}, dimensions=1024,
                               timeout=30, tag=f"q2-determinism-{p}")
            calls.append({"p": p, "ti": ti, "rep": rep, "ok": r["ok"], "lat": r["latency_s"], "err": r["error"],
                          "served": r["provider"], "v": np.array(r["vectors"][0]) if r["ok"] else None})

def variants(vs):
    reps = []; lab = []
    for v in vs:
        for k, c in enumerate(reps):
            if v @ c / np.linalg.norm(v) / np.linalg.norm(c) > 0.99999:
                lab.append(k); break
        else:
            reps.append(v); lab.append(len(reps) - 1)
    return reps, lab

report = {"providers": {}, "cross": {}}
for p in providers:
    pc = [c for c in calls if c["p"] == p]
    okc = [c for c in pc if c["ok"]]
    lat = np.array([c["lat"] for c in okc]) if okc else np.array([np.nan])
    per_text = {}
    for ti in range(len(texts)):
        vs = [c["v"] for c in okc if c["ti"] == ti]
        reps, lab = variants(vs)
        for c, l in zip([c for c in okc if c["ti"] == ti], lab):
            c["variant"] = l
        cs = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for i, a in enumerate(reps) for b in reps[i + 1:]]
        bit = len({v.tobytes() for v in vs})
        per_text[ti] = {"calls": len(vs), "variants": len(reps), "bit_distinct": bit,
                        "variant_counts": [lab.count(k) for k in range(len(reps))],
                        "min_cos_between_variants": min(cs) if cs else 1.0}
        # latency by variant
        for k in range(len(reps)):
            per_text[ti][f"median_lat_variant{k}"] = float(np.median([c["lat"] for c in okc if c["ti"] == ti and c["variant"] == k]))
    report["providers"][p] = {"calls": len(pc), "failed": len(pc) - len(okc), "errors": [c["err"] for c in pc if not c["ok"]][:5],
                              "served": sorted({str(c["served"]) for c in pc}),
                              "lat_median": float(np.median(lat)), "lat_p95": float(np.percentile(lat, 95)), "lat_max": float(lat.max()),
                              "per_text": per_text}
# cross-provider: cosine between each provider's majority variant
for ti in range(len(texts)):
    maj = {}
    for p in providers:
        okc = [c for c in calls if c["p"] == p and c["ok"] and c["ti"] == ti]
        if okc:
            from collections import Counter
            k = Counter(c["variant"] for c in okc).most_common(1)[0][0]
            maj[p] = next(c["v"] for c in okc if c["variant"] == k)
            # also every variant
            for kk in sorted({c["variant"] for c in okc}):
                maj[f"{p}#v{kk}"] = next(c["v"] for c in okc if c["variant"] == kk)
    keys = sorted(maj)
    report["cross"][ti] = {f"{a}|{b}": round(float(maj[a] @ maj[b] / np.linalg.norm(maj[a]) / np.linalg.norm(maj[b])), 6)
                           for i, a in enumerate(keys) for b in keys[i + 1:]}
json.dump(report, open(INV / f"q2_determinism_{'_'.join(providers)}.json", "w"), indent=1)
for p, r in report["providers"].items():
    print(p, {k: v for k, v in r.items() if k != "per_text"})
    for ti, t in r["per_text"].items(): print("   text", ti, t)
print(json.dumps(report["cross"], indent=1)[:3000])
print("spent", orclient.spent())
