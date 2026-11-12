"""Q2: summarise the mitigation arms: failure rate, median/p95/max wall time, repeatability, match with the stored-document variant."""
import base64, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from app.evaluation.catalog_evaluation.runs import decode_vector
INV = Path(__file__).resolve().parent; DD = Path("data/evaluation/catalog50k")
dec = lambda s: np.frombuffer(base64.b64decode(s), dtype=np.float32)
frozen = {m: {json.loads(l)["query_id"]: decode_vector(json.loads(l)["vector"]) for l in open(DD / f"query_embeddings/{m}.jsonl")}
          for m in ("qwen__qwen3-embedding-8b", "baai__bge-m3")}
recs = [json.loads(l) for l in open(INV / "q2_round2_calls.jsonl")]
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
rows = []
for arm in sorted({r["arm"] for r in recs}):
    rr = [r for r in recs if r["arm"] == arm]
    ok = [r for r in rr if r["ok"]]; w = np.array([r["wall_s"] for r in rr])
    fz = frozen["baai__bge-m3" if "bge" in arm else "qwen__qwen3-embedding-8b"]
    match = [cos(dec(r["vector"]), fz[r["query_id"]]) for r in ok]
    byq = defaultdict(list)
    for r in ok: byq[r["query_id"]].append(dec(r["vector"]))
    multi = [vs for vs in byq.values() if len(vs) >= 2]
    same = sum(1 for vs in multi if min(cos(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]) > 0.99999)
    mincos = min(min(cos(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]) for vs in multi)
    provs = defaultdict(int)
    for r in ok: provs[r["provider"]] += 1
    errs = defaultdict(int)
    for r in rr:
        if not r["ok"]: errs[str(r["error"])[:40]] += 1
    rows.append({"arm": arm, "calls": len(rr), "failed": len(rr) - len(ok), "fail_rate": round((len(rr) - len(ok)) / len(rr), 4),
                 "median_s": round(float(np.median(w)), 3), "p95_s": round(float(np.percentile(w, 95)), 3),
                 "p99_s": round(float(np.percentile(w, 99)), 3),
                 "ok_median_s": round(float(np.median([r["wall_s"] for r in ok])), 3),
                 "mean_requests_per_call": round(float(np.mean([r["attempts"] for r in rr])), 2),
                 "queries_same_vector_all_reps": f"{same}/{len(multi)}", "min_cos_between_reps": round(mincos, 6),
                 "share_matching_stored_variant": round(float(np.mean([m > 0.99999 for m in match])), 3),
                 "min_cos_to_stored_variant": round(min(match), 6), "providers": dict(provs), "errors": dict(errs)})
for r in rows: print(json.dumps(r, ensure_ascii=False))
json.dump(rows, open(INV / "q2_round2_summary.json", "w"), indent=1, ensure_ascii=False)
