"""Q2: quality of qwen query vectors by DeepInfra variant (V14 operational repetitions, A+B queries, existing judgments).
Variant 'match' = cosine > 0.99999 with the frozen E1 vector (which equals the stored-document variant)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from app.evaluation.catalog_evaluation.runs import decode_vector
from app.evaluation.catalog_evaluation.results import load_grades
from app.evaluation.catalog_evaluation.analysis import query_metrics
from app.evaluation.catalog_evaluation import design as D
DD = Path("data/evaluation/catalog50k")
pool = json.load(open(DD / "pool.json"))
queries = {x["query_id"]: x for x in json.load(open(DD / "queries/queries.json"))["queries"]}
fq = {json.loads(l)["query_id"]: decode_vector(json.loads(l)["vector"]) for l in open(DD / "query_embeddings/qwen__qwen3-embedding-8b.jsonl")}
recs = [json.loads(l) for l in open(DD / "operational/qwen__qwen3-embedding-8b.jsonl")]
out = {}
for judge in D.JUDGES:
    g = load_grades(DD, judge, pool)
    by = defaultdict(lambda: defaultdict(list))
    paired = []
    for r in recs:
        if r["error"] or queries[r["query_id"]]["stratum"] not in "AB":
            continue
        v = decode_vector(r["vector"]); f = fq[r["query_id"]]
        c = float(v @ f / np.linalg.norm(v) / np.linalg.norm(f)); kind = "match" if c > 0.99999 else "other"
        m = query_metrics(r["top10"], g[queries[r["query_id"]]["need_id"]])
        by[kind]["ndcg"].append(m["ndcg_at_10"]); by[kind]["judged"].append(m["judged_at_10"]); by[kind]["lat"].append(r["latency_ms"])
        by[kind]["qid"].append(r["query_id"])
    # paired per query: mean ndcg of 'other' minus 'match' where both exist
    diffs = []
    for q in set(by["match"]["qid"]) & set(by["other"]["qid"]):
        a = np.mean([n for n, qq in zip(by["match"]["ndcg"], by["match"]["qid"]) if qq == q])
        b = np.mean([n for n, qq in zip(by["other"]["ndcg"], by["other"]["qid"]) if qq == q])
        diffs.append(b - a)
    out[judge] = {k: {"calls": len(v["ndcg"]), "ndcg10_mean": round(float(np.mean(v["ndcg"])), 4),
                      "judged10_mean": round(float(np.mean(v["judged"])), 3),
                      "latency_median_ms": float(np.median(v["lat"]))} for k, v in by.items()}
    out[judge]["paired_other_minus_match"] = {"queries": len(diffs), "mean": round(float(np.mean(diffs)), 4),
                                              "abs_mean": round(float(np.mean(np.abs(diffs))), 4)}
print(json.dumps(out, indent=1))
json.dump(out, open(DD / "investigation/q2_variant_quality.json", "w"), indent=1)
