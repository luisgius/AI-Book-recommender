"""
Q1: (a) ranking restricted to each query's judged pool (both models see the same candidates, so
collection-size distractors are removed); (b) recall@100 of grade>=2 works split by the work's
language, by text length and by whether the work's language equals the query's; (c) sanity: E1
scores recomputed from the stored qwen vectors.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from app.evaluation.catalog_evaluation.runs import decode_vector
from app.evaluation.catalog_evaluation.results import load_grades
from app.evaluation.catalog_evaluation.analysis import query_metrics
from app.evaluation.catalog_evaluation import design as D
INV = Path(__file__).resolve().parent
DD = Path("data/evaluation/catalog50k")
W = {}
for l in open(INV / "work_texts.jsonl"):
    r = json.loads(l); W[r["id"]] = r
allids = sorted(W)
pos = {k: i for i, k in enumerate(allids)}
B = np.vstack([np.load(f"data/catalog_deploy/bge_m3/shard_{i:03d}.npy") for i in range(25)])
qids = json.load(open(INV / "pool_qwen_doc_vectors.ids.json")); Q = np.load(INV / "pool_qwen_doc_vectors.npy")
Q = Q / np.linalg.norm(Q, axis=1, keepdims=True); qpos = {k: i for i, k in enumerate(qids)}
fq = {m: {json.loads(l)["query_id"]: decode_vector(json.loads(l)["vector"]) for l in open(DD / f"query_embeddings/{m}.jsonl")}
      for m in ("qwen__qwen3-embedding-8b", "baai__bge-m3")}
queries = {x["query_id"]: x for x in json.load(open(DD / "queries/queries.json"))["queries"]}
AB = [q for q, x in queries.items() if x["stratum"] in "AB"]
pool = json.load(open(DD / "pool.json"))
runs = {s: {json.loads(l)["query_id"]: json.loads(l) for l in open(DD / f"runs/{s}.jsonl")} for s in ("E1", "E2")}
out = {}
# (c) sanity
d = []
for q in AB[:30]:
    top = runs["E1"][q]["ids"][0]; v = fq["qwen__qwen3-embedding-8b"][q]
    d.append(abs(float(Q[qpos[top]] @ (v / np.linalg.norm(v))) - runs["E1"][q]["scores"][0]))
out["E1_score_recompute_max_abs_diff"] = max(d)
for judge in (D.PRIMARY_JUDGE, D.SECOND_JUDGE):
    grades = load_grades(DD, judge, pool)
    # (a) within pool
    res = defaultdict(list)
    for q in AB:
        need = queries[q]["need_id"]; g = grades[need]; cand = [w["work_id"] for w in pool[need]]
        vq = fq["qwen__qwen3-embedding-8b"][q]; vb = fq["baai__bge-m3"][q]
        sq = Q[[qpos[c] for c in cand]] @ (vq / np.linalg.norm(vq)); sb = B[[pos[c] for c in cand]] @ (vb / np.linalg.norm(vb))
        res["qwen"].append(query_metrics([cand[i] for i in np.argsort(-sq, kind="stable")], g)["ndcg_at_10"])
        res["bge"].append(query_metrics([cand[i] for i in np.argsort(-sb, kind="stable")], g)["ndcg_at_10"])
        res["E1_full"].append(query_metrics(runs["E1"][q]["ids"], g)["ndcg_at_10"])
        res["E2_full"].append(query_metrics(runs["E2"][q]["ids"], g)["ndcg_at_10"])
    out[f"within_pool_ndcg10_{judge}"] = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
    # (b) recall@100 of grade>=2 works, splits
    split = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for q in AB:
        need = queries[q]["need_id"]; qlang = queries[q]["language"]
        rel = [w for w, gr in grades[need].items() if gr >= 2]
        e1 = set(runs["E1"][q]["ids"][:100]); e2 = set(runs["E2"][q]["ids"][:100])
        for w in rel:
            c = W[w]["chars"]; lang = W[w].get("language") or "?"
            keys = {"len": "<400" if c < 400 else "400-800" if c < 800 else "800-1200" if c < 1200 else ">=1200",
                    "work_lang": lang if lang in ("en", "es", "fr", "de") else "other",
                    "same_lang_as_query": str(lang == qlang)}
            for k, v in keys.items():
                split[k][v][0] += w in e1; split[k][v][1] += w in e2
                split[k][v].append(1) if False else None
            split["all"]["all"][0] += w in e1; split["all"]["all"][1] += w in e2
    cnt = defaultdict(lambda: defaultdict(int))
    for q in AB:
        need = queries[q]["need_id"]; qlang = queries[q]["language"]
        for w, gr in grades[need].items():
            if gr >= 2:
                c = W[w]["chars"]; lang = W[w].get("language") or "?"
                cnt["len"]["<400" if c < 400 else "400-800" if c < 800 else "800-1200" if c < 1200 else ">=1200"] += 1
                cnt["work_lang"][lang if lang in ("en", "es", "fr", "de") else "other"] += 1
                cnt["same_lang_as_query"][str(lang == qlang)] += 1
                cnt["all"]["all"] += 1
    out[f"recall100_grade2plus_{judge}"] = {k: {v: {"n": cnt[k][v], "E1": round(a[0] / cnt[k][v], 3), "E2": round(a[1] / cnt[k][v], 3)}
                                              for v, a in vs.items()} for k, vs in split.items()}
print(json.dumps(out, indent=1))
json.dump(out, open(INV / "q1_pool_and_splits.json", "w"), indent=1)
