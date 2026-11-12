"""Q1: rank of each known-item target (stratum B, by title, hand-mapped for b-zh-6) under E1 (top 100 run) and E2 (full exact ranking)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from app.evaluation.catalog_evaluation.runs import decode_vector
INV = "data/evaluation/catalog50k/investigation"
rows = [json.loads(l) for l in open(f"{INV}/work_texts.jsonl")]; ids = [r["id"] for r in rows]
B = np.vstack([np.load(f"data/catalog_deploy/bge_m3/shard_{i:03d}.npy") for i in range(25)])
qs = {x["query_id"]: x for x in json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]}
fb = {json.loads(l)["query_id"]: decode_vector(json.loads(l)["vector"]) for l in open("data/evaluation/catalog50k/query_embeddings/baai__bge-m3.jsonl")}
E1 = {json.loads(l)["query_id"]: json.loads(l)["ids"] for l in open("data/evaluation/catalog50k/runs/E1.jsonl")}
out = []
for q, x in qs.items():
    if x["type"] != "known_item": continue
    title = x["target_title"] if q != "b-zh-6" else "Dream of the Red Chamber"
    targets = {r["id"] for r in rows if r["title"].strip().lower() == title.strip().lower()}
    s = B @ fb[q]; order = np.argsort(-s, kind="stable"); r2 = next((k + 1 for k, j in enumerate(order) if ids[j] in targets), None)
    r1 = next((k + 1 for k, i in enumerate(E1[q]) if i in targets), ">100")
    out.append({"query_id": q, "target": title, "E1_rank": r1, "E2_rank": r2}); print(q, title[:30], "E1", r1, "E2", r2)
json.dump(out, open(f"{INV}/q1_known_items.json", "w"), indent=1, ensure_ascii=False)
