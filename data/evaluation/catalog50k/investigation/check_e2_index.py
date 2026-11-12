"""Q1 checks on the stored E2 (bge-m3) vectors: alignment, norms, duplicates, brute force vs ExactIndex, query vectors."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from app.evaluation.catalog_evaluation.e2_index import ExactIndex
from app.evaluation.catalog_evaluation.runs import decode_vector

INV = Path("data/evaluation/catalog50k/investigation")
idx = ExactIndex.load(Path("data/catalog_deploy/bge_m3"))
rows = [json.loads(l) for l in open(INV / "work_texts.jsonl")]
sorted_ids = [r["id"] for r in rows]
print("index size", len(idx.ids), "unique", len(set(idx.ids)))
print("ids == loader ids sorted:", idx.ids == sorted_ids)
raw = np.vstack([np.load(f"data/catalog_deploy/bge_m3/shard_{i:03d}.npy") for i in range(25)])
n = np.linalg.norm(raw, axis=1)
print("raw dim", raw.shape, "nan", np.isnan(raw).sum(), "norm min/med/max", n.min(), np.median(n), n.max())
# exact duplicate vectors vs duplicate texts
_, inv_v, cnt_v = np.unique(raw.round(5), axis=0, return_inverse=True, return_counts=True)
print("duplicate vector groups", (cnt_v > 1).sum())
texts = [r["text"] for r in rows]
print("duplicate text groups", sum(1 for c in __import__("collections").Counter(texts).values() if c > 1))
# query vectors
q = {}
for l in open("data/evaluation/catalog50k/query_embeddings/baai__bge-m3.jsonl"):
    r = json.loads(l); q[r["query_id"]] = decode_vector(r["vector"])
qn = np.array([np.linalg.norm(v) for v in q.values()])
print("queries", len(q), "dims", {len(v) for v in q.values()}, "norm min/max", qn.min(), qn.max())
# brute force vs index vs run file
run = {json.loads(l)["query_id"]: json.loads(l) for l in open("data/evaluation/catalog50k/runs/E2.jsonl")}
mism = 0
M = raw / n[:, None]
for qid, v in q.items():
    s = M @ (v / np.linalg.norm(v))
    top = [sorted_ids[i] for i in np.argsort(-s, kind="stable")[:100]]
    got = run[qid]["ids"]
    if top[:10] != got[:10]:
        mism += 1
print("queries whose brute-force top10 != E2 run top10:", mism, "of", len(q))
# score distribution
s_all = []
for qid, v in list(q.items())[:20]:
    s = M @ (v / np.linalg.norm(v)); s_all.append((np.sort(s)[-1], np.sort(s)[-10], np.median(s)))
print("top1/top10/median cosine (first 20 queries):", np.round(np.mean(s_all, axis=0), 3))
