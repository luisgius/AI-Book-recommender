"""
Q2: which DeepInfra variant made the stored Neon document vectors and the frozen E1 query vectors?
8 pooled works: their text sent 8 times to DeepInfra (as documents, no instruction) and 3 times to Nebius;
each fresh vector's cosine with the stored halfvec vector. 6 queries: 8 calls to DeepInfra each, cosine
with the frozen E1 vector. Variant = fast/slow group within DeepInfra (cosine > 0.99999 to each other).
"""
import json, random, sys
from pathlib import Path
import numpy as np
INV = Path(__file__).resolve().parent; sys.path.insert(0, str(INV)); sys.path.insert(0, ".")
import orclient
from app.evaluation.catalog_evaluation.runs import decode_vector
from app.infrastructure.search.text_encoding import QWEN3_QUERY_INSTRUCTION
M = "qwen/qwen3-embedding-8b"
W = {}
for l in open(INV / "work_texts.jsonl"):
    r = json.loads(l); W[r["id"]] = r
ids = json.load(open(INV / "pool_qwen_doc_vectors.ids.json")); S = np.load(INV / "pool_qwen_doc_vectors.npy")
cos = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
rng = random.Random(1); pick = rng.sample(range(len(ids)), 8)
out = {"docs": [], "queries": []}
def run(text, stored, prov, n):
    rows = []
    for _ in range(n):
        r = orclient.embed([text], M, provider={"order": [prov], "allow_fallbacks": False}, dimensions=1024, timeout=30, tag=f"q2-docvariant-{prov}")
        if r["ok"]:
            rows.append({"prov": prov, "lat": r["latency_s"], "cos_stored": round(cos(r["vectors"][0], stored), 6)})
    return rows
for i in pick:
    rows = run(W[ids[i]]["text"], S[i], "deepinfra", 8) + run(W[ids[i]]["text"], S[i], "nebius", 3)
    out["docs"].append({"id": ids[i], "calls": rows}); print("doc", [(r["prov"][:2], r["cos_stored"], r["lat"]) for r in rows])
fq = {json.loads(l)["query_id"]: decode_vector(json.loads(l)["vector"]) for l in open("data/evaluation/catalog50k/query_embeddings/qwen__qwen3-embedding-8b.jsonl")}
qs = {x["query_id"]: x for x in json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]}
for q in ("q02", "b-en-6", "b-zh-3", "c04-de", "q20", "b-ru-2"):
    rows = run(QWEN3_QUERY_INSTRUCTION + qs[q]["text"], fq[q], "deepinfra", 8)
    out["queries"].append({"query_id": q, "calls": rows}); print("query", q, [(r["cos_stored"], r["lat"]) for r in rows])
json.dump(out, open(INV / "q2_doc_variant.json", "w"), indent=1)
print("spent", orclient.spent())
