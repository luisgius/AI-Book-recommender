"""Q1: is a bge-m3 document vector the same alone and inside the original 32-text batch? Compared with the stored E2 vectors."""
import json, random, sys
from pathlib import Path
import numpy as np
INV = Path(__file__).resolve().parent
sys.path.insert(0, str(INV)); sys.path.insert(0, ".")
import orclient
from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder

PIN = {"order": ["deepinfra"], "allow_fallbacks": False}
rows = [json.loads(l) for l in open(INV / "work_texts.jsonl")]
ids = [r["id"] for r in rows]
stored = {}
def stored_vec(i):
    shard = i // 2000
    if shard not in stored:
        stored[shard] = np.load(f"data/catalog_deploy/bge_m3/shard_{shard:03d}.npy")
    return stored[shard][i % 2000]
cos = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
enc = OpenRouterEmbeddingEncoder.__new__(OpenRouterEmbeddingEncoder)
enc._batch_size, enc._max_batch_chars = 32, 20000
out = {}
# 1) the first two original batches of shard 7, re-sent exactly as batched in V14
shard = 7
texts = [r["text"] for r in rows[shard*2000:(shard+1)*2000]]
batches = OpenRouterEmbeddingEncoder.batches(enc, texts)
res = []
pos = 0
for b in batches[:2]:
    orclient.check()
    r = orclient.embed(b, "baai/bge-m3", provider=PIN, tag="q1-batch")
    for j, v in enumerate(r["vectors"]):
        gi = shard*2000 + pos + j
        res.append({"i": gi, "chars": len(b[j]), "cos_batch_vs_stored": cos(v, stored_vec(gi)), "vb": v})
    pos += len(b)
# 2) the same works, one per request
for x in res:
    r = orclient.embed([rows[x["i"]]["text"]], "baai/bge-m3", provider=PIN, tag="q1-single")
    x["cos_single_vs_stored"] = cos(r["vectors"][0], stored_vec(x["i"]))
    x["cos_single_vs_batch"] = cos(r["vectors"][0], x["vb"])
cb = np.array([x["cos_batch_vs_stored"] for x in res]); cs = np.array([x["cos_single_vs_stored"] for x in res])
csb = np.array([x["cos_single_vs_batch"] for x in res])
summary = {"works": len(res), "batch_sizes": [len(b) for b in batches[:2]],
           "batch_vs_stored": [float(cb.min()), float(np.median(cb))],
           "single_vs_stored": [float(cs.min()), float(np.median(cs)), float(cs.max())],
           "single_vs_batch": [float(csb.min()), float(np.median(csb))],
           "corr(single_vs_stored, chars)": float(np.corrcoef(cs, [x["chars"] for x in res])[0, 1])}
print(json.dumps(summary, indent=1))
for x in sorted(res, key=lambda x: x["cos_single_vs_stored"])[:8]:
    print(round(x["cos_single_vs_stored"], 4), x["chars"], rows[x["i"]]["title"][:60])
json.dump({"summary": summary, "per_work": [{k: v for k, v in x.items() if k != "vb"} for x in res]},
          open(INV / "q1_batch_vs_single.json", "w"), indent=1)
print("spent", orclient.spent())
