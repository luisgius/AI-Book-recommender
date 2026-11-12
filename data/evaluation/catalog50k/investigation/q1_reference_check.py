"""Q1: does the API's baai/bge-m3 reproduce the dense similarities printed in the BGE-M3 model card?
Model card (BAAI/bge-m3, FlagEmbedding README, dense embedding example, fp16):
  sentences_1 = ["What is BGE M3?", "Defination of BM25"]
  sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
                 "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query tokens that appear in each document"]
  similarity = [[0.6265, 0.3477], [0.3499, 0.678]]
"""
import json, sys
from pathlib import Path
import numpy as np
INV = Path(__file__).resolve().parent; sys.path.insert(0, str(INV)); import orclient
s1 = ["What is BGE M3?", "Defination of BM25"]
s2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
      "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query tokens that appear in each document"]
ref = [[0.6265, 0.3477], [0.3499, 0.678]]
out = {"reference": ref}
for prov in ("deepinfra", "parasail"):
    r = orclient.embed(s1 + s2, "baai/bge-m3", provider={"order": [prov], "allow_fallbacks": False}, tag="q1-ref")
    if not r["ok"]:
        out[prov] = {"error": r["error"]}; print(prov, r["error"]); continue
    v = np.array(r["vectors"]); v /= np.linalg.norm(v, axis=1, keepdims=True)
    sim = (v[:2] @ v[2:].T).round(4).tolist()
    out[prov] = {"similarity": sim, "dim": v.shape[1], "provider": r["provider"]}
    print(prov, r["provider"], sim)
json.dump(out, open(INV / "q1_reference_check.json", "w"), indent=1)
