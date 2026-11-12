"""
Q1: title self-retrieval. 200 works drawn with seed 0; the query is the work's title alone.
bge-m3: no prefix, exact cosine over the stored E2 matrix (all 50,000 ranks).
qwen3-embedding-8b: deployed query instruction, 1024 d, DeepInfra pinned; exact search in Neon
(the E1 repository, read-only, one query at a time), top 100 (rank > 100 reported as miss).
A hit is the work itself or another work with the identical title.
Also: re-encode the 168 evaluation queries with bge-m3 and compare with the frozen vectors.
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
INV = Path(__file__).resolve().parent; sys.path.insert(0, str(INV)); sys.path.insert(0, ".")
import orclient
from dotenv import load_dotenv; load_dotenv(".env")
from app.evaluation.catalog_evaluation.runs import decode_vector
from app.infrastructure.search.text_encoding import QWEN3_QUERY_INSTRUCTION

PIN = {"order": ["deepinfra"], "allow_fallbacks": False}
rows = [json.loads(l) for l in open(INV / "work_texts.jsonl")]
ids = [r["id"] for r in rows]
by_title = {}
for r in rows:
    by_title.setdefault(r["title"].strip().lower(), set()).add(r["id"])
M = np.vstack([np.load(f"data/catalog_deploy/bge_m3/shard_{i:03d}.npy") for i in range(25)])
rng = random.Random(0)
sample = rng.sample(range(len(rows)), 200)

# --- frozen bge query vectors vs fresh
frozen = {json.loads(l)["query_id"]: decode_vector(json.loads(l)["vector"])
          for l in open("data/evaluation/catalog50k/query_embeddings/baai__bge-m3.jsonl")}
qs = {x["query_id"]: x for x in json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]}
fresh_cos = []
for qid, v in frozen.items():
    orclient.check()
    r = orclient.embed([qs[qid]["text"]], "baai/bge-m3", provider=PIN, tag="q1-frozen-recheck")
    if r["ok"]:
        f = np.array(r["vectors"][0]); fresh_cos.append(float(f @ v / np.linalg.norm(f) / np.linalg.norm(v)))
print("frozen vs fresh bge query cosine: n=%d min=%.6f median=%.6f" % (len(fresh_cos), min(fresh_cos), np.median(fresh_cos)))

# --- bge self-retrieval (local)
res = []
for i in sample:
    t = rows[i]["title"]
    r = orclient.embed([t], "baai/bge-m3", provider=PIN, tag="q1-self-bge")
    v = np.array(r["vectors"][0]); s = M @ (v / np.linalg.norm(v))
    order = np.argsort(-s, kind="stable")
    same = by_title[t.strip().lower()]
    rank = next(k + 1 for k, j in enumerate(order) if ids[j] in same)
    res.append({"i": i, "title": t, "chars": rows[i]["chars"], "bge_rank": rank})

# --- qwen self-retrieval (Neon exact, top 100)
from app.infrastructure.config.settings import database_url
from app.infrastructure.db.postgres_connection import PostgresDatabase
from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
db = PostgresDatabase(database_url(), pool_max_size=1, statement_timeout_ms=60_000)
repo = PgVectorSearchRepository(db, "catalog_deploy", mode="exact")
for x in res:
    r = None
    for _ in range(4):
        r = orclient.embed([QWEN3_QUERY_INSTRUCTION + x["title"]], "qwen/qwen3-embedding-8b", provider=PIN,
                           dimensions=1024, tag="q1-self-qwen")
        if r["ok"]:
            break
        time.sleep(2)
    if not r["ok"]:
        x["qwen_rank"] = None; continue
    hits = repo.search(r["vectors"][0], max_results=100)
    same = by_title[x["title"].strip().lower()]
    x["qwen_rank"] = next((h.rank for h in hits if str(h.book.id) in same), 101)
db.close()

def summ(key):
    rk = np.array([x[key] for x in res if x.get(key) is not None])
    return {"n": int(len(rk)), "hit@1": float((rk == 1).mean()), "hit@10": float((rk <= 10).mean()),
            "hit@100": float((rk <= 100).mean()), "mrr@100": float(np.mean([1 / r if r <= 100 else 0 for r in rk]))}
summary = {"bge-m3": summ("bge_rank"), "qwen3-8b": summ("qwen_rank"),
           "frozen_vs_fresh_bge_cosine": {"n": len(fresh_cos), "min": min(fresh_cos), "median": float(np.median(fresh_cos))}}
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "per_work": res}, open(INV / "q1_self_retrieval.json", "w"), indent=1, ensure_ascii=False)
print("spent", orclient.spent())
