"""Q2: retrieval quality of Nebius (and SiliconFlow) qwen query vectors against the DeepInfra-made document vectors.
Exact search in Neon (E1 repository, read-only, sequential) with the first successful vector of each query from round 1
arms A1 (nebius) and A2 (siliconflow); compared with the frozen E1 run of the same queries. Existing judgments; unjudged = 0."""
import base64, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")
from app.evaluation.catalog_evaluation.results import load_grades
from app.evaluation.catalog_evaluation.analysis import query_metrics
from app.evaluation.catalog_evaluation import design as D
from app.infrastructure.config.settings import database_url
from app.infrastructure.db.postgres_connection import PostgresDatabase
from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository
INV = Path(__file__).resolve().parent; DD = Path("data/evaluation/catalog50k")
pool = json.load(open(DD / "pool.json"))
queries = {x["query_id"]: x for x in json.load(open(DD / "queries/queries.json"))["queries"]}
E1 = {json.loads(l)["query_id"]: json.loads(l)["ids"] for l in open(DD / "runs/E1.jsonl")}
R = [json.loads(l) for l in open(INV / "q2_mitigations_calls.jsonl")]
db = PostgresDatabase(database_url(), pool_max_size=1, statement_timeout_ms=60_000)
repo = PgVectorSearchRepository(db, "catalog_deploy", mode="exact")
tops = {}
for arm in ("A1", "A2"):
    seen = {}
    for r in R:
        if r["arm"].startswith(arm) and r["ok"] and r["query_id"] not in seen:
            seen[r["query_id"]] = np.frombuffer(base64.b64decode(r["vector"]), dtype=np.float32)
    tops[arm] = {q: [str(h.book.id) for h in repo.search(list(map(float, v)), max_results=10)] for q, v in seen.items()}
db.close()
out = {}
for judge in D.JUDGES:
    g = load_grades(DD, judge, pool)
    for arm, t in tops.items():
        rows = []
        for q, ids in t.items():
            need = queries[q]["need_id"]
            m = query_metrics(ids, g[need]); m1 = query_metrics(E1[q][:10], g[need])
            rows.append((m["ndcg_at_10"], m1["ndcg_at_10"], m["judged_at_10"], len(set(ids) & set(E1[q][:10])) / 10, queries[q]["stratum"]))
        ab = [r for r in rows if r[4] in "AB"]
        out[f"{arm}|{judge}"] = {"queries": len(rows), "AB_queries": len(ab),
                                 "ndcg10_arm_AB": round(float(np.mean([r[0] for r in ab])), 4),
                                 "ndcg10_E1_same_queries_AB": round(float(np.mean([r[1] for r in ab])), 4),
                                 "judged10_arm": round(float(np.mean([r[2] for r in rows])), 3),
                                 "top10_overlap_with_E1": round(float(np.mean([r[3] for r in rows])), 3)}
print(json.dumps(out, indent=1)); json.dump(out, open(INV / "q2_nebius_quality.json", "w"), indent=1)
