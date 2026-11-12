"""Read (only) the stored qwen document vectors of the pooled works from Neon, in chunks of 500, one connection."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")
from app.infrastructure.config.settings import database_url
from app.infrastructure.db.postgres_connection import PostgresDatabase
INV = Path(__file__).resolve().parent
pool = json.load(open("data/evaluation/catalog50k/pool.json"))
ids = sorted({w["work_id"] for ws in pool.values() for w in ws})
db = PostgresDatabase(database_url(), pool_max_size=1, statement_timeout_ms=60_000)
vecs = {}
for i in range(0, len(ids), 500):
    rows = db.query("SELECT id::text AS id, embedding::text AS e FROM catalog_deploy.works WHERE id = ANY(%(ids)s::uuid[])",
                    {"ids": ids[i:i + 500]})
    for r in rows:
        vecs[r["id"]] = np.array(json.loads(r["e"]), dtype=np.float32)
db.close()
order = sorted(vecs)
np.save(INV / "pool_qwen_doc_vectors.npy", np.vstack([vecs[k] for k in order]))
json.dump(order, open(INV / "pool_qwen_doc_vectors.ids.json", "w"))
print(len(ids), len(vecs))
