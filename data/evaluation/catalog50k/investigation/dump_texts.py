"""Dump the per-work searchable text (the loader's) of the 50k works to a local jsonl, with lengths and language count."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from app.evaluation.catalog_evaluation.runs import LocalCatalog
from app.evaluation.catalog_evaluation import design as D

out = Path("data/evaluation/catalog50k/investigation/work_texts.jsonl")
cat = LocalCatalog(D.SOURCE_CATALOG)
n = 0
with out.open("w", encoding="utf-8") as f:
    for b in sorted(cat.iter_all(), key=lambda b: str(b.id)):
        t = b.get_searchable_text()
        f.write(json.dumps({"id": str(b.id), "title": b.title, "language": getattr(b, "language", None),
                            "chars": len(t), "text": t}, ensure_ascii=False) + "\n")
        n += 1
print(n)
