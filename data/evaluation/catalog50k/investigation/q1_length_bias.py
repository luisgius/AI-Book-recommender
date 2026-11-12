"""Q1: length of the works each system puts in its top 10 (A+B queries), against the catalog and against the works judged relevant."""
import json, numpy as np
from collections import defaultdict
INV = "data/evaluation/catalog50k/investigation"
W = {}
for l in open(f"{INV}/work_texts.jsonl"):
    r = json.loads(l); W[r["id"]] = r
qs = {x["query_id"]: x for x in json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]}
AB = [q for q, x in qs.items() if x["stratum"] in "AB"]
allc = np.array([r["chars"] for r in W.values()])
print("catalog chars median %d, share < 400 chars %.3f" % (np.median(allc), (allc < 400).mean()))
out = {}
for s in ("S1", "S2", "E1", "E2", "S5"):
    run = {json.loads(l)["query_id"]: json.loads(l) for l in open(f"data/evaluation/catalog50k/runs/{s}.jsonl")}
    c = np.array([W[i]["chars"] for q in AB for i in run[q]["ids"][:10]])
    out[s] = {"median_chars_top10": float(np.median(c)), "share_lt_400": float((c < 400).mean())}
    print(s, out[s])
# grade>=2 works (primary judge): their length
src = {(r["source"] if "source" in r else None) for r in []}
json.dump(out, open(f"{INV}/q1_length_bias.json", "w"), indent=1)
