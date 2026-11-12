"""
Q2 round 3: B0, B2 and a three-provider hedge B4, 6 repetitions, to catch a DeepInfra episode. Round 2 docstring follows.\n(after round 1 showed DeepInfra's failures come in episodes and other providers fail independently):
B0 control = A0 (deepinfra, 3 s x2, 5 s); B1 deepinfra, one attempt, 10 s (uncensored tail);
B2 cross-provider hedge: deepinfra at 0 s, nebius at 1.0 s, deepinfra again at 2.5 s, first success wins, 5 s;
B3 same as B2 with the second request at 0.6 s.
Same 60 queries x 3 repetitions, interleaved call by call. Each record keeps which request (provider) won.
"""
import base64, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
INV = Path(__file__).resolve().parent; sys.path.insert(0, str(INV)); sys.path.insert(0, ".")
import orclient
from q2_mitigations import sequential, QW, pin  # noqa: E402  (module body guarded below)
from app.infrastructure.search.text_encoding import QWEN3_QUERY_INSTRUCTION
POOL = ThreadPoolExecutor(max_workers=8)

def cross_hedge(schedule, deadline_s):
    """schedule: [(start_s, provider)]; each request launched at its start time unless one already succeeded."""
    def call(text, tag):
        t0 = time.perf_counter(); futs = {}; last = None; i = 0
        while True:
            now = time.perf_counter() - t0
            while i < len(schedule) and schedule[i][0] <= now:
                p = schedule[i][1]
                futs[POOL.submit(orclient.embed, [text], QW, provider=pin(p), dimensions=1024,
                                 timeout=max(0.5, deadline_s - now), tag=tag)] = p
                i += 1
            for f in [f for f in futs if f.done()]:
                r = f.result(); p = futs.pop(f)
                if r["ok"]:
                    return {"ok": True, "wall": time.perf_counter() - t0, "attempts": i, "provider": r["provider"], "v": r["vectors"][0]}
                last = r["error"]
                if i < len(schedule):  # a failure launches the next request at once
                    schedule_now = list(schedule); schedule_now[i] = (now, schedule[i][1]); schedule[:] = schedule_now
            left = deadline_s - (time.perf_counter() - t0)
            if left <= 0 or (not futs and i >= len(schedule)):
                break
            nxt = schedule[i][0] - (time.perf_counter() - t0) if i < len(schedule) else left
            if futs:
                wait(list(futs), timeout=max(0.01, min(left, nxt)), return_when=FIRST_COMPLETED)
            else:
                time.sleep(max(0.0, min(left, nxt)))
        return {"ok": False, "wall": time.perf_counter() - t0, "attempts": i, "error": last or "deadline"}
    return call

def hedge_factory(schedule):
    def call(text, tag):
        return cross_hedge(list(schedule), 5.0)(text, tag)
    return call

ARMS = {
    "B0 qwen deepinfra, deployed (3 s x2, 5 s) [control]": sequential(QW, pin("deepinfra"), 3.0, 2, 5.0, 1024),
    "B2 qwen hedge deepinfra 0 s / nebius 1.0 s / deepinfra 2.5 s, 5 s": hedge_factory([(0.0, "deepinfra"), (1.0, "nebius"), (2.5, "deepinfra")]),
    "B4 qwen hedge deepinfra 0 s / nebius 1.0 s / siliconflow 2.0 s, 5 s": hedge_factory([(0.0, "deepinfra"), (1.0, "nebius"), (2.0, "siliconflow")]),
}
qs = json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]
chosen = sorted(random.Random(0).sample(qs, 60), key=lambda q: q["query_id"])
OUT = INV / "q2_round3_calls.jsonl"
done = {(json.loads(l)["arm"], json.loads(l)["query_id"], json.loads(l)["rep"]) for l in OUT.open()} if OUT.exists() else set()
rng = random.Random(11)
for rep in (1, 2, 3, 4, 5, 6):
    for q in chosen:
        arms = list(ARMS); rng.shuffle(arms)
        for arm in arms:
            if (arm, q["query_id"], rep) in done: continue
            orclient.check()
            r = ARMS[arm](QWEN3_QUERY_INSTRUCTION + q["text"], "q2-r2:" + arm[:2])
            rec = {"arm": arm, "query_id": q["query_id"], "rep": rep, "t": time.strftime("%H:%M:%S"), "ok": r["ok"],
                   "wall_s": round(r["wall"], 4), "attempts": r["attempts"], "provider": r.get("provider"), "error": r.get("error"),
                   "vector": base64.b64encode(np.asarray(r["v"], dtype=np.float32).tobytes()).decode() if r["ok"] else None}
            with OUT.open("a") as f: f.write(json.dumps(rec) + "\n")
    print("rep", rep, "done; spent", orclient.spent(), flush=True)
