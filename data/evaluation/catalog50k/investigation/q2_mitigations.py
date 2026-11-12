"""
Q2: candidate mitigations, measured with real calls. 60 queries (seed 0 from the 168) x 3 repetitions = 180
calls per arm; arms interleaved call by call (order shuffled per query) so that every arm sees the same
moments of provider load. Each call is one *policy* (attempts, timeouts, deadline) as a web request would run it.
Recorded per call: success, wall time, attempts, providers answering, the vector.
"""
import json, random, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
INV = Path(__file__).resolve().parent; sys.path.insert(0, str(INV)); sys.path.insert(0, ".")
import orclient
from app.infrastructure.search.text_encoding import QWEN3_QUERY_INSTRUCTION

QW, BG = "qwen/qwen3-embedding-8b", "baai/bge-m3"
pin = lambda p: {"order": [p], "allow_fallbacks": False}

def sequential(model, provider, attempt_s, attempts, deadline_s, dims):
    """Deployed-style policy: attempts in sequence, each cut to the time left; no attempt starts with < 0.5 s left."""
    def call(text, tag):
        t0 = time.perf_counter(); n = 0; last = None
        while n < attempts:
            left = deadline_s - (time.perf_counter() - t0)
            if left < 0.5: break
            n += 1
            r = orclient.embed([text], model, provider=provider, dimensions=dims, timeout=min(attempt_s, left), tag=tag)
            if r["ok"]:
                return {"ok": True, "wall": time.perf_counter() - t0, "attempts": n, "provider": r["provider"], "v": r["vectors"][0]}
            last = r["error"]
        return {"ok": False, "wall": time.perf_counter() - t0, "attempts": n, "error": last}
    return call

POOL = ThreadPoolExecutor(max_workers=8)
def hedged(model, provider, hedge_after_s, deadline_s, dims, max_requests=3):
    """Send one request; if no answer after hedge_after_s (or it fails), send another; first success wins."""
    def call(text, tag):
        t0 = time.perf_counter(); futs = []; n = 0; last = None
        def launch():
            nonlocal n
            n += 1
            left = max(0.5, deadline_s - (time.perf_counter() - t0))
            futs.append(POOL.submit(orclient.embed, [text], model, provider=provider, dimensions=dims, timeout=left, tag=tag))
        launch()
        next_hedge = t0 + hedge_after_s
        while True:
            now = time.perf_counter(); left = deadline_s - (now - t0)
            if left <= 0: break
            pending = [f for f in futs if not f.done()]
            done = [f for f in futs if f.done()]
            for f in done:
                r = f.result()
                if r["ok"]:
                    return {"ok": True, "wall": time.perf_counter() - t0, "attempts": n, "provider": r["provider"], "v": r["vectors"][0]}
                last = r["error"]
            futs[:] = pending
            if (not pending or now >= next_hedge) and n < max_requests and left > 0.5:
                launch(); next_hedge = time.perf_counter() + hedge_after_s; continue
            if not pending and n >= max_requests: break
            wait(pending, timeout=max(0.01, min(left, next_hedge - now if n < max_requests else left)), return_when=FIRST_COMPLETED)
        return {"ok": False, "wall": time.perf_counter() - t0, "attempts": n, "error": last or "deadline"}
    return call

ARMS = {
    "A0 qwen deepinfra, deployed (3 s x2, 5 s)": (QW, sequential(QW, pin("deepinfra"), 3.0, 2, 5.0, 1024)),
    "A1 qwen nebius, deployed (3 s x2, 5 s)": (QW, sequential(QW, pin("nebius"), 3.0, 2, 5.0, 1024)),
    "A2 qwen siliconflow, deployed (3 s x2, 5 s)": (QW, sequential(QW, pin("siliconflow"), 3.0, 2, 5.0, 1024)),
    "A3 qwen deepinfra, short attempts (1.5 s x3, 5 s)": (QW, sequential(QW, pin("deepinfra"), 1.5, 3, 5.0, 1024)),
    "A4 qwen deepinfra, hedged (2nd request at 1.0 s, max 3, 5 s)": (QW, hedged(QW, pin("deepinfra"), 1.0, 5.0, 1024)),
    "A5 qwen deepinfra then nebius fallback (3 s x2, 5 s)": (QW, sequential(QW, {"order": ["deepinfra", "nebius"], "allow_fallbacks": True}, 3.0, 2, 5.0, 1024)),
    "A6 bge-m3 deepinfra, deployed (3 s x2, 5 s)": (BG, sequential(BG, pin("deepinfra"), 3.0, 2, 5.0, None)),
}
if __name__ == "__main__":
    qs = json.load(open("data/evaluation/catalog50k/queries/queries.json"))["queries"]
    chosen = sorted(random.Random(0).sample(qs, 60), key=lambda q: q["query_id"])
    OUT = INV / "q2_mitigations_calls.jsonl"
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["arm"], json.loads(l)["query_id"], json.loads(l)["rep"]) for l in OUT.open()}
    rng = random.Random(42)
    for rep in (1, 2, 3):
        for q in chosen:
            arms = list(ARMS); rng.shuffle(arms)
            for arm in arms:
                if (arm, q["query_id"], rep) in done: continue
                orclient.check()
                model, call = ARMS[arm]
                text = (QWEN3_QUERY_INSTRUCTION if model == QW else "") + q["text"]
                r = call(text, "q2-mit:" + arm[:2])
                rec = {"arm": arm, "query_id": q["query_id"], "rep": rep, "t": time.strftime("%H:%M:%S"),
                       "ok": r["ok"], "wall_s": round(r["wall"], 4), "attempts": r["attempts"], "provider": r.get("provider"),
                       "error": r.get("error"), "vector": orclient.base64.b64encode(np.asarray(r["v"], dtype=np.float32).tobytes()).decode() if r["ok"] else None}
                with OUT.open("a") as f: f.write(json.dumps(rec) + "\n")
        print("rep", rep, "done; spent", orclient.spent(), flush=True)
    POOL.shutdown(wait=False)
