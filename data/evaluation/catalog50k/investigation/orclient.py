"""
Minimal OpenRouter embeddings client for the investigation, with a spend ledger.

- Reads OPENROUTER_API_KEY from .env (never printed). Sends only the bearer
  token (no HTTP-Referer, no X-Title, a generic User-Agent).
- Every call (ok or failed) is appended to ledger.jsonl with model, provider
  answered, status, latency, tokens and the cost OpenRouter reports.
- check() stops at the cap (1.50 USD) summed from the ledger.
"""
import base64, json, os, struct, threading, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

INV = Path(__file__).resolve().parent
LEDGER = INV / "ledger.jsonl"
CAP_USD = 1.50
URL = "https://openrouter.ai/api/v1/embeddings"
_lock = threading.Lock()


def _key() -> str:
    for line in (INV.parents[3] / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no key")


KEY = _key()
CLIENT = httpx.Client(headers={"Authorization": f"Bearer {KEY}", "User-Agent": "python-httpx"},
                      http2=False)


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum((json.loads(l).get("cost") or 0.0) for l in LEDGER.open())


def check() -> None:
    if spent() >= CAP_USD:
        raise RuntimeError("spending cap reached")


def decode(raw: Any) -> List[float]:
    if isinstance(raw, str):
        d = base64.b64decode(raw)
        return list(struct.unpack(f"<{len(d)//4}f", d))
    return [float(x) for x in raw]


def embed(inputs: List[str], model: str, *, provider: Optional[Dict[str, Any]] = None,
          dimensions: Optional[int] = None, timeout: float = 30.0, tag: str = "") -> Dict[str, Any]:
    """One request. Returns dict(ok, vectors, provider, status, error, latency_s, cost, tokens)."""
    body: Dict[str, Any] = {"model": model, "input": inputs, "encoding_format": "base64"}
    if dimensions:
        body["dimensions"] = dimensions
    if provider:
        body["provider"] = provider
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model,
                           "req_provider": provider, "n_inputs": len(inputs),
                           "chars": sum(len(x) for x in inputs)}
    out: Dict[str, Any] = {"ok": False, "vectors": None}
    try:
        r = CLIENT.post(URL, json=body, timeout=timeout)
        rec["status"] = r.status_code
        rec["latency_s"] = round(time.perf_counter() - t0, 4)
        try:
            j = r.json()
        except Exception:
            j = {"_text": r.text[:300]}
        if r.status_code == 200 and isinstance(j.get("data"), list):
            items = sorted(j["data"], key=lambda it: int(it["index"]))
            out["vectors"] = [decode(it["embedding"]) for it in items]
            out["ok"] = True
            u = j.get("usage") or {}
            rec.update(provider=j.get("provider"), id=j.get("id"), cost=u.get("cost"),
                       tokens=u.get("prompt_tokens"), served_model=j.get("model"))
        else:
            err = j.get("error") if isinstance(j, dict) else None
            rec["error"] = (json.dumps(err)[:400] if err else json.dumps(j)[:400])
            rec["provider"] = (err or {}).get("metadata", {}).get("provider_name") if isinstance(err, dict) else None
    except httpx.TimeoutException as e:
        rec.update(status=None, error=f"timeout:{type(e).__name__}", latency_s=round(time.perf_counter() - t0, 4))
    except httpx.HTTPError as e:
        rec.update(status=None, error=f"http:{type(e).__name__}:{str(e)[:150]}", latency_s=round(time.perf_counter() - t0, 4))
    with _lock:
        with LEDGER.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    out.update({k: rec.get(k) for k in ("provider", "status", "error", "latency_s", "cost", "tokens", "id")})
    return out
