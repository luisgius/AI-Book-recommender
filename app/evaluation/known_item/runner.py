"""
Runs of the known-item evaluation against the deployed catalog (phase 2).

It reuses the catalog evaluation's network side (catalog_evaluation.pipeline
.Context): the same Neon schema, the same Postgres BM25 and pgvector
adapters, the same pinned query encoder and API reranker, the same frozen
query embeddings and the same run-file format and stores. Steps, all
resumable (a query already stored is never run again):

1. embed: one frozen `qwen/qwen3-embedding-8b` vector per query (paid, tiny);
2. database: S1 and S2 at depth 100 and their top 60 (S1@60, S2@60), the
   fusion inputs, one query at a time on one connection (free);
3. fusion (pure): S3 = RRF of S1@60 and S2@60; S3w@w for every w of the
   grid (free);
4. weight: w chosen on the development half from S3w@w only, written to
   weight_selection.json before any test-half reranking; S3w = S3w@w;
5. estimate: the reranker cost of the test half from the length of the texts
   that will be sent; the run stops here when it exceeds what is left under
   the cap;
6. rerank (paid, test half only): S4, S5 and S5w reorder the top 30 of S2,
   S3 and S3w with the API reranker.

The cross-encoder scores every (query, book) pair on its own, so each book
of the union of the three heads of a query is scored once (requests of at
most 30 books) and each head is then sorted by those scores, ties keeping
the head's order, exactly as ApiReranker.rerank sorts one head. This avoids
paying three times for a book that is in two or three heads.

Every paid call goes through the ledger (catalog_evaluation.ledger) with a
cap of 1.50 USD: ledger.check() runs before each call and stops the run once
the recorded spend reaches the cap.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from app.evaluation.catalog_evaluation import design as CD
from app.evaluation.catalog_evaluation.ledger import SpendLedger
from app.evaluation.catalog_evaluation.pipeline import API_WORKERS, CALL_ATTEMPTS, Context, embed_queries, frozen_vector, ms
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.catalog_evaluation.runs import JsonlStore, embedding_store, rrf_fuse, run_store
from app.evaluation.known_item import analysis as KA
from app.evaluation.known_item import design as D
from app.evaluation.known_item.fusion import weighted_rrf

RERANK_SCORES = "rerank_scores.jsonl"
ESTIMATE_FILE = "rerank_estimate.json"


class EstimateAboveCap(RuntimeError):
    """The estimated reranker cost exceeds what is left under the cap: nothing is reranked."""


# =============================================================================
# Pure helpers
# =============================================================================


def eval_query(query: Mapping[str, Any]) -> EvalQuery:
    """The catalog evaluation's query record for a known-item query (the stores and the encoder read it)."""
    return EvalQuery(query_id=query["query_id"], need_id=query["query_id"], text=query["text"],
                     language=query["language"], type=query["query_type"], stratum="K", need=query["text"])


def record(query_id: str, pairs: Sequence, timings: Mapping[str, float], error: Optional[str] = None,
           **extra: Any) -> Dict[str, Any]:
    """A run line in V14's format (pipeline._record), at most DEPTH results."""
    return {"query_id": query_id, "ids": [str(i) for i, _ in pairs][: D.DEPTH],
            "scores": [None if s is None else float(s) for _, s in pairs][: D.DEPTH],
            "timings_ms": dict(timings), "error": error, "extra": extra}


def head_union(heads: Sequence[Sequence[str]]) -> List[str]:
    """The ids of several heads, each once, in order of first appearance."""
    seen: Dict[str, None] = {}
    for head in heads:
        for book_id in head:
            seen.setdefault(book_id, None)
    return list(seen)


def order_head(ids: Sequence[str], scores: Mapping[str, float], depth: int = D.RERANK_DEPTH) -> List[str]:
    """
    The list with its first `depth` ids sorted by reranker score, best first
    (a stable sort: equal scores keep the list's order, as ApiReranker.rerank),
    and the tail unchanged.
    """
    head = list(ids[:depth])
    ordered = sorted(range(len(head)), key=lambda i: scores[head[i]], reverse=True)
    return [head[i] for i in ordered] + list(ids[depth:])


def chunks(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def estimate_usd(texts_chars: int) -> float:
    """Reranker cost of sending `texts_chars` characters (documents and query), at V14's observed rate."""
    return texts_chars / 1_000_000 * D.RERANK_USD_PER_MILLION_CHARS


# =============================================================================
# Steps
# =============================================================================


def load_sample(data_dir: Path) -> Dict[str, Any]:
    path = data_dir / D.SAMPLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `python -m app.evaluation.known_item sample` first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_database(ctx: Context, queries: Sequence[EvalQuery]) -> None:
    """S1, S2 (depth 100) and S1@60, S2@60, sequentially on one connection, as V14's run_database_systems (without E1)."""
    stores = {name: run_store(ctx.data_dir, name) for name in (D.S1, D.S2, "S1@60", "S2@60")}
    lexical = ctx.lexical()
    approximate = ctx.vector("approximate")
    embeddings = embedding_store(ctx.data_dir, CD.QWEN_EMBEDDING)
    for number, query in enumerate(queries, start=1):
        qid = query.query_id
        if all(qid in store for store in stores.values()):
            continue
        embedding_ms = embeddings.get(qid)["latency_ms"]
        vector = list(map(float, frozen_vector(ctx, CD.QWEN_EMBEDDING, qid)))
        for name, depth, repo, kind in ((D.S1, D.DEPTH, lexical, "lexical"),
                                        ("S1@60", D.FUSION_CANDIDATES, lexical, "lexical"),
                                        (D.S2, D.DEPTH, approximate, "vector"),
                                        ("S2@60", D.FUSION_CANDIDATES, approximate, "vector")):
            if qid in stores[name]:
                continue
            started = time.perf_counter()
            if kind == "lexical":
                results = repo.search(query.text, max_results=depth)
            else:
                results = repo.search(vector, max_results=depth)
            elapsed = ms(time.perf_counter() - started)
            timings = {"lexical": elapsed} if kind == "lexical" else {"embedding": embedding_ms, "vector": elapsed}
            stores[name].add(record(qid, [(str(r.book.id), r.final_score) for r in results], timings))
        if number % 50 == 0:
            ctx.log(f"database: {number}/{len(queries)}")


def run_fusions(data_dir: Path, queries: Sequence[EvalQuery]) -> None:
    """S3 and S3w@w for every w of the grid, from the stored S1@60 and S2@60 (pure, free)."""
    lexical, vector = run_store(data_dir, "S1@60"), run_store(data_dir, "S2@60")
    targets = {D.S3: None, **{D.grid_name(w): w for w in D.WEIGHT_GRID}}
    stores = {name: run_store(data_dir, name) for name in targets}
    for query in queries:
        qid = query.query_id
        if qid not in lexical or qid not in vector:
            continue
        a, b = lexical.get(qid)["ids"], vector.get(qid)["ids"]
        for name, weight in targets.items():
            if qid in stores[name]:
                continue
            fused = rrf_fuse(a, b) if weight is None else weighted_rrf(a, b, weight)
            stores[name].add(record(qid, fused, {}, lexical_weight=weight))


def choose_weight(data_dir: Path, queries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    The weight chosen on the development half (written once to
    weight_selection.json, then read back), and S3w = S3w@w for every query.
    """
    path = data_dir / D.SELECTION_FILE
    if path.exists():
        selection = json.loads(path.read_text(encoding="utf-8"))
    else:
        dev = KA.split(queries, D.DEV)
        runs = {D.grid_name(w): run_store(data_dir, D.grid_name(w)).records for w in D.WEIGHT_GRID}
        selection = {**KA.select_weight(KA.per_query(dev, runs), dev),
                     "chosen": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    source = run_store(data_dir, D.grid_name(selection["chosen_weight"]))
    target = run_store(data_dir, D.S3W)
    for qid, line in source.records.items():
        if qid not in target:
            target.add(line)
    return selection


def rerank_plan(ctx: Context, queries: Sequence[EvalQuery], scored: JsonlStore) -> Dict[str, Any]:
    """Per test query, the books still to score (union of the three heads) and the estimated cost."""
    bases = {name: run_store(ctx.data_dir, base) for name, base in D.RERANKED.items()}
    plan: Dict[str, List[str]] = {}
    chars = 0
    for query in queries:
        known = (scored.get(query.query_id) or {}).get("scores", {})
        heads = [bases[name].get(query.query_id)["ids"][: D.RERANK_DEPTH] for name in D.RERANKED]
        todo = [b for b in head_union(heads) if b not in known]
        plan[query.query_id] = todo
        for book in ctx.catalog.books(todo):
            chars += len(book.get_searchable_text()[: D.RERANK_MAX_DOCUMENT_CHARS]) + len(query.text)
    return {"plan": plan, "books": sum(len(v) for v in plan.values()), "chars": chars,
            "estimated_usd": estimate_usd(chars)}


def _score_query(ctx: Context, reranker: Any, query: EvalQuery, todo: Sequence[str],
                 scored: JsonlStore) -> Optional[str]:
    """Score the books of `todo` for one query in requests of RERANK_CHUNK; returns an error or None."""
    from app.domain.entities import SearchResult

    scores: Dict[str, float] = dict((scored.get(query.query_id) or {}).get("scores", {}))
    errors: List[str] = []
    for part in chunks(todo, D.RERANK_CHUNK):
        books = ctx.catalog.books(part)
        results = [SearchResult(book=b, final_score=1.0, rank=i, source="hybrid") for i, b in enumerate(books, 1)]
        for _ in range(CALL_ATTEMPTS):
            ctx.ledger.check()  # outside the try: reaching the cap stops the run
            try:
                reranked = reranker.rerank(query.text, results)
            except Exception as error:  # noqa: BLE001 - retried, then reported
                errors.append(type(error).__name__)
                continue
            scores.update({str(r.book.id): float(r.rerank_score) for r in reranked})
            break
        else:
            return f"reranker failed {CALL_ATTEMPTS} times: {errors}"
    scored.add({"query_id": query.query_id, "scores": scores, "failed_attempts": errors})
    return None


def run_reranked(ctx: Context, queries: Sequence[EvalQuery], plan: Mapping[str, Sequence[str]]) -> None:
    """S4, S5 and S5w on the test half (one scoring of the union of heads per query)."""
    reranker = ctx.reranker()
    scored = JsonlStore(ctx.data_dir / RERANK_SCORES)
    stores = {name: run_store(ctx.data_dir, name) for name in D.RERANKED}
    bases = {name: run_store(ctx.data_dir, base) for name, base in D.RERANKED.items()}

    def one(query: EvalQuery) -> None:
        qid = query.query_id
        if all(qid in store for store in stores.values()):
            return
        started = time.perf_counter()
        error = _score_query(ctx, reranker, query, plan.get(qid, []), scored) if plan.get(qid) else None
        elapsed = ms(time.perf_counter() - started)
        scores = (scored.get(qid) or {}).get("scores", {})
        for name, store in stores.items():
            base = bases[name].get(qid)
            if error is None:
                ids = order_head(base["ids"], scores)
                head = [scores[b] for b in ids[: D.RERANK_DEPTH]]
                store.add({"query_id": qid, "ids": ids, "scores": head + [None] * (len(ids) - len(head)),
                           "timings_ms": {**base["timings_ms"], "rerank_union": elapsed}, "error": None,
                           "extra": {"base": D.RERANKED[name]}})
            else:  # V14's fallback: the base order is kept and the error recorded
                store.add({"query_id": qid, "ids": base["ids"], "scores": [None] * len(base["ids"]),
                           "timings_ms": dict(base["timings_ms"]), "error": error,
                           "extra": {"base": D.RERANKED[name], "fallback": "base order kept"}})

    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        list(pool.map(one, queries))
    ctx.log(f"reranked systems: {len(queries)} test queries")


def run_all(data_dir: Path, ledger: SpendLedger, log: Callable[[str], None], stage: str = "known-item") -> Dict[str, Any]:
    """Steps 1 to 6 (see the module docstring). Raises EstimateAboveCap or SpendCapReached to stop."""
    sample = load_sample(data_dir)
    queries = sample["queries"]
    all_eval = [eval_query(q) for q in queries]
    test_eval = [eval_query(q) for q in KA.split(queries, D.TEST)]
    ctx = Context(data_dir, ledger, stage, log)
    try:
        embed_queries(ctx, CD.QWEN_EMBEDDING, all_eval)
        missing = [q.query_id for q in all_eval if q.query_id not in embedding_store(data_dir, CD.QWEN_EMBEDDING)]
        if missing:
            raise RuntimeError(f"{len(missing)} queries have no frozen embedding; run again to retry them")
        run_database(ctx, all_eval)
        run_fusions(data_dir, all_eval)
        selection = choose_weight(data_dir, queries)
        log(f"Lexical weight chosen on the development half: {selection['chosen_weight']}")
        scored = JsonlStore(data_dir / RERANK_SCORES)
        estimate = rerank_plan(ctx, test_eval, scored)
        left = ledger.cap_usd - ledger.total()
        (data_dir / ESTIMATE_FILE).write_text(json.dumps(
            {k: v for k, v in estimate.items() if k != "plan"} | {"left_under_cap_usd": left},
            indent=2), encoding="utf-8")
        log(f"Reranking estimate: {estimate['books']} books, {estimate['estimated_usd']:.3f} USD; "
            f"{left:.3f} USD left under the cap")
        if estimate["estimated_usd"] > left:
            raise EstimateAboveCap(f"estimated reranker cost {estimate['estimated_usd']:.3f} USD exceeds the "
                                   f"{left:.3f} USD left under the {ledger.cap_usd:.2f} USD cap; nothing reranked")
        run_reranked(ctx, test_eval, estimate["plan"])
        return {"selection": selection, "estimate": {k: v for k, v in estimate.items() if k != "plan"}}
    finally:
        ctx.close()
