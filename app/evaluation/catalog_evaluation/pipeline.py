"""
Network side of stage 2: Neon (read-only) and OpenRouter.

The cloud adapters are called directly (no public URL, so neither the
visitors' daily budget nor the client's network is involved): Postgres BM25
(PostgresLexicalSearchRepository), the approximate and the exact pgvector
search (PgVectorSearchRepository), the API reranker (ApiReranker) and the
system LLM as listwise re-ranker (LLMRecommender, V07 variant R1). Database
calls are sequential on one connection (the live site shares the compute);
API calls run on a few threads.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.e2_index import ExactIndex, build_vectors
from app.evaluation.catalog_evaluation.ledger import LedgerLLMRecorder, LedgerRecorder, SpendLedger
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.catalog_evaluation.runs import (
    JsonlStore,
    LocalCatalog,
    apply_head_order,
    decode_vector,
    embedding_store,
    encode_vector,
    rrf_fuse,
    run_store,
)

API_WORKERS = 4
LLM_WORKERS = 6
CALL_ATTEMPTS = 3  # attempts of a whole call (each with its own HTTP retries) before a fallback is stored


def ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


class Context:
    """Lazily built adapters shared by the stages."""

    def __init__(self, data_dir: Path, ledger: SpendLedger, stage: str, log: Callable[[str], None]) -> None:
        self.data_dir = data_dir
        self.ledger = ledger
        self.stage = stage
        self.log = log
        self._db = None
        self.catalog = LocalCatalog(D.SOURCE_CATALOG)

    @property
    def db(self):
        if self._db is None:
            from app.infrastructure.config.settings import database_url
            from app.infrastructure.db.postgres_connection import PostgresDatabase

            self._db = PostgresDatabase(database_url(), pool_max_size=1, statement_timeout_ms=120_000)
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()

    def encoder(self, model: str, *, deployed: bool = False):
        """The API encoder of `model`, pinned like the deployment; deployed=True: the deployed query deadlines."""
        from app.infrastructure.search.openrouter_embedding_encoder import OpenRouterEmbeddingEncoder

        dimensions = D.QWEN_DIMENSIONS if model == D.QWEN_EMBEDDING else None
        query_call = ({"timeout_s": D.DEPLOYED_TIMEOUT_S, "max_retries": D.DEPLOYED_RETRIES,
                       "deadline_s": D.DEPLOYED_DEADLINE_S} if deployed else None)
        return OpenRouterEmbeddingEncoder(model, dimensions=dimensions, upstream_provider=D.EMBEDDING_PIN,
                                          usage_recorder=LedgerRecorder(self.ledger, self.stage),
                                          timeout_s=D.EVAL_TIMEOUT_S, max_retries=D.EVAL_RETRIES,
                                          query_call=query_call)

    def reranker(self):
        from app.infrastructure.search.api_reranker import ApiReranker

        return ApiReranker(D.RERANKER, upstream_provider=D.RERANKER_PIN,
                           usage_recorder=LedgerRecorder(self.ledger, self.stage),
                           timeout_s=D.EVAL_TIMEOUT_S, max_retries=D.EVAL_RETRIES)

    def recommender(self):
        from app.infrastructure.config.settings import LLMSettings
        from app.infrastructure.llm.langchain_llm_client import build_chat_model
        from app.infrastructure.llm.llm_recommender import (
            RECOMMENDER_MAX_TOKENS,
            RERANK_OPERATION,
            LangChainCompletion,
            LLMRecommender,
        )

        settings = LLMSettings.from_env({"LLM_PROVIDER": "openrouter", "LLM_MODEL": D.SYSTEM_LLM})
        recorder = LedgerLLMRecorder(self.ledger, self.stage, RERANK_OPERATION)
        chat_model = build_chat_model(settings.provider, settings.model, temperature=0.0,
                                      base_url=settings.base_url, num_ctx=settings.num_ctx,
                                      reasoning=settings.reasoning, max_tokens=RECOMMENDER_MAX_TOKENS,
                                      timeout_s=settings.timeout_s, usage_recorder=recorder)
        return LLMRecommender(LangChainCompletion(chat_model, usage_recorder=recorder), settings.model)

    def lexical(self):
        from app.infrastructure.search.postgres_lexical_search_repository import PostgresLexicalSearchRepository

        return PostgresLexicalSearchRepository(self.db, D.SCHEMA, ranking="bm25")

    def vector(self, mode: str):
        from app.infrastructure.search.openrouter_embedding_encoder import api_model_identity
        from app.infrastructure.search.pgvector_search_repository import PgVectorSearchRepository

        identity = api_model_identity(D.QWEN_EMBEDDING, D.QWEN_DIMENSIONS, D.EMBEDDING_PIN)
        return PgVectorSearchRepository(self.db, D.SCHEMA, mode=mode, expected_identity=identity)


# =============================================================================
# Frozen query embeddings
# =============================================================================


def embed_queries(ctx: Context, model: str, queries: Sequence[EvalQuery]) -> Dict[str, Any]:
    """Encode every query not yet stored, once, with the long evaluation deadlines."""
    store = embedding_store(ctx.data_dir, model)
    encoder = ctx.encoder(model)
    todo = [q for q in queries if q.query_id not in store]
    failures = []

    def one(query: EvalQuery) -> None:
        for attempt in range(1, CALL_ATTEMPTS + 1):
            ctx.ledger.check()
            started = time.perf_counter()
            try:
                vector = encoder.generate_query_embedding(query.text)
            except Exception as error:  # noqa: BLE001 - retried, then reported
                failures.append({"query_id": query.query_id, "attempt": attempt, "error": type(error).__name__})
                continue
            store.add({"query_id": query.query_id, "model": model, "provider": D.EMBEDDING_PIN,
                       "dimension": len(vector), "latency_ms": ms(time.perf_counter() - started),
                       "attempts": attempt, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "vector": encode_vector(vector)})
            return

    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        list(pool.map(one, todo))
    ctx.log(f"{model}: {len(todo)} queries encoded in this run, {len(store.records)} stored, "
            f"{len(failures)} failed attempts")
    return {"encoded": len(todo), "stored": len(store.records), "failed_attempts": failures}


def frozen_vector(ctx: Context, model: str, query_id: str) -> np.ndarray:
    record = embedding_store(ctx.data_dir, model).get(query_id)
    if record is None:
        raise KeyError(f"no frozen {model} embedding for {query_id}: run the embed stage")
    return decode_vector(record["vector"])


def build_e2(ctx: Context) -> int:
    encoder = ctx.encoder(D.BGE_EMBEDDING)
    books = list(ctx.catalog.iter_all())
    return build_vectors(books, encoder, D.E2_DIR, check=ctx.ledger.check, log=ctx.log)


# =============================================================================
# Run files
# =============================================================================


def _record(query_id: str, pairs: Sequence, timings: Dict[str, float], error: Optional[str] = None,
            **extra: Any) -> Dict[str, Any]:
    return {"query_id": query_id, "ids": [str(i) for i, _ in pairs][: D.DEPTH],
            "scores": [None if s is None else float(s) for _, s in pairs][: D.DEPTH],
            "timings_ms": timings, "error": error, "extra": extra}


def run_database_systems(ctx: Context, queries: Sequence[EvalQuery]) -> None:
    """S1, S2, E1 (Postgres), the fusion inputs and S3 (pure), sequentially, one query at a time."""
    stores = {name: run_store(ctx.data_dir, name) for name in (D.S1, D.S2, D.E1, D.S3, "S1@60", "S2@60")}
    lexical = ctx.lexical()
    approximate = ctx.vector("approximate")
    exact = ctx.vector("exact")
    embeddings = embedding_store(ctx.data_dir, D.QWEN_EMBEDDING)
    for number, query in enumerate(queries, start=1):
        qid = query.query_id
        if all(qid in store for store in stores.values()):
            continue
        embedding_ms = embeddings.get(qid)["latency_ms"]
        vector = list(map(float, frozen_vector(ctx, D.QWEN_EMBEDDING, qid)))
        out: Dict[str, Dict[str, Any]] = {}
        for name, depth, repo, kind in ((D.S1, D.DEPTH, lexical, "lexical"), ("S1@60", D.FUSION_CANDIDATES, lexical, "lexical"),
                                        (D.S2, D.DEPTH, approximate, "vector"), ("S2@60", D.FUSION_CANDIDATES, approximate, "vector"),
                                        (D.E1, D.DEPTH, exact, "vector")):
            if qid in stores[name]:
                out[name] = stores[name].get(qid)
                continue
            started = time.perf_counter()
            if kind == "lexical":
                results = repo.search(query.text, max_results=depth)
            else:
                results = repo.search(vector, max_results=depth)
            elapsed = ms(time.perf_counter() - started)
            timings = {"lexical": elapsed} if kind == "lexical" else {"embedding": embedding_ms, "vector": elapsed}
            record = _record(qid, [(str(r.book.id), r.final_score) for r in results], timings)
            stores[name].add(record)
            out[name] = record
        if qid not in stores[D.S3]:
            started = time.perf_counter()
            fused = rrf_fuse(out["S1@60"]["ids"], out["S2@60"]["ids"])
            timings = {"lexical": out["S1@60"]["timings_ms"]["lexical"], "embedding": embedding_ms,
                       "vector": out["S2@60"]["timings_ms"]["vector"], "fusion": ms(time.perf_counter() - started)}
            stores[D.S3].add(_record(qid, fused, timings))
        if number % 10 == 0:
            ctx.log(f"database systems: {number}/{len(queries)}")


def run_e2(ctx: Context, queries: Sequence[EvalQuery]) -> None:
    store = run_store(ctx.data_dir, D.E2)
    todo = [q for q in queries if q.query_id not in store]
    if not todo:
        return
    index = ExactIndex.load(D.E2_DIR)
    embeddings = embedding_store(ctx.data_dir, D.BGE_EMBEDDING)
    for query in todo:
        vector = frozen_vector(ctx, D.BGE_EMBEDDING, query.query_id)
        started = time.perf_counter()
        pairs = index.search(vector, D.DEPTH)
        store.add(_record(query.query_id, pairs, {"embedding": embeddings.get(query.query_id)["latency_ms"],
                                                    "vector": ms(time.perf_counter() - started)}))


def _rerank_one(ctx: Context, reranker: Any, query: EvalQuery, base: Dict[str, Any]) -> Dict[str, Any]:
    """Rerank the head of `base` with the API reranker; after CALL_ATTEMPTS failures keep the base order."""
    from app.domain.entities import SearchResult

    head_ids = base["ids"][: D.RERANK_DEPTH]
    books = ctx.catalog.books(head_ids)
    results = [SearchResult(book=b, final_score=1.0, rank=i, source="hybrid") for i, b in enumerate(books, start=1)]
    errors = []
    for _ in range(CALL_ATTEMPTS):
        ctx.ledger.check()
        started = time.perf_counter()
        try:
            reranked = reranker.rerank(query.text, results)
        except Exception as error:  # noqa: BLE001
            errors.append(type(error).__name__)
            continue
        elapsed = ms(time.perf_counter() - started)
        position = {book_id: i for i, book_id in enumerate(head_ids)}
        order = [position[str(r.book.id)] for r in reranked]
        ids = apply_head_order(base["ids"], order, D.RERANK_DEPTH)
        head_scores = [r.rerank_score for r in reranked]
        scores = head_scores + [None] * (len(ids) - len(head_scores))
        return {"query_id": query.query_id, "ids": ids, "scores": scores,
                "timings_ms": {**base["timings_ms"], "rerank": elapsed}, "error": None,
                "extra": {"failed_attempts": errors, "top_rerank_score": head_scores[0] if head_scores else None}}
    return {"query_id": query.query_id, "ids": base["ids"], "scores": [None] * len(base["ids"]),
            "timings_ms": dict(base["timings_ms"]), "error": f"reranker failed {CALL_ATTEMPTS} times: {errors}",
            "extra": {"failed_attempts": errors, "top_rerank_score": None, "fallback": "base order kept"}}


def run_reranked(ctx: Context, queries: Sequence[EvalQuery]) -> None:
    """S4 = S2 + reranker, S5 = S3 + reranker (top 30)."""
    reranker = ctx.reranker()
    for name, base_name in ((D.S4, D.S2), (D.S5, D.S3)):
        store = run_store(ctx.data_dir, name)
        base = run_store(ctx.data_dir, base_name)
        todo = [q for q in queries if q.query_id not in store and q.query_id in base]

        def one(query: EvalQuery, store=store, base=base) -> None:
            store.add(_rerank_one(ctx, reranker, query, base.get(query.query_id)))

        with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
            list(pool.map(one, todo))
        ctx.log(f"{name}: {len(todo)} queries in this run")


def _llm_one(ctx: Context, recommender: Any, query: EvalQuery, base: Dict[str, Any]) -> Dict[str, Any]:
    head_ids = base["ids"][: D.RERANK_DEPTH]
    books = ctx.catalog.books(head_ids)
    errors = []
    for _ in range(CALL_ATTEMPTS):
        ctx.ledger.check()
        started = time.perf_counter()
        try:
            outcome = recommender.rank_books(query.text, books)
        except Exception as error:  # noqa: BLE001 - request failed
            errors.append(type(error).__name__)
            continue
        elapsed = ms(time.perf_counter() - started)
        ids = apply_head_order(base["ids"], outcome.order, D.RERANK_DEPTH)
        output = outcome.output
        return {"query_id": query.query_id, "ids": ids, "scores": [None] * len(ids),
                "timings_ms": {**base["timings_ms"], "llm": elapsed}, "error": None,
                "extra": {"failed_attempts": errors, "answer_failed": outcome.failed, "valid_ids": len(outcome.valid_ids),
                          "unknown_ids": len(outcome.unknown_ids), "duplicate_ids": len(outcome.duplicate_ids),
                          "filled": outcome.filled, "output_error": output.error,
                          "input_tokens": output.input_tokens, "output_tokens": output.output_tokens,
                          "cost_usd": output.cost_usd}}
    return {"query_id": query.query_id, "ids": base["ids"], "scores": [None] * len(base["ids"]),
            "timings_ms": dict(base["timings_ms"]), "error": f"LLM request failed {CALL_ATTEMPTS} times: {errors}",
            "extra": {"failed_attempts": errors, "fallback": "fused order kept"}}


def run_llm(ctx: Context, queries: Sequence[EvalQuery]) -> None:
    """S6 and S6b: the system LLM re-orders the top 30 of S3 (two independent runs)."""
    recommender = ctx.recommender()
    base = run_store(ctx.data_dir, D.S3)
    for name in (D.S6, D.S6B):
        store = run_store(ctx.data_dir, name)
        todo = [q for q in queries if q.query_id not in store and q.query_id in base]

        def one(query: EvalQuery, store=store) -> None:
            store.add(_llm_one(ctx, recommender, query, base.get(query.query_id)))

        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            list(pool.map(one, todo))
        ctx.log(f"{name}: {len(todo)} queries in this run")


def explain_exact_scan(ctx: Context) -> Dict[str, Any]:
    """EXPLAIN of the E1 statement with the exact settings: it must be a sequential scan, no ANN index."""
    repo = ctx.vector("exact")
    vector = [0.0] * repo.dimension()
    vector[0] = 1.0
    sql, params, settings = repo.build_sql(D.DEPTH, None, exact=True)
    from app.infrastructure.db.postgres_rows import vector_literal

    params["vector"] = vector_literal(vector)
    plan = [row["QUERY PLAN"] for row in ctx.db.query("EXPLAIN " + sql, params, settings=settings)]
    approx_sql, approx_params, approx_settings = ctx.vector("approximate").build_sql(D.DEPTH, None, exact=False)
    approx_params["vector"] = params["vector"]
    approx_plan = [row["QUERY PLAN"] for row in ctx.db.query("EXPLAIN " + approx_sql, approx_params,
                                                                 settings=approx_settings)]
    text = "\n".join(plan)
    return {"exact_settings": {k: str(v) for k, v in settings.items()}, "exact_plan": plan,
            "exact_is_sequential_scan": "Seq Scan" in text and "works_embedding_ann" not in text,
            "approximate_plan": approx_plan,
            "approximate_uses_ann_index": "works_embedding_ann" in "\n".join(approx_plan)}
