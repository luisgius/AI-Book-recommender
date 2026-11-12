"""
Sections 8 and 10 (network side).

Section 8, operational measurements: every query is encoded three times by
each embedding model, with the deployed deadlines (3 s per attempt, one
retry, 5 s per call) and no cache, sequentially per model (as a single
user would). Each call is recorded (latency, error) in
<data>/operational/<model>.jsonl keyed "<query_id>#<repetition>", so an
interrupted run continues. The top 10 of each successful vector is the
exact search of that model (E1 in Postgres for qwen, the local E2 index for
bge-m3); a vector bit-identical to an earlier repetition of the same query
reuses that repetition's top 10 instead of scanning again.

Section 10, explanations: 40 queries of A and B drawn with seed 0, the top 3
of S5, an explanation by the system LLM through LangChainLLMClient (the
deployed client); citation precision before the guardrail and the language
of the explanation, as in V10.
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence

import numpy as np

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.pipeline import Context, ms
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.catalog_evaluation.runs import JsonlStore, decode_vector, encode_vector, model_slug, run_store

EXPLAIN_WORKERS = 4


def operational_store(ctx: Context, model: str) -> JsonlStore:
    return JsonlStore(ctx.data_dir / "operational" / f"{model_slug(model)}.jsonl", key="key")


def measure_operational(ctx: Context, model: str, queries: Sequence[EvalQuery]) -> None:
    store = operational_store(ctx, model)
    encoder = ctx.encoder(model, deployed=True)
    if model == D.BGE_EMBEDDING:
        from app.evaluation.catalog_evaluation.e2_index import ExactIndex

        index = ExactIndex.load(D.E2_DIR)
        search = lambda vector: [i for i, _ in index.search(vector, D.CUTOFF)]  # noqa: E731
    else:
        repo = ctx.vector("exact")
        search = lambda vector: [str(r.book.id) for r in repo.search(list(map(float, vector)), D.CUTOFF)]  # noqa: E731
    for repetition in range(1, D.OPERATIONAL_REPETITIONS + 1):
        for query in queries:
            key = f"{query.query_id}#{repetition}"
            if key in store:
                continue
            ctx.ledger.check()
            started = time.perf_counter()
            try:
                vector = encoder.generate_query_embedding(query.text)
                error = None
            except Exception as exc:  # noqa: BLE001 - a failed call is a measurement
                vector, error = None, f"{type(exc).__name__}: {str(exc)[:160]}"
            latency = ms(time.perf_counter() - started)
            top10 = None
            if vector is not None:
                earlier = [store.get(f"{query.query_id}#{r}") for r in range(1, repetition)]
                same = next((e for e in earlier if e and e.get("vector") == encode_vector(vector)), None)
                top10 = same["top10"] if same else search(np.asarray(vector, dtype=np.float32))
            store.add({"key": key, "query_id": query.query_id, "repetition": repetition, "model": model,
                       "latency_ms": latency, "error": error, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "vector": encode_vector(vector) if vector is not None else None, "top10": top10})
        ctx.log(f"operational {model}: repetition {repetition} done")


def explanation_queries(queries: Sequence[EvalQuery]) -> List[EvalQuery]:
    """40 queries of strata A and B drawn with seed 0 (from the list sorted by id)."""
    pool = sorted((q for q in queries if q.stratum in D.STATISTICAL_STRATA), key=lambda q: q.query_id)
    chosen = random.Random(D.SEED).sample(pool, D.EXPLANATION_QUERIES)
    return sorted(chosen, key=lambda q: q.query_id)


def run_explanations(ctx: Context, queries: Sequence[EvalQuery]) -> None:
    from app.evaluation.llm_model_selection import ExplanationTask, build_system_client, explanation_record
    from app.infrastructure.config.settings import LLMSettings
    from app.ingestion.language_detection import detect_language

    store = JsonlStore(ctx.data_dir / "explanations" / "explanations.jsonl", key="key")
    s5 = run_store(ctx.data_dir, D.S5)
    settings = LLMSettings.from_env({"LLM_PROVIDER": "openrouter", "LLM_MODEL": D.SYSTEM_LLM})
    tasks = []
    for query in explanation_queries(queries):
        ids = s5.get(query.query_id)["ids"][: D.EXPLANATION_RESULTS]
        for rank, book in enumerate(ctx.catalog.books(ids), start=1):
            if f"{query.query_id}#{rank}" not in store:
                tasks.append(ExplanationTask(query.query_id, query.language, query.text, rank, book))
    local = threading.local()

    def one(task: ExplanationTask) -> None:
        if not hasattr(local, "client"):
            local.client = build_system_client(D.SYSTEM_LLM, settings.reasoning, [])
            local.raw = []
            local.client.raw_output_observer = lambda operation, output: local.raw.append(output)
        client = local.client
        ctx.ledger.check()
        first = len(client.usage_recorder.records)
        local.raw.clear()
        started = time.perf_counter()
        explanation = client.generate_grounded_explanation(task.query_text, task.book)
        wall = time.perf_counter() - started
        calls = client.usage_recorder.records[first:]
        for call in calls:
            ctx.ledger.add(ctx.stage, call.operation, call.model, call.cost_usd, input_tokens=call.input_tokens,
                           output_tokens=call.output_tokens, latency_s=round(call.latency_s, 3), error=call.error)
        record = explanation_record(task, local.raw[0] if local.raw else None, explanation, calls, wall)
        text = f"{record.get('summary', '')} {record.get('reasoning', '')}".strip()
        record.update(key=f"{task.query_id}#{task.rank}", detected_language=detect_language(text),
                      model=D.SYSTEM_LLM, date=time.strftime("%Y-%m-%dT%H:%M:%S"))
        store.add(record)

    with ThreadPoolExecutor(max_workers=EXPLAIN_WORKERS) as pool:
        list(pool.map(one, tasks))
    ctx.log(f"explanations: {len(tasks)} in this run, {len(store.records)} stored")
