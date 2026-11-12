"""
Stage 3: pooling and judging (protocol section 5).

Pool: for every information need, the union of the top 10 of every pooled
system (S1-S6 with both runs of S6, E1, E2) over the queries of that need
(one query, or the four language versions of a stratum C need). Both
judges grade the whole pool, pointwise, against the need's English
description (stratum A: the query itself), with the V05 prompt and scale
(0-3) through the project's judge adapter. Judgments are cached on disk
with their provenance (model, prompt version, date) in one file per judge
under <data>/judgments/, keyed by (need_id, source, source_id); a re-run
only judges what is missing.
"""

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.ledger import LedgerLLMRecorder, SpendLedger
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.relevance_judge import GRADES, JudgmentCache, LLMJudgment

JUDGE_WORKERS = 8
JUDGE_PASSES = 3  # failed pairs are retried in later passes


def build_pool(queries: Sequence[EvalQuery], runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
               systems: Sequence[str] = D.POOL_SYSTEMS, k: int = D.POOL_K) -> Dict[str, List[str]]:
    """{need_id: work ids in first-seen order} from the top k of every system and query of the need."""
    pool: Dict[str, List[str]] = {}
    seen: Dict[str, Set[str]] = defaultdict(set)
    for query in queries:
        ids = pool.setdefault(query.need_id, [])
        for system in systems:
            record = runs.get(system, {}).get(query.query_id)
            if record is None:
                continue
            for book_id in record["ids"][:k]:
                if book_id not in seen[query.need_id]:
                    seen[query.need_id].add(book_id)
                    ids.append(book_id)
    return pool


def needs_of(queries: Sequence[EvalQuery]) -> Dict[str, EvalQuery]:
    """One representative query per need (the first), carrying the need text."""
    out: Dict[str, EvalQuery] = {}
    for query in queries:
        out.setdefault(query.need_id, query)
    return out


def judge_cache(data_dir: Path, model: str) -> JudgmentCache:
    from app.infrastructure.llm.prompts_judge import RELEVANCE_JUDGE_PROMPT_VERSION

    return JudgmentCache.for_judge(data_dir / "judgments", model, RELEVANCE_JUDGE_PROMPT_VERSION)


def grades_by_need(cache: JudgmentCache, source_ids: Mapping[Tuple[str, str], str]) -> Dict[str, Dict[str, int]]:
    """{need_id: {work_id: grade}} from a cache; source_ids maps (source, source_id) -> work id."""
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for judgment in cache.judgments():
        work_id = source_ids.get((judgment.source, judgment.source_id))
        if work_id is not None:
            out[judgment.query_id][work_id] = judgment.grade
    return dict(out)


def build_judge(model: str, ledger: SpendLedger, stage: str):
    """The project's LangChain judge for `model` (V05 prompt, temperature 0), recording to the ledger."""
    from app.infrastructure.config.settings import JudgeSettings
    from app.infrastructure.llm.langchain_llm_client import build_chat_model
    from app.infrastructure.llm.relevance_judge import RELEVANCE_JUDGE_OPERATION, LangChainRelevanceJudge

    settings = JudgeSettings.from_env({"JUDGE_PROVIDER": "openrouter", "JUDGE_MODEL": model,
                                       "JUDGE_REASONING": D.JUDGE_REASONING})
    recorder = LedgerLLMRecorder(ledger, stage, RELEVANCE_JUDGE_OPERATION)
    chat_model = build_chat_model(settings.provider, settings.model, temperature=settings.temperature,
                                  base_url=settings.base_url, num_ctx=settings.num_ctx, reasoning=settings.reasoning,
                                  max_tokens=settings.max_tokens, timeout_s=settings.timeout_s,
                                  usage_recorder=recorder)
    return LangChainRelevanceJudge(chat_model, settings.model, usage_recorder=recorder), settings


def judge_pool(model: str, tasks: Sequence[Tuple[str, str, Any]], cache: JudgmentCache, ledger: SpendLedger,
               *, stage: str = "3-judge", log: Callable[[str], None] = print,
               workers: int = JUDGE_WORKERS, judge_factory=None) -> Dict[str, Any]:
    """
    Judge every (need_id, need_text, book) task not yet in the cache, with
    `workers` threads (one judge per thread). A failed pair is not cached
    and is retried in the next pass, up to JUDGE_PASSES passes.
    """
    factory = judge_factory or (lambda: build_judge(model, ledger, stage)[0])
    local = threading.local()
    lock = threading.Lock()
    stats = {"failures": 0, "judged": 0, "errors": defaultdict(int)}

    def judge_for_thread():
        if not hasattr(local, "judge"):
            local.judge = factory()
        return local.judge

    def one(task: Tuple[str, str, Any]) -> None:
        need_id, need_text, book = task
        ledger.check()
        judge = judge_for_thread()
        try:
            grade, rationale = judge.judge(need_text, book)
            if grade not in GRADES:
                raise ValueError(f"grade {grade!r}")
        except Exception as error:  # noqa: BLE001 - retried in the next pass
            with lock:
                stats["failures"] += 1
                stats["errors"][type(error).__name__] += 1
            return
        judgment = LLMJudgment(query_id=need_id, source=book.source, source_id=book.source_id or "", grade=grade,
                               rationale=rationale, model=judge.model_name, prompt_version=judge.prompt_version,
                               date=datetime.now(UTC).isoformat(timespec="seconds"), query_text=need_text,
                               title=book.title)
        with lock:
            cache.add(judgment)
            stats["judged"] += 1
            if stats["judged"] % 200 == 0:
                log(f"{model}: {stats['judged']} judged in this run")

    passes = 0
    for passes in range(1, JUDGE_PASSES + 1):
        todo = [t for t in tasks if (t[0], t[2].source, t[2].source_id or "") not in cache]
        if not todo:
            break
        log(f"{model}: pass {passes}, {len(todo)} pairs to judge")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, todo))
    remaining = sum(1 for t in tasks if (t[0], t[2].source, t[2].source_id or "") not in cache)
    return {"model": model, "pairs": len(tasks), "judged_in_run": stats["judged"],
            "failed_calls_in_run": stats["failures"], "errors": dict(stats["errors"]),
            "passes": passes, "unjudged": remaining}
