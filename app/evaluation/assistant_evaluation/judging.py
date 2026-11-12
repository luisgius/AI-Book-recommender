"""
Stage 4: judgments (protocol section 4).

The pairs to judge are the (need, work) pairs the measures read:
- A and B: the top 10 of P on the request and of the assistant's run-1 reply,
  and every work it recommended;
- the agent called directly: every work it recommended;
- E: the top 10 of the assistant's last reply of the conversation and its
  recommendations, the top 10 of P on the vague request and of P on the
  original B query, all against the hidden B need;
- F: the top 10 of the assistant's reply and its recommendations, and the
  top 10 of P, against the request itself.
D is measured by its route only.

A pair V14 already judged (same need id, so the same need text, prompt and
judge) is reused from V14's caches, which are only read. Every other pair is
judged by both of V14's judges, with V14's judge adapter and settings, into
separate caches under <data>/judgments/. judgments/pairs.json lists, per
need, the works the measures read and V14's pooled works, with their source
ids, so the report maps judgments to works without opening the catalog.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.assistant_evaluation import runs as R
from app.evaluation.assistant_evaluation.analysis import last_reply, reply_list, reply_picks
from app.evaluation.assistant_evaluation.sets import Request, need_texts
from app.evaluation.catalog_evaluation.judging import judge_cache, judge_pool
from app.evaluation.catalog_evaluation.ledger import SpendLedger
from app.evaluation.relevance_judge import JudgmentCache

PAIRS_FILE = "pairs.json"
JUDGING_RUNS = "judging_runs.jsonl"


def v14_cache(model: str) -> JudgmentCache:
    """V14's cache of `model` (read only: nothing is ever added to it here)."""
    return judge_cache(D.V14_DIR, model)


def new_cache(data_dir: Path, model: str) -> JudgmentCache:
    return judge_cache(data_dir, model)


def first_measurement_cache(data_dir: Path, model: str) -> JudgmentCache | None:
    """
    The first measurement's cache of `model` when data_dir is the re-measurement's (V25), else None.

    Read only: the re-measurement reuses the pairs the first one judged
    (same need id, prompt and judge) and judges only new pairs, into its own
    cache.
    """
    return judge_cache(D.DATA_DIR, model) if D.is_v25(data_dir) else None


def pairs_needed(requests: Sequence[Request], data: Mapping[str, Any]) -> Dict[str, List[str]]:
    """{need_id: work ids in first-seen order} that the measures read (see the module docstring)."""
    out: Dict[str, List[str]] = {}

    def add(need_id: str, ids: Sequence[str]) -> None:
        bucket = out.setdefault(need_id, [])
        for work_id in ids:
            if work_id not in bucket:
                bucket.append(work_id)

    p = data["P"]
    turns = data["turns"]
    conversations = data["conversations"]
    agent = data["agent"]
    by_id = {r.request_id: r for r in requests}
    for r in requests:
        if r.set == D.SET_D:
            continue
        if r.set in (D.SET_A, D.SET_B, D.SET_F):
            add(r.need_id, (p.get(r.text) or {}).get("ids", [])[: D.CUTOFF])
            reply = (turns.get(R.turn_key(D.RUN_1, r.request_id, 1)) or {}).get("reply")
            add(r.need_id, reply_list(reply)[: D.CUTOFF])
            add(r.need_id, reply_picks(reply))
        if r.set == D.SET_E:
            add(r.need_id, (p.get(r.text) or {}).get("ids", [])[: D.CUTOFF])
            add(r.need_id, (p.get(r.source_query or "") or {}).get("ids", [])[: D.CUTOFF])
            reply = last_reply(conversations.get(r.request_id), turns, r.request_id)
            add(r.need_id, reply_list(reply)[: D.CUTOFF])
            add(r.need_id, reply_picks(reply))
    for request_id, record in agent.items():
        request = by_id.get(request_id)
        if request is not None and record.get("agent"):
            add(request.need_id, [x["work_id"] for x in record["agent"]["recommendations"]])
    return out


def load_run_data(data_dir: Path) -> Dict[str, Any]:
    return {"P": R.load_store(data_dir, R.P_FILE, "text"), "turns": R.load_store(data_dir, R.TURNS_FILE, "key"),
            "conversations": R.load_store(data_dir, R.CONVERSATIONS_FILE, "request_id"),
            "agent": R.load_store(data_dir, R.AGENT_FILE, "request_id")}


def judge_all(data_dir: Path, requests: Sequence[Request], ledger: SpendLedger, *,
              log: Callable[[str], None] = print, judge_factory=None) -> Dict[str, Any]:
    from app.evaluation.catalog_evaluation.runs import LocalCatalog

    needed = pairs_needed(requests, load_run_data(data_dir))
    texts = need_texts(requests)
    v14_pool = json.loads(D.V14_POOL.read_text(encoding="utf-8"))
    catalog = LocalCatalog(D.SOURCE_CATALOG)
    document: Dict[str, Any] = {}
    books: Dict[Tuple[str, str], Any] = {}
    for need_id, ids in needed.items():
        found = catalog.books(ids)
        works = {w["work_id"]: w for w in v14_pool.get(need_id, [])}
        for work_id, book in zip(ids, found):
            books[(need_id, work_id)] = book
            works.setdefault(work_id, {"work_id": work_id, "source": book.source, "source_id": book.source_id or ""})
        document[need_id] = {"need": texts[need_id], "measured": list(ids), "works": list(works.values())}
    out_dir = data_dir / "judgments"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PAIRS_FILE).write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")

    stats_all: Dict[str, Any] = {}

    def judge(model: str) -> None:
        old = v14_cache(model)
        first = first_measurement_cache(data_dir, model)
        cache = new_cache(data_dir, model)
        tasks = []
        reused = reused_first = 0
        for (need_id, work_id), book in books.items():
            key = (need_id, book.source, book.source_id or "")
            if key in old:
                reused += 1
                continue
            if first is not None and key in first:
                reused_first += 1
                continue
            tasks.append((need_id, texts[need_id], book))
        log(f"{model}: {len(books)} pairs measured, {reused} judged in V14, {reused_first} judged in the first "
            f"measurement, {len(tasks)} to judge here")
        stats = judge_pool(model, tasks, cache, ledger, stage="4-judge", log=log, judge_factory=judge_factory)
        stats.update({"reused_from_v14": reused, "measured_pairs": len(books),
                      "date": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if first is not None:
            stats["reused_from_first_measurement"] = reused_first
        with (out_dir / JUDGING_RUNS).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats) + "\n")
        stats_all[model] = stats
        log(json.dumps(stats))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(judge, D.JUDGES))
    return stats_all


def load_grades(data_dir: Path, model: str, pairs: Mapping[str, Any],
                v14: JudgmentCache | None = None, new: JudgmentCache | None = None) -> Dict[str, Dict[str, int]]:
    """{need_id: {work_id: grade}} of one judge over the works of pairs.json (V14's judgments, then this task's)."""
    v14 = v14 if v14 is not None else v14_cache(model)
    new = new if new is not None else new_cache(data_dir, model)
    first = first_measurement_cache(data_dir, model)
    out: Dict[str, Dict[str, int]] = {}
    for need_id, entry in pairs.items():
        grades: Dict[str, int] = {}
        for work in entry["works"]:
            key = (need_id, work["source"], work["source_id"])
            judgment = v14.get(key) or (first.get(key) if first is not None else None) or new.get(key)
            if judgment is not None:
                grades[work["work_id"]] = judgment.grade
        out[need_id] = grades
    return out


def judged_counts(data_dir: Path, pairs: Mapping[str, Any]) -> Dict[str, Any]:
    """Per judge: measured pairs, how many come from V14 and how many were judged here, and the unjudged."""
    out: Dict[str, Any] = {}
    for model in D.JUDGES:
        v14, new = v14_cache(model), new_cache(data_dir, model)
        first = first_measurement_cache(data_dir, model)
        counts = {"measured_pairs": 0, "from_v14": 0, "judged_here": 0, "unjudged": 0}
        if first is not None:
            counts["from_first_measurement"] = 0
        source: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for need_id, entry in pairs.items():
            for work in entry["works"]:
                source[(need_id, work["work_id"])] = (work["source"], work["source_id"])
            for work_id in entry["measured"]:
                counts["measured_pairs"] += 1
                key = (need_id, *source[(need_id, work_id)])
                if key in v14:
                    counts["from_v14"] += 1
                elif first is not None and key in first:
                    counts["from_first_measurement"] += 1
                elif key in new:
                    counts["judged_here"] += 1
                else:
                    counts["unjudged"] += 1
        counts["cache_size_here"] = len(new)
        out[model] = counts
    return out


def needs_with_work(pairs: Mapping[str, Any]) -> Set[str]:
    return set(pairs)
