"""
Stage 1: build the request sets (A, B and D from V14; E and F generated).

The generator is V14's (`google/gemini-3.8-flash`, temperature 0.7,
reasoning effort low, JSON output), called through V14's Generator class:
each call is saved under <data>/sets/generation/ before the next one starts
(prompt, model, provider, date, raw answer, reported usage), so a re-run
reuses it and pays nothing twice. As in V14, every request that fails a
mechanical check is sent back to the generator in the same conversation with
the reason, for at most MAX_FIX_ROUNDS rounds, and every replacement is
logged in checks.json.
"""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.assistant_evaluation import sets as S
from app.evaluation.catalog_evaluation.generation import (
    GENERATOR_REASONING,
    GENERATOR_TEMPERATURE,
    Generator,
    catalog_titles,
    find_target,
)
from app.evaluation.catalog_evaluation.ledger import SpendLedger
from app.evaluation.catalog_evaluation.queries import load_queries, parse_json_answer

MAX_FIX_ROUNDS = 3
REQUESTS_FILE = "requests.json"
CHECKS_FILE = "checks.json"


def eligible_works(db_path: Path = D.SOURCE_CATALOG) -> List[S.Work]:
    """Works with a title and a description of at least F_MIN_DESCRIPTION characters, in a B language."""
    out = []
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        for row in conn.execute("SELECT id, title, authors, description, language, categories, published_date FROM books"):
            work_id, title, authors, description, language, categories, published = row
            language = (language or "").lower()
            if not title or len(description or "") < D.F_MIN_DESCRIPTION or language not in D.B_LANGUAGES:
                continue
            out.append(S.Work(work_id=str(work_id), title=title, authors=json.loads(authors or "[]"), language=language,
                              categories=json.loads(categories or "[]"),
                              year=(str(published)[:4] if published else None), description=description))
    return out


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _fix_loop(name: str, generator: Generator, conversation: List[Dict[str, str]], items: Dict[str, Dict[str, Any]],
              check: Callable[[Dict[str, Dict[str, Any]]], Dict[str, Any]], fix_prompt: str,
              log: Callable[[str], None]) -> Dict[str, Any]:
    """Send every request with a problem back to the generator (same conversation), up to MAX_FIX_ROUNDS rounds."""
    fixes: List[Dict[str, Any]] = []
    first = check(items)
    for round_number in range(1, MAX_FIX_ROUNDS + 1):
        result = check(items)
        reasons: Dict[str, List[str]] = {}
        for problem in result["problems"]:
            if problem["request_id"]:
                reasons.setdefault(problem["request_id"], []).append(f"{problem['check']}: {problem['detail']}")
        if not reasons:
            break
        log(f"{name} fix round {round_number}: {len(reasons)} requests")
        for request_id, why in sorted(reasons.items()):
            prompt = fix_prompt.format(id=request_id, reason="; ".join(why))
            record = generator.ask(f"{name}-fix-{round_number}-{request_id}-{_digest(prompt)}",
                                   conversation + [{"role": "user", "content": prompt}])
            try:
                answer = S.parse_requests_answer(parse_json_answer(record["raw_output"]))
                replacement = answer.get(request_id) or next(iter(answer.values()))
            except (ValueError, KeyError, StopIteration, json.JSONDecodeError) as error:
                fixes.append({"round": round_number, "request_id": request_id, "reasons": why, "error": str(error)})
                continue
            fixes.append({"round": round_number, "request_id": request_id, "reasons": why,
                          "before": items.get(request_id), "after": replacement})
            items[request_id] = {**replacement, "id": request_id}
    return {"first_pass": first, "fixes": fixes, "final": check(items)}


def build_sets(out_dir: Path, ledger: SpendLedger, *, generator: Optional[Generator] = None,
               works: Optional[Sequence[S.Work]] = None, titles=None,
               log: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    Write out_dir/requests.json (A, B, D, E, F) and out_dir/checks.json.

    Returns the content of checks.json.
    """
    queries = load_queries(D.V14_QUERIES)
    test_queries = json.loads(D.TEST_QUERIES.read_text(encoding="utf-8"))
    base = S.v14_requests(queries, test_queries)
    generator = generator or Generator(out_dir / "generation", ledger, model=D.GENERATOR_MODEL)

    # --- E ---------------------------------------------------------------------
    b_queries = [q for q in queries if q.stratum == D.SET_B]
    needs = S.sample_e_needs(b_queries)
    needs_by_id = {q.query_id: q for q in needs}
    e_conversation = [{"role": "user", "content": S.e_prompt(needs)}]
    record = generator.ask("E", e_conversation)
    e_conversation = e_conversation + [{"role": "assistant", "content": record["raw_output"]}]
    e_items = S.parse_requests_answer(parse_json_answer(record["raw_output"]))

    def check_e(items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        texts = {k: v.get("text", "") for k, v in items.items()}
        requests = S.e_requests(needs, texts)
        result = S.check_e(requests, needs_by_id)
        for problem in result["problems"]:  # problems name e-<b id>; the generator knows the b id
            if problem["request_id"]:
                problem["request_id"] = problem["request_id"][2:]
        return result

    e_result = _fix_loop("E", generator, e_conversation, e_items, check_e, S.E_FIX_PROMPT, log)
    e_requests = S.e_requests(needs, {k: v.get("text", "") for k, v in e_items.items()})

    # --- F ---------------------------------------------------------------------
    works = list(works) if works is not None else eligible_works()
    titles = titles if titles is not None else catalog_titles(D.SOURCE_CATALOG)
    tasks = S.pick_f_works(works)
    tasks_by_id = {t["id"]: t for t in tasks}
    f_conversation = [{"role": "user", "content": S.f_prompt(tasks)}]
    record = generator.ask("F", f_conversation)
    f_conversation = f_conversation + [{"role": "assistant", "content": record["raw_output"]}]
    f_items = S.parse_requests_answer(parse_json_answer(record["raw_output"]))

    def target(title: str, author: Optional[str]) -> List[str]:
        return find_target(title, author, titles)

    def check_f(items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        return S.check_f(S.f_requests(tasks, items), tasks_by_id, items, target)

    f_result = _fix_loop("F", generator, f_conversation, f_items, check_f, S.F_FIX_PROMPT, log)
    f_requests = S.f_requests(tasks, f_items)

    requests = base + e_requests + f_requests
    agent_sample = S.sample_agent_requests(requests)
    meta = {"protocol": D.PROTOCOL_PATH, "protocol_commit": D.PROTOCOL_COMMIT,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "generator_model": generator.model, "agent_sample": agent_sample}
    S.save_requests(out_dir / REQUESTS_FILE, requests, meta)
    document = {
        "generator": {"model": generator.model, "temperature": GENERATOR_TEMPERATURE,
                      "reasoning": GENERATOR_REASONING,
                      "prompt_versions": {"E": S.E_PROMPT_VERSION, "F": S.F_PROMPT_VERSION}},
        "e_sample": {"seed": D.E_SEED, "b_query_ids": [q.query_id for q in needs]},
        "f_tasks": [{"id": t["id"], "kind": t["kind"], "language": t["language"],
                     "works": [{"work_id": w.work_id, "title": w.title, "authors": w.authors} for w in t["works"]]}
                    for t in tasks],
        "f_eligible_works": len(works),
        "agent_sample": {"seed": D.AGENT_SEED, "request_ids": agent_sample},
        "E": e_result, "F": f_result,
        "sets": S.describe_sets(requests),
    }
    (out_dir / CHECKS_FILE).write_text(json.dumps(document, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return document
