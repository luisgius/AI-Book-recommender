"""
Stage 1: generate strata B, C and D with an LLM and check them mechanically.

The generator is a flash-tier model of a vendor the evaluation uses nowhere
else (Google; chosen from OpenRouter's live model list). It is given the
catalog's language shares and its most frequent subjects, never a system
output. Each call is saved before the next one starts (prompt, model, date,
raw answer, reported cost), so a re-run reuses it and pays nothing twice.

After the first pass every problem found by check_queries, and every
known-item query whose target work is not in the catalog, is sent back to
the generator with the reason, for at most MAX_FIX_ROUNDS rounds. Every
replacement is logged and reported.
"""

import hashlib
import json
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.evaluation.catalog_evaluation.design import (
    B_LANGUAGES,
    CATALOG_WORKS,
    EVAL_RETRIES,
    EVAL_TIMEOUT_S,
    LANGUAGE_NAMES,
    SOURCE_CATALOG,
    STRATUM_B,
    STRATUM_C,
    STRATUM_D,
)
from app.evaluation.catalog_evaluation.ledger import SpendLedger
from app.evaluation.catalog_evaluation.queries import (
    FIX_PROMPT,
    EvalQuery,
    b_prompt,
    b_queries,
    c_prompt,
    c_queries,
    check_queries,
    d_prompt,
    d_queries,
    fold,
    parse_json_answer,
    stratum_a,
)

GENERATOR_MODEL = "google/gemini-3.8-flash"
GENERATOR_TEMPERATURE = 0.7
GENERATOR_REASONING = {"effort": "low"}
MAX_FIX_ROUNDS = 3
SUBJECTS_SHOWN = 60
STAGE = "1-queries"


# =============================================================================
# Catalog facts given to the generator (languages and subjects only)
# =============================================================================


def catalog_facts(db_path: Path = SOURCE_CATALOG, subjects: int = SUBJECTS_SHOWN) -> Dict[str, Any]:
    """Language shares of the works and the most frequent subjects, from the catalog."""
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        languages = Counter()
        categories = Counter()
        total = 0
        for language, raw in conn.execute("SELECT language, categories FROM books"):
            total += 1
            languages[(language or "unknown").lower()] += 1
            for category in json.loads(raw or "[]"):
                categories[category] += 1
    shares = {code: languages[code] / total for code in B_LANGUAGES}
    return {"works": total, "language_shares": shares,
            "subjects": [name for name, _ in categories.most_common(subjects)]}


def catalog_titles(db_path: Path = SOURCE_CATALOG) -> List[Tuple[str, str, List[str]]]:
    """(id, title, authors) of every work, for the known-item membership check."""
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        return [(row[0], row[1], json.loads(row[2] or "[]"))
                for row in conn.execute("SELECT id, title, authors FROM books")]


def find_target(title: Optional[str], author: Optional[str], works: Sequence[Tuple[str, str, List[str]]]) -> List[str]:
    """
    Catalog works that are the known item.

    A work matches when its folded title equals the folded target title, or
    when the catalog title starts with the target title followed by a space
    or a colon (subtitles) and the last word of the target author's name
    appears in one of the work's authors. The author is not required for an
    exact title match, because the catalog often stores authors transliterated
    (e.g. "Fyodor Dostoyevsky" for a title in Cyrillic).
    """
    if not title:
        return []
    target = fold(title).strip(" .")
    surname = fold(author).split()[-1] if author and author.split() else None
    found = []
    for work_id, work_title, authors in works:
        candidate = fold(work_title).strip(" .")
        if candidate == target:
            found.append(work_id)
            continue
        prefix = len(target) >= 4 and (candidate.startswith(target + " ") or candidate.startswith(target + ":"))
        if prefix and surname and any(surname in fold(name) for name in authors):
            found.append(work_id)
    return found


# =============================================================================
# Calls (cached on disk)
# =============================================================================


class Generator:
    """Chat calls to the generator through OpenRouterHttp (no identifying header), cached per call id."""

    def __init__(self, directory: Path, ledger: SpendLedger, http: Any = None, model: str = GENERATOR_MODEL) -> None:
        self.directory = directory
        self.ledger = ledger
        self.model = model
        self._http = http

    def _client(self):
        if self._http is None:
            from app.infrastructure.search.openrouter_http import OpenRouterHttp

            self._http = OpenRouterHttp(timeout_s=120.0, max_retries=EVAL_RETRIES)
        return self._http

    def ask(self, call_id: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """The saved call `call_id`, or a new call saved under that id."""
        path = self.directory / f"{call_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        self.ledger.check()
        body = {"model": self.model, "messages": messages, "temperature": GENERATOR_TEMPERATURE,
                "reasoning": GENERATOR_REASONING, "response_format": {"type": "json_object"}}
        started = time.perf_counter()
        response = self._client().post("chat/completions", body)
        payload = response.body
        usage = payload.get("usage") or {}
        cost = usage.get("cost")
        content = payload["choices"][0]["message"].get("content") or ""
        record = {
            "call_id": call_id, "model": self.model, "provider": payload.get("provider"),
            "date": datetime.now(UTC).isoformat(timespec="seconds"), "request": body,
            "raw_output": content, "usage": usage, "latency_s": round(time.perf_counter() - started, 3),
        }
        self.ledger.add(STAGE, "query_generation", self.model, float(cost) if cost is not None else None,
                        input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record


def _first_pass(generator: Generator, facts: Mapping[str, Any]) -> Tuple[List[EvalQuery], Dict[str, List[Dict[str, str]]]]:
    shares, subjects, works = facts["language_shares"], facts["subjects"], facts["works"]
    conversations: Dict[str, List[Dict[str, str]]] = {}
    queries: List[EvalQuery] = []
    for language in B_LANGUAGES:
        messages = [{"role": "user", "content": b_prompt(language, shares, subjects, works)}]
        record = generator.ask(f"B-{language}", messages)
        conversations[f"B-{language}"] = messages + [{"role": "assistant", "content": record["raw_output"]}]
        queries += b_queries(language, parse_json_answer(record["raw_output"]))
    messages = [{"role": "user", "content": c_prompt(shares, subjects, works)}]
    record = generator.ask("C", messages)
    conversations["C"] = messages + [{"role": "assistant", "content": record["raw_output"]}]
    queries += c_queries(parse_json_answer(record["raw_output"]))
    messages = [{"role": "user", "content": d_prompt(shares, subjects, works)}]
    record = generator.ask("D", messages)
    conversations["D"] = messages + [{"role": "assistant", "content": record["raw_output"]}]
    queries += d_queries(parse_json_answer(record["raw_output"]))
    return queries, conversations


def _conversation_of(query: EvalQuery) -> str:
    if query.stratum == STRATUM_B:
        return f"B-{query.language}"
    return "C" if query.stratum == STRATUM_C else "D"


def _item_of(query: EvalQuery) -> Dict[str, Any]:
    item: Dict[str, Any] = {"type": query.type, "text": query.text, "need": query.need}
    if query.type == "known_item":
        item.update(target_title=query.target_title, target_author=query.target_author)
    if query.stratum == STRATUM_D:
        item["language"] = query.language
    return item


def _replacement(query: EvalQuery, answer: Mapping[str, Any]) -> EvalQuery:
    """The query rebuilt from a one-item fix answer (same id, language, stratum and need id)."""
    items = answer.get("queries") or answer.get("needs") or []
    if not items:
        raise ValueError("empty fix answer")
    item = items[0]
    if query.stratum == STRATUM_C:  # a C fix rewrites one language version of the same need
        text = item.get(query.language) or item.get("text") or ""
        return EvalQuery(query.query_id, query.need_id, " ".join(str(text).split()), query.language, query.type,
                         query.stratum, query.need, notes=list(query.notes))
    return EvalQuery(
        query.query_id, query.need_id, " ".join(str(item.get("text", "")).split()), query.language,
        query.type if query.stratum == STRATUM_D else str(item.get("type", query.type)), query.stratum,
        " ".join(str(item.get("need", query.need)).split()),
        target_title=item.get("target_title") if query.type == "known_item" else None,
        target_author=item.get("target_author") if query.type == "known_item" else None,
        notes=list(query.notes),
    )


def known_item_targets(queries: Sequence[EvalQuery], works) -> Dict[str, List[str]]:
    return {q.query_id: find_target(q.target_title, q.target_author, works) for q in queries if q.type == "known_item"}


def generate_queries(
    test_queries: Sequence[Mapping[str, Any]],
    out_dir: Path,
    ledger: SpendLedger,
    *,
    generator: Optional[Generator] = None,
    facts: Optional[Mapping[str, Any]] = None,
    works=None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Build the query file: A unchanged, B, C and D generated, checked and fixed.

    Writes out_dir/queries.json (the query set), out_dir/checks.json (checks
    of the first pass and of the final set, every fix) and the generator
    calls under out_dir/generation/. Returns the content of checks.json.
    """
    facts = dict(facts or catalog_facts())
    works = works if works is not None else catalog_titles()
    generator = generator or Generator(out_dir / "generation", ledger)
    generated, conversations = _first_pass(generator, facts)
    queries = stratum_a(test_queries) + generated
    first_checks = check_queries(queries)
    targets = known_item_targets(queries, works)
    fixes: List[Dict[str, Any]] = []

    for round_number in range(1, MAX_FIX_ROUNDS + 1):
        checks = check_queries(queries)
        targets = known_item_targets(queries, works)
        reasons: Dict[str, List[str]] = {}
        for problem in checks["problems"]:
            if problem["query_id"] and problem["check"] in ("language", "title words", "duplicate", "empty", "known item"):
                reasons.setdefault(problem["query_id"], []).append(f"{problem['check']}: {problem['detail']}")
        for query_id, found in targets.items():
            if not found:
                reasons.setdefault(query_id, []).append(
                    "known item: the work named in target_title / target_author was not found in the catalog by "
                    "title and author; choose a more widely known work, and give its title exactly as published")
        if not reasons:
            break
        log(f"Fix round {round_number}: {len(reasons)} queries")
        index = {q.query_id: i for i, q in enumerate(queries)}
        for query_id, why in sorted(reasons.items()):
            query = queries[index[query_id]]
            if query.stratum == "A":
                continue
            conversation = conversations[_conversation_of(query)]
            prompt = FIX_PROMPT.format(reason="; ".join(why), item=json.dumps(_item_of(query), ensure_ascii=False),
                                       language_name=LANGUAGE_NAMES.get(query.language, query.language))
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
            record = generator.ask(f"fix-{round_number}-{query_id}-{digest}",
                                   conversation + [{"role": "user", "content": prompt}])
            try:
                replacement = _replacement(query, parse_json_answer(record["raw_output"]))
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                fixes.append({"round": round_number, "query_id": query_id, "reasons": why, "error": str(error)})
                continue
            replacement = EvalQuery(**{**replacement.to_dict(),
                                       "notes": replacement.notes + [f"replaced in fix round {round_number}: {'; '.join(why)}"]})
            fixes.append({"round": round_number, "query_id": query_id, "reasons": why,
                          "before": _item_of(query), "after": _item_of(replacement)})
            queries[index[query_id]] = replacement

    final_checks = check_queries(queries)
    targets = known_item_targets(queries, works)
    document = {
        "generator": {"model": generator.model, "temperature": GENERATOR_TEMPERATURE,
                      "reasoning": GENERATOR_REASONING, "prompt_version": "v1"},
        "catalog_facts_shown": facts,
        "first_pass_checks": first_checks,
        "fixes": fixes,
        "final_checks": final_checks,
        "known_item_targets": targets,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queries.json").write_text(json.dumps(
        {"protocol": "docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md",
         "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
         "generator_model": generator.model,
         "queries": [q.to_dict() for q in queries]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "checks.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document
