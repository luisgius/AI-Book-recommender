"""
Request sets of the evaluation (protocol section 3): the pure parts.

- A, B and D are V14's strata, unchanged (A carries its category, so the ten
  "ambiguous" queries can be reported apart).
- E: thirty needs of stratum B sampled with seed 0; for each one V14's
  generator writes a vague request of one to three words in the need's
  language. The need stays hidden from the assistant; the simulated user and
  the judges see it.
- F: twenty multi-step requests (compare two works, "something like X but
  ...", facts about one work) written by the generator from catalog records
  of works that are in the catalog, without seeing any system output.

This module holds the request record, the sampling, the prompts, the parsing
and the mechanical checks. The network calls are in generation.py.
"""

import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.catalog_evaluation.queries import (
    STOPWORDS,
    EvalQuery,
    fold,
    language_check,
    normalized_text,
    title_overlap,
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")


@dataclass(frozen=True)
class Request:
    """
    One request sent to the assistant (the first user message of a conversation).

    need_id names the information need the judgments belong to: V14's need
    for A, B and D; the hidden B need for E (so V14's judgments of that need
    are reused); the request itself for F.
    """

    request_id: str
    set: str
    text: str
    language: str
    need_id: str
    need: str
    category: str = ""  # A: the test-query category; B: the V14 type; F: the kind
    source_query: Optional[str] = None  # E: the original B query (hidden from the assistant)
    works: List[str] = field(default_factory=list)  # F: catalog work ids the request is about
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Request":
        values = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        values["works"] = list(values.get("works") or [])
        values["notes"] = list(values.get("notes") or [])
        return cls(**values)


# =============================================================================
# A, B and D from V14
# =============================================================================


def v14_requests(queries: Sequence[EvalQuery], test_queries: Sequence[Mapping[str, Any]]) -> List[Request]:
    """Requests of sets A, B and D: V14's strata unchanged (A with its category)."""
    categories = {q["query_id"]: q.get("category", "") for q in test_queries}
    out = []
    for q in queries:
        if q.stratum not in (D.SET_A, D.SET_B, D.SET_D):
            continue
        category = categories.get(q.query_id, "") if q.stratum == D.SET_A else q.type
        out.append(Request(request_id=q.query_id, set=q.stratum, text=q.text, language=q.language,
                           need_id=q.need_id, need=q.need, category=category))
    return out


def sample_e_needs(b_queries: Sequence[EvalQuery], size: int = D.E_SIZE, seed: int = D.E_SEED) -> List[EvalQuery]:
    """`size` B queries drawn at random with `seed` from the B queries sorted by id (protocol: semilla 0)."""
    ordered = sorted(b_queries, key=lambda q: q.query_id)
    chosen = random.Random(seed).sample(ordered, size)
    return chosen


def sample_agent_requests(requests: Sequence[Request], size: int = D.AGENT_SAMPLE, seed: int = D.AGENT_SEED) -> List[str]:
    """Ids of the A and B requests the agent is called on directly (seed 0)."""
    ab = sorted(r.request_id for r in requests if r.set in (D.SET_A, D.SET_B))
    return sorted(random.Random(seed).sample(ab, size))


# =============================================================================
# Set E: prompts, parsing and checks
# =============================================================================

E_PROMPT_VERSION = "v1"

E_PROMPT = """You are helping to evaluate the assistant of a multilingual book catalog. For each reader below we know what they really want (their "need") and the precise query they would type if they knew exactly how to say it. We want, instead, the VAGUE first message such a reader might send to a librarian before explaining themselves.

For each reader write one vague request:
- one to three words, in the reader's language (given by its code), as typed into a chat box (Chinese and Japanese: a short phrase of at most 8 characters);
- general enough that a librarian would need to ask a follow-up question: it may name a broad genre, form or field ("novels", "history", "something to read"), but must NOT give away the specific subject, period, place, characters, author or title of the need;
- not a copy or shortening of the precise query.

Readers (JSON): {readers}

Answer with JSON: {{"requests": [{{"id": "<id>", "text": "..."}}]}} with one entry per reader, in the same order."""

E_FIX_PROMPT = """The vague request for reader {id} does not meet the rules: {reason}

Write a new vague request for that reader only, meeting every rule above. Answer with JSON: {{"requests": [{{"id": "{id}", "text": "..."}}]}}"""


def e_readers(needs: Sequence[EvalQuery]) -> List[Dict[str, str]]:
    return [{"id": q.query_id, "language": q.language, "need": q.need, "precise_query": q.text} for q in needs]


def e_prompt(needs: Sequence[EvalQuery]) -> str:
    return E_PROMPT.format(readers=json.dumps(e_readers(needs), ensure_ascii=False))


def e_request_id(b_query_id: str) -> str:
    return f"e-{b_query_id}"


def e_requests(needs: Sequence[EvalQuery], texts: Mapping[str, str]) -> List[Request]:
    """Set E requests (need ids are the B need ids, so the hidden need is the judged one)."""
    out = []
    for q in needs:
        out.append(Request(request_id=e_request_id(q.query_id), set=D.SET_E,
                           text=" ".join(str(texts.get(q.query_id, "")).split()), language=q.language,
                           need_id=q.need_id, need=q.need, category=q.type, source_query=q.text))
    return out


def content_words(text: str) -> List[str]:
    """Folded words of three or more letters that are not function words of any catalog language."""
    return [w for w in _WORD.findall(fold(text)) if len(w) >= 3 and w not in STOPWORDS and not _CJK.search(w)]


def cjk_chars(text: str) -> int:
    return len(_CJK.findall(text))


SHARED_PREFIX = 5  # inflected forms ("мифология" / "мифологию") share their first letters


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    return len(a) >= SHARED_PREFIX and len(b) >= SHARED_PREFIX and a[:SHARED_PREFIX] == b[:SHARED_PREFIX]


def shared_words(vague: str, original: str) -> List[str]:
    """
    Content words (or, for Chinese and Japanese, two-character sequences) the two texts share.

    Two words are the same when equal, or when both have at least
    SHARED_PREFIX letters and begin with the same SHARED_PREFIX letters, so
    inflected forms count as shared.
    """
    originals = content_words(original)
    found = sorted({w for w in content_words(vague) if any(_same_word(w, o) for o in originals)})
    cjk = "".join(_CJK.findall(vague))
    for i in range(len(cjk) - 1):
        if cjk[i:i + 2] in original:
            found.append(cjk[i:i + 2])
    return sorted(set(found))


def check_e(requests: Sequence[Request], needs: Mapping[str, EvalQuery]) -> Dict[str, Any]:
    """
    Mechanical checks of set E.

    Per request: not empty; length (1-3 words; Chinese and Japanese at most
    E_MAX_CJK_CHARS characters); language (V14's rule; "undecided" is listed,
    not a problem); not the original query; it does not give its need away:
    at most E_MAX_SHARED_WORDS content word shared with the precise query,
    and for a known-item need no word of the work's title and not the
    author's surname. Plus: the count and no duplicate texts.
    """
    problems: List[Dict[str, Any]] = []
    verdicts: Dict[str, Any] = {}
    if len(requests) != D.E_SIZE:
        problems.append({"request_id": None, "check": "count", "detail": f"{len(requests)} requests, expected {D.E_SIZE}"})
    seen: Dict[str, str] = {}
    for r in requests:
        need = needs[r.request_id[2:]] if r.request_id.startswith("e-") else None
        text = r.text.strip()
        if not text:
            problems.append({"request_id": r.request_id, "check": "empty", "detail": "empty text"})
            continue
        if r.language in ("zh", "ja") and cjk_chars(text):
            size, limit, unit = cjk_chars(text), D.E_MAX_CJK_CHARS, "characters"
        else:
            size, limit, unit = len(text.split()), D.E_MAX_WORDS, "words"
        if size > limit:
            problems.append({"request_id": r.request_id, "check": "length", "detail": f"{size} {unit}, at most {limit}"})
        verdict, evidence = language_check(text, r.language)
        verdicts[r.request_id] = {"verdict": verdict, "evidence": evidence, "text": text}
        if verdict == "mismatch":
            problems.append({"request_id": r.request_id, "check": "language", "detail": f"not {r.language}: {evidence}"})
        key = normalized_text(text)
        if key in seen:
            problems.append({"request_id": r.request_id, "check": "duplicate", "detail": f"same text as {seen[key]}"})
        seen.setdefault(key, r.request_id)
        if need is None:
            continue
        if normalized_text(need.text) == key:
            problems.append({"request_id": r.request_id, "check": "gives need away", "detail": "same as the precise query"})
        shared = shared_words(text, need.text)
        if len(shared) > D.E_MAX_SHARED_WORDS:
            problems.append({"request_id": r.request_id, "check": "gives need away",
                             "detail": f"shares {shared} with the precise query"})
        if need.type == "known_item":
            words = title_overlap(text, need.target_title or "")
            surname = fold(need.target_author or "").split()[-1:] if need.target_author else []
            if surname and surname[0] in set(_WORD.findall(fold(text))):
                words = words + surname
            if words:
                problems.append({"request_id": r.request_id, "check": "gives need away",
                                 "detail": f"names the work or author: {words}"})
    return {"problems": problems, "language": verdicts,
            "language_summary": _count(v["verdict"] for v in verdicts.values())}


def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


# =============================================================================
# Set F: works, prompts, parsing and checks
# =============================================================================

F_PROMPT_VERSION = "v1"

F_KIND_RULES = {
    "compare": "a request that asks to compare the two works below (e.g. which one suits a given reader better, or how they differ), naming both by their titles",
    "like_but": "a request for books like the work below but different in one stated way (e.g. shorter, more recent, set elsewhere, lighter in tone), naming the work by its title",
    "facts": "a question about the work below itself (e.g. what it is about, who wrote it and when, whether it suits a given reader), naming it by its title",
}

F_PROMPT = """You are helping to evaluate the assistant of a multilingual book catalog. Its users sometimes ask for things that need several steps. Below are {count} tasks; each gives one or two catalog records (title, authors, year, language, subjects, description) and the kind of request to write.

For each task write one request, as a real reader would type it into the assistant's chat box:
- in the language given by the task's "language" code;
- of the kind the task asks for;
- naming each work exactly by the title shown in its record (you may keep the title in its original language);
- one or two sentences, without ids, links or the description's wording copied.

You will not see the catalog itself nor any answer of the assistant.

Tasks (JSON): {tasks}

Answer with JSON: {{"requests": [{{"id": "<task id>", "text": "...", "works_named": ["<title exactly as in the record>", ...]}}]}} with one entry per task, in the same order."""

F_FIX_PROMPT = """The request for task {id} does not meet the rules: {reason}

Write a new request for that task only, meeting every rule above. Answer with JSON: {{"requests": [{{"id": "{id}", "text": "...", "works_named": [...]}}]}}"""


@dataclass(frozen=True)
class Work:
    """A catalog record shown to the generator."""

    work_id: str
    title: str
    authors: List[str]
    language: str
    categories: List[str]
    year: Optional[str]
    description: str

    def shown(self, chars: int = D.F_DESCRIPTION_SHOWN) -> Dict[str, Any]:
        return {"title": self.title, "authors": self.authors, "year": self.year, "language": self.language,
                "subjects": self.categories[:6], "description": self.description[:chars]}


def pick_f_works(works: Sequence[Work], kinds: Mapping[str, int] = D.F_KINDS, seed: int = D.F_SEED) -> List[Dict[str, Any]]:
    """
    The F tasks: works drawn with `seed` from the eligible works (sorted by
    id), one per task; a "compare" task adds a second work that shares an
    author with the first, or else its first subject, or else any work of
    the same language. Returns [{"id", "kind", "language", "works": [Work, ...]}].
    """
    ordered = sorted(works, key=lambda w: w.work_id)
    rng = random.Random(seed)
    total = sum(kinds.values())
    first = rng.sample(ordered, total)
    used = {w.work_id for w in first}
    by_author: Dict[str, List[Work]] = {}
    by_subject: Dict[str, List[Work]] = {}
    by_language: Dict[str, List[Work]] = {}
    for w in ordered:
        for a in w.authors[:1]:
            by_author.setdefault(fold(a), []).append(w)
        for c in w.categories[:1]:
            by_subject.setdefault(c, []).append(w)
        by_language.setdefault(w.language, []).append(w)
    tasks = []
    number = 0
    for kind, count in kinds.items():
        for _ in range(count):
            work = first[number]
            number += 1
            chosen = [work]
            if kind == "compare":
                for pool in ([by_author.get(fold(a), []) for a in work.authors[:1]]
                             + [by_subject.get(c, []) for c in work.categories[:1]]
                             + [by_language.get(work.language, [])]):
                    candidates = [w for w in pool if w.work_id not in used and fold(w.title) != fold(work.title)]
                    if candidates:
                        other = rng.choice(candidates)
                        used.add(other.work_id)
                        chosen.append(other)
                        break
            tasks.append({"id": f"f{number:02d}", "kind": kind, "language": work.language, "works": chosen})
    return tasks


def f_prompt(tasks: Sequence[Mapping[str, Any]]) -> str:
    shown = [{"id": t["id"], "kind": F_KIND_RULES[t["kind"]], "language": t["language"],
              "records": [w.shown() for w in t["works"]]} for t in tasks]
    return F_PROMPT.format(count=len(tasks), tasks=json.dumps(shown, ensure_ascii=False))


def f_requests(tasks: Sequence[Mapping[str, Any]], answers: Mapping[str, Mapping[str, Any]]) -> List[Request]:
    """Set F requests; the need is the request itself (judged against the request)."""
    out = []
    for t in tasks:
        text = " ".join(str((answers.get(t["id"]) or {}).get("text", "")).split())
        out.append(Request(request_id=t["id"], set=D.SET_F, text=text, language=t["language"], need_id=t["id"],
                           need=text, category=t["kind"], works=[w.work_id for w in t["works"]]))
    return out


def check_f(requests: Sequence[Request], tasks: Mapping[str, Mapping[str, Any]],
            answers: Mapping[str, Mapping[str, Any]], find_target) -> Dict[str, Any]:
    """
    Mechanical checks of set F.

    Per request: not empty, at most 400 characters; language (V14's rule);
    every work of its task named (its folded title, or the part before a
    colon, appears in the request); every title the generator says it named
    found in the catalog by V14's known-item match (find_target), and among
    the task's works. Plus: the count and no duplicates.
    """
    problems: List[Dict[str, Any]] = []
    verdicts: Dict[str, Any] = {}
    found_works: Dict[str, Any] = {}
    if len(requests) != D.F_SIZE:
        problems.append({"request_id": None, "check": "count", "detail": f"{len(requests)} requests, expected {D.F_SIZE}"})
    seen: Dict[str, str] = {}
    for r in requests:
        text = r.text.strip()
        if not text:
            problems.append({"request_id": r.request_id, "check": "empty", "detail": "empty text"})
            continue
        if len(text) > 400:
            problems.append({"request_id": r.request_id, "check": "length", "detail": f"{len(text)} characters"})
        verdict, evidence = language_check(text, r.language)
        verdicts[r.request_id] = {"verdict": verdict, "evidence": evidence, "text": text}
        if verdict == "mismatch":
            problems.append({"request_id": r.request_id, "check": "language", "detail": f"not {r.language}: {evidence}"})
        key = normalized_text(text)
        if key in seen:
            problems.append({"request_id": r.request_id, "check": "duplicate", "detail": f"same text as {seen[key]}"})
        seen.setdefault(key, r.request_id)
        task = tasks[r.request_id]
        for work in task["works"]:
            if not title_named(text, work.title):
                problems.append({"request_id": r.request_id, "check": "work not named",
                                 "detail": f"title {work.title!r} not in the request"})
        named = list((answers.get(r.request_id) or {}).get("works_named") or [])
        task_ids = {w.work_id for w in task["works"]}
        found_works[r.request_id] = {}
        for title in named:
            ids = find_target(str(title), None)
            found_works[r.request_id][str(title)] = ids
            if not ids:
                problems.append({"request_id": r.request_id, "check": "work not in catalog", "detail": str(title)})
            elif not set(ids) & task_ids:
                problems.append({"request_id": r.request_id, "check": "work not in catalog",
                                 "detail": f"{title!r} matches other works than the task's"})
    return {"problems": problems, "language": verdicts, "works_found": found_works,
            "language_summary": _count(v["verdict"] for v in verdicts.values())}


def title_named(text: str, title: str) -> bool:
    """True if the folded title (or its part before a colon or a period) appears in the folded text."""
    folded = " ".join(_WORD.findall(fold(text)))
    candidates = [title, re.split(r"[:.;(]", title)[0]]
    for candidate in candidates:
        key = " ".join(_WORD.findall(fold(candidate)))
        if key and key in folded:
            return True
        cjk = "".join(_CJK.findall(candidate))
        if cjk and cjk in text:
            return True
    return False


# =============================================================================
# Files
# =============================================================================


def save_requests(path, requests: Sequence[Request], meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "requests": [r.to_dict() for r in requests]}, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_requests(path) -> List[Request]:
    with open(path, encoding="utf-8") as handle:
        return [Request.from_dict(item) for item in json.load(handle)["requests"]]


def by_set(requests: Sequence[Request], *names: str) -> List[Request]:
    return [r for r in requests if r.set in names]


def parse_requests_answer(answer: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{id: item} from a generator answer {"requests": [{"id", ...}]}."""
    return {str(item.get("id")): dict(item) for item in answer.get("requests") or [] if item.get("id")}


def is_known_item(need: EvalQuery) -> bool:
    return need.type == "known_item"


def describe_sets(requests: Sequence[Request]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    langs: Dict[str, Dict[str, int]] = {}
    cats: Dict[str, Dict[str, int]] = {}
    for r in requests:
        counts[r.set] = counts.get(r.set, 0) + 1
        langs.setdefault(r.set, {})[r.language] = langs.get(r.set, {}).get(r.language, 0) + 1
        cats.setdefault(r.set, {})[r.category] = cats.get(r.set, {}).get(r.category, 0) + 1
    return {"counts": counts, "languages": langs, "categories": cats}


def need_texts(requests: Sequence[Request]) -> Dict[str, str]:
    """{need_id: the text the judges grade against} (one per need)."""
    out: Dict[str, str] = {}
    for r in requests:
        out.setdefault(r.need_id, r.need)
    return out


def split_tasks(tasks: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Mapping[str, Any]], List[str]]:
    return {t["id"]: t for t in tasks}, [t["id"] for t in tasks]
