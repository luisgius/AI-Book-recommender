"""
Set G of the re-measurement (V25 addendum, section 3): a held-out set written
after the revised routing instruction was committed, by V14's generator,
from a list of topics written here, without the generator seeing the routing
instruction or any output of the system.

- 60 valid requests on non-literary topics (ten fields, six topics each), in
  six languages (es, en, fr, de, it, pt), half as a few keywords typed into
  the search box and half as a sentence asking for something to read on the
  topic. Expected route: anything but "out_of_scope".
- 20 out-of-scope requests of the four kinds the addendum names (a fact about
  the present moment, a calculation, writing a program, conversation with no
  request), five of each. Expected route: "out_of_scope".

Languages and forms are assigned to the topics in code (seed 0) so that
every language gets ten valid requests, five of each form. The generator
returns each request with a faithful English translation, which the
mechanical checks read: the valid ones must name their topic (a content word
shared with the topic, inflected forms included) and the sentences must ask
for something to read; the out-of-scope ones must not mention books or
reading and must look like their kind. Language (V14's rule), form, count and
duplicates (within G and against every request of A, B, D, E and F) are
checked on the request itself. A request that fails is sent back to the
generator with the reason, as for E and F (at most three rounds).

This module holds the pure parts and the generation call; the files are
<data>/sets/g_topics.json (the list as given to the generator),
<data>/sets/generation/G*.json (prompt, model, provider, date, raw output)
and <data>/sets/g_checks.json.
"""

import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.assistant_evaluation import sets as S
from app.evaluation.catalog_evaluation.generation import GENERATOR_REASONING, GENERATOR_TEMPERATURE, Generator
from app.evaluation.catalog_evaluation.ledger import SpendLedger
from app.evaluation.catalog_evaluation.queries import language_check, normalized_text, parse_json_answer

G_PROMPT_VERSION = "v1"
G_SEED = 0
G_LANGUAGES = ("es", "en", "fr", "de", "it", "pt")
FORM_KEYWORDS, FORM_SENTENCE = "keywords", "sentence"
KEYWORDS_MAX_WORDS = 4
SENTENCE_MIN_WORDS = 5
VALID, OUT_OF_SCOPE = "valid", "out_of_scope"

# Ten non-literary fields, six topics each (the addendum names cooking,
# computing, health, travel, economics, science, trades and sport "and
# others"; the last two fields are gardening and home and family). Written
# for this set; none is taken from the requests of A, B, D, E or F.
G_TOPICS: Dict[str, List[str]] = {
    "cooking": ["sourdough bread baking", "Thai curry cooking", "vegetarian meal prep", "cake decorating",
                "fermenting vegetables such as kimchi and sauerkraut", "barbecue and smoking meat"],
    "computing": ["relational databases and SQL", "cybersecurity for small businesses",
                  "building websites with JavaScript", "Linux system administration",
                  "algorithms and data structures", "electronics projects with a Raspberry Pi"],
    "health": ["living with type 2 diabetes", "sleep problems and insomnia", "exercises for back pain",
               "nutrition during pregnancy", "coping with anxiety", "first aid"],
    "travel": ["backpacking in Southeast Asia", "walking the Camino de Santiago", "a road trip around Iceland",
               "travelling in Japan on a budget", "hiking in the Alps", "visiting Morocco"],
    "economics": ["personal finance and saving money", "investing in the stock market for beginners",
                  "inflation and central banks", "starting a small business", "behavioural economics",
                  "the history of money"],
    "science": ["black holes", "the science of climate change", "human genetics and DNA",
                "quantum physics for beginners", "coral reef marine biology", "volcanoes and earthquakes"],
    "trades": ["woodworking and carpentry", "home plumbing repairs", "basic household electrical wiring",
               "welding", "car mechanics", "sewing and tailoring"],
    "sport": ["marathon training", "chess openings", "football tactics", "yoga for beginners",
              "rock climbing", "swimming technique"],
    "gardening": ["growing a vegetable garden", "bonsai", "composting", "caring for houseplants", "permaculture",
                  "pruning fruit trees"],
    "home and family": ["training a puppy", "keeping backyard chickens", "toddler sleep", "decluttering the home",
                        "planning a wedding", "understanding cat behaviour"],
}

# The four kinds of out-of-scope request of the addendum, five each: the gist
# the generator turns into a request.
G_OUT_OF_SCOPE: Dict[str, List[str]] = {
    "present_fact": ["the current price of bitcoin", "the live score of tonight's tennis final",
                     "whether the nearest post office is open right now", "the traffic on the ring road at this moment",
                     "today's lottery results"],
    "calculation": ["what 15 percent of 240 is", "converting 72 degrees Fahrenheit to Celsius",
                    "solving the equation 3x + 7 = 22", "the monthly payment of a 200000 mortgage at 3 percent over 25 years",
                    "how many days there are between 3 March and 19 August"],
    "program": ["writing a Python function that removes duplicates from a list",
                "writing an SQL query that counts orders per customer", "writing a bash script that renames photos by date",
                "writing a regular expression that validates an email address",
                "writing an HTML page with a contact form"],
    "conversation": ["greeting the assistant and asking how it is", "asking the assistant to tell a joke",
                     "thanking the assistant and saying goodbye", "asking the assistant its favourite colour",
                     "saying they are bored and want to chat"],
}

# English words that mark a request for something to read (in the translation).
READING_WORDS = frozenset(
    "book books read reading reads novel novels guide guides guidebook manual manuals handbook handbooks "
    "textbook textbooks literature ebook library libraries author authors title titles bibliography".split()
)
_WORD = re.compile(r"[a-z]+")
_KIND_SIGNS = {
    "present_fact": re.compile(r"\b(now|today|tonight|current|currently|right now|at the moment|this moment|live|latest)\b"),
    "calculation": re.compile(r"\d"),
    "program": re.compile(r"\b(write|code|script|function|program|query|regex|regular expression|html)\b"),
}


def g_tasks(seed: int = G_SEED) -> List[Dict[str, Any]]:
    """
    The 80 generation tasks with their id, kind, language and form.

    Valid: the 60 topics in a seeded order; task i gets language
    G_LANGUAGES[i % 6] and form keywords when (i // 6) is even, so every
    language has five of each form. Out of scope: the 20 gists in a seeded
    order, language G_LANGUAGES[i % 6].
    """
    rng = random.Random(seed)
    topics = [(field, topic) for field, items in G_TOPICS.items() for topic in items]
    rng.shuffle(topics)
    tasks = []
    for i, (field, topic) in enumerate(topics):
        tasks.append({"id": f"g{i + 1:02d}", "group": VALID, "field": field, "topic": topic,
                      "language": G_LANGUAGES[i % len(G_LANGUAGES)],
                      "form": FORM_KEYWORDS if (i // len(G_LANGUAGES)) % 2 == 0 else FORM_SENTENCE})
    gists = [(kind, gist) for kind, items in G_OUT_OF_SCOPE.items() for gist in items]
    rng.shuffle(gists)
    for i, (kind, gist) in enumerate(gists):
        tasks.append({"id": f"g-oos{i + 1:02d}", "group": OUT_OF_SCOPE, "field": kind, "topic": gist,
                      "language": G_LANGUAGES[i % len(G_LANGUAGES)], "form": "any"})
    return tasks


G_PROMPT = """You are helping to test the search box of a multilingual online library catalog: people type into it what they want, in their own language. We need realistic messages of two kinds.

1. Valid messages (group "valid"): a reader wants something to read on the given topic.
   - form "keywords": one to four words, as typed into a search box: just the topic, no verb, no question mark, no full stop.
   - form "sentence": one sentence of six to twenty-five words asking for something to read on the topic (a book, a guide, a manual, an introduction...), written as a person would type it.
2. Other messages (group "out_of_scope"): a person types into the same box something that is not a request for anything to read; the task's "topic" says what they want. Write it as they would type it. It must not mention books, reading, libraries or authors.

Rules for every message: in the language given by the task's code; natural and varied (do not start every message the same way); no ids or quotation marks around the whole message. For each one also give a faithful English translation.

Tasks (JSON): {tasks}

Answer with JSON: {{"requests": [{{"id": "<task id>", "text": "...", "english": "..."}}]}} with one entry per task, in the same order."""

G_FIX_PROMPT = """The message for task {id} does not meet the rules: {reason}

Write a new message for that task only, meeting every rule above. Answer with JSON: {{"requests": [{{"id": "{id}", "text": "...", "english": "..."}}]}}"""


def g_prompt(tasks: Sequence[Mapping[str, Any]]) -> str:
    shown = [{k: t[k] for k in ("id", "group", "topic", "language", "form")} for t in tasks]
    return G_PROMPT.format(tasks=json.dumps(shown, ensure_ascii=False))


def english_words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def check_g(items: Mapping[str, Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]],
            other_texts: Sequence[str]) -> Dict[str, Any]:
    """
    Mechanical checks of set G (see the module docstring).

    items: {task id: {"text", "english"}}; other_texts: the texts of A, B, D,
    E and F, which no G request may repeat.
    """
    problems: List[Dict[str, Any]] = []
    verdicts: Dict[str, Any] = {}
    for group, expected in ((VALID, 60), (OUT_OF_SCOPE, 20)):
        n = sum(1 for t in tasks if t["group"] == group and str(items.get(t["id"], {}).get("text", "")).strip())
        if n != expected:
            problems.append({"request_id": None, "check": "count", "detail": f"{n} {group} requests, expected {expected}"})
    others = {normalized_text(t) for t in other_texts}
    seen: Dict[str, str] = {}
    for task in tasks:
        tid = task["id"]
        item = items.get(tid) or {}
        text = " ".join(str(item.get("text", "")).split())
        english = str(item.get("english", ""))
        if not text:
            problems.append({"request_id": tid, "check": "empty", "detail": "no text"})
            continue
        verdict, evidence = language_check(text, task["language"])
        verdicts[tid] = {"verdict": verdict, "evidence": evidence, "text": text}
        if verdict == "mismatch":
            problems.append({"request_id": tid, "check": "language", "detail": f"not {task['language']}: {evidence}"})
        key = normalized_text(text)
        if key in seen:
            problems.append({"request_id": tid, "check": "duplicate", "detail": f"same text as {seen[key]}"})
        seen.setdefault(key, tid)
        if key in others:
            problems.append({"request_id": tid, "check": "duplicate", "detail": "same text as a request of A-F"})
        words = english_words(english)
        if not words:
            problems.append({"request_id": tid, "check": "translation", "detail": "no English translation"})
            continue
        size = len(text.split())
        if task["group"] == VALID:
            if task["form"] == FORM_KEYWORDS and (size > KEYWORDS_MAX_WORDS or text.rstrip()[-1:] in "?.!"):
                problems.append({"request_id": tid, "check": "form",
                                 "detail": f"keywords: {size} words (at most {KEYWORDS_MAX_WORDS}), no final punctuation"})
            if task["form"] == FORM_SENTENCE and size < SENTENCE_MIN_WORDS:
                problems.append({"request_id": tid, "check": "form",
                                 "detail": f"sentence: {size} words (at least {SENTENCE_MIN_WORDS})"})
            if not S.shared_words(english, task["topic"]):
                problems.append({"request_id": tid, "check": "topic",
                                 "detail": f"the translation '{english}' does not name the topic '{task['topic']}'"})
            if task["form"] == FORM_SENTENCE and not READING_WORDS.intersection(words):
                problems.append({"request_id": tid, "check": "asks for reading",
                                 "detail": f"the translation '{english}' does not ask for something to read"})
        else:
            reading = sorted(READING_WORDS.intersection(words))
            if reading:
                problems.append({"request_id": tid, "check": "asks for books",
                                 "detail": f"the translation mentions {reading}"})
            sign = _KIND_SIGNS.get(task["field"])
            if sign is not None and not sign.search(english.lower()):
                problems.append({"request_id": tid, "check": "kind",
                                 "detail": f"the translation '{english}' does not read as a {task['field']} request"})
    return {"problems": problems, "language": verdicts,
            "language_summary": dict(sorted(_count(v["verdict"] for v in verdicts.values()).items()))}


def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def g_requests(tasks: Sequence[Mapping[str, Any]], items: Mapping[str, Mapping[str, Any]]) -> List[S.Request]:
    """Set G as requests: category "valid" or "out_of_scope"; notes: field, form or kind, English translation."""
    out = []
    for t in tasks:
        item = items.get(t["id"]) or {}
        out.append(S.Request(request_id=t["id"], set=D.SET_G, text=" ".join(str(item.get("text", "")).split()),
                             language=t["language"], need_id=t["id"], need=t["topic"], category=t["group"],
                             notes=[t["field"], t["form"], str(item.get("english", ""))]))
    return out


def build_set_g(sets_dir: Path, ledger: SpendLedger, *, generator: Optional[Generator] = None,
                log: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    Generate and check G, append it to sets_dir/requests.json and write g_topics.json and g_checks.json.

    requests.json must already hold A, B, D, E and F (copied from the first
    measurement); G replaces any G requests it held. Returns g_checks.json.
    """
    from app.evaluation.assistant_evaluation.generation import _fix_loop

    tasks = g_tasks()
    (sets_dir / "g_topics.json").write_text(json.dumps(
        {"prompt_version": G_PROMPT_VERSION, "seed": G_SEED, "languages": G_LANGUAGES, "topics": G_TOPICS,
         "out_of_scope": G_OUT_OF_SCOPE, "tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = json.loads((sets_dir / "requests.json").read_text(encoding="utf-8"))
    base = [S.Request.from_dict(r) for r in meta["requests"] if r["set"] != D.SET_G]
    other_texts = [r.text for r in base]
    generator = generator or Generator(sets_dir / "generation", ledger, model=D.GENERATOR_MODEL)
    conversation = [{"role": "user", "content": g_prompt(tasks)}]
    record = generator.ask("G", conversation)
    conversation = conversation + [{"role": "assistant", "content": record["raw_output"]}]
    items = S.parse_requests_answer(parse_json_answer(record["raw_output"]))
    result = _fix_loop("G", generator, conversation, items, lambda x: check_g(x, tasks, other_texts),
                       G_FIX_PROMPT, log)
    requests = base + g_requests(tasks, items)
    meta_out = {k: v for k, v in meta.items() if k != "requests"}
    meta_out["g"] = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                     "generator_model": generator.model, "prompt_version": G_PROMPT_VERSION}
    S.save_requests(sets_dir / "requests.json", requests, meta_out)
    document = {"generator": {"model": generator.model, "temperature": GENERATOR_TEMPERATURE,
                              "reasoning": GENERATOR_REASONING, "prompt_version": G_PROMPT_VERSION,
                              "date": record.get("date"), "provider": record.get("provider")},
                "tasks": tasks, "G": result,
                "counts": _count(f"{t['group']}/{t['language']}/{t['form']}" for t in tasks)}
    (sets_dir / "g_checks.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, default=str),
                                            encoding="utf-8")
    return document
