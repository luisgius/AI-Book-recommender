"""
Query set of the evaluation (protocol section 3): four strata.

- A: the 44 queries of app/evaluation/test_queries.json, unchanged; their
  information need is their own text.
- B: 72 new queries, six in each of twelve catalog languages (three
  thematic, two descriptive, one known-item described without words of its
  title).
- C: ten needs, each written in Spanish, English, French and German (40
  queries sharing the judgments of their need).
- D: twelve requests a book catalog cannot answer.

B, C and D are written by an LLM of another vendor than the judges, the
system LLM and the retrieval models, from the catalog's languages and
subject list only (it never sees a system output). This module holds the
pure parts: the query record, the generation prompts, the parsing and the
mechanical checks (counts, duplicates, language, title words in known-item
queries). The network call is in generate_queries.
"""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.evaluation.catalog_evaluation.design import (
    A_TYPE,
    B_LANGUAGES,
    B_TYPES,
    C_LANGUAGES,
    C_NEEDS,
    D_QUERIES,
    EXPECTED_COUNTS,
    LANGUAGE_NAMES,
    STRATUM_A,
    STRATUM_B,
    STRATUM_C,
    STRATUM_D,
    UNANSWERABLE_TYPE,
)
from app.ingestion.language_detection import _FUNCTION_WORDS, _WORD_WEIGHT


@dataclass(frozen=True)
class EvalQuery:
    """
    One evaluation query.

    need_id identifies the information need the judgments belong to: the
    query's own id, except in stratum C, where the four language versions of
    a need share one need_id (and its judgments).
    """

    query_id: str
    need_id: str
    text: str
    language: str
    type: str
    stratum: str
    need: str
    target_title: Optional[str] = None  # known-item queries: the work meant (never shown to a system)
    target_author: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalQuery":
        values = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        values["notes"] = list(values.get("notes") or [])
        return cls(**values)


def stratum_a(test_queries: Sequence[Mapping[str, Any]]) -> List[EvalQuery]:
    """Stratum A: the original queries unchanged; the need is the query text."""
    return [
        EvalQuery(query_id=q["query_id"], need_id=q["query_id"], text=q["text"], language=q["language"],
                  type=A_TYPE, stratum=STRATUM_A, need=q["text"])
        for q in test_queries
    ]


def load_queries(path) -> List[EvalQuery]:
    with open(path, encoding="utf-8") as handle:
        return [EvalQuery.from_dict(item) for item in json.load(handle)["queries"]]


def by_stratum(queries: Iterable[EvalQuery], *strata: str) -> List[EvalQuery]:
    return [q for q in queries if q.stratum in strata]


# =============================================================================
# Generation prompts (the generator sees languages and subjects only)
# =============================================================================

GENERATION_PROMPT_VERSION = "v1"

_CATALOG_BLOCK = """You are helping to build a test collection for a multilingual book search engine.

The catalog has {works:,} works (novels, essays, poetry, plays, non-fiction, scholarly open-access books).
Share of works by language of the work: {languages}.
Most frequent subjects in the catalog: {subjects}.

You will not see the catalog itself nor any search result."""

_B_PROMPT = _CATALOG_BLOCK + """

Write {total} search queries in {language_name} (language code "{language}"), as real readers would type them into the search box of this catalog:
- {thematic} "thematic" queries: a subject, genre or theme, a few words (e.g. a topic and a period).
- {descriptive} "descriptive" queries: a sentence describing the kind of book wanted (in English it would start like "a novel about ...", "a book that explains ...").
- {known_item} "known_item" query: a reader looking for one specific, widely known work written in {language_name} or very famous among {language_name} readers, described by its plot, characters or author WITHOUT using any word of its title (not even translated) and without the title's characters for Chinese or Japanese.

Rules:
- Every query must be written entirely in {language_name}.
- Vary subjects and periods; do not repeat a query.
- For each query give "need": one or two sentences in English describing what the reader wants (what a relevant book must be about). A relevance assessor will grade books against this description.
- For the known_item query also give "target_title" (the work's title in its original language) and "target_author".

Answer with JSON: {{"queries": [{{"type": "thematic" | "descriptive" | "known_item", "text": "...", "need": "...", "target_title": "... or null", "target_author": "... or null"}}]}}"""

_C_PROMPT = _CATALOG_BLOCK + """

Write {needs} different information needs that this catalog could plausibly answer (a mix of thematic and descriptive requests on varied subjects), and write each one as a search query in Spanish, English, French and German. The four versions of a need must ask for exactly the same thing, as a native speaker of each language would type it (not a word-for-word translation).

For each need give "need": one or two sentences in English describing what a relevant book must be about, "type": "thematic" or "descriptive", and the four queries.

Answer with JSON: {{"needs": [{{"need": "...", "type": "thematic" | "descriptive", "es": "...", "en": "...", "fr": "...", "de": "..."}}]}}"""

_D_PROMPT = _CATALOG_BLOCK + """

Write {count} queries that a user might type into this search box but that a book catalog cannot answer at all, because no book can satisfy them: practical, real-time or personal requests (for example the timetable of a train, today's weather, the price of a flight, a phone number, the opening hours of a shop). One query in each of these languages: {query_languages}.

Rules: each query entirely in its language; varied requests; do not mention books, reading or literature.
For each query give "need" in English (what the user wants).

Answer with JSON: {{"queries": [{{"language": "<code>", "text": "...", "need": "..."}}]}}"""


def _languages_line(language_shares: Mapping[str, float]) -> str:
    return ", ".join(f"{LANGUAGE_NAMES.get(code, code)} ({code}) {share:.1%}" for code, share in language_shares.items())


def b_prompt(language: str, language_shares: Mapping[str, float], subjects: Sequence[str], works: int) -> str:
    return _B_PROMPT.format(
        works=works, languages=_languages_line(language_shares), subjects=", ".join(subjects),
        total=sum(B_TYPES.values()), language=language, language_name=LANGUAGE_NAMES[language],
        thematic=B_TYPES["thematic"], descriptive=B_TYPES["descriptive"], known_item=B_TYPES["known_item"],
    )


def c_prompt(language_shares: Mapping[str, float], subjects: Sequence[str], works: int) -> str:
    return _C_PROMPT.format(works=works, languages=_languages_line(language_shares),
                            subjects=", ".join(subjects), needs=C_NEEDS)


def d_prompt(language_shares: Mapping[str, float], subjects: Sequence[str], works: int) -> str:
    query_languages = ", ".join(f"{LANGUAGE_NAMES[c]} ({c})" for c in B_LANGUAGES)
    return _D_PROMPT.format(works=works, languages=_languages_line(language_shares),
                            subjects=", ".join(subjects), count=D_QUERIES, query_languages=query_languages)


FIX_PROMPT = """The following query does not meet the rules: {reason}

Original item: {item}

Write one replacement item of the same kind, in the same language ({language_name}), meeting every rule above. Answer with JSON in the same format as before, with a single item in the list."""


# =============================================================================
# Parsing generator output
# =============================================================================


def parse_json_answer(content: str) -> Dict[str, Any]:
    """The JSON object of a model answer (tolerates a fenced code block around it)."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[start:end + 1])


def b_queries(language: str, answer: Mapping[str, Any]) -> List[EvalQuery]:
    """Stratum B queries of one language from the generator's answer, ids b-<lang>-<n>."""
    out = []
    for number, item in enumerate(answer.get("queries", []), start=1):
        qtype = str(item.get("type", "")).strip()
        query_id = f"b-{language}-{number}"
        out.append(EvalQuery(
            query_id=query_id, need_id=query_id, text=" ".join(str(item.get("text", "")).split()),
            language=language, type=qtype, stratum=STRATUM_B, need=" ".join(str(item.get("need", "")).split()),
            target_title=(item.get("target_title") or None) if qtype == "known_item" else None,
            target_author=(item.get("target_author") or None) if qtype == "known_item" else None,
        ))
    return out


def c_queries(answer: Mapping[str, Any]) -> List[EvalQuery]:
    """Stratum C queries: need c<nn>, query ids c<nn>-<lang>."""
    out = []
    for number, item in enumerate(answer.get("needs", []), start=1):
        need_id = f"c{number:02d}"
        for language in C_LANGUAGES:
            out.append(EvalQuery(
                query_id=f"{need_id}-{language}", need_id=need_id, text=" ".join(str(item.get(language, "")).split()),
                language=language, type=str(item.get("type", "")).strip(), stratum=STRATUM_C,
                need=" ".join(str(item.get("need", "")).split()),
            ))
    return out


def d_queries(answer: Mapping[str, Any]) -> List[EvalQuery]:
    """Stratum D queries, ids d-<lang>."""
    out = []
    for item in answer.get("queries", []):
        language = str(item.get("language", "")).strip().lower()
        query_id = f"d-{language}"
        out.append(EvalQuery(query_id=query_id, need_id=query_id, text=" ".join(str(item.get("text", "")).split()),
                             language=language, type=UNANSWERABLE_TYPE, stratum=STRATUM_D,
                             need=" ".join(str(item.get("need", "")).split())))
    return out


# =============================================================================
# Mechanical checks
# =============================================================================

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NON_LATIN = {"ru": "cyrillic", "zh": "han", "ja": "japanese", "ar": "arabic"}


def fold(text: str) -> str:
    """Lowercase without diacritics (NFKD, combining marks dropped)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def script_counts(text: str) -> Counter:
    """Letters per writing system (latin, cyrillic, arabic, han, kana, other)."""
    counts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            counts["kana"] += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts["han"] += 1
        elif 0x0400 <= code <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            counts["arabic"] += 1
        elif code < 0x0250 or 0x1E00 <= code <= 0x1EFF:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def language_check(text: str, language: str) -> Tuple[str, str]:
    """
    Is the query written in `language`? Returns (verdict, evidence), verdict
    "ok", "mismatch" or "undecided".

    Rule. Non-Latin languages (ru, zh, ja, ar) by writing system: most
    letters must be in the language's script (Japanese: Han or kana with at
    least one kana; a text with kana is never Chinese). Latin languages: most
    letters Latin, then the function words of the catalog's language
    detector (app/ingestion/language_detection.py), each word weighted by
    1/number of languages that share it: "mismatch" if another language
    scores higher than the stated one, "ok" if the stated one has the top
    score, "undecided" when the query has no function word at all (short
    keyword queries). Undecided queries are listed for a manual check.
    """
    counts = script_counts(text)
    letters = sum(counts.values())
    if letters == 0:
        return "mismatch", "no letters"
    if language in NON_LATIN:
        if language == "ja":
            cjk = counts["han"] + counts["kana"]
            if cjk <= letters / 2:
                return "mismatch", f"{cjk}/{letters} Han or kana letters"
            if counts["kana"] == 0:
                return "undecided", "Han characters only (Chinese or Japanese)"
            return "ok", f"{counts['kana']} kana, {counts['han']} Han"
        script = NON_LATIN[language]
        if counts[script] <= letters / 2:
            return "mismatch", f"{counts[script]}/{letters} {script} letters"
        if language == "zh" and counts["kana"]:
            return "mismatch", "kana present (Japanese)"
        return "ok", f"{counts[script]}/{letters} {script} letters"
    if counts["latin"] <= letters / 2:
        return "mismatch", f"{counts['latin']}/{letters} Latin letters"
    tokens = [t.lower().split("'")[-1].split("’")[-1] for t in _WORD.findall(text)]
    scores: Counter = Counter()
    for token in tokens:
        for lang, weight in _WORD_WEIGHT.get(token, {}).items():
            scores[lang] += weight
    if not scores:
        return "undecided", "no function word"
    best = max(scores.values())
    own = scores.get(language, 0.0)
    evidence = ", ".join(f"{lang} {value:.2f}" for lang, value in scores.most_common(3))
    if language not in _FUNCTION_WORDS:
        return "undecided", f"no word list for {language}"
    if own + 1e-9 < best:
        return "mismatch", evidence
    return "ok", evidence


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
STOPWORDS = frozenset(word for words in _FUNCTION_WORDS.values() for word in words)


def title_overlap(query: str, title: str) -> List[str]:
    """
    Words of the title that appear in the query (known-item rule).

    Latin, Cyrillic and Arabic text: folded words of three or more letters
    that are not function words of any catalog language. Chinese and
    Japanese: any two consecutive Han or kana characters of the title.
    """
    found: List[str] = []
    query_folded = fold(query)
    title_folded = fold(title)
    query_words = set(_WORD.findall(query_folded))
    for word in _WORD.findall(title_folded):
        if _CJK.search(word):
            continue
        if len(word) >= 3 and word not in STOPWORDS and word in query_words:
            found.append(word)
    cjk_title = "".join(_CJK.findall(title))
    for i in range(len(cjk_title) - 1):
        bigram = cjk_title[i:i + 2]
        if bigram in query:
            found.append(bigram)
    return sorted(set(found))


def normalized_text(text: str) -> str:
    return " ".join(_WORD.findall(fold(text)))


def check_queries(queries: Sequence[EvalQuery]) -> Dict[str, Any]:
    """
    Mechanical checks of the whole query set.

    Returns counts per stratum, language and type, duplicates (same folded
    text), the language verdict of every query, the known-item queries that
    contain title words, and the list of problems (an empty list means the
    set passes; "undecided" language verdicts are not problems, they are
    listed for a manual check).
    """
    problems: List[Dict[str, Any]] = []
    counts = Counter(q.stratum for q in queries)
    for stratum, expected in EXPECTED_COUNTS.items():
        if counts.get(stratum, 0) != expected:
            problems.append({"query_id": None, "check": "count", "detail": f"stratum {stratum}: {counts.get(stratum, 0)} queries, expected {expected}"})
    by_language_type = Counter((q.stratum, q.language, q.type) for q in queries)
    for language in B_LANGUAGES:
        for qtype, expected in B_TYPES.items():
            got = by_language_type.get((STRATUM_B, language, qtype), 0)
            if got != expected:
                problems.append({"query_id": None, "check": "count", "detail": f"B {language} {qtype}: {got}, expected {expected}"})
    c_needs = Counter(q.need_id for q in queries if q.stratum == STRATUM_C)
    for need_id, number in c_needs.items():
        if number != len(C_LANGUAGES):
            problems.append({"query_id": need_id, "check": "count", "detail": f"need {need_id} has {number} versions"})
    d_languages = Counter(q.language for q in queries if q.stratum == STRATUM_D)
    for language in B_LANGUAGES:
        if d_languages.get(language, 0) != 1:
            problems.append({"query_id": None, "check": "count", "detail": f"D {language}: {d_languages.get(language, 0)} queries, expected 1"})

    seen: Dict[str, str] = {}
    duplicates = []
    ids = Counter(q.query_id for q in queries)
    for query_id, number in ids.items():
        if number > 1:
            problems.append({"query_id": query_id, "check": "duplicate id", "detail": f"{number} times"})
    for q in queries:
        key = normalized_text(q.text)
        if not key:
            problems.append({"query_id": q.query_id, "check": "empty", "detail": "empty text"})
            continue
        if not q.need.strip():
            problems.append({"query_id": q.query_id, "check": "empty", "detail": "empty need"})
        if key in seen:
            duplicates.append([seen[key], q.query_id])
            problems.append({"query_id": q.query_id, "check": "duplicate", "detail": f"same text as {seen[key]}"})
        else:
            seen[key] = q.query_id

    language_verdicts = {}
    for q in queries:
        if q.stratum == STRATUM_A:
            continue  # unchanged by design
        verdict, evidence = language_check(q.text, q.language)
        language_verdicts[q.query_id] = {"verdict": verdict, "evidence": evidence, "language": q.language, "text": q.text}
        if verdict == "mismatch":
            problems.append({"query_id": q.query_id, "check": "language", "detail": f"not {q.language}: {evidence}"})

    title_words = {}
    for q in queries:
        if q.type != "known_item":
            continue
        if not q.target_title:
            problems.append({"query_id": q.query_id, "check": "known item", "detail": "no target title"})
            continue
        overlap = title_overlap(q.text, q.target_title)
        title_words[q.query_id] = overlap
        if overlap:
            problems.append({"query_id": q.query_id, "check": "title words", "detail": f"contains {overlap}"})

    return {
        "counts_by_stratum": dict(sorted(counts.items())),
        "counts_by_stratum_language_type": [
            {"stratum": s, "language": lang, "type": t, "queries": n}
            for (s, lang, t), n in sorted(by_language_type.items())
        ],
        "duplicates": duplicates,
        "language": language_verdicts,
        "language_summary": dict(Counter(v["verdict"] for v in language_verdicts.values())),
        "title_words": title_words,
        "problems": problems,
    }
