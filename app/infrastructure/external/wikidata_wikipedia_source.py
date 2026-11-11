"""
Wikidata + Wikipedia bulk source (implements the BulkBooksSource port).

What it reads
-------------
Works of a fixed set of "book-like" Wikidata classes (literary work, written
work, book, novel series, ...) that have a Wikipedia article in at least one
of the configured languages. For each work:

- Wikidata (CC0): labels, authors (P50, else creator P170, else
  "Unknown"), genres (P136), publication year (P577 with at least year
  precision), language of the work (P407, mapped to ISO 639-1 through the
  language item's P218; else the language tag of the title P1476), Open
  Library ids (P648),
  Project Gutenberg ids (P2034), ISBN-13 (P212), DOI (P356) and the number
  of Wikipedia language editions with an article (popularity signal).
- Wikipedia (CC BY-SA): the plain-text intro of the article in each
  configured language, used as that language's description. Each description
  keeps the article URL as its attribution.

How it downloads (separate step, cached on disk, resumable)
-----------------------------------------------------------
1. SPARQL: one query per piece of data (articles, labels, authors, ...),
   each saved as a CSV file in ``<download_dir>/sparql/``. A file that exists
   is not fetched again. The public Wikidata Query Service (WDQS) cannot
   enumerate this set inside its 60 s limit (a bare count of literary works
   with an English article took 59.7 s on 2026-11-11 and the paged variant
   timed out), so the default endpoint is QLever (University of Freiburg),
   which serves the same Wikidata RDF and answers these queries in seconds.
   The queries are plain SPARQL 1.1 and also run on WDQS for smaller sets.
2. DBpedia abstracts (the Wikipedia intros extracted by DBpedia from late
   2022 revisions), fetched in bulk through QLever's DBpedia endpoint and
   cached in ``<download_dir>/dbpedia/``.
3. MediaWiki action API, only for articles DBpedia lacks: intro extracts, 20
   titles per request (the TextExtracts limit for intros), with
   ``maxlag=5`` and a rate limit. Wikimedia allows 10 requests per minute to
   clients whose User-Agent has no contact details (200 with one), so this
   step is slow; results are appended to ``<download_dir>/extracts/<lang>.jsonl``
   and titles already there are skipped, so an interrupted run resumes.

Parsing (``iter_books``) reads only these files.
"""

import csv
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

from app.domain.book_normalization import clean_isbn13, parse_year, to_iso639_1
from app.domain.entities import Book
from app.domain.ports import BulkBooksSource
from app.domain.value_objects import BookDescription, BookMetadata, SourceLicense, SourceRecord
from app.infrastructure.external.bulk_download import PoliteHttpClient

logger = logging.getLogger(__name__)

SOURCE_NAME = "wikidata"

DEFAULT_LANGUAGES: Tuple[str, ...] = (
    "en", "es", "fr", "de", "it", "pt", "ca", "nl", "ru", "zh", "ja", "ar",
)

QLEVER_ENDPOINT = "https://qlever.dev/api/wikidata"
QLEVER_DBPEDIA_ENDPOINT = "https://qlever.dev/api/dbpedia"
WDQS_ENDPOINT = "https://query.wikidata.org/sparql"

# Direct P31 classes treated as books. Chosen from the classes most used by
# Wikidata items that have an author (P50) and an English article; comics,
# films, games and legal texts are left out on purpose.
BOOK_CLASSES: Dict[str, str] = {
    "Q7725634": "literary work",
    "Q47461344": "written work",
    "Q571": "book",
    "Q1667921": "novel series",
    "Q277759": "book series",
    "Q13593966": "literary trilogy",
    "Q725377": "graphic novel",
    "Q20540385": "non-fiction literary work",
    "Q47068459": "children's book series",
    "Q104213567": "light novel series",
    "Q83790": "textbook",
    "Q193495": "monograph",
    "Q384515": "treatise",
    "Q179461": "religious text",
    "Q699": "fairy tale",
    "Q13136": "reference work",
}

WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
# DBpedia's abstracts come from Wikipedia revisions of late 2022 (before the
# switch to 4.0 in June 2023), so that text is under CC BY-SA 3.0.
DBPEDIA_TEXT_LICENSE = "CC BY-SA 3.0"
WIKIDATA_LICENSE = "CC0 1.0"
EXTRACTS_PER_REQUEST = 20  # TextExtracts maximum when exintro is set
MAX_EXTRACT_CHARS_KEPT = 4000  # raw intros are cut to this on load (the pipeline keeps ~1,200)
MAX_CONSECUTIVE_FAILURES = 10
UNKNOWN_AUTHOR = "Unknown"  # failed batches in a row before the extract download stops

_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

# Property queries that return (item, value) rows.
_ID_PROPERTIES = {"openlibrary": "P648", "gutenberg": "P2034", "isbn13": "P212", "doi": "P356"}


def wiki_url(language: str) -> str:
    """Base IRI of a Wikipedia edition as used by schema:isPartOf."""
    return f"https://{language}.wikipedia.org/"


def article_url(language: str, title: str) -> str:
    """Public URL of a Wikipedia article (also its attribution URL)."""
    return f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}"


def _item_set(languages: Sequence[str]) -> str:
    """SPARQL group binding ?item to every book-like work with an article in the languages."""
    classes = " ".join(f"wd:{qid}" for qid in BOOK_CLASSES)
    wikis = " ".join(f"<{wiki_url(lang)}>" for lang in languages)
    return (
        "{ SELECT DISTINCT ?item WHERE { "
        f"VALUES ?cls {{ {classes} }} ?item wdt:P31 ?cls . "
        f"VALUES ?wiki {{ {wikis} }} ?article schema:about ?item ; schema:isPartOf ?wiki . "
        "} }"
    )


def _label_coalesce(entity: str, out: str, languages: Sequence[str]) -> str:
    """OPTIONAL label patterns plus a COALESCE picking the first available label."""
    order = ["en", "mul"] + [lang for lang in languages if lang != "en"]
    parts = []
    names = []
    for index, lang in enumerate(order):
        var = f"?l{index}"
        names.append(var)
        parts.append(f'OPTIONAL {{ {entity} rdfs:label {var} . FILTER(LANG({var}) = "{lang}") }}')
    return " ".join(parts) + f" BIND(COALESCE({', '.join(names)}) AS {out})"


def build_queries(languages: Sequence[str]) -> Dict[str, str]:
    """
    Build every SPARQL query of the download step, keyed by output file name.

    Each query returns a small number of columns and no row explosion, which
    keeps every result well inside the endpoint's time limit.
    """
    item_set = _item_set(languages)
    wikis = " ".join(f"<{wiki_url(lang)}>" for lang in languages)
    label_langs = ", ".join(f'"{lang}"' for lang in ["mul", *languages])
    classes = " ".join(f"wd:{qid}" for qid in BOOK_CLASSES)
    queries = {
        "articles": (
            "SELECT DISTINCT ?item ?wiki ?title WHERE { "
            f"VALUES ?cls {{ {classes} }} ?item wdt:P31 ?cls . "
            f"VALUES ?wiki {{ {wikis} }} "
            "?article schema:about ?item ; schema:isPartOf ?wiki ; schema:name ?title . }"
        ),
        "wikipedia_editions": (
            "SELECT ?item (COUNT(DISTINCT ?article) AS ?editions) WHERE { "
            f"{item_set} ?article schema:about ?item ; schema:isPartOf ?site . "
            '?site wikibase:wikiGroup "wikipedia" . } GROUP BY ?item'
        ),
        "labels": (
            "SELECT ?item ?lang ?label WHERE { "
            f"{item_set} ?item rdfs:label ?label . BIND(LANG(?label) AS ?lang) "
            f"FILTER(?lang IN ({label_langs})) }}"
        ),
        "authors": (
            "SELECT ?item ?author ?name WHERE { "
            f"{item_set} ?item wdt:P50 ?author . "
            f"{_label_coalesce('?author', '?name', languages)} }}"
        ),
        "creators": (
            "SELECT ?item ?author ?name WHERE { "
            f"{item_set} ?item wdt:P170 ?author . "
            f"{_label_coalesce('?author', '?name', languages)} }}"
        ),
        "genres": (
            "SELECT ?item ?genre ?name WHERE { "
            f"{item_set} ?item wdt:P136 ?genre . "
            f"{_label_coalesce('?genre', '?name', languages)} }}"
        ),
        "dates": (
            "SELECT ?item ?time ?precision WHERE { "
            f"{item_set} ?item p:P577 ?statement . ?statement psv:P577 ?value . "
            "?value wikibase:timeValue ?time ; wikibase:timePrecision ?precision . }"
        ),
        "languages": (
            "SELECT ?item ?code WHERE { "
            f"{item_set} ?item wdt:P407 ?language . ?language wdt:P218 ?code . }}"
        ),
        # The original title (P1476) carries the language of the work as its
        # language tag: the fallback when P407 is missing.
        "title_languages": (
            "SELECT ?item (LANG(?title) AS ?lang) WHERE { "
            f"{item_set} ?item wdt:P1476 ?title . }}"
        ),
    }
    for name, prop in _ID_PROPERTIES.items():
        queries[f"id_{name}"] = f"SELECT ?item ?value WHERE {{ {item_set} ?item wdt:{prop} ?value . }}"
    return {name: _PREFIXES + query for name, query in queries.items()}


def _qid(iri: str) -> str:
    """'http://www.wikidata.org/entity/Q480' -> 'Q480'."""
    return iri.rsplit("/", 1)[-1]


def _wiki_language(iri: str) -> str:
    """'https://es.wikipedia.org/' -> 'es'."""
    return iri.split("//", 1)[-1].split(".", 1)[0]


@dataclass
class _WorkData:
    """Everything known about one Wikidata work while the CSV files are joined."""

    articles: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    authors: List[Tuple[str, str]] = field(default_factory=list)
    creators: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    title_languages: List[str] = field(default_factory=list)
    openlibrary: List[str] = field(default_factory=list)
    gutenberg: List[str] = field(default_factory=list)
    isbn13: List[str] = field(default_factory=list)
    doi: List[str] = field(default_factory=list)
    editions: Optional[int] = None


class WikidataWikipediaSource(BulkBooksSource):
    """BulkBooksSource over Wikidata works and their Wikipedia intros."""

    def __init__(
        self,
        download_dir: Path,
        languages: Sequence[str] = DEFAULT_LANGUAGES,
        sparql_endpoint: str = QLEVER_ENDPOINT,
        dbpedia_endpoint: str = QLEVER_DBPEDIA_ENDPOINT,
    ) -> None:
        """
        Args:
            download_dir: Directory with the cached SPARQL results and extracts
            languages: Wikipedia languages, in priority order
            sparql_endpoint: Wikidata SPARQL endpoint used by download()
            dbpedia_endpoint: DBpedia SPARQL endpoint used by download()
        """
        self._dir = Path(download_dir)
        self._languages = tuple(languages)
        self._endpoint = sparql_endpoint
        self._dbpedia_endpoint = dbpedia_endpoint
        self._rejected: Dict[str, int] = {}

    # ------------------------------------------------------------------ port

    def get_source_name(self) -> str:
        """Source identifier."""
        return SOURCE_NAME

    def get_license(self) -> SourceLicense:
        """Wikidata data is CC0; Wikipedia intros are CC BY-SA 4.0 (attribute per record)."""
        return SourceLicense(
            source=SOURCE_NAME,
            name="Wikidata + Wikipedia",
            license="Wikidata: CC0 1.0; Wikipedia text: CC BY-SA 4.0 (API) / CC BY-SA 3.0 (via DBpedia)",
            terms_url="https://www.wikidata.org/wiki/Wikidata:Licensing ; "
            "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
            attribution=(
                "Each Wikipedia description keeps the article URL as attribution; the text "
                "is CC BY-SA (3.0 when taken from DBpedia, 4.0 when taken from the API) and "
                "share-alike applies to it."
            ),
            notes=(
                "Structured data from Wikidata needs no attribution (CC0). Wikidata was "
                "queried through QLever (live-updated copy of the official dump) because "
                "WDQS times out on this query. Intros come from DBpedia abstracts (Wikipedia "
                "revisions of late 2022) and, where DBpedia lacks one, from the Wikipedia API."
            ),
        )

    def get_rejected_counts(self) -> Dict[str, int]:
        """Works skipped by iter_books(), by reason."""
        return dict(self._rejected)

    def iter_books(self, limit: Optional[int] = None) -> Iterator[Book]:
        """
        Join the cached SPARQL results and extracts into one Book per work.

        Works are yielded by decreasing number of Wikipedia editions, so a
        ``limit`` keeps the best-known works.

        Raises:
            FileNotFoundError: If the SPARQL results have not been downloaded
        """
        self._rejected = {}
        works = self._load_works()
        extracts = self._load_extracts(works)
        abstracts = self._load_dbpedia_abstracts()
        ordered = sorted(works.items(), key=lambda kv: (-(kv[1].editions or 0), kv[0]))
        produced = 0
        for qid, data in ordered:
            if limit is not None and produced >= limit:
                break
            book = self._to_book(qid, data, extracts, abstracts.get(qid, {}))
            if book is not None:
                produced += 1
                yield book

    # -------------------------------------------------------------- download

    def download(
        self,
        sparql_client: PoliteHttpClient,
        wiki_client: PoliteHttpClient,
        extract_languages: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        workers: int = 1,
        time_budget_s: Optional[float] = None,
    ) -> Dict[str, int]:
        """
        Fetch the SPARQL results, DBpedia abstracts and missing Wikipedia intros.

        Steps (each cached and resumable):

        1. Wikidata SPARQL results (``download_sparql``).
        2. DBpedia abstracts for every work (``download_dbpedia_abstracts``):
           the Wikipedia intros extracted by DBpedia, fetched in bulk through
           SPARQL. They cover most works that have an English article.
        3. Wikipedia API intros only for the (language, article) pairs DBpedia
           lacks (``download_extracts``). Works with no description at all
           come first (one article each: the work's own language if it has
           one, else English, else the configured order with Spanish first);
           then the remaining pairs, Spanish first and then the configured
           order. Batches run language by language in that order. Wikimedia
           allows 10 requests per minute to clients whose User-Agent carries
           no contact details, so this step is slow and can be stopped and
           resumed at any time.

        Args:
            sparql_client: Client for the SPARQL endpoints
            wiki_client: Client for the Wikipedia APIs (its rate limit applies)
            extract_languages: Languages whose intros may be fetched from the API
            limit: Only consider the ``limit`` works with most editions
            workers: Parallel requests to Wikipedia (at most 3 per Wikimedia)
            time_budget_s: Stop the API step after this many seconds

        Returns:
            Number of (language, article) pairs still without a description
        """
        self.download_sparql(sparql_client)
        works = self._load_works()
        selected = sorted(works.items(), key=lambda kv: (-(kv[1].editions or 0), kv[0]))
        if limit is not None:
            selected = selected[:limit]
        self.download_dbpedia_abstracts(sparql_client, [qid for qid, _ in selected])
        abstracts = self._load_dbpedia_abstracts()
        languages = list(extract_languages or self._languages)
        phases = self.plan_extract_phases(selected, abstracts, languages)
        return self.download_extracts(wiki_client, phases, workers, time_budget_s)

    def plan_extract_phases(
        self,
        works: Sequence[Tuple[str, "_WorkData"]],
        abstracts: Dict[str, Dict[str, str]],
        languages: Sequence[str],
    ) -> List[Dict[str, List[str]]]:
        """
        Order the Wikipedia API requests: rescue undescribed works first.

        Returns:
            Two phases, each mapping language -> article titles to fetch
        """
        api_order = ["es", *[lang for lang in languages if lang != "es"]] if "es" in languages else list(languages)
        rescue: Dict[str, List[str]] = {lang: [] for lang in api_order}
        enrich: Dict[str, List[str]] = {lang: [] for lang in api_order}
        for qid, data in works:
            have = abstracts.get(qid, {})
            missing = [lang for lang in api_order if lang in data.articles and lang not in have]
            if not missing:
                continue
            if not any(lang in have for lang in data.articles):
                # One article per undescribed work: its own language's, else
                # English (so a work of unknown language is not described, and
                # then labelled, in Spanish only because Spanish comes first).
                work_language = _work_language(data, self._languages)
                preferred = [work_language, "en", *api_order]
                first = next(lang for lang in preferred if lang in missing)
                rescue[first].append(data.articles[first])
                missing.remove(first)
            for lang in missing:
                enrich[lang].append(data.articles[lang])
        return [rescue, enrich]

    def download_dbpedia_abstracts(
        self, client: PoliteHttpClient, qids: Sequence[str], batch_size: int = 2000
    ) -> None:
        """
        Fetch DBpedia abstracts (Wikipedia intros) for the works, in cached batches.

        DBpedia links its resources to Wikidata with owl:sameAs and stores the
        intro of the article in many languages as dbo:abstract. The batches
        are cut from the sorted QID list, so a re-run reuses the cached files.
        """
        out_dir = self._dir / "dbpedia"
        out_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(set(qids), key=lambda q: int(q[1:]))
        langs = ", ".join(f'"{lang}"' for lang in self._languages)
        for index, start in enumerate(range(0, len(ordered), batch_size)):
            target = out_dir / f"abstracts_{index:04d}.csv"
            if target.exists() and target.stat().st_size > 0:
                continue
            values = " ".join(f"<http://www.wikidata.org/entity/{q}>" for q in ordered[start:start + batch_size])
            query = (
                "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                "PREFIX dbo: <http://dbpedia.org/ontology/>\n"
                "SELECT ?item ?resource ?lang (SUBSTR(STR(?abstract), 1, 2000) AS ?text) WHERE { "
                f"VALUES ?item {{ {values} }} ?resource owl:sameAs ?item . "
                "?resource dbo:abstract ?abstract . BIND(LANG(?abstract) AS ?lang) "
                f"FILTER(?lang IN ({langs})) }}"
            )
            response = client.post(self._dbpedia_endpoint, data={"query": query}, headers={"Accept": "text/csv"})
            if not response.text.startswith("item"):
                raise RuntimeError(f"Unexpected DBpedia answer: {response.text[:300]}")
            tmp = target.with_suffix(".tmp")
            tmp.write_text(response.text, encoding="utf-8")
            tmp.replace(target)
            logger.info("DBpedia batch %d: %d rows", index, response.text.count("\n") - 1)

    def _load_dbpedia_abstracts(self) -> Dict[str, Dict[str, str]]:
        """Cached DBpedia abstracts: QID -> language -> text."""
        abstracts: Dict[str, Dict[str, str]] = {}
        for path in sorted((self._dir / "dbpedia").glob("abstracts_*.csv")):
            with open(path, encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["text"]:
                        abstracts.setdefault(_qid(row["item"]), {}).setdefault(row["lang"], row["text"])
        return abstracts

    def download_sparql(self, client: PoliteHttpClient) -> None:
        """Run every SPARQL query whose CSV file is not cached yet."""
        out_dir = self._dir / "sparql"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, query in build_queries(self._languages).items():
            target = out_dir / f"{name}.csv"
            if target.exists() and target.stat().st_size > 0:
                continue
            (out_dir / f"{name}.rq").write_text(query, encoding="utf-8")
            logger.info("SPARQL %s ...", name)
            response = client.post(
                self._endpoint, data={"query": query}, headers={"Accept": "text/csv"}
            )
            if not response.text.startswith("item"):
                raise RuntimeError(f"Unexpected SPARQL answer for {name}: {response.text[:300]}")
            tmp = target.with_suffix(".tmp")
            tmp.write_text(response.text, encoding="utf-8")
            tmp.replace(target)
            logger.info("SPARQL %s: %d rows", name, response.text.count("\n") - 1)

    def download_extracts(
        self,
        client: PoliteHttpClient,
        phases: List[Dict[str, List[str]]],
        workers: int = 1,
        time_budget_s: Optional[float] = None,
    ) -> Dict[str, int]:
        """
        Fetch Wikipedia intro extracts through the API (resumable).

        Titles are grouped in batches of 20 of the same language and fetched
        phase by phase, language by language, in the given order. Each batch
        is appended to the language's JSONL file as soon as it arrives, and
        titles already in the file are skipped, so the step can be stopped
        and run again. After MAX_CONSECUTIVE_FAILURES failed batches in a
        row, or when the time budget runs out, no further request is sent.

        Args:
            client: Client for the Wikipedia APIs (its rate limit applies)
            phases: Ordered list of {language: titles} to fetch
            workers: Parallel requests (Wikimedia asks for at most 3)
            time_budget_s: Stop sending requests after this many seconds

        Returns:
            Number of requested titles still missing, per language
        """
        out_dir = self._dir / "extracts"
        out_dir.mkdir(parents=True, exist_ok=True)
        languages = list(dict.fromkeys(lang for phase in phases for lang in phase))
        locks = {lang: threading.Lock() for lang in languages}
        done_by_lang = {lang: _done_titles(out_dir / f"{lang}.jsonl") for lang in languages}
        tasks: List[Tuple[str, List[str]]] = []
        for number, phase in enumerate(phases, start=1):
            for lang, lang_titles in phase.items():
                remaining = [t for t in dict.fromkeys(lang_titles) if t not in done_by_lang[lang]]
                logger.info("Extracts phase %d %s: %d titles, %d to fetch",
                            number, lang, len(lang_titles), len(remaining))
                for start in range(0, len(remaining), EXTRACTS_PER_REQUEST):
                    tasks.append((lang, remaining[start:start + EXTRACTS_PER_REQUEST]))

        # Wall-clock deadline: on macOS time.monotonic() stops while the
        # machine is suspended, which would stretch the budget.
        deadline = time.time() + time_budget_s if time_budget_s else None
        progress = {"done": 0, "consecutive_failures": 0}
        progress_lock = threading.Lock()

        def run(task: Tuple[str, List[str]]) -> None:
            lang, batch = task
            if progress["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                return  # circuit open: stop sending requests, rerun later to resume
            if deadline is not None and time.time() > deadline:
                return
            try:
                pages = fetch_intro_extracts(client, lang, batch)
            except RuntimeError as exc:
                with progress_lock:
                    progress["consecutive_failures"] += 1
                logger.error("Extract batch failed for %s (%s); it will be retried on the next run",
                             lang, str(exc)[:200])
                return
            with locks[lang], open(out_dir / f"{lang}.jsonl", "a", encoding="utf-8") as out:
                for record in pages:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
            with progress_lock:
                progress["consecutive_failures"] = 0
                progress["done"] += 1
                if progress["done"] % 50 == 0:
                    logger.info("Extract batches done: %d / %d", progress["done"], len(tasks))

        with ThreadPoolExecutor(max_workers=max(1, min(workers, 3))) as pool:
            list(pool.map(run, tasks))
        if progress["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            logger.error("Stopped after %d consecutive failed batches", MAX_CONSECUTIVE_FAILURES)

        missing: Dict[str, int] = {}
        for lang in languages:
            done = _done_titles(out_dir / f"{lang}.jsonl")
            wanted = {t for phase in phases for t in phase.get(lang, [])}
            missing[lang] = sum(1 for t in wanted if t not in done)
        return missing

    # --------------------------------------------------------------- parsing

    def _read_csv(self, name: str) -> Iterable[Dict[str, str]]:
        path = self._dir / "sparql" / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run the download step for wikidata first")
        with open(path, encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)

    def _optional_rows(self, each, name: str):
        """Rows of a cached query added after the first download (empty if not fetched)."""
        if not (self._dir / "sparql" / f"{name}.csv").exists():
            logger.warning("No %s.csv: run the download step to fetch it", name)
            return iter(())
        return each(name)

    def _load_works(self) -> Dict[str, _WorkData]:
        """Join the per-property CSV files into one record per work."""
        works: Dict[str, _WorkData] = {}
        for row in self._read_csv("articles"):
            lang = _wiki_language(row["wiki"])
            if lang in self._languages:
                works.setdefault(_qid(row["item"]), _WorkData()).articles[lang] = row["title"]

        def each(name: str) -> Iterable[Tuple[_WorkData, Dict[str, str]]]:
            for row in self._read_csv(name):
                data = works.get(_qid(row["item"]))
                if data is not None:
                    yield data, row

        for data, row in each("wikipedia_editions"):
            data.editions = int(row["editions"])
        for data, row in each("labels"):
            data.labels.setdefault(row["lang"], row["label"])
        for data, row in each("authors"):
            name = (row.get("name") or "").strip()
            author_qid = _qid(row["author"])
            if name and not any(q == author_qid for q, _ in data.authors):
                data.authors.append((author_qid, name))
        for data, row in self._optional_rows(each, "creators"):
            # Fallback author: creator (P170) when the work has no author (P50).
            name = (row.get("name") or "").strip()
            if name and not data.authors:
                data.creators.append(name)
        for data, row in each("genres"):
            name = (row.get("name") or "").strip()
            if name and name not in data.genres:
                data.genres.append(name)
        for data, row in each("dates"):
            if int(row["precision"] or 0) >= 9:  # 9 = year, 10 = month, 11 = day
                year = parse_year(row["time"])
                if year is not None:
                    data.years.append(year)
        for data, row in each("languages"):
            code = to_iso639_1(row["code"])
            if code and code not in data.languages:
                data.languages.append(code)
        for data, row in self._optional_rows(each, "title_languages"):
            code = to_iso639_1(row["lang"])
            if code and code not in data.title_languages:
                data.title_languages.append(code)
        for name in _ID_PROPERTIES:
            for data, row in each(f"id_{name}"):
                values: List[str] = getattr(data, name)
                value = row["value"].strip()
                if value and value not in values:
                    values.append(value)
        return works

    def _load_extracts(self, works: Dict[str, _WorkData]) -> Dict[str, Dict[str, str]]:
        """Read the cached intros: language -> requested title -> extract."""
        wanted: Dict[str, Set[str]] = {lang: set() for lang in self._languages}
        for data in works.values():
            for lang, title in data.articles.items():
                wanted[lang].add(title)
        extracts: Dict[str, Dict[str, str]] = {}
        for lang in self._languages:
            path = self._dir / "extracts" / f"{lang}.jsonl"
            extracts[lang] = {}
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue  # last line of a run that was interrupted mid-write
                    text = record.get("extract")
                    if text and record["title"] in wanted[lang]:
                        extracts[lang][record["title"]] = text[:MAX_EXTRACT_CHARS_KEPT]
        return extracts

    def _to_book(
        self,
        qid: str,
        data: _WorkData,
        extracts: Dict[str, Dict[str, str]],
        abstracts: Dict[str, str],
    ) -> Optional[Book]:
        """Build the raw Book of one work (None, counted, if it has no title or author)."""
        work_language = _work_language(data, self._languages)
        title = _pick_title(data, work_language, self._languages)
        if not title:
            self._reject("no_title")
            return None
        # Works with no author (P50) nor creator (P170) are traditional or
        # anonymous texts (scriptures, epics, "Lazarillo de Tormes") or works
        # whose author Wikidata does not record. They are kept with the label
        # "Unknown" (Project Gutenberg's label for the same case) instead of
        # being dropped; "Anonymous" would be wrong for the second group.
        authors = [name for _, name in data.authors] or list(data.creators) or [UNKNOWN_AUTHOR]

        descriptions = []
        provenance = [
            SourceRecord(SOURCE_NAME, qid, WIKIDATA_LICENSE, f"https://www.wikidata.org/wiki/{qid}")
        ]
        for lang in self._languages:
            article = data.articles.get(lang)
            if not article:
                continue
            # A fresh intro from the Wikipedia API wins over DBpedia's 2022 copy.
            text = extracts.get(lang, {}).get(article)
            license_name = WIKIPEDIA_LICENSE
            if not text and abstracts.get(lang):
                text = abstracts[lang][:MAX_EXTRACT_CHARS_KEPT]
                license_name = DBPEDIA_TEXT_LICENSE
            if not text:
                continue
            url = article_url(lang, article)
            descriptions.append(
                BookDescription(
                    text=text,
                    language=lang,
                    source="wikipedia",
                    license=license_name,
                    attribution_url=url,
                )
            )
            provenance.append(SourceRecord("wikipedia", f"{lang}:{article}", license_name, url))

        titles = tuple(
            (lang, label)
            for lang, label in sorted(data.labels.items())
            if lang != "mul" and label != title
        )
        isbn13s = tuple(dict.fromkeys(i for i in (clean_isbn13(v) for v in data.isbn13) if i))
        metadata = BookMetadata(
            isbn13=isbn13s[0] if isbn13s else None,
            work_language=work_language,
            descriptions=tuple(descriptions),
            titles=titles,
            wikidata_id=qid,
            openlibrary_ids=tuple(v for v in data.openlibrary if v.startswith("OL") and v.endswith("W")),
            gutenberg_ids=tuple(v for v in data.gutenberg if v.isdigit()),
            isbn13s=isbn13s,
            doi=data.doi[0].lower() if data.doi else None,
            wikipedia_editions=data.editions,
            provenance=tuple(provenance),
        )
        year = min(data.years) if data.years else None
        return Book(
            id=uuid5(NAMESPACE_URL, f"{SOURCE_NAME}:{qid}"),
            title=title,
            authors=authors,
            description=None,
            language=work_language,
            categories=list(data.genres),
            published_date=_year_to_date(year),
            source=SOURCE_NAME,
            source_id=qid,
            metadata=metadata,
        )

    def _reject(self, reason: str) -> None:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1


def fetch_intro_extracts(client: PoliteHttpClient, language: str, titles: List[str]) -> List[Dict]:
    """
    Fetch the plain-text intro of up to 20 Wikipedia articles.

    Follows title normalisation and redirects, and the API's continuation
    when not every extract fits in one answer.

    Returns:
        One record per requested title: title, page_title, pageid, extract
        (extract is None for missing pages or empty intros)
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exsectionformat": "plain",
        "exlimit": str(EXTRACTS_PER_REQUEST),
        "redirects": "1",
        "maxlag": "5",
        "titles": "|".join(titles),
    }
    url = f"https://{language}.wikipedia.org/w/api.php"
    pages: Dict[str, Dict] = {}
    aliases: Dict[str, str] = {}
    cont: Dict[str, str] = {}
    while True:
        response = client.get(url, params={**params, **cont})
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"API error {body['error'].get('code')}: {body['error'].get('info')}")
        query = body.get("query", {})
        for item in query.get("normalized", []) + query.get("redirects", []):
            aliases[item["from"]] = item["to"]
        for page in query.get("pages", []):
            known = pages.setdefault(page["title"], page)
            if page.get("extract") and not known.get("extract"):
                known["extract"] = page["extract"]
        if "continue" not in body:
            break
        cont = {k: str(v) for k, v in body["continue"].items()}

    records = []
    for title in titles:
        final = title
        for _ in range(3):  # normalisation, then up to two redirect hops
            final = aliases.get(final, final)
        page = pages.get(final, {})
        records.append({
            "title": title,
            "page_title": page.get("title"),
            "pageid": page.get("pageid"),
            "extract": page.get("extract") or None,
        })
    return records


def _done_titles(path: Path) -> Set[str]:
    """Titles already present in an extracts JSONL file."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["title"])
            except (ValueError, KeyError):
                continue  # a line cut by an interrupted write: refetch it
    return done


def _work_language(data: "_WorkData", priority: Sequence[str]) -> Optional[str]:
    """Language of the work: P407 if stated, else the language tag of its title (P1476)."""
    return _first_in_priority(data.languages, priority) or _first_in_priority(data.title_languages, priority)


def _first_in_priority(codes: Sequence[str], priority: Sequence[str]) -> Optional[str]:
    """First code in priority order, else the first code at all."""
    for lang in priority:
        if lang in codes:
            return lang
    return codes[0] if codes else None


def _pick_title(data: _WorkData, work_language: Optional[str], languages: Sequence[str]) -> Optional[str]:
    """Label in the work's language, else the multilingual label, else by language priority."""
    for lang in [work_language, "mul", *languages]:
        if lang and data.labels.get(lang):
            return data.labels[lang]
    if data.labels:
        return next(iter(data.labels.values()))
    for lang in languages:
        if lang in data.articles:
            return data.articles[lang].split(" (")[0]
    return None


def _year_to_date(year: Optional[int]) -> Optional[datetime]:
    """Year -> January 1st of that year (the catalog stores year precision only)."""
    if year is None or year < 1:
        return None
    return datetime(year, 1, 1)
