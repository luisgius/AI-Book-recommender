"""
Sample of books, queries and relevant works of the known-item evaluation (pure).

Sampling rule (PROTOCOL.md, section 3):
1. A book is eligible when its language is one of the twelve languages of
   V14, it has a non-empty title, its first author names a person (not empty
   and not a placeholder such as "Unknown", "Anonymous" or "Various"), and its
   normalised title is unique in the whole catalog (50 000 works).
2. Per language, the eligible books are sorted by id and shuffled with a
   random generator seeded with SEED; books are taken in that order, skipping
   a book whose normalised first author was already taken (in any language),
   until BOOKS_PER_LANGUAGE are taken. The skip keeps the 300 author queries
   distinct.
3. Split: per language, the taken books are shuffled again with the same
   generator; the first 13 go to the development half for the languages in
   even positions of LANGUAGES and the first 12 for those in odd positions
   (150 + 150).

Queries (three per book, built mechanically): T = the title as stored;
TA = the title, a space and the first author; A = the first author alone.
Relevant works: for T and TA, the book itself; for A, every catalog work
whose author list holds an author with the same normalised name.
"""

import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence, Set

from app.evaluation.known_item import design as D

_NOT_WORD = re.compile(r"[\W_]+", re.UNICODE)


class BookLike(Protocol):
    """The fields of the domain Book the sampling reads."""

    id: Any
    title: str
    authors: List[str]
    language: str


def normalise(text: str) -> str:
    """
    Normalised form of a title or an author name: Unicode NFKC, case folded,
    every run of characters that are not letters or digits turned into one
    space, trimmed. "Dune." and "dune" are equal; accents are kept.
    """
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _NOT_WORD.sub(" ", folded).strip()


def first_author(book: BookLike) -> str:
    return book.authors[0].strip() if book.authors else ""


def catalog_indexes(books: Iterable[BookLike]) -> Dict[str, Any]:
    """
    One pass over the catalog: title counts by normalised title, works by
    normalised author name, and the light records of every book.
    """
    titles: Counter = Counter()
    by_author: Dict[str, Set[str]] = defaultdict(set)
    records: Dict[str, Dict[str, Any]] = {}
    for book in books:
        book_id = str(book.id)
        titles[normalise(book.title)] += 1
        for author in book.authors:
            key = normalise(author)
            if key:
                by_author[key].add(book_id)
        records[book_id] = {"book_id": book_id, "title": book.title, "authors": list(book.authors),
                            "language": book.language}
    return {"titles": titles, "by_author": by_author, "records": records}


def eligibility(record: Mapping[str, Any], titles: Mapping[str, int]) -> str:
    """'ok' or the first reason the book is not eligible."""
    if record["language"] not in D.LANGUAGES:
        return "language"
    if not normalise(record["title"]):
        return "no_title"
    if not record["authors"] or normalise(record["authors"][0]) in ("",) + D.PLACEHOLDER_AUTHORS:
        return "no_author"
    if titles[normalise(record["title"])] != 1:
        return "title_not_unique"
    return "ok"


def sample_books(indexes: Mapping[str, Any], per_language: int = D.BOOKS_PER_LANGUAGE,
                 seed: int = D.SEED) -> Dict[str, Any]:
    """The sample (steps 1-3 of the rule) with its counts per language."""
    rng = random.Random(seed)
    records, titles = indexes["records"], indexes["titles"]
    reasons: Dict[str, Counter] = defaultdict(Counter)
    eligible: Dict[str, List[str]] = defaultdict(list)
    for book_id in sorted(records):
        record = records[book_id]
        reason = eligibility(record, titles)
        reasons[record["language"] if record["language"] in D.LANGUAGES else "other"][reason] += 1
        if reason == "ok":
            eligible[record["language"]].append(book_id)
    taken_authors: Set[str] = set()
    chosen: Dict[str, List[str]] = {}
    skipped_author: Counter = Counter()
    for language in D.LANGUAGES:
        candidates = list(eligible[language])
        rng.shuffle(candidates)
        chosen[language] = []
        for book_id in candidates:
            if len(chosen[language]) == per_language:
                break
            author = normalise(records[book_id]["authors"][0])
            if author in taken_authors:
                skipped_author[language] += 1
                continue
            taken_authors.add(author)
            chosen[language].append(book_id)
        if len(chosen[language]) < per_language:
            raise ValueError(f"only {len(chosen[language])} eligible books in {language}, need {per_language}")
    halves: Dict[str, str] = {}
    for position, language in enumerate(D.LANGUAGES):
        order = list(chosen[language])
        rng.shuffle(order)
        n_dev = (per_language + 1) // 2 if position % 2 == 0 else per_language // 2
        for i, book_id in enumerate(order):
            halves[book_id] = D.DEV if i < n_dev else D.TEST
    counts = {language: {"catalog": sum(reasons[language].values()), "eligible": len(eligible[language]),
                         "not_eligible": {k: v for k, v in reasons[language].items() if k != "ok"},
                         "skipped_repeated_author": skipped_author[language],
                         "sampled": len(chosen[language]),
                         "dev": sum(halves[b] == D.DEV for b in chosen[language]),
                         "test": sum(halves[b] == D.TEST for b in chosen[language])}
              for language in D.LANGUAGES}
    counts["other_languages"] = {"catalog": sum(reasons["other"].values())}
    books = [{**records[b], "half": halves[b]} for language in D.LANGUAGES for b in chosen[language]]
    return {"books": books, "counts": counts}


def build_queries(books: Sequence[Mapping[str, Any]], by_author: Mapping[str, Set[str]]) -> List[Dict[str, Any]]:
    """Three queries per sampled book, with their relevant work ids."""
    queries = []
    for number, book in enumerate(books, start=1):
        author = book["authors"][0].strip()
        relevant_by_author = sorted(by_author[normalise(author)])
        texts = {D.T: book["title"], D.TA: f"{book['title']} {author}", D.A: author}
        for query_type in D.QUERY_TYPES:
            queries.append({"query_id": f"k{number:03d}-{query_type}", "book_id": book["book_id"],
                            "query_type": query_type, "language": book["language"], "half": book["half"],
                            "text": texts[query_type],
                            "relevant": relevant_by_author if query_type == D.A else [book["book_id"]]})
    return queries


def build_sample(books: Iterable[BookLike]) -> Dict[str, Any]:
    """The whole sample document (books, counts, queries) from an iterable over the catalog."""
    indexes = catalog_indexes(books)
    sample = sample_books(indexes)
    queries = build_queries(sample["books"], indexes["by_author"])
    a_sizes = sorted(len(q["relevant"]) for q in queries if q["query_type"] == D.A)
    return {
        "rule": (__doc__ or "").strip(),
        "seed": D.SEED,
        "catalog_works": len(indexes["records"]),
        "languages": list(D.LANGUAGES),
        "books_per_language": D.BOOKS_PER_LANGUAGE,
        "counts": sample["counts"],
        "author_query_relevant": {"min": a_sizes[0], "median": a_sizes[len(a_sizes) // 2], "max": a_sizes[-1],
                                  "single_work": sum(1 for n in a_sizes if n == 1)},
        "books": sample["books"],
        "queries": queries,
    }
