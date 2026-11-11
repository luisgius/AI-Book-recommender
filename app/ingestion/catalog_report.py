"""
Statistics and Markdown report of a built multi-source catalog.

``compute_catalog_report`` walks the catalog once (streaming, so it works on
a million books) and combines what it counts with the per-source ingestion
counters stored by the build. ``render_markdown`` turns the result into
``CATALOG_REPORT.md``; the same dict is written as JSON next to it.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

from app.domain.entities import Book

LENGTH_BUCKETS = [(200, 399), (400, 599), (600, 799), (800, 999), (1000, 1199), (1200, 10**9)]
IDENTIFIERS = ("wikidata_id", "openlibrary_ids", "gutenberg_ids", "isbn13s", "doi")


def _family(source: str) -> str:
    """Catalog source of a provenance record ('wikipedia' text comes with 'wikidata')."""
    return "wikidata" if source == "wikipedia" else source


def _percentile(sorted_values: List[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def compute_catalog_report(
    books: Iterable[Book],
    build_sources: Sequence[Dict],
    languages: Sequence[str],
    top_subjects: int = 30,
) -> Dict:
    """
    Compute every statistic of the catalog report.

    Args:
        books: All books of the catalog (streamed)
        build_sources: Per-source records stored by the build (status, stats, licence)
        languages: Configured languages, in priority order
        top_subjects: How many subjects to list

    Returns:
        A JSON-serialisable dict
    """
    total = 0
    by_language: Counter = Counter()
    work_language_known = 0
    description_languages: Counter = Counter()
    primary_language: Counter = Counter()
    multi_language = 0
    n_descriptions: Counter = Counter()
    lengths: List[int] = []
    has = Counter()
    subjects: Counter = Counter()
    families: Counter = Counter()
    family_sets: Counter = Counter()
    description_sources: Counter = Counter()
    description_licenses: Counter = Counter()
    auto_primary = 0
    popularity = Counter()
    years: List[int] = []

    for book in books:
        total += 1
        meta = book.metadata
        by_language[book.language or "unknown"] += 1
        if meta and meta.work_language:
            work_language_known += 1
        descriptions = meta.descriptions if meta else ()
        langs = [d.language for d in descriptions if d.language]
        for lang in langs:
            description_languages[lang] += 1
        if descriptions:
            primary_language[descriptions[0].language or "unknown"] += 1
            if descriptions[0].auto_generated:
                auto_primary += 1
        for description in descriptions:
            description_sources[description.source] += 1
            description_licenses[description.license or "unknown"] += 1
        n_descriptions[min(len(descriptions), 5)] += 1
        if len(set(langs)) > 1:
            multi_language += 1
        if book.description:
            lengths.append(len(book.description))
        year = book.get_published_year()
        if year is not None:
            has["year"] += 1
            years.append(year)
        if book.categories:
            has["subjects"] += 1
            subjects.update(book.categories)
        if meta:
            for name in IDENTIFIERS:
                if getattr(meta, name):
                    has[name] += 1
            if meta.wikipedia_editions:
                popularity["wikipedia_editions"] += 1
            if meta.gutenberg_downloads:
                popularity["gutenberg_downloads"] += 1
            sources = sorted({_family(r.source) for r in meta.provenance})
            for source in sources:
                families[source] += 1
            family_sets[" + ".join(sources) or "unknown"] += 1

    lengths.sort()
    years.sort()
    length_buckets = {
        (f"{low}-{high}" if high < 10**9 else f"{low}+"): sum(1 for n in lengths if low <= n <= high)
        for low, high in LENGTH_BUCKETS
    }

    def share(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    sources_summary = []
    for record in build_sources:
        stats = record.get("stats") or {}
        sources_summary.append({
            "source": record["source"],
            "status": record["status"],
            "read": stats.get("read"),
            "kept_after_filter": stats.get("kept"),
            "rejected": stats.get("rejected", {}),
            "inserted_as_new": stats.get("inserted"),
            "merged_into_existing": stats.get("merged"),
            "merged_cross_source": stats.get("merged_cross_source"),
            "merged_by_rule": stats.get("merged_by_rule", {}),
            "years_dropped": stats.get("years_dropped"),
            "seconds": stats.get("seconds"),
            "descriptions_language_detected": stats.get("descriptions_detected"),
            "works_with_this_source": families.get(record["source"], 0),
            "license": record.get("license"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
        })

    return {
        "total_works": total,
        "sources": sources_summary,
        "works_by_source_combination": dict(family_sets.most_common()),
        "works_with_more_than_one_source": sum(c for k, c in family_sets.items() if "+" in k),
        "work_language": {
            "distribution": dict(by_language.most_common()),
            "stated_by_a_source": work_language_known,
            "taken_from_primary_description": total - work_language_known,
        },
        "description_language": {
            "works_with_a_description_in": {
                lang: description_languages.get(lang, 0) for lang in languages
            },
            "primary_description_language": dict(primary_language.most_common()),
            "works_with_more_than_one_description_language": multi_language,
            "share_with_more_than_one_description_language": share(multi_language),
            "descriptions_per_work": {str(k) if k < 5 else "5+": v for k, v in sorted(n_descriptions.items())},
            "descriptions_by_source": dict(description_sources.most_common()),
            "descriptions_by_license": dict(description_licenses.most_common()),
            "primary_description_auto_generated": auto_primary,
        },
        "primary_description_length": {
            "min": lengths[0] if lengths else 0,
            "p25": _percentile(lengths, 0.25),
            "median": _percentile(lengths, 0.5),
            "p75": _percentile(lengths, 0.75),
            "max": lengths[-1] if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "buckets": length_buckets,
        },
        "coverage": {
            "year": share(has["year"]),
            "subjects": share(has["subjects"]),
            "isbn13": share(has["isbn13s"]),
            "wikidata_id": share(has["wikidata_id"]),
            "openlibrary_id": share(has["openlibrary_ids"]),
            "gutenberg_id": share(has["gutenberg_ids"]),
            "doi": share(has["doi"]),
            "popularity_wikipedia_editions": share(popularity["wikipedia_editions"]),
            "popularity_gutenberg_downloads": share(popularity["gutenberg_downloads"]),
        },
        "years": {
            "min": years[0] if years else None,
            "median": _percentile(years, 0.5) if years else None,
            "max": years[-1] if years else None,
        },
        "top_subjects": dict(subjects.most_common(top_subjects)),
    }


def _table(headers: List[str], rows: List[List]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return lines


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_markdown(report: Dict, title: str = "Multi-source catalog report", extra: Optional[List[str]] = None) -> str:
    """Render the report dict as Markdown."""
    total = report["total_works"]
    out = [f"# {title}", "", f"Total works in the catalog: **{total:,}**", ""]
    if extra:
        out += extra + [""]

    out += ["## Records per source", ""]
    rows = []
    for s in report["sources"]:
        rejected = ", ".join(f"{k} {v:,}" for k, v in sorted(s["rejected"].items(), key=lambda kv: -kv[1]))
        rows.append([
            s["source"], s["status"], f"{s['read'] or 0:,}", f"{s['kept_after_filter'] or 0:,}",
            f"{s['inserted_as_new'] or 0:,}", f"{s['merged_into_existing'] or 0:,}",
            f"{s['merged_cross_source'] or 0:,}", f"{s['works_with_this_source']:,}", rejected,
            s.get("seconds"),
        ])
    out += _table(
        ["Source", "Status", "Read", "Kept (passed filter)", "New works", "Merged", "Merged across sources",
         "Works containing it", "Rejected (reason count)", "Ingest time (s)"],
        rows,
    )
    out += ["", "Merges by rule:", ""]
    for s in report["sources"]:
        if s["merged_by_rule"]:
            out.append(f"- {s['source']}: " + ", ".join(f"{k} {v:,}" for k, v in s["merged_by_rule"].items()))
    out += ["", f"Works built from more than one source: **{report['works_with_more_than_one_source']:,}**", ""]
    out += _table(["Sources of the work", "Works"], [[k, f"{v:,}"] for k, v in report["works_by_source_combination"].items()])

    wl = report["work_language"]
    out += ["", "## Language of the work", "",
            f"Stated by a source: {wl['stated_by_a_source']:,}; taken from the primary description: "
            f"{wl['taken_from_primary_description']:,}.", ""]
    out += _table(["Language", "Works", "Share"],
                  [[k, f"{v:,}", _pct(v / total if total else 0)] for k, v in wl["distribution"].items()][:25])

    dl = report["description_language"]
    out += ["", "## Description languages", "",
            f"Works with descriptions in more than one language: {dl['works_with_more_than_one_description_language']:,} "
            f"({_pct(dl['share_with_more_than_one_description_language'])}).",
            f"Primary descriptions generated automatically (Gutenberg summaries): {dl['primary_description_auto_generated']:,}.", ""]
    out += _table(["Language", "Works with a description in it", "Works whose primary description is in it"],
                  [[k, f"{v:,}", f"{dl['primary_description_language'].get(k, 0):,}"]
                   for k, v in dl["works_with_a_description_in"].items()])
    out += ["", "Descriptions per work: " + ", ".join(f"{k}: {v:,}" for k, v in dl["descriptions_per_work"].items()), ""]
    out += ["Descriptions by source: " + ", ".join(f"{k} {v:,}" for k, v in dl["descriptions_by_source"].items()), ""]

    ln = report["primary_description_length"]
    out += ["## Primary description length (characters)", "",
            f"min {ln['min']}, p25 {ln['p25']}, median {ln['median']}, p75 {ln['p75']}, max {ln['max']}, mean {ln['mean']}", ""]
    out += _table(["Length", "Works"], [[k, f"{v:,}"] for k, v in ln["buckets"].items()])

    cv = report["coverage"]
    out += ["", "## Field coverage", ""]
    out += _table(["Field", "Share of works"], [[k, _pct(v)] for k, v in cv.items()])
    yr = report["years"]
    out += ["", f"Publication years: min {yr['min']}, median {yr['median']}, max {yr['max']}.", ""]

    out += ["## Top subjects", ""]
    out += _table(["Subject", "Works"], [[k, f"{v:,}"] for k, v in report["top_subjects"].items()])

    out += ["", "## Licences and attribution", ""]
    for s in report["sources"]:
        lic = s.get("license") or {}
        attribution = str(lic.get("attribution", "?")).rstrip(".")
        out.append(f"- **{lic.get('name', s['source'])}** ({s['source']}): {lic.get('license', '?')}. "
                   f"Terms: {lic.get('terms_url', '?')}. Attribution: {attribution}. {lic.get('notes', '')}")
    out += ["", "Descriptions by licence: " + ", ".join(f"{k} {v:,}" for k, v in dl["descriptions_by_license"].items()), ""]
    return "\n".join(out) + "\n"
