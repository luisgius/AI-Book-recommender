"""
Offline parsing tests of the bulk sources (Wikidata/Wikipedia, Gutenberg,
DOAB, Open Library dump), each from small fixture files written to tmp_path.
"""

import csv
import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, List

import httpx
import pytest

from app.infrastructure.external.bulk_download import PoliteHttpClient
from app.infrastructure.external.doab_source import DoabSource, classification_labels, parse_languages
from app.infrastructure.external.gutenberg_source import GutenbergSource, parse_gutenberg_rdf
from app.infrastructure.external.openlibrary_dump_source import OpenLibraryDumpSource
from app.infrastructure.external.wikidata_wikipedia_source import (
    WikidataWikipediaSource,
    build_queries,
    fetch_intro_extracts,
)

LONG_ES = (
    "Don Quijote de la Mancha es una novela escrita por el español Miguel de Cervantes Saavedra. "
    "Publicada su primera parte con el título de El ingenioso hidalgo don Quijote de la Mancha a "
    "comienzos de 1605, es la obra más destacada de la literatura española y una de las principales "
    "de la literatura universal."
)
LONG_EN = (
    "Don Quixote is a Spanish novel by Miguel de Cervantes. It follows the adventures of a hidalgo "
    "who reads so many chivalric romances that he loses his mind and decides to become a knight-errant "
    "to revive chivalry and serve his nation."
)
WD = "http://www.wikidata.org/entity/"


def write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# ------------------------------------------------------------------ wikidata

def make_wikidata_fixture(root: Path) -> None:
    sparql = root / "sparql"
    write_csv(sparql / "articles.csv", ["item", "wiki", "title"], [
        [WD + "Q480", "https://en.wikipedia.org/", "Don Quixote"],
        [WD + "Q480", "https://es.wikipedia.org/", "Don Quijote de la Mancha"],
        [WD + "Q1", "https://es.wikipedia.org/", "Lazarillo de Tormes"],
        [WD + "Q2", "https://pl.wikipedia.org/", "Ignored language"],
    ])
    write_csv(sparql / "wikipedia_editions.csv", ["item", "editions"], [[WD + "Q480", 125], [WD + "Q1", 40]])
    write_csv(sparql / "labels.csv", ["item", "lang", "label"], [
        [WD + "Q480", "es", "Don Quijote de la Mancha"], [WD + "Q480", "en", "Don Quixote"],
        [WD + "Q480", "mul", "Don Quijote"], [WD + "Q1", "es", "Lazarillo de Tormes"],
    ])
    write_csv(sparql / "authors.csv", ["item", "author", "name"], [[WD + "Q480", WD + "Q5682", "Miguel de Cervantes"]])
    write_csv(sparql / "creators.csv", ["item", "author", "name"], [])
    write_csv(sparql / "genres.csv", ["item", "genre", "name"], [[WD + "Q480", WD + "Q8261", "novel"]])
    write_csv(sparql / "dates.csv", ["item", "time", "precision"], [
        [WD + "Q480", "1615-01-01T00:00:00Z", 9], [WD + "Q480", "1605-01-01T00:00:00Z", 9],
        [WD + "Q1", "1554-01-01T00:00:00Z", 7],  # century precision: ignored
    ])
    write_csv(sparql / "languages.csv", ["item", "code"], [[WD + "Q480", "es"], [WD + "Q1", "es"]])
    write_csv(sparql / "id_openlibrary.csv", ["item", "value"], [[WD + "Q480", "OL455567W"], [WD + "Q480", "OL1M"]])
    write_csv(sparql / "id_gutenberg.csv", ["item", "value"], [[WD + "Q480", "2000"], [WD + "Q480", "996"]])
    write_csv(sparql / "id_isbn13.csv", ["item", "value"], [])
    write_csv(sparql / "id_doi.csv", ["item", "value"], [])
    write_csv(root / "dbpedia" / "abstracts_0000.csv", ["item", "resource", "lang", "text"], [
        [WD + "Q480", "http://dbpedia.org/resource/Don_Quixote", "en", LONG_EN],
        [WD + "Q480", "http://dbpedia.org/resource/Don_Quixote", "es", "Texto antiguo de DBpedia."],
    ])
    (root / "extracts").mkdir(parents=True, exist_ok=True)
    with open(root / "extracts" / "es.jsonl", "w", encoding="utf-8") as handle:
        for title, text in [("Don Quijote de la Mancha", LONG_ES), ("Lazarillo de Tormes", "La vida de Lazarillo " * 12)]:
            handle.write(json.dumps({"title": title, "page_title": title, "pageid": 1, "extract": text}) + "\n")
        handle.write('{"title": "cut by an interrupted wr')  # partial last line is skipped


class TestWikidataWikipediaSource:
    def test_joins_sparql_results_dbpedia_and_extracts(self, tmp_path):
        make_wikidata_fixture(tmp_path)
        books = {b.source_id: b for b in WikidataWikipediaSource(tmp_path).iter_books()}
        quixote = books["Q480"]
        assert quixote.title == "Don Quijote de la Mancha"  # label in the work's language
        assert quixote.authors == ["Miguel de Cervantes"]
        assert quixote.get_published_year() == 1605
        assert quixote.language == "es" and quixote.categories == ["novel"]
        meta = quixote.metadata
        assert meta.wikidata_id == "Q480" and meta.wikipedia_editions == 125
        assert meta.openlibrary_ids == ("OL455567W",)  # edition ids (M) are dropped
        assert meta.gutenberg_ids == ("2000", "996")
        by_lang = {d.language: d for d in meta.descriptions}
        assert by_lang["es"].text == LONG_ES and by_lang["es"].license == "CC BY-SA 4.0"  # API wins
        assert by_lang["en"].text == LONG_EN and by_lang["en"].license == "CC BY-SA 3.0"  # DBpedia
        assert by_lang["en"].attribution_url == "https://en.wikipedia.org/wiki/Don_Quixote"
        assert {r.source for r in meta.provenance} == {"wikidata", "wikipedia"}

    def test_work_without_author_gets_unknown_and_no_coarse_year(self, tmp_path):
        make_wikidata_fixture(tmp_path)
        books = {b.source_id: b for b in WikidataWikipediaSource(tmp_path).iter_books()}
        assert books["Q1"].authors == ["Unknown"]
        assert books["Q1"].published_date is None

    def test_limit_keeps_most_edited_works_and_unconfigured_wikis_are_ignored(self, tmp_path):
        make_wikidata_fixture(tmp_path)
        books = list(WikidataWikipediaSource(tmp_path).iter_books(limit=1))
        assert [b.source_id for b in books] == ["Q480"]
        assert "Q2" not in {b.source_id for b in WikidataWikipediaSource(tmp_path).iter_books()}

    def test_extract_plan_rescues_undescribed_works_first(self, tmp_path):
        make_wikidata_fixture(tmp_path)
        source = WikidataWikipediaSource(tmp_path)
        works = sorted(source._load_works().items())
        abstracts = {"Q480": {"en": LONG_EN}}
        rescue, enrich = source.plan_extract_phases(works, abstracts, ["en", "es"])
        assert rescue["es"] == ["Lazarillo de Tormes"]  # no description at all: its own language
        assert enrich["es"] == ["Don Quijote de la Mancha"]  # described in English already
        assert rescue["en"] == [] and enrich["en"] == []

    def test_extract_plan_prefers_english_for_works_of_unknown_language(self, tmp_path):
        make_wikidata_fixture(tmp_path)
        with open(tmp_path / "sparql" / "articles.csv", "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([WD + "Q7", "https://es.wikipedia.org/", "Ciclo de la Puerta de la Muerte"])
            writer.writerow([WD + "Q7", "https://en.wikipedia.org/", "The Death Gate Cycle"])
        source = WikidataWikipediaSource(tmp_path)
        works = sorted(source._load_works().items())
        rescue, enrich = source.plan_extract_phases(works, {}, ["en", "es"])
        assert "The Death Gate Cycle" in rescue["en"]
        assert "Ciclo de la Puerta de la Muerte" in enrich["es"]

    def test_queries_are_built_for_every_cached_file(self):
        queries = build_queries(["en", "es"])
        assert {"articles", "labels", "authors", "creators", "dates", "id_gutenberg"} <= set(queries)
        assert "<https://es.wikipedia.org/>" in queries["articles"]

    def test_fetch_intro_extracts_follows_redirects_and_continuation(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(dict(request.url.params))
            if "excontinue" not in request.url.params:
                body = {
                    "continue": {"excontinue": "1", "continue": "||"},
                    "query": {
                        "normalized": [{"from": "don quixote", "to": "Don quixote"}],
                        "redirects": [{"from": "Don quixote", "to": "Don Quixote"}],
                        "pages": [{"pageid": 1, "title": "Don Quixote", "extract": "Intro A"},
                                  {"pageid": 2, "title": "Hamlet"}],
                    },
                }
            else:
                body = {"query": {"pages": [{"pageid": 2, "title": "Hamlet", "extract": "Intro B"},
                                            {"title": "Missing page", "missing": True}]}}
            return httpx.Response(200, json=body)

        client = PoliteHttpClient(rate_per_second=1000, client=httpx.Client(transport=httpx.MockTransport(handler)))
        records = fetch_intro_extracts(client, "en", ["don quixote", "Hamlet", "Missing page"])
        assert [r["extract"] for r in records] == ["Intro A", "Intro B", None]
        assert calls[0]["maxlag"] == "5" and calls[0]["exintro"] == "1"

    def test_client_honours_retry_after_on_429(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("app.infrastructure.external.bulk_download.time.sleep", sleeps.append)
        answers = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={})])
        client = PoliteHttpClient(rate_per_second=1000, client=httpx.Client(
            transport=httpx.MockTransport(lambda request: next(answers))))
        assert client.get("https://example.org").status_code == 200
        assert 7.0 in sleeps and client.rate_limit_events == 1


# ----------------------------------------------------------------- gutenberg

RDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xml:base="http://www.gutenberg.org/"
  xmlns:pgterms="http://www.gutenberg.org/2009/pgterms/" xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcam="http://purl.org/dc/dcam/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <pgterms:ebook rdf:about="ebooks/{id}">
    <dcterms:issued>2004-12-01</dcterms:issued>
    <pgterms:downloads>{downloads}</pgterms:downloads>
    <dcterms:creator><pgterms:agent rdf:about="2009/agents/1"><pgterms:name>Cervantes Saavedra, Miguel de</pgterms:name></pgterms:agent></dcterms:creator>
    <dcterms:title>{title}</dcterms:title>
    <pgterms:marc520>{summary} (This is an automatically generated summary.)</pgterms:marc520>
    <dcterms:language><rdf:Description><rdf:value>es</rdf:value></rdf:Description></dcterms:language>
    <dcterms:subject><rdf:Description><dcam:memberOf rdf:resource="http://purl.org/dc/terms/LCSH"/><rdf:value>Knights and knighthood -- Spain -- Fiction</rdf:value></rdf:Description></dcterms:subject>
    <dcterms:subject><rdf:Description><dcam:memberOf rdf:resource="http://purl.org/dc/terms/LCC"/><rdf:value>PQ</rdf:value></rdf:Description></dcterms:subject>
    <pgterms:bookshelf><rdf:Description><rdf:value>Category: Novels</rdf:value></rdf:Description></pgterms:bookshelf>
    <dcterms:type><rdf:Description><rdf:value>{type}</rdf:value></rdf:Description></dcterms:type>
  </pgterms:ebook>
</rdf:RDF>"""


def make_gutenberg_archive(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    records = [
        dict(id=2000, downloads=1500, title="Don Quijote\nedición completa", summary=LONG_EN, type="Text"),
        dict(id=9999, downloads=3, title="A recording", summary="Audio.", type="Sound"),
    ]
    with tarfile.open(root / "rdf-files.tar.bz2", "w:bz2") as archive:
        for record in records:
            data = RDF_TEMPLATE.format(**record).encode("utf-8")
            info = tarfile.TarInfo(f"cache/epub/{record['id']}/pg{record['id']}.rdf")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class TestGutenbergSource:
    def test_parse_rdf_flags_generated_summary(self):
        record = parse_gutenberg_rdf(RDF_TEMPLATE.format(
            id=1, downloads=5, title="T", summary="Summary text.", type="Text").encode())
        assert record["auto_generated"] is True and record["summary"] == "Summary text."
        assert record["subjects"] == ["Knights and knighthood -- Spain -- Fiction"]  # LCC codes left out
        assert record["downloads"] == 5 and record["languages"] == ["es"]

    def test_iter_books_from_archive(self, tmp_path):
        make_gutenberg_archive(tmp_path)
        source = GutenbergSource(tmp_path)
        [book] = list(source.iter_books())
        assert book.title == "Don Quijote: edición completa"
        assert book.authors == ["Cervantes Saavedra, Miguel de"]  # normalised later by the pipeline
        assert book.published_date is None  # 'issued' is the e-text date, not publication
        assert book.metadata.gutenberg_ids == ("2000",) and book.metadata.gutenberg_downloads == 1500
        [description] = book.metadata.descriptions
        assert description.auto_generated and description.language is None
        assert source.get_rejected_counts() == {"not_text": 1}


# ---------------------------------------------------------------------- doab

DOAB_HEADER = [
    "id", "dc.type", "dc.title", "dc.contributor.author", "dc.contributor.editor", "dc.date.issued",
    "dc.language", "dc.description.abstract", "dc.subject.classification", "dc.subject.other",
    "oapen.relation.isbn", "BITSTREAM ISBN", "oapen.identifier.doi", "oapen.relation.isPublishedBy_publisher.name",
    "oapen.pages", "BITSTREAM License", "dc.identifier.uri",
]


class TestDoabSource:
    def make_export(self, root: Path) -> None:
        write_csv(root / "doab_export.csv", DOAB_HEADER, [
            ["u1", "book", "Historia de la lectura", "Pérez, Ana||Ruiz, Luis", "", "2019", "Spanish[spa]",
             LONG_ES, "thema EDItEUR::D Biography, Literature::DS Literature::DSB Literary studies",
             "lectura||Historia del libro", "978-84-376-0494-7||9780306406157", "", "10.1234/ABC",
             "Editorial X", "250", "https://creativecommons.org/licenses/by/4.0/", "https://directory.doabooks.org/handle/1"],
            ["u2", "chapter", "A chapter", "Doe, Jane", "", "2020", "English[eng]", LONG_EN, "", "", "", "", "", "", "", "", ""],
            ["u3", "book", "Edited volume", "", "Smith, Ann", "2021", "Italian[ita]||English[eng]", LONG_EN, "", "", "", "", "", "", "", "", ""],
        ])

    def test_helpers(self):
        assert parse_languages("Italian[ita]||English[eng]") == ["it", "en"]
        assert classification_labels("thema EDItEUR::J Society::JN Education::JNM Higher education") == ["Higher education"]

    def test_iter_books(self, tmp_path):
        self.make_export(tmp_path)
        source = DoabSource(tmp_path)
        books = list(source.iter_books())
        assert [b.source_id for b in books] == ["u1", "u3"]
        first = books[0]
        assert first.authors == ["Pérez, Ana", "Ruiz, Luis"] and first.get_published_year() == 2019
        assert first.language == "es" and first.metadata.isbn13s == ("9788437604947", "9780306406157")
        assert first.metadata.doi == "10.1234/abc" and first.metadata.page_count == 250
        assert first.categories == ["Literary studies", "Historia del libro"]  # one-word lowercase keyword dropped
        assert "by/4.0" in first.metadata.provenance[0].license
        assert books[1].authors == ["Smith, Ann"]  # editors when there is no author
        assert books[1].language is None  # two languages: none is the work's language
        assert source.get_rejected_counts() == {"not_a_book": 1}


# -------------------------------------------------------------- open library

def dump_line(record_type: str, record: Dict) -> str:
    return f"{record_type}\t{record['key']}\t1\t2026-10-01T00:00:00\t{json.dumps(record)}\n"


class TestOpenLibraryDumpSource:
    def make_dumps(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        works = [
            {"key": "/works/OL1W", "title": "Long work", "description": {"type": "/type/text", "value": LONG_EN},
             "authors": [{"author": {"key": "/authors/OL1A"}, "type": {"key": "/type/author_role"}}],
             "subjects": ["Fiction", "Accessible book"], "first_publish_date": "June 1925", "covers": [42]},
            {"key": "/works/OL2W", "title": "Short work", "description": "Too short.",
             "authors": [{"author": {"key": "/authors/OL1A"}}]},
            {"key": "/works/OL3W", "title": "No description"},
            {"key": "/works/OL4W", "title": "Orphan work", "description": LONG_ES,
             "authors": [{"author": {"key": "/authors/OL9A"}}]},
        ]
        with gzip.open(root / "ol_dump_works_2026-10-31.txt.gz", "wt", encoding="utf-8") as handle:
            for work in works:
                handle.write(dump_line("/type/work", work))
        with gzip.open(root / "ol_dump_authors_2026-10-31.txt.gz", "wt", encoding="utf-8") as handle:
            handle.write(dump_line("/type/author", {"key": "/authors/OL1A", "name": "F. Scott Fitzgerald"}))
            handle.write(dump_line("/type/author", {"key": "/authors/OL2A", "name": "Unused"}))

    def test_three_pass_parsing(self, tmp_path):
        self.make_dumps(tmp_path)
        source = OpenLibraryDumpSource(tmp_path)
        [book] = list(source.iter_books())
        assert book.title == "Long work" and book.authors == ["F. Scott Fitzgerald"]
        assert book.get_published_year() == 1925
        assert book.metadata.openlibrary_ids == ("OL1W",)
        assert book.metadata.thumbnail_url.endswith("/42-M.jpg")
        assert book.metadata.descriptions[0].language is None
        assert source.get_rejected_counts() == {"no_or_short_description": 2, "no_author": 1}

    def test_candidates_are_cached_and_counts_restored(self, tmp_path):
        self.make_dumps(tmp_path)
        list(OpenLibraryDumpSource(tmp_path).iter_books())
        (tmp_path / "ol_dump_works_2026-10-31.txt.gz").unlink()  # second run must not need the dump
        source = OpenLibraryDumpSource(tmp_path)
        assert len(list(source.iter_books())) == 1
        assert source.get_rejected_counts()["no_or_short_description"] == 2

    def test_runs_from_the_candidates_cache_when_the_works_dump_is_gone(self, tmp_path, caplog):
        self.make_dumps(tmp_path)
        list(OpenLibraryDumpSource(tmp_path).iter_books())
        (tmp_path / "ol_dump_works_2026-10-31.txt.gz").unlink()
        source = OpenLibraryDumpSource(tmp_path)
        assert source.is_available()
        with caplog.at_level("INFO"):
            assert [b.authors for b in source.iter_books()] == [["F. Scott Fitzgerald"]]
        assert "using the cached candidates" in caplog.text

    def test_without_the_authors_dump_names_cannot_be_resolved(self, tmp_path):
        self.make_dumps(tmp_path)
        list(OpenLibraryDumpSource(tmp_path).iter_books())
        (tmp_path / "ol_dump_works_2026-10-31.txt.gz").unlink()
        (tmp_path / "ol_dump_authors_2026-10-31.txt.gz").unlink()
        source = OpenLibraryDumpSource(tmp_path)
        assert source.is_available()  # the cache exists ...
        with pytest.raises(FileNotFoundError, match="author names cannot be resolved"):
            list(source.iter_books())  # ... but works would have no authors

    def test_not_available_without_dump_or_cache(self, tmp_path):
        assert not OpenLibraryDumpSource(tmp_path).is_available()


# ------------------------------------------------------------------ download

class TestResumableDownload:
    def test_resume_keeps_the_original_segments(self, tmp_path, monkeypatch):
        import app.infrastructure.external.bulk_download as bd

        payload = bytes(range(256)) * 40  # 10,240 bytes
        dest = tmp_path / "dump.gz"
        # A previous run with 4 segments left the first 100 bytes of each part.
        size = len(payload) // 4
        for i in range(4):
            (tmp_path / f"dump.gz.part{i}").write_bytes(payload[i * size:i * size + 100])

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                return httpx.Response(200, headers={"Content-Length": str(len(payload)), "Accept-Ranges": "bytes"})
            start, end = request.headers["Range"].removeprefix("bytes=").split("-")
            return httpx.Response(206, content=payload[int(start):int(end) + 1])

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client
        monkeypatch.setattr(bd.httpx, "Client", lambda *a, **k: real_client(*a, transport=transport, **k))
        bd.download_file("https://example.org/dump.gz", dest, segments=8)
        assert dest.read_bytes() == payload
        assert not list(tmp_path.glob("dump.gz.part*"))
