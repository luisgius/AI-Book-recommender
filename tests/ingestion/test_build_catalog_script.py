"""End-to-end test of scripts/build_catalog.py on tiny offline fixtures."""

import csv
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import build_catalog

SUMMARY = (
    "\"Don Quijote\" by Miguel de Cervantes is a novel about a man who reads so many books of chivalry "
    "that he decides to become a knight-errant himself, riding out with his squire Sancho Panza in search "
    "of adventures across La Mancha. (This is an automatically generated summary.)"
)
ABSTRACT = (
    "Este libro estudia la recepción del Quijote en la cultura española del siglo veinte, desde la "
    "crítica literaria hasta el cine y la televisión, y muestra cómo la novela de Cervantes se convirtió "
    "en un símbolo nacional y en un objeto de debate político y estético."
)
RDF = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns:pgterms="http://www.gutenberg.org/2009/pgterms/" xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcam="http://purl.org/dc/dcam/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <pgterms:ebook rdf:about="ebooks/2000">
    <pgterms:downloads>1500</pgterms:downloads>
    <dcterms:creator><pgterms:agent><pgterms:name>Cervantes Saavedra, Miguel de, 1547-1616</pgterms:name></pgterms:agent></dcterms:creator>
    <dcterms:title>Don Quijote</dcterms:title>
    <pgterms:marc520>{summary}</pgterms:marc520>
    <dcterms:language><rdf:Description><rdf:value>es</rdf:value></rdf:Description></dcterms:language>
    <dcterms:type><rdf:Description><rdf:value>Text</rdf:value></rdf:Description></dcterms:type>
  </pgterms:ebook>
</rdf:RDF>"""


@pytest.fixture
def downloads(tmp_path: Path) -> Path:
    root = tmp_path / "downloads"
    (root / "gutenberg").mkdir(parents=True)
    data = RDF.format(summary=SUMMARY).encode("utf-8")
    with tarfile.open(root / "gutenberg" / "rdf-files.tar.bz2", "w:bz2") as archive:
        info = tarfile.TarInfo("cache/epub/2000/pg2000.rdf")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    (root / "doab").mkdir()
    with open(root / "doab" / "doab_export.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "dc.type", "dc.title", "dc.contributor.author", "dc.date.issued", "dc.language",
                         "dc.description.abstract", "dc.identifier.uri"])
        writer.writerow(["u1", "book", "El Quijote en el siglo XX", "Pérez, Ana", "2015", "Spanish[spa]",
                         ABSTRACT, "https://directory.doabooks.org/handle/1"])
    return root


def test_ingest_report_and_select(tmp_path, downloads):
    out = tmp_path / "catalog_v2" / "catalog.db"
    common = ["--downloads", str(downloads), "--out", str(out)]
    assert build_catalog.main(["ingest", "--sources", "gutenberg", "doab", *common]) == 0
    assert build_catalog.main(["ingest", "--sources", "gutenberg", "doab", *common]) == 0  # finished: skipped
    assert build_catalog.main(["report", *common]) == 0

    report = json.loads((out.parent / "catalog_report.json").read_text())
    assert report["total_works"] == 2
    assert [s["source"] for s in report["sources"]] == ["gutenberg", "doab"]
    assert (out.parent / "CATALOG_REPORT.md").read_text().startswith("# Multi-source catalog report")

    small = tmp_path / "catalog_small" / "catalog.db"
    assert build_catalog.main(["select", "--size", "1", "--from", str(out), "--out", str(small)]) == 0
    selection = json.loads((small.parent / "selection.json").read_text())
    assert selection["size"] == 1 and selection["language_mix"] == {"es": 1}

    # V17: the licence records travel with the selection, so its report lists them
    assert build_catalog.main(["report", "--out", str(small)]) == 0
    small_report = json.loads((small.parent / "catalog_report.json").read_text())
    assert [s["source"] for s in small_report["sources"]] == ["gutenberg", "doab"]
    assert all(s["license"] for s in small_report["sources"])
    assert small_report["selection"]["size"] == 1
    markdown = (small.parent / "CATALOG_REPORT.md").read_text()
    assert "This catalog is a selection of 1 works" in markdown and "## Licences and attribution" in markdown


def test_reading_build_sources_of_a_catalog_without_them(tmp_path):
    import sqlite3

    plain = tmp_path / "plain.db"
    sqlite3.connect(plain).execute("CREATE TABLE books (id TEXT)").connection.close()

    assert build_catalog.read_build_sources(plain) == []


def test_refuses_to_write_the_original_catalog(downloads):
    with pytest.raises(SystemExit, match="original catalog"):
        build_catalog.main(["ingest", "--sources", "doab", "--downloads", str(downloads),
                            "--out", str(build_catalog.PROTECTED_DB)])
