"""Tests of the manual review sheet of the LLM-generated queries and requests: no network, temporary paths only."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from app.evaluation import query_review as QR
from app.evaluation.assistant_evaluation.sets import Request, load_requests
from app.evaluation.catalog_evaluation.queries import EvalQuery, load_queries

BOM = b"\xef\xbb\xbf"


def _write_sets(root: Path) -> Tuple[Path, List[Path]]:
    """A tiny catalog query file and two assistant measurements; returns (queries path, assistant dirs)."""
    queries = [
        EvalQuery("a1", "a1", "original query", "en", "original", "A", "original query"),
        EvalQuery("b-es-1", "b-es-1", "novelas de misterio", "es", "thematic", "B", "Mystery novels."),
        EvalQuery("b-es-6", "b-es-6", "novela de un hidalgo loco", "es", "known_item", "B", "Don Quixote.",
                  target_title="Don Quijote", target_author="Cervantes"),
        EvalQuery("c01-en", "c01", "space novels", "en", "thematic", "C", "Space novels."),
        EvalQuery("c01-es", "c01", "novelas espaciales", "es", "thematic", "C", "Space novels."),
        EvalQuery("d-en", "d-en", "train times", "en", "unanswerable", "D", "Train timetable."),
    ]
    queries_path = root / "queries.json"
    queries_path.write_text(json.dumps({"queries": [q.to_dict() for q in queries]}), encoding="utf-8")
    e = Request("e-b-es-1", "E", "misterio", "es", "b-es-1", "Mystery novels.", "thematic",
                source_query="novelas de misterio")
    f = Request("f01", "F", "Compare X and Y", "en", "f01", "Compare X and Y", "compare", works=["w1", "w2"])
    g = Request("g01", "G", "pan de masa madre", "es", "g01", "sourdough bread baking", "valid",
                notes=["cooking", "keywords", "sourdough bread"])
    oos = Request("g-oos01", "G", "2+2?", "en", "g-oos01", "a sum", "out_of_scope", notes=["calculation", "any", "2+2?"])
    first = [Request("a1", "A", "original query", "en", "a1", "original query"), e, f]
    checks = {"f_tasks": [{"id": "f01", "works": [{"work_id": "w1", "title": "X", "authors": ["Ann"]},
                                                  {"work_id": "w2", "title": "Y", "authors": []}]}]}
    dirs = []
    for name, requests in (("v22", first), ("v25", first + [g, oos])):
        (root / name / "sets").mkdir(parents=True)
        (root / name / "sets" / "requests.json").write_text(
            json.dumps({"requests": [r.to_dict() for r in requests]}), encoding="utf-8")
        (root / name / "sets" / "checks.json").write_text(json.dumps(checks), encoding="utf-8")
        dirs.append(root / name)
    return queries_path, dirs


def _items(tmp_path: Path) -> List[Dict[str, str]]:
    queries_path, dirs = _write_sets(tmp_path)
    return QR.review_items(queries_path, dirs)


def _write_filled(path: Path, items, verdicts: Dict[str, tuple], delimiter: str = ",", bom: bool = False) -> Path:
    """A filled copy of the sheet: verdicts maps item_id to (verdict, comment)."""
    rows = []
    for item in items:
        verdict, comment = verdicts.get(item["item_id"], ("", ""))
        rows.append({**item, "verdict": verdict, "comment": comment})
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QR.SHEET_COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ------------------------------------------------------------------ the sheet


def test_sheet_from_published_files_is_complete_and_deterministic(tmp_path):
    items = QR.review_items()
    assert QR.counts(items) == {"catalogo": {"B": 72, "C": 40, "D": 12}, "asistente": {"E": 30, "F": 20, "G": 80}}
    expected = {q.query_id for q in load_queries(QR.CATALOG_QUERIES) if q.stratum in ("B", "C", "D")}
    for directory in QR.ASSISTANT_DIRS:
        expected |= {r.request_id for r in load_requests(directory / "sets" / "requests.json")
                     if r.set in ("E", "F", "G")}
    ids = [item["item_id"] for item in items]
    assert len(ids) == len(expected) and set(ids) == expected  # every generated item, once
    order = [(QR.EVALUATIONS.index(item["evaluation"]), item["stratum"], item["item_id"]) for item in items]
    assert order == sorted(order)
    assert all(item["text"] and item["need"] and item["language"] for item in items)
    assert all(item["context"] for item in items if item["stratum"] in ("E", "F", "G"))

    QR.build_sheet(items, tmp_path / "one")
    QR.build_sheet(QR.review_items(), tmp_path / "two")
    sheet = (tmp_path / "one" / QR.SHEET_FILE).read_bytes()
    assert sheet == (tmp_path / "two" / QR.SHEET_FILE).read_bytes()
    assert sheet.startswith(BOM)  # Excel needs it to show the non-Latin scripts
    with (tmp_path / "one" / QR.SHEET_FILE).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == QR.SHEET_COLUMNS
    assert rows == items and all(row["verdict"] == row["comment"] == "" for row in rows)
    instructions = (tmp_path / "one" / QR.INSTRUCTIONS_FILE).read_text(encoding="utf-8")
    assert "254 en total" in instructions and "query_review summary --sheet" in instructions


def test_rows_order_context_and_requests_listed_once(tmp_path):
    items = _items(tmp_path)
    assert [i["item_id"] for i in items] == ["b-es-1", "b-es-6", "c01-en", "c01-es", "d-en",
                                             "e-b-es-1", "f01", "g-oos01", "g01"]
    by_id = {i["item_id"]: i for i in items}
    assert by_id["b-es-6"]["context"] == "obra: Don Quijote (Cervantes)"
    assert by_id["e-b-es-1"]["context"] == "consulta de B b-es-1: novelas de misterio"
    assert by_id["f01"]["context"] == "obras: X (Ann); Y ()"
    assert by_id["g01"]["context"] == "campo: cooking; forma: keywords; traducción del generador: sourdough bread"
    assert by_id["g-oos01"]["context"] == "tipo: calculation; traducción del generador: 2+2?"


def test_a_request_that_differs_between_measurements_is_an_error(tmp_path):
    queries_path, dirs = _write_sets(tmp_path)
    path = dirs[1] / "sets" / "requests.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["requests"][1]["text"] = "otra cosa"  # e-b-es-1 in the second measurement
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(QR.SheetError, match="e-b-es-1"):
        QR.review_items(queries_path, dirs)


def test_a_sheet_with_verdicts_is_never_overwritten(tmp_path):
    items = _items(tmp_path)
    QR.build_sheet(items, tmp_path / "out")
    QR.build_sheet(items, tmp_path / "out")  # still empty: rebuilding is fine
    _write_filled(tmp_path / "out" / QR.SHEET_FILE, items, {"f01": ("ok", "")})
    with pytest.raises(QR.SheetError, match="already holds verdicts"):
        QR.build_sheet(items, tmp_path / "out")


# ------------------------------------------------------------------ the summary


def test_summary_counts_and_flagged_items(tmp_path):
    items = _items(tmp_path)
    sheet = _write_filled(tmp_path / "filled.csv", items, {
        "b-es-1": ("ok", ""), "b-es-6": (" Mal ", "names the title"), "c01-en": ("dudosa", "too formal"),
        "e-b-es-1": ("OK", ""), "g01": ("mal", "not about bread")})
    summary = QR.summarize(items, QR.read_sheet(sheet, items))
    assert summary["total"] == {"items": 9, "reviewed": 5, "ok": 2, "dudosa": 1, "mal": 2}
    catalog = summary["by_evaluation"]["catalogo"]
    assert catalog["B"] == {"items": 2, "reviewed": 2, "ok": 1, "dudosa": 0, "mal": 1,
                            "flagged": [{"item_id": "b-es-6", "verdict": "mal", "comment": "names the title"}]}
    assert catalog["C"]["flagged"] == [{"item_id": "c01-en", "verdict": "dudosa", "comment": "too formal"}]
    assert catalog["D"] == {"items": 1, "reviewed": 0, "ok": 0, "dudosa": 0, "mal": 0, "flagged": []}
    assert summary["by_evaluation"]["asistente"]["G"]["mal"] == 1


def test_a_sheet_with_a_byte_order_mark_or_semicolons_reads_the_same(tmp_path):
    items = _items(tmp_path)
    verdicts = {"d-en": ("ok", "fine"), "f01": ("dudosa", "one work only; unclear")}
    plain = QR.read_sheet(_write_filled(tmp_path / "plain.csv", items, verdicts), items)
    assert QR.read_sheet(_write_filled(tmp_path / "bom.csv", items, verdicts, bom=True), items) == plain
    assert QR.read_sheet(_write_filled(tmp_path / "semi.csv", items, verdicts, delimiter=";", bom=True), items) == plain
    assert plain["f01"] == {"verdict": "dudosa", "comment": "one work only; unclear"}


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows[0].update(verdict="bien"), "unknown verdict"),
    (lambda rows: rows[0].update(item_id="b-es-9"), "unknown or changed: \\['b-es-9'\\]"),
    (lambda rows: rows.pop(), "missing: \\['g01'\\]"),
    (lambda rows: rows.append(dict(rows[0])), "repeated: \\['b-es-1'\\]"),
])
def test_a_sheet_with_changed_ids_or_unknown_verdicts_is_rejected(tmp_path, change, message):
    items = _items(tmp_path)
    rows = [dict(item) for item in items]
    change(rows)
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QR.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(QR.SheetError, match=message):
        QR.read_sheet(path, items)


def test_a_missing_column_file_or_encoding_is_rejected(tmp_path):
    items = _items(tmp_path)
    no_verdict = tmp_path / "no_verdict.csv"
    no_verdict.write_text("item_id,comment\nb-es-1,\n", encoding="utf-8")
    with pytest.raises(QR.SheetError, match="no column \\['verdict'\\]"):
        QR.read_sheet(no_verdict, items)
    with pytest.raises(QR.SheetError, match="does not exist"):
        QR.read_sheet(tmp_path / "absent.csv", items)
    latin1 = tmp_path / "latin1.csv"
    latin1.write_bytes("item_id,verdict,comment\nb-es-1,mal,traducción\n".encode("latin-1"))
    with pytest.raises(QR.SheetError, match="CSV UTF-8"):
        QR.read_sheet(latin1, items)


# ------------------------------------------------------------------ command line


def test_command_line_builds_and_summarizes_in_the_given_paths(tmp_path):
    assert QR.build_parser().parse_args(["summary"]).output == QR.OUT_DIR / QR.SUMMARY_FILE
    assert QR.main(["sheet", "--out-dir", str(tmp_path)]) == 0
    sheet = tmp_path / QR.SHEET_FILE
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        items = list(csv.DictReader(handle))
    first = items[0]["item_id"]
    _write_filled(sheet, items, {first: ("mal", "wrong language")}, bom=True)
    output = tmp_path / "summary.json"
    assert QR.main(["summary", "--sheet", str(sheet), "--output", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["total"] == {"items": 254, "reviewed": 1, "ok": 0, "dudosa": 0, "mal": 1}
    assert document["by_evaluation"]["catalogo"]["B"]["flagged"] == [
        {"item_id": first, "verdict": "mal", "comment": "wrong language"}]
    _write_filled(sheet, items, {first: ("maybe", "")})
    with pytest.raises(SystemExit, match="error: unknown verdict"):
        QR.main(["summary", "--sheet", str(sheet), "--output", str(output)])
