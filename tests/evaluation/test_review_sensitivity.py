"""
Tests of the sensitivity analysis of the manual review (V27): the published
figures are reproduced exactly with no exclusion, an exclusion changes the
counts it should, and unknown ids are rejected. They read the published
evaluation files and write only to temporary paths; no network, no LLM, no
database.
"""

import csv
import json

import pytest

from app.evaluation import query_review as QR
from app.evaluation import review_sensitivity as RS
from app.evaluation.assistant_evaluation import design as AD
from app.evaluation.assistant_evaluation.results import build_results as build_assistant_results
from app.evaluation.catalog_evaluation import design as CD
from tests.evaluation.test_query_review import _items, _write_filled

# ------------------------------------------------------------------ reproduction


def test_no_exclusion_reproduces_the_published_figures_exactly():
    """The proof that the recomputation uses the published method: same numbers as each results.json."""
    assert RS.catalog_figures(CD.DATA_DIR) == RS.published_catalog(CD.DATA_DIR)
    for data_dir in (AD.DATA_DIR, AD.V25_DIR):
        assert RS.assistant_figures(data_dir) == RS.published_assistant(data_dir)


def test_a_figure_that_is_not_reproduced_stops_the_analysis():
    published = {"metrics": {"S1": {"mean": 0.5, "ci": [0.4, 0.6]}}, "conclusions": {"H1": "sustained"}}
    assert RS.first_difference(published, json.loads(json.dumps(published))) is None
    recomputed = {"metrics": {"S1": {"mean": 0.5, "ci": [0.4, 0.61]}}, "conclusions": {"H1": "sustained"}}
    assert RS.first_difference(published, recomputed) == "metrics.S1.ci[1]"
    assert RS.first_difference(published, {"metrics": published["metrics"]}) == "conclusions"
    with pytest.raises(RS.SensitivityError, match="does not reproduce the published figure `metrics.S1.ci\\[1\\]`"):
        RS.check_reproduction("catalog", published, recomputed)


def test_changed_figures_lists_every_value_that_differs_but_not_per_request_rows():
    published = {"m": {"n": 30, "mean": 0.3, "ci": [0.2, 0.4], "rows": [{"id": "e1"}], "by": {"x": 1.0}},
                 "other": {"x": 1}}
    without = {"m": {"n": 23, "mean": 0.3, "ci": [0.2, 0.5], "rows": [], "by": {}}, "other": {"x": 2}}
    compared, changed = RS.changed_figures(published, without, ["m"])
    assert compared == 4  # n, mean, ci, by.x: the rows are a listing and "other" is not a compared section
    assert changed == [{"figure": "m.n", "published": 30, "without": 23},
                       {"figure": "m.ci", "published": [0.2, 0.4], "without": [0.2, 0.5]},
                       {"figure": "m.by.x", "published": 1.0, "without": None}]  # a group left without items


# ------------------------------------------------------------------ the exclusion set


def test_flagged_items_are_the_dudosa_and_mal_rows_of_the_sheet(tmp_path):
    items = _items(tmp_path)
    sheet = _write_filled(tmp_path / "filled.csv", items, {
        "b-es-1": ("ok", ""), "c01-en": ("dudosa", "too formal"), "g01": (" Mal ", "not about bread")})
    flagged = RS.flagged_items(sheet, items)
    assert [(x["item_id"], x["verdict"], x["comment"]) for x in flagged] == [
        ("c01-en", "dudosa", "too formal"), ("g01", "mal", "not about bread")]


def test_unknown_ids_in_the_exclusion_set_are_rejected():
    with pytest.raises(RS.SensitivityError, match="unknown item ids in the exclusion set: \\['no-such-item'\\]"):
        RS.sensitivity(["b-zh-6", "no-such-item"])
    with pytest.raises(ValueError, match="does not have: \\['g18'\\]"):  # G exists only in the re-measurement
        build_assistant_results(AD.DATA_DIR, ["g18"])


# ------------------------------------------------------------------ an exclusion, end to end


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The command on a filled copy of the published sheet with five flagged items, written to a temporary dir."""
    root = tmp_path_factory.mktemp("sensitivity")
    flagged = {"b-zh-6": ("mal", "x"), "c09-fr": ("dudosa", "y"), "e-b-ca-1": ("dudosa", "z"),
               "f03": ("dudosa", "w"), "g18": ("dudosa", "v")}
    rows = [{**item, "verdict": flagged.get(item["item_id"], ("ok", ""))[0],
             "comment": flagged.get(item["item_id"], ("ok", ""))[1]} for item in QR.review_items()]
    sheet = root / "sheet.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QR.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    out_dir = root / "out"
    assert QR.main(["sensitivity", "--sheet", str(sheet), "--out-dir", str(out_dir)]) == 0
    return json.loads((out_dir / RS.JSON_FILE).read_text(encoding="utf-8")), (out_dir / RS.MD_FILE).read_text(
        encoding="utf-8")


def test_an_exclusion_changes_the_counts_as_expected(run):
    document, _ = run
    assert [x["item_id"] for x in document["excluded"]] == ["b-zh-6", "c09-fr", "e-b-ca-1", "f03", "g18"]
    assert document["in_both_evaluations"] == ["b-zh-6"]
    catalog = document["evaluations"][RS.CATALOG]
    assert catalog["excluded"] == {"b-zh-6": "B", "c09-fr": "C"}
    assert catalog["items"]["ab"] == [116, 115]
    for judge in RS.JUDGES:
        assert {row["queries"] for row in catalog["published"]["metrics"][judge]["summary_ab"].values()} == {116}
        assert {row["queries"] for row in catalog["without"]["metrics"][judge]["summary_ab"].values()} == {115}
        assert {row["queries"] for row in catalog["without"]["metrics"][judge]["contrasts"].values()} == {115}
    before, after = catalog["items"]["stratum_c"]
    assert (before["needs"], before["pairs"], before["versions_by_language"]["fr"]) == (10, 60, 10)
    assert (after["needs"], after["pairs"], after["versions_by_language"]["fr"]) == (10, 57, 9)

    for name in (RS.V22, RS.V25):
        e = document["evaluations"][name]
        assert e["excluded"] == ({"b-zh-6": "B", "e-b-ca-1": "E", "f03": "F"}
                                 | ({"g18": "G"} if name == RS.V25 else {}))
        for judge in RS.JUDGES:
            n = {h: (c["published"]["requests"], c["without"]["requests"]) for h, c in e["contrasts"][judge].items()}
            assert n == {"H7": (116, 115), "H8": (30, 29), "H9": (20, 19)}
        b = next(d for d in e["decisions"] if "questions on B" in d["decision"])
        assert b["published_detail"].startswith("1/72" if name == RS.V22 else "0/72")
        assert b["without_detail"].startswith("1/71" if name == RS.V22 else "0/71")
        agent = next(f for f in e["figures"] if f["figure"].startswith("Direct agent") and "primary" in f["figure"])
        assert agent["items"][1] == agent["items"][0] - 1  # b-zh-6 is in the direct-agent sample
    g = next(d for d in document["evaluations"][RS.V25]["decisions"] if "valid G requests" in d["decision"])
    assert g["published_detail"].startswith("2/60") and g["without_detail"].startswith("2/59")


def test_the_report_says_what_the_analysis_is_and_how_to_regenerate_it(run):
    document, markdown = run
    assert document["decided_after_results"] is True and document["declared_in_protocols"] is False
    first = markdown.split("## Excluded items")[0]
    assert markdown.startswith("# Sensitivity of the evaluations to the doubtful generated queries (V27)")
    assert "Decided after the results" in first and "does not replace or modify any published result" in first
    assert RS.COMMAND in first
    assert all(f"`{i}`" in markdown for i in ("b-zh-6", "c09-fr", "e-b-ca-1", "f03", "g18"))
    assert "## Figures not recomputed" in markdown
