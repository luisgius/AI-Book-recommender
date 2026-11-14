"""Tests of the order annotation (V31): pure logic and local files only, no network, no database."""

import csv
import json
import math
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from scipy import stats

from app.domain.entities import Book
from app.evaluation.catalog_evaluation import annotation_order as O
from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.cli import build_parser
from app.evaluation.catalog_evaluation.queries import EvalQuery


def q(qid, stratum, language):
    return EvalQuery(qid, qid, f"text {qid}", language, "thematic", stratum, f"need {qid}")


def v14_like_queries():
    """The shape of the V14 set: A 24 en + 20 es, B 6 per language, C and D outside the sample."""
    out = [q(f"a-en-{i:02d}", "A", "en") for i in range(24)] + [q(f"a-es-{i:02d}", "A", "es") for i in range(20)]
    for language in ("en", "es", "fr", "de"):
        out += [q(f"b-{language}-{i}", "B", language) for i in range(6)]
    return out + [q("c01-es", "C", "es"), q("c01-en", "C", "en"), q("d01", "D", "en")]


# ------------------------------------------------------------------ query sample


def test_select_queries_balances_languages_and_takes_stratum_b_first():
    selected, record = O.select_queries(v14_like_queries())
    assert len(selected) == 30 and record["fewer_than_target"] is False
    assert record["selected"] == {"en": {"B": 6, "A": 9}, "es": {"B": 6, "A": 9}}
    assert {x.stratum for x in selected} == {"A", "B"} and {x.language for x in selected} == {"en", "es"}
    assert {x.query_id for x in selected if x.stratum == "B"} == {f"b-{l}-{i}" for l in ("en", "es") for i in range(6)}
    again, _ = O.select_queries(list(reversed(v14_like_queries())))
    assert [x.query_id for x in again] == [x.query_id for x in selected]  # depends on the set and the seed only
    other, _ = O.select_queries(v14_like_queries(), seed=1)
    assert {x.query_id for x in other} != {x.query_id for x in selected}


def test_block_2_is_every_other_eligible_query():
    queries = v14_like_queries()
    block_1, _ = O.select_queries(queries)
    block_2, record = O.remaining_queries(queries, block_1)
    assert len(block_2) == 26 and record["selected"] == {"en": {"B": 0, "A": 15}, "es": {"B": 0, "A": 11}}
    assert not {x.query_id for x in block_1} & {x.query_id for x in block_2}
    assert {x.query_id for x in block_1 + block_2} == {x.query_id for x in O.eligible_queries(queries)}


def test_select_queries_with_few_queries_takes_all_and_reports_the_count():
    queries = [q("a-en-1", "A", "en"), q("b-en-1", "B", "en"), q("b-es-1", "B", "es"), q("b-fr-1", "B", "fr")]
    selected, record = O.select_queries(queries)
    assert len(selected) == 3 and record["fewer_than_target"] is True and record["queries"] == 3
    assert O.remaining_queries(queries, selected)[0] == []


def test_balanced_quota_gives_the_rest_to_the_language_with_queries_left():
    assert O.balanced_quota({"en": 30, "es": 30}, 30) == {"en": 15, "es": 15}
    assert O.balanced_quota({"en": 5, "es": 40}, 30) == {"en": 5, "es": 25}
    assert O.balanced_quota({"en": 31, "es": 30}, 31) == {"en": 16, "es": 15}


def test_select_queries_samples_stratum_b_when_it_exceeds_the_quota():
    queries = [q(f"b-en-{i}", "B", "en") for i in range(20)] + [q(f"b-es-{i}", "B", "es") for i in range(20)]
    selected, record = O.select_queries(queries + [q("a-en-1", "A", "en")], n=10)
    assert record["selected"] == {"en": {"B": 5, "A": 0}, "es": {"B": 5, "A": 0}}
    assert len(selected) == 10


# ------------------------------------------------------------------ sheet and key


def _book(work_id, year=None, language="en"):
    published = date(year, 1, 1) if year else None
    return Book(id=uuid4(), title=f"Title {work_id}", authors=["A One", "B Two"], source="s", source_id=work_id,
                description=f"  description {work_id}  ", language=language, published_date=published)


GOOD = {"f", "g", "s", "t", "u", "v"}  # the works only S3 or S5 bring in _toy()


def _toy():
    """Three queries; the four systems overlap, so each query has fewer than 20 pairs."""
    queries = [q("b-en-1", "B", "en"), q("q02", "A", "es"), q("q03", "A", "en")]
    lists = {
        "b-en-1": {D.S2: ["a", "b", "c", "d", "e"], D.S3: ["b", "f", "a", "g", "c"],
                   D.S4: ["c", "a", "b", "e", "d"], D.S5: ["c", "b", "f", "a", "g"]},
        "q02": {D.S2: ["m", "n", "o", "p", "r"], D.S3: ["s", "m", "n", "t", "o"],
                D.S4: ["n", "m", "p", "o", "r"], D.S5: ["n", "s", "m", "t", "o"]},
        "q03": {D.S2: ["h", "i", "j", "k", "l"], D.S3: ["u", "h", "i", "v", "j"],
                D.S4: ["i", "h", "k", "j", "l"], D.S5: ["i", "u", "h", "v", "j"]},
    }
    books = {w: _book(w, year=2000 if w != "a" else None) for by in lists.values() for top in by.values() for w in top}
    earlier_key = {"p001": {"need_id": "q02", "query_id": "q02", "work_id": "s"},
                   "p002": {"need_id": "q99", "query_id": "q99", "work_id": "a"}}
    return queries, lists, books, earlier_key


def _rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_order_sheet_is_blind_grouped_by_query_and_keyed(tmp_path):
    queries, lists, books, earlier_key = _toy()
    summary = O.build_order_sheet([queries], lists, books, tmp_path, earlier_key)
    rows = _rows(tmp_path / O.SHEET_FILE)
    assert (tmp_path / O.SHEET_FILE).read_bytes().startswith(b"\xef\xbb\xbf")  # opens as UTF-8 in a spreadsheet
    assert tuple(rows[0].keys()) == O.SHEET_COLUMNS
    assert not {"system", "rank", "ranks", "judge"} & set(rows[0].keys())
    assert all(r["grade"] == "" and r["grade_2"] == "" and r["block"] == "1" for r in rows)
    key = json.loads((tmp_path / O.KEY_FILE).read_text())
    # each (query, work) of the union of the four top 5 appears once
    expected = {(qid, w) for qid, by in lists.items() for top in by.values() for w in top}
    seen = [(key[r["pair_id"]]["query_id"], key[r["pair_id"]]["work_id"]) for r in rows]
    assert len(seen) == len(set(seen)) == len(expected) == summary["pairs"] and set(seen) == expected
    # one query at a time: the rows of a query are contiguous
    order = [key[r["pair_id"]]["query_id"] for r in rows]
    assert [x for i, x in enumerate(order) if i == 0 or order[i - 1] != x] == summary["blocks"]["block_1"]["query_order"]
    # the book is shown as in the earlier sheet, plus its language and year
    row = next(r for r in rows if key[r["pair_id"]]["work_id"] == "a")
    assert (row["title"], row["authors"], row["description"]) == ("Title a", "A One; B Two", "description a")
    assert (row["book_language"], row["year"]) == ("en", "")
    # the key rebuilds every system's top 5
    assert O.lists_from_key(key) == lists
    runs = {s: {qid: {"ids": by[s] + ["x"] * 95} for qid, by in lists.items()} for s in O.ORDER_SYSTEMS}
    assert O.check_key_against_runs(key, runs) == []
    runs[D.S5]["q02"]["ids"] = list(reversed(runs[D.S5]["q02"]["ids"]))
    assert O.check_key_against_runs(key, runs) == ["q02"]
    # the overlap with the earlier sheet is in the key only
    assert summary["blocks"]["block_1"]["overlap_with_earlier_sheet"] == {"pairs": 1, "earlier_pair_ids": ["p001"]}
    assert [p for p, v in key.items() if v["earlier_pair_id"]] == [r["pair_id"] for r in rows
                                                                   if key[r["pair_id"]]["work_id"] == "s"]
    text = (tmp_path / O.INSTRUCTIONS_FILE).read_text()
    assert "grade_2" in text and "Perfectamente relevante" in text and O.KEY_FILE in text


def test_adding_block_2_leaves_block_1_unchanged(tmp_path):
    queries, lists, books, earlier_key = _toy()
    one = O.build_order_sheet([queries[:2]], lists, books, tmp_path / "one", earlier_key)
    two = O.build_order_sheet([queries[:2], queries[2:]], lists, books, tmp_path / "two", earlier_key)
    rows_one, rows_two = _rows(tmp_path / "one" / O.SHEET_FILE), _rows(tmp_path / "two" / O.SHEET_FILE)
    n = one["pairs"]
    assert [r["block"] for r in rows_two] == ["1"] * n + ["2"] * (len(rows_two) - n)
    assert rows_two[:n] == rows_one  # block 1 rows: same pairs, same order, same ids
    assert rows_two[n]["pair_id"] == f"r{n + 1:03d}"
    key = json.loads((tmp_path / "two" / O.KEY_FILE).read_text())
    assert {v["query_id"] for v in key.values() if v["block"] == 2} == {"q03"}
    assert two["blocks"]["block_2"]["pairs"] == len(rows_two) - n
    text = (tmp_path / "two" / O.INSTRUCTIONS_FILE).read_text()
    assert "Bloque 2 (opcional)" in text and f"{len(rows_two) - n} filas" in text


def test_order_sheet_is_reproducible(tmp_path):
    queries, lists, books, earlier_key = _toy()
    O.build_order_sheet([queries[:2], queries[2:]], lists, books, tmp_path / "one", earlier_key)
    O.build_order_sheet([list(reversed(queries[:2])), queries[2:]], lists, books, tmp_path / "two", earlier_key)
    for name in (O.SHEET_FILE, O.KEY_FILE):  # the instructions name their own folder
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_top_lists_reads_the_stored_runs_and_refuses_a_missing_line():
    queries = [q("q01", "A", "en")]
    runs = {s: {"q01": {"ids": [f"{s}-{i}" for i in range(100)]}} for s in O.ORDER_SYSTEMS}
    assert O.top_lists(queries, runs)["q01"][D.S4] == [f"S4-{i}" for i in range(5)]
    del runs[D.S3]["q01"]
    with pytest.raises(ValueError):
        O.top_lists(queries, runs)


# ------------------------------------------------------------------ statistics


@pytest.mark.parametrize("x,y", [
    ([4, 3, 2, 1], [4, 3, 2, 1]),
    ([4, 3, 2, 1], [1, 2, 3, 4]),
    ([0.8, 0.6, 0.7, 0.5], [0.9, 0.6, 0.6, 0.1]),
    ([1, 1, 2, 3], [1, 2, 2, 3]),
])
def test_kendall_tau_b_matches_scipy(x, y):
    assert O.kendall_tau_b(x, y) == pytest.approx(stats.kendalltau(x, y).statistic)


def test_kendall_tau_b_of_a_constant_vector_is_undefined():
    assert O.kendall_tau_b([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_minimum_detectable_difference_formula():
    diffs = [0.1, -0.2, 0.3, 0.0, -0.1, 0.25, 0.05, -0.05, 0.15, -0.3]
    out = O.minimum_detectable_difference(diffs)
    sd = stats.tstd(diffs)
    factor = stats.t.ppf(0.975, 9) + stats.t.ppf(0.8, 9)
    assert out["sd"] == pytest.approx(sd)
    assert out["minimum_detectable_difference"] == pytest.approx(factor * sd / math.sqrt(10))


def test_design_power_on_a_subset_of_queries(tmp_path):
    queries, lists, books, earlier_key = _toy()
    O.build_order_sheet([queries[:2], queries[2:]], lists, books, tmp_path, earlier_key)
    key = json.loads((tmp_path / O.KEY_FILE).read_text())
    judges = {"judge/one": _judge(lists)}
    assert O.design_power(key, judges)["queries"] == 3
    block_1 = O.design_power(key, judges, ["b-en-1", "q02"])
    assert block_1["queries"] == 2 and block_1["by_judge"]["judge/one"]["contrasts"]["H1"]["queries"] == 2


# ------------------------------------------------------------------ the declared analysis


def _write(path: Path, rows, bom=True):
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=O.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _judge(lists, flip=False):
    """{need_id: {work_id: grade}}: GOOD works 3 and the rest 1 (S3 beats S2); flip: GOOD 0 and the rest 3."""
    grade = (lambda w: 0 if w in GOOD else 3) if flip else (lambda w: 3 if w in GOOD else 1)
    return {qid: {w: grade(w) for top in by.values() for w in top} for qid, by in lists.items()}


def _filled(tmp_path, blocks, grade_of):
    """Build a sheet with `blocks` and fill `grade` with grade_of(key entry); returns key and rows."""
    queries, lists, books, earlier_key = _toy()
    by_id = {x.query_id: x for x in queries}
    O.build_order_sheet([[by_id[i] for i in b] for b in blocks], lists, books, tmp_path, earlier_key)
    key = json.loads((tmp_path / O.KEY_FILE).read_text())
    rows = _rows(tmp_path / O.SHEET_FILE)
    for row in rows:
        row["grade"] = grade_of(key[row["pair_id"]])
    return lists, key, rows


def test_order_agreement_end_to_end(tmp_path):
    queries, lists, _, _ = _toy()
    judge = _judge(lists)
    judges = {"judge/one": judge, "judge/two": judge}
    truth = lambda pair: str(judge[pair["need_id"]][pair["work_id"]])  # noqa: E731
    lists, key, rows = _filled(tmp_path, [[x.query_id for x in queries]], truth)
    rows[0]["grade"] = ""  # the person copies the judge, leaves one cell empty and writes one invalid cell
    rows[1]["grade"] = "2.5"
    first_query = key[rows[0]["pair_id"]]["query_id"]
    second_rows = [r for r in rows if key[r["pair_id"]]["query_id"] == first_query]
    for row in second_rows:  # a second person grades the whole first query, one row differently
        row["grade_2"] = truth(key[row["pair_id"]])
    second_rows[3]["grade_2"] = "0" if second_rows[3]["grade_2"] != "0" else "3"
    filled = tmp_path / "filled.csv"
    _write(filled, rows)
    earlier = tmp_path / "earlier.csv"
    with earlier.open("w", encoding="utf-8", newline="") as handle:
        handle.write("pair_id,grade\np001,2\np002,1\n")

    out = O.order_agreement(filled, key, judges, earlier)
    cells = out["cells"][O.FIRST]
    assert (cells["graded"], cells["empty"], cells["invalid"]) == (len(rows) - 2, 1, {rows[1]["pair_id"]: "2.5"})
    block = out[O.FIRST]
    # one block only: block 1 is the primary set; its query with empty cells is still used
    assert block["selection"]["primary_set"] == "block 1" and len(block["selection"]["primary"]) == 3
    figures = block["primary"]["figures"]
    # the person and the judges are scored on the same pairs, so equal grades give equal figures
    assert figures[O.FIRST]["ndcg_at_5"] == figures["judge/one"]["ndcg_at_5"]
    assert figures[O.FIRST]["contrasts"]["H1"]["difference"] > 0
    assert block["primary"]["versus_judges"]["judge/one"]["signs"]["H1"]["agree"] is True
    assert block["primary"]["versus_judges"]["judge/one"]["kendall_tau_b"] == pytest.approx(1.0)
    assert block["primary"]["reading_rule"]["H1"] == "confirmed by a person"
    agreement = block["pair_agreement_with_judges"]["judge/one"]
    assert agreement["pairs"] == len(rows) - 2 and agreement["overall"]["exact_agreement"] == 1.0
    # the empty and the invalid pair are unjudged for the judges too
    left_out = {rows[0]["pair_id"], rows[1]["pair_id"]}
    masked = O.by_query(key, {p: g for p, g in O.judge_by_pair(key, judge).items() if p not in left_out})
    unmasked = O.by_query(key, O.judge_by_pair(key, judge))
    query_ids = sorted(lists)
    assert figures["judge/one"]["per_query"] == O.ndcg_table(lists, masked, query_ids)
    assert figures["judge/one"]["per_query"] != O.ndcg_table(lists, unmasked, query_ids)
    # self-agreement on the pair of the earlier sheet (work s of q02, earlier grade 2, new grade 3 if graded)
    new_pair = next(p for p, v in key.items() if v["earlier_pair_id"] == "p001")
    assert out["self_agreement"]["overlap_pairs"] == 1 and out["self_agreement"]["graded_earlier"] == 1
    assert out["self_agreement"]["agreement"]["pairs"] == int(new_pair not in left_out)
    # the second annotator: the whole first query, which the first annotator graded only in part
    second = out[O.SECOND]
    assert out["cells"][O.SECOND]["graded"] == len(second_rows)
    assert second["selection"]["primary"] == [first_query]
    between = second["pair_agreement_between_annotators"]
    assert between["pairs"] == len(second_rows) - 2
    common = second["versus_first_annotator_on_common_queries"]
    assert (common["queries"], common["pairs"]) == (1, len(second_rows) - 2)


def test_block_2_counts_only_when_every_query_is_complete(tmp_path):
    queries, lists, _, _ = _toy()
    judge = _judge(lists)
    judges = {"judge/one": judge, "judge/two": judge}
    truth = lambda pair: str(judge[pair["need_id"]][pair["work_id"]])  # noqa: E731
    blocks = [["b-en-1"], ["q02", "q03"]]

    def run(name, edit):
        lists_, key, rows = _filled(tmp_path / name, blocks, truth)
        edit(key, rows)
        _write(tmp_path / name / "filled.csv", rows)
        return O.order_agreement(tmp_path / name / "filled.csv", key, judges)[O.FIRST]

    def blank(query_ids, how_many=None):
        def edit(key, rows):
            chosen = [r for r in rows if key[r["pair_id"]]["query_id"] in query_ids]
            for row in chosen[:how_many]:
                row["grade"] = ""
        return edit

    # every query of block 2 complete (one empty cell in block 1 is allowed): blocks 1 and 2 are primary
    full = run("full", blank({"b-en-1"}, 1))
    assert full["selection"]["primary_set"] == "blocks 1 and 2" and len(full["selection"]["primary"]) == 3
    assert full["primary"]["queries"] == 3 and full["secondary_block_1"]["queries"] == 1
    assert "reading_rule" not in full["secondary_block_1"] and "block_2_descriptive" not in full
    # q03 graded in part: block 1 is primary, q02 is descriptive, q03 is never used
    part = run("part", blank({"q03"}, 2))
    assert part["selection"]["primary_set"] == "block 1" and part["selection"]["primary"] == ["b-en-1"]
    assert part["selection"]["partial_not_used"] == ["q03"]
    assert part["block_2_descriptive"]["queries"] == 1 and part["selection"]["block_2_descriptive"] == ["q02"]
    assert "secondary_block_1" not in part
    # block 2 not started: block 1 only, nothing descriptive
    none = run("none", blank({"q02", "q03"}))
    assert none["selection"]["primary"] == ["b-en-1"] and none["selection"]["partial_not_used"] == []
    assert "block_2_descriptive" not in none
    # the graded pairs of a partial block 2 query still count for the agreement per pair
    total = sum(len({w for top in by.values() for w in top}) for by in lists.values())
    assert part["pair_agreement_with_judges"]["judge/one"]["pairs"] == total - 2
    assert part["primary"]["pairs"] == len({w for top in lists["b-en-1"].values() for w in top})


def test_reading_rule_flags_a_sign_that_differs(tmp_path):
    queries, lists, _, _ = _toy()
    judge, opposite = _judge(lists), _judge(lists, flip=True)
    lists, key, rows = _filled(tmp_path, [[x.query_id for x in queries]],
                               lambda pair: str(opposite[pair["need_id"]][pair["work_id"]]))
    filled = tmp_path / "filled.csv"
    _write(filled, rows, bom=False)
    out = O.order_agreement(filled, key, {"judge/one": judge, "judge/two": judge})
    primary = out[O.FIRST]["primary"]
    assert primary["figures"][O.FIRST]["contrasts"]["H1"]["sign"] == -1
    assert primary["figures"]["judge/one"]["contrasts"]["H1"]["sign"] == 1
    assert primary["reading_rule"]["H1"] == "not confirmed by a person"
    assert primary["reading_rule"]["order_holds_on_this_sample"] is False
    assert out[O.SECOND] is None and out["cells"][O.SECOND]["graded"] == 0


def test_second_annotator_from_a_separate_copy(tmp_path):
    queries, lists, _, _ = _toy()
    judge = _judge(lists)
    lists, key, first = _filled(tmp_path, [[x.query_id for x in queries]],
                                lambda pair: str(judge[pair["need_id"]][pair["work_id"]]))
    copy = _rows(tmp_path / O.SHEET_FILE)
    for row in copy:
        row["grade_2"] = "1"
    _write(tmp_path / "first.csv", first)
    _write(tmp_path / "copy.csv", copy)
    out = O.order_agreement(tmp_path / "first.csv", key, {"judge/one": judge}, None, tmp_path / "copy.csv")
    assert out["cells"][O.SECOND]["graded"] == len(copy)
    assert out[O.SECOND]["pair_agreement_between_annotators"]["pairs"] == len(copy)


def test_cli_has_the_order_subcommands():
    parser = build_parser()
    assert parser.parse_args(["order-sheet"]).func.__name__ == "cmd_order_sheet"
    args = parser.parse_args(["order-agreement", "--sheet", "x.csv", "--second-sheet", "y.csv"])
    assert (args.func.__name__, args.sheet, args.second_sheet) == ("cmd_order_agreement", "x.csv", "y.csv")
