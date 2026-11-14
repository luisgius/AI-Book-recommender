"""Tests of the post hoc scope analysis of the catalog evaluation (V29, part A); no network."""

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.catalog_evaluation.scope import SUBSETS, build_markdown, excluded_ids, subset_ids


def _query(qid: str, stratum: str, language: str) -> EvalQuery:
    return EvalQuery(query_id=qid, need_id=qid, text="x", language=language, type="t", stratum=stratum, need="x")


QUERIES = [_query("a1", "A", "en"), _query("b1", "B", "en"), _query("b2", "B", "zh"), _query("b3", "B", "ja"),
           _query("c1", "C", "fr"), _query("d1", "D", "zh")]


def test_subsets_select_the_declared_queries():
    by_key = {s.key: s for s in SUBSETS}
    assert subset_ids(QUERIES, by_key["a"]) == ["b1", "b2", "b3"]
    assert subset_ids(QUERIES, by_key["b"]) == ["a1", "b1"]
    assert subset_ids(QUERIES, by_key["c"]) == ["b1"]
    assert subset_ids(QUERIES, by_key["d"]) == ["b1", "b2", "b3", "c1", "d1"]
    assert excluded_ids(QUERIES, by_key["c"]) == ["b2", "b3", "d1"]
    assert by_key["b"].strata == D.STATISTICAL_STRATA


def _figures(h3: float, p_holm: float):
    contrasts = {name: {"system": "S3", "baseline": "S2", "queries": 10, "difference": -0.1, "ci": [-0.2, -0.05],
                        "p": 0.001, "p_holm": 0.002} for name in ("H1", "H2", "H3", "H4", "H5", "H6")}
    contrasts["H3"] = dict(contrasts["H3"], difference=h3, p_holm=p_holm)
    summary = {s: {"ndcg_at_10": {"mean": 0.5}} for s in D.SYSTEM_NAMES}
    return {"metrics": {j: {"summary_ab": summary, "contrasts": contrasts} for j in ("primary", "second")}}


def test_markdown_states_post_hoc_and_flags_a_changed_significance():
    full, sub = _figures(-0.03, 0.001), _figures(-0.01, 0.2)
    document = {"date": "2026-11-14", "review_date": "2026-11-13", "command": "cmd",
                "full": {"queries": 116, "figures": full},
                "subsets": [{"key": "a", "label": "Estrato B solo", "queries": 10, "queries_by_stratum": {"B": 10},
                             "figures": sub, "changed_decisions": ["H3 significant at 0.05, primary judge"],
                             "decisions": []}]}
    text = build_markdown(document)
    assert "post hoc, fuera del protocolo declarado" in text
    assert "| H3 (S3 - S2) | 10 | -0.010 | [-0.200, -0.050] | 0.200 | no | sí |" in text
