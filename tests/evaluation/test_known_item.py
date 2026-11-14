"""Tests of the known-item evaluation (V29, part B): pure parts only, no network, no database."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from app.evaluation.catalog_evaluation.runs import rrf_fuse
from app.evaluation.known_item import analysis as KA
from app.evaluation.known_item import design as D
from app.evaluation.known_item.fusion import weighted_rrf
from app.evaluation.known_item.runner import chunks, estimate_usd, head_union, order_head
from app.evaluation.known_item.sampling import build_queries, build_sample, catalog_indexes, eligibility, normalise


# =============================================================================
# Weighted RRF
# =============================================================================


def test_weighted_rrf_scores_follow_the_formula():
    fused = dict(weighted_rrf(["a", "b"], ["b", "c"], 0.25, k=60))
    assert fused["a"] == pytest.approx(0.25 / 61)
    assert fused["b"] == pytest.approx(0.25 / 62 + 0.75 / 61)
    assert fused["c"] == pytest.approx(0.75 / 62)


def test_weighted_rrf_at_one_half_keeps_the_order_of_the_deployed_fusion():
    lexical, vector = ["a", "b", "c", "d"], ["d", "e", "a", "f"]
    assert [i for i, _ in weighted_rrf(lexical, vector, 0.5)] == [i for i, _ in rrf_fuse(lexical, vector)]


def test_weighted_rrf_extreme_weights_follow_one_branch():
    lexical, vector = ["a", "b", "c"], ["c", "b", "a"]
    assert [i for i, _ in weighted_rrf(lexical, vector, 0.9)] == ["a", "b", "c"]
    assert [i for i, _ in weighted_rrf(lexical, vector, 0.1)] == ["c", "b", "a"]


def test_weighted_rrf_ties_keep_lexical_first_insertion_order():
    # x is only lexical at rank 1, y only vector at rank 1: equal weights tie, lexical first
    assert [i for i, _ in weighted_rrf(["x"], ["y"], 0.5)] == ["x", "y"]


def test_weighted_rrf_rejects_a_weight_outside_zero_one():
    with pytest.raises(ValueError):
        weighted_rrf(["a"], ["b"], 1.5)


# =============================================================================
# Sampling
# =============================================================================


@dataclass
class FakeBook:
    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    language: str = "en"


def _catalog(per_language: int = 3) -> List[FakeBook]:
    books = []
    for language in D.LANGUAGES:
        for i in range(per_language):
            books.append(FakeBook(f"{language}-{i:02d}", f"Title {language} {i}", [f"Author {language} {i}"], language))
    return books


def test_normalise_folds_case_punctuation_and_spaces_but_keeps_accents():
    assert normalise("  Dune. ") == normalise("dune") == "dune"
    assert normalise("Cien  años de\tsoledad!") == "cien años de soledad"
    assert normalise("Brothers_Grimm") == "brothers grimm"
    assert normalise("Gabriel García Márquez") != normalise("Gabriel Garcia Marquez")


def test_eligibility_reasons():
    indexes = catalog_indexes([FakeBook("1", "Same"), FakeBook("2", "same."), FakeBook("3", "Unique", ["A"]),
                               FakeBook("4", "Other", ["Unknown"]), FakeBook("5", "Latin", ["A"], "la")])
    titles, records = indexes["titles"], indexes["records"]
    assert eligibility(records["1"], titles) == "no_author"
    assert eligibility({**records["1"], "authors": ["A"]}, titles) == "title_not_unique"
    assert eligibility(records["3"], titles) == "ok"
    assert eligibility(records["4"], titles) == "no_author"
    assert eligibility(records["5"], titles) == "language"


def test_sample_is_stratified_split_and_reproducible():
    from app.evaluation.known_item import sampling

    indexes = catalog_indexes(_catalog(5))
    first = sampling.sample_books(indexes, per_language=2, seed=7)
    second = sampling.sample_books(catalog_indexes(_catalog(5)), per_language=2, seed=7)
    assert first == second
    assert len(first["books"]) == 2 * len(D.LANGUAGES)
    assert {b["language"] for b in first["books"]} == set(D.LANGUAGES)
    halves = [b["half"] for b in first["books"]]
    assert halves.count(D.DEV) == halves.count(D.TEST) == len(D.LANGUAGES)


def test_sample_takes_each_first_author_once():
    books = _catalog(4)
    # every English book by the same author: only one can be taken
    books = [FakeBook(b.id, b.title, ["Same Person"], b.language) if b.language == "en" else b for b in books]
    books += [FakeBook("en-x", "Extra", ["Other Person"], "en")]
    from app.evaluation.known_item import sampling

    sample = sampling.sample_books(catalog_indexes(books), per_language=2, seed=1)
    authors = [normalise(b["authors"][0]) for b in sample["books"]]
    assert len(authors) == len(set(authors))
    english = sorted(b["authors"][0] for b in sample["books"] if b["language"] == "en")
    assert english == ["Other Person", "Same Person"]


def test_sample_fails_when_a_language_has_too_few_books():
    from app.evaluation.known_item import sampling

    with pytest.raises(ValueError):
        sampling.sample_books(catalog_indexes(_catalog(1)), per_language=2, seed=0)


def test_queries_and_relevance():
    catalog = [FakeBook("b1", "Dune", ["Frank Herbert"]), FakeBook("b2", "Dune Messiah", ["frank herbert."]),
               FakeBook("b3", "Other", ["Someone", "Frank Herbert"]), FakeBook("b4", "Else", ["Frank Herberts"])]
    indexes = catalog_indexes(catalog)
    book = {**indexes["records"]["b1"], "half": D.TEST}
    queries = {q["query_type"]: q for q in build_queries([book], indexes["by_author"])}
    assert queries[D.T]["text"] == "Dune" and queries[D.T]["relevant"] == ["b1"]
    assert queries[D.TA]["text"] == "Dune Frank Herbert" and queries[D.TA]["relevant"] == ["b1"]
    assert queries[D.A]["text"] == "Frank Herbert"
    assert queries[D.A]["relevant"] == ["b1", "b2", "b3"]  # normalised exact match; b4 is another name
    assert {q["query_id"] for q in queries.values()} == {"k001-T", "k001-TA", "k001-A"}


def test_build_sample_document():
    document = build_sample(_catalog(D.BOOKS_PER_LANGUAGE + 2))
    assert len(document["books"]) == D.BOOKS == 300
    assert len(document["queries"]) == 900
    assert sum(b["half"] == D.DEV for b in document["books"]) == D.DEV_BOOKS
    assert document["counts"]["en"]["eligible"] == D.BOOKS_PER_LANGUAGE + 2


# =============================================================================
# Metrics, weight choice, contrasts, decision
# =============================================================================


def _q(qid: str, qtype: str, relevant: List[str], half: str = D.TEST, language: str = "en") -> Dict[str, Any]:
    return {"query_id": qid, "query_type": qtype, "relevant": relevant, "half": half, "language": language}


def test_query_metrics_title_and_author():
    title = KA.query_metrics(["x", "b", "y"], _q("1", D.T, ["b"]))
    assert title == {"mrr_at_10": 0.5, "success_at_1": 0.0, "success_at_10": 1.0}
    assert KA.query_metrics(["b"], _q("1", D.TA, ["b"]))["success_at_1"] == 1.0
    beyond = KA.query_metrics([f"n{i}" for i in range(10)] + ["b"], _q("1", D.T, ["b"]))
    assert beyond["mrr_at_10"] == 0.0 and beyond["success_at_10"] == 0.0
    author = KA.query_metrics(["a1", "x", "a2"], _q("1", D.A, ["a1", "a2"]))
    assert author["success_at_10"] == 1.0
    assert 0.0 < author["ndcg_at_10"] < 1.0
    assert KA.query_metrics(["a1", "a2"], _q("1", D.A, ["a1", "a2"]))["ndcg_at_10"] == pytest.approx(1.0)


def _scores_for(values: Dict[str, float], queries) -> Dict[str, Dict[str, float]]:
    return {q["query_id"]: {D.PRIMARY_METRIC[q["query_type"]]: values[q["query_type"]]} for q in queries}


def test_select_weight_maximises_the_mean_over_types_and_breaks_ties_low():
    dev = [_q("t", D.T, ["b"], D.DEV), _q("ta", D.TA, ["b"], D.DEV), _q("a", D.A, ["b"], D.DEV)]
    scores = {D.grid_name(0.1): _scores_for({D.T: 0.2, D.TA: 0.2, D.A: 0.2}, dev),
              D.grid_name(0.25): _scores_for({D.T: 0.9, D.TA: 0.3, D.A: 0.3}, dev),  # mean 0.5
              D.grid_name(0.75): _scores_for({D.T: 0.5, D.TA: 0.5, D.A: 0.5}, dev),  # mean 0.5 (tie)
              D.grid_name(0.9): _scores_for({D.T: 1.0, D.TA: 0.1, D.A: 0.1}, dev)}
    selection = KA.select_weight(scores, dev)
    assert selection["chosen_weight"] == 0.25
    assert selection["objective"][D.grid_name(0.75)] == pytest.approx(0.5)


def _family_scores(gain_s5: float, gain_s5w: float, n: int = 30):
    queries = [_q(f"{qt}{i}", qt, ["b"]) for qt in D.QUERY_TYPES for i in range(n)]
    scores: Dict[str, Dict[str, Dict[str, float]]] = {D.S4: {}, D.S5: {}, D.S5W: {}}
    for i, q in enumerate(queries):
        metric = D.PRIMARY_METRIC[q["query_type"]]
        base = 0.3 + 0.01 * (i % 7)
        scores[D.S4][q["query_id"]] = {metric: base}
        scores[D.S5][q["query_id"]] = {metric: base + (gain_s5 if q["query_type"] == D.T else 0.0)}
        scores[D.S5W][q["query_id"]] = {metric: base + (gain_s5w if q["query_type"] == D.T else 0.0) * (i % 2)}
    return scores, queries


def test_contrasts_are_six_with_holm_and_drive_the_decision():
    scores, queries = _family_scores(gain_s5=0.2, gain_s5w=0.0)
    family = KA.contrasts(scores, queries)
    assert sorted(family) == sorted(c.name for c in D.CONTRASTS) and len(family) == 6
    assert all(row["p_holm"] >= row["p"] for row in family.values())
    assert family["C1-T"]["significant"] and family["C1-T"]["difference"] == pytest.approx(0.2)
    assert not family["C2-T"]["significant"] and not family["C1-A"]["significant"]
    decision = KA.decision(family)
    assert decision["lexical_helps_for"] == [D.T]
    assert decision["per_type"][D.A]["outcome"].startswith("the conclusion extends")


def test_a_significantly_worse_fusion_does_not_count_as_help():
    scores, queries = _family_scores(gain_s5=-0.2, gain_s5w=0.0)
    family = KA.contrasts(scores, queries)
    assert family["C1-T"]["significant"] and family["C1-T"]["difference"] < 0
    assert KA.decision(family)["lexical_helps_for"] == []


def test_secondary_reading_drops_chinese_and_japanese():
    scores, queries = _family_scores(0.1, 0.1, n=4)
    queries = [dict(q, language="zh") if q["query_id"].endswith("0") else q for q in queries]
    secondary = KA.secondary_contrasts(scores, queries)
    assert secondary["C1-T"]["queries"] == 3
    assert "p_holm" not in secondary["C1-T"]


def test_contrasts_are_declared_with_reranker_on_both_sides():
    assert [(c.system, c.baseline) for c in D.CONTRASTS if c.query_type == D.T] == [(D.S5, D.S4), (D.S5W, D.S4)]
    assert D.SPEND_CAP_USD == 1.5
    assert 0.5 not in D.WEIGHT_GRID


# =============================================================================
# Reranking helpers of the runner (pure)
# =============================================================================


def test_head_union_keeps_first_appearance_order():
    assert head_union([["a", "b"], ["b", "c"], ["d", "a"]]) == ["a", "b", "c", "d"]


def test_order_head_sorts_the_head_stably_and_keeps_the_tail():
    scores = {"a": 0.1, "b": 0.9, "c": 0.9}
    assert order_head(["a", "b", "c", "t1", "t2"], scores, depth=3) == ["b", "c", "a", "t1", "t2"]


def test_order_head_matches_reranking_the_head_alone():
    # Scoring the union once and sorting each head equals sorting each head by its own scores.
    scores = {"a": 0.3, "b": 0.8, "c": 0.5, "d": 0.1}
    for head in (["a", "b", "c"], ["d", "c", "a"]):
        assert order_head(head, scores, depth=3) == sorted(head, key=lambda i: scores[i], reverse=True)


def test_chunks_and_estimate():
    assert chunks(list("abcde"), 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert estimate_usd(1_000_000) == pytest.approx(D.RERANK_USD_PER_MILLION_CHARS)


# =============================================================================
# Offline steps of the runner and the report, on a temporary directory
# =============================================================================


def test_fusions_weight_choice_and_report_offline(tmp_path):
    import json

    from app.evaluation.catalog_evaluation.runs import run_store
    from app.evaluation.known_item.report import write_report
    from app.evaluation.known_item.runner import choose_weight, eval_query, record, run_fusions

    document = build_sample(_catalog(D.BOOKS_PER_LANGUAGE))
    (tmp_path / D.SAMPLE_FILE).write_text(json.dumps(document), encoding="utf-8")
    queries = document["queries"]
    for q in queries:  # lexical finds the book first; vector finds it third
        target = q["relevant"][0]
        lexical = [target] + [f"l{i}" for i in range(59)]
        vector = ["v0", "v1", target] + [f"v{i}" for i in range(2, 59)]
        for name, ids in ((D.S1, lexical), ("S1@60", lexical), (D.S2, vector), ("S2@60", vector)):
            run_store(tmp_path, name).add(record(q["query_id"], [(i, 1.0) for i in ids], {}))
    run_fusions(tmp_path, [eval_query(q) for q in queries])
    selection = choose_weight(tmp_path, queries)
    assert selection["chosen_weight"] == 0.1  # every weight puts the book first: the tie goes to the smallest w
    assert (tmp_path / D.SELECTION_FILE).exists()
    for name, base in D.RERANKED.items():  # stand-in for the paid reranking: the base order
        for line in run_store(tmp_path, base).records.values():
            run_store(tmp_path, name).add(line)
    results = write_report(tmp_path)
    assert results["test"]["queries"] == 450
    assert results["test"]["summary"][D.T][D.S1]["mrr_at_10"]["mean"] == pytest.approx(1.0)
    assert results["test"]["summary"][D.T][D.S2]["mrr_at_10"]["mean"] == pytest.approx(1 / 3)
    assert set(results["contrasts"]) == {c.name for c in D.CONTRASTS}
    report = (tmp_path / D.REPORT_FILE).read_text(encoding="utf-8")
    assert report.startswith("# Evaluación")
    assert "## Desviaciones" in report and "## Límites (sección 9 del protocolo)" in report
    assert len(results["limits"]) == 7  # the seven bullets of the protocol's section 9
    assert "fuera de la familia" in report
