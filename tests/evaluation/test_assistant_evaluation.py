"""Tests of the pure logic of the assistant evaluation (V22) and of its offline report."""

import json
import socket
from pathlib import Path

import pytest

from app.evaluation.assistant_evaluation import analysis as A
from app.evaluation.assistant_evaluation import design as D
from app.evaluation.assistant_evaluation import sets as S
from app.evaluation.assistant_evaluation.simulated_user import (
    STOP_ANSWERED,
    STOP_ERROR,
    STOP_MAX_TURNS,
    run_conversation,
    user_prompt,
)
from app.evaluation.catalog_evaluation.queries import EvalQuery


def _b(qid, text, language="en", qtype="thematic", need="a need", title=None, author=None):
    return EvalQuery(qid, qid, text, language, qtype, "B", need, target_title=title, target_author=author)


# =============================================================================
# Sets
# =============================================================================


def test_v14_requests_keep_a_b_d_and_a_categories():
    queries = [EvalQuery("q1", "q1", "python", "en", "original", "A", "python"),
               _b("b-en-1", "gothic novels"),
               EvalQuery("c01-en", "c01", "x", "en", "thematic", "C", "n"),
               EvalQuery("d-en", "d-en", "train times", "en", "unanswerable", "D", "trains")]
    out = S.v14_requests(queries, [{"query_id": "q1", "category": "ambiguous"}])
    assert [r.request_id for r in out] == ["q1", "b-en-1", "d-en"]
    assert out[0].category == "ambiguous" and out[1].category == "thematic"
    assert A.route_group(out[0]) == "A ambiguous" and A.route_group(out[1]) == "B"


def test_e_sample_is_seeded_and_sorted_independent():
    qs = [_b(f"b-en-{i}", f"query {i}") for i in range(10)]
    first = [q.query_id for q in S.sample_e_needs(qs, size=4, seed=0)]
    again = [q.query_id for q in S.sample_e_needs(list(reversed(qs)), size=4, seed=0)]
    assert first == again and len(set(first)) == 4


def test_shared_words_count_inflected_forms():
    assert S.shared_words("славянская мифология", "книга про мифологию древних славян") == ["мифология", "славянская"]
    assert S.shared_words("gothic fiction", "early Victorian gothic horror novels") == ["gothic"]
    assert S.shared_words("science fiction", "a novel about an astronomer") == []


def test_check_e_flags_length_give_away_and_known_item():
    needs = {"b-en-1": _b("b-en-1", "early Victorian gothic horror novels"),
             "b-en-2": _b("b-en-2", "a novel about a whale hunt", qtype="known_item", title="Moby-Dick", author="Herman Melville"),
             "b-ja-1": _b("b-ja-1", "孤島に取り残された少年たちの小説", language="ja")}
    requests = [S.Request("e-b-en-1", "E", "victorian gothic horror", "en", "b-en-1", "n"),
                S.Request("e-b-en-2", "E", "Melville novels", "en", "b-en-2", "n"),
                S.Request("e-b-ja-1", "E", "海外文学の小説を読みたいです", "ja", "b-ja-1", "n")]
    problems = S.check_e(requests, needs)["problems"]
    checks = {(p["request_id"], p["check"]) for p in problems}
    assert ("e-b-en-1", "gives need away") in checks
    assert ("e-b-en-2", "gives need away") in checks  # the author's surname
    assert ("e-b-ja-1", "length") in checks  # more than 8 CJK characters
    assert ("e-b-en-1", "length") not in checks


def test_check_e_accepts_a_vague_request():
    needs = {"b-en-1": _b("b-en-1", "early Victorian gothic horror novels")}
    out = S.check_e([S.Request("e-b-en-1", "E", "gothic fiction", "en", "b-en-1", "n")], needs)
    assert [p for p in out["problems"] if p["request_id"]] == []


def _work(i, author="A", subject="s", language="en"):
    return S.Work(f"w{i:03d}", f"Title {i}", [author], language, [subject], "1900", "d" * 400)


def test_pick_f_works_is_seeded_and_pairs_comparisons():
    works = [_work(i, author=f"auth{i % 5}") for i in range(60)]
    tasks = S.pick_f_works(works, kinds={"compare": 3, "like_but": 2, "facts": 1}, seed=0)
    assert [t["kind"] for t in tasks] == ["compare"] * 3 + ["like_but"] * 2 + ["facts"]
    assert all(len(t["works"]) == 2 for t in tasks[:3]) and all(len(t["works"]) == 1 for t in tasks[3:])
    for t in tasks[:3]:
        assert t["works"][0].authors == t["works"][1].authors  # same author first
    assert tasks == S.pick_f_works(works, kinds={"compare": 3, "like_but": 2, "facts": 1}, seed=0)


def test_title_named_and_check_f():
    assert S.title_named("Could you compare Justine and Clea?", "Clea")
    assert S.title_named("Books like Los retos de la diferencia please", "Los retos de la diferencia : Los actores")
    assert not S.title_named("Books like Ringworld", "Seventh Son")
    work = _work(1)
    tasks = {"f01": {"id": "f01", "kind": "facts", "language": "en", "works": [work]}}
    answers = {"f01": {"text": "What is Title 1 about?", "works_named": ["Title 1", "Unknown Book"]}}
    requests = S.f_requests(list(tasks.values()), answers)
    out = S.check_f(requests, tasks, answers, lambda title, author: ["w001"] if title == "Title 1" else [])
    details = [(p["check"], p["detail"]) for p in out["problems"] if p["request_id"]]
    assert details == [("work not in catalog", "Unknown Book")]


# =============================================================================
# The simulated-user loop
# =============================================================================


def _reply(route, message="m", options=()):
    return {"route": route, "message": message, "options": list(options)}


def test_conversation_stops_when_the_assistant_searches():
    calls = []

    def assistant(turn, messages):
        calls.append([m["content"] for m in messages])
        return _reply("clarify", "Which kind?", ["Novels", "Poetry"]) if turn == 1 else _reply("search")

    out = run_conversation("literature", assistant, lambda t, m, r: r["options"][0])
    assert out["stop"] == STOP_ANSWERED and out["questions"] == 1 and len(out["replies"]) == 2
    assert calls[1] == ["literature", "Which kind?", "Novels"]
    assert out["messages"][1]["route"] == "clarify"


def test_conversation_safety_stop_and_error():
    out = run_conversation("x", lambda t, m: _reply("clarify", "?", ["a", "b"]), lambda t, m, r: "a", max_turns=3)
    assert out["stop"] == STOP_MAX_TURNS and len(out["replies"]) == 3 and out["questions"] == 3
    out = run_conversation("x", lambda t, m: None, lambda t, m, r: "a")
    assert out["stop"] == STOP_ERROR and out["replies"] == []


def test_user_prompt_shows_need_options_and_language_but_no_results():
    prompt = user_prompt("gothic novels of the 1840s", "es", [{"role": "user", "content": "novelas", "route": None}],
                         ["Terror", "Romance"])
    assert "gothic novels of the 1840s" in prompt and "Terror" in prompt and "Spanish" in prompt
    assert "route" not in prompt


# =============================================================================
# Routes, thresholds, stability
# =============================================================================


def test_route_matrix_rates_and_thresholds():
    sets_of = {f"a{i}": "A" for i in range(50)} | {f"b{i}": "B" for i in range(50)}
    routes = {k: "search" for k in sets_of}
    routes["a0"] = "out_of_scope"  # 1/100 = 1 % <= 2 %
    for i in range(6):
        routes[f"b{i}"] = "clarify"  # 6/50 = 12 % > 10 %
    rates = A.route_rates(routes, sets_of, ["out_of_scope", "search"])
    assert rates["out_of_scope_ab"]["rate"] == pytest.approx(0.01)
    assert rates["clarify_b"]["rate"] == pytest.approx(0.12)
    assert rates["d_detection"]["count"] == 1
    th = A.thresholds(rates)
    assert th["rows"]["out_of_scope_ab"]["met"] and not th["rows"]["clarify_b"]["met"] and not th["both_met"]
    routes["a1"] = "out_of_scope"
    routes["a2"] = "out_of_scope"  # 3 % > 2 %
    assert not A.thresholds(A.route_rates(routes, sets_of, []))["rows"]["out_of_scope_ab"]["met"]
    matrix = A.route_matrix([("B", "search"), ("B", "clarify"), ("D", "error")])
    assert matrix["B"] == {"search": 1, "clarify": 1, "total": 2} and matrix["D"] == {"error": 1, "total": 1}


def test_threshold_boundaries_are_inclusive():
    sets_of = {f"b{i}": "B" for i in range(10)}
    routes = {k: "search" for k in sets_of}
    routes["b0"] = "clarify"  # exactly 10 %
    assert A.thresholds(A.route_rates(routes, sets_of, []))["rows"]["clarify_b"]["met"]


def test_stability():
    out = A.stability({"a": "search", "b": "clarify", "c": "search"}, {"a": "search", "b": "search", "c": "search"},
                      {"a": "A", "b": "B", "c": "B"})
    assert out["same_route"] == 2 and out["changes"] == {"clarify -> search": 1} and out["changed_requests"] == ["b"]
    assert out["by_set"]["B"] == {"requests": 2, "same_route": 1, "agreement": 0.5}


# =============================================================================
# Lists, recommendation precision against P, grounding, contrasts
# =============================================================================


def test_reply_list_and_picks_by_route():
    search = {"route": "search", "search": {"ids": ["x", "y"], "rerank_scores": [1, 1]},
              "recommendations": [{"work_id": "y", "citations": []}], "agent": None}
    agent = {"route": "agent", "search": None, "recommendations": [{"work_id": "z"}],
             "agent": {"recommendations": [{"work_id": "z"}, {"work_id": "w"}]}}
    assert A.reply_list(search) == ["x", "y"] and A.reply_picks(search) == ["y"]
    assert A.reply_list(agent) == ["z", "w"]
    assert A.reply_list({"route": "clarify"}) == [] and A.reply_list(None) == []
    assert A.reply_route(None) == "error"


def test_precision_against_p_at_the_same_k():
    grades = {"a": 3, "b": 1, "c": 2, "d": 0}
    assert A.precision_at(["a", "b"], grades, 2) == 0.5  # grade 1 is not "2 or 3"
    assert A.precision_at(["c", "a", "b"], grades, 2) == 1.0
    assert A.precision_at(["unjudged"], grades, 1) == 0.0
    assert A.precision_at([], grades, 0) is None
    assert A.ndcg10([], grades) == 0.0 and A.ndcg10(["a"], {}) == 0.0


def test_search_grounding_counts_drop_reasons():
    reply = {"route": "search", "search": {"ids": [f"w{i}" for i in range(10)]},
             "recommendations": [{"work_id": "w0", "citations": [{"field": "title", "snippet": "t"}]}]}
    raw = [{"output": {"picks": [{"book_id": "w0", "citations": [1, 2]}, {"book_id": "w9", "citations": [1]},
                                 {"book_id": "w1", "citations": [1]}, {"book_id": "w0", "citations": []}]}}]
    row = A.search_grounding(reply, raw, answer_context=6)
    assert row["picks_proposed"] == 4 and row["picks_kept"] == 1
    assert row["dropped_reasons"] == {"not_shown": 1, "no_valid_citation": 1, "repeat": 1}
    assert row["citations_proposed"] == 4 and row["citations_kept"] == 1
    total = A.sum_grounding([row, None, {"answer_call": False}])
    assert total["replies"] == 2 and total["replies_without_answer_output"] == 1 and total["picks_dropped"] == 3


def test_contrast_family_holm_and_conclusions():
    up = ([1.0] * 30, [0.0] * 30)
    flat = ([0.5, 0.4] * 15, [0.4, 0.5] * 15)
    fam = A.contrast_family({"H7": up, "H8": flat, "H9": ([], [])})
    assert set(fam) == {"H7", "H8"}  # an empty contrast is not tested
    assert fam["H7"]["difference"] == 1.0 and fam["H7"]["p_holm"] < 0.05 and fam["H8"]["p_holm"] >= fam["H8"]["p"]
    concl = A.conclusions(fam, fam)
    assert concl == {"H7": "sustained", "H8": "not significant"}


def test_reply_language_verdict():
    assert A.reply_language_verdict("Aquí tienes algunas novelas góticas que encajan con lo que buscas en el catálogo.", "es")[0] == "ok"
    assert A.reply_language_verdict("Here are some gothic novels that match what you are looking for in the catalog.", "es") == ("mismatch", "en")
    assert A.reply_language_verdict("", "en")[0] == "undecided"


# =============================================================================
# Offline report
# =============================================================================


def _jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _search_reply(ids, picks=(), route="search", query="q"):
    return {"route": route, "forced_route": None, "message": "Here are some books that match what you are looking for today.",
            "options": [], "search_query": query, "filters": None, "clarifications": 0, "notices": [], "timings_ms": {"route": 500.0},
            "model": "m", "usage": {}, "search": {"ids": list(ids), "rerank_scores": [0.9] * len(ids), "search_mode": "vector_only",
                                                  "degraded": False, "degradation_reason": None, "notices": [], "latency_ms": 1.0,
                                                  "timings_ms": {}},
            "recommendations": [{"work_id": w, "reason": "r", "citations": [{"field": "title", "snippet": "t"}]} for w in picks],
            "agent": None}


def _turn(run, rid, set_name, reply, turn=1):
    return {"key": f"{run}:{rid}:{turn}", "run": run, "request_id": rid, "set": set_name, "turn": turn, "messages": [],
            "reply": reply, "router_raw": [], "answer_raw": [], "llm_calls": [{"operation": "assistant", "step": "route",
            "model": "m", "latency_s": 1.0, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.001, "error": None}],
            "api_calls": [], "wall_ms": 2000.0, "attempts": 1, "failed_attempts": [], "commit": "abc",
            "date": "2026-11-12T10:00:00"}


@pytest.fixture
def offline_dir(tmp_path, monkeypatch):
    from app.evaluation.assistant_evaluation import judging
    from app.evaluation.relevance_judge import JudgmentCache, LLMJudgment
    from app.infrastructure.llm.prompts_judge import RELEVANCE_JUDGE_PROMPT_VERSION

    d = tmp_path / "assistant"
    reqs = [S.Request("q1", "A", "python", "en", "q1", "python", category="ambiguous"),
            S.Request("b-en-1", "B", "gothic novels", "en", "b-en-1", "gothic"),
            S.Request("d-en", "D", "train times", "en", "d-en", "trains"),
            S.Request("e-b-en-1", "E", "fiction", "en", "b-en-1", "gothic", source_query="gothic novels"),
            S.Request("f01", "F", "Compare X and Y", "en", "f01", "Compare X and Y", category="compare", works=["w1", "w2"])]
    S.save_requests(d / "sets" / "requests.json", reqs, {"agent_sample": ["q1"]})
    (d / "sets" / "checks.json").write_text(json.dumps({"sets": S.describe_sets(reqs)}), encoding="utf-8")
    _jsonl(d / "runs" / "P.jsonl", [{"text": t, "ids": ["w1", "w2", "w3"], "reranked": True}
                                     for t in ("python", "gothic novels", "fiction", "Compare X and Y")])
    turns = [_turn("run1", "q1", "A", _search_reply(["w1", "w2"], ["w1"])),
             _turn("run2", "q1", "A", _search_reply(["w1"])),
             _turn("run1", "b-en-1", "B", {**_search_reply([]), "route": "clarify", "message": "Which?", "options": ["a", "b"]}),
             _turn("run2", "b-en-1", "B", _search_reply(["w3"])),
             _turn("run1", "d-en", "D", {**_search_reply([]), "route": "out_of_scope", "search": None}),
             _turn("run1", "e-b-en-1", "E", {**_search_reply([]), "route": "clarify", "search": None}),
             _turn("run1", "e-b-en-1", "E", _search_reply(["w2", "w1"], ["w2"]), turn=2),
             _turn("run1", "f01", "F", {**_search_reply([], route="agent"), "search": None,
                                        "recommendations": [{"work_id": "w1", "reason": "r", "citations": []}],
                                        "agent": {"answer": "a", "recommendations": [{"work_id": "w1", "reason": "r"}],
                                                  "steps": [{"tool": "search_books", "arguments": {}, "result_count": 3,
                                                             "books": ["w1"]}], "dropped_recommendations": 1,
                                                  "reflections": 0, "model": "m"}})]
    _jsonl(d / "runs" / "turns.jsonl", turns)
    _jsonl(d / "runs" / "conversations.jsonl", [{"request_id": "e-b-en-1", "stop": "answered", "questions": 1,
                                                  "assistant_turns": 2, "messages": [], "user_calls": []}])
    _jsonl(d / "runs" / "agent_direct.jsonl", [{"request_id": "q1", "set": "A", "agent": {
        "answer": "a", "recommendations": [{"work_id": "w2", "reason": "r"}], "steps": [], "dropped_recommendations": 0,
        "reflections": 0, "model": "m"}, "llm_calls": [], "api_calls": [], "wall_ms": 9000.0, "commit": "abc", "date": "2026-11-12T10:01:00"}])
    works = [{"work_id": w, "source": "s", "source_id": w} for w in ("w1", "w2", "w3")]
    pairs = {n: {"need": n, "measured": ["w1", "w2", "w3"], "works": works} for n in ("q1", "b-en-1", "f01")}
    (d / "judgments").mkdir(parents=True)
    (d / "judgments" / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    empty = tmp_path / "v14"
    for model in D.JUDGES:
        cache = JudgmentCache.for_judge(d / "judgments", model, RELEVANCE_JUDGE_PROMPT_VERSION)
        for need in pairs:
            for w, g in (("w1", 3), ("w2", 2), ("w3", 0)):
                cache.add(LLMJudgment(query_id=need, source="s", source_id=w, grade=g, rationale="", model=model,
                                      prompt_version=RELEVANCE_JUDGE_PROMPT_VERSION, date="2026-11-12", query_text=need, title=w))
    monkeypatch.setattr(judging, "v14_cache", lambda model: JudgmentCache.for_judge(empty, model, RELEVANCE_JUDGE_PROMPT_VERSION))
    # the system description saved when the measurement ran (the report reads nothing else about the system)
    (d / "system.json").write_text(json.dumps({
        "commit": "abc", "llm": {"model": "saved/model", "temperature": 0.0, "reasoning": False},
        "assistant": {"answer_context_books": 6}}), encoding="utf-8")
    return d


def test_offline_report_needs_no_network(offline_dir, tmp_path, monkeypatch):
    from app.evaluation.assistant_evaluation.cli import main

    def refuse(*args, **kwargs):
        raise AssertionError("the offline report opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    output = tmp_path / "REPORT.md"
    assert main(["--data-dir", str(offline_dir), "report", "--offline", "--output", str(output)]) == 0
    text = output.read_text(encoding="utf-8")
    assert "Evaluation of the assistant and the agent (V22)" in text
    assert D.PROTOCOL_COMMIT in text and "LLM judges only" in text and "simulated" in text
    assert "## Deviations from the protocol" in text and "## Limits" in text
    results = json.loads((offline_dir / "results.json").read_text(encoding="utf-8"))
    # run 1: B asked a question (1/1 > 10 %), nobody was out of scope
    th = results["routes"]["thresholds"]["run1"]
    assert th["rows"]["clarify_b"]["met"] is False and th["rows"]["out_of_scope_ab"]["met"] is True
    assert results["routes"]["stability"]["same_route"] == 1
    assert results["routes"]["rates"]["run1"]["d_detection"]["count"] == 1
    primary = results["metrics"]["primary"]
    assert primary["e"]["asked"] == 1 and primary["f"]["with_recommendations"] == 1
    assert primary["f"]["rows"][0]["precision_picks"] == 1.0 and primary["f"]["rows"][0]["precision_P"] == 1.0
    assert results["grounding"]["agent_in_assistant"]["dropped_not_from_tools"] == 1
    assert set(results["conclusions"]) == {"H7", "H8", "H9"}
    # no re-measurement next to offline_dir: no second report, and the published one is not touched
    assert not (tmp_path / D.V25_REPORT_MD.name).exists()


# =============================================================================
# Re-measurement (V25 addendum): set G, thresholds, both reports offline
# =============================================================================


def test_g_tasks_balance_languages_forms_and_kinds():
    from collections import Counter

    from app.evaluation.assistant_evaluation import set_g

    tasks = set_g.g_tasks()
    valid = [t for t in tasks if t["group"] == "valid"]
    oos = [t for t in tasks if t["group"] == "out_of_scope"]
    assert len(valid) == 60 and len(oos) == 20 and len({t["id"] for t in tasks}) == 80
    assert set(Counter((t["language"], t["form"]) for t in valid).values()) == {5}
    assert Counter(t["field"] for t in oos) == {k: 5 for k in set_g.G_OUT_OF_SCOPE}
    assert Counter(t["field"] for t in valid) == {k: 6 for k in set_g.G_TOPICS}
    assert set_g.g_tasks() == tasks  # seeded


def test_check_g_flags_each_rule():
    from app.evaluation.assistant_evaluation import set_g

    tasks = [{"id": "g1", "group": "valid", "field": "cooking", "topic": "sourdough bread baking", "language": "en",
              "form": "keywords"},
             {"id": "g2", "group": "valid", "field": "cooking", "topic": "Thai curry cooking", "language": "en",
              "form": "sentence"},
             {"id": "g3", "group": "out_of_scope", "field": "calculation", "topic": "15 percent of 240", "language": "en",
              "form": "any"}]
    good = {"g1": {"text": "sourdough bread", "english": "sourdough bread"},
            "g2": {"text": "I am looking for a book about Thai curry for beginners",
                   "english": "I am looking for a book about Thai curry for beginners"},
            "g3": {"text": "What is 15 percent of 240?", "english": "What is 15 percent of 240?"}}
    result = set_g.check_g(good, tasks, other_texts=[])
    assert [p for p in result["problems"] if p["request_id"]] == []
    bad = {"g1": {"text": "how to bake sourdough bread at home?", "english": "how to bake sourdough bread at home?"},
           "g2": {"text": "Teach me to cook Thai curry tonight please", "english": "Teach me to cook Thai curry tonight please"},
           "g3": {"text": "Which book explains percentages?", "english": "Which book explains percentages?"}}
    checks = {(p["request_id"], p["check"]) for p in set_g.check_g(bad, tasks, other_texts=["sourdough bread"])["problems"]}
    assert ("g1", "form") in checks and ("g2", "asks for reading") in checks
    assert ("g3", "asks for books") in checks and ("g3", "kind") in checks
    dup = set_g.check_g(good, tasks, other_texts=["Sourdough  bread"])["problems"]
    assert any(p["request_id"] == "g1" and p["check"] == "duplicate" for p in dup)


def test_g_routes_threshold_allows_one_refusal_in_sixty():
    rows = [{"group": "valid", "language": "es", "field": "cooking", "form": "keywords", "route": "search"}] * 59
    rows = rows + [{"group": "valid", "language": "es", "field": "cooking", "form": "keywords", "route": "out_of_scope"}]
    rows = rows + [{"group": "out_of_scope", "language": "en", "field": "program", "form": "any", "route": "out_of_scope"}] * 20
    g = A.g_routes(rows)
    assert g["threshold"]["count"] == 1 and g["threshold"]["met"] is True
    assert g["out_of_scope"]["detection"]["count"] == 20
    two = A.g_routes(rows[:58] + [rows[-21]] * 2 + rows[60:])
    assert two["threshold"]["count"] == 2 and two["threshold"]["met"] is False


def test_spend_cap_and_v25_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "V25_DIR", tmp_path / "v25")
    assert D.spend_cap(tmp_path / "v25") == D.V25_SPEND_CAP_USD == 2.5
    assert D.spend_cap(tmp_path / "other") == D.SPEND_CAP_USD


@pytest.fixture
def second_dir(offline_dir, tmp_path, monkeypatch):
    """A re-measurement directory next to offline_dir: A-F as there, plus G, a log and the instructions record."""
    import shutil

    from app.evaluation.relevance_judge import JudgmentCache
    from app.infrastructure.llm.prompts_judge import RELEVANCE_JUDGE_PROMPT_VERSION

    d = tmp_path / "assistant_v25"
    monkeypatch.setattr(D, "DATA_DIR", offline_dir)
    monkeypatch.setattr(D, "V25_DIR", d)
    for name in ("sets", "runs"):
        shutil.copytree(offline_dir / name, d / name)
    (d / "judgments").mkdir(parents=True)
    # pairs.json of the second measurement; its judgments all come from the first measurement's caches
    shutil.copy(offline_dir / "judgments" / "pairs.json", d / "judgments" / "pairs.json")
    for model in D.JUDGES:
        assert len(JudgmentCache.for_judge(d / "judgments", model, RELEVANCE_JUDGE_PROMPT_VERSION)) == 0
    meta = json.loads((d / "sets" / "requests.json").read_text(encoding="utf-8"))
    reqs = [S.Request.from_dict(r) for r in meta["requests"]]
    g = [S.Request("g01", "G", "pan de masa madre", "es", "g01", "sourdough", category="valid",
                   notes=["cooking", "keywords", "sourdough bread"]),
         S.Request("g02", "G", "un libro de cocina tailandesa", "es", "g02", "thai", category="valid",
                   notes=["cooking", "sentence", "a Thai cooking book"]),
         S.Request("g-oos01", "G", "15 % de 240", "es", "g-oos01", "calc", category="out_of_scope",
                   notes=["calculation", "any", "15 % of 240"])]
    S.save_requests(d / "sets" / "requests.json", reqs + g, {k: v for k, v in meta.items() if k != "requests"})
    turns = [json.loads(line) for line in (d / "runs" / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
    turns += [_turn("run1", "g01", "G", _search_reply(["w1"])),
              _turn("run1", "g02", "G", {**_search_reply([]), "route": "out_of_scope", "search": None}),
              _turn("run1", "g-oos01", "G", {**_search_reply([]), "route": "out_of_scope", "search": None})]
    _jsonl(d / "runs" / "turns.jsonl", turns)
    (d / "sets" / "g_checks.json").write_text(json.dumps({"generator": {"model": "gen"}, "G": {"fixes": []}}),
                                              encoding="utf-8")
    instruction = {"router_system_prompt": "x", "sha256": "s", "prompts_version": "v", "commit": "c",
                   "route_lines": ['- "out_of_scope": old']}
    (d / "instructions.json").write_text(json.dumps({"first": instruction, "second": {
        **instruction, "router_system_prompt": "y", "route_lines": ['- "out_of_scope": new']}}), encoding="utf-8")
    (d / "system.json").write_text(json.dumps({
        "commit": "def", "llm": {"model": "saved/model", "temperature": 0.0, "reasoning": False},
        "llm_openrouter_extra_body": {"reasoning": {"enabled": False}}, "assistant": {"answer_context_books": 6}}),
        encoding="utf-8")
    (d / "assistant.log").write_text(
        "WARNING x: Assistant answer failed: Could not parse response content as the length limit was reached - "
        "CompletionUsage(completion_tokens=1024, prompt_tokens=9, completion_tokens_details=CompletionTokensDetails("
        "reasoning_tokens=0))\n", encoding="utf-8")
    return d


def test_offline_report_covers_both_measurements(offline_dir, second_dir, tmp_path, monkeypatch):
    from app.evaluation.assistant_evaluation import judging
    from app.evaluation.assistant_evaluation.cli import main

    def refuse(*args, **kwargs):
        raise AssertionError("the offline report opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    first_md, second_md = tmp_path / "REPORT.md", tmp_path / "REPORT_V25.md"
    assert main(["--data-dir", str(offline_dir), "report", "--offline", "--output", str(first_md), "--skip-second"]) == 0
    first_files = {p: p.read_bytes() for p in (first_md, offline_dir / "results.json", offline_dir / "per_request.json")}
    assert main(["--data-dir", str(offline_dir), "report", "--offline", "--output", str(first_md),
                 "--second-data-dir", str(second_dir), "--second-output", str(second_md)]) == 0
    # the first measurement's report and results do not change when the second one is reported
    assert {p: p.read_bytes() for p in first_files} == first_files
    text = second_md.read_text(encoding="utf-8")
    assert "Re-measurement of the assistant" in text and "## The two instructions" in text
    assert '- "out_of_scope": old' in text and '- "out_of_scope": new' in text
    assert "**not met**" in text and "LLM judges only" in text and "## Deviations from the addendum" in text
    second = json.loads((second_dir / "results.json").read_text(encoding="utf-8"))
    v = second["v25"]
    assert v["g"]["threshold"]["count"] == 1 and v["g"]["threshold"]["requests"] == 2
    assert v["g"]["out_of_scope"]["detection"]["count"] == 1
    assert v["thresholds"]["required_met"] is False and v["spend_cap_usd"] == D.V25_SPEND_CAP_USD
    # G stays out of the measures both measurements share
    assert "G valid" not in second["routes"]["matrices"]["run1"]
    assert sum(second["routes"]["matrices"]["run1"]["B"].values()) == 2  # clarify 1 + total 1
    # every judgment of the second measurement comes from the first one's caches
    counts = second["judgments"]["counts"][D.PRIMARY_JUDGE]
    assert counts["from_first_measurement"] == counts["measured_pairs"] and counts["judged_here"] == 0
    assert judging.first_measurement_cache(offline_dir, D.PRIMARY_JUDGE) is None
    assert "| second (V25) | 1 | 0 | [0] |" in text  # one answer call at the cap, no reasoning


def test_report_outputs_default_to_the_published_paths_for_the_design_directories():
    assert D.report_md(D.DATA_DIR) == D.REPORT_MD
    assert D.second_data_dir(D.DATA_DIR) == D.V25_DIR
    assert D.second_report_md(D.V25_DIR) == D.V25_REPORT_MD


def test_report_outputs_follow_the_data_directory(offline_dir, second_dir, tmp_path):
    """With only --data-dir, every output of the report is written next to the given directories."""
    from app.evaluation.assistant_evaluation.cli import main

    assert second_dir == D.second_data_dir(offline_dir)  # the fixture's layout is the default one
    assert main(["--data-dir", str(offline_dir), "report", "--offline"]) == 0
    for path in (tmp_path / D.REPORT_MD.name, tmp_path / D.V25_REPORT_MD.name, offline_dir / "results.json",
                 offline_dir / "per_request.json", second_dir / "results.json", second_dir / "per_request.json"):
        assert path.is_file(), path
    assert "Re-measurement of the assistant" in (tmp_path / D.V25_REPORT_MD.name).read_text(encoding="utf-8")


def test_offline_reports_ignore_environment_and_say_what_was_not_recorded(offline_dir, second_dir, tmp_path, monkeypatch):
    """Both offline reports describe the system only from the saved system.json of each measurement."""
    from app.evaluation.assistant_evaluation.cli import main

    first_json = offline_dir / "system.json"
    saved = json.loads(first_json.read_text(encoding="utf-8"))
    envs = [
        {},
        {"LLM_REASONING": "high", "LLM_MODEL": "other/model", "LLM_PROVIDER": "ollama", "LLM_TEMPERATURE": "0.9",
         "OPENROUTER_API_KEY": "sk-not-a-real-key", "SEARCH_PROFILE": "local", "JUDGE_REASONING": "high"},
    ]
    names = ("LLM_REASONING", "LLM_MODEL", "LLM_PROVIDER", "LLM_TEMPERATURE", "OPENROUTER_API_KEY", "SEARCH_PROFILE",
             "JUDGE_REASONING")
    outputs = []
    for i, env in enumerate(envs):
        for n in names:
            monkeypatch.delenv(n, raising=False)
        for n, v in env.items():
            monkeypatch.setenv(n, v)
        first_md, second_md = tmp_path / f"r{i}.md", tmp_path / f"r{i}_v25.md"
        assert main(["--data-dir", str(offline_dir), "report", "--offline", "--output", str(first_md),
                     "--second-data-dir", str(second_dir), "--second-output", str(second_md)]) == 0
        outputs.append((first_md.read_bytes(), second_md.read_bytes()))
    assert outputs[0] == outputs[1]
    text = outputs[0][1].decode("utf-8")
    assert "saved/model" in outputs[0][0].decode("utf-8") and "other/model" not in text
    assert "first (V22) reasoning `False`, request body not recorded" in text

    # a field the first measurement did not save is stated as not recorded, never guessed
    first_json.write_text(json.dumps({**saved, "llm": {"model": "saved/model"}}), encoding="utf-8")
    first_md, second_md = tmp_path / "n.md", tmp_path / "n_v25.md"
    assert main(["--data-dir", str(offline_dir), "report", "--offline", "--output", str(first_md),
                 "--second-data-dir", str(second_dir), "--second-output", str(second_md)]) == 0
    assert "first (V22) reasoning `not recorded`" in second_md.read_text(encoding="utf-8")
    assert "reasoning not recorded" in first_md.read_text(encoding="utf-8")
