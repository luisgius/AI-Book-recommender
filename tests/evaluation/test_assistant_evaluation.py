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
