"""
Stages 2 and 3: the reference P, the assistant's turns and the direct agent.

Files under <data>/runs/ (JSON Lines, one record per line, written and synced
as soon as it is done, so every part resumes where it stopped):

- P.jsonl, keyed by text: P (the search service as deployed, no LLM) on a
  request text, max_results=100: work ids, rerank scores, mode, degradation,
  timings. A search whose reranker or embedding failed is repeated (up to
  TURN_ATTEMPTS searches) so that P is not a fallback.
- depth_check.jsonl, keyed by text: the approximate vector search at the
  candidate depth of a 10-result page (60) and of P (200): are the 30
  candidates the reranker sees the same? (database only, no paid call)
- turns.jsonl, keyed by "<run>:<request id>:<turn>": one assistant turn, the
  messages sent, the serialized reply, the raw router and answer outputs,
  every LLM and API call made during it, wall time.
- conversations.jsonl, keyed by request id: the E conversations.
- agent_direct.jsonl, keyed by request id: the agent called directly.
- the simulated user's calls under <data>/runs/simulated_user/.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.assistant_evaluation.sets import Request, by_set, load_requests
from app.evaluation.assistant_evaluation.simulated_user import SimulatedUser, run_conversation
from app.evaluation.assistant_evaluation.system import CAPTURE
from app.evaluation.catalog_evaluation.runs import JsonlStore

P_FILE = "P.jsonl"
DEPTH_FILE = "depth_check.jsonl"
TURNS_FILE = "turns.jsonl"
CONVERSATIONS_FILE = "conversations.jsonl"
AGENT_FILE = "agent_direct.jsonl"
DEPTH_SMALL, DEPTH_LARGE = 60, 200  # candidate depths: max(10, 30) * 2 and max(100, 30) * 2


def runs_dir(data_dir: Path) -> Path:
    return data_dir / "runs"


def store(data_dir: Path, name: str, key: str) -> JsonlStore:
    return JsonlStore(runs_dir(data_dir) / name, key=key)


def ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


def turn_key(run: str, request_id: str, turn: int) -> str:
    return f"{run}:{request_id}:{turn}"


# =============================================================================
# Serialization
# =============================================================================


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def serialize_search(response: Any) -> Optional[Dict[str, Any]]:
    if response is None:
        return None
    metadata = response.metadata
    return {
        "ids": [str(r.book.id) for r in response.results],
        "rerank_scores": [r.rerank_score for r in response.results],
        "search_mode": response.search_mode, "degraded": response.degraded,
        "degradation_reason": response.degradation_reason, "notices": list(response.notices),
        "latency_ms": response.latency_ms,
        "timings_ms": dict(metadata.timings_ms) if metadata is not None else {},
    }


def serialize_agent(answer: Any) -> Optional[Dict[str, Any]]:
    if answer is None:
        return None
    return {
        "answer": answer.answer,
        "recommendations": [{"work_id": str(r.book.id), "reason": r.reason} for r in answer.recommendations],
        "steps": [{"tool": s.tool, "arguments": _plain(s.arguments), "result_count": s.result_count,
                   "books": [b.book_id for b in s.books]} for s in answer.steps],
        "dropped_recommendations": answer.dropped_recommendations, "reflections": answer.reflections,
        "model": answer.model,
    }


def serialize_reply(reply: Any) -> Dict[str, Any]:
    """The parts of an AssistantReply the measures need (work ids, not whole books)."""
    return {
        "route": reply.route, "forced_route": reply.forced_route, "message": reply.message,
        "options": list(reply.options), "search_query": reply.search_query,
        "filters": _plain(reply.filters), "clarifications": reply.clarifications,
        "notices": list(reply.notices), "timings_ms": dict(reply.timings_ms), "model": reply.model,
        "usage": _plain(reply.usage),
        "search": serialize_search(reply.search_response),
        "recommendations": [{"work_id": str(r.book.id), "reason": r.reason,
                             "citations": [{"field": c.chunk_id, "snippet": c.snippet} for c in r.citations]}
                            for r in reply.recommendations],
        "agent": serialize_agent(reply.agent_answer),
    }


def reranked(search: Optional[Dict[str, Any]]) -> bool:
    """True when the head of the list carries rerank scores (the reranker answered)."""
    return bool(search and search["ids"] and search["rerank_scores"][0] is not None)


# =============================================================================
# Reference P
# =============================================================================


def p_texts(requests: Sequence[Request]) -> List[str]:
    """Texts P runs on: the A, B, E and F requests (B covers E's original queries)."""
    texts: List[str] = []
    for r in requests:
        if r.set in (D.SET_A, D.SET_B, D.SET_E, D.SET_F) and r.text not in texts:
            texts.append(r.text)
    return texts


def run_reference(system: Any, requests: Sequence[Request], limit: Optional[int] = None) -> None:
    from app.domain.value_objects import SearchQuery

    p_store = store(system.data_dir, P_FILE, "text")
    todo = [t for t in p_texts(requests) if t not in p_store][:limit]

    def one(text: str) -> None:
        attempts: List[Dict[str, Any]] = []
        record: Dict[str, Any] = {}
        for attempt in range(1, D.TURN_ATTEMPTS + 1):
            system.ledger.check()
            started = time.perf_counter()
            try:
                response = system.search.search_with_fallback(SearchQuery(text=text, max_results=D.P_DEPTH))
            except Exception as error:  # noqa: BLE001 - retried, then recorded
                attempts.append({"attempt": attempt, "error": f"{type(error).__name__}: {error}"[:200]})
                continue
            search = serialize_search(response)
            record = {"text": text, **search, "wall_ms": ms(time.perf_counter() - started), "attempts": attempt,
                      "failed_attempts": attempts, "reranked": reranked(search), "commit": system.commit,
                      "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if record["reranked"] and not search["degraded"]:
                break
            attempts.append({"attempt": attempt, "error": "degraded or not reranked"})
        if not record:
            record = {"text": text, "ids": [], "error": attempts, "attempts": D.TURN_ATTEMPTS, "reranked": False,
                      "commit": system.commit, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
        p_store.add(record)

    system.log(f"P: {len(todo)} texts to run")
    with ThreadPoolExecutor(max_workers=D.WORKERS) as pool:
        list(pool.map(one, todo))
    system.log(f"P: {len(p_store.records)} texts stored")


def run_depth_check(system: Any, requests: Sequence[Request], limit: Optional[int] = None) -> None:
    """Database only: are the 30 vector candidates the reranker sees the same at depth 60 and at depth 200?"""
    depth = store(system.data_dir, DEPTH_FILE, "text")
    todo = [t for t in p_texts(requests) if t not in depth][:limit]
    for text in todo:
        vector = system.encoder.generate_query_embedding(text)  # stored by P: no API call
        small = [str(r.book.id) for r in system.vector.search(query_embedding=vector, max_results=DEPTH_SMALL)]
        large = [str(r.book.id) for r in system.vector.search(query_embedding=vector, max_results=DEPTH_LARGE)]
        head = system.reranker_settings.depth
        depth.add({"text": text, "head": head, "same_head_set": set(small[:head]) == set(large[:head]),
                   "same_head_order": small[:head] == large[:head],
                   "head_overlap": len(set(small[:head]) & set(large[:head]))})
    system.log(f"depth check: {len(depth.records)} texts")


# =============================================================================
# Assistant turns
# =============================================================================


def assistant_turn(system: Any, request: Request, run: str, turn: int, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One request to the assistant (ConversationalAssistant.reply), cached in turns.jsonl."""
    from app.domain.value_objects import ChatMessage

    turns = system.turns
    key = turn_key(run, request.request_id, turn)
    if key in turns:
        return turns.get(key)
    errors: List[str] = []
    record: Dict[str, Any] = {}
    for attempt in range(1, D.TURN_ATTEMPTS + 1):
        system.ledger.check()
        capture: Dict[str, List[Any]] = {}
        token = CAPTURE.set(capture)
        started = time.perf_counter()
        try:
            reply = system.assistant.reply([ChatMessage(role=m["role"], content=m["content"], route=m.get("route"))
                                            for m in messages], max_results=D.ASSISTANT_MAX_RESULTS)
        except Exception as error:  # noqa: BLE001 - retried, then recorded
            errors.append(f"{type(error).__name__}: {error}"[:300])
            continue
        finally:
            CAPTURE.reset(token)
        record = {"key": key, "run": run, "request_id": request.request_id, "set": request.set, "turn": turn,
                  "messages": messages, "reply": serialize_reply(reply), "router_raw": capture.get("router", []),
                  "answer_raw": capture.get("answer", []), "llm_calls": capture.get("llm_calls", []),
                  "api_calls": capture.get("api_calls", []), "wall_ms": ms(time.perf_counter() - started),
                  "attempts": attempt, "failed_attempts": errors, "commit": system.commit,
                  "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
        break
    if not record:
        record = {"key": key, "run": run, "request_id": request.request_id, "set": request.set, "turn": turn,
                  "messages": messages, "reply": None, "error": errors, "attempts": D.TURN_ATTEMPTS,
                  "commit": system.commit, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    turns.add(record)
    return record


def run_single_turns(system: Any, requests: Sequence[Request], run: str, limit: Optional[int] = None) -> None:
    todo = [r for r in requests if turn_key(run, r.request_id, 1) not in system.turns][:limit]
    system.log(f"{run}: {len(todo)} single-turn requests to run")

    def one(request: Request) -> None:
        assistant_turn(system, request, run, 1, [{"role": "user", "content": request.text, "route": None}])

    with ThreadPoolExecutor(max_workers=D.WORKERS) as pool:
        list(pool.map(one, todo))


def run_e(system: Any, requests: Sequence[Request], limit: Optional[int] = None) -> None:
    """Set E: one conversation per request, the simulated user answering the assistant's questions."""
    conversations = store(system.data_dir, CONVERSATIONS_FILE, "request_id")
    user = SimulatedUser(runs_dir(system.data_dir) / "simulated_user", system.ledger, system.stage)
    todo = [r for r in by_set(requests, D.SET_E) if r.request_id not in conversations][:limit]
    system.log(f"E: {len(todo)} conversations to run")

    def one(request: Request) -> None:
        user_calls: List[Dict[str, Any]] = []

        def assistant(turn: int, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            record = assistant_turn(system, request, D.RUN_1, turn, messages)
            reply = record.get("reply")
            return None if reply is None else reply

        def answer(turn: int, messages: List[Dict[str, Any]], reply: Dict[str, Any]) -> str:
            shown = messages + [{"role": "assistant", "content": reply.get("message") or "", "route": "clarify"}]
            call = user.answer(f"{request.request_id}-{turn}", request.need, request.language, shown,
                               list(reply.get("options") or []))
            text = call.get("reply") or call.get("chosen_option") or (reply.get("options") or ["?"])[0]
            user_calls.append({"turn": turn, "reply": text, "chosen_option": call.get("chosen_option"),
                               "raw_output": call.get("raw_output"), "model": call.get("model"),
                               "empty_reply_replaced": not call.get("reply")})
            return text

        result = run_conversation(request.text, assistant, answer)
        conversations.add({"request_id": request.request_id, "stop": result["stop"], "questions": result["questions"],
                           "assistant_turns": len(result["replies"]), "messages": result["messages"],
                           "user_calls": user_calls, "date": time.strftime("%Y-%m-%dT%H:%M:%S")})

    with ThreadPoolExecutor(max_workers=D.WORKERS) as pool:
        list(pool.map(one, todo))


def run_agent_direct(system: Any, requests: Sequence[Request], ids: Sequence[str], limit: Optional[int] = None) -> None:
    """The agent called directly (SearchAgent.answer) on the sample of A and B."""
    agent_store = store(system.data_dir, AGENT_FILE, "request_id")
    index = {r.request_id: r for r in requests}
    todo = [index[i] for i in ids if i not in agent_store][:limit]
    system.log(f"agent: {len(todo)} requests to run")

    def one(request: Request) -> None:
        errors: List[str] = []
        record: Dict[str, Any] = {}
        for attempt in range(1, D.TURN_ATTEMPTS + 1):
            system.ledger.check()
            capture: Dict[str, List[Any]] = {}
            token = CAPTURE.set(capture)
            started = time.perf_counter()
            try:
                answer = system.agent.answer(request.text)
            except Exception as error:  # noqa: BLE001
                errors.append(f"{type(error).__name__}: {error}"[:300])
                continue
            finally:
                CAPTURE.reset(token)
            record = {"request_id": request.request_id, "set": request.set, "agent": serialize_agent(answer),
                      "llm_calls": capture.get("llm_calls", []), "api_calls": capture.get("api_calls", []),
                      "wall_ms": ms(time.perf_counter() - started), "attempts": attempt, "failed_attempts": errors,
                      "commit": system.commit, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
            break
        if not record:
            record = {"request_id": request.request_id, "set": request.set, "agent": None, "error": errors,
                      "commit": system.commit, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
        agent_store.add(record)

    with ThreadPoolExecutor(max_workers=D.WORKERS) as pool:
        list(pool.map(one, todo))


def run_part(system: Any, requests: Sequence[Request], part: str, limit: Optional[int] = None) -> None:
    if not hasattr(system, "turns"):
        system.turns = store(system.data_dir, TURNS_FILE, "key")
    if part == "ab1":
        run_single_turns(system, by_set(requests, D.SET_A, D.SET_B), D.RUN_1, limit)
    elif part == "ab2":
        run_single_turns(system, by_set(requests, D.SET_A, D.SET_B), D.RUN_2, limit)
    elif part == "d":
        run_single_turns(system, by_set(requests, D.SET_D), D.RUN_1, limit)
    elif part == "f":
        run_single_turns(system, by_set(requests, D.SET_F), D.RUN_1, limit)
    elif part == "e":
        run_e(system, requests, limit)
    elif part == "agent":
        import json

        meta = json.loads((system.data_dir / "sets" / "requests.json").read_text(encoding="utf-8"))
        run_agent_direct(system, requests, meta["agent_sample"], limit)
    else:
        raise ValueError(f"unknown part {part}")


def load_store(data_dir: Path, name: str, key: str) -> Dict[str, Dict[str, Any]]:
    return store(data_dir, name, key).records


__all__ = ["run_reference", "run_depth_check", "run_part", "serialize_reply", "load_store", "load_requests",
           "turn_key", "reranked", "Callable"]
