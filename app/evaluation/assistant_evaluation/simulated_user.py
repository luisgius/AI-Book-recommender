"""
The simulated user of set E (protocol section 3) and the conversation loop.

The user is an LLM of another vendor than the system LLM. It is given the
hidden need, the conversation so far (its own messages and the assistant's
questions) and the assistant's latest question with its suggested answers,
and answers with one short sentence coherent with the need: one of the
options if one fits, otherwise its own words. It never sees a result.

run_conversation is pure: the assistant and the user are functions, so the
loop is tested with fakes. The conversation ends when the assistant answers
with anything but a question (books, the agent, out of scope) or after
E_MAX_ASSISTANT_TURNS assistant turns (a safety stop; the assistant itself
stops asking after two questions).
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from app.evaluation.assistant_evaluation import design as D
from app.evaluation.catalog_evaluation.ledger import SpendLedger
from app.evaluation.catalog_evaluation.queries import parse_json_answer

USER_PROMPT_VERSION = "v1"

USER_PROMPT = """You are role-playing a reader who is chatting with the assistant of a book catalog.

What you are looking for (private, the assistant does not know it): {need}

The conversation so far (JSON, oldest first): {conversation}

The assistant's latest message is a question, with these suggested answers: {options}

Reply as that reader would, in {language_name}: one short sentence that answers the question and is coherent with what you are looking for. If one of the suggested answers fits what you are looking for, reply with it (you may add a few words); if none fits, write your own answer. Do not ask questions back and do not mention that you are role-playing. You have not seen any search result.

Answer with JSON: {{"reply": "...", "chosen_option": "<the suggested answer you chose, or empty if none fits>"}}"""

STOP_ANSWERED = "answered"  # the assistant replied with something other than a question
STOP_MAX_TURNS = "max_turns"
STOP_ERROR = "error"


def user_prompt(need: str, language: str, conversation: List[Mapping[str, Any]], options: List[str]) -> str:
    shown = [{"role": m["role"], "content": m["content"]} for m in conversation]
    return USER_PROMPT.format(need=need, conversation=json.dumps(shown, ensure_ascii=False),
                              options=json.dumps(options, ensure_ascii=False),
                              language_name=D.LANGUAGE_NAMES.get(language, language))


def run_conversation(
    first_message: str,
    assistant: Callable[[int, List[Dict[str, Any]]], Optional[Dict[str, Any]]],
    user: Callable[[int, List[Dict[str, Any]], Dict[str, Any]], str],
    max_turns: int = D.E_MAX_ASSISTANT_TURNS,
) -> Dict[str, Any]:
    """
    Run one conversation.

    Args:
        first_message: The user's first message (the vague request)
        assistant: (turn, messages) -> serialized reply (with "route",
            "message", "options"), or None when the turn failed
        user: (turn, messages, reply) -> the user's answer to the question
        max_turns: Safety stop on assistant turns

    Returns:
        {"messages": the conversation as sent last, "replies": [reply per
        assistant turn], "stop": why it ended, "questions": questions asked}
    """
    messages: List[Dict[str, Any]] = [{"role": "user", "content": first_message, "route": None}]
    replies: List[Dict[str, Any]] = []
    stop = STOP_MAX_TURNS
    for turn in range(1, max_turns + 1):
        reply = assistant(turn, [dict(m) for m in messages])
        if reply is None:
            stop = STOP_ERROR
            break
        replies.append(reply)
        if reply.get("route") != "clarify":
            stop = STOP_ANSWERED
            break
        if turn == max_turns:
            break
        answer = user(turn, [dict(m) for m in messages], reply)
        messages.append({"role": "assistant", "content": reply.get("message") or "clarify", "route": "clarify"})
        messages.append({"role": "user", "content": answer, "route": None})
    return {"messages": messages, "replies": replies, "stop": stop,
            "questions": sum(1 for r in replies if r.get("route") == "clarify")}


class SimulatedUser:
    """Chat calls to the simulated user through OpenRouterHttp, cached on disk per call id."""

    def __init__(self, directory: Path, ledger: SpendLedger, stage: str, http: Any = None,
                 model: str = D.SIMULATED_USER_MODEL) -> None:
        self.directory = directory
        self.ledger = ledger
        self.stage = stage
        self.model = model
        self._http = http

    def _client(self):
        if self._http is None:
            from app.infrastructure.search.openrouter_http import OpenRouterHttp

            self._http = OpenRouterHttp(timeout_s=60.0, max_retries=3)
        return self._http

    def answer(self, call_id: str, need: str, language: str, conversation: List[Mapping[str, Any]],
               options: List[str]) -> Dict[str, Any]:
        """The saved call, or a new one: {"reply", "chosen_option", "raw_output", ...}."""
        path = self.directory / f"{call_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        self.ledger.check()
        messages = [{"role": "user", "content": user_prompt(need, language, list(conversation), options)}]
        body = {"model": self.model, "messages": messages, "temperature": D.SIMULATED_USER_TEMPERATURE,
                "max_tokens": 200, "response_format": {"type": "json_object"}}
        started = time.perf_counter()
        response = self._client().post("chat/completions", body)
        payload = response.body
        usage = payload.get("usage") or {}
        cost = usage.get("cost")
        content = payload["choices"][0]["message"].get("content") or ""
        try:
            parsed = parse_json_answer(content)
            reply = " ".join(str(parsed.get("reply", "")).split())
            chosen = " ".join(str(parsed.get("chosen_option", "")).split())
        except (ValueError, json.JSONDecodeError):
            reply, chosen = " ".join(content.split()), ""
        record = {"call_id": call_id, "model": self.model, "provider": payload.get("provider"),
                  "date": datetime.now(UTC).isoformat(timespec="seconds"), "prompt_version": USER_PROMPT_VERSION,
                  "request": body, "raw_output": content, "reply": reply, "chosen_option": chosen,
                  "usage": usage, "latency_s": round(time.perf_counter() - started, 3)}
        self.ledger.add(self.stage, "simulated_user", self.model, float(cost) if cost is not None else None,
                        input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record
