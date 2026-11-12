"""
API endpoint of the conversational assistant (the LangGraph entry point).

The client sends the whole conversation on every request; the server stores
nothing. The assistant decides whether to ask one clarifying question,
search, hand over to the agent, or say the request is out of scope.

Without an LLM, or with the day's LLM budget spent, the endpoint does not
fail: it runs the plain search of the last user message and says why in
`notices` ("llm_unavailable" or "llm_budget_spent"), as POST /search does.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.rate_limiting import LLM
from app.api.v1 import schemas as api
from app.api.v1.converters import api_messages_to_domain, domain_reply_to_api
from app.api.v1.dependencies import (
    get_assistant,
    get_rest_rate_limits,
    get_search_service,
    get_usage_budget,
)
from app.domain.ports import BUDGET_LLM_REQUEST
from app.domain.services import NOTICE_LLM_BUDGET, NOTICE_LLM_UNAVAILABLE, plain_search_reply
from app.domain.value_objects import bound_conversation, last_user_message
from app.infrastructure.llm.prompts_assistant import fixed_reply, guess_language

router = APIRouter()

DEFAULT_MAX_RESULTS = 10


def take_llm_unit() -> bool:
    """
    One unit of the day's LLM budget for this request (V17).

    Returns:
        True without a budget or when granted; False when spent or when the
        budget cannot be read (fail closed: no LLM call without a counted unit)
    """
    budget = get_usage_budget()
    if budget is None:
        return True
    try:
        return bool(budget.try_consume(BUDGET_LLM_REQUEST))
    except Exception:  # noqa: BLE001
        return False


@router.post("/assistant/chat", response_model=api.AssistantChatResponse)
def assistant_chat(
    request: api.AssistantChatRequest,
    http_request: Request,
    assistant=Depends(get_assistant),
    service=Depends(get_search_service),
) -> api.AssistantChatResponse:
    """
    Reply to the last user message of a conversation.

    Each request counts against the per-client LLM rate limit and takes one
    unit of the day's LLM budget, whatever route it takes.

    Raises:
        400: The conversation does not end with a user message
        422: More than 50 messages, or a message over 5000 characters
        429: The client is over its LLM rate limit
        503: The search is unavailable
    """
    get_rest_rate_limits().enforce(LLM, http_request)
    start = time.time()
    messages = api_messages_to_domain(request.messages)
    max_results = request.max_results or DEFAULT_MAX_RESULTS
    try:
        if assistant is not None and take_llm_unit():
            reply = assistant.reply(messages, max_results=max_results)
        else:
            notice = NOTICE_LLM_UNAVAILABLE if assistant is None else NOTICE_LLM_BUDGET
            last = last_user_message(bound_conversation(messages))
            message = fixed_reply("plain_search", guess_language(last), last)
            reply = plain_search_reply(service, messages, max_results, message=message, notice=notice)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    return domain_reply_to_api(reply, latency_ms=(time.time() - start) * 1000)
