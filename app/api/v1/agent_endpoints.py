"""
API endpoint for agentic search (Phase 4: tool use, ReAct, self-reflection).

The agent decides which catalog tools to call; this module only converts
HTTP input/output and maps errors.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.rate_limiting import LLM
from app.api.v1 import schemas as api
from app.api.v1.converters import domain_agent_step_to_api, domain_book_to_api
from app.api.v1.dependencies import get_rest_rate_limits, get_search_agent, get_usage_budget
from app.domain.ports import BUDGET_LLM_REQUEST

router = APIRouter()

AGENT_BUDGET_MESSAGE = (
    "The day's budget of LLM requests of this public demo is spent, so the agent is not "
    "available until 00:00 UTC. Plain search still works."
)

AGENT_NO_LLM_MESSAGE = (
    "The agent needs an LLM and none is configured on this server (set OPENROUTER_API_KEY, "
    "see .env.example). Plain search still works."
)


def require_llm_allowance(http_request: Request) -> None:
    """
    Per-client LLM rate limit and one unit of the day's LLM budget (V17).

    Raises:
        HTTPException: 429 when the client is over LLM_RATE_LIMIT_PER_MINUTE
            or the day's LLM budget is spent (a budget that cannot be read
            counts as spent)
    """
    get_rest_rate_limits().enforce(LLM, http_request)
    budget = get_usage_budget()
    if budget is None:
        return
    try:
        granted = budget.try_consume(BUDGET_LLM_REQUEST)
    except Exception:  # noqa: BLE001 - fail closed: no LLM call without a counted unit
        granted = False
    if not granted:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=AGENT_BUDGET_MESSAGE)


@router.post("/agent/ask", response_model=api.AgentAskResponse, dependencies=[Depends(require_llm_allowance)])
def ask_agent(
    request: api.AgentAskRequest,
    agent=Depends(get_search_agent),
) -> api.AgentAskResponse:
    """
    Answer a book request with the LangGraph search agent.

    The agent searches the catalog with tools (the search, book details,
    similar books), possibly several times, and recommends only books its
    tools returned. The response includes the tool-call trace.

    Raises:
        429: The client is over its LLM rate limit, or the day's LLM budget is spent
        503: No LLM is configured or it cannot be reached
    """
    if agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=AGENT_NO_LLM_MESSAGE)
    start = time.time()
    try:
        result = agent.answer(request.question)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    return api.AgentAskResponse(
        question=result.question,
        answer=result.answer,
        recommendations=[
            api.AgentRecommendation(book=domain_book_to_api(r.book), reason=r.reason)
            for r in result.recommendations
        ],
        steps=[
            domain_agent_step_to_api(s)
            for s in result.steps
        ],
        model=result.model,
        dropped_recommendations=result.dropped_recommendations,
        reflections=result.reflections,
        latency_ms=(time.time() - start) * 1000,
    )
