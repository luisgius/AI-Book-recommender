"""
API endpoint for agentic search (Phase 4: tool use, ReAct, self-reflection).

The agent decides which catalog tools to call; this module only converts
HTTP input/output and maps errors.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1 import schemas as api
from app.api.v1.converters import domain_book_to_api
from app.api.v1.dependencies import get_search_agent

router = APIRouter()


@router.post("/agent/ask", response_model=api.AgentAskResponse)
def ask_agent(
    request: api.AgentAskRequest,
    agent=Depends(get_search_agent),
) -> api.AgentAskResponse:
    """
    Answer a book request with the LangGraph search agent.

    The agent searches the catalog with tools (hybrid search, book details,
    similar books), possibly several times, and recommends only books its
    tools returned. The response includes the tool-call trace.

    Raises:
        503: No LLM is configured or it cannot be reached
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agentic search needs an LLM (set LLM_PROVIDER and start the model server)",
        )
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
            api.AgentStep(tool=s.tool, arguments=s.arguments, result_count=s.result_count)
            for s in result.steps
        ],
        model=result.model,
        dropped_recommendations=result.dropped_recommendations,
        reflections=result.reflections,
        latency_ms=(time.time() - start) * 1000,
    )
