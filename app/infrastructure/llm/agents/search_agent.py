"""
LangGraph implementation of the SearchAgent port (Phase 4: tool use, ReAct,
self-reflection).

The agent is an explicit StateGraph rather than a prebuilt black box, so every
step can be inspected and explained:

    START -> agent --(tool calls)--> tools --> agent      (ReAct loop)
               |
               +--(no tool calls or round limit)--> finalize -> verify
                                                                  |
                         (no valid recommendation, 1 retry) <-----+--> END
                         back to agent with a reflection message

- agent:    the LLM with the catalog tools bound decides what to call (ReAct).
- tools:    LangGraph ToolNode runs the tools over the domain services.
- finalize: the LLM turns the tool observations (not its own memory) into a
            structured recommendation list.
- verify:   deterministic guardrail. Recommendations whose book_id no tool
            returned are dropped; if nothing valid remains, the agent is sent
            back once to search again (self-reflection).
"""

import json
import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from uuid import UUID

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.domain.entities import AgentAnswer, AgentRecommendation, AgentStep, AgentStepBook, Book
from app.domain.ports import BookCatalogRepository
from app.domain.services import SearchService
from app.domain.value_objects import SearchFilters, SearchQuery
from app.infrastructure.llm.prompts_agent import (
    AGENT_FINALIZE_PROMPT,
    AGENT_REFLECTION_PROMPT,
    AGENT_SYSTEM_PROMPT,
)
from app.infrastructure.llm.structured_output import with_structured_output
from app.infrastructure.llm.usage import operation_config

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
MAX_REFLECTIONS = 1
MAX_RECOMMENDATIONS = 5
MAX_TOOL_RESULTS = 8
DESCRIPTION_PREVIEW_CHARS = 300


# =============================================================================
# Structured output of the finalize node
# =============================================================================


class RecommendedBookLLM(BaseModel):
    """One recommendation as produced by the LLM (verified afterwards)."""

    book_id: str = Field(description="book_id exactly as returned by a tool")
    reason: str = Field(description="One sentence, based only on the book data shown")


class RecommendationListLLM(BaseModel):
    """Final answer of the agent."""

    answer: str = Field(description="Short answer to the user's request")
    recommendations: List[RecommendedBookLLM] = Field(
        default_factory=list, description="Recommended books, best first"
    )


# =============================================================================
# Graph state
# =============================================================================


class AgentState(TypedDict):
    """State carried between the nodes of the agent graph."""

    question: str
    messages: Annotated[List[AnyMessage], add_messages]
    tool_rounds: int
    reflections: int
    final: Optional[RecommendationListLLM]
    dropped: int


# =============================================================================
# Tools over the domain services
# =============================================================================


def _book_to_observation(book: Book, description_chars: int = DESCRIPTION_PREVIEW_CHARS) -> Dict[str, Any]:
    """Compact JSON view of a book for the LLM (what the agent is allowed to know)."""
    description = book.description or ""
    if len(description) > description_chars:
        description = description[: description_chars - 3] + "..."
    return {
        "book_id": str(book.id),
        "title": book.title,
        "authors": book.authors,
        "year": book.get_published_year(),
        "language": book.language,
        "categories": book.categories,
        "description": description,
    }


def build_catalog_tools(search_service: SearchService, catalog_repo: BookCatalogRepository) -> list:
    """
    Create the LangChain tools the agent can call.

    Each tool wraps a domain operation and returns JSON, so the observations
    are both readable by the LLM and parseable by the verify node.
    """

    @tool
    def search_books(
        query: str,
        language: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> str:
        """Search the catalog for books matching the query (search by meaning, in any language).

        Args:
            query: What to look for, e.g. "spanish civil war novel" or "recetas de pasta".
            language: Optional ISO 639-1 code ("es", "en"); only if the user asked for it.
            min_year: Optional earliest publication year.
            max_year: Optional latest publication year.
        """
        filters = SearchFilters(language=language.lower() if language else None,
                                min_year=min_year, max_year=max_year)
        response = search_service.search_with_fallback(
            SearchQuery(text=query, filters=filters, max_results=MAX_TOOL_RESULTS)
        )
        return json.dumps([_book_to_observation(r.book) for r in response.results], ensure_ascii=False)

    @tool
    def get_book_details(book_id: str) -> str:
        """Get the full record of one book (complete description) by its book_id."""
        book = catalog_repo.get_by_id(UUID(book_id))
        if book is None:
            return json.dumps({"error": f"No book with id {book_id}"})
        return json.dumps(_book_to_observation(book, description_chars=1500), ensure_ascii=False)

    @tool
    def find_similar_books(book_id: str) -> str:
        """Find books similar to a given book (by meaning), using its book_id."""
        results = search_service.find_similar_books(UUID(book_id), max_results=MAX_TOOL_RESULTS)
        return json.dumps([_book_to_observation(r.book) for r in results], ensure_ascii=False)

    return [search_books, get_book_details, find_similar_books]


def _tool_text(message: ToolMessage) -> str:
    """Tool output as text (ToolNode returns strings; content blocks are joined)."""
    if isinstance(message.content, str):
        return message.content
    return "".join(part if isinstance(part, str) else str(part.get("text", "")) for part in message.content)


def _observed_books(messages: List[AnyMessage]) -> Dict[str, Dict[str, Any]]:
    """Collect every book returned by a tool call, keyed by book_id."""
    observed: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(_tool_text(message))
        except (TypeError, json.JSONDecodeError):
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict) and "book_id" in item:
                observed[item["book_id"]] = item
    return observed


def _returned_books(message: ToolMessage) -> Optional[List[AgentStepBook]]:
    """The books in one tool output (None when the output is not the tools' JSON)."""
    try:
        payload = json.loads(_tool_text(message))
    except (TypeError, json.JSONDecodeError):
        return None
    items = payload if isinstance(payload, list) else [payload]
    return [
        AgentStepBook(book_id=str(item["book_id"]), title=str(item.get("title") or ""))
        for item in items
        if isinstance(item, dict) and "book_id" in item
    ]


def _tool_steps(messages: List[AnyMessage]) -> List[AgentStep]:
    """Rebuild the tool-call trace (tool, arguments, books returned)."""
    books_by_call: Dict[str, List[AgentStepBook]] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            books_by_call[message.tool_call_id] = _returned_books(message) or []
    steps: List[AgentStep] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                books = books_by_call.get(call.get("id") or "", [])
                steps.append(AgentStep(tool=call["name"], arguments=dict(call["args"]),
                                       result_count=len(books), books=books))
    return steps


# =============================================================================
# The agent
# =============================================================================


class LangGraphSearchAgent:
    """SearchAgent implemented as an explicit LangGraph ReAct graph with verification."""

    def __init__(
        self,
        model,
        search_service: SearchService,
        catalog_repo: BookCatalogRepository,
        model_name: str = "unknown",
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        max_reflections: int = MAX_REFLECTIONS,
    ) -> None:
        """
        Args:
            model: LangChain chat model supporting bind_tools and with_structured_output
            search_service: Domain search service used by the search tools
            catalog_repo: Catalog used for book details and to hydrate results
            model_name: Model identifier reported in the answer
            max_tool_rounds: Maximum agent turns that call tools
            max_reflections: Maximum re-searches after a failed verification
        """
        self._catalog = catalog_repo
        self._model_name = model_name
        self._max_tool_rounds = max_tool_rounds
        self._max_reflections = max_reflections
        self._tools = build_catalog_tools(search_service, catalog_repo)
        self._model_with_tools = model.bind_tools(self._tools)
        self._finalizer = with_structured_output(model, RecommendationListLLM)
        self.graph = self._build_graph()

    # -- nodes -----------------------------------------------------------------

    def _agent_node(self, state: AgentState) -> dict:
        response = self._model_with_tools.invoke([SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]])
        return {
            "messages": [response],
            "tool_rounds": state["tool_rounds"] + (1 if response.tool_calls else 0),
        }

    def _route_after_agent(self, state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        if wants_tools and state["tool_rounds"] <= self._max_tool_rounds:
            return "tools"
        return "finalize"

    def _finalize_node(self, state: AgentState) -> dict:
        observed = _observed_books(state["messages"])
        prompt = AGENT_FINALIZE_PROMPT.format(
            question=state["question"],
            observations=json.dumps(list(observed.values()), ensure_ascii=False, indent=1) if observed else "[]",
            max_recommendations=MAX_RECOMMENDATIONS,
        )
        final = self._finalizer.invoke([HumanMessage(prompt)])
        return {"final": final}

    def _verify_node(self, state: AgentState) -> dict:
        observed_ids = set(_observed_books(state["messages"]))
        final = state["final"]
        if final is None:
            raise RuntimeError("verify reached without a finalized answer")
        valid, seen = [], set()
        for rec in final.recommendations:
            if rec.book_id in observed_ids and rec.book_id not in seen:
                valid.append(rec)
                seen.add(rec.book_id)
        dropped = len(final.recommendations) - len(valid)

        if not valid and state["reflections"] < self._max_reflections:
            logger.info("Agent verification failed (%s dropped); reflecting once", dropped)
            return {
                "messages": [HumanMessage(AGENT_REFLECTION_PROMPT)],
                "reflections": state["reflections"] + 1,
                "final": None,
                "dropped": state["dropped"] + dropped,
            }
        trimmed = RecommendationListLLM(answer=final.answer, recommendations=valid[:MAX_RECOMMENDATIONS])
        return {"final": trimmed, "dropped": state["dropped"] + dropped}

    def _route_after_verify(self, state: AgentState) -> str:
        return "agent" if state["final"] is None else END

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self._tools))
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("verify", self._verify_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._route_after_agent, {"tools": "tools", "finalize": "finalize"})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("finalize", "verify")
        workflow.add_conditional_edges("verify", self._route_after_verify, {"agent": "agent", END: END})
        return workflow.compile()

    # -- port ------------------------------------------------------------------

    def answer(self, question: str) -> AgentAnswer:
        """Run the graph for one request (SearchAgent port)."""
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        initial: AgentState = {
            "question": question,
            "messages": [HumanMessage(question)],
            "tool_rounds": 0,
            "reflections": 0,
            "final": None,
            "dropped": 0,
        }
        recursion_limit = 4 * (self._max_tool_rounds + 2) * (self._max_reflections + 1)
        try:
            # The "agent" label reaches every LLM call of the graph (usage records)
            state = self.graph.invoke(
                initial, config={"recursion_limit": recursion_limit, **operation_config("agent")}
            )
        except ConnectionError as e:
            raise RuntimeError(f"LLM unreachable: {e}") from e

        final = state["final"]
        if final is None:  # the graph always ends after verify sets a final answer
            raise RuntimeError("Agent graph ended without a final answer")
        recommendations = []
        for rec in final.recommendations:
            book = self._catalog.get_by_id(UUID(rec.book_id))
            if book is not None:
                recommendations.append(AgentRecommendation(book=book, reason=rec.reason))

        return AgentAnswer(
            question=question,
            answer=final.answer,
            recommendations=recommendations,
            steps=_tool_steps(state["messages"]),
            model=self._model_name,
            dropped_recommendations=state["dropped"],
            reflections=state["reflections"],
        )
