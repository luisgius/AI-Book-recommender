"""
LangGraph implementation of the ConversationalAssistant port: the entry point of LLM requests.

One StateGraph routes each request to one of four paths:

    START -> route --clarify------> clarify ---------------------------> END
                   --search-------> retrieve -> answer -> verify ------> END
                   --agent--------> agent --(agent failed)--> retrieve
                                          --(answered)----------------> END
                   --out_of_scope-> out_of_scope ----------------------> END

- route:        one LLM call with strict structured output (RouteDecisionLLM),
                then deterministic rules (apply_routing_rules): at most
                MAX_CLARIFYING_TURNS questions per conversation, a reply to a
                question is out of scope only if the model labels it
                "unrelated" (not a non-answer), filters only when the user's
                own words state them, and invalid output falls back to a
                search of the last message.
- clarify:      returns the question and its suggested answers. No search.
- retrieve:     the evaluated search pipeline (SearchService), always run on
                this route: retrieval is a fixed node, not a tool to skip.
- answer:       one LLM call that recommends up to three retrieved books with
                verbatim citations (RAG generation step).
- verify:       deterministic guardrail. Picks that are not among the shown
                books are dropped; citations go through the same check as the
                explanations (guardrails.filter_valid_citations); a pick with
                no valid citation is dropped.
- agent:        hands the condensed request to the tool-calling SearchAgent.
- out_of_scope: a fixed reply. No search, no further LLM call.

The server is stateless: the conversation arrives with each request, is
bounded (value_objects.bound_conversation) and nothing is stored. There is no
checkpointer. Every LLM call is recorded by the LLM client's usage recorder
under the operation "assistant" with the node name as step (the agent's
calls keep the operation "agent"); a per-request recorder adds the tokens and
cost of this request to the reply, and each node's time is returned.
"""

import logging
import operator
import time
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain.entities import AgentAnswer, AssistantRecommendation, AssistantReply, Book
from app.domain.ports import SearchAgent
from app.domain.services import SearchService
from app.domain.value_objects import (
    MAX_CLARIFYING_TURNS,
    ROUTE_AGENT,
    ROUTE_CLARIFY,
    ROUTE_OUT_OF_SCOPE,
    ROUTE_SEARCH,
    ChatMessage,
    SearchFilters,
    SearchQuery,
    SearchResponse,
    bound_conversation,
    count_clarifications,
    last_user_message,
)
from app.infrastructure.llm.agents.search_agent import _book_to_observation
from app.infrastructure.llm.filter_guards import guard_filters
from app.infrastructure.llm.guardrails import convert_citation_to_domain, filter_valid_citations
from app.infrastructure.llm.prompts_assistant import (
    build_answer_messages,
    build_router_messages,
    fixed_reply,
    guess_language,
    reply_language,
)
from app.infrastructure.llm.schemas_assistant import (
    MAX_OPTIONS,
    MAX_PICKS,
    GroundedAnswerLLM,
    RouteDecisionLLM,
)
from app.infrastructure.llm.structured_output import with_structured_output
from app.infrastructure.llm.usage import LLMUsageRecorder, operation_config

logger = logging.getLogger(__name__)

# Retrieved books shown to the answer call (the rest are returned, not discussed).
ANSWER_CONTEXT_BOOKS = 6
# Characters of each description the answer call sees.
ANSWER_DESCRIPTION_CHARS = 600
# Token cap of the routing call. Its output is under 300 tokens; the cap
# makes a model that loops (whitespace, repetition) fail in about a second
# instead of running to the client's cap, and the fallback then searches.
ROUTER_MAX_TOKENS = 400
# Plausible publication years for a filter the router extracted.
MIN_FILTER_YEAR = 1000
MAX_FILTER_YEAR = 2100

ASSISTANT_OPERATION = "assistant"

# Reasons recorded in AssistantReply.forced_route when code overrides the model.
FORCED_INVALID_OUTPUT = "invalid_router_output"
FORCED_CLARIFICATION_CAP = "clarification_cap"
FORCED_INVALID_CLARIFICATION = "invalid_clarification"
FORCED_REPLY_TO_CLARIFICATION = "reply_to_clarification"
FORCED_AGENT_FAILED = "agent_failed"

# Notices added to the reply (codes, like SearchResponse.notices).
NOTICE_ANSWER_FAILED = "assistant_answer_failed"
NOTICE_AGENT_FAILED = "agent_failed"


def _merge(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    """Reducer of the timings: each node adds its own entry."""
    return {**left, **right}


class AssistantState(TypedDict, total=False):
    """State carried between the nodes of the assistant graph."""

    conversation: List[ChatMessage]
    clarifications: int
    max_results: int
    # set by route
    route: str
    forced_route: Optional[str]
    language: str
    search_query: str
    filters: SearchFilters
    question: str
    options: List[str]
    # set by the route-specific nodes
    message: str
    search_response: Optional[SearchResponse]
    answer: Optional[GroundedAnswerLLM]
    recommendations: List[AssistantRecommendation]
    agent_answer: Optional[AgentAnswer]
    notices: Annotated[List[str], operator.add]
    timings_ms: Annotated[Dict[str, float], _merge]


# =============================================================================
# Deterministic rules applied to the model's routing output
# =============================================================================


def _user_text(conversation: List[ChatMessage]) -> str:
    """Everything the user wrote, oldest first (the words filters must come from)."""
    return " ".join(m.content for m in conversation if m.role == "user")


def _answers_a_clarification(conversation: List[ChatMessage]) -> bool:
    """True if the message before the last user message is a clarifying question."""
    if len(conversation) < 2 or conversation[-2].role != "assistant":
        return False
    return conversation[-2].route in (None, ROUTE_CLARIFY)


def validated_filters(decision: RouteDecisionLLM, user_text: str) -> SearchFilters:
    """
    The filters of a routing decision that the user's own words state.

    Language: a two-letter code the user asked for ("en español", "in
    English"); years: plausible, in order, and only if the user wrote digits.
    The rules are those of the query understanding path (filter_guards.py).
    """
    language = (decision.language or "").strip().lower() or None
    if language is not None and not (len(language) == 2 and language.isalpha()):
        language = None

    def plausible(year: Optional[int]) -> Optional[int]:  # 0 means "not given"
        return year if year is not None and MIN_FILTER_YEAR <= year <= MAX_FILTER_YEAR else None

    min_year, max_year = plausible(decision.min_year), plausible(decision.max_year)
    if min_year is not None and max_year is not None and min_year > max_year:
        min_year = max_year = None
    return guard_filters(SearchFilters(language=language, min_year=min_year, max_year=max_year), user_text)


def apply_routing_rules(
    decision: Optional[RouteDecisionLLM],
    conversation: List[ChatMessage],
    clarifications: int,
    max_clarifications: int = MAX_CLARIFYING_TURNS,
) -> Dict[str, Any]:
    """
    Turn the model's routing output into the route the graph takes.

    Args:
        decision: Parsed routing output, or None when the call failed or the
            output was invalid
        conversation: The bounded conversation
        clarifications: Clarifying questions already asked in it
        max_clarifications: The cap on clarifying questions

    Returns:
        State update: route, forced_route, language, search_query, filters,
        question, options
    """
    last = last_user_message(conversation)
    if decision is None:
        return {
            "route": ROUTE_SEARCH, "forced_route": FORCED_INVALID_OUTPUT, "language": guess_language(last),
            "search_query": last, "filters": SearchFilters(), "question": "", "options": [],
        }

    route = decision.route
    forced: Optional[str] = None
    query = " ".join(decision.search_query.split())
    question = " ".join((decision.question or "").split())
    options: List[str] = []
    for option in decision.options:
        option = " ".join(option.split())
        if option and option not in options:
            options.append(option)
    options = options[:MAX_OPTIONS]

    if route == ROUTE_CLARIFY:
        if clarifications >= max_clarifications:
            route, forced = ROUTE_SEARCH, FORCED_CLARIFICATION_CAP
        elif not question or len(options) < 2:
            route, forced = ROUTE_SEARCH, FORCED_INVALID_CLARIFICATION
        if forced is not None and not query:
            query = _user_text(conversation)  # whatever is known so far
    # A reply to a question is out of scope only when it plainly asks about
    # something else: "I don't know" (non_answer) searches with what is known.
    if (route == ROUTE_OUT_OF_SCOPE and _answers_a_clarification(conversation)
            and decision.last_message_kind != "unrelated"):
        route, forced = ROUTE_SEARCH, FORCED_REPLY_TO_CLARIFICATION
        query = query or _user_text(conversation)
    if route in (ROUTE_SEARCH, ROUTE_AGENT) and not query:
        query = last

    return {
        "route": route,
        "forced_route": forced,
        "language": reply_language(decision.reply_language, last),
        "search_query": query if route in (ROUTE_SEARCH, ROUTE_AGENT) else "",
        "filters": validated_filters(decision, _user_text(conversation)) if route == ROUTE_SEARCH else SearchFilters(),
        "question": question if route == ROUTE_CLARIFY else "",
        "options": options if route == ROUTE_CLARIFY else [],
    }


def verify_picks(answer: GroundedAnswerLLM, shown: Dict[str, Book]) -> List[AssistantRecommendation]:
    """
    Keep the picks that name a shown book and have at least one verified citation.

    Citations are checked with the explanations' guardrail: each snippet must
    be a substring (after normalising case and spaces) of the cited field of
    that book.
    """
    recommendations: List[AssistantRecommendation] = []
    seen = set()
    for pick in answer.picks:
        book = shown.get(pick.book_id.strip())
        if book is None or pick.book_id in seen:
            continue
        valid = filter_valid_citations(pick.citations, book)
        if not valid:
            continue
        seen.add(pick.book_id)
        recommendations.append(AssistantRecommendation(
            book=book,
            reason=" ".join(pick.reason.split()),
            citations=[convert_citation_to_domain(c, book.id) for c in valid],
        ))
    return recommendations[:MAX_PICKS]


# =============================================================================
# The graph
# =============================================================================


class LangGraphAssistant:
    """ConversationalAssistant implemented as an explicit LangGraph router graph."""

    def __init__(
        self,
        model,
        search_service: SearchService,
        agent: Optional[SearchAgent],
        model_name: str = "unknown",
        max_clarifications: int = MAX_CLARIFYING_TURNS,
    ) -> None:
        """
        Args:
            model: LangChain chat model supporting with_structured_output
            search_service: The evaluated search pipeline (retrieve node)
            agent: The tool-calling agent for multi-step requests; None sends
                those requests to the search route
            model_name: Model identifier reported in the reply
            max_clarifications: Cap on clarifying questions per conversation
        """
        self._search = search_service
        self._agent = agent
        self._model_name = model_name
        self._max_clarifications = max_clarifications
        self._router = with_structured_output(_with_max_tokens(model, ROUTER_MAX_TOKENS), RouteDecisionLLM)
        self._answerer = with_structured_output(model, GroundedAnswerLLM)
        self.graph = self._build_graph()

    # -- nodes -----------------------------------------------------------------

    def _route_node(self, state: AssistantState) -> dict:
        conversation = state["conversation"]
        left = max(0, self._max_clarifications - state["clarifications"])
        try:
            decision = self._router.invoke(build_router_messages(conversation, left))
            if not isinstance(decision, RouteDecisionLLM):
                raise ValueError(f"unexpected routing output {type(decision).__name__}")
        except Exception as e:  # noqa: BLE001 - any failure degrades to plain search
            logger.warning(f"Routing failed, searching the last message: {e}")
            decision = None
        update = apply_routing_rules(decision, conversation, state["clarifications"], self._max_clarifications)
        if update["route"] == ROUTE_AGENT and self._agent is None:
            update["route"] = ROUTE_SEARCH
        return update

    def _clarify_node(self, state: AssistantState) -> dict:
        return {"message": state["question"]}

    def _retrieve_node(self, state: AssistantState) -> dict:
        response = self._search.search_with_fallback(SearchQuery(
            text=state["search_query"], filters=state["filters"], max_results=state["max_results"],
        ))
        return {"search_response": response, "notices": list(response.notices)}

    def _answer_node(self, state: AssistantState) -> dict:
        results = state["search_response"].results
        if not results:
            return {"answer": None}
        books = [_book_to_observation(r.book, ANSWER_DESCRIPTION_CHARS) for r in results[:ANSWER_CONTEXT_BOOKS]]
        try:
            answer = self._answerer.invoke(
                build_answer_messages(state["search_query"], books, state["language"], MAX_PICKS)
            )
            if not isinstance(answer, GroundedAnswerLLM):
                raise ValueError(f"unexpected answer output {type(answer).__name__}")
        except Exception as e:  # noqa: BLE001 - the results are still returned
            logger.warning(f"Assistant answer failed, returning the results only: {e}")
            return {"answer": None, "notices": [NOTICE_ANSWER_FAILED]}
        return {"answer": answer}

    def _verify_node(self, state: AssistantState) -> dict:
        results = state["search_response"].results
        query, language = state["search_query"], state["language"]
        if not results:
            return {"message": fixed_reply("no_results", language, query), "recommendations": []}
        answer = state.get("answer")
        shown = {str(r.book.id): r.book for r in results[:ANSWER_CONTEXT_BOOKS]}
        recommendations = verify_picks(answer, shown) if answer is not None else []
        if not recommendations:
            return {"message": fixed_reply("results_only", language, query), "recommendations": []}
        return {"message": " ".join(answer.message.split()), "recommendations": recommendations}

    def _agent_node(self, state: AssistantState) -> dict:
        try:
            answer = self._agent.answer(state["search_query"])
        except Exception as e:  # noqa: BLE001 - degrade to the search route
            logger.warning(f"Agent failed, using the search route: {e}")
            return {"route": ROUTE_SEARCH, "forced_route": FORCED_AGENT_FAILED, "notices": [NOTICE_AGENT_FAILED]}
        recommendations = [
            AssistantRecommendation(book=r.book, reason=r.reason) for r in answer.recommendations
        ]
        return {"agent_answer": answer, "message": answer.answer, "recommendations": recommendations}

    def _out_of_scope_node(self, state: AssistantState) -> dict:
        return {"message": fixed_reply("out_of_scope", state["language"])}

    # -- edges -----------------------------------------------------------------

    @staticmethod
    def _after_route(state: AssistantState) -> str:
        return state["route"]

    @staticmethod
    def _after_agent(state: AssistantState) -> str:
        return "retrieve" if state["route"] == ROUTE_SEARCH else END

    def _build_graph(self):
        def timed(name: str, node):
            def run(state: AssistantState) -> dict:
                started = time.perf_counter()
                update = node(state)
                return {**update, "timings_ms": {name: round((time.perf_counter() - started) * 1000, 1)}}
            return run

        workflow = StateGraph(AssistantState)
        workflow.add_node("route", timed("route", self._route_node))
        workflow.add_node("clarify", timed("clarify", self._clarify_node))
        workflow.add_node("retrieve", timed("retrieve", self._retrieve_node))
        workflow.add_node("answer", timed("answer", self._answer_node))
        workflow.add_node("verify", timed("verify", self._verify_node))
        workflow.add_node("agent", timed("agent", self._agent_node))
        workflow.add_node("out_of_scope", timed("out_of_scope", self._out_of_scope_node))

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges("route", self._after_route, {
            ROUTE_CLARIFY: "clarify",
            ROUTE_SEARCH: "retrieve",
            ROUTE_AGENT: "agent",
            ROUTE_OUT_OF_SCOPE: "out_of_scope",
        })
        workflow.add_edge("clarify", END)
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", "verify")
        workflow.add_edge("verify", END)
        workflow.add_conditional_edges("agent", self._after_agent, {"retrieve": "retrieve", END: END})
        workflow.add_edge("out_of_scope", END)
        return workflow.compile()

    def mermaid(self) -> str:
        """The graph as Mermaid text (for the report)."""
        return self.graph.get_graph().draw_mermaid()

    # -- port ------------------------------------------------------------------

    def reply(self, messages: List[ChatMessage], max_results: int = 10) -> AssistantReply:
        """Run the graph for one request (ConversationalAssistant port)."""
        conversation = bound_conversation(messages)  # raises ValueError if nothing to reply to
        clarifications = count_clarifications(messages)
        request_usage = LLMUsageRecorder(default_operation=ASSISTANT_OPERATION)
        initial: AssistantState = {
            "conversation": conversation,
            "clarifications": clarifications,
            "max_results": max_results,
            "notices": [],
            "timings_ms": {},
            "recommendations": [],
        }
        config = {**operation_config(ASSISTANT_OPERATION), "callbacks": [request_usage]}
        state = self.graph.invoke(initial, config=config)

        route = state["route"]
        return AssistantReply(
            route=route,
            message=state.get("message", ""),
            options=state.get("options", []) if route == ROUTE_CLARIFY else [],
            search_query=state.get("search_query") or None,
            filters=state.get("filters") if route == ROUTE_SEARCH else None,
            search_response=state.get("search_response"),
            recommendations=state.get("recommendations", []),
            agent_answer=state.get("agent_answer"),
            notices=list(dict.fromkeys(state.get("notices", []))),
            timings_ms=dict(state.get("timings_ms", {})),
            forced_route=state.get("forced_route"),
            clarifications=clarifications + (1 if route == ROUTE_CLARIFY else 0),
            model=self._model_name,
            usage=request_usage_summary(request_usage),
        )


def _with_max_tokens(model, max_tokens: int):
    """
    A copy of a LangChain chat model with another token cap.

    Chat models are Pydantic models; model_copy keeps the client, the
    callbacks (usage recorder) and every other option. Objects without a
    max_tokens field (test doubles, other clients) are returned unchanged.
    """
    fields = getattr(type(model), "model_fields", {})
    if "max_tokens" not in fields:
        return model
    return model.model_copy(update={"max_tokens": max_tokens})


def request_usage_summary(recorder: LLMUsageRecorder) -> Dict[str, Any]:
    """
    Calls, tokens and reported cost of one request's LLM calls.

    Sums over the calls that reported a value; None when none did. A failed
    call reports nothing, so `failed_calls` says how many sums are partial.
    """
    records = recorder.records
    def total(values):
        values = [v for v in values if v is not None]
        return sum(values) if values else None
    return {
        "llm_calls": len(records),
        "failed_calls": sum(1 for r in records if r.error is not None),
        "input_tokens": total(r.input_tokens for r in records),
        "output_tokens": total(r.output_tokens for r in records),
        "cost_usd": total(r.cost_usd for r in records),
        "by_node": [
            {"operation": r.operation, "step": r.step, "latency_s": round(r.latency_s, 3),
             "cost_usd": r.cost_usd, "error": r.error}
            for r in records
        ],
    }
