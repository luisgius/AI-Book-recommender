"""
LangGraph StateGraph for query understanding.

This module implements a multi-step flow that:
1. Classifies query intent (recommendation, factual, exploratory)
2. Extracts filters from natural language (language, category, year)
3. Reformulates the query for better retrieval
4. Applies intent-specific post-processing

Flow:
    START 
    → parse_intent 
    → extract_filters 
    → reformulate_query 
    → [conditional routing based on intent_type]
    → handle_recommendation / handle_factual / handle_exploratory
    → END

The flow uses graceful degradation: if any node fails, it sets the error
field in state and continues with fallback values.

Architecture:
- Uses LangChain's with_structured_output() for Pydantic validation
- Returns final_query (optimized) or original_query (fallback)
- Intent-specific nodes can customize query processing per intent type
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from ..schemas_query import (
    QueryIntentLLM,
    ExtractedFiltersLLM,
    ReformulatedQueryLLM
)
from ..prompts_query import (
    INTENT_EXTRACTION_PROMPT,
    FILTER_EXTRACTION_PROMPT,
    QUERY_REFORMULATION_PROMPT
)

class QueryUnderstandingState(TypedDict):
    """
    State for the query understanding LangGraph flow.

    Fields are updated by nodes as the flow progresses:
    - parse_intent node → updates 'intent'
    - extract_filters node → updates 'filters'
    - reformulate_query node → updates 'reformulated'
    - intent-specific nodes → update 'final_query' and 'query_variations'

    If any node fails, it sets 'error' instead of its target field.
    """
    original_query: str
    intent: QueryIntentLLM | None
    filters: ExtractedFiltersLLM | None
    reformulated: ReformulatedQueryLLM | None
    final_query: str | None
    query_variations: list[str]
    error: str | None


def parse_intent_node(state: QueryUnderstandingState, llm) -> dict:
    """
    Node 1: Classify query intent.

    Returns:
        {"intent": QueryIntentLLM} on success
        {"intent": None, "error": "..."} on failure
    """
    try:

        query = state["original_query"]

        prompt_text = INTENT_EXTRACTION_PROMPT.format(query = query)

        llm_with_structure = llm.with_structured_output(QueryIntentLLM)
        intent_result = llm_with_structure.invoke(prompt_text)

        return {"intent":intent_result}
    
    except Exception as e:
        # Graceful degradation 
        return {
            "intent": None,
            "error": f"Intent parsing failed: {str(e)}"
        }

def extract_filters_node(state: QueryUnderstandingState, llm) -> dict:
    """
    Node 2: Extract filters from natural language.
    
    Returns:
        {"filters": ExtractedFiltersLLM} on success
        {"filters": None, "error": "..."} on failure
    """
    try:
        query = state["original_query"]
        prompt_text = FILTER_EXTRACTION_PROMPT.format(query=query)
        
        llm_with_structure = llm.with_structured_output(ExtractedFiltersLLM)
        filters_result = llm_with_structure.invoke(prompt_text)
        
        return {"filters": filters_result}
        
    except Exception as e:
        return {
            "filters": None,
            "error": f"Filter extraction failed: {str(e)}"
        }

def reformulate_query_node(state: QueryUnderstandingState, llm) -> dict:
    """
    Node 3: Reformulate query for better retrieval.

    Returns:
        {"reformulated": ReformulatedQueryLLM} on success
        {"reformulated": None, "error": "..."} on failure
    """
    try:
        query = state["original_query"]
        prompt_text = QUERY_REFORMULATION_PROMPT.format(query=query)

        llm_with_structure = llm.with_structured_output(ReformulatedQueryLLM)
        reformulated_result = llm_with_structure.invoke(prompt_text)

        return {"reformulated": reformulated_result}

    except Exception as e:
        return {
            "reformulated": None,
            "error": f"Query reformulation failed: {str(e)}"
        }


# ==============================================================================
# Intent-Specific Handler Nodes
# ==============================================================================

def handle_recommendation_node(state: QueryUnderstandingState) -> dict:
    """
    Node 4a: Handle recommendation intent.

    For recommendation queries, we prioritize the reformulated query
    (optimized for similarity search) and generate a keyword-based
    variation for multi-query retrieval.

    Variations (2):
    - Primary: the reformulated query (concept-focused)
    - Secondary: top keywords joined (topic-focused)

    Returns:
        {"final_query": str, "query_variations": list[str]}
    """
    if state["reformulated"] is not None:
        final = state["reformulated"].optimized_query
        keywords = state["reformulated"].keywords
    else:
        final = state["original_query"]
        keywords = []

    # Build variations: primary query + keyword-based alternative
    variations = [final]
    if keywords:
        keyword_variation = " ".join(keywords[:3])
        if keyword_variation != final:
            variations.append(keyword_variation)

    return {"final_query": final, "query_variations": variations}


def handle_factual_node(state: QueryUnderstandingState) -> dict:
    """
    Node 4b: Handle factual intent.

    For factual queries, we keep the original query structure
    (exact matches are important for facts) and use the reformulated
    query as a second variation.

    Variations (2):
    - Primary: original query (preserves exact wording for BM25)
    - Secondary: reformulated query (cleaned for broader matching)

    Returns:
        {"final_query": str, "query_variations": list[str]}
    """
    if state["reformulated"] is not None:
        reformulated = state["reformulated"].optimized_query
    else:
        reformulated = state["original_query"]

    original = state["original_query"]

    # For factual: use reformulated as primary but keep original as variation
    final = reformulated
    variations = [final]
    if original != reformulated:
        variations.append(original)

    return {"final_query": final, "query_variations": variations}


def handle_exploratory_node(state: QueryUnderstandingState) -> dict:
    """
    Node 4c: Handle exploratory intent.

    For exploratory queries, we generate up to 3 variations to cast
    the widest net. Keywords are split into two groups so each
    variation "sees" different parts of the catalog.

    Variations (up to 3):
    - Primary: reformulated query expanded with top keywords
    - Secondary: first half of keywords (one perspective)
    - Tertiary: second half of keywords (another perspective)

    Returns:
        {"final_query": str, "query_variations": list[str]}
    """
    if state["reformulated"] is not None and state["reformulated"].keywords:
        base_query = state["reformulated"].optimized_query
        keywords = state["reformulated"].keywords
        top_keywords = " ".join(keywords[:5])
        final = f"{base_query} {top_keywords}"

        # Split keywords into two groups for different perspectives
        variations = [final]
        mid = len(keywords) // 2
        if mid > 0:
            first_half = " ".join(keywords[:mid])
            if first_half != final:
                variations.append(first_half)
        if mid < len(keywords):
            second_half = " ".join(keywords[mid:])
            if second_half != final and second_half not in variations:
                variations.append(second_half)

    elif state["reformulated"] is not None:
        final = state["reformulated"].optimized_query
        variations = [final]
    else:
        final = state["original_query"]
        variations = [final]

    return {"final_query": final, "query_variations": variations}


# ==============================================================================
# Conditional Routing
# ==============================================================================

def route_by_intent(state: QueryUnderstandingState) -> Literal["recommendation", "factual", "exploratory"]:
    """
    Routing function for conditional edges.

    Determines which intent-specific handler to route to based on
    the classified intent. Falls back to "exploratory" if intent is None.

    Args:
        state: Current graph state

    Returns:
        One of: "recommendation", "factual", "exploratory"
    """
    # If intent parsing failed, default to exploratory (safest/most general)
    if state["intent"] is None:
        return "exploratory"

    # Otherwise, route based on classified intent type
    return state["intent"].intent_type


# ==============================================================================
# Graph Builder
# ==============================================================================

def create_query_understanding_graph(llm):
    """
    Build and compile the query understanding LangGraph StateGraph.

    Flow:
        START
        → parse_intent
        → extract_filters
        → reformulate_query
        → [conditional routing based on intent_type]
        → handle_recommendation / handle_factual / handle_exploratory
        → END

    Args:
        llm: LangChain LLM instance (must support with_structured_output())

    Returns:
        Compiled StateGraph ready for invocation

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        >>> graph = create_query_understanding_graph(llm)
        >>> result = graph.invoke({"original_query": "books like 1984"})
        >>> print(result["final_query"])
    """
    # Initialize the StateGraph with our state schema
    workflow = StateGraph(QueryUnderstandingState)

    # Add LLM nodes (wrap llm parameter into lambda)
    workflow.add_node("parse_intent", lambda state: parse_intent_node(state, llm))
    workflow.add_node("extract_filters", lambda state: extract_filters_node(state, llm))
    workflow.add_node("reformulate_query", lambda state: reformulate_query_node(state, llm))

    # Add intent-specific handler nodes (no llm needed)
    workflow.add_node("handle_recommendation", handle_recommendation_node)
    workflow.add_node("handle_factual", handle_factual_node)
    workflow.add_node("handle_exploratory", handle_exploratory_node)

    # Linear edges for the first 3 nodes
    workflow.add_edge(START, "parse_intent")
    workflow.add_edge("parse_intent", "extract_filters")
    workflow.add_edge("extract_filters", "reformulate_query")

    # Conditional routing after reformulation
    workflow.add_conditional_edges(
        "reformulate_query",
        route_by_intent,
        {
            "recommendation": "handle_recommendation",
            "factual": "handle_factual",
            "exploratory": "handle_exploratory"
        }
    )

    # All intent handlers lead to END
    workflow.add_edge("handle_recommendation", END)
    workflow.add_edge("handle_factual", END)
    workflow.add_edge("handle_exploratory", END)

    # Compile the graph
    return workflow.compile()