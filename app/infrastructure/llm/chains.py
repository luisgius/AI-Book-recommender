"""
LangChain chains for LLM operations.

This module defines LangChain chains (composable pipelines) for different
LLM-powered tasks. Chains combine prompts, LLMs, and parsers into reusable units.

Architecture:
- Chains are created as functions that take an LLM and return a runnable
- This allows dependency injection and testing with mock LLMs
- Uses LCEL (LangChain Expression Language) for composition: prompt | llm | parser
"""

from langchain_core.runnables import RunnableLambda

from .prompts import build_grounded_explanation_prompt
from .prompts_judge import build_llm_judge_prompt
from .schemas import GroundedExplanationLLM
from .schemas_judge import ExplanationJudgmentLLM


def create_grounded_explanation_chain(llm):
    """
    Create a LangChain chain for generating grounded explanations with citations.

    This chain:
    1. Takes book context and query as inputs
    2. Formats them into a grounded explanation prompt
    3. Calls the LLM with structured output (Pydantic schema)
    4. Returns GroundedExplanationLLM with summary, reasoning, citations, confidence

    The chain uses .with_structured_output() for native structured generation,
    which is more robust than manual JSON parsing.

    Args:
        llm: A LangChain ChatModel (e.g., ChatOpenAI or ChatAnthropic)
             Must support structured output (OpenAI function calling or Anthropic tools)

    Returns:
        A runnable chain that accepts:
        - Input: dict with keys {query_text, book_id, title, authors, categories, description}
        - Output: GroundedExplanationLLM (Pydantic model)

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        >>> chain = create_grounded_explanation_chain(llm)
        >>> result = chain.invoke({
        ...     "query_text": "books about AI",
        ...     "book_id": "123...",
        ...     "title": "Artificial Intelligence: A Modern Approach",
        ...     "authors": "Russell, Norvig",
        ...     "categories": "Computer Science, AI",
        ...     "description": "Comprehensive AI textbook..."
        ... })
        >>> print(result.summary)
        >>> print(result.citations)
    """
    # Step 1: Add structured output capability to the LLM
    # This uses native function calling (OpenAI) or tools (Anthropic)
    # instead of relying on the LLM to generate valid JSON in its response
    llm_with_structure = llm.with_structured_output(GroundedExplanationLLM)

    # Step 2: Create prompt formatter function
    # This function converts the input dict into a formatted prompt message
    def format_prompt(inputs: dict) -> list:
        """
        Format inputs into a prompt message for the LLM.

        Takes a dict with book fields and query, calls build_grounded_explanation_prompt()
        to construct the full prompt text, and returns it in LangChain messages format.

        Args:
            inputs: Dict with keys:
                - query_text: User's search query
                - book_id: UUID of the book as string
                - title: Book title
                - authors: Comma-separated author names
                - categories: Comma-separated category names
                - description: Book description text

        Returns:
            List with a single message dict in LangChain format:
            [{"role": "user", "content": "<full prompt text>"}]
        """
        # Build the complete prompt using our versioned prompt template
        prompt_text = build_grounded_explanation_prompt(
            query_text=inputs["query_text"],
            book_id=inputs["book_id"],
            title=inputs["title"],
            authors=inputs["authors"],
            categories=inputs["categories"],
            description=inputs["description"]
        )

        # Return in LangChain messages format
        # ChatModels expect a list of message dicts
        return [{"role": "user", "content": prompt_text}]

    # Step 3: Wrap function in RunnableLambda
    # RunnableLambda makes any Python function compatible with LCEL (|)
    prompt_formatter = RunnableLambda(format_prompt)

    # Step 4: Compose the chain using LCEL
    # The | operator chains runnables: output of left becomes input of right
    # Flow: inputs dict → format_prompt → messages → llm → GroundedExplanationLLM
    chain = prompt_formatter | llm_with_structure

    return chain


def create_llm_judge_chain(llm):
    """
    Create a LangChain chain for evaluating explanation quality (LLM-as-Judge).

    This chain implements Block 3's evaluation pipeline:
    1. Takes an explanation with its context (query, book, citations)
    2. Formats them into the judge prompt with evaluation rubrics
    3. Calls the LLM with structured output (Pydantic schema)
    4. Returns ExplanationJudgmentLLM with scores and reasoning for 3 dimensions

    The LLM acts as an expert evaluator, scoring the explanation on:
    - Groundedness: Are claims supported by citations?
    - Clarity: Is it well-structured and understandable?
    - Relevance: Does it address the user's query intent?

    Args:
        llm: A LangChain ChatModel (e.g., ChatOpenAI or ChatAnthropic)
             Must support structured output (OpenAI function calling or Anthropic tools)

    Returns:
        A runnable chain that accepts:
        - Input: dict with keys {query_text, book_title, explanation_text, citations}
        - Output: ExplanationJudgmentLLM (Pydantic model with 3 JudgmentDimension objects)

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        >>> chain = create_llm_judge_chain(llm)
        >>> result = chain.invoke({
        ...     "query_text": "books about AI",
        ...     "book_title": "Artificial Intelligence: A Modern Approach",
        ...     "explanation_text": "This book is highly relevant because...",
        ...     "citations": [Citation(...), Citation(...)]
        ... })
        >>> print(result.groundedness.score)  # 5
        >>> print(result.groundedness.reasoning)  # "All claims supported by citations"
    """
    # Step 1: Add structured output capability
    # This ensures the LLM returns a valid ExplanationJudgmentLLM object
    # instead of free-form text that we'd have to parse
    llm_with_structure = llm.with_structured_output(ExplanationJudgmentLLM)

    # Step 2: Create prompt formatter function
    def format_judge_prompt(inputs: dict) -> list:
        """
        Format inputs into the judge evaluation prompt.

        Takes the evaluation context and constructs the full judge prompt
        with rubrics and instructions.

        Args:
            inputs: Dict with keys:
                - query_text: User's original search query
                - book_title: Title of the book being explained
                - explanation_text: The generated explanation to evaluate
                - citations: List[Citation] objects from the explanation

        Returns:
            List with a single message dict in LangChain format:
            [{"role": "user", "content": "<full judge prompt>"}]
        """
        # Build the complete judge prompt using our versioned template
        prompt_text = build_llm_judge_prompt(
            query_text=inputs["query_text"],
            book_title=inputs["book_title"],
            explanation_text=inputs["explanation_text"],
            citations=inputs["citations"]
        )

        # Return in LangChain messages format
        return [{"role": "user", "content": prompt_text}]

    # Step 3: Wrap function in RunnableLambda for LCEL compatibility
    prompt_formatter = RunnableLambda(format_judge_prompt)

    # Step 4: Compose the chain using LCEL (| operator)
    # Flow: inputs dict → format_judge_prompt → messages → llm → ExplanationJudgmentLLM
    chain = prompt_formatter | llm_with_structure

    return chain


# =============================================================================
# FUTURE CHAINS (not yet implemented)
# =============================================================================

"""
Chains implemented:
✅ Block 1: Grounded Explanation Chain - Generates explanations with citations
✅ Block 2: Query Understanding - Implemented via LangGraph (see graphs/query_understanding.py)
✅ Block 3: LLM-as-Judge Chain - Evaluates explanation quality on 3 dimensions

Future chains to implement:

1. Agentic Search Chain (Block 4 - Optional):
   - Uses: create_react_agent with search_catalog and get_similar_books tools
   - For: Complex multi-step queries requiring tool use and planning
   - Pattern: ReAct (Reasoning + Acting) with LangChain AgentExecutor
"""
