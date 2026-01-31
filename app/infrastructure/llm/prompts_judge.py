"""
Prompts for LLM-as-Judge evaluation (Block 3).

All prompts are versioned for reproducibility and evaluation tracking.
Changes to prompts should increment the version number and be documented.

Versioning format: "vX.Y"
- X increments for major changes (structure, rules, format)
- Y increments for minor changes (wording, examples)
"""

import json
from typing import List

from app.domain.entities import Citation


# =============================================================================
# LLM-AS-JUDGE EVALUATION
# =============================================================================

LLM_JUDGE_PROMPT_VERSION = "v1.0"
"""
Version history:
- v1.0 (2024-12-22): Initial implementation with 3 dimensions (groundedness, clarity, relevance)
"""


def build_llm_judge_prompt(
    query_text: str,
    book_title: str,
    explanation_text: str,
    citations: List[Citation]
) -> str:
    """
    Build the prompt for LLM-as-Judge evaluation.

    This function constructs a prompt that:
    1. Defines the LLM's role as an evaluator
    2. Provides the context (query, book, explanation, citations)
    3. Specifies the 3 evaluation dimensions with clear rubrics
    4. Requires structured output with scores and reasoning

    Args:
        query_text: The user's original search query
        book_title: Title of the book being explained
        explanation_text: The generated explanation to evaluate
        citations: List of Citation objects from the explanation

    Returns:
        Complete prompt string ready for LLM
    """
    # Format citations as JSON for clear presentation
    citations_data = [
        {
            "chunk_id": c.chunk_id,
            "snippet": c.snippet,
            "relevance_score": c.relevance_score
        }
        for c in citations
    ]
    citations_json = json.dumps(citations_data, indent=2, ensure_ascii=False)

    return f"""You are an expert evaluator assessing the quality of book recommendation explanations.

Your task is to evaluate an explanation that was generated to help a user understand why a book is relevant to their search query.

=== CONTEXT ===

USER QUERY: "{query_text}"

BOOK TITLE: "{book_title}"

GENERATED EXPLANATION:
{explanation_text}

CITATIONS PROVIDED:
{citations_json}

=== EVALUATION CRITERIA ===

Rate the explanation on these 3 dimensions using a 1-5 scale:

1. GROUNDEDNESS (Are claims supported by evidence?)
   - Score 5: Every claim in the explanation has explicit citation support. The explanation only states what can be verified from the provided book information.
   - Score 4: Almost all claims are supported. Minor statements may lack direct citation but don't contradict evidence.
   - Score 3: Most major claims are supported. Some statements go slightly beyond what citations show.
   - Score 2: Several important claims lack citation support. Explanation includes unsupported assertions.
   - Score 1: Major claims are not supported by citations. Explanation appears to fabricate or assume information.

2. CLARITY (Is it clear and well-structured?)
   - Score 5: Crystal clear, logically organized, easy to understand. Reader immediately knows why the book is relevant.
   - Score 4: Clear and well-structured with minor areas that could be improved.
   - Score 3: Understandable but could be more clearly written or better organized.
   - Score 2: Somewhat confusing structure or wording. Reader has to work to understand the point.
   - Score 1: Confusing, poorly structured, hard to follow. Fails to communicate effectively.

3. RELEVANCE (Does it address the user's query?)
   - Score 5: Directly and fully addresses the query intent. Explains exactly why this book matches what the user is looking for.
   - Score 4: Addresses the query well with minor tangential information.
   - Score 3: Partially relevant. Addresses some aspects of the query but misses others.
   - Score 2: Weakly relevant. Only tangentially connects to what the user asked for.
   - Score 1: Not relevant. Fails to explain why the book matches the user's query.

=== INSTRUCTIONS ===

For each dimension:
1. Carefully analyze the explanation against the criteria above
2. Assign a score from 1-5
3. Provide a brief reasoning (max 200 characters) explaining your score

Be objective and consistent. Base your judgment solely on the content provided.
Do not assume information not present in the explanation or citations.

Provide your evaluation now:"""


# =============================================================================
# SYSTEM MESSAGES
# =============================================================================

LLM_JUDGE_SYSTEM_MESSAGE = """You are an expert evaluator for book recommendation explanations.

Your role is to objectively assess explanation quality across three dimensions:
- Groundedness: Are claims supported by cited evidence?
- Clarity: Is the explanation clear and well-organized?
- Relevance: Does it address the user's query intent?

You provide scores (1-5) with brief, constructive reasoning.
Be fair, consistent, and base judgments only on provided content."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_citations_for_judge(citations: List[Citation]) -> str:
    """
    Format citations as a readable string for the judge prompt.

    Args:
        citations: List of Citation objects

    Returns:
        Formatted string representation of citations
    """
    if not citations:
        return "No citations provided."

    lines = []
    for i, c in enumerate(citations, start=1):
        lines.append(
            f"{i}. [{c.chunk_id}] \"{c.snippet}\" (relevance: {c.relevance_score:.2f})"
        )

    return "\n".join(lines)


def get_judge_prompt_metadata() -> dict[str, str]:
    """
    Get metadata about judge prompts for logging and evaluation.

    Returns:
        Dictionary mapping prompt names to version strings
    """
    return {
        "llm_judge": LLM_JUDGE_PROMPT_VERSION,
    }


# =============================================================================
# NOTES FOR FUTURE ITERATIONS
# =============================================================================

"""
MVP SCOPE (Block 3):
- Single prompt, 3 dimensions, stored artifacts
- Do NOT over-engineer

POTENTIAL IMPROVEMENTS (if needed):

1. Calibration Examples:
   - Add 2-3 calibration examples showing scores across the range
   - Helps LLM understand the scale better

2. Pairwise Comparison:
   - Instead of absolute scores, compare two explanations
   - "Which explanation is better for groundedness?"

3. Chain-of-Thought:
   - Ask LLM to list evidence before scoring
   - "First, list all claims in the explanation. Then, for each claim..."

4. Multi-Judge Ensemble:
   - Run same evaluation 3 times with temp > 0
   - Average scores for more stable results

5. Dimension Weighting:
   - Allow configurable weights per dimension
   - e.g., Groundedness 50%, Clarity 25%, Relevance 25%

6. Negative Examples:
   - Add examples of BAD explanations with low scores
   - Helps calibrate the lower end of the scale
"""
