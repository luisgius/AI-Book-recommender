"""
Prompts for LLM interactions.

All prompts are versioned for reproducibility and evaluation tracking.
Changes to prompts should increment the version number and be documented.

Versioning format: "vX.Y"
- X increments for major changes (structure, rules, format)
- Y increments for minor changes (wording, examples)
"""

# =============================================================================
# GROUNDED EXPLANATION GENERATION
# =============================================================================

GROUNDED_EXPLANATION_PROMPT_VERSION = "v1.0"
"""
Version history:
- v1.0 (2024-12-15): Initial implementation with JSON context, citation rules, and few-shot example
"""


def build_grounded_explanation_prompt(
    query_text: str,
    book_id: str,
    title: str,
    authors: str,
    categories: str,
    description: str
) -> str:
    """
    Build the prompt for grounded explanation generation.

    This function constructs a prompt that:
    1. Defines the LLM's role and task
    2. Provides strict grounding rules
    3. Presents book context in JSON format
    4. Includes a diverse few-shot example (content neutralized)
    5. Specifies the expected output structure

    Args:
        query_text: The user's search query
        book_id: UUID of the book as string
        title: Book title
        authors: Comma-separated author names
        categories: Comma-separated category names
        description: Book description text

    Returns:
        Complete prompt string ready for LLM
    """
    return f"""You are a book recommendation assistant specialized in providing evidence-based explanations.

Your task is to explain why a book is relevant to a user's query. Your explanation MUST be grounded in the book's actual data - every claim you make must be supported by a citation from the book information provided.

STRICT RULES:
1. Every claim you make MUST be supported by a citation from the book data
2. Citations MUST reference actual text from the book fields (title, description, categories, authors)
3. Each citation snippet MUST be verbatim text from the cited field (max 200 characters)
4. Do NOT invent, assume, or paraphrase information not explicitly present
5. If you cannot find sufficient evidence to support a claim, do not make that claim
6. Provide at least ONE citation - explanations without citations will be rejected

=== BOOK CONTEXT (JSON) ===
{{
  "book_id": "{book_id}",
  "title": "{title}",
  "authors": "{authors}",
  "categories": "{categories}",
  "description": "{description}"
}}

=== CITATION FORMAT ===
For each claim, you must provide:
- chunk_id: Which field you're citing (one of: "title", "description", "categories", "authors")
- snippet: Exact verbatim text from that field (max 200 chars)
- relevance_score: How strongly this evidence supports your claim (0.0 to 1.0)

Example citation structure:
{{
  "book_id": "{book_id}",
  "chunk_id": "description",
  "snippet": "comprehensive introduction to machine learning",
  "relevance_score": 0.9
}}

=== FEW-SHOT EXAMPLE (for format understanding only - do NOT copy content) ===

Query: "books about sustainable gardening"
Book Context:
{{
  "book_id": "example-uuid-123",
  "title": "The Urban Gardener's Handbook",
  "authors": "Jane Smith, Robert Green",
  "categories": "Gardening, Sustainability, Urban Living",
  "description": "A practical guide to creating eco-friendly gardens in urban spaces using composting and water conservation techniques."
}}

Output:
{{
  "summary": "This book directly addresses sustainable gardening practices in urban environments.",
  "reasoning": "The book is highly relevant because it focuses on eco-friendly gardening methods [CITE: description] and includes sustainability as a core topic [CITE: categories]. The guide covers composting and water conservation [CITE: description], which are key sustainable practices.",
  "citations": [
    {{
      "book_id": "example-uuid-123",
      "chunk_id": "description",
      "snippet": "eco-friendly gardens in urban spaces using composting and water conservation techniques",
      "relevance_score": 0.95
    }},
    {{
      "book_id": "example-uuid-123",
      "chunk_id": "categories",
      "snippet": "Sustainability",
      "relevance_score": 0.90
    }}
  ],
  "confidence": 0.92
}}

=== YOUR TASK ===

User Query: "{query_text}"

Instructions:
1. Analyze the book context above (NOT the example)
2. Identify evidence that shows why this book is relevant to the query
3. Construct your explanation with proper citations
4. Use ONLY information from the book context provided
5. Return JSON following the exact structure shown

IMPORTANT: The example above is ONLY to show you the format. Do NOT use any content from the example in your response. Base your answer entirely on the actual book context provided.

Generate your grounded explanation now:"""


# =============================================================================
# SYSTEM MESSAGES
# =============================================================================

GROUNDED_EXPLANATION_SYSTEM_MESSAGE = """You are a book recommendation assistant that provides evidence-based explanations.

Core principles:
- Every claim must be grounded in provided book data
- Citations must reference actual text, never invented or paraphrased
- If evidence is insufficient, acknowledge limitations rather than speculate
- Prioritize accuracy and traceability over eloquence

Output format: Structured JSON with summary, reasoning, citations, and confidence score."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_book_context_json(
    book_id: str,
    title: str,
    authors: list[str],
    categories: list[str],
    description: str | None
) -> str:
    """
    Format book data as JSON string for inclusion in prompt.

    Args:
        book_id: Book UUID as string
        title: Book title
        authors: List of author names
        categories: List of category/genre names
        description: Book description (may be None)

    Returns:
        JSON-formatted string representation of book context
    """
    import json

    context = {
        "book_id": book_id,
        "title": title,
        "authors": ", ".join(authors),
        "categories": ", ".join(categories),
        "description": description or "No description available"
    }

    return json.dumps(context, indent=2, ensure_ascii=False)


def get_prompt_metadata() -> dict[str, str]:
    """
    Get metadata about all prompts for logging and evaluation.

    Returns:
        Dictionary mapping prompt names to version strings
    """
    return {
        "grounded_explanation": GROUNDED_EXPLANATION_PROMPT_VERSION,
    }


# =============================================================================
# NOTES FOR FUTURE ITERATIONS
# =============================================================================

"""
MITIGATION STRATEGIES IMPLEMENTED:

1. Content Neutralization:
   - Few-shot example uses DIFFERENT domain (gardening vs AI/tech books)
   - Prevents copying of specific terms/phrases

2. Explicit Delimiters:
   - Clear "=== SECTIONS ===" separate context, examples, and task
   - Reduces confusion about what is input vs example

3. Negative Instructions:
   - "Do NOT use any content from the example" explicitly stated
   - Reminds LLM that example is for format only

4. Chain of Thought (in example):
   - Shows reasoning process: identify evidence -> construct explanation -> cite
   - Model learns the LOGIC, not just the output

FUTURE IMPROVEMENTS (if needed):

5. Example Shuffling:
   - When adding more examples, randomize their order per request
   - Mitigates recency bias

6. Temperature Tuning:
   - Start with temp=0 for determinism
   - Increase slightly (0.2-0.3) if responses feel too rigid

7. Dynamic Example Selection (RAG):
   - Store pool of diverse examples
   - Retrieve most similar to current query (but different domain)
   - Requires embeddings infrastructure

8. Prompt Ablation Testing:
   - Version with/without few-shot
   - Version with different rule phrasings
   - Measure citation precision and groundedness scores
"""
