"""
Prompt templates for query understanding with LLMs.

This module contains all prompts used in the query understanding LangGraph flow.
Prompts are versioned to enable A/B testing and reproducibility.

Version naming convention:
- PROMPT_NAME_V1_0: Initial version
- PROMPT_NAME_V1_1: Minor improvements (better examples, clarifications)
- PROMPT_NAME_V2_0: Major rewrite (different approach)

Current versions in use:
- Intent extraction: v1.0
- Filter extraction: v1.0
- Query reformulation: v1.0
"""

# Version metadata
PROMPTS_VERSION = "1.0"
CREATED_DATE = "2024-12-18"  

# ==============================================================================
# Prompt 1: Intent Classification
# ==============================================================================

INTENT_EXTRACTION_PROMPT_V1_0 = """You are an expert at understanding search queries for a book recommendation system.

Your task is to classify the user's query into EXACTLY ONE of these intent types:

1. "recommendation" - User wants books similar to a specific book, author, or topic
   Key indicators: "like", "similar to", "more", "recommend", comparison language
   Examples:
   - "books like 1984"
   - "similar to Isaac Asimov"
   - "more fantasy novels like Lord of the Rings"
   - "recommend sci-fi like Dune"

2. "factual" - User wants specific information about a book, author, or literary fact
   Key indicators: "who", "what", "when", "where", "is there", specific questions
   Examples:
   - "who wrote Dune"
   - "when was 1984 published"
   - "is there a sequel to Foundation"
   - "what genre is Pride and Prejudice"

3. "exploratory" - User wants to discover books in a topic, genre, or category
   Key indicators: "about", "on", topic keywords, genre names, broad search
   Examples:
   - "science fiction about artificial intelligence"
   - "best fantasy novels"
   - "books on Roman history"
   - "popular thrillers from 2020"

IMPORTANT CONTEXT:
This classification determines the search strategy:
- recommendation → Emphasize vector similarity search
- factual → Emphasize lexical (keyword) search + direct lookup
- exploratory → Balanced hybrid search with diversity

EDGE CASES:
- If the query is ambiguous, choose the most likely intent and set confidence < 0.7
- If the query contains multiple intents, prioritize the PRIMARY intent
- If the query is very short (1-2 words), it's usually "exploratory"

Your response must be valid JSON matching this schema:
{{
  "intent_type": "recommendation" | "factual" | "exploratory",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation (1-2 sentences) of why you chose this classification"
}}

User query: {query}

Classify this query now.
"""

# ==============================================================================
# Prompt 2: Filter Extraction
# ==============================================================================

FILTER_EXTRACTION_PROMPT_V1_0 = """You are an expert at extracting structured filters from natural language search queries.

Your task is to extract the following filters from the user's query (ONLY if explicitly mentioned):

1. **language**: ISO 639-1 language code (2 letters)
   - "English books" → "en"
   - "Spanish novels" → "es" (if referring to language)
   - "French literature" → "fr"
   - "libros en español" → "es"
   - ONLY extract if the user specifies a language to filter by

2. **category**: Book genre/category (use standard names)
   - "science fiction" → "Science Fiction"
   - "fantasy" → "Fantasy"
   - "history" or "historical" → "History"
   - "biography" → "Biography"
   - "fiction" → "Fiction"
   - "non-fiction" → "Non-Fiction"

3. **min_year**: Minimum publication year (integer)
   - "recent books" → 2020 (current year - 5)
   - "from the 90s" → 1990
   - "published after 2010" → 2010
   - "last 5 years" → 2020 (current year - 5)

4. **max_year**: Maximum publication year (integer)
   - "from the 90s" → 1999
   - "before 2000" → 1999
   - "published in 2020" → 2020 (also set min_year=2020)

5. **author_hint**: Author name if mentioned (string)
   - "books by Asimov" → "Isaac Asimov"
   - "Stephen King novels" → "Stephen King"
   - Note: This is a hint, not a strict filter (we don't support author filtering yet)

IMPORTANT RULES:
- Set a field to null if NOT mentioned in the query
- For ambiguous queries like "Spanish novels":
  * Use context clues: "Spanish novels" likely means language="es"
  * "novels by Spanish authors" would be author_hint="Spanish authors" + category="Fiction"
  * When in doubt, prefer language filter over category
- "recent" means last 5 years (min_year = current_year - 5)
- For decades: "90s" = 1990-1999, "2000s" = 2000-2009
- Normalize category names to title case

EXAMPLES:

Query: "recent science fiction books"
→ {{
  "language": null,
  "category": "Science Fiction",
  "min_year": 2020,
  "max_year": null,
  "author_hint": null
}}

Query: "Spanish novels from the 90s"
→ {{
  "language": "es",
  "category": "Fiction",
  "min_year": 1990,
  "max_year": 1999,
  "author_hint": null
}}

Query: "books by Isaac Asimov about robots"
→ {{
  "language": null,
  "category": "Science Fiction",
  "min_year": null,
  "max_year": null,
  "author_hint": "Isaac Asimov"
}}

Query: "fantasy novels" (no filters explicitly mentioned)
→ {{
  "language": null,
  "category": "Fantasy",
  "min_year": null,
  "max_year": null,
  "author_hint": null
}}

Your response must be valid JSON matching the schema above.

User query: {query}

Extract filters now.
"""

# ==============================================================================
# Prompt 3: Query Reformulation
# ==============================================================================

QUERY_REFORMULATION_PROMPT_V1_0 = """You are an expert at optimizing search queries for hybrid retrieval systems (BM25 + vector search).

Your task is to reformulate the user's query to improve search quality by:
1. Removing noise words (filler, politeness, conversational language)
2. Fixing typos and grammatical errors
3. Expanding with synonyms or related terms (when helpful)
4. Focusing on the core search intent

NOISE WORDS TO REMOVE:
- Filler: "I want", "I'm looking for", "Can you", "Please", "Help me find"
- Questions: "Do you have", "Are there any", "Could you show me"
- Politeness: "please", "thank you", "if possible"
- Vague quantifiers: "some", "any", "a few"

OPTIMIZATION STRATEGIES:
- Keep domain-specific terms: "artificial intelligence", "machine learning", "neural networks"
- Expand abbreviations if helpful: "AI" → "artificial intelligence AI" (keep both)
- Add genre context if implied: "Asimov" → "Asimov science fiction"
- Preserve author names and book titles exactly as written
- For "books like X", expand to: "X similar style themes"

EXAMPLES:

Query: "I'm looking for some books about artificial intelligence please"
→ {{
  "optimized_query": "artificial intelligence AI",
  "keywords": ["artificial intelligence", "AI", "technology", "machine learning"],
  "removed_noise": ["I'm looking for", "some", "about", "please"]
}}

Query: "Can you recommend sci-fi books similar to Dune?"
→ {{
  "optimized_query": "science fiction Dune similar epic space opera",
  "keywords": ["science fiction", "Dune", "space opera", "epic", "similar"],
  "removed_noise": ["Can you recommend", "books", "to", "?"]
}}

Query: "books by azimov about robots" (typo)
→ {{
  "optimized_query": "Isaac Asimov robots science fiction",
  "keywords": ["Isaac Asimov", "Asimov", "robots", "science fiction", "AI"],
  "removed_noise": ["books", "by", "about"]
}}

Query: "fantasy novels like Lord of the Rings"
→ {{
  "optimized_query": "fantasy Lord of the Rings LOTR epic adventure magic",
  "keywords": ["fantasy", "Lord of the Rings", "LOTR", "epic", "adventure", "magic", "Tolkien"],
  "removed_noise": ["novels", "like"]
}}

Query: "recent thrillers" (already concise)
→ {{
  "optimized_query": "recent thrillers suspense mystery",
  "keywords": ["thrillers", "recent", "suspense", "mystery", "crime"],
  "removed_noise": []
}}

IMPORTANT RULES:
- Do NOT remove domain keywords (book genres, author names, topics)
- Do NOT over-expand (max 10 keywords)
- Preserve the original query's core meaning
- If the query is already concise, minimal changes are OK
- Keywords should be relevant to BOOK SEARCH (not general web search)

Your response must be valid JSON matching this schema:
{{
  "optimized_query": "string (max 300 chars)",
  "keywords": ["list", "of", "max", "10", "keywords"],
  "removed_noise": ["list", "of", "removed", "phrases"]
}}

User query: {query}

Reformulate this query now.
"""

# ==============================================================================
# Active Prompts (currently used)
# ==============================================================================

# These are the prompts currently used in production
INTENT_EXTRACTION_PROMPT = INTENT_EXTRACTION_PROMPT_V1_0
FILTER_EXTRACTION_PROMPT = FILTER_EXTRACTION_PROMPT_V1_0
QUERY_REFORMULATION_PROMPT = QUERY_REFORMULATION_PROMPT_V1_0

# ==============================================================================
# Helper: Get all prompts with versions
# ==============================================================================

def get_prompt_versions() -> dict[str, str]:
    """
    Get the current version of all prompts.

    Useful for logging and evaluation to track which prompt version was used.

    Returns:
        Dictionary mapping prompt names to version strings
    """
    return {
        "intent_extraction": "v1.0",
        "filter_extraction": "v1.0",
        "query_reformulation": "v1.0",
        "prompts_module": PROMPTS_VERSION
    }


def format_prompt(prompt_template: str, **kwargs) -> str:
    """
    Format a prompt template with variables.

    Args:
        prompt_template: The prompt template string with {variable} placeholders
        **kwargs: Variable values to substitute

    Returns:
        Formatted prompt string

    Example:
        >>> format_prompt(INTENT_EXTRACTION_PROMPT, query="books like 1984")
        "You are an expert... User query: books like 1984..."
    """
    return prompt_template.format(**kwargs)
