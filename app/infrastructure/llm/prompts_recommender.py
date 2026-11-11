"""
Prompts of the LLM recommender (task V07).

Two prompts, both short so they can be printed in the report:

- Listwise (R1 and R2): the model receives a numbered list of catalog books
  and the query, and returns the ids of the books it recommends, best first,
  with one sentence each. R1 sends the top 30 retrieved candidates, R2 the
  whole catalog; the prompt is the same, only the list differs.
- Closed book (R3): the model receives only the query and recommends real
  books (title and author) from its own knowledge; no catalog is shown.

Layout: the book list comes first and the query last, so a long list is read
before the question and the prefix (instructions and list) is identical for
every query when the list is the whole catalog (providers can cache it).

Each book is shown with a short id (B1, B2, ...), its title, authors, year,
language, categories and its description cut to DESCRIPTION_MAX_CHARS
characters at a word boundary ("..." included).

Versioning format: "vX.Y" (X: structure or rules, Y: wording). The version
is recorded with every cached output.
"""

from typing import List, Optional, Sequence, Tuple

RECOMMENDER_PROMPT_VERSION = "v1.0"
"""
Version history:
- v1.0 (2026-11-11): listwise prompt (ids from a given list) and closed-book
  prompt (title and author from the model's own knowledge), 10 books each.
"""

# Longest description shown per book, "..." included.
DESCRIPTION_MAX_CHARS = 500
ELLIPSIS = "..."
MISSING = "(none)"

LISTWISE_SYSTEM_PROMPT = (
    "You are a book recommender for a library catalog. You may only recommend books "
    "from the list you are given, and you refer to each book by its id."
)

LISTWISE_USER_TEMPLATE = """Books:

{books}

Query: "{query}"

Choose the {n} books from the list that best match the query, best first. Use only ids from the list, each at most once. For each book give its id and one short sentence, in the language of the query, saying why it matches."""

CLOSED_BOOK_SYSTEM_PROMPT = "You are a book recommender."

CLOSED_BOOK_USER_TEMPLATE = """Query: "{query}"

Recommend the {n} real, published books that best match the query, best first. For each book give its exact title, its main author, and one short sentence, in the language of the query, saying why it matches."""


def truncate_description(description: Optional[str], limit: int = DESCRIPTION_MAX_CHARS) -> str:
    """
    A description as shown to the model: whitespace collapsed and at most
    `limit` characters, cut at a word boundary and ended with "...".

    Returns:
        The text, or "(none)" when the catalog has no description
    """
    text = " ".join((description or "").split())
    if not text:
        return MISSING
    if len(text) <= limit:
        return text
    cut = text[: limit - len(ELLIPSIS)]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + ELLIPSIS


def format_book_entry(
    book_id: str,
    title: str,
    authors: Sequence[str],
    year: Optional[int],
    language: Optional[str],
    categories: Sequence[str],
    description: Optional[str],
) -> str:
    """
    One book of the list, in two lines:

        [B7] title: ...; authors: ...; year: ...; language: ...; categories: ...
        description: ...
    """
    header = (
        f"[{book_id}] title: {' '.join(title.split())}; "
        f"authors: {', '.join(authors) if authors else 'unknown'}; "
        f"year: {year if year is not None else 'unknown'}; "
        f"language: {language or 'unknown'}; "
        f"categories: {', '.join(categories) if categories else MISSING}"
    )
    return f"{header}\ndescription: {truncate_description(description)}"


def build_listwise_prompt(query_text: str, book_entries: Sequence[str], n: int) -> Tuple[str, str]:
    """
    The listwise prompt (R1 and R2).

    Args:
        query_text: The user's query
        book_entries: The books, as format_book_entry returns them, in the
            order they are shown
        n: How many books to recommend

    Returns:
        (system message, user message)
    """
    user = LISTWISE_USER_TEMPLATE.format(books="\n\n".join(book_entries), query=query_text.strip(), n=n)
    return LISTWISE_SYSTEM_PROMPT, user


def build_closed_book_prompt(query_text: str, n: int) -> Tuple[str, str]:
    """The closed-book prompt (R3): (system message, user message)."""
    return CLOSED_BOOK_SYSTEM_PROMPT, CLOSED_BOOK_USER_TEMPLATE.format(query=query_text.strip(), n=n)


def prompt_texts() -> List[Tuple[str, str]]:
    """(name, text) of every template, for the report."""
    return [
        ("listwise system", LISTWISE_SYSTEM_PROMPT),
        ("listwise user", LISTWISE_USER_TEMPLATE),
        ("closed-book system", CLOSED_BOOK_SYSTEM_PROMPT),
        ("closed-book user", CLOSED_BOOK_USER_TEMPLATE),
    ]
