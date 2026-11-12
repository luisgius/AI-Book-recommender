"""
Pydantic schemas of the assistant graph's two LLM calls (graphs/assistant.py).

- RouteDecisionLLM: the routing call. One strict structured output with the
  route, the condensed query, the filters the user asked for and, for the
  clarify route, the question and its suggested answers.
- GroundedAnswerLLM: the answer of the search route. A short message and up
  to MAX_PICKS recommended books, each with citations that the existing
  citation guardrail (guardrails.filter_valid_citations) verifies.

As in schemas.py, the JSON schemas hold only standard keywords and limits on
free text are soft (cut, not rejected). Every rule that matters (the
clarification cap, valid routes, which books may be recommended, which
filters apply) is enforced in code afterwards, not trusted to these schemas.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .schemas import CitationLLM, soft_max_items, soft_max_length

MAX_OPTIONS = 4
MAX_PICKS = 3

RouteName = Literal["clarify", "search", "agent", "out_of_scope"]


class RouteDecisionLLM(BaseModel):
    """What the routing node decided for the last user message."""

    route: RouteName = Field(
        description=(
            "clarify: too vague to search well, ask one question; search: specific enough to "
            "search the catalog; agent: needs several steps (compare books, books like a given "
            "one but different in some way, questions about one specific book); out_of_scope: "
            "not a request a book catalog can serve"
        )
    )
    reply_language: Annotated[str, soft_max_length(5)] = Field(
        description="ISO 639-1 code of the language of the user's LAST message, e.g. 'es' or 'en'"
    )
    search_query: Annotated[str, soft_max_length(300)] = Field(
        description=(
            "search and agent: the whole conversation condensed into one standalone request, in "
            "the user's words and language. clarify and out_of_scope: empty string"
        )
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 code of the language the BOOKS must be in, only if the user asked for it",
    )
    min_year: int | None = Field(default=None, description="Earliest publication year, only if the user gave one")
    max_year: int | None = Field(default=None, description="Latest publication year, only if the user gave one")
    question: Annotated[str, soft_max_length(200)] | None = Field(
        default=None, description="clarify only: one short follow-up question in the reply language"
    )
    options: Annotated[list[Annotated[str, soft_max_length(60)]], soft_max_items(MAX_OPTIONS)] = Field(
        default_factory=list,
        description="clarify only: 2 to 4 short suggested answers in the reply language; otherwise empty",
    )
    reason: Annotated[str, soft_max_length(160)] = Field(
        description="One short sentence on why this route was chosen (for the trace)"
    )


class AnswerPickLLM(BaseModel):
    """One recommended book of the search route's answer."""

    book_id: str = Field(description="book_id exactly as shown in the book list")
    reason: Annotated[str, soft_max_length(300)] = Field(
        description="One sentence, in the reply language, saying why it fits, based only on its data"
    )
    citations: Annotated[list[CitationLLM], soft_max_items(3)] = Field(
        description="1 to 3 verbatim snippets from this book's fields that support the reason"
    )


class GroundedAnswerLLM(BaseModel):
    """Answer of the search route, recommending from the retrieved books."""

    message: Annotated[str, soft_max_length(400)] = Field(
        description="One or two sentences in the reply language introducing the recommendations"
    )
    picks: Annotated[list[AnswerPickLLM], soft_max_items(MAX_PICKS)] = Field(
        description=f"Up to {MAX_PICKS} books from the list that best answer the request, best first"
    )
