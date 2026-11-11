"""
Pydantic schemas of the LLM recommender's structured outputs (task V07).

- CandidateRecommendationsLLM: ids from the list shown in the prompt, best
  first, with one reason each (listwise prompt, R1 and R2).
- BookRecommendationsLLM: title and author of books the model knows, with
  one reason each (closed-book prompt, R3).

The number of recommendations and the validity of the ids are checked in
code (llm_recommender.py), not in the schema: an id outside the list is a
finding to count, not an error that discards the answer. Free text has
soft length limits (see soft_max_length in schemas.py).
"""

from typing import Annotated, List

from pydantic import BaseModel, Field

from .schemas import soft_max_length

REASON_MAX_CHARS = 300
TITLE_MAX_CHARS = 300
AUTHOR_MAX_CHARS = 200


class RecommendedCandidateLLM(BaseModel):
    """One recommended book of the list, by id."""

    id: str = Field(description="Id of a book from the list, for example B7")
    reason: Annotated[str, soft_max_length(REASON_MAX_CHARS)] = Field(
        max_length=REASON_MAX_CHARS, description="One short sentence: why the book matches the query"
    )


class CandidateRecommendationsLLM(BaseModel):
    """Answer to the listwise prompt."""

    recommendations: List[RecommendedCandidateLLM] = Field(description="The recommended books, best first")


class RecommendedBookLLM(BaseModel):
    """One recommended book, named by title and author."""

    title: Annotated[str, soft_max_length(TITLE_MAX_CHARS)] = Field(
        max_length=TITLE_MAX_CHARS, description="Exact title of the book"
    )
    author: Annotated[str, soft_max_length(AUTHOR_MAX_CHARS)] = Field(
        max_length=AUTHOR_MAX_CHARS, description="Main author of the book"
    )
    reason: Annotated[str, soft_max_length(REASON_MAX_CHARS)] = Field(
        max_length=REASON_MAX_CHARS, description="One short sentence: why the book matches the query"
    )


class BookRecommendationsLLM(BaseModel):
    """Answer to the closed-book prompt."""

    recommendations: List[RecommendedBookLLM] = Field(description="The recommended books, best first")
