"""
The LLM as a book recommender (task V07): one adapter, three ways to ask.

- Listwise over retrieved candidates (R1, RAG): LLMListwiseReranker
  implements the domain port Reranker, so SearchService can use it as its
  third stage in place of the cross-encoder. It receives the top fused
  candidates, asks the model for the 10 it recommends (by id, best first,
  one reason each) and returns all candidates re-ordered: the model's picks
  first, then the other candidates in the order they arrived.
- Listwise over the whole catalog (R2, long context, no retrieval):
  LLMRecommender.rank_books with every catalog book in the prompt.
- Closed book (R3, no catalog): LLMRecommender.recommend_closed_book asks
  for real books by title and author from the model's own knowledge.

Grounding rules of the listwise modes (pure functions below, tested):
- An id is read after trimming spaces and surrounding brackets, in upper
  case ("[b7] " -> "B7"), and must be one of the ids shown.
- Ids that were not shown are dropped and counted (unknown_ids); a repeated
  id is dropped and counted (duplicate_ids).
- If fewer than 10 valid ids come back, the top 10 is completed with the
  remaining candidates in the order they were shown (fused order for R1,
  catalog order for R2); `filled` counts those positions.
- An answer that cannot be parsed counts as zero valid ids: the list keeps
  the order it was shown in and the outcome records the error.

The model call goes through a Completion: a function from LLMRequest to
LLMOutput. LangChainCompletion sends the request to a LangChain chat model
with strict structured output and records latency, tokens and the cost the
provider reports (LLMUsageRecorder, one per call). The evaluation wraps it
with an on-disk cache; tests use plain functions.
"""

import logging
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import httpx
import openai
from pydantic import BaseModel

from app.domain.entities import Book, SearchResult
from app.domain.ports import Reranker
from app.infrastructure.config.settings import LLMSettings

from .langchain_llm_client import build_chat_model
from .prompts_recommender import (
    RECOMMENDER_PROMPT_VERSION,
    build_closed_book_prompt,
    build_listwise_prompt,
    format_book_entry,
)
from .schemas_recommender import BookRecommendationsLLM, CandidateRecommendationsLLM
from .structured_output import with_structured_output
from .usage import OPERATION_METADATA_KEY, LLMUsageRecorder

logger = logging.getLogger(__name__)

# Operation labels of the recommender's calls in the usage records.
RERANK_OPERATION = "listwise_rerank"
LONG_CONTEXT_OPERATION = "long_context_recommend"
CLOSED_BOOK_OPERATION = "closed_book_recommend"

# Books recommended per query (the product shows 10 results).
TOP_N = 10

# Ten ids or titles with one sentence each fit well under this cap.
RECOMMENDER_MAX_TOKENS = 2048

# Errors of the request itself (network, HTTP status, timeout): there is no
# answer to record, so they propagate and the caller may retry. Any other
# exception during the call means the model answered with something that
# is not valid for the schema; that is recorded as the call's output.
REQUEST_ERRORS = (openai.APIError, httpx.HTTPError, ConnectionError, TimeoutError)

# Longest error text kept in an output record.
MAX_ERROR_CHARS = 500


# =============================================================================
# Requests and outputs
# =============================================================================


@dataclass(frozen=True)
class LLMRequest:
    """One recommender call: the messages sent and how the call is labelled."""

    operation: str
    query_text: str
    system_prompt: str
    user_prompt: str
    schema: Type[BaseModel]

    def messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass(frozen=True)
class LLMOutput:
    """
    The model's answer to one request, with the usage of the call.

    Attributes:
        parsed: The structured answer (the schema's fields), None if unusable
        content: The text the model returned, when available
        error: Why the answer is unusable; None when parsed is set
        latency_s: Wall clock of the call, network included
        input_tokens, output_tokens: As the provider reported them (None if not)
        cached_input_tokens: Input tokens read from the provider's prompt cache
        cost_usd: Cost the provider reported (OpenRouter), None if not reported
    """

    parsed: Optional[Dict[str, Any]]
    content: Optional[str] = None
    error: Optional[str] = None
    latency_s: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LLMOutput":
        return cls(**{name: data.get(name) for name in cls.__dataclass_fields__})


# A function that answers a request: LangChainCompletion, a cache, a test double.
Completion = Callable[[LLMRequest], LLMOutput]


def _cached_input_tokens(message: Any) -> Optional[int]:
    """Prompt-cache reads reported in a LangChain message's usage_metadata, if any."""
    usage = getattr(message, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    value = details.get("cache_read")
    return int(value) if isinstance(value, (int, float)) else None


class LangChainCompletion:
    """
    Sends LLMRequests to a LangChain chat model with strict structured output.

    Each call gets its own LLMUsageRecorder (passed in the run config), so the
    latency, tokens and cost of that call are known even when several calls
    run at once. The chat model may carry a recorder of its own as well.
    """

    def __init__(self, chat_model: Any, usage_recorder: Optional[LLMUsageRecorder] = None) -> None:
        """
        Args:
            chat_model: LangChain chat model that supports structured output
            usage_recorder: The recorder attached to chat_model, if any (it
                sums every call; exposed for the caller)
        """
        self._chat_model = chat_model
        self.usage_recorder = usage_recorder

    @classmethod
    def from_settings(cls, settings: LLMSettings, max_tokens: int = RECOMMENDER_MAX_TOKENS) -> "LangChainCompletion":
        """The system LLM from LLMSettings, at temperature 0 and with a fixed token cap."""
        recorder = LLMUsageRecorder()
        chat_model = build_chat_model(
            settings.provider,
            settings.model,
            temperature=0.0,
            base_url=settings.base_url,
            num_ctx=settings.num_ctx,
            reasoning=settings.reasoning,
            max_tokens=max_tokens,
            timeout_s=settings.timeout_s,
            usage_recorder=recorder,
        )
        return cls(chat_model, usage_recorder=recorder)

    def __call__(self, request: LLMRequest) -> LLMOutput:
        """
        Ask the model.

        Raises:
            REQUEST_ERRORS: When the request itself fails (nothing to record)
        """
        recorder = LLMUsageRecorder(default_operation=request.operation)
        runnable = with_structured_output(self._chat_model, request.schema, include_raw=True)
        config = {"callbacks": [recorder], "metadata": {OPERATION_METADATA_KEY: request.operation}}
        started = time.perf_counter()
        raw = parsed = None
        error: Optional[str] = None
        try:
            result = runnable.invoke(request.messages(), config=config)
        except REQUEST_ERRORS:
            raise
        except Exception as exc:  # the model answered, but not with valid JSON for the schema
            error = f"{type(exc).__name__}: {exc}"
        else:
            raw, parsed = result.get("raw"), result.get("parsed")
            if result.get("parsing_error") is not None:
                error = f"{type(result['parsing_error']).__name__}: {result['parsing_error']}"
            elif parsed is None:
                error = "no structured answer"
        wall_s = time.perf_counter() - started

        records = recorder.records
        record = records[-1] if records else None
        usage = getattr(raw, "usage_metadata", None) or {}
        content = getattr(raw, "content", None)
        return LLMOutput(
            parsed=parsed.model_dump() if isinstance(parsed, BaseModel) else parsed,
            content=content if isinstance(content, str) else None,
            error=error[:MAX_ERROR_CHARS] if error else None,
            latency_s=record.latency_s if record else wall_s,
            input_tokens=record.input_tokens if record else usage.get("input_tokens"),
            output_tokens=record.output_tokens if record else usage.get("output_tokens"),
            cached_input_tokens=_cached_input_tokens(raw),
            cost_usd=record.cost_usd if record else None,
        )


# =============================================================================
# Grounding rules (pure functions)
# =============================================================================


def candidate_ids(count: int) -> List[str]:
    """The ids shown for `count` books: B1, B2, ..., in the order shown."""
    return [f"B{position}" for position in range(1, count + 1)]


def normalize_id(raw: Any) -> str:
    """An id as the model wrote it, trimmed, without brackets, in upper case."""
    return str(raw).strip().strip("[]").strip().upper()


def check_ids(returned: Sequence[Any], shown: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split the ids the model returned.

    Args:
        returned: Ids as the model wrote them, in its order
        shown: Ids shown in the prompt

    Returns:
        (valid: distinct shown ids in the model's order,
         unknown: ids that were not shown, as written,
         duplicates: repeats of a valid id, as written)
    """
    allowed = set(shown)
    valid: List[str] = []
    unknown: List[str] = []
    duplicates: List[str] = []
    for raw in returned:
        book_id = normalize_id(raw)
        if book_id not in allowed:
            unknown.append(str(raw))
        elif book_id in valid:
            duplicates.append(str(raw))
        else:
            valid.append(book_id)
    return valid, unknown, duplicates


def complete_order(chosen: Sequence[int], count: int, top_n: int = TOP_N) -> Tuple[List[int], int]:
    """
    Final order of `count` candidates: the chosen ones first, then the rest
    in the order they were shown.

    Args:
        chosen: Indices (0-based, distinct) of the model's valid picks, best first
        count: Number of candidates shown
        top_n: Size of the list the user sees

    Returns:
        (order: every index once, filled: positions of the top n taken from
         the shown order because the model returned fewer valid picks)
    """
    picked = set(chosen)
    order = list(chosen) + [index for index in range(count) if index not in picked]
    filled = max(0, min(top_n, count) - len(chosen))
    return order, filled


# =============================================================================
# Outcomes
# =============================================================================


@dataclass(frozen=True)
class ListwiseOutcome:
    """
    What one listwise call returned and how it was turned into an order.

    Attributes:
        query_text: The query
        books: The books shown, in the order shown
        shown_ids: Ids shown in the prompt (B1..Bn), same order
        returned_ids: Ids as the model wrote them, in its order
        valid_ids: Distinct shown ids in the model's order
        unknown_ids: Returned ids that were not shown (dropped)
        duplicate_ids: Repeated ids (dropped)
        order: Final order, as indices into the books shown
        filled: Top-n positions completed from the shown order
        reasons: Valid id -> the model's sentence
        output: The model's answer and the usage of the call
    """

    query_text: str
    books: List[Book]
    shown_ids: List[str]
    returned_ids: List[str]
    valid_ids: List[str]
    unknown_ids: List[str]
    duplicate_ids: List[str]
    order: List[int]
    filled: int
    reasons: Dict[str, str]
    output: LLMOutput

    @property
    def ranked_books(self) -> List[Book]:
        """Every book shown, in the final order (the model's picks first)."""
        return [self.books[index] for index in self.order]

    @property
    def completed(self) -> bool:
        """True when the top n had to be completed from the shown order."""
        return self.filled > 0

    @property
    def failed(self) -> bool:
        """True when the answer could not be used at all."""
        return self.output.error is not None


@dataclass(frozen=True)
class ClosedBookOutcome:
    """The books a closed-book call recommended, as the model named them."""

    query_text: str
    recommendations: List[Dict[str, str]]  # title, author, reason
    output: LLMOutput

    @property
    def failed(self) -> bool:
        return self.output.error is not None


def _recommendations(output: LLMOutput) -> List[Dict[str, Any]]:
    """The recommendation items of a usable answer; none when the answer is unusable."""
    if output.error is not None or not output.parsed:
        return []
    return list(output.parsed.get("recommendations") or [])


# =============================================================================
# The recommender
# =============================================================================


def book_entry(book_id: str, book: Book) -> str:
    """A catalog book as shown in the listwise prompt."""
    return format_book_entry(
        book_id,
        title=book.title,
        authors=book.authors,
        year=book.get_published_year(),
        language=book.language,
        categories=book.categories,
        description=book.description,
    )


def listwise_request(query_text: str, books: Sequence[Book], operation: str, top_n: int = TOP_N) -> LLMRequest:
    """The listwise request for a query and the books to choose from (in the order shown)."""
    entries = [book_entry(book_id, book) for book_id, book in zip(candidate_ids(len(books)), books)]
    system, user = build_listwise_prompt(query_text, entries, top_n)
    return LLMRequest(operation, query_text, system, user, CandidateRecommendationsLLM)


def closed_book_request(query_text: str, top_n: int = TOP_N) -> LLMRequest:
    """The closed-book request for a query."""
    system, user = build_closed_book_prompt(query_text, top_n)
    return LLMRequest(CLOSED_BOOK_OPERATION, query_text, system, user, BookRecommendationsLLM)


class LLMRecommender:
    """
    The system LLM used as a recommender: listwise over given books, or closed book.

    Attributes:
        model_name: Model id, recorded with every output
        prompt_version: Version of the recommender prompts
        top_n: Books recommended per query
    """

    def __init__(self, complete: Completion, model_name: str, *, top_n: int = TOP_N) -> None:
        """
        Args:
            complete: Answers LLMRequests (LangChainCompletion, a cache, a double)
            model_name: Model id of the completion
            top_n: Books to recommend per query
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        self.complete = complete
        self.model_name = model_name
        self.prompt_version = RECOMMENDER_PROMPT_VERSION
        self.top_n = top_n

    @classmethod
    def from_settings(cls, settings: LLMSettings) -> "LLMRecommender":
        """The system LLM of LLMSettings (temperature 0, RECOMMENDER_MAX_TOKENS)."""
        return cls(LangChainCompletion.from_settings(settings), settings.model)

    def rank_books(self, query_text: str, books: Sequence[Book], operation: str = RERANK_OPERATION) -> ListwiseOutcome:
        """
        Ask the model to choose the top n of `books` for the query (R1, R2).

        Args:
            query_text: The user's query
            books: The books to choose from, in the order they are shown
            operation: Label of the call (RERANK_OPERATION or LONG_CONTEXT_OPERATION)

        Returns:
            The outcome; outcome.order ranks every book (picks first)
        """
        request = listwise_request(query_text, books, operation, self.top_n)
        output = self.complete(request)
        items = _recommendations(output)
        shown = candidate_ids(len(books))
        returned = [str(item.get("id", "")) for item in items]
        valid, unknown, duplicates = check_ids(returned, shown)
        position = {book_id: index for index, book_id in enumerate(shown)}
        order, filled = complete_order([position[book_id] for book_id in valid], len(books), self.top_n)
        reasons: Dict[str, str] = {}
        for item in items:
            book_id = normalize_id(item.get("id", ""))
            if book_id in position and book_id not in reasons:
                reasons[book_id] = " ".join(str(item.get("reason", "")).split())
        return ListwiseOutcome(
            query_text=query_text,
            books=list(books),
            shown_ids=shown,
            returned_ids=returned,
            valid_ids=valid,
            unknown_ids=unknown,
            duplicate_ids=duplicates,
            order=order,
            filled=filled,
            reasons=reasons,
            output=output,
        )

    def recommend_closed_book(self, query_text: str) -> ClosedBookOutcome:
        """Ask the model for real books (title and author) for the query, without a catalog (R3)."""
        output = self.complete(closed_book_request(query_text, self.top_n))
        items = _recommendations(output)
        recommendations = [
            {key: " ".join(str(item.get(key, "")).split()) for key in ("title", "author", "reason")}
            for item in items
        ]
        return ClosedBookOutcome(query_text=query_text, recommendations=recommendations, output=output)


class LLMListwiseReranker(Reranker):
    """
    Reranker port backed by one listwise LLM call (R1: the LLM recommends
    from the retrieved candidates).

    SearchService sends it the top `rerank_depth` fused candidates. The
    model's picks come first in its order, then the other candidates in
    fused order. The model returns an order, not a score, so rerank_score
    is 1 / new rank. An unusable answer keeps the fused order (with a
    warning); a failed request raises RuntimeError, and SearchService then
    keeps the fused order itself.
    """

    def __init__(
        self,
        recommender: LLMRecommender,
        observer: Optional[Callable[[ListwiseOutcome], None]] = None,
    ) -> None:
        """
        Args:
            recommender: The LLM recommender
            observer: Optional hook that receives every outcome (evaluation
                only, to count invalid ids and completed lists); None in
                production
        """
        self._recommender = recommender
        self.observer = observer

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Re-order the candidates with one LLM call.

        Returns:
            New SearchResult objects, every input once, with rank and
            rerank_score set; fusion scores unchanged

        Raises:
            RuntimeError: If the request to the model fails
        """
        if not results:
            return []
        try:
            outcome = self._recommender.rank_books(query_text, [result.book for result in results])
        except Exception as error:
            raise RuntimeError(f"LLM reranking failed: {type(error).__name__}: {error}") from error
        if outcome.failed:
            logger.warning("LLM reranker answer unusable (%s); keeping the fused order", outcome.output.error)
        if self.observer is not None:
            self.observer(outcome)
        return [
            replace(results[index], rank=position, rerank_score=1.0 / position)
            for position, index in enumerate(outcome.order, start=1)
        ]

    def get_model_name(self) -> str:
        """Model id of the LLM."""
        return self._recommender.model_name

    def is_ready(self) -> bool:
        """Always True: the model is remote and nothing is loaded locally."""
        return True
