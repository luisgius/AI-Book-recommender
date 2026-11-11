"""
LLM relevance judge: grades how relevant a book is to a test query (0-3).

Infrastructure adapter for the RelevanceJudge protocol defined in
app/evaluation/relevance_judge.py (the protocol is structural, so this module
does not import the evaluation package). It is used only by the evaluation,
to grade (query, book) pairs that have no judgment in the test collection.

How a judgment is made (pointwise, one judge, as in UMBRELA):
1. build_relevance_judge_prompt (prompts_judge.py) shows the query and the
   catalog fields of the book; it never shows an existing grade, a system
   name, a rank or a score.
2. The chat model returns a RelevanceGradeLLM (schemas_judge.py): a
   one-sentence rationale and the grade, via structured output.

The chat model is built with build_chat_model, the same factory the system
LLM client uses, from JudgeSettings: temperature 0, thinking off (or its
lowest effort for models that cannot switch it off) and a small token cap,
so a judgment is reproducible and cannot run away. Its calls are recorded
by usage_recorder under the operation "relevance_judge".
"""

from typing import Optional, Tuple

from app.domain.entities import Book
from app.infrastructure.config.settings import JudgeSettings

from .langchain_llm_client import build_chat_model
from .prompts_judge import RELEVANCE_JUDGE_PROMPT_VERSION, build_relevance_judge_prompt
from .schemas_judge import RelevanceGradeLLM
from .structured_output import with_structured_output
from .usage import LLMUsageRecorder

# Operation label of the judge's LLM calls in the usage records.
RELEVANCE_JUDGE_OPERATION = "relevance_judge"


def build_book_prompt(query_text: str, book: Book) -> str:
    """The relevance-judging prompt for a query and a catalog book."""
    return build_relevance_judge_prompt(
        query_text=query_text,
        title=book.title,
        authors=book.authors,
        categories=book.categories,
        year=book.get_published_year(),
        language=book.language,
        description=book.description,
    )


class LangChainRelevanceJudge:
    """
    Grades (query, book) pairs with a LangChain chat model.

    Attributes:
        model_name: Judge model id, recorded with every judgment
        prompt_version: Version of the judging prompt, recorded likewise
        usage_recorder: Records latency and tokens of every judge call when
            the judge is built with from_settings; None otherwise
    """

    def __init__(
        self,
        chat_model,
        model_name: str,
        usage_recorder: Optional[LLMUsageRecorder] = None,
    ) -> None:
        """
        Args:
            chat_model: LangChain chat model that supports structured output
            model_name: Model id to record with each judgment
            usage_recorder: The recorder attached to chat_model, if any
        """
        self.model_name = model_name
        self.prompt_version = RELEVANCE_JUDGE_PROMPT_VERSION
        self.usage_recorder = usage_recorder
        self._structured_model = with_structured_output(chat_model, RelevanceGradeLLM)

    @classmethod
    def from_settings(cls, settings: JudgeSettings) -> "LangChainRelevanceJudge":
        """Build the judge from JudgeSettings (temperature 0, thinking off, token cap)."""
        recorder = LLMUsageRecorder(default_operation=RELEVANCE_JUDGE_OPERATION)
        chat_model = build_chat_model(
            settings.provider,
            settings.model,
            temperature=settings.temperature,
            base_url=settings.base_url,
            num_ctx=settings.num_ctx,
            reasoning=settings.reasoning,
            max_tokens=settings.max_tokens,
            timeout_s=settings.timeout_s,
            usage_recorder=recorder,
        )
        return cls(chat_model, settings.model, usage_recorder=recorder)

    def judge(self, query_text: str, book: Book) -> Tuple[int, str]:
        """
        Grade one (query, book) pair.

        Args:
            query_text: The test query
            book: The catalog book

        Returns:
            (grade 0-3, one-sentence rationale)

        Raises:
            Exception: Whatever the model call raises (connection error,
                timeout, output that does not fit the schema). The caller
                decides whether to retry; nothing is cached on failure.
        """
        prompt = build_book_prompt(query_text, book)
        result = self._structured_model.invoke([{"role": "user", "content": prompt}])
        return result.grade, " ".join(result.rationale.split())
