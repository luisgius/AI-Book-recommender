"""Explanation text: no citation markers, and a cut never falls inside a word or a marker."""

from uuid import uuid4

from app.domain.entities import Book
from app.infrastructure.llm.explanation_text import shorten_at_boundary, strip_citation_markers
from app.infrastructure.llm.guardrails import EXPLANATION_MAX_CHARS, validate_grounded_explanation
from app.infrastructure.llm.schemas import CitationLLM, GroundedExplanationLLM


def make_book() -> Book:
    return Book(
        id=uuid4(),
        title="Pattern Recognition and Machine Learning",
        authors=["Christopher Bishop"],
        description="A comprehensive introduction to machine learning and pattern recognition.",
        categories=["Computers"],
    )


class TestStripCitationMarkers:
    def test_complete_markers_in_both_bracket_styles_are_removed(self):
        text = "It covers machine learning [CITE: description] and pattern recognition (CITE: title)."
        assert strip_citation_markers(text) == "It covers machine learning and pattern recognition."

    def test_a_marker_cut_at_the_end_is_removed(self):
        for cut in ("[C", "[", "(CITE:", "[CITE: descr", "(cite"):
            text = f"The title is 'Pattern Recognition and Machine Learning' {cut}"
            assert strip_citation_markers(text) == "The title is 'Pattern Recognition and Machine Learning'"

    def test_ordinary_parentheses_are_kept(self):
        text = "A classic (first published in 2006) about Bayesian methods."
        assert strip_citation_markers(text) == text


class TestShortenAtBoundary:
    def test_a_text_that_fits_is_unchanged(self):
        assert shorten_at_boundary("Short text.", 50) == "Short text."

    def test_cuts_at_the_last_sentence_end(self):
        text = "First sentence is here. Second one is longer and does not fit in the limit at all."
        assert shorten_at_boundary(text, 40) == "First sentence is here."

    def test_cuts_at_a_word_boundary_with_an_ellipsis_when_no_sentence_ends(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = shorten_at_boundary(text, 30)
        assert len(result) <= 30 and result.endswith("…")
        assert result[:-1] in text and text[len(result) - 1] == " "  # a whole word before the ellipsis

    def test_never_cuts_inside_a_marker(self):
        text = "x" * 10 + " and it covers machine learning [CITE: description] which matters a lot"
        cut_at = text.index("[CITE") + 3  # the old blind cut would leave "[CI"
        result = shorten_at_boundary(text, cut_at)
        assert "[" not in result and len(result) <= cut_at

    def test_a_decimal_point_is_not_a_sentence_end(self):
        text = "Covers version 3.5 of the language in depth and much more besides"
        assert not shorten_at_boundary(text, 20).endswith("3.")


class TestStoredExplanation:
    def explanation(self, reasoning: str) -> GroundedExplanationLLM:
        return GroundedExplanationLLM(
            summary="A machine learning textbook.",
            reasoning=reasoning,
            citations=[CitationLLM(book_id="x", chunk_id="description",
                                   snippet="introduction to machine learning", relevance_score=0.9)],
            confidence=0.9,
        )

    def test_stored_text_has_no_markers(self):
        book = make_book()
        raw = self.explanation("It is an introduction to machine learning [CITE: description].")
        result = validate_grounded_explanation(raw, book, "machine learning", "m")
        assert result.text == "It is an introduction to machine learning."
        assert len(result.citations) == 1  # the citations are still returned separately

    def test_an_over_long_reasoning_ends_at_a_boundary_without_a_marker_fragment(self):
        """The case seen in the page: the 500-character cut fell inside '[CITE: ...]'."""
        book = make_book()
        sentence = "The book is titled 'Pattern Recognition and Machine Learning' [CITE: title]. "
        reasoning = (sentence * 8).strip()
        raw = self.explanation(reasoning)  # the schema's soft limit applies here
        result = validate_grounded_explanation(raw, book, "machine learning", "m")
        assert "CITE" not in result.text and "[" not in result.text and "(" not in result.text
        assert len(result.text) <= EXPLANATION_MAX_CHARS
        assert result.text.endswith(".")
