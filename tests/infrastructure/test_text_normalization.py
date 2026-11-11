"""
Unit tests for the text normalization module used by BM25 search.

They cover accent folding, the "ñ" option, punctuation handling, stopword
removal, the all-stopword fallback and the tokenizer lookup.
"""

import unicodedata

import pytest

from app.infrastructure.search.text_normalization import (
    AVAILABLE_TOKENIZERS,
    NORMALIZED_TOKENIZER,
    STOPWORDS,
    WHITESPACE_TOKENIZER,
    fold_accents,
    get_tokenizer,
    normalized_tokenize,
    whitespace_tokenize,
)


class TestFoldAccents:
    """Tests for fold_accents."""

    def test_removes_spanish_accents(self):
        assert fold_accents("programación análisis histórica") == (
            "programacion analisis historica"
        )

    def test_removes_accents_on_uppercase(self):
        assert fold_accents("ÁRBOL Él ÚLTIMO") == "ARBOL El ULTIMO"

    def test_removes_other_diacritics(self):
        assert fold_accents("pingüino café naïve") == "pinguino cafe naive"

    def test_folds_enye_by_default(self):
        assert fold_accents("España niño Ñandú") == "Espana nino Nandu"

    def test_keep_enye_preserves_enye_and_folds_other_accents(self):
        assert fold_accents("España Ñandú", keep_enye=True) == "España Ñandu"

    def test_keep_enye_handles_decomposed_input(self):
        # "n" followed by a combining tilde (U+0303) is the decomposed form of "ñ"
        decomposed = unicodedata.normalize("NFD", "año")
        assert len(decomposed) == 4

        assert fold_accents(decomposed, keep_enye=True) == "año"
        assert fold_accents(decomposed) == "ano"

    def test_plain_ascii_is_unchanged(self):
        assert fold_accents("machine learning, 2nd ed.") == "machine learning, 2nd ed."

    def test_empty_string(self):
        assert fold_accents("") == ""


class TestStopwords:
    """Tests for the embedded stopword set."""

    def test_contains_spanish_and_english_function_words(self):
        for word in ["de", "la", "los", "para", "y", "the", "of", "and", "with", "is"]:
            assert word in STOPWORDS

    def test_stopwords_are_stored_folded_and_lowercase(self):
        for word in STOPWORDS:
            assert word == fold_accents(word.lower())

    def test_cross_language_content_words_are_not_stopwords(self):
        # Spanish function words that are content words in English
        for word in ["sin", "son", "era"]:
            assert word not in STOPWORDS


class TestWhitespaceTokenizer:
    """The legacy tokenizer must keep its exact behaviour."""

    def test_is_lower_split(self):
        text = "Hello World, This is a TEST!"
        assert whitespace_tokenize(text) == text.lower().split()
        assert whitespace_tokenize(text) == ["hello", "world,", "this", "is", "a", "test!"]

    def test_keeps_accents_and_stopwords(self):
        assert whitespace_tokenize("Programación de la IA") == [
            "programación",
            "de",
            "la",
            "ia",
        ]


class TestNormalizedTokenizer:
    """Tests for the normalized tokenizer."""

    def test_lowercases_and_folds_accents(self):
        assert normalized_tokenize("Programación Análisis") == ["programacion", "analisis"]

    def test_folds_enye(self):
        assert normalized_tokenize("Novela histórica España") == [
            "novela",
            "historica",
            "espana",
        ]

    def test_strips_punctuation(self):
        assert normalized_tokenize("python, (pandas); numpy! \"data-science\"") == [
            "python",
            "pandas",
            "numpy",
            "data",
            "science",
        ]

    def test_removes_spanish_stopwords(self):
        assert normalized_tokenize("programación en python para análisis de datos") == [
            "programacion",
            "python",
            "analisis",
            "datos",
        ]

    def test_removes_english_stopwords(self):
        assert normalized_tokenize("Spanish civil war novels in English translation") == [
            "spanish",
            "civil",
            "war",
            "novels",
            "english",
            "translation",
        ]

    def test_keeps_numbers(self):
        assert normalized_tokenize("Siglo XIX, 1984") == ["siglo", "xix", "1984"]

    def test_all_stopword_text_falls_back_to_unfiltered_tokens(self):
        assert normalized_tokenize("The") == ["the"]
        assert normalized_tokenize("de la, y el") == ["de", "la", "y", "el"]

    def test_empty_and_punctuation_only_text(self):
        assert normalized_tokenize("") == []
        assert normalized_tokenize("  ¿?!  ") == []

    def test_accented_and_unaccented_queries_give_same_tokens(self):
        assert normalized_tokenize("introducción a la inteligencia artificial") == (
            normalized_tokenize("introduccion a la inteligencia artificial")
        )


class TestGetTokenizer:
    """Tests for the tokenizer lookup."""

    def test_returns_named_tokenizers(self):
        assert get_tokenizer(WHITESPACE_TOKENIZER) is whitespace_tokenize
        assert get_tokenizer(NORMALIZED_TOKENIZER) is normalized_tokenize

    def test_available_tokenizers(self):
        assert AVAILABLE_TOKENIZERS == {"whitespace", "normalized"}

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            get_tokenizer("stemmed")
