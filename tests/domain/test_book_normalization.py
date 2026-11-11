"""Tests for the catalog normalisation rules (app/domain/book_normalization.py)."""

import pytest

from app.domain.book_normalization import (
    author_surname,
    clean_title,
    clean_isbn13,
    clean_subjects,
    fold_accents,
    is_plausible_year,
    normalize_author_name,
    parse_year,
    strip_markup,
    title_key,
    to_iso639_1,
    truncate_text,
)


class TestLanguageCodes:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("es", "es"), ("spa", "es"), ("Spanish", "es"), ("Español", "es"), ("es-ES", "es"),
            ("pt_BR", "pt"), ("zh-Hans", "zh"), ("/languages/fre", "fr"), ("ger", "de"),
            ("dut", "nl"), ("cat", "ca"), ("jpn", "ja"), ("ara", "ar"), ("rus", "ru"),
        ],
    )
    def test_known_codes_and_names(self, value, expected):
        assert to_iso639_1(value) == expected

    @pytest.mark.parametrize("value", [None, "", "mul", "und", "zxx", "klingon"])
    def test_unknown_or_not_a_language(self, value):
        assert to_iso639_1(value) is None


class TestYears:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1813-01-28T00:00:00Z", 1813), ("1605", 1605), (1999, 1999), ("c. 1850", 1850),
            ("May 1999", 1999), ("[1923?]", 1923), ("-0800-01-01T00:00:00Z", -800), ("", None),
            (None, None), ("unknown", None),
        ],
    )
    def test_parse_year(self, value, expected):
        assert parse_year(value) == expected

    def test_plausible_year(self):
        assert is_plausible_year(1605, 2027)
        assert not is_plausible_year(-800, 2027)
        assert not is_plausible_year(20013, 2027)
        assert not is_plausible_year(None, 2027)


class TestAuthors:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Austen, Jane, 1775-1817", "Jane Austen"),
            ("Doyle, Arthur Conan, Sir, 1859-1930", "Arthur Conan Doyle"),
            ("Wells, H. G. (Herbert George), 1866-1946", "H. G. Wells"),
            ("Cervantes Saavedra, Miguel de, 1547-1616", "Miguel de Cervantes Saavedra"),
            ("Miguel de Cervantes", "Miguel de Cervantes"),
            ("Plato", "Plato"),
            ("Homer, active 8th century B.C.", "Homer"),
            ("Homer, 751? BCE-651? BCE", "Homer"),
            ("50 Cent", "50 Cent"),
            ("Martin Luther King, Jr.", "Martin Luther King"),
            ("  Gabriel   García Márquez ", "Gabriel García Márquez"),
            ("", None),
        ],
    )
    def test_normalize_author_name(self, raw, expected):
        assert normalize_author_name(raw) == expected

    def test_surname_is_folded_last_word(self):
        assert author_surname("Gabriel García Márquez") == "marquez"
        assert author_surname("Martin Luther King Jr.") == "king"
        assert author_surname("村上春樹") == "村上春樹"


class TestSubjects:
    def test_splits_lcsh_and_drops_chronological_and_housekeeping(self):
        subjects = [
            "England -- Social life and customs -- 19th century -- Fiction",
            "Accessible book", "Protected DAISY", "nyt:hardcover-fiction=2008-01-01",
            "Category: British Literature", "Browsing: Fiction", "PR", "Love stories",
        ]
        assert clean_subjects(subjects) == [
            "England", "Social life and customs", "Fiction", "British Literature", "Love stories",
        ]

    def test_caps_and_deduplicates_case_insensitively(self):
        subjects = [f"Topic {i}" for i in range(20)] + ["topic 1"]
        cleaned = clean_subjects(subjects, max_count=10)
        assert len(cleaned) == 10
        assert cleaned[0] == "Topic 0"


class TestDescriptions:
    def test_strip_markup(self):
        raw = (
            "<p>A <b>novel</b> by [[Miguel de Cervantes|Cervantes]] ( ; ) about a "
            "[knight](https://example.org) &amp; his squire.[1]</p>\n\n"
            "----------\n**Also contained in:**\n- [Other](/works/OL1W)"
        )
        assert strip_markup(raw) == "A novel by Cervantes about a knight & his squire."

    def test_strip_zero_width_characters(self):
        assert strip_markup("Don Quijote​ es una novela") == "Don Quijote es una novela"

    def test_truncate_at_sentence_end(self):
        text = "First sentence here. " * 100
        cut = truncate_text(text, 1200)
        assert len(cut) <= 1200
        assert cut.endswith(".")

    def test_truncate_without_sentence_end_adds_ellipsis(self):
        text = "word " * 400
        cut = truncate_text(text, 1200)
        assert len(cut) <= 1200 and cut.endswith("...")

    def test_short_text_unchanged(self):
        assert truncate_text("Short.", 1200) == "Short."


class TestMatchKeys:
    def test_fold_accents(self):
        assert fold_accents("Cien años de soledad, García") == "Cien anos de soledad, Garcia"

    @pytest.mark.parametrize(
        "title, expected",
        [
            ("The Time Machine: An Invention", "time machine"),
            ("Frankenstein; or, The Modern Prometheus", "frankenstein"),
            ("Cien años de soledad", "cien anos de soledad"),
            ("El ingenioso hidalgo don Quijote de la Mancha", "ingenioso hidalgo don quijote de la mancha"),
            ("L'Étranger", "etranger"),
        ],
    )
    def test_title_key(self, title, expected):
        assert title_key(title) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("978-84-376-0494-7", "9788437604947"),
            ("0-306-40615-2", "9780306406157"),
            ("9788437604948", None),  # wrong check digit
            ("12345", None),
        ],
    )
    def test_clean_isbn13(self, value, expected):
        assert clean_isbn13(value) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Alterit&agrave;", "Alterità"),
        ("Picciola : $b The prisoner of Fenestrella", "Picciola : The prisoner of Fenestrella"),
        ("  Don  Quijote /", "Don Quijote"),
    ],
)
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected
