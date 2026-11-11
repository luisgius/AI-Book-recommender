"""
Normalisation rules for catalog records coming from several sources.

Pure functions over strings and numbers (standard library only), shared by
the source adapters and the ingestion pipeline:

- language codes: ISO 639-1/639-2/639-3, MARC codes, locale tags and English
  or native language names -> ISO 639-1 ("spa", "Spanish", "es-ES" -> "es");
- years: parse a year out of dates such as "1813-01-28" or "c. 1850" and
  check that it is plausible;
- author names: "Austen, Jane, 1775-1817" -> "Jane Austen";
- subjects: drop housekeeping tags, split Library of Congress headings, cap;
- descriptions: strip HTML/wiki/Markdown markup and truncate at a sentence;
- match keys: accent folding and a normalised title used for de-duplication.

The accent folding mirrors ``fold_accents`` in the search infrastructure
(``app/infrastructure/search/text_normalization.py``), which the domain must
not import.
"""

import html
import re
import unicodedata
from typing import Iterable, List, Optional

# ---------------------------------------------------------------- languages

# ISO 639-1 code -> other spellings (ISO 639-2/B, 639-2/T, 639-3, names).
_LANGUAGE_ALIASES = {
    "en": ("eng", "english"),
    "es": ("spa", "spanish", "castilian", "español", "espanol", "castellano"),
    "fr": ("fre", "fra", "french", "français", "francais"),
    "de": ("ger", "deu", "german", "deutsch"),
    "it": ("ita", "italian", "italiano"),
    "pt": ("por", "portuguese", "português", "portugues"),
    "ca": ("cat", "catalan", "català", "valencian"),
    "nl": ("dut", "nld", "dutch", "flemish", "nederlands"),
    "ru": ("rus", "russian"),
    "zh": ("chi", "zho", "chinese", "cmn", "mandarin"),
    "ja": ("jpn", "japanese"),
    "ar": ("ara", "arabic"),
    "la": ("lat", "latin"),
    "el": ("gre", "ell", "greek", "modern greek"),
    "pl": ("pol", "polish"),
    "sv": ("swe", "swedish"),
    "da": ("dan", "danish"),
    "no": ("nor", "norwegian"),
    "nb": ("nob", "norwegian bokmål", "bokmål"),
    "nn": ("nno", "norwegian nynorsk", "nynorsk"),
    "fi": ("fin", "finnish"),
    "hu": ("hun", "hungarian"),
    "cs": ("cze", "ces", "czech"),
    "sk": ("slo", "slk", "slovak"),
    "sl": ("slv", "slovenian", "slovene"),
    "hr": ("hrv", "croatian"),
    "sr": ("srp", "serbian"),
    "bs": ("bos", "bosnian"),
    "bg": ("bul", "bulgarian"),
    "ro": ("rum", "ron", "romanian"),
    "uk": ("ukr", "ukrainian"),
    "be": ("bel", "belarusian"),
    "tr": ("tur", "turkish"),
    "he": ("heb", "hebrew"),
    "yi": ("yid", "yiddish"),
    "fa": ("per", "fas", "persian", "farsi"),
    "hi": ("hin", "hindi"),
    "bn": ("ben", "bengali"),
    "ta": ("tam", "tamil"),
    "ur": ("urd", "urdu"),
    "ko": ("kor", "korean"),
    "vi": ("vie", "vietnamese"),
    "th": ("tha", "thai"),
    "id": ("ind", "indonesian"),
    "ms": ("may", "msa", "malay"),
    "tl": ("tgl", "tagalog", "filipino", "fil"),
    "sw": ("swa", "swahili"),
    "af": ("afr", "afrikaans"),
    "eo": ("epo", "esperanto"),
    "gl": ("glg", "galician", "galego"),
    "eu": ("baq", "eus", "basque", "euskara"),
    "cy": ("wel", "cym", "welsh"),
    "ga": ("gle", "irish"),
    "gd": ("gla", "scottish gaelic", "gaelic"),
    "br": ("bre", "breton"),
    "oc": ("oci", "occitan"),
    "is": ("ice", "isl", "icelandic"),
    "et": ("est", "estonian"),
    "lv": ("lav", "latvian"),
    "lt": ("lit", "lithuanian"),
    "sq": ("alb", "sqi", "albanian"),
    "mk": ("mac", "mkd", "macedonian"),
    "ka": ("geo", "kat", "georgian"),
    "hy": ("arm", "hye", "armenian"),
    "az": ("aze", "azerbaijani"),
    "kk": ("kaz", "kazakh"),
    "fy": ("fry", "frisian", "western frisian"),
    "ia": ("ina", "interlingua"),
    "mt": ("mlt", "maltese"),
    "lb": ("ltz", "luxembourgish"),
}

_ALIAS_TO_ISO = {alias: iso for iso, aliases in _LANGUAGE_ALIASES.items() for alias in aliases}
_ALIAS_TO_ISO.update({iso: iso for iso in _LANGUAGE_ALIASES})

# Codes that mean "no single language" (multiple, undetermined, no content).
_NOT_A_LANGUAGE = {"mul", "und", "zxx", "mis", "multiple languages", "undetermined"}


def to_iso639_1(value: Optional[str]) -> Optional[str]:
    """
    Map a language code or name to its ISO 639-1 code.

    Accepts ISO 639-1 ("es"), ISO 639-2/3 and MARC codes ("spa"), locale tags
    ("es-ES", "pt_BR", "zh-Hans"), Open Library keys ("/languages/spa") and
    English or native names ("Spanish", "Español").

    Args:
        value: The code or name as given by a source

    Returns:
        The two-letter code, or None if unknown or not a single language
    """
    if not value:
        return None
    text = value.strip().lower().rsplit("/", 1)[-1]
    if text in _NOT_A_LANGUAGE:
        return None
    if text in _ALIAS_TO_ISO:
        return _ALIAS_TO_ISO[text]
    base = re.split(r"[-_]", text, maxsplit=1)[0]
    return _ALIAS_TO_ISO.get(base)


# -------------------------------------------------------------------- years

_ISO_YEAR = re.compile(r"^\s*([+-]?)(\d{1,4})(?:-\d{1,2}){0,2}(?:T|\s|$)")
_ANY_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def parse_year(value: object) -> Optional[int]:
    """
    Extract a year from a date value.

    Handles integers, ISO dates ("1813-01-28", "-0800-01-01T00:00:00Z"), and
    free text with a four-digit year ("c. 1850", "May 1999", "[1923?]").

    Args:
        value: A date in any of those forms

    Returns:
        The year (negative for BCE ISO dates), or None if none is found
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    match = _ISO_YEAR.match(text)
    if match:
        year = int(match.group(2))
        return -year if match.group(1) == "-" else year
    match = _ANY_YEAR.search(text)
    return int(match.group(1)) if match else None


def is_plausible_year(year: Optional[int], max_year: int) -> bool:
    """
    Check that a publication year can be stored and is not in the future.

    The catalog stores dates as Python datetimes, so years before 1 CE are
    not representable; years after ``max_year`` are data errors.

    Args:
        year: Year to check
        max_year: Latest acceptable year (usually next year)

    Returns:
        True if 1 <= year <= max_year
    """
    return year is not None and 1 <= year <= max_year


# ------------------------------------------------------------------ authors

# A comma-separated part of a name, after the first, that contains a digit is
# a life date ("1775-1817", "751? BCE-651? BCE", "active 8th century B.C.").
_LIFE_DATES = re.compile(r"\d")
_ROLE = re.compile(r"\s*[\[(][^\])]*[\])]\s*")
_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}


def normalize_author_name(name: Optional[str]) -> Optional[str]:
    """
    Normalise an author name to "Given Surname" order.

    - drops life dates (comma-separated parts after the first that contain a
      digit) and bracketed roles or expansions
      ("Wells, H. G. (Herbert George), 1866-1946" -> "H. G. Wells");
    - turns "Surname, Given[, title]" into "Given Surname"
      ("Doyle, Arthur Conan, Sir, 1859-1930" -> "Arthur Conan Doyle");
    - collapses whitespace.

    Args:
        name: Name as given by a source

    Returns:
        The normalised name, or None if nothing is left
    """
    if not name:
        return None
    text = _ROLE.sub(" ", name).strip().strip(",;").strip()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    # The first part is the name itself ("50 Cent" keeps its digits); dates come after it.
    parts = parts[:1] + [part for part in parts[1:] if not _LIFE_DATES.search(part)]
    if not parts:
        return None
    if len(parts) >= 2 and parts[1].lower() not in _NAME_SUFFIXES:
        text = f"{parts[1]} {parts[0]}"
    else:
        text = parts[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def author_surname(name: str) -> str:
    """
    Accent-folded, lowercase surname of a normalised name (its last word).

    Names written without spaces (e.g. Chinese or Japanese) are returned whole.
    """
    tokens = [t for t in re.split(r"\s+", fold_accents(name.lower()).strip()) if t]
    while len(tokens) > 1 and tokens[-1].strip(".") in {s.strip(".") for s in _NAME_SUFFIXES}:
        tokens.pop()
    return re.sub(r"[^\w]", "", tokens[-1]) if tokens else ""


# ----------------------------------------------------------------- subjects

_HOUSEKEEPING = re.compile(
    r"^(?:accessible book|protected daisy|in library|lending library|"
    r"overdrive|large type books|internet archive wishlist|open library staff picks|"
    r"long now manual for civilization|reading level-|nyt:|award:|series:|"
    r"place:|person:|time:|collectionid:|ebooks?$|electronic books?$|"
    r"translations into |"
    r"fiction, general$|general$|other$)",
    re.IGNORECASE,
)
_CHRONOLOGICAL = re.compile(r"^(?:\d{1,4}(?:th|st|nd|rd)? century|\d{3,4}(?:-\d{2,4})?s?)$", re.IGNORECASE)
_PREFIXES_TO_STRIP = re.compile(r"^(?:browsing|category|subject|genre)\s*:\s*", re.IGNORECASE)


def clean_subjects(subjects: Iterable[str], max_count: int = 10) -> List[str]:
    """
    Clean and cap a list of subjects or genres.

    - strips prefixes such as "Browsing: " (Gutenberg bookshelves);
    - splits Library of Congress headings on " -- " and drops chronological
      subdivisions ("England -- Social life -- 19th century -- Fiction" ->
      "England", "Social life", "Fiction");
    - drops housekeeping tags (e.g. "Accessible book", "nyt:...") and codes;
    - capitalises all-lowercase subjects (Wikidata genre labels) so the same
      subject is spelled the same way across sources;
    - removes case-insensitive duplicates and keeps at most ``max_count``.

    Args:
        subjects: Raw subjects in source order
        max_count: Maximum number of subjects kept

    Returns:
        Cleaned subjects in first-seen order
    """
    cleaned: List[str] = []
    seen = set()
    for subject in subjects:
        if not subject:
            continue
        subject = _PREFIXES_TO_STRIP.sub("", subject.strip())
        for part in re.split(r"\s+--\s+", subject):
            part = re.sub(r"\s+", " ", part).strip(" .;,")
            if not part or len(part) > 80 or _HOUSEKEEPING.match(part):
                continue
            if _CHRONOLOGICAL.match(part) or not re.search(r"[^\W\d_]", part):
                continue
            if re.fullmatch(r"[A-Z]{1,3}\d*(?:\.\d+)?", part):  # classification codes ("PR", "PZ3")
                continue
            if part == part.lower():
                part = part[0].upper() + part[1:]  # "science fiction" -> "Science fiction"
            key = part.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(part)
            if len(cleaned) >= max_count:
                return cleaned
    return cleaned


# ------------------------------------------------------------- descriptions

_TAG = re.compile(r"<[^>]+>")
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_EXTERNAL_LINK = re.compile(r"\[(?:https?://\S+)\s+([^\]]*)\]")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
_FOOTNOTE_DEF = re.compile(r"^\s*\[\d+\]:\s*\S+.*$", re.MULTILINE)
_FOOTNOTE_REF = re.compile(r"\[(?:\d+|[a-z])\]")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_URL = re.compile(r"https?://\S+")
_EMPTY_PARENS = re.compile(r"\(\s*[;,:]?\s*\)")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")


def strip_markup(text: Optional[str]) -> str:
    """
    Turn a description into plain text.

    Removes HTML tags and entities, wiki and Markdown links (keeping their
    label), templates, footnote markers, bare URLs, emphasis marks and the
    empty parentheses left by removed pronunciations, and collapses all
    whitespace. Open Library's "----------" separator (followed by "also
    contained in" lists) ends the description.

    Args:
        text: Raw description

    Returns:
        Plain text (possibly empty)
    """
    if not text:
        return ""
    text = text.split("\n----------", 1)[0].split("\n-----", 1)[0]
    text = html.unescape(html.unescape(text))
    text = _ZERO_WIDTH.sub("", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>", " ", text)
    text = _TAG.sub(" ", text)
    text = _FOOTNOTE_DEF.sub(" ", text)
    for _ in range(2):
        text = _TEMPLATE.sub(" ", text)
    text = _WIKI_LINK.sub(r"\1", text)
    text = _EXTERNAL_LINK.sub(r"\1", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _FOOTNOTE_REF.sub("", text)
    text = _URL.sub(" ", text)
    text = re.sub(r"'{2,}|\*{1,3}(?=\S)|(?<=\S)\*{1,3}|__", "", text)
    text = _EMPTY_PARENS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


_SENTENCE_END = re.compile(r"[.!?。！？](?:[\"'”»)]*)(?=\s|$)")


def truncate_text(text: str, max_chars: int = 1200) -> str:
    """
    Shorten a text to at most ``max_chars``, cutting at a sentence end.

    If no sentence ends in the second half of the window, the cut falls on
    the last space and "..." is appended.

    Args:
        text: Plain text
        max_chars: Maximum length of the result

    Returns:
        The text, unchanged if already short enough
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    if ends and ends[-1] >= max_chars // 2:
        return window[: ends[-1]].strip()
    cut = window[: max_chars - 3].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "..."


# ------------------------------------------------------------------- titles

_MARC_SUBFIELD = re.compile(r"\s*\$[a-z]\s+")


def clean_title(title: Optional[str]) -> str:
    """
    Clean a title for display: HTML entities decoded ("Alterit&agrave;" ->
    "Alterità"), MARC subfield markers removed ("Picciola : $b The prisoner"
    -> "Picciola : The prisoner"), whitespace collapsed, trailing "/" or ":"
    removed.
    """
    if not title:
        return ""
    text = html.unescape(title)
    text = _MARC_SUBFIELD.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" /:;,").strip()


# --------------------------------------------------------------- match keys

def fold_accents(text: str) -> str:
    """Remove diacritics with NFKD decomposition ("Quijote" stays, "Pérez" -> "Perez")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


_LEADING_ARTICLES = {
    "the", "a", "an", "el", "la", "los", "las", "lo", "un", "una", "le", "les",
    "l", "der", "die", "das", "ein", "eine", "il", "i", "gli", "o", "os", "as",
    "de", "het", "een",
}
_SUBTITLE = re.compile(r"\s*(?:[:;/(\[]|\s[-–—]\s|,\s*or[,\s])")


def title_key(title: str) -> str:
    """
    Normalised title used to match the same work across sources.

    Lowercase, accents folded, subtitle dropped (after ":", ";", "/", "(",
    " - " or ", or"), punctuation removed and one leading article dropped:
    "The Time Machine: An Invention" -> "time machine".

    Args:
        title: Title as stored

    Returns:
        The key (may be empty for titles made only of punctuation)
    """
    text = fold_accents(title.lower())
    head = _SUBTITLE.split(text, maxsplit=1)[0]
    if head.strip():
        text = head
    text = re.sub(r"[^\w\s]", " ", text.replace("'", " "))
    tokens = text.split()
    if len(tokens) > 1 and tokens[0] in _LEADING_ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def clean_isbn13(value: Optional[str]) -> Optional[str]:
    """
    Return a valid ISBN-13 as 13 digits, converting ISBN-10 when needed.

    Args:
        value: ISBN with or without hyphens/spaces

    Returns:
        The 13-digit ISBN, or None if the value is not a valid ISBN
    """
    if not value:
        return None
    raw = re.sub(r"[^0-9Xx]", "", value)
    if len(raw) == 10:
        if not _valid_isbn10(raw):
            return None
        body = "978" + raw[:9]
        return body + _isbn13_check_digit(body)
    if len(raw) == 13 and raw.isdigit() and raw[:3] in {"978", "979"}:
        return raw if _isbn13_check_digit(raw[:12]) == raw[12] else None
    return None


def _isbn13_check_digit(first12: str) -> str:
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(first12))
    return str((10 - total % 10) % 10)


def _valid_isbn10(raw: str) -> bool:
    if not raw[:9].isdigit() or not (raw[9].isdigit() or raw[9] in "Xx"):
        return False
    total = sum((10 - i) * int(d) for i, d in enumerate(raw[:9]))
    total += 10 if raw[9] in "Xx" else int(raw[9])
    return total % 11 == 0
