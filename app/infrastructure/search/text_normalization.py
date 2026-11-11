"""
Text normalization and tokenization for lexical (BM25) search.

The catalog is bilingual (English and Spanish), so a plain
``text.lower().split()`` tokenizer has three problems:

- accents: "programación" and "programacion" are different tokens, and users
  often type Spanish without accents;
- punctuation: "python," and "python" are different tokens;
- stopwords: in short queries, words like "de", "la" or "the" carry as much
  weight as the content words.

This module offers two named tokenizers behind a single lookup
(``get_tokenizer``), so the BM25 repository can persist the name of the one
it used and an ablation can compare both:

- ``"whitespace"``: the legacy behaviour, exactly ``text.lower().split()``.
- ``"normalized"``: lowercase, fold accents, split on word characters and
  drop Spanish/English stopwords.

The same tokenizer must be applied to documents and queries; otherwise
their tokens would not match.

Pure Python (``unicodedata`` and ``re``), no external dependency.
"""

import re
import unicodedata
from typing import Callable, Dict, FrozenSet, List

# Tokenizer names (the values persisted with a BM25 index).
WHITESPACE_TOKENIZER = "whitespace"
NORMALIZED_TOKENIZER = "normalized"

Tokenizer = Callable[[str], List[str]]

_ENYE = frozenset("ñÑ")

# Runs of unicode word characters (letters, digits, underscore).
_WORD_PATTERN = re.compile(r"\w+")

# Spanish function words, already lowercased and accent-folded
# ("según" -> "segun", "había" -> "habia", "él"/"el" -> "el").
_SPANISH_STOPWORDS: FrozenSet[str] = frozenset(
    {
        # Articles and contractions
        "el", "la", "los", "las", "lo", "un", "una", "unos", "unas", "al", "del",
        # Prepositions ("sin" is left out: it is a content word in English)
        "a", "ante", "bajo", "con", "contra", "de", "desde", "durante", "en",
        "entre", "hacia", "hasta", "mediante", "para", "por", "segun",
        "sobre", "tras",
        # Conjunctions
        "y", "e", "o", "u", "ni", "pero", "sino", "que", "porque", "aunque",
        "si", "como", "cuando",
        # Pronouns, possessives and demonstratives
        "yo", "tu", "ella", "ello", "ellos", "ellas", "nosotros", "nosotras",
        "vosotros", "vosotras", "usted", "ustedes", "me", "te", "se", "nos",
        "os", "le", "les", "mi", "mis", "tus", "su", "sus", "nuestro",
        "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros",
        "vuestras", "este", "esta", "estos", "estas", "ese", "esa", "esos",
        "esas", "aquel", "aquella", "aquellos", "aquellas", "esto", "eso",
        "aquello", "quien", "quienes", "cual", "cuales", "cuyo", "cuya",
        # Auxiliaries: forms of ser, estar, haber
        # ("son" and "era" are left out: content words in English)
        "soy", "eres", "es", "somos", "fue", "fueron", "sea", "estan",
        "estaba", "he", "has", "ha", "hemos", "han", "habia", "hay",
    }
)

# English function words (lowercase; English has no accents to fold).
_ENGLISH_STOPWORDS: FrozenSet[str] = frozenset(
    {
        # Articles
        "a", "an", "the",
        # Prepositions
        "about", "at", "by", "for", "from", "in", "into", "of", "on", "over",
        "to", "under", "with", "within", "without", "between", "through",
        "during",
        # Conjunctions
        "and", "or", "but", "nor", "as", "if", "because", "than", "while",
        # Pronouns, possessives and demonstratives
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
        "his", "she", "her", "it", "its", "they", "them", "their", "this",
        "that", "these", "those", "what", "which", "who", "whom", "whose",
        # Auxiliaries and modals
        "am", "is", "are", "was", "were", "be", "been", "being", "do",
        "does", "did", "have", "has", "had", "will", "would", "shall",
        "should", "can", "could", "may", "might", "must",
    }
)

# Union of both languages: one set, so the tokenizer needs no language
# detection. Words that are function words in one language but content
# words in the other ("sin", "son", "era") are deliberately excluded.
STOPWORDS: FrozenSet[str] = _SPANISH_STOPWORDS | _ENGLISH_STOPWORDS


def fold_accents(text: str, keep_enye: bool = False) -> str:
    """
    Remove diacritics from text ("programación" -> "programacion").

    Uses Unicode NFKD decomposition, which splits an accented letter into
    its base letter plus a combining mark, and then drops the combining
    marks. NFKD also maps compatibility characters to their plain form
    (for example the ligature "ﬁ" becomes "fi").

    Args:
        text: Text to fold.
        keep_enye: If True, "ñ" and "Ñ" are preserved (in Spanish they are
            a separate letter, not an accented "n"). If False, they become
            "n" and "N".

    Returns:
        The text without combining marks.
    """
    if not keep_enye:
        return _strip_combining_marks(text)

    # NFC first, so a decomposed "n" + combining tilde becomes one "ñ" char.
    composed = unicodedata.normalize("NFC", text)
    return "".join(
        char if char in _ENYE else _strip_combining_marks(char)
        for char in composed
    )


def _strip_combining_marks(text: str) -> str:
    """Decompose text with NFKD and drop every combining mark."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def whitespace_tokenize(text: str) -> List[str]:
    """
    Legacy tokenizer: lowercase and split on whitespace.

    Punctuation stays attached to words ("python," != "python") and accents
    are kept ("programación" != "programacion"). Kept unchanged so existing
    indexes keep working and so it can serve as an ablation baseline.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


def normalized_tokenize(text: str) -> List[str]:
    """
    Bilingual tokenizer: lowercase, fold accents, split on words, drop stopwords.

    Steps:
    1. Lowercase.
    2. Fold accents, including "ñ" -> "n" (users often type "espana").
    3. Extract runs of word characters, which drops punctuation.
    4. Remove Spanish and English stopwords. If that would leave no tokens
       (a text made only of stopwords, e.g. "the"), the tokens are kept
       without stopword removal so the output is never emptied by it.

    Args:
        text: Text to tokenize (a document or a query).

    Returns:
        List of normalized tokens.
    """
    folded = fold_accents(text.lower())
    tokens = _WORD_PATTERN.findall(folded)
    content_tokens = [token for token in tokens if token not in STOPWORDS]
    return content_tokens if content_tokens else tokens


_TOKENIZERS: Dict[str, Tokenizer] = {
    WHITESPACE_TOKENIZER: whitespace_tokenize,
    NORMALIZED_TOKENIZER: normalized_tokenize,
}

AVAILABLE_TOKENIZERS: FrozenSet[str] = frozenset(_TOKENIZERS)


def get_tokenizer(name: str) -> Tokenizer:
    """
    Look up a tokenizer by name.

    Args:
        name: "whitespace" or "normalized".

    Returns:
        The tokenizer function (str -> list of tokens).

    Raises:
        ValueError: If the name is not a known tokenizer.
    """
    try:
        return _TOKENIZERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown tokenizer {name!r}. "
            f"Available: {sorted(AVAILABLE_TOKENIZERS)}"
        ) from None
