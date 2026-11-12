"""
Clean-up of the explanation text written by the model.

The explanation prompt asks the model to mark its sources inline with
"[CITE: field]". Those markers are for the model, not for the reader: the
verified citations are returned separately (Explanation.citations), so the
text a user sees must not contain them.

Two things used to leave broken markers in the text ("... Learning' [C"):

1. The `reasoning` field has a soft length limit (500 characters). The old
   validator kept the first 500 characters, so the cut could fall inside a
   marker.
2. The markers were kept in the stored text and only the web page removed
   the complete ones, so a cut marker stayed visible.

`shorten_at_boundary` replaces the blind cut (sentence end, else word
boundary, never inside an open bracket) and `strip_citation_markers`
removes every marker, complete or cut, before the text is stored.
"""

import re

# A complete marker: "[CITE: description]", "(CITE: title)", "[cite:categories]".
_MARKER = re.compile(r"\s*[\[(]\s*CITE\b[^\])]*[\])]", re.IGNORECASE)
# A marker cut at the end of the text: "[", "[C", "(CITE", "(CITE: descr".
_CUT_MARKER = re.compile(r"\s*[\[(]\s*(?:C(?:I(?:T(?:E)?)?)?)?\s*(?::[^\])]*)?$", re.IGNORECASE)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?])")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")

ELLIPSIS = "…"


def strip_citation_markers(text: str) -> str:
    """
    Remove the inline citation markers from an explanation.

    Complete markers anywhere and a cut marker at the end are removed; the
    spaces they leave before punctuation are closed up.

    Args:
        text: Explanation text as the model wrote it

    Returns:
        The text without markers, with single spaces
    """
    cleaned = _MARKER.sub("", text)
    cleaned = _CUT_MARKER.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    return " ".join(cleaned.split())


def shorten_at_boundary(text: str, limit: int) -> str:
    """
    Shorten a text to at most `limit` characters without cutting a word or a marker.

    Ends at the last sentence end within the limit when that keeps at least
    half of it; otherwise at the last word boundary, followed by "…". A
    bracket opened and not closed before the cut (a cut citation marker) is
    dropped with what follows it.

    Args:
        text: The text to shorten
        limit: Maximum length of the result, ellipsis included

    Returns:
        The text unchanged if it fits, else the shortened text
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    last_open = max(window.rfind("["), window.rfind("("))
    if last_open > max(window.rfind("]"), window.rfind(")")):
        window = window[:last_open]

    # Sentence ends are found in the whole text, so "3.5" is not one.
    sentence_ends = [m.end() for m in _SENTENCE_END.finditer(text) if m.end() <= len(window)]
    if sentence_ends and sentence_ends[-1] >= limit // 2:
        return text[: sentence_ends[-1]]

    head = window[: limit - len(ELLIPSIS)]
    space = head.rfind(" ")
    if space > 0 and len(head) < len(text) and not text[len(head)].isspace():
        head = head[:space]
    return head.rstrip(" ,;:-([") + ELLIPSIS
