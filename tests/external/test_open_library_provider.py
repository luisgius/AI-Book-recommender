"""Tests for OpenLibraryProvider with a fake HTTP session (no network)."""

from unittest.mock import patch

import pytest
import requests

from app.infrastructure.external.open_library_provider import OpenLibraryProvider


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._payload


class FakeSession:
    """Routes GET requests by path; records every call."""

    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def get(self, url, params=None, timeout=None):
        path = url.replace(OpenLibraryProvider.BASE_URL, "")
        self.calls.append((path, params))
        handler = self.routes.get(path)
        if handler is None:
            return FakeResponse({}, status=404)
        return handler(params) if callable(handler) else handler


SEARCH = {"docs": [
    {"key": "/works/OL1W", "title": "Guerra civil", "author_name": ["Ignacio Agustí"], "first_publish_year": 1972,
     "language": ["spa"], "subject": ["Fiction", "Spain Civil War, 1936-1939", "a", "b", "c", "d"],
     "isbn": ["8408000000", "9788408000000"], "publisher": ["Planeta"], "number_of_pages_median": 574,
     "cover_i": 42},
    {"key": "/works/OL2W", "title": "No authors here"},
    {"key": "/works/OL3W", "title": "Bilingual", "author_name": ["X"], "language": ["spa", "eng"],
     "first_sentence": ["It begins."]},
]}


def provider(routes, **kwargs):
    return OpenLibraryProvider(session=FakeSession(routes), request_delay_s=0, **kwargs)


def test_maps_search_results_to_books():
    p = provider({"/search.json": FakeResponse(SEARCH),
                  "/works/OL1W.json": FakeResponse({"description": {"type": "/type/text", "value": "Una novela."}}),
                  "/works/OL3W.json": FakeResponse({})})

    books = p.search_books("guerra civil", max_results=10)

    assert [b.title for b in books] == ["Guerra civil", "Bilingual"]  # work without authors skipped
    first = books[0]
    assert first.source == "open_library" and first.source_id == "OL1W"
    assert first.language == "es"
    assert first.description == "Una novela."
    assert first.get_published_year() == 1972
    assert len(first.categories) == 5
    assert first.metadata.isbn13 == "9788408000000" and first.metadata.isbn == "8408000000"
    assert first.metadata.page_count == 574
    assert first.metadata.thumbnail_url.endswith("/b/id/42-M.jpg")


def test_falls_back_to_first_sentence_and_leaves_ambiguous_language_unset():
    p = provider({"/search.json": FakeResponse(SEARCH), "/works/OL1W.json": FakeResponse({}),
                  "/works/OL3W.json": FakeResponse({})})

    bilingual = p.search_books("x")[1]

    assert bilingual.description == "It begins."
    assert bilingual.language is None


def test_language_filter_uses_marc_codes_and_sets_the_book_language():
    session = FakeSession({"/search.json": FakeResponse({"docs": SEARCH["docs"][:1]}),
                           "/works/OL1W.json": FakeResponse({})})
    p = OpenLibraryProvider(session=session, request_delay_s=0)

    books = p.search_books("guerra civil", language="ES")

    assert session.calls[0][1]["q"] == "guerra civil language:spa"
    assert books[0].language == "es"


def test_descriptions_can_be_skipped_to_save_requests():
    session = FakeSession({"/search.json": FakeResponse(SEARCH)})
    p = OpenLibraryProvider(session=session, fetch_descriptions=False, request_delay_s=0)

    p.search_books("guerra civil")

    assert [path for path, _ in session.calls] == ["/search.json"]


def test_respects_max_results():
    p = provider({"/search.json": FakeResponse(SEARCH)}, fetch_descriptions=False)
    assert len(p.search_books("x", max_results=1)) == 1


@pytest.mark.parametrize("kwargs", [{"query": "  "}, {"query": "x", "language": "xx"}, {"query": "x", "max_results": 0}])
def test_invalid_input_is_rejected(kwargs):
    with pytest.raises(ValueError):
        provider({}).search_books(**kwargs)


def test_transient_errors_are_retried_then_succeed():
    responses = [FakeResponse({}, status=503), FakeResponse({"docs": []})]
    p = provider({"/search.json": lambda params: responses.pop(0)})
    with patch("app.infrastructure.external.open_library_provider.time.sleep"):
        assert p.search_books("x") == []


def test_persistent_failure_raises_runtime_error():
    p = provider({"/search.json": FakeResponse({}, status=500)})
    with patch("app.infrastructure.external.open_library_provider.time.sleep"):
        with pytest.raises(RuntimeError):
            p.search_books("x")


def test_get_book_by_id_resolves_author_names():
    p = provider({
        "/works/OL1W.json": FakeResponse({"title": "Guerra civil", "description": "Texto.",
                                         "subjects": ["Fiction"],
                                         "authors": [{"author": {"key": "/authors/OL9A"}}]}),
        "/authors/OL9A.json": FakeResponse({"name": "Ignacio Agustí"}),
    })

    book = p.get_book_by_id("OL1W")

    assert book.title == "Guerra civil" and book.authors == ["Ignacio Agustí"]
    assert book.description == "Texto." and book.source_id == "OL1W"


def test_get_book_by_id_unknown_work_is_none():
    assert provider({}).get_book_by_id("OL404W") is None
