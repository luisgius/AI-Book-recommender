"""Tests for the static web UI served at /ui."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import UI_DIR, app

client = TestClient(app)


def test_ui_page_is_served():
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="search-form"' in response.text


def test_ui_assets_are_served():
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/styles.css").status_code == 200


def test_root_points_to_the_ui():
    assert client.get("/").json()["ui"] == "/ui/"


def test_ui_never_renders_api_data_as_html():
    """Book data comes from external APIs, so it must only reach the DOM as text."""
    script = (Path(UI_DIR) / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script


def test_ui_has_no_external_assets():
    """Scripts, styles and images are served by the app; only plain links (licences) point outside."""
    import re

    page = (Path(UI_DIR) / "index.html").read_text(encoding="utf-8")
    assets = re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", page)
    assert assets and not [tag for tag in assets if "http://" in tag or "https://" in tag]


def test_ui_attributes_sources_and_shows_budget_notices():
    """Each result names its source and licence (Wikipedia text is CC BY-SA), and budget notices are shown."""
    page = (Path(UI_DIR) / "index.html").read_text(encoding="utf-8")
    script = (Path(UI_DIR) / "app.js").read_text(encoding="utf-8")
    assert 'class="attribution"' in page and 'id="about"' in page and 'id="notice"' in page
    assert "renderAttribution(book)" in script and "attribution_url" in script
    assert "search_budget_spent" in script and "llm_budget_spent" in script


def test_ui_explains_a_server_without_llm():
    """Without an LLM the UI names the missing key (notice and disabled LLM options)."""
    script = (Path(UI_DIR) / "app.js").read_text(encoding="utf-8")
    assert "llm_unavailable" in script and "OPENROUTER_API_KEY" in script


# ---------------------------------------------------------------------------
# V19: examples, waiting states, why-row, agent panel, how it works, quality
# ---------------------------------------------------------------------------

def _static(name: str) -> str:
    return (Path(UI_DIR) / name).read_text(encoding="utf-8")


def test_ui_offers_example_queries_in_several_languages():
    """Clickable examples let an examiner try the system at once, in more than one language."""
    page, script = _static("index.html"), _static("app.js")
    assert 'id="search-examples"' in page and 'id="agent-examples"' in page
    assert "SEARCH_EXAMPLES" in script and "AGENT_EXAMPLES" in script
    import re

    languages = set(re.findall(r'lang: "([a-z]{2})"', script))
    assert {"es", "en", "fr", "de"} <= languages


def test_ui_shows_a_waiting_state_that_mentions_the_cold_start():
    """A slow first request shows a plain-language waiting state with a cancel button."""
    page, script = _static("index.html"), _static("app.js")
    assert 'id="search-wait"' in page and 'id="agent-wait"' in page
    assert 'id="search-cancel"' in page and 'id="agent-cancel"' in page
    assert "AbortController" in script and "COLD_START_HINT" in script and "10 y 20 segundos" in script


def test_ui_explains_errors_and_degraded_searches_in_spanish():
    """429 (rate limit and daily budget), 503 (no LLM) and degraded searches have their own messages."""
    script = _static("app.js")
    assert "describeError" in script
    for case in ("s === 429", "s === 503", "s === 0", "response.degraded", "lexical_only"):
        assert case in script
    assert "Retry-After" in script


def test_ui_shows_why_each_result_was_returned():
    """Each result shows the stages that found it (BM25, embeddings, reranker, final) and the raw scores on demand."""
    page, script = _static("index.html"), _static("app.js")
    assert 'class="trail"' in page and 'class="raw"' in page
    for field in ("lexical_score", "vector_score", "rerank_score", "final_score"):
        assert field in script
    assert "stageRanks" in script and "Reordenador" in script


def test_ui_has_an_agent_mode_with_answer_books_and_steps():
    """The agent endpoint is reachable from the UI and its tool-call trace is shown."""
    page, script = _static("index.html"), _static("app.js")
    assert 'id="tab-agent"' in page and "Preguntar al agente" in page
    assert 'id="agent-answer"' in page and 'id="agent-books"' in page and 'id="agent-steps"' in page
    assert '"/agent/ask"' in script
    for tool in ("search_books", "get_book_details", "find_similar_books"):
        assert tool in script
    assert "dropped_recommendations" in script and "reflections" in script


def test_ui_agent_books_open_the_same_detail_view_as_search_results():
    page, script = _static("index.html"), _static("app.js")
    assert 'id="book-dialog"' in page
    assert script.count("openBook(") >= 3  # definition, search results, agent recommendations


def test_ui_explains_the_pipeline_with_an_inline_diagram():
    page = _static("index.html")
    assert 'id="how"' in page and "<svg" in page and 'role="img"' in page
    for stage in ("BM25", "embeddings", "Fusión RRF", "Reordenador", "LLM"):
        assert stage in page


def test_ui_respects_colour_scheme_motion_and_keyboard_users():
    page, style = _static("index.html"), _static("styles.css")
    assert "prefers-color-scheme: dark" in style and "prefers-reduced-motion" in style
    assert ":focus-visible" in style and 'class="skip-link"' in page
    assert 'role="tablist"' in page and 'aria-selected' in page


def test_ui_every_form_control_has_a_label():
    """Every input, select and textarea is named by a <label for=...> or is wrapped in a label."""
    import re

    page = _static("index.html")
    label_for = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', page))
    for match in re.finditer(r"<(input|select|textarea)\b[^>]*>", page):
        tag = match.group(0)
        control_id = re.search(r'\bid="([^"]+)"', tag)
        preceding = page[: match.start()]
        wrapped = preceding.rfind("<label") > preceding.rfind("</label>")
        assert wrapped or (control_id and control_id.group(1) in label_for), tag


def test_ui_static_files_load_nothing_from_other_hosts():
    """No web fonts, CDN scripts or remote fetches: the page works offline and sends data nowhere else."""
    import re

    style, script = _static("styles.css"), _static("app.js")
    assert "@import" not in style and not re.search(r"url\(\s*['\"]?https?:", style)
    assert not re.search(r"fetch\(\s*['\"`]https?:", script)
    assert 'const API = "/api/v1"' in script
