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
    assert 'id="search-examples"' in page and 'id="assistant-examples"' in page
    assert "SEARCH_EXAMPLES" in script and "ASSISTANT_EXAMPLES" in script
    import re

    languages = set(re.findall(r'lang: "([a-z]{2})"', script))
    assert {"es", "en", "fr", "de"} <= languages


def test_ui_shows_a_waiting_state_that_mentions_the_cold_start():
    """A slow first request shows a plain-language waiting state with a cancel button."""
    page, script = _static("index.html"), _static("app.js")
    assert 'id="search-wait"' in page and 'id="assistant-wait"' in page
    assert 'id="search-cancel"' in page and 'id="assistant-cancel"' in page
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


def test_ui_agent_steps_name_the_books_each_tool_returned():
    """The agent's trace (shown in the assistant's routing detail) lists the books of each tool call."""
    script = _static("app.js")
    for tool in ("search_books", "get_book_details", "find_similar_books"):
        assert tool in script
    assert "step.books" in script and "tool-books" in script


def test_ui_assistant_books_open_the_same_detail_view_as_search_results():
    page, script = _static("index.html"), _static("app.js")
    assert 'id="book-dialog"' in page
    # The assistant reuses the search result card (title opens the dialog) for its books
    assert script.count("openBook(") >= 2 and "renderResult(result, ranks, \"search\", searchMode)" in script


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


# ---------------------------------------------------------------------------
# V20: assistant mode (POST /assistant/chat), true stage positions, explanations
# ---------------------------------------------------------------------------


def test_ui_assistant_mode_replaces_the_agent_mode():
    page, script = _static("index.html"), _static("app.js")
    assert 'id="tab-assistant"' in page and 'id="panel-assistant"' in page
    assert 'id="tab-agent"' not in page and "Preguntar al agente" not in page
    assert '"/assistant/chat"' in script and '"/agent/ask"' not in script
    assert 'id="chat-thread"' in page and 'id="chat-input"' in page and 'id="chat-new"' in page


def test_ui_assistant_examples_cover_the_three_kinds_of_request_in_several_languages():
    import re

    script = _static("app.js")
    block = script[script.index("const ASSISTANT_EXAMPLES"):script.index("];", script.index("const ASSISTANT_EXAMPLES"))]
    assert len(re.findall(r"\{ text:", block)) == 3
    assert len(set(re.findall(r'lang: "([a-z]{2})"', block))) == 3


def test_ui_keeps_the_conversation_in_memory_and_echoes_each_route():
    """No browser storage; the whole conversation is sent and each assistant turn carries its route."""
    page, script = _static("index.html"), _static("app.js")
    for storage in ("localStorage", "sessionStorage", "indexedDB", "document.cookie"):
        assert storage not in script
    assert "JSON.stringify({ messages: conversation })" in script
    assert 'role: "assistant", content: response.message' in script and "route: response.route" in script
    assert "no se guarda en el navegador ni en el servidor" in " ".join(page.split())


def test_ui_follows_the_server_limits_of_the_conversation():
    """The page's limits are the API's (50 messages) and what the assistant reads of a message (500 characters)."""
    import re

    from app.api.v1.schemas import MAX_REQUEST_MESSAGES
    from app.domain.value_objects import MAX_CLARIFYING_TURNS, MAX_MESSAGE_CHARS

    page, script = _static("index.html"), _static("app.js")
    assert f"const MAX_CHAT_MESSAGES = {MAX_REQUEST_MESSAGES};" in script
    assert f"const MAX_CHAT_CHARS = {MAX_MESSAGE_CHARS};" in script
    assert f"const MAX_CLARIFYING_TURNS = {MAX_CLARIFYING_TURNS};" in script
    textarea = re.search(r'<textarea id="chat-input"[^>]*>', page).group(0)
    assert f'maxlength="{MAX_MESSAGE_CHARS}"' in textarea


def test_ui_shows_clarifying_options_as_buttons_that_send_the_answer():
    script = _static("app.js")
    assert 'response.route === "clarify"' in script and "renderOptions(response.options)" in script
    assert "sendChat(option)" in script


def test_ui_routing_detail_covers_every_node_and_forced_route():
    """Each reply shows the path through the graph with node times, forced routes, the query, filters and steps."""
    import re

    script = _static("app.js")
    graph = (Path(UI_DIR).parents[1] / "infrastructure" / "llm" / "graphs" / "assistant.py").read_text(encoding="utf-8")
    nodes = set(re.findall(r'add_node\("([a-z_]+)"', graph))
    forced = set(re.findall(r'^FORCED_[A-Z_]+ = "([a-z_]+)"', graph, re.M))
    assert nodes and forced
    for node in nodes:
        assert f"{node}:" in script.split("const NODE_NAMES")[1].split("};")[0]
    for reason in forced:
        assert f"{reason}:" in script.split("const FORCED_ROUTE_TEXT")[1].split("};")[0]
    assert 'el("details", "route-detail")' in script
    for field in ("timings_ms", "forced_route", "search_query", "filters", "response.steps", "usage"):
        assert field in script


def test_ui_handles_every_outcome_of_the_assistant_endpoint():
    """Degraded replies (no LLM, budget spent), notices of the graph, 400, 422, 429, 503 and cancellation."""
    script = _static("app.js")
    for code in ("llm_unavailable", "llm_budget_spent", "assistant_answer_failed", "agent_failed"):
        assert code in script.split("const ASSISTANT_NOTICE_TEXT")[1].split("};")[0]
    assert "ASSISTANT_ERRORS" in script
    for status in ("400:", "422:", "503:"):
        assert status in script.split("const ASSISTANT_ERRORS")[1].split("};")[0]
    assert "s === 429" in script and 'error.name === "AbortError"' in script
    assert "15 y 60 segundos" in script  # the slow agent route


def test_ui_announces_new_replies_without_moving_focus():
    page, script = _static("index.html"), _static("app.js")
    assert 'id="chat-live"' in page and 'aria-live="polite"' in page
    assert "announce(replySummary(response))" in script
    assert "keepFocus: true" in script and "aria-disabled" in script


def test_ui_uses_the_true_stage_positions_and_flags_approximations_only_without_them():
    script = _static("app.js")
    for field in ("lexical_rank", "vector_rank", "rerank_rank"):
        assert field in script
    assert script.count("Puestos aproximados") == 1  # only in the branch for a server without positions


def test_ui_no_longer_strips_citation_markers():
    """The server stores explanations without [CITE: ...] markers, so the page shows the text as it comes."""
    assert "CITE" not in _static("app.js")


def test_ui_explains_the_assistant_graph_with_an_inline_diagram():
    page = _static("index.html")
    assert 'id="graph-title"' in page and page.count('role="img"') >= 2
    for node in ("route", "clarify", "retrieve", "answer", "verify", "agent", "out_of_scope"):
        assert node in page.split('id="graph-title"')[1].split("</svg>")[0]
    for name in ("Enrutar", "Preguntar", "Buscar", "Responder", "Verificar", "Agente", "Fuera de ámbito"):
        assert f"<strong>{name}" in page


# ---------------------------------------------------------------------------
# Candidate mode (SEARCH_CANDIDATES): the page describes what the server runs
# ---------------------------------------------------------------------------


def _how_variant(page: str, mode: str) -> str:
    """The block of the "Cómo funciona" section that describes one candidate mode."""
    start = page.index(f'<div class="how-grid" data-candidates="{mode}"')
    return page[start:page.index("</ol>", start)]


def test_ui_describes_both_candidate_modes_and_picks_the_one_health_reports():
    page, script = _static("index.html"), _static("app.js")
    vector, hybrid = _how_variant(page, "vector"), _how_variant(page, "hybrid")
    # Vector mode: no fusion stage; keyword search is drawn and listed as the fallback only
    assert "RRF" not in vector and "fusion" not in vector.lower() and "Fusión" not in vector
    assert "solo si falla" in vector and "dg-fallback" in vector and "Reordenador" in vector
    # Hybrid mode keeps the published pipeline, hidden until /health asks for it
    assert "Fusión RRF" in hybrid and 'data-candidates="hybrid" hidden' in page
    # The mode comes from /health; a server that does not report it is an older, hybrid one
    assert "health.profile.candidates" in script and ': "hybrid";' in script
    assert "showPipeline(candidateMode)" in script
    # The arrowhead lives outside both diagrams, so it renders whichever one is shown
    assert page.count('<marker id="arrow"') == 1 and 'class="dg-defs"' in page


def test_ui_why_row_and_status_do_not_claim_a_keyword_search_in_vector_mode():
    page, script = _static("index.html"), _static("app.js")
    assert 'searchMode === "vector_only" ? "no se usó"' in script
    assert 'response.search_mode === "vector_only"' in script
    assert '"Puntuación de recuperación (final_score)"' in script
    assert "híbrid" not in page.split('id="how"')[0] and "Búsqueda híbrida" not in script
    # The assistant's results use the server's mode, not a fixed "hybrid"
    assert 'candidateMode === "hybrid" ? "hybrid" : "vector_only"' in script
