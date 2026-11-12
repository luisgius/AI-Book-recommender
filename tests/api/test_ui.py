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
