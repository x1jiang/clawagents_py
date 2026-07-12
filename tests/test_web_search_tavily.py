"""Hermetic tests for Tavily-backed ``web_search``."""

from __future__ import annotations

import asyncio
import json

import pytest

from clawagents.tools import web
from clawagents.tools.web import WebSearchTool


def test_web_search_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = asyncio.run(WebSearchTool().execute({"query": "python asyncio"}))
    assert result.success is False
    assert "TAVILY_API_KEY" in (result.error or "")


def test_web_search_formats_tavily_payload(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    def fake_post(api_key: str, body: dict, timeout: int):
        assert api_key == "tvly-test"
        assert body["query"] == "clawagents web_search"
        assert body["max_results"] == 3
        assert body["search_depth"] == "basic"
        assert body["include_answer"] is True
        return 200, {
            "answer": "A short answer.",
            "results": [
                {
                    "title": "Docs",
                    "url": "https://example.com/docs",
                    "content": "Snippet about clawagents.",
                    "score": 0.91,
                },
                {
                    "title": "Blog",
                    "url": "https://example.com/blog",
                    "content": "More text.",
                    "score": 0.7,
                },
            ],
        }

    monkeypatch.setattr(web, "_tavily_post", fake_post)
    result = asyncio.run(
        WebSearchTool().execute(
            {"query": "clawagents web_search", "max_results": 3}
        )
    )
    assert result.success is True
    assert "A short answer." in result.output
    assert "https://example.com/docs" in result.output
    assert "[0.91] Docs" in result.output


def test_web_search_http_401(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "bad")
    monkeypatch.setattr(
        web, "_tavily_post", lambda *a, **k: (401, {"detail": {"error": "bad key"}})
    )
    result = asyncio.run(WebSearchTool().execute({"query": "x"}))
    assert result.success is False
    assert "401" in (result.error or "")


def test_web_search_clamps_max_results(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    seen: dict = {}

    def fake_post(api_key: str, body: dict, timeout: int):
        seen.update(body)
        return 200, {"results": []}

    monkeypatch.setattr(web, "_tavily_post", fake_post)
    asyncio.run(WebSearchTool().execute({"query": "q", "max_results": 99}))
    assert seen["max_results"] == web.MAX_SEARCH_RESULTS


def test_format_tavily_results_empty():
    text = web._format_tavily_results({"results": []}, "hello")
    assert "Query: hello" in text
    assert "(no results)" in text


def test_tavily_post_builds_json_request(monkeypatch: pytest.MonkeyPatch):
    """Exercise the HTTPS path with a fake connection (no network)."""

    class FakeResp:
        status = 200

        def read(self, n: int):
            return json.dumps(
                {"results": [{"title": "T", "url": "https://ex.com", "content": "c"}]}
            ).encode()

    class FakeConn:
        def __init__(self, *a, **k):
            self.requested = None

        def request(self, method, path, body=None, headers=None):
            self.requested = (method, path, body, headers)

        def getresponse(self):
            return FakeResp()

        def close(self):
            return None

    fake = FakeConn()
    monkeypatch.setattr(
        web.http.client, "HTTPSConnection", lambda *a, **k: fake
    )
    status, payload = web._tavily_post(
        "tvly-x", {"query": "q", "max_results": 1}, timeout=5
    )
    assert status == 200
    assert payload["results"][0]["title"] == "T"
    method, path, body, headers = fake.requested
    assert method == "POST"
    assert path == "/search"
    assert json.loads(body)["query"] == "q"
    assert headers["Authorization"] == "Bearer tvly-x"
