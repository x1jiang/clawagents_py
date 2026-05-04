"""Tests for the Exa-powered ``web_search`` tool.

The Exa SDK is imported lazily inside ``ExaSearchTool.execute`` so we mock at
the source module path (``exa_py.Exa``) rather than the consuming module.
Tests do not require the ``exa-py`` package to be installed: missing-dep
behaviour is exercised via the unset-key path, and the success path injects
a fake module into ``sys.modules`` before the import line runs.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from clawagents.tools.exa_search import (
    ExaSearchResult,
    ExaSearchTool,
    _build_contents_kwarg,
    _coerce_str_list,
    _format_results,
    _parse_results,
    exa_search_tools,
)


# ─── Pure helpers ─────────────────────────────────────────────────────────


def test_tool_is_registered():
    names = [t.name for t in exa_search_tools]
    assert names == ["web_search"]


def test_coerce_str_list_handles_csv_and_lists():
    assert _coerce_str_list(None) is None
    assert _coerce_str_list("") is None
    assert _coerce_str_list("a, b ,c") == ["a", "b", "c"]
    assert _coerce_str_list(["x", " y ", ""]) == ["x", "y"]
    assert _coerce_str_list(123) is None


def test_build_contents_kwarg_supports_combined_modes():
    contents = _build_contents_kwarg(
        text=True, text_max_chars=400,
        highlights=True, summary=True, summary_query="why",
    )
    assert contents == {
        "text": {"maxCharacters": 400},
        "highlights": True,
        "summary": {"query": "why"},
    }


def test_build_contents_kwarg_returns_none_when_empty():
    assert _build_contents_kwarg(
        text=False, text_max_chars=None,
        highlights=False, summary=False, summary_query=None,
    ) is None


def test_build_contents_kwarg_text_boolean_when_no_cap():
    contents = _build_contents_kwarg(
        text=True, text_max_chars=None,
        highlights=False, summary=False, summary_query=None,
    )
    assert contents == {"text": True}


# ─── Response parsing + snippet fallback ──────────────────────────────────


_FIXTURE_RAW: Dict[str, Any] = {
    "results": [
        {
            "title": "Hit with highlights",
            "url": "https://example.com/a",
            "publishedDate": "2026-01-15",
            "author": "A. Person",
            "highlights": ["first excerpt", "second excerpt"],
        },
        {
            "title": "Hit with only summary",
            "url": "https://example.com/b",
            "summary": "Short summary text.",
        },
        {
            "title": "Hit with only text",
            "url": "https://example.com/c",
            "text": "long body " * 200,
        },
        {
            "title": "Hit with nothing",
            "url": "https://example.com/d",
        },
    ]
}


def test_parse_results_from_raw_dict():
    results = _parse_results(_FIXTURE_RAW)
    assert len(results) == 4
    assert results[0].title == "Hit with highlights"
    assert results[0].url == "https://example.com/a"
    assert results[0].published_date == "2026-01-15"
    assert results[0].author == "A. Person"
    assert results[0].highlights == ["first excerpt", "second excerpt"]


def test_parse_results_from_sdk_objects():
    """The SDK returns objects with attributes; the parser must accept both."""

    class _R:
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    payload = types.SimpleNamespace(
        results=[
            _R(
                title="t1",
                url="https://x.test/1",
                published_date="2026-02-02",
                author="auth",
                text=None,
                summary=None,
                highlights=["h"],
            )
        ]
    )
    results = _parse_results(payload)
    assert results[0].url == "https://x.test/1"
    assert results[0].snippet == "h"


def test_snippet_falls_back_through_highlights_summary_text():
    rs = _parse_results(_FIXTURE_RAW)
    # Highlights win when present.
    assert rs[0].snippet == "first excerpt … second excerpt"
    # Summary used when highlights missing.
    assert rs[1].snippet == "Short summary text."
    # Text is truncated to 500 chars + ellipsis when used as fallback.
    snippet = rs[2].snippet
    assert snippet.endswith("…")
    assert len(snippet) <= 501
    # Empty when nothing available.
    assert rs[3].snippet == ""


def test_format_results_includes_query_and_urls():
    rs = _parse_results(_FIXTURE_RAW)
    out = _format_results("hello world", rs)
    assert "hello world" in out
    assert "https://example.com/a" in out
    assert "Hit with highlights" in out
    assert "first excerpt" in out


def test_format_results_handles_empty_list():
    out = _format_results("nada", [])
    assert "No results" in out
    assert "nada" in out


# ─── execute() — disabled / validation paths ──────────────────────────────


@pytest.mark.asyncio
async def test_execute_rejects_empty_query(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")
    res = await ExaSearchTool().execute({"query": "  "})
    assert res.success is False
    assert "No query" in (res.error or "")


@pytest.mark.asyncio
async def test_execute_disabled_when_api_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    res = await ExaSearchTool().execute({"query": "anything"})
    assert res.success is False
    assert "EXA_API_KEY" in (res.error or "")


@pytest.mark.asyncio
async def test_execute_rejects_unknown_search_type(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")
    res = await ExaSearchTool().execute({"query": "q", "type": "telepathic"})
    assert res.success is False
    assert "Unknown search type" in (res.error or "")


@pytest.mark.asyncio
async def test_execute_rejects_unknown_category(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")
    res = await ExaSearchTool().execute({"query": "q", "category": "vibes"})
    assert res.success is False
    assert "Unknown category" in (res.error or "")


# ─── execute() — happy path with a fake exa_py module ─────────────────────


class _FakeExaClient:
    """Stand-in for ``exa_py.Exa``: records the kwargs it was called with."""

    last_call: Optional[Dict[str, Any]] = None
    last_query: Optional[str] = None
    last_method: Optional[str] = None
    headers: Dict[str, str]

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {}

    def search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        type(self).last_query = query
        type(self).last_call = kwargs
        type(self).last_method = "search"
        return _FIXTURE_RAW

    def search_and_contents(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        type(self).last_query = query
        type(self).last_call = kwargs
        type(self).last_method = "search_and_contents"
        return _FIXTURE_RAW


@pytest.fixture
def fake_exa_module(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake ``exa_py`` module so the lazy import inside execute()
    resolves to ``_FakeExaClient`` without requiring the real SDK.
    """
    _FakeExaClient.last_call = None
    _FakeExaClient.last_query = None
    _FakeExaClient.last_method = None

    fake_module = types.ModuleType("exa_py")
    fake_module.Exa = _FakeExaClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exa_py", fake_module)
    return _FakeExaClient


@pytest.mark.asyncio
async def test_execute_happy_path_sets_integration_header(
    monkeypatch: pytest.MonkeyPatch, fake_exa_module: type[_FakeExaClient]
):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")

    captured_headers: Dict[str, str] = {}
    original_init = fake_exa_module.__init__

    def init_capture(self: Any, api_key: str) -> None:  # noqa: ANN401
        original_init(self, api_key)
        captured_headers["holder"] = self.headers  # type: ignore[assignment]

    monkeypatch.setattr(fake_exa_module, "__init__", init_capture)

    res = await ExaSearchTool().execute({"query": "vector databases"})
    assert res.success is True
    assert "vector databases" in res.output
    # Tracking header is required for usage attribution.
    assert captured_headers["holder"]["x-exa-integration"] == "clawagents-py"


@pytest.mark.asyncio
async def test_execute_passes_filters_through_to_sdk(
    monkeypatch: pytest.MonkeyPatch, fake_exa_module: type[_FakeExaClient]
):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")
    res = await ExaSearchTool().execute(
        {
            "query": "rust async runtime",
            "num_results": 3,
            "type": "neural",
            "category": "research paper",
            "include_domains": "arxiv.org, openreview.net",
            "exclude_domains": "medium.com",
            "start_published_date": "2025-01-01",
            "end_published_date": "2026-05-01",
            "user_location": "US",
            "highlights": True,
            "summary": True,
            "summary_query": "what's new",
            "text": False,
        }
    )
    assert res.success is True
    call = fake_exa_module.last_call or {}
    assert fake_exa_module.last_query == "rust async runtime"
    assert call["num_results"] == 3
    assert call["type"] == "neural"
    assert call["category"] == "research paper"
    assert call["include_domains"] == ["arxiv.org", "openreview.net"]
    assert call["exclude_domains"] == ["medium.com"]
    assert call["start_published_date"] == "2025-01-01"
    assert call["end_published_date"] == "2026-05-01"
    assert call["user_location"] == "US"
    # Both highlights and summary requested simultaneously.
    assert call["contents"]["highlights"] is True
    assert call["contents"]["summary"] == {"query": "what's new"}
    # search_and_contents is the right entry point when contents are requested.
    assert fake_exa_module.last_method == "search_and_contents"


@pytest.mark.asyncio
async def test_execute_uses_plain_search_when_no_contents_requested(
    monkeypatch: pytest.MonkeyPatch, fake_exa_module: type[_FakeExaClient]
):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")
    res = await ExaSearchTool().execute(
        {"query": "minimal", "highlights": False, "summary": False, "text": False}
    )
    assert res.success is True
    assert fake_exa_module.last_method == "search"
    assert "contents" not in (fake_exa_module.last_call or {})


@pytest.mark.asyncio
async def test_execute_caps_num_results(
    monkeypatch: pytest.MonkeyPatch, fake_exa_module: type[_FakeExaClient]
):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")
    await ExaSearchTool().execute({"query": "q", "num_results": 9999})
    assert (fake_exa_module.last_call or {})["num_results"] == 25
    await ExaSearchTool().execute({"query": "q", "num_results": -3})
    assert (fake_exa_module.last_call or {})["num_results"] == 1


@pytest.mark.asyncio
async def test_execute_surfaces_sdk_errors(
    monkeypatch: pytest.MonkeyPatch, fake_exa_module: type[_FakeExaClient]
):
    monkeypatch.setenv("EXA_API_KEY", "sk-test")

    def boom(self: Any, query: str, **kwargs: Any) -> Any:  # noqa: ANN401
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(fake_exa_module, "search", boom)
    monkeypatch.setattr(fake_exa_module, "search_and_contents", boom)
    res = await ExaSearchTool().execute({"query": "q"})
    assert res.success is False
    assert "Exa search failed" in (res.error or "")
    assert "upstream 503" in (res.error or "")


# ─── ExaSearchResult dataclass ────────────────────────────────────────────


def test_exa_search_result_is_immutable():
    r = ExaSearchResult(title="t", url="https://x.y")
    with pytest.raises((AttributeError, Exception)):
        r.title = "nope"  # type: ignore[misc]
