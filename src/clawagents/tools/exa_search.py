"""Exa Search Tool — neural and keyword web search via the Exa API.

Adds a ``web_search`` tool that complements the existing ``web_fetch``:
``web_fetch`` reads a single known URL, ``web_search`` discovers URLs and
returns ranked results with optional inline content (highlights, summaries,
or full text). Useful for documentation lookups, news, research papers,
company/people profiles, and any task that needs an LLM to find sources
before reading them.

Configuration
-------------
Set ``EXA_API_KEY`` in the environment. The tool is registered eagerly but
will return a structured error if the key is missing, so callers can
detect the disabled state without the SDK installed.

The Exa Python SDK (``exa-py``) is imported lazily inside ``execute`` so the
runtime dependency is only required when the tool actually runs.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clawagents.tools.registry import Tool, ToolResult

DEFAULT_NUM_RESULTS = 5
MAX_NUM_RESULTS = 25
DEFAULT_SEARCH_TYPE = "auto"
_VALID_SEARCH_TYPES = frozenset({
    "auto", "neural", "fast", "deep-lite", "deep", "deep-reasoning", "instant",
})
_VALID_CATEGORIES = frozenset({
    "company", "research paper", "news", "personal site",
    "financial report", "people",
})
_INTEGRATION_HEADER = "clawagents-py"


@dataclass(frozen=True)
class ExaSearchResult:
    """One result row returned from the Exa /search endpoint.

    Fields mirror the API response. ``snippet`` is a derived view that
    cascades through ``highlights``, ``summary``, and ``text`` so callers
    can render a single excerpt without re-implementing the fallback.
    """

    title: str
    url: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    text: Optional[str] = None
    summary: Optional[str] = None
    highlights: List[str] = field(default_factory=list)

    @property
    def snippet(self) -> str:
        if self.highlights:
            return " … ".join(h.strip() for h in self.highlights if h)
        if self.summary:
            return self.summary.strip()
        if self.text:
            text = self.text.strip()
            return text[:500] + ("…" if len(text) > 500 else "")
        return ""


def _coerce_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    elif isinstance(value, list):
        items = [str(v).strip() for v in value]
    else:
        return None
    items = [v for v in items if v]
    return items or None


def _parse_results(payload: Any) -> List[ExaSearchResult]:
    """Parse an Exa SDK response (or raw dict) into ``ExaSearchResult``.

    The SDK returns objects with attributes; the raw HTTP response is a
    dict under ``results``. Accept both so the parsing is reusable in
    tests where a JSON fixture is more convenient than a fake SDK object.
    """
    raw_results = getattr(payload, "results", None)
    if raw_results is None and isinstance(payload, dict):
        raw_results = payload.get("results", [])
    if raw_results is None:
        return []

    parsed: List[ExaSearchResult] = []
    for item in raw_results:
        if isinstance(item, dict):
            title = item.get("title") or ""
            url = item.get("url") or ""
            published = item.get("publishedDate") or item.get("published_date")
            author = item.get("author")
            text = item.get("text")
            summary = item.get("summary")
            highlights = item.get("highlights") or []
        else:
            title = getattr(item, "title", "") or ""
            url = getattr(item, "url", "") or ""
            published = getattr(item, "published_date", None) or getattr(item, "publishedDate", None)
            author = getattr(item, "author", None)
            text = getattr(item, "text", None)
            summary = getattr(item, "summary", None)
            highlights = getattr(item, "highlights", None) or []
        parsed.append(
            ExaSearchResult(
                title=str(title),
                url=str(url),
                published_date=str(published) if published else None,
                author=str(author) if author else None,
                text=str(text) if text else None,
                summary=str(summary) if summary else None,
                highlights=[str(h) for h in highlights if h],
            )
        )
    return parsed


def _format_results(query: str, results: List[ExaSearchResult]) -> str:
    if not results:
        return f"No results for query: {query!r}"
    lines: List[str] = [f"Exa search results for {query!r} ({len(results)} hits):", ""]
    for i, r in enumerate(results, start=1):
        header = f"{i}. {r.title or '(untitled)'}"
        meta_bits: List[str] = [r.url]
        if r.published_date:
            meta_bits.append(f"published {r.published_date}")
        if r.author:
            meta_bits.append(f"by {r.author}")
        lines.append(header)
        lines.append("   " + " · ".join(meta_bits))
        snippet = r.snippet
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_contents_kwarg(
    *,
    text: bool,
    text_max_chars: Optional[int],
    highlights: bool,
    summary: bool,
    summary_query: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Assemble the ``contents`` dict for the SDK call.

    Returns ``None`` when no content is requested so the SDK can use its
    default (no body fetched). The Exa API accepts multiple content types
    in the same request — they are not mutually exclusive.
    """
    contents: Dict[str, Any] = {}
    if text:
        contents["text"] = {"maxCharacters": text_max_chars} if text_max_chars else True
    if highlights:
        contents["highlights"] = True
    if summary:
        contents["summary"] = {"query": summary_query} if summary_query else {}
    return contents or None


class ExaSearchTool:
    """Search the web through the Exa API.

    Honours the same async/registry contract as ``WebFetchTool``. Marked
    ``cacheable = True`` and ``parallel_safe = True`` because a search
    request is a stateless read; the registry's path-scope rules will
    serialise duplicate queries automatically via the cache layer.
    """

    name = "web_search"
    cacheable = True
    parallel_safe = True
    keywords = [
        "search the web", "google", "find documentation",
        "research", "news", "papers", "exa",
    ]
    description = (
        "Search the web with Exa and return ranked results with optional "
        "inline content. Use this to discover URLs, find documentation or "
        "papers, or pull recent news; use web_fetch to read a specific URL "
        "you already know. Supports neural/auto/fast/keyword search types, "
        "domain filtering, date ranges, and category-scoped queries "
        "(company, research paper, news, personal site, financial report, "
        "people). Requires EXA_API_KEY in the environment."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "query": {
            "type": "string",
            "description": "Search query. Plain natural-language works best with neural/auto.",
            "required": True,
        },
        "num_results": {
            "type": "number",
            "description": f"Number of results to return (1-{MAX_NUM_RESULTS}). Default {DEFAULT_NUM_RESULTS}.",
        },
        "type": {
            "type": "string",
            "description": (
                "Search algorithm: auto (default), neural, fast, instant, "
                "deep-lite, deep, deep-reasoning."
            ),
        },
        "category": {
            "type": "string",
            "description": (
                "Restrict to one category: company, research paper, news, "
                "personal site, financial report, people."
            ),
        },
        "include_domains": {
            "type": "string",
            "description": "Comma-separated allowlist of domains (e.g. 'arxiv.org,nature.com').",
        },
        "exclude_domains": {
            "type": "string",
            "description": "Comma-separated blocklist of domains.",
        },
        "include_text": {
            "type": "string",
            "description": "Comma-separated phrases that must appear verbatim in result text.",
        },
        "exclude_text": {
            "type": "string",
            "description": "Comma-separated phrases that must NOT appear in result text.",
        },
        "start_published_date": {
            "type": "string",
            "description": "ISO 8601 lower bound on publication date (e.g. '2025-01-01').",
        },
        "end_published_date": {
            "type": "string",
            "description": "ISO 8601 upper bound on publication date.",
        },
        "user_location": {
            "type": "string",
            "description": "Two-letter ISO country code to bias results (e.g. 'US').",
        },
        "highlights": {
            "type": "boolean",
            "description": "Return LLM-selected highlight excerpts. Default true.",
        },
        "summary": {
            "type": "boolean",
            "description": "Return a short LLM-generated summary per result. Default false.",
        },
        "summary_query": {
            "type": "string",
            "description": "Optional guidance for the summary (only used when summary=true).",
        },
        "text": {
            "type": "boolean",
            "description": "Include full page text per result. Default false (cheaper, smaller).",
        },
        "text_max_chars": {
            "type": "number",
            "description": "Cap on full-text length per result when text=true.",
        },
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(success=False, output="", error="No query provided")

        api_key = os.environ.get("EXA_API_KEY", "").strip()
        if not api_key:
            return ToolResult(
                success=False, output="",
                error=(
                    "EXA_API_KEY is not set. Add it to your environment "
                    "(or .env) to enable web_search."
                ),
            )

        try:
            num_results = int(args.get("num_results", DEFAULT_NUM_RESULTS))
        except (TypeError, ValueError):
            num_results = DEFAULT_NUM_RESULTS
        num_results = max(1, min(MAX_NUM_RESULTS, num_results))

        search_type = str(args.get("type") or DEFAULT_SEARCH_TYPE).strip().lower()
        if search_type not in _VALID_SEARCH_TYPES:
            return ToolResult(
                success=False, output="",
                error=(
                    f"Unknown search type {search_type!r}. "
                    f"Expected one of: {sorted(_VALID_SEARCH_TYPES)}"
                ),
            )

        category = args.get("category")
        if category is not None:
            category = str(category).strip().lower() or None
            if category and category not in _VALID_CATEGORIES:
                return ToolResult(
                    success=False, output="",
                    error=(
                        f"Unknown category {category!r}. "
                        f"Expected one of: {sorted(_VALID_CATEGORIES)}"
                    ),
                )

        kwargs: Dict[str, Any] = {
            "num_results": num_results,
            "type": search_type,
        }
        if category:
            kwargs["category"] = category

        for src, dst in (
            ("include_domains", "include_domains"),
            ("exclude_domains", "exclude_domains"),
            ("include_text", "include_text"),
            ("exclude_text", "exclude_text"),
        ):
            coerced = _coerce_str_list(args.get(src))
            if coerced:
                kwargs[dst] = coerced

        if start := args.get("start_published_date"):
            kwargs["start_published_date"] = str(start)
        if end := args.get("end_published_date"):
            kwargs["end_published_date"] = str(end)
        if loc := args.get("user_location"):
            kwargs["user_location"] = str(loc)

        highlights_flag = bool(args.get("highlights", True))
        summary_flag = bool(args.get("summary", False))
        text_flag = bool(args.get("text", False))
        text_max_chars = args.get("text_max_chars")
        try:
            text_max_chars = int(text_max_chars) if text_max_chars is not None else None
        except (TypeError, ValueError):
            text_max_chars = None
        contents = _build_contents_kwarg(
            text=text_flag,
            text_max_chars=text_max_chars,
            highlights=highlights_flag,
            summary=summary_flag,
            summary_query=args.get("summary_query") and str(args["summary_query"]),
        )
        if contents is not None:
            kwargs["contents"] = contents

        try:
            from exa_py import Exa
        except ImportError:
            return ToolResult(
                success=False, output="",
                error=(
                    "exa-py is not installed. Run `pip install 'clawagents[exa]'` "
                    "or `pip install exa-py` to enable web_search."
                ),
            )

        try:
            client = Exa(api_key)
            try:
                client.headers["x-exa-integration"] = _INTEGRATION_HEADER
            except (AttributeError, TypeError):
                pass

            response = await asyncio.to_thread(
                client.search_and_contents if contents is not None else client.search,
                query,
                **kwargs,
            )
        except Exception as err:  # noqa: BLE001 — surface a one-line tool error
            return ToolResult(
                success=False, output="",
                error=f"Exa search failed: {err}",
            )

        results = _parse_results(response)
        return ToolResult(success=True, output=_format_results(query, results))


exa_search_tools: List[Tool] = [ExaSearchTool()]
