"""Web Fetch Tool — retrieve content from a URL.

Useful for reading documentation, API responses, or any web resource.
Returns plain text with HTML tags stripped for readability.
"""

import re
import asyncio
from typing import Any, Dict, List
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from clawagents.tools.registry import Tool, ToolResult

MAX_RESPONSE_CHARS = 50_000
DEFAULT_TIMEOUT_S = 15


def _strip_html(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<nav[\s\S]*?</nav>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<footer[\s\S]*?</footer>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&")
    html = html.replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&quot;", '"').replace("&#39;", "'")
    html = re.sub(r"\s{2,}", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


class WebFetchTool:
    name = "web_fetch"
    description = (
        "Fetch content from a URL. Returns the text content of the page. "
        "Useful for reading documentation, API responses, or checking web resources. "
        "HTML is stripped for readability. JSON responses are returned as-is."
    )
    parameters = {
        "url": {"type": "string", "description": "The URL to fetch", "required": True},
        "timeout": {"type": "number", "description": f"Timeout in seconds. Default: {DEFAULT_TIMEOUT_S}"},
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        url = str(args.get("url", ""))
        try:
            timeout = max(1, int(args.get("timeout", DEFAULT_TIMEOUT_S)))
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT_S

        if not url:
            return ToolResult(success=False, output="", error="No URL provided")

        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError
        except Exception:
            return ToolResult(success=False, output="", error=f"Invalid URL: {url}")

        loop = asyncio.get_running_loop()
        try:
            def _fetch():
                req = Request(url, headers={"User-Agent": "ClawAgents/1.0"})
                resp = urlopen(req, timeout=timeout)
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8", errors="replace")
                return status, content_type, body

            status, content_type, body = await loop.run_in_executor(None, _fetch)

            if len(body) > MAX_RESPONSE_CHARS:
                body = body[:MAX_RESPONSE_CHARS] + f"\n...(truncated at {MAX_RESPONSE_CHARS} chars)"

            if "html" in content_type.lower():
                body = _strip_html(body)

            return ToolResult(success=True, output=f"[{status}] {url}\n\n{body}")

        except HTTPError as e:
            return ToolResult(success=False, output="", error=f"HTTP {e.code}: {e.reason}")
        except URLError as e:
            return ToolResult(success=False, output="", error=f"web_fetch failed: {e.reason}")
        except TimeoutError:
            return ToolResult(success=False, output="", error=f"Request timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"web_fetch failed: {str(e)}")


web_tools: List[Tool] = [WebFetchTool()]
