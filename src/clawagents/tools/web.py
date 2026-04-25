"""Web Fetch Tool — retrieve content from a URL.

Useful for reading documentation, API responses, or any web resource.
Returns plain text with HTML tags stripped for readability.

Security
--------
``web_fetch`` is callable by the LLM with arbitrary URLs, so it can be
weaponized for SSRF (e.g. asking the agent to read cloud metadata at
``http://169.254.169.254/`` or internal services on ``localhost``). To
prevent that we:

* restrict to ``http`` / ``https``,
* resolve the hostname and reject loopback, link-local, private (RFC1918),
  unspecified, multicast, and reserved addresses unless explicitly opted
  in via ``CLAWAGENTS_WEB_ALLOW_PRIVATE=1``,
* **disable automatic redirects** and revalidate every hop. A naive
  validator that only checks the original URL is bypassable: a public
  attacker-controlled host can return ``302 Location: http://127.0.0.1/``
  or ``http://169.254.169.254/...`` and a default ``urlopen`` will follow
  it without re-checking.

If you genuinely need to hit private endpoints (dev environments,
internal docs servers), set the env var or run a custom tool that
bypasses ``web_fetch``.
"""

import os
import re
import asyncio
import ipaddress
import socket
from typing import Any, Dict, List
from urllib.request import urlopen, Request, build_opener, HTTPRedirectHandler
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse, urljoin

from clawagents.tools.registry import Tool, ToolResult

MAX_RESPONSE_CHARS = 50_000
DEFAULT_TIMEOUT_S = 15
MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = ("http", "https")


def _is_private_address(host: str) -> bool:
    """Return True if *host* resolves to a non-public IP we should refuse.

    Covers loopback, link-local, private RFC1918, unspecified, multicast,
    and reserved ranges. Also blocks the EC2/IMDS metadata IP explicitly
    so it's caught even if a future stdlib release relaxes a category.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return True
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_unspecified
            or ip.is_multicast
            or ip.is_reserved
        ):
            return True
        if str(ip) in {"169.254.169.254", "fd00:ec2::254"}:
            return True
    return False


class _NoFollowRedirectHandler(HTTPRedirectHandler):
    """Suppress automatic redirect following.

    Returning ``None`` from :meth:`redirect_request` causes the underlying
    ``OpenerDirector`` to fall through to ``HTTPDefaultErrorHandler``,
    which raises ``HTTPError`` for the 3xx response. The caller catches
    that error, reads ``Location``, and revalidates the hop.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _validate_hop(url: str, allow_private: bool) -> str | None:
    """Validate a URL prior to network I/O. Returns an error message or None."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError
    except Exception:
        return f"Invalid URL: {url}"

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return f"Refusing scheme '{parsed.scheme}'. web_fetch only allows http/https."

    if not allow_private:
        host = parsed.hostname or ""
        if not host or _is_private_address(host):
            return (
                f"Refusing to fetch '{host or url}': resolves to a private/loopback/"
                "link-local/reserved address. Set CLAWAGENTS_WEB_ALLOW_PRIVATE=1 to override."
            )
    return None


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
    cacheable = True
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

        allow_private = os.environ.get(
            "CLAWAGENTS_WEB_ALLOW_PRIVATE", ""
        ).strip() in ("1", "true", "yes")

        loop = asyncio.get_running_loop()
        opener = build_opener(_NoFollowRedirectHandler())

        def _fetch_one(target: str):
            req = Request(target, headers={"User-Agent": "ClawAgents/1.0"})
            return opener.open(req, timeout=timeout)

        current = url
        try:
            for hop in range(MAX_REDIRECTS + 1):
                err = _validate_hop(current, allow_private)
                if err is not None:
                    return ToolResult(success=False, output="", error=err)

                try:
                    resp = await loop.run_in_executor(None, _fetch_one, current)
                except HTTPError as e:
                    if 300 <= e.code < 400:
                        if hop >= MAX_REDIRECTS:
                            return ToolResult(
                                success=False,
                                output="",
                                error=f"Too many redirects (>{MAX_REDIRECTS}) starting at {url}",
                            )
                        location = e.headers.get("Location")
                        if not location:
                            return ToolResult(
                                success=False,
                                output="",
                                error=f"HTTP {e.code} without Location header at {current}",
                            )
                        current = urljoin(current, location)
                        continue
                    return ToolResult(success=False, output="", error=f"HTTP {e.code}: {e.reason}")

                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8", errors="replace")

                if len(body) > MAX_RESPONSE_CHARS:
                    body = body[:MAX_RESPONSE_CHARS] + f"\n...(truncated at {MAX_RESPONSE_CHARS} chars)"

                if "html" in content_type.lower():
                    body = _strip_html(body)

                return ToolResult(success=True, output=f"[{status}] {current}\n\n{body}")

            return ToolResult(
                success=False,
                output="",
                error=f"Too many redirects (>{MAX_REDIRECTS}) starting at {url}",
            )

        except URLError as e:
            return ToolResult(success=False, output="", error=f"web_fetch failed: {e.reason}")
        except TimeoutError:
            return ToolResult(success=False, output="", error=f"Request timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"web_fetch failed: {str(e)}")


web_tools: List[Tool] = [WebFetchTool()]
