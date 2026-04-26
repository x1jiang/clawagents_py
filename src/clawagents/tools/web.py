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

import http.client
import os
import re
import asyncio
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urljoin

from clawagents.tools.registry import Tool, ToolResult

MAX_RESPONSE_CHARS = 50_000
MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MiB hard cap on body bytes read
DEFAULT_TIMEOUT_S = 15
MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = ("http", "https")
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _ip_is_private(ip: ipaddress._BaseAddress) -> bool:
    if (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
    ):
        return True
    return str(ip) in {"169.254.169.254", "fd00:ec2::254"}


def _is_private_address(host: str, *, _resolved: list[str] | None = None) -> bool:
    """Return True if *host* is or resolves to a non-public IP.

    When *_resolved* is supplied, skip DNS and use those IPs directly —
    this lets the orchestrator share a single ``getaddrinfo`` between
    the privacy check and the IP pin. Tests monkey-patch this function
    to flip 127.0.0.1 between "private" and "public"; their fakes don't
    inspect *_resolved*, so kwarg-only is fine.
    """
    if _resolved is not None:
        for ip_str in _resolved:
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return True
            if _ip_is_private(ip):
                return True
        return False
    try:
        return _ip_is_private(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    if not infos:
        return True
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if _ip_is_private(ip):
            return True
    return False


@dataclass(frozen=True)
class PinnedTarget:
    scheme: str
    host: str
    port: int
    ip: str
    path: str


def _validate_hop(url: str, allow_private: bool) -> str | PinnedTarget:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError
    except Exception:
        return f"Invalid URL: {url}"

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return f"Refusing scheme '{parsed.scheme}'. web_fetch only allows http/https."

    host = parsed.hostname or ""
    if not host:
        return f"Invalid URL (no host): {url}"

    port = parsed.port or _DEFAULT_PORTS[scheme]
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    # One DNS resolution shared between the privacy check and the
    # connection pin: closes the TOCTOU window between them and saves
    # an RTT per redirect hop.
    try:
        ipaddress.ip_address(host)
        ip: str | None = host
        resolved_ips: list[str] | None = None  # IP literal: skip DNS
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return f"DNS lookup failed for '{host}'"
        if not infos:
            return f"DNS lookup returned no records for '{host}'"
        resolved_ips = [info[4][0] for info in infos]
        ip = resolved_ips[0]

    if not allow_private and _is_private_address(host, _resolved=resolved_ips):
        return (
            f"Refusing to fetch '{host}': resolves to a private/loopback/"
            "link-local/reserved address. Set CLAWAGENTS_WEB_ALLOW_PRIVATE=1 to override."
        )

    if ip is None:
        return f"DNS lookup failed for '{host}'"
    return PinnedTarget(scheme=scheme, host=host, port=port, ip=ip, path=path)


def _fetch_pinned(
    target: PinnedTarget,
    timeout: int,
) -> Tuple[int, Dict[str, str], bytes]:
    """Open a single HTTP(S) connection to ``target.ip`` with the original
    hostname in the ``Host`` header and (for TLS) as SNI. Pinning the IP
    neutralises DNS rebinding for this hop. Redirect responses are
    returned as-is (no auto-follow).
    """
    headers = {
        "Host": target.host if target.port in (80, 443) else f"{target.host}:{target.port}",
        "User-Agent": "ClawAgents/1.0",
        "Connection": "close",
        "Accept-Encoding": "identity",
    }
    if target.scheme == "https":
        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(
            target.ip, target.port, timeout=timeout, context=ctx, server_hostname=target.host,
        )
    else:
        conn = http.client.HTTPConnection(target.ip, target.port, timeout=timeout)
    try:
        conn.request("GET", target.path, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        # Cap by Content-Length first if reasonable.
        clen_raw = resp.getheader("Content-Length")
        try:
            clen = int(clen_raw) if clen_raw is not None else None
        except ValueError:
            clen = None
        if clen is not None and clen > MAX_RESPONSE_BYTES:
            # Drain just enough to give a readable preview, then bail.
            body = resp.read(MAX_RESPONSE_BYTES)
        else:
            # Stream-read up to MAX_RESPONSE_BYTES; reject larger payloads.
            body = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                body = body[:MAX_RESPONSE_BYTES]
        # Snapshot headers as a plain dict (last-write-wins on duplicates).
        # Lowercase keys so callers can do simple ``hdrs["location"]``
        # lookups regardless of how the server cased the header.
        hdrs: Dict[str, str] = {}
        for k, v in resp.getheaders():
            hdrs[k.lower()] = v
        return status, hdrs, body
    finally:
        conn.close()


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
    keywords = ["fetch url", "http request", "read webpage", "download text", "documentation"]
    description = (
        "Fetch content from a URL. Returns the text content of the page. "
        "Useful for reading documentation, API responses, or checking web resources. "
        "HTML is stripped for readability. JSON responses are returned as-is."
    )
    parameters: Dict[str, Dict[str, Any]] = {
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
        current = url
        try:
            for hop in range(MAX_REDIRECTS + 1):
                hop_info = _validate_hop(current, allow_private)
                if isinstance(hop_info, str):
                    return ToolResult(success=False, output="", error=hop_info)

                try:
                    status, headers, body = await loop.run_in_executor(
                        None, _fetch_pinned, hop_info, timeout
                    )
                except TimeoutError:
                    return ToolResult(
                        success=False, output="",
                        error=f"Request timed out after {timeout}s",
                    )
                except OSError as e:
                    return ToolResult(success=False, output="", error=f"web_fetch failed: {e}")

                if 300 <= status < 400:
                    if hop >= MAX_REDIRECTS:
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"Too many redirects (>{MAX_REDIRECTS}) starting at {url}",
                        )
                    location = headers.get("location")
                    if not location:
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"HTTP {status} without Location header at {current}",
                        )
                    next_url = urljoin(current, location)
                    if hop_info.scheme == "https" and next_url.lower().startswith("http://"):
                        return ToolResult(
                            success=False,
                            output="",
                            error=(
                                "Refusing redirect: HTTPS endpoint sent a "
                                "Location pointing to http:// (TLS downgrade)"
                            ),
                        )
                    current = next_url
                    continue

                if not (200 <= status < 300):
                    return ToolResult(success=False, output="", error=f"HTTP {status}")

                content_type = headers.get("content-type") or ""
                text = body.decode("utf-8", errors="replace")

                if len(text) > MAX_RESPONSE_CHARS:
                    text = text[:MAX_RESPONSE_CHARS] + f"\n...(truncated at {MAX_RESPONSE_CHARS} chars)"

                if "html" in content_type.lower():
                    text = _strip_html(text)

                return ToolResult(success=True, output=f"[{status}] {current}\n\n{text}")

            return ToolResult(
                success=False,
                output="",
                error=f"Too many redirects (>{MAX_REDIRECTS}) starting at {url}",
            )

        except (OSError, ssl.SSLError) as e:
            return ToolResult(success=False, output="", error=f"web_fetch failed: {e}")


web_tools: List[Tool] = [WebFetchTool()]
