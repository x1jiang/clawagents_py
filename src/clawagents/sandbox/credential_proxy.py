"""Credential proxy for sandboxed agent environments.

Real API keys never enter subprocess/container environments.
The proxy intercepts requests and injects credentials transparently.

Usage::

    proxy = CredentialProxy({"Authorization": "Bearer sk-..."})
    url = proxy.start()          # e.g. "http://127.0.0.1:54321"
    # point sub-agent at url, strip real keys from its env
    proxy.stop()

Uses only stdlib (http.server + urllib.request) — no extra dependencies.
"""

from __future__ import annotations

import http.server
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_DEFAULT_UPSTREAM_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that injects credentials and forwards requests."""

    # Set by CredentialProxy before the server starts
    credentials: dict[str, str] = {}
    upstream_base_urls: dict[str, str] = _DEFAULT_UPSTREAM_BASE_URLS

    # ── suppress default request logging ──────────────────────────────────
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        pass

    def _send_plain(self, status: int, text: str) -> None:
        body_bytes = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def _select_upstream(self) -> str:
        if self.headers.get("anthropic-version") or self.headers.get("x-api-key"):
            return self.upstream_base_urls.get("anthropic", _DEFAULT_UPSTREAM_BASE_URLS["anthropic"])
        return self.upstream_base_urls.get("openai", _DEFAULT_UPSTREAM_BASE_URLS["openai"])

    def _target_url(self) -> str:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.scheme and parsed.netloc:
            return self.path

        base = self._select_upstream().rstrip("/")
        base_parsed = urllib.parse.urlparse(base)
        path = self.path if self.path.startswith("/") else f"/{self.path}"
        base_path = base_parsed.path.rstrip("/")
        if base_path and path.startswith(f"{base_path}/"):
            return urllib.parse.urlunparse((
                base_parsed.scheme, base_parsed.netloc, path, "", "", "",
            ))
        return urllib.parse.urljoin(f"{base}/", path.lstrip("/"))

    def _is_allowed_target(self, target: str) -> bool:
        parsed = urllib.parse.urlparse(target)
        origin = (parsed.scheme, parsed.netloc)
        return origin in {
            (
                urllib.parse.urlparse(url).scheme,
                urllib.parse.urlparse(url).netloc,
            )
            for url in self.upstream_base_urls.values()
        }

    def _credential_applies_to_target(self, header_name: str, target: str) -> bool:
        host = urllib.parse.urlparse(target).hostname or ""
        lower_name = header_name.lower()
        if "anthropic" in host:
            return lower_name == "x-api-key"
        if "openai" in host:
            return lower_name == "authorization"
        return True

    def _forward(self, body: bytes | None = None) -> None:
        try:
            target = self._target_url()
            if not self._is_allowed_target(target):
                self._send_plain(403, f"Refusing to proxy untrusted upstream: {target}")
                return

            # Build the upstream request
            req = urllib.request.Request(target)
            req.method = self.command

            # Copy headers from client, then inject credentials
            for key, value in self.headers.items():
                lower = key.lower()
                # skip hop-by-hop headers
                if lower in ("host", "content-length", "transfer-encoding", "connection"):
                    continue
                req.add_header(key, value)

            for header_name, header_value in self.credentials.items():
                if self._credential_applies_to_target(header_name, target):
                    req.add_header(header_name, header_value)

            if body:
                req.data = body

            with _NO_REDIRECT_OPENER.open(req, timeout=60) as resp:
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    lower = key.lower()
                    if lower in ("transfer-encoding", "connection"):
                        continue
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as exc:
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                lower = key.lower()
                if lower in ("transfer-encoding", "connection"):
                    continue
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(exc.read())
        except Exception as exc:
            body_bytes = str(exc).encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

    def _read_body(self) -> bytes | None:
        length = self.headers.get("Content-Length")
        if length:
            return self.rfile.read(int(length))
        return None

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward(self._read_body())

    def do_PUT(self) -> None:
        self._forward(self._read_body())

    def do_PATCH(self) -> None:
        self._forward(self._read_body())

    def do_DELETE(self) -> None:
        self._forward()

    def do_HEAD(self) -> None:
        self._forward()

    def do_OPTIONS(self) -> None:
        self._forward()


class CredentialProxy:
    """Lightweight HTTP proxy that injects API credentials into forwarded requests.

    Args:
        credentials: Mapping of header name → header value to inject.
            Example: ``{"Authorization": "Bearer sk-...", "x-api-key": "..."}``
        host: Bind address (default ``"127.0.0.1"``).
        port: Port to listen on. ``0`` means OS auto-assigns a free port.
    """

    def __init__(
        self,
        credentials: dict[str, str],
        host: str = "127.0.0.1",
        port: int = 0,
        upstream_base_urls: dict[str, str] | None = None,
    ) -> None:
        self._credentials = dict(credentials)
        self._host = host
        self._port = port
        self._upstream_base_urls = dict(upstream_base_urls or _DEFAULT_UPSTREAM_BASE_URLS)
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    def start(self) -> str:
        """Start the proxy and return its base URL (e.g. ``"http://127.0.0.1:54321"``).

        The proxy runs in a daemon thread so it does not block process exit.
        Calling :meth:`stop` is still recommended for clean shutdown.
        """
        if self._server is not None:
            return self._url  # type: ignore[return-value]

        # Build a handler class with the credentials baked in via class attribute
        creds = self._credentials

        class _BoundHandler(_ProxyHandler):
            credentials = creds
            upstream_base_urls = self._upstream_base_urls

        self._server = http.server.HTTPServer((self._host, self._port), _BoundHandler)
        actual_port = self._server.server_address[1]
        self._url = f"http://{self._host}:{actual_port}"

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="credential-proxy",
        )
        self._thread.start()
        return self._url

    def stop(self) -> None:
        """Shut down the proxy server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._url = None

    @property
    def url(self) -> str | None:
        """The proxy URL after :meth:`start` is called, else ``None``."""
        return self._url

    def __enter__(self) -> "CredentialProxy":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
