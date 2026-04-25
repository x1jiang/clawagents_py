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
import urllib.request
from typing import Any


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that injects credentials and forwards requests."""

    # Set by CredentialProxy before the server starts
    credentials: dict[str, str] = {}

    # ── suppress default request logging ──────────────────────────────────
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        pass

    def _forward(self, body: bytes | None = None) -> None:
        target = self.path  # proxy receives absolute URLs or path-only

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
            req.add_header(header_name, header_value)

        if body:
            req.data = body

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
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
    ) -> None:
        self._credentials = dict(credentials)
        self._host = host
        self._port = port
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
