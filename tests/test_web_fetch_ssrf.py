"""Hermetic SSRF tests for ``web_fetch``.

Scenarios covered:
  1. A redirect from a "public" host to a private IP is refused at hop 2 —
     this is the bypass that motivated the redirect-aware rewrite.
  2. A redirect chain that exceeds ``MAX_REDIRECTS`` is rejected.
  3. A direct fetch of a private IP is refused at hop 1 (regression check).
  4. A redirect from a "public" host to another "public" host succeeds —
     proving we did not break legitimate redirects.

The "public" classification is faked by monkey-patching
``_is_private_address``; the test server still binds to 127.0.0.1 because
that is the only host we can rely on in CI sandboxes.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from clawagents.tools import web
from clawagents.tools.web import WebFetchTool


def _start_server(handler_cls: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def _make_handler(routes: dict[str, tuple[int, dict[str, str], bytes]]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            entry = routes.get(self.path)
            if entry is None:
                self.send_response(404)
                self.end_headers()
                return
            status, headers, body = entry
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, *args, **kwargs):  # noqa: D401
            return

    return Handler


@pytest.fixture
def fake_validator(monkeypatch: pytest.MonkeyPatch):
    """Patch ``_is_private_address`` so 127.0.0.1 is 'public' for hop 1
    while every other IP is treated according to a caller-provided
    allowlist. The fixture returns a setter so each test can declare which
    redirect targets count as private.
    """

    private_overrides: set[str] = set()

    def fake(host: str, **_kwargs: object) -> bool:
        if host in private_overrides:
            return True
        if host == "127.0.0.1":
            return False
        return True

    monkeypatch.setattr(web, "_is_private_address", fake)

    def configure(*hosts: str) -> None:
        private_overrides.update(hosts)

    return configure


def test_redirect_to_private_ip_is_refused(fake_validator):
    """Bypass scenario: hop 1 looks public, hop 2 is 169.254.169.254."""
    fake_validator("169.254.169.254")
    routes = {
        "/r": (
            302,
            {"Location": "http://169.254.169.254/latest/meta-data/"},
            b"",
        ),
    }
    server, port = _start_server(_make_handler(routes))
    try:
        tool = WebFetchTool()
        result = asyncio.run(tool.execute({"url": f"http://127.0.0.1:{port}/r"}))
    finally:
        server.shutdown()

    assert result.success is False
    assert "169.254.169.254" in (result.error or "")
    assert "private" in (result.error or "")


def test_redirect_to_loopback_is_refused(fake_validator):
    """Hop 1 'public' (127.0.0.1 stub), hop 2 = 10.0.0.1 (RFC1918)."""
    fake_validator("10.0.0.1")
    routes = {
        "/r": (302, {"Location": "http://10.0.0.1/admin"}, b""),
    }
    server, port = _start_server(_make_handler(routes))
    try:
        tool = WebFetchTool()
        result = asyncio.run(tool.execute({"url": f"http://127.0.0.1:{port}/r"}))
    finally:
        server.shutdown()

    assert result.success is False
    assert "10.0.0.1" in (result.error or "")


def test_redirect_chain_too_long(fake_validator):
    """Loop redirect: every hop is 'public'; bail out after MAX_REDIRECTS."""
    routes = {"/r": (302, {"Location": "/r"}, b"")}
    server, port = _start_server(_make_handler(routes))
    try:
        tool = WebFetchTool()
        result = asyncio.run(tool.execute({"url": f"http://127.0.0.1:{port}/r"}))
    finally:
        server.shutdown()

    assert result.success is False
    assert "Too many redirects" in (result.error or "")


def test_direct_private_is_still_refused():
    """Without the fake validator, the real check refuses 127.0.0.1."""
    tool = WebFetchTool()
    result = asyncio.run(tool.execute({"url": "http://127.0.0.1/x"}))
    assert result.success is False
    assert "127.0.0.1" in (result.error or "")
    assert "private" in (result.error or "")


def test_public_to_public_redirect_succeeds(fake_validator):
    """A relative 302 to another 'public' endpoint on the same host must
    follow through and return the body of the final hop. Proves the
    hardening did not break legitimate redirect chains.
    """
    routes = {
        "/r": (302, {"Location": "/dest"}, b""),
        "/dest": (200, {"Content-Type": "text/plain"}, b"final-body"),
    }
    server, port = _start_server(_make_handler(routes))
    try:
        tool = WebFetchTool()
        result = asyncio.run(tool.execute({"url": f"http://127.0.0.1:{port}/r"}))
    finally:
        server.shutdown()

    assert result.success is True, result.error
    assert "final-body" in result.output
    assert "/dest" in result.output
