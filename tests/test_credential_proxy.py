from __future__ import annotations

import threading
import http.client
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from clawagents.sandbox.credential_proxy import CredentialProxy


def _start_server(handler_cls: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def test_proxy_forwards_path_only_requests_to_configured_upstream() -> None:
    seen: dict[str, str | None] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args, **kwargs):  # noqa: D401
            return

    upstream, port = _start_server(Handler)
    proxy = CredentialProxy(
        {"Authorization": "Bearer real"},
        upstream_base_urls={"openai": f"http://127.0.0.1:{port}/v1"},
    )
    url = proxy.start()
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=5) as resp:
            assert resp.read() == b"ok"
    finally:
        proxy.stop()
        upstream.shutdown()

    assert seen == {"path": "/v1/models", "auth": "Bearer real"}


def test_proxy_rejects_absolute_urls_outside_configured_upstreams() -> None:
    proxy = CredentialProxy(
        {"Authorization": "Bearer real"},
        upstream_base_urls={"openai": "https://api.openai.com/v1"},
    )
    url = proxy.start()
    try:
        parsed = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            conn.request("GET", "https://example.com/v1/models")
            resp = conn.getresponse()
            assert resp.status == 403
        finally:
            conn.close()
    finally:
        proxy.stop()


def test_proxy_rejects_protocol_downgrade_for_allowed_hosts() -> None:
    proxy = CredentialProxy(
        {"Authorization": "Bearer real"},
        upstream_base_urls={"openai": "https://api.openai.com/v1"},
    )
    url = proxy.start()
    try:
        parsed = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            conn.request("GET", "http://api.openai.com/v1/models")
            resp = conn.getresponse()
            assert resp.status == 403
        finally:
            conn.close()
    finally:
        proxy.stop()


def test_proxy_does_not_follow_upstream_redirects_with_credentials() -> None:
    captured: list[str | None] = []

    class CaptureHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            captured.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args, **kwargs):  # noqa: D401
            return

    capture_server, capture_port = _start_server(CaptureHandler)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{capture_port}/capture")
            self.end_headers()

        def log_message(self, *args, **kwargs):  # noqa: D401
            return

    upstream, upstream_port = _start_server(RedirectHandler)
    proxy = CredentialProxy(
        {"Authorization": "Bearer real"},
        upstream_base_urls={"openai": f"http://127.0.0.1:{upstream_port}/v1"},
    )
    url = proxy.start()
    try:
        parsed = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            assert resp.status == 302
        finally:
            conn.close()
    finally:
        proxy.stop()
        upstream.shutdown()
        capture_server.shutdown()

    assert captured == []
