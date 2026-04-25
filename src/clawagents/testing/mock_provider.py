"""Deterministic fake LLM service for offline e2e tests.

Inspired by ``claw-code-main/rust/crates/mock-anthropic-service/src/lib.rs``
and ``mock_parity_scenarios.json``.

The harness binds a tiny stdlib ``ThreadingHTTPServer`` on
``127.0.0.1:<auto>`` and answers any POST with a scenario-shaped JSON
response. The :class:`OpenAIProvider` / :class:`AnthropicProvider` /
:class:`GeminiProvider` clients can be pointed at it via:

    - ``OPENAI_BASE_URL=http://127.0.0.1:<port>``
    - ``ANTHROPIC_BASE_URL=http://127.0.0.1:<port>``
    - ``GOOGLE_API_BASE_URL=http://127.0.0.1:<port>``

Scenario routing
----------------
A request picks a scenario via either:

    1. an HTTP header ``X-Parity-Scenario: <name>``, or
    2. a system message preamble of the form
       ``PARITY_SCENARIO: <name>`` (any of the JSON message bodies).

A request matches a :class:`Scenario` if its ``request_predicate`` returns
True OR any of its ``keyword_match`` strings appears in a stringified
request body / picked scenario name. The first match in the registered
order wins; otherwise a 404 with ``{"error": "scenario_not_found"}`` is
returned.

No new runtime deps — pure stdlib.
"""

from __future__ import annotations

import http.server
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from urllib.parse import urlparse


__all__ = [
    "MockLLMService",
    "Scenario",
    "BUILTIN_SCENARIOS",
    "PARITY_SCENARIO_HEADER",
    "PARITY_SCENARIO_MARKER",
]


PARITY_SCENARIO_HEADER = "X-Parity-Scenario"
PARITY_SCENARIO_MARKER = "PARITY_SCENARIO:"

# Type aliases for clarity
ResponseFactory = Callable[[dict[str, Any]], dict[str, Any]]
RequestPredicate = Callable[[dict[str, Any]], bool]


@dataclass
class Scenario:
    """One parity scenario: how to match a request and what to return.

    Either ``request_predicate`` or ``keyword_match`` must be provided
    (both is fine).

    Args:
        name: stable identifier; can be used as the scenario tag.
        response: either a fixed JSON-able dict OR a callable taking the
            decoded request body and returning a JSON-able dict.
        request_predicate: optional, takes the decoded request body
            (parsed from JSON; ``{}`` if non-JSON) and returns True if
            this scenario should serve the request.
        keyword_match: optional list of substrings; matches if any
            substring is found in the request body's text representation.
        status: HTTP status to return (default 200).
        headers: extra response headers to send.
    """

    name: str
    response: dict[str, Any] | ResponseFactory
    request_predicate: RequestPredicate | None = None
    keyword_match: list[str] = field(default_factory=list)
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)

    def matches(self, body: dict[str, Any], scenario_tag: str | None) -> bool:
        if scenario_tag and scenario_tag == self.name:
            return True
        if self.request_predicate is not None:
            try:
                if self.request_predicate(body):
                    return True
            except Exception:
                # Predicates must be defensive; if one raises we move on.
                pass
        if self.keyword_match:
            haystack = json.dumps(body, default=str)
            for kw in self.keyword_match:
                if kw in haystack:
                    return True
        return False

    def render(self, body: dict[str, Any]) -> dict[str, Any]:
        if callable(self.response):
            return self.response(body)
        return self.response


# ─── Built-in OpenAI-shaped scenario presets ──────────────────────────


def _chat_completion(
    *,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    model: str = "mock-gpt",
) -> dict[str, Any]:
    """Build an OpenAI Chat Completions response payload."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-mock-001",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


def _streaming_text_response(_body: dict[str, Any]) -> dict[str, Any]:
    return _chat_completion(
        content="hello from mock streaming text",
        finish_reason="stop",
    )


def _single_tool_call_response(_body: dict[str, Any]) -> dict[str, Any]:
    return _chat_completion(
        content="",
        tool_calls=[
            {
                "id": "call_mock_1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": json.dumps({"text": "hi"}),
                },
            }
        ],
        finish_reason="tool_calls",
    )


def _multi_tool_turn_response(_body: dict[str, Any]) -> dict[str, Any]:
    return _chat_completion(
        content="",
        tool_calls=[
            {
                "id": "call_mock_a",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "/tmp/a"}),
                },
            },
            {
                "id": "call_mock_b",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "/tmp/b"}),
                },
            },
        ],
        finish_reason="tool_calls",
    )


def _bash_permission_denied_response(_body: dict[str, Any]) -> dict[str, Any]:
    # Model "tries" to run a shell command that will be denied by the
    # client-side permission layer.
    return _chat_completion(
        content="",
        tool_calls=[
            {
                "id": "call_mock_bash",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"cmd": "rm -rf /"}),
                },
            }
        ],
        finish_reason="tool_calls",
    )


def _truncated_json_recovery_response(_body: dict[str, Any]) -> dict[str, Any]:
    # Args string is intentionally truncated/malformed so the client's
    # JSON-recovery path is exercised. Clients are expected to repair
    # the trailing brace.
    return _chat_completion(
        content="",
        tool_calls=[
            {
                "id": "call_mock_trunc",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"text": "hello world"',  # missing trailing }
                },
            }
        ],
        finish_reason="tool_calls",
    )


BUILTIN_SCENARIOS: list[Scenario] = [
    Scenario(name="streaming_text", response=_streaming_text_response),
    Scenario(name="single_tool_call", response=_single_tool_call_response),
    Scenario(name="multi_tool_turn", response=_multi_tool_turn_response),
    Scenario(
        name="bash_permission_denied",
        response=_bash_permission_denied_response,
    ),
    Scenario(
        name="truncated_json_recovery",
        response=_truncated_json_recovery_response,
    ),
]


# ─── HTTP plumbing ────────────────────────────────────────────────────


def _extract_scenario_tag(
    headers: dict[str, str], body: dict[str, Any]
) -> str | None:
    """Pull the scenario name out of the X-Parity-Scenario header or a
    ``PARITY_SCENARIO: <name>`` system message preamble.
    """
    # Headers are case-insensitive; normalise.
    for k, v in headers.items():
        if k.lower() == PARITY_SCENARIO_HEADER.lower():
            return v.strip() or None

    messages = body.get("messages") if isinstance(body, dict) else None
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str) and PARITY_SCENARIO_MARKER in content:
                line = content.split(PARITY_SCENARIO_MARKER, 1)[1].strip()
                # Take everything up to the next whitespace / newline.
                tag = line.split()[0] if line else ""
                if tag:
                    return tag
    return None


class _MockHandler(http.server.BaseHTTPRequestHandler):
    # Class attributes set by _make_handler_cls. Defaults satisfy mypy.
    scenarios: list[Scenario] = []
    request_log: list[dict[str, Any]] = []

    # Keep stdout quiet during tests.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_body(self) -> tuple[dict[str, Any], bytes]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        body: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    body = parsed
            except json.JSONDecodeError:
                pass
        return body, raw

    def _headers_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.headers.items()}

    def _write_json(
        self, status: int, payload: dict[str, Any], extra_headers: dict[str, str]
    ) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        for k, v in extra_headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(encoded)

    def _serve(self) -> None:
        body, _raw = self._read_body()
        headers = self._headers_dict()
        path = urlparse(self.path).path
        scenario_tag = _extract_scenario_tag(headers, body)

        # Record for tests.
        self.request_log.append(
            {
                "method": self.command,
                "path": path,
                "headers": headers,
                "body": body,
                "scenario_tag": scenario_tag,
            }
        )

        for scenario in self.scenarios:
            if scenario.matches(body, scenario_tag):
                payload = scenario.render(body)
                self._write_json(scenario.status, payload, scenario.headers)
                return

        # Unmatched.
        self._write_json(
            404,
            {
                "error": "scenario_not_found",
                "scenario_tag": scenario_tag,
                "path": path,
                "available": [s.name for s in self.scenarios],
            },
            {},
        )

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler)
        self._serve()

    def do_GET(self) -> None:  # noqa: N802
        # Health probe convenience.
        if self.path.rstrip("/") in ("/health", "/_mock/health"):
            self._write_json(200, {"ok": True}, {})
            return
        self._serve()


def _make_handler_cls(
    scenarios: list[Scenario], request_log: list[dict[str, Any]]
) -> type[_MockHandler]:
    class _BoundHandler(_MockHandler):
        pass

    _BoundHandler.scenarios = scenarios
    _BoundHandler.request_log = request_log
    return _BoundHandler


# ─── Service ──────────────────────────────────────────────────────────


class MockLLMService:
    """Deterministic fake LLM service.

    Usage::

        with MockLLMService() as mock:
            os.environ["OPENAI_BASE_URL"] = mock.url + "/v1"
            ...   # run real provider clients against the mock

    Args:
        port: bind port (``0`` → OS picks a free one). Default ``0``.
        scenarios: scenario list. Defaults to :data:`BUILTIN_SCENARIOS`.
            Pass ``[]`` for an empty service that always 404s.
        host: bind host (default ``"127.0.0.1"``).
    """

    def __init__(
        self,
        *,
        port: int = 0,
        scenarios: Iterable[Scenario] | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self._host = host
        self._port = port
        self._scenarios: list[Scenario] = (
            list(scenarios) if scenarios is not None else list(BUILTIN_SCENARIOS)
        )
        self.request_log: list[dict[str, Any]] = []
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self) -> str:
        if self._server is not None:
            assert self._url is not None
            return self._url
        handler_cls = _make_handler_cls(self._scenarios, self.request_log)
        self._server = http.server.ThreadingHTTPServer(
            (self._host, self._port), handler_cls
        )
        actual_port = self._server.server_address[1]
        self._url = f"http://{self._host}:{actual_port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="mock-llm-service",
        )
        self._thread.start()
        return self._url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._url = None

    # ── scenarios ──────────────────────────────────────────────────

    def add_scenario(self, scenario: Scenario) -> None:
        """Append a custom scenario after construction.

        Note: scenarios already registered with the live handler will see
        the new entry too — both share the same list reference.
        """
        self._scenarios.append(scenario)

    @property
    def scenarios(self) -> list[Scenario]:
        return self._scenarios

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("MockLLMService not started")
        return self._url

    # ── context manager ───────────────────────────────────────────

    def __enter__(self) -> "MockLLMService":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
