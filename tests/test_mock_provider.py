"""Tests for the deterministic mock LLM service.

Hits the bound HTTP server with stdlib ``urllib`` (we avoid pulling
``requests`` as a hard test dep), and verifies scenario routing,
shape, and the not-found path.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from clawagents.testing.mock_provider import (
    BUILTIN_SCENARIOS,
    MockLLMService,
    PARITY_SCENARIO_HEADER,
    Scenario,
)


def _post(
    url: str, body: dict, headers: dict[str, str] | None = None
) -> tuple[int, dict]:
    """POST JSON, return (status, parsed body)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Read the body so callers can assert on error payloads.
        return e.code, json.loads(e.read().decode("utf-8"))


# ─── Lifecycle ────────────────────────────────────────────────────────


def test_starts_and_binds_a_port_then_stops() -> None:
    svc = MockLLMService()
    try:
        url = svc.start()
        assert url.startswith("http://127.0.0.1:")
        assert svc.url == url
        # Health probe.
        with urllib.request.urlopen(url + "/health", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read())["ok"] is True
    finally:
        svc.stop()


def test_context_manager_starts_and_stops() -> None:
    with MockLLMService() as svc:
        assert svc.url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(svc.url + "/health", timeout=5) as resp:
            assert resp.status == 200


def test_url_before_start_raises() -> None:
    svc = MockLLMService()
    with pytest.raises(RuntimeError):
        _ = svc.url


# ─── Built-in scenario routing ────────────────────────────────────────


@pytest.mark.parametrize(
    "scenario_name",
    [
        "streaming_text",
        "single_tool_call",
        "multi_tool_turn",
        "bash_permission_denied",
        "truncated_json_recovery",
    ],
)
def test_routes_via_header(scenario_name: str) -> None:
    with MockLLMService() as svc:
        status, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
            headers={PARITY_SCENARIO_HEADER: scenario_name},
        )
    assert status == 200
    # OpenAI-shaped response.
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"


def test_routes_via_system_message_preamble() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "PARITY_SCENARIO: streaming_text"},
            {"role": "user", "content": "anything"},
        ]
    }
    with MockLLMService() as svc:
        status, payload = _post(svc.url + "/v1/chat/completions", body)
    assert status == 200
    assert "hello from mock" in payload["choices"][0]["message"]["content"]


def test_single_tool_call_shape() -> None:
    with MockLLMService() as svc:
        _, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "go"}]},
            headers={PARITY_SCENARIO_HEADER: "single_tool_call"},
        )
    msg = payload["choices"][0]["message"]
    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "echo"
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_multi_tool_turn_shape() -> None:
    with MockLLMService() as svc:
        _, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "go"}]},
            headers={PARITY_SCENARIO_HEADER: "multi_tool_turn"},
        )
    tool_calls = payload["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    names = [tc["function"]["name"] for tc in tool_calls]
    assert names == ["read_file", "read_file"]


def test_truncated_json_recovery_returns_malformed_args() -> None:
    with MockLLMService() as svc:
        _, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "go"}]},
            headers={PARITY_SCENARIO_HEADER: "truncated_json_recovery"},
        )
    args = payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    # Intentionally missing trailing brace so the client must repair it.
    assert args.endswith('"hello world"')
    assert not args.endswith("}")


# ─── Scenario-not-found ───────────────────────────────────────────────


def test_scenario_not_found_returns_404_and_lists_available() -> None:
    with MockLLMService() as svc:
        status, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
            headers={PARITY_SCENARIO_HEADER: "not_a_real_scenario"},
        )
    assert status == 404
    assert payload["error"] == "scenario_not_found"
    assert payload["scenario_tag"] == "not_a_real_scenario"
    assert "streaming_text" in payload["available"]


def test_no_scenario_tag_with_no_predicate_returns_404() -> None:
    # Empty scenario list → everything 404s.
    with MockLLMService(scenarios=[]) as svc:
        status, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
    assert status == 404
    assert payload["error"] == "scenario_not_found"
    assert payload["available"] == []


# ─── Custom scenarios ─────────────────────────────────────────────────


def test_custom_scenario_via_keyword_match() -> None:
    custom = Scenario(
        name="haiku_about_cats",
        keyword_match=["cat"],
        response={"id": "cust", "object": "chat.completion", "model": "m"},
    )
    with MockLLMService(scenarios=[custom]) as svc:
        _, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "write a cat haiku"}]},
        )
    assert payload["id"] == "cust"


def test_custom_scenario_via_request_predicate() -> None:
    def is_long(body: dict) -> bool:
        msgs = body.get("messages") or []
        return any(len(m.get("content", "")) > 50 for m in msgs)

    custom = Scenario(
        name="long_input",
        request_predicate=is_long,
        response={"id": "long", "object": "chat.completion", "model": "m"},
    )
    with MockLLMService(scenarios=[custom]) as svc:
        # Short → 404
        status_short, _ = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        # Long → matches
        status_long, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x" * 100}]},
        )
    assert status_short == 404
    assert status_long == 200
    assert payload["id"] == "long"


def test_callable_response_sees_request_body() -> None:
    seen: dict = {}

    def echo(body: dict) -> dict:
        seen.update(body)
        return {"id": "echoed", "object": "chat.completion", "model": "m"}

    custom = Scenario(name="echo_back", response=echo)
    with MockLLMService(scenarios=[custom]) as svc:
        _, payload = _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "ping"}]},
            headers={PARITY_SCENARIO_HEADER: "echo_back"},
        )
    assert payload["id"] == "echoed"
    assert seen["messages"][0]["content"] == "ping"


def test_request_log_records_traffic() -> None:
    with MockLLMService() as svc:
        _post(
            svc.url + "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "logme"}]},
            headers={PARITY_SCENARIO_HEADER: "streaming_text"},
        )
        assert any(
            r.get("scenario_tag") == "streaming_text" for r in svc.request_log
        )
        assert svc.request_log[-1]["body"]["messages"][0]["content"] == "logme"


# ─── Sanity ───────────────────────────────────────────────────────────


def test_builtin_scenarios_have_unique_names() -> None:
    names = [s.name for s in BUILTIN_SCENARIOS]
    assert len(names) == len(set(names))
    assert "streaming_text" in names
