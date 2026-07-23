"""OpenAI Responses API auto-routing + message/tool conversion."""

from __future__ import annotations

from types import SimpleNamespace

from clawagents.providers.llm import (
    _apply_responses_reasoning,
    _chat_tools_to_responses_tools,
    _messages_to_responses_input,
    _parse_responses_result,
    prefers_responses_api,
)


def test_prefers_responses_for_gpt56_and_codex_on_official():
    assert prefers_responses_api("gpt-5.6-luna")
    assert prefers_responses_api("gpt-5.5-pro")
    assert prefers_responses_api("gpt-5.2-codex")
    assert prefers_responses_api("codex-mini")
    assert not prefers_responses_api("gpt-4o")
    assert not prefers_responses_api("gpt-5.4")


def test_prefers_chat_completions_on_compatible_proxies():
    # Auto still uses Responses for GPT-5.5/5.6 on custom hosts (Responses-only
    # corporate gateways). Force chat_completions when the proxy is chat-only.
    assert prefers_responses_api(
        "gpt-5.6-luna", base_url="http://localhost:11434/v1",
    )
    assert prefers_responses_api(
        "gpt-5.6-luna", base_url="https://bag.example/api/v1",
    )
    assert prefers_responses_api(
        "gpt-5.6-luna", base_url="https://api.openai.com/v1",
    )
    assert not prefers_responses_api("gpt-5.6", api_type="azure")
    assert not prefers_responses_api(
        "gpt-5.6-luna",
        base_url="http://localhost:11434/v1",
        wire_api="chat_completions",
    )
    assert prefers_responses_api(
        "gpt-4o",
        base_url="https://proxy.example/v1",
        wire_api="responses",
    )


def test_prefers_responses_for_reasoning_plus_tools():
    assert prefers_responses_api(
        "o3", has_tools=True, reasoning_effort="high",
    )
    assert not prefers_responses_api(
        "o3", has_tools=True, reasoning_effort="none",
    )
    assert not prefers_responses_api(
        "o3", has_tools=False, reasoning_effort="high",
    )


def test_messages_to_responses_input_maps_tools():
    instructions, items = _messages_to_responses_input(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "run echo"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"x":1}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
    )
    assert instructions == "You are helpful."
    assert items[0] == {"role": "user", "content": "run echo"}
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_1"
    assert items[1]["name"] == "echo"
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "ok",
    }


def test_chat_tools_to_responses_tools_flattens():
    flat = _chat_tools_to_responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert flat == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


def test_apply_responses_reasoning_keeps_effort_with_tools():
    kwargs: dict = {}
    _apply_responses_reasoning(kwargs, preferred="high")
    assert kwargs["reasoning"] == {"effort": "high"}


def test_parse_responses_result_text_and_tools():
    resp = SimpleNamespace(
        output_text="",
        usage=SimpleNamespace(
            total_tokens=42,
            input_tokens=10,
            input_tokens_details=SimpleNamespace(
                cached_tokens=3,
                cache_write_tokens=7,
            ),
        ),
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="hello")],
            ),
            SimpleNamespace(
                type="function_call",
                name="echo",
                arguments='{"x": 2}',
                call_id="call_9",
                id="fc_9",
            ),
        ],
    )
    text, calls, total, prompt, cached, cache_write = _parse_responses_result(resp)
    assert text == "hello"
    assert total == 42
    assert prompt == 10
    assert cached == 3
    assert cache_write == 7
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].tool_name == "echo"
    assert calls[0].args == {"x": 2}
    assert calls[0].tool_call_id == "call_9"
