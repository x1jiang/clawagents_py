"""Situation suite: nested array-of-object tool schemas + tool-pair sanitizers.

Mirrors the live Gemini 400 class (``properties[command].items: missing field``)
and the luna/Responses orphan-tool investigations without hitting a network.
"""

from __future__ import annotations

from typing import Any

import pytest

from clawagents.mcp.tool_bridge import _normalize_input_schema
from clawagents.providers.llm import (
    LLMMessage,
    NativeToolSchema,
    _chat_tools_to_responses_tools,
    _openai_chat_messages,
    _sanitize_openai_tool_pairs,
    _to_gemini_tools,
    _to_openai_tools,
)
from clawagents.providers.tool_schema import gemini_array_items_valid


# Minimal MCP-style schema matching ctx_batch_execute / similar batch tools.
_BATCH_MCP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "commands": {
            "type": "array",
            "description": "Commands to run",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Short label"},
                    "command": {"type": "string", "description": "Shell command"},
                },
                "required": ["label", "command"],
            },
        },
        "timeout_ms": {"type": "integer", "description": "Timeout"},
    },
    "required": ["commands"],
}


def _batch_params() -> dict[str, dict[str, Any]]:
    return _normalize_input_schema(_BATCH_MCP_SCHEMA)


def _batch_schema() -> NativeToolSchema:
    return NativeToolSchema(
        name="ctx_batch_execute",
        description="Run labeled commands",
        parameters=_batch_params(),
    )


# ---------------------------------------------------------------------------
# Schema situations
# ---------------------------------------------------------------------------


def test_bridge_preserves_array_of_object_items():
    params = _batch_params()
    items = params["commands"]["items"]
    assert items["type"] == "object"
    assert "label" in items["properties"]
    assert "command" in items["properties"]
    assert set(items.get("required") or []) == {"label", "command"}
    assert params["commands"]["required"] is True
    assert params["timeout_ms"]["required"] is False


def test_openai_emitter_keeps_nested_properties():
    oai = _to_openai_tools([_batch_schema()])
    commands = oai[0]["function"]["parameters"]["properties"]["commands"]
    assert commands["type"] == "array"
    assert commands["items"]["type"] == "object"
    assert commands["items"]["properties"]["label"]["type"] == "string"
    assert commands["items"]["properties"]["command"]["type"] == "string"
    assert set(commands["items"]["required"]) == {"label", "command"}


def test_responses_tools_inherit_nested_schema_from_openai():
    oai = _to_openai_tools([_batch_schema()])
    resp = _chat_tools_to_responses_tools(oai)
    assert resp is not None
    commands = resp[0]["parameters"]["properties"]["commands"]
    assert commands["items"]["properties"]["command"]["type"] == "string"


def test_gemini_emitter_has_items_with_properties_not_bare_object():
    gem = _to_gemini_tools([_batch_schema()])
    decl = gem[0]["function_declarations"][0]
    commands = decl["parameters"]["properties"]["commands"]
    assert commands["type"] == "ARRAY"
    items = commands["items"]
    assert items["type"] == "OBJECT"
    # The live 400 was "items: missing field" / bare OBJECT — properties must exist.
    assert "properties" in items
    assert items["properties"]["label"]["type"] == "STRING"
    assert items["properties"]["command"]["type"] == "STRING"
    assert set(items["required"]) == {"label", "command"}
    assert gemini_array_items_valid(decl["parameters"])


def test_gemini_rejects_bare_object_items_shape_as_invalid():
    """Document the pre-fix failure mode for the situation harness."""
    bare = {
        "type": "OBJECT",
        "properties": {
            "commands": {"type": "ARRAY", "items": {"type": "OBJECT"}},
        },
    }
    # Bare OBJECT items still have a type — gemini_array_items_valid only checks
    # type presence. The API still 400s without properties; emitter must add them.
    assert gemini_array_items_valid(bare)
    fixed = _to_gemini_tools([_batch_schema()])[0]["function_declarations"][0]
    items = fixed["parameters"]["properties"]["commands"]["items"]
    assert items.get("properties"), "nested properties required to avoid Gemini 400"


@pytest.mark.parametrize(
    "depth_schema",
    [
        {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                }
            },
            "required": ["matrix"],
        },
        {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name"],
                    },
                }
            },
            "required": ["groups"],
        },
    ],
)
def test_deeper_nesting_survives_all_emitters(depth_schema: dict[str, Any]):
    params = _normalize_input_schema(depth_schema)
    schema = NativeToolSchema(name="nested", description="n", parameters=params)
    oai = _to_openai_tools([schema])[0]["function"]["parameters"]
    gem = _to_gemini_tools([schema])[0]["function_declarations"][0]["parameters"]
    assert gemini_array_items_valid(gem)
    # Spot-check first top-level array still has typed items
    for props in (oai["properties"], gem["properties"]):
        top = next(iter(props.values()))
        assert "items" in top


# ---------------------------------------------------------------------------
# Sanitizer situations (real OpenAI chat → Responses path input shape)
# ---------------------------------------------------------------------------


def _assistant_with_calls(*ids: str) -> LLMMessage:
    return LLMMessage(
        role="assistant",
        content="",
        tool_calls_meta=[
            {"id": i, "name": "execute", "args": {"command": "true"}} for i in ids
        ],
    )


def test_sanitize_drops_orphan_tool_result():
    messages = [
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="tool", content="orphan", tool_call_id="missing"),
        LLMMessage(role="assistant", content="ok"),
    ]
    formatted = _sanitize_openai_tool_pairs(_openai_chat_messages(messages))
    assert all(m.get("role") != "tool" for m in formatted)
    assert formatted[-1]["role"] == "assistant"


def test_sanitize_backfills_dangling_tool_call():
    messages = [
        LLMMessage(role="user", content="hi"),
        _assistant_with_calls("c1", "c2"),
        LLMMessage(role="tool", content="done", tool_call_id="c1"),
    ]
    formatted = _sanitize_openai_tool_pairs(_openai_chat_messages(messages))
    tool_ids = [m["tool_call_id"] for m in formatted if m.get("role") == "tool"]
    assert set(tool_ids) == {"c1", "c2"}
    backfill = next(m for m in formatted if m.get("tool_call_id") == "c2")
    assert "cancelled" in backfill["content"].lower() or "interrupted" in backfill["content"].lower()


def test_sanitize_parallel_partial_results():
    messages = [
        LLMMessage(role="user", content="hi"),
        _assistant_with_calls("a", "b", "c"),
        LLMMessage(role="tool", content="1", tool_call_id="a"),
        LLMMessage(role="tool", content="3", tool_call_id="c"),
    ]
    formatted = _sanitize_openai_tool_pairs(_openai_chat_messages(messages))
    ids = [m["tool_call_id"] for m in formatted if m.get("role") == "tool"]
    assert set(ids) == {"a", "b", "c"}
    assert ids.count("b") == 1


def test_sanitize_then_responses_input_does_not_raise():
    """End-to-end: sanitize → responses convert (luna path) for messy history."""
    from clawagents.providers.llm import _messages_to_responses_input

    messages = [
        LLMMessage(role="system", content="You are helpful."),
        LLMMessage(role="user", content="run tools"),
        _assistant_with_calls("x1"),
        LLMMessage(role="tool", content="orphan-wrong", tool_call_id="nope"),
        LLMMessage(role="tool", content="ok", tool_call_id="x1"),
        _assistant_with_calls("x2"),  # dangling — needs backfill
    ]
    formatted = _sanitize_openai_tool_pairs(_openai_chat_messages(messages))
    instructions, items = _messages_to_responses_input(formatted)
    assert instructions and "helpful" in instructions
    assert isinstance(items, list) and items
    call_ids = {
        it.get("call_id")
        for it in items
        if it.get("type") in ("function_call", "function_call_output")
    }
    assert "nope" not in call_ids
    assert "x1" in call_ids
    assert "x2" in call_ids


def test_flat_string_array_still_works():
    schema = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["paths"],
    }
    params = _normalize_input_schema(schema)
    gem = _to_gemini_tools(
        [NativeToolSchema(name="read_many", description="d", parameters=params)]
    )
    items = gem[0]["function_declarations"][0]["parameters"]["properties"]["paths"]["items"]
    assert items == {"type": "STRING"} or items.get("type") == "STRING"


# ---------------------------------------------------------------------------
# Gemini leading-turn / pair situations (cross-check existing sanitizer)
# ---------------------------------------------------------------------------


def test_gemini_leading_model_dropped():
    from clawagents.providers.llm import _sanitize_gemini_contents

    out = _sanitize_gemini_contents(
        [
            {"role": "model", "parts": [{"text": "orphan"}]},
            {"role": "user", "parts": [{"text": "hi"}]},
        ]
    )
    assert out == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_gemini_backfills_missing_function_response():
    from clawagents.providers.llm import _sanitize_gemini_contents

    out = _sanitize_gemini_contents(
        [
            {"role": "user", "parts": [{"text": "hi"}]},
            {
                "role": "model",
                "parts": [{"function_call": {"name": "execute", "args": {}, "id": "g1"}}],
            },
        ]
    )
    assert out[-1]["role"] == "user"
    assert out[-1]["parts"][0]["function_response"]["id"] == "g1"


def test_anthropic_emitter_preserves_nested_items():
    from clawagents.providers.tool_schema import emit_openai_schema_node

    params = _batch_params()
    prop = emit_openai_schema_node(params["commands"])
    assert prop["items"]["properties"]["command"]["type"] == "string"
