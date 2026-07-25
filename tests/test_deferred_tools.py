"""Cache-preserving tool activation: which tools defer, and referencing them."""

from __future__ import annotations

import json

from clawagents.providers.deferred_tools import (
    split_deferred_tools,
    tool_reference_blocks,
)
from clawagents.providers.llm import LLMMessage, NativeToolSchema


def _schema(name: str) -> NativeToolSchema:
    return NativeToolSchema(name, f"{name} description", {})


def _activation(call_id: str, added: list[str]) -> LLMMessage:
    return LLMMessage(
        role="tool",
        content=json.dumps({"activated": "web", "tools": added}),
        tool_call_id=call_id,
        added_tool_names=added,
    )


def test_tools_added_by_a_tool_result_are_deferred():
    messages = [
        LLMMessage(role="user", content="search the web"),
        _activation("c1", ["web_search", "web_fetch"]),
    ]
    schemas = [_schema("read_file"), _schema("web_search"), _schema("web_fetch")]

    immediate, deferred = split_deferred_tools(messages, schemas)

    assert [s.name for s in immediate] == ["read_file"]
    assert sorted(deferred) == ["web_fetch", "web_search"]


def test_deferred_stays_deferred_once_the_model_calls_it():
    """Stability is the point: the deferred set must not churn as turns arrive.

    Promoting a tool into the prefix after it gets called would rewrite the
    cached prefix — the exact cost this mechanism avoids — so an activation
    that deferred a tool keeps deferring it for the rest of the run.
    """
    messages = [
        _activation("c1", ["web_search"]),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c2", "name": "web_search", "args": {}}],
        ),
    ]
    immediate, deferred = split_deferred_tools(messages, [_schema("web_search")])

    assert immediate == []
    assert list(deferred) == ["web_search"]


def test_a_tool_called_before_its_activation_is_not_deferred():
    """Already established in the prefix — referencing it would be redundant."""
    messages = [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c1", "name": "web_search", "args": {}}],
        ),
        _activation("c2", ["web_search"]),
    ]
    immediate, deferred = split_deferred_tools(messages, [_schema("web_search")])

    assert [s.name for s in immediate] == ["web_search"]
    assert deferred == {}


def test_disabled_keeps_every_tool_immediate():
    messages = [_activation("c1", ["web_search"])]
    schemas = [_schema("read_file"), _schema("web_search")]

    immediate, deferred = split_deferred_tools(messages, schemas, enabled=False)

    assert sorted(s.name for s in immediate) == ["read_file", "web_search"]
    assert deferred == {}


def test_duplicate_schemas_collapse_by_name():
    immediate, _ = split_deferred_tools([], [_schema("read_file"), _schema("read_file")])
    assert len(immediate) == 1


def test_reference_blocks_introduce_each_tool_exactly_once():
    msg = _activation("c1", ["web_search", "web_fetch"])
    seen: set[str] = set()

    first = tool_reference_blocks(msg, {"web_search", "web_fetch"}, seen)
    assert first == [
        {"type": "tool_reference", "tool_name": "web_search"},
        {"type": "tool_reference", "tool_name": "web_fetch"},
    ]

    # A repeat reference to an already-introduced tool is rejected upstream.
    assert tool_reference_blocks(msg, {"web_search", "web_fetch"}, seen) == []


def test_reference_blocks_skip_tools_that_are_not_deferred():
    msg = _activation("c1", ["web_search"])
    assert tool_reference_blocks(msg, set(), set()) == []


def test_activation_tool_records_what_it_added():
    """The activating ToolResult must carry the names for the transcript."""
    from clawagents.tools.registry import ToolResult

    result = ToolResult(True, "{}", added_tool_names=["web_search"])
    assert result.added_tool_names == ["web_search"]
    # Default stays None so ordinary results are unaffected.
    assert ToolResult(True, "ok").added_tool_names is None
