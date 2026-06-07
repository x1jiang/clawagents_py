"""Machine-readable CLI output formats (OpenHarness-style)."""

from __future__ import annotations

import json
import sys
from enum import Enum
from typing import Any, Callable

from clawagents.graph.agent_loop import AgentState, EventKind, OnEvent


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


def parse_output_format(value: str | None) -> OutputFormat:
    if not value:
        return OutputFormat.TEXT
    normalized = value.strip().lower()
    for fmt in OutputFormat:
        if fmt.value == normalized:
            return fmt
    raise ValueError(f"Unsupported output format: {value!r} (use text, json, stream-json)")


def serialize_agent_state(state: AgentState) -> dict[str, Any]:
    usage = state.usage.to_dict() if hasattr(state.usage, "to_dict") else {}
    return {
        "status": state.status,
        "result": state.result,
        "iterations": state.iterations,
        "max_iterations": state.max_iterations,
        "tool_calls": state.tool_calls,
        "final_output": state.final_output,
        "guardrail_triggered": state.guardrail_triggered,
        "trajectory_file": state.trajectory_file,
        "session_file": state.session_file,
        "usage": usage,
    }


def print_agent_output(state: AgentState, fmt: OutputFormat) -> None:
    if fmt == OutputFormat.TEXT:
        if state.result:
            print(state.result)
        return
    if fmt == OutputFormat.JSON:
        print(json.dumps(serialize_agent_state(state), ensure_ascii=False))
        return
    # stream-json final envelope
    print(json.dumps({"type": "result", **serialize_agent_state(state)}, ensure_ascii=False), flush=True)


def make_stream_json_emitter() -> OnEvent:
    """NDJSON event emitter for --output-format stream-json."""

    def emit(kind: EventKind, data: dict[str, Any]) -> None:
        print(json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)

    return emit


def chain_on_event(primary: OnEvent | None, secondary: OnEvent | None) -> OnEvent:
    def emit(kind: EventKind, data: dict[str, Any]) -> None:
        if primary:
            primary(kind, data)
        if secondary:
            secondary(kind, data)

    return emit
