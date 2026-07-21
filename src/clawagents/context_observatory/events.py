"""Typed event dataclasses for context management observability.

Each event captures a specific moment in the agent loop's context pipeline:
LLM calls, compaction, content crush, output trim, and budget snapshots.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageSnapshot:
    """Lightweight snapshot of a single message in the context window."""

    role: str
    content_preview: str  # first N chars for display
    content_length: int  # full char count
    token_count: int
    has_tool_calls: bool = False
    tool_call_id: str | None = None
    # Optional full content for deep inspection (populated when available)
    full_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "content_preview": self.content_preview,
            "content_length": self.content_length,
            "token_count": self.token_count,
        }
        if self.has_tool_calls:
            d["has_tool_calls"] = True
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.full_content is not None:
            d["full_content"] = self.full_content
        return d


@dataclass
class ToolCallSnapshot:
    """Snapshot of a single tool call issued by the LLM in its response."""

    call_id: str = ""
    tool_name: str = ""
    args_preview: str = ""      # first N chars of serialized args
    args_length: int = 0        # full serialized args length
    success: bool | None = None  # None = not yet completed
    output_preview: str = ""
    output_length: int = 0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "args_preview": self.args_preview,
            "args_length": self.args_length,
        }
        if self.success is not None:
            d["success"] = self.success
        if self.output_preview:
            d["output_preview"] = self.output_preview
        if self.output_length:
            d["output_length"] = self.output_length
        if self.duration_ms:
            d["duration_ms"] = self.duration_ms
        return d


@dataclass
class ContextEvent:
    """Base event — every event has a turn index, timestamp, and kind."""

    turn: int
    kind: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "kind": self.kind,
            "timestamp": self.timestamp,
        }


@dataclass
class LLMCallEvent(ContextEvent):
    """Snapshot of the full message list sent to the LLM at each turn.

    ``system_prompt_breakdown`` maps each detected memory component
    (core_memory, facts, rules, repo_map, …) to its token count.
    """

    kind: str = "llm_call"
    model: str = ""
    messages: list[MessageSnapshot] = field(default_factory=list)
    system_prompt_breakdown: dict[str, int] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_input_tokens: int = 0
    context_window: int = 0
    utilization_pct: float = 0.0
    # Per-role token totals for budget analysis
    tokens_by_role: dict[str, int] = field(default_factory=dict)
    # v2: extended fields for richer per-turn data
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    tool_calls_made: list[ToolCallSnapshot] = field(default_factory=list)
    response_text_preview: str = ""
    response_text_length: int = 0
    # Cumulative stats up to and including this turn
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    cumulative_cost_usd: float = 0.0
    # Label for multi-call turns, e.g. " (Call 1/2)"
    call_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "system_prompt_breakdown": self.system_prompt_breakdown,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "context_window": self.context_window,
            "utilization_pct": self.utilization_pct,
            "tokens_by_role": self.tokens_by_role,
            "cache_creation_tokens": self.cache_creation_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "tool_calls_made": [t.to_dict() for t in self.tool_calls_made],
            "response_text_preview": self.response_text_preview,
            "response_text_length": self.response_text_length,
            "cumulative_input_tokens": self.cumulative_input_tokens,
            "cumulative_output_tokens": self.cumulative_output_tokens,
            "cumulative_cost_usd": self.cumulative_cost_usd,
        })
        return d


@dataclass
class CompactionEvent(ContextEvent):
    """Fired when context compaction starts or finishes."""

    kind: str = "compaction"
    phase: str = ""  # "start" | "end"
    tokens_before: int = 0
    tokens_after: int = 0
    messages_before: int = 0
    messages_after: int = 0
    messages_dropped: int = 0
    savings_pct: float = 0.0
    budget: int = 0
    summary_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "phase": self.phase,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "messages_dropped": self.messages_dropped,
            "savings_pct": self.savings_pct,
            "budget": self.budget,
            "summary_preview": self.summary_preview,
        })
        return d


@dataclass
class CrushEvent(ContextEvent):
    """Fired when a tool output is crushed by content_crush."""

    kind: str = "crush"
    tool_name: str = ""
    content_kind: str = ""  # json/search/log/code/html/diff/test/prose
    original_chars: int = 0
    crushed_chars: int = 0
    saved_chars: int = 0
    original_tokens: int = 0
    crushed_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "tool_name": self.tool_name,
            "content_kind": self.content_kind,
            "original_chars": self.original_chars,
            "crushed_chars": self.crushed_chars,
            "saved_chars": self.saved_chars,
            "original_tokens": self.original_tokens,
            "crushed_tokens": self.crushed_tokens,
        })
        return d


@dataclass
class TrimEvent(ContextEvent):
    """Fired when a verbose assistant/user message is trimmed."""

    kind: str = "trim"
    role: str = ""
    original_chars: int = 0
    trimmed_chars: int = 0
    saved_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "role": self.role,
            "original_chars": self.original_chars,
            "trimmed_chars": self.trimmed_chars,
            "saved_chars": self.saved_chars,
        })
        return d


@dataclass
class BudgetSnapshot(ContextEvent):
    """Snapshot of per-role token budget allocation vs actual usage."""

    kind: str = "budget"
    system_tokens: int = 0
    tool_tokens: int = 0
    user_assistant_tokens: int = 0
    image_tokens: int = 0
    budget_limits: dict[str, int] = field(default_factory=dict)
    actual_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "system_tokens": self.system_tokens,
            "tool_tokens": self.tool_tokens,
            "user_assistant_tokens": self.user_assistant_tokens,
            "image_tokens": self.image_tokens,
            "budget_limits": self.budget_limits,
            "actual_usage": self.actual_usage,
        })
        return d


# Union type for type checking
AnyContextEvent = LLMCallEvent | CompactionEvent | CrushEvent | TrimEvent | BudgetSnapshot
