"""ACP wire-format message dataclasses.

These mirror the relevant subset of Zed's Agent Client Protocol. They are
intentionally hermetic — no dependency on the official ``acp`` package —
so library users can construct, log, and round-trip ACP frames in tests
without installing any optional extras.

The schema covers the messages that an agent server emits or accepts:

* :class:`PromptRequest`     — incoming ``session/prompt`` request
* :class:`SessionUpdate`     — outgoing ``session/update`` notification union
* :class:`AgentMessageChunk` — assistant text fragment
* :class:`AgentThoughtChunk` — assistant reasoning fragment
* :class:`ToolCallStart`     — tool call invocation begin
* :class:`ToolCallComplete`  — tool call result/finish
* :class:`PermissionRequest` — outgoing ``request_permission`` to client
* :class:`PermissionDecision`— incoming reply
* :class:`StopReason`        — terminal status of a prompt cycle

Use :func:`encode_update` / :func:`decode_update` for the
``session/update`` discriminated union.
"""

from __future__ import annotations

import enum
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union


class StopReason(str, enum.Enum):
    """Terminal status reported when a prompt cycle ends."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_TURN_REQUESTS = "max_turn_requests"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"
    ERROR = "error"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ──────────────────────────────────────────────────────────────────────
# Inbound: prompt request
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PromptRequest:
    """Incoming ``session/prompt`` request from the IDE.

    The ACP protocol carries content as a list of typed blocks
    (text/image/resource), but the most common case is a single text
    block — :attr:`text` is the joined plain-text view. The original
    blocks are preserved in :attr:`blocks` for callers that need them.
    """

    session_id: str
    text: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromptRequest":
        session_id = str(payload.get("sessionId") or payload.get("session_id") or "")
        prompt = payload.get("prompt") or []
        if isinstance(prompt, dict):
            prompt = [prompt]
        if not isinstance(prompt, list):
            prompt = []
        text_parts: List[str] = []
        for block in prompt:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        return cls(
            session_id=session_id,
            text="\n".join(text_parts),
            blocks=list(prompt),
            raw=dict(payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "prompt": list(self.blocks)
            if self.blocks
            else [{"type": "text", "text": self.text}],
        }


# ──────────────────────────────────────────────────────────────────────
# Outbound: session/update variants
# ──────────────────────────────────────────────────────────────────────


@dataclass
class AgentMessageChunk:
    """Streamed assistant message text fragment."""

    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": self.text},
        }


@dataclass
class AgentThoughtChunk:
    """Streamed assistant reasoning text fragment.

    Mapped to ACP's ``agent_thought_chunk`` update type so IDEs can show
    chain-of-thought separately from user-visible message text.
    """

    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": self.text},
        }


@dataclass
class ToolCallStart:
    """Notification that the agent has begun invoking a tool."""

    tool_call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None

    @classmethod
    def new(
        cls,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> "ToolCallStart":
        return cls(
            tool_call_id=_new_id("tc"),
            name=name,
            arguments=dict(arguments or {}),
            label=label,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionUpdate": "tool_call",
            "toolCallId": self.tool_call_id,
            # ``title`` is REQUIRED by the ACP schema (spec-strict clients
            # reject frames without it); name/label are kept for local
            # round-trips and ignored by conforming peers.
            "title": self.label or self.name,
            "kind": _infer_tool_kind(self.name),
            "name": self.name,
            "label": self.label or self.name,
            "status": "in_progress",
            "rawInput": dict(self.arguments),
        }


@dataclass
class ToolCallComplete:
    """Notification that a previously-started tool call finished.

    ``output`` may be plain text or an arbitrary JSON-serialisable
    structure; ACP carries it back as a content block.
    """

    tool_call_id: str
    name: str
    output: Optional[Any] = None
    error: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None

    @property
    def status(self) -> str:
        return "failed" if self.error else "completed"

    def to_dict(self) -> Dict[str, Any]:
        # Spec ``ToolCallContent`` wraps each ContentBlock as
        # {"type": "content", "content": block}; bare text blocks fail
        # schema validation on spec-strict clients.
        content_blocks: List[Dict[str, Any]] = []
        if self.error:
            content_blocks.append(
                {"type": "content", "content": {"type": "text", "text": str(self.error)}}
            )
        elif self.output is not None:
            text = (
                self.output
                if isinstance(self.output, str)
                else _safe_dumps(self.output)
            )
            content_blocks.append(
                {"type": "content", "content": {"type": "text", "text": text}}
            )
        payload: Dict[str, Any] = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": self.tool_call_id,
            "status": self.status,
            "name": self.name,
        }
        if content_blocks:
            payload["content"] = content_blocks
        if self.arguments is not None:
            payload["rawInput"] = dict(self.arguments)
        return payload


SessionUpdate = Union[
    AgentMessageChunk, AgentThoughtChunk, ToolCallStart, ToolCallComplete
]


# ──────────────────────────────────────────────────────────────────────
# Bidirectional: permission request / decision
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PermissionRequest:
    """Server → client request to allow a tool call.

    Returned via :class:`PermissionDecision`; in ACP this is the
    ``session/request_permission`` call.
    """

    tool_call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toolCallId": self.tool_call_id,
            "name": self.name,
            "rawInput": dict(self.arguments),
            "description": self.description or self.name,
        }


@dataclass
class PermissionDecision:
    """Client → server response to a permission request."""

    allowed: bool
    rationale: Optional[str] = None
    one_time: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PermissionDecision":
        outcome = payload.get("outcome") or payload.get("decision") or "denied"
        if isinstance(outcome, dict):
            kind = outcome.get("kind") or outcome.get("type")
        else:
            kind = str(outcome)
        allowed = str(kind).lower() in {"allow", "allowed", "approve", "approved"}
        rationale = payload.get("rationale") or payload.get("reason")
        return cls(
            allowed=allowed,
            rationale=str(rationale) if rationale is not None else None,
            one_time=not bool(payload.get("remember")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": {"kind": "allow" if self.allowed else "denied"},
            "rationale": self.rationale,
        }


# ──────────────────────────────────────────────────────────────────────
# Codec helpers
# ──────────────────────────────────────────────────────────────────────


def encode_update(update: SessionUpdate) -> Dict[str, Any]:
    """Serialise any ``SessionUpdate`` variant to its wire form."""
    if hasattr(update, "to_dict"):
        return update.to_dict()
    raise TypeError(f"Unsupported update type: {type(update).__name__}")


def decode_update(payload: Mapping[str, Any]) -> SessionUpdate:
    """Reverse of :func:`encode_update`. Used by clients/tests.

    Returns the appropriate dataclass based on the ``sessionUpdate``
    discriminator. Raises :class:`ValueError` for unknown variants.
    """

    kind = payload.get("sessionUpdate")
    if kind == "agent_message_chunk":
        text = ((payload.get("content") or {}).get("text")) or ""
        return AgentMessageChunk(text=str(text))
    if kind == "agent_thought_chunk":
        text = ((payload.get("content") or {}).get("text")) or ""
        return AgentThoughtChunk(text=str(text))
    if kind == "tool_call":
        return ToolCallStart(
            tool_call_id=str(payload.get("toolCallId") or _new_id("tc")),
            # Wire frames from spec-strict peers carry only ``title``.
            name=str(payload.get("name") or payload.get("title") or ""),
            arguments=dict(payload.get("rawInput") or {}),
            label=payload.get("label") or payload.get("title"),
        )
    if kind == "tool_call_update":
        content = payload.get("content") or []
        text_out: Optional[str] = None
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                # Spec shape: {"type": "content", "content": ContentBlock};
                # also accept bare ContentBlocks from older frames.
                if block.get("type") == "content" and isinstance(
                    block.get("content"), dict
                ):
                    block = block["content"]
                if block.get("type") == "text":
                    text_out = str(block.get("text") or "")
                    break
        status = str(payload.get("status") or "completed").lower()
        return ToolCallComplete(
            tool_call_id=str(payload.get("toolCallId") or ""),
            name=str(payload.get("name") or ""),
            output=text_out if status != "failed" else None,
            error=text_out if status == "failed" else None,
            arguments=dict(payload.get("rawInput") or {}) or None,
        )
    raise ValueError(f"Unknown sessionUpdate variant: {kind!r}")


def _infer_tool_kind(name: str) -> str:
    """Map a clawagents tool name onto the closest ACP ``ToolKind``."""
    n = (name or "").lower()
    if re.search(r"(^|_)(read|cat|glob|ls|list|tree)($|_)", n):
        return "read"
    if re.search(r"(write|edit|patch|apply|create_file)", n):
        return "edit"
    if re.search(r"(delete|remove|rm)($|_)", n):
        return "delete"
    if re.search(r"(move|rename)", n):
        return "move"
    if re.search(r"(grep|search|find)", n):
        return "search"
    if re.search(r"(exec|bash|shell|run_command|command)", n):
        return "execute"
    if "think" in n:
        return "think"
    if re.search(r"(web|fetch|http|browse)", n):
        return "fetch"
    return "other"


def _safe_dumps(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)
