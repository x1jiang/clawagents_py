"""Bridge between a ClawAgents agent loop and ACP session updates.

:class:`AgentSession` accepts events emitted by ClawAgents' agent loop
(text deltas, reasoning fragments, tool starts/completions) and forwards
them to a sink as ACP-shaped updates. The sink is decoupled from the
JSON-RPC server so unit tests can exercise the translation layer with
just an in-memory list.

The shape of the events accepted here intentionally mirrors what
``clawagents.graph.agent_loop.OnEvent`` already emits, so wiring a
ClawAgents agent into ACP is just::

    sink: list = []
    sess = AgentSession(session_id="s1", sink=sink.append)

    def on_event(kind, payload):
        sess.dispatch(kind, payload or {})

    agent_state.on_event = on_event
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, List, Mapping, Optional, Union
from collections import deque

from clawagents.acp.messages import (
    AgentMessageChunk,
    AgentThoughtChunk,
    PermissionDecision,
    PermissionRequest,
    SessionUpdate,
    StopReason,
    ToolCallComplete,
    ToolCallStart,
    encode_update,
)


# A sink may be sync or async. Sync is used in tests; async in real
# servers (``await conn.session_update(...)``).
SessionEventSink = Union[
    Callable[[Dict[str, Any]], None],
    Callable[[Dict[str, Any]], Awaitable[None]],
]

# Permission requester — async in real servers.
PermissionRequester = Callable[[PermissionRequest], Awaitable[PermissionDecision]]


@dataclass
class AgentSession:
    """Wraps an ACP session and translates agent events into updates.

    Parameters
    ----------
    session_id:
        Identifier supplied by the IDE in ``session/new`` / ``session/prompt``.
    sink:
        Callable invoked for every encoded :class:`SessionUpdate` dict.
        May be sync (tests) or async (real server). For async sinks,
        callers should ``await session.adispatch(...)`` instead of
        :meth:`dispatch`.
    permission_requester:
        Optional async callable used by :meth:`request_permission`.
    """

    session_id: str
    sink: Optional[SessionEventSink] = None
    permission_requester: Optional[PermissionRequester] = None

    _tool_ids_by_name: Dict[str, Deque[str]] = field(default_factory=dict)
    _tool_args_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _emitted: List[Dict[str, Any]] = field(default_factory=list)
    _stop_reason: Optional[StopReason] = None

    @property
    def emitted(self) -> List[Dict[str, Any]]:
        """In-order log of every wire-form update this session produced.

        Useful for assertions in tests; the live server clears this with
        :meth:`reset_emitted` between prompt cycles.
        """
        return list(self._emitted)

    def reset_emitted(self) -> None:
        self._emitted = []

    @property
    def stop_reason(self) -> Optional[StopReason]:
        return self._stop_reason

    # ── Event dispatch ────────────────────────────────────────────

    def dispatch(self, kind: str, payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Translate one agent-loop event into zero or more ACP updates.

        Returns the list of wire-form payloads emitted, so callers that
        need to plumb them through their own transport can do so without
        relying on :attr:`sink`.
        """

        updates = list(self._translate(kind, payload))
        for raw in updates:
            self._emitted.append(raw)
            self._emit_sync(raw)
        return updates

    async def adispatch(
        self, kind: str, payload: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        """Async variant of :meth:`dispatch` for real ACP servers."""

        updates = list(self._translate(kind, payload))
        for raw in updates:
            self._emitted.append(raw)
            await self._emit_async(raw)
        return updates

    # ── Permission gate ───────────────────────────────────────────

    async def request_permission(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
    ) -> PermissionDecision:
        """Ask the IDE whether a particular tool call may proceed.

        If no :attr:`permission_requester` is configured the call is
        allowed by default — matching ``trust=True`` mode.
        """

        if self.permission_requester is None:
            return PermissionDecision(
                allowed=True, rationale="no requester configured"
            )
        req = PermissionRequest(
            tool_call_id=_make_tool_call_id(),
            name=name,
            arguments=dict(arguments or {}),
            description=description,
        )
        return await self.permission_requester(req)

    # ── Translation core ──────────────────────────────────────────

    def _translate(
        self, kind: str, payload: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        # Normalise: agent loop events use dotted names like
        # "tool.started" / "llm.delta"; we accept either form.
        k = kind.replace(".", "_")

        if k == "llm_delta" or k == "message_delta" or k == "message_text":
            text = _coerce_text(payload.get("text") or payload.get("delta"))
            if text:
                out.append(encode_update(AgentMessageChunk(text=text)))
            return out

        if k in {"reasoning", "reasoning_delta", "thought", "thinking"}:
            text = _coerce_text(payload.get("text") or payload.get("delta"))
            if text:
                out.append(encode_update(AgentThoughtChunk(text=text)))
            return out

        if k in {"tool_started", "tool_start"}:
            name = str(payload.get("name") or payload.get("tool") or "tool")
            args_raw = payload.get("arguments") or payload.get("args") or {}
            args = _coerce_args(args_raw)
            tc = ToolCallStart.new(name, arguments=args)
            queue = self._tool_ids_by_name.setdefault(name, deque())
            queue.append(tc.tool_call_id)
            self._tool_args_by_id[tc.tool_call_id] = args
            out.append(encode_update(tc))
            return out

        if k in {"tool_completed", "tool_complete", "tool_finished", "tool_end"}:
            name = str(payload.get("name") or payload.get("tool") or "tool")
            existing = self._tool_ids_by_name.get(name)
            tc_id = existing.popleft() if existing else _make_tool_call_id()
            args = self._tool_args_by_id.pop(tc_id, {})
            error = payload.get("error")
            output = payload.get("output") or payload.get("result")
            update = ToolCallComplete(
                tool_call_id=tc_id,
                name=name,
                output=output if not error else None,
                error=str(error) if error else None,
                arguments=args,
            )
            out.append(encode_update(update))
            return out

        if k in {"run_finished", "agent_finished", "stop"}:
            self._stop_reason = _coerce_stop(payload.get("reason"))
            return out

        if k in {"run_error", "agent_error", "error"}:
            self._stop_reason = StopReason.ERROR
            return out

        return out

    # ── Sink emission ─────────────────────────────────────────────

    def _emit_sync(self, raw: Dict[str, Any]) -> None:
        if self.sink is None:
            return
        result = self.sink(raw)
        if hasattr(result, "__await__"):
            # caller passed an async sink to a sync dispatch — tell them.
            raise TypeError(
                "AgentSession.dispatch() received an async sink. "
                "Use AgentSession.adispatch() instead."
            )

    async def _emit_async(self, raw: Dict[str, Any]) -> None:
        if self.sink is None:
            return
        result = self.sink(raw)
        if result is not None and hasattr(result, "__await__"):
            await result


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_args(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (json.JSONDecodeError, ValueError):
            return {"raw": value}
    return {}


def _coerce_stop(value: Any) -> StopReason:
    if isinstance(value, StopReason):
        return value
    if isinstance(value, str):
        try:
            return StopReason(value.lower())
        except ValueError:
            pass
    return StopReason.END_TURN


def _make_tool_call_id() -> str:
    import uuid

    return f"tc_{uuid.uuid4().hex[:12]}"
