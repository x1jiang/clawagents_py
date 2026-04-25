"""Capture live agent runs as :class:`Trajectory` objects.

:class:`RLRecorder` plugs into ``agent.on_event`` and assembles a
training-ready trajectory as the agent emits its event stream. It is
intentionally separate from :class:`clawagents.trajectory.TrajectoryRecorder`
(which focuses on observability/scoring metadata): :class:`RLRecorder`
captures the *prompt → assistant → tool* sequence in a shape that
TRL/Atropos/SLIME can consume directly.

Usage::

    rec = RLRecorder()
    agent = create_claw_agent(name="claw")
    agent.on_event = rec.observe
    answer = agent.run("solve x^2 = 16")
    traj = rec.finalise(prompt="solve x^2 = 16", final=answer)

The recorder only mutates state via the public ``observe`` method, so
the same instance can record a single run end-to-end without subclass-
ing or threading lock plumbing through it.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from clawagents.rl.trajectory import ToolCall, Trajectory


logger = logging.getLogger(__name__)


@dataclass
class RecorderConfig:
    """Knobs for what gets captured during a run.

    Defaults are tuned for "small enough to ship, big enough to train":
    we capture full assistant content but truncate large tool outputs
    (``max_tool_result_chars``) so a single trajectory stays small.
    """

    max_tool_result_chars: int = 8_000
    capture_thinking: bool = False
    capture_system_prompt: bool = True
    redact_tool_args: bool = False  # opt-in: useful for OSS dataset releases


@dataclass
class _PendingToolCall:
    """Bookkeeping for a tool_call event that hasn't paired with a result yet."""

    id: str
    name: str
    arguments: dict[str, Any]


class RLRecorder:
    """Streams agent events into a :class:`Trajectory`.

    The recorder is *additive*: every call to :meth:`observe` appends
    to the in-progress trajectory. Call :meth:`finalise` when the agent
    is done to attach the final assistant message and metadata.
    """

    def __init__(
        self,
        task: str = "",
        model: str = "",
        config: RecorderConfig | None = None,
    ) -> None:
        self.config = config or RecorderConfig()
        self.trajectory = Trajectory(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            model=model,
        )
        self._pending_calls: dict[str, _PendingToolCall] = {}
        self._current_assistant: str | None = None
        self._current_tool_calls: list[ToolCall] = []
        # Tool messages are emitted *after* the assistant turn that issued them
        # (the canonical OpenAI ChatML order). We stash them here until we
        # flush the assistant block, then drain them as tool steps.
        self._pending_tool_messages: list[tuple[str, str, str, bool]] = []
        self._finalised = False

    # ── public API ──────────────────────────────────────────────────

    def observe(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        """Event handler — bind this to ``agent.on_event``.

        Mirrors the :data:`clawagents.graph.agent_loop.OnEvent` contract.
        Unknown event kinds are silently ignored.
        """
        if self._finalised:
            return
        data = dict(payload or {})
        try:
            handler = self._dispatch.get(kind)
            if handler is not None:
                handler(self, data)
        except Exception:  # pragma: no cover - never let RL break the agent
            logger.debug("RLRecorder: failed handling %s event", kind, exc_info=True)

    def add_user(self, content: str, **meta: Any) -> None:
        self.trajectory.add_user(content, **meta)

    def add_system(self, content: str, **meta: Any) -> None:
        if self.config.capture_system_prompt:
            self.trajectory.add_system(content, **meta)

    def finalise(
        self,
        prompt: str | None = None,
        final: str | None = None,
        reward: float | None = None,
        **metadata: Any,
    ) -> Trajectory:
        """Flush any pending assistant turn and return the trajectory.

        ``prompt``, if given, is prepended as a ``user`` step *only* if
        the trajectory doesn't already have a user/system step. ``final``
        is appended as a final assistant message if non-empty.
        """
        if self._finalised:
            return self.trajectory

        if prompt and not any(
            s.role in ("user", "system") for s in self.trajectory.steps
        ):
            self.trajectory.add_user(prompt)

        self._flush_assistant()

        if final and final != (self.trajectory.final_assistant.content if self.trajectory.final_assistant else None):
            self.trajectory.add_assistant(final)

        if reward is not None:
            self.trajectory.reward = float(reward)
        if metadata:
            self.trajectory.metadata.update(metadata)

        self._finalised = True
        return self.trajectory

    # ── event handlers ──────────────────────────────────────────────

    def _on_assistant_message(self, data: dict[str, Any]) -> None:
        text = str(data.get("content") or data.get("message") or "")
        if not text:
            return
        # If we already have results from a prior tool round, the prior
        # assistant turn is complete — flush it before starting a new one.
        if self._pending_tool_messages:
            self._flush_assistant()
        if self._current_assistant is None:
            self._current_assistant = text
        else:
            self._current_assistant = (self._current_assistant + "\n" + text).strip()

    def _on_assistant_delta(self, data: dict[str, Any]) -> None:
        delta = str(data.get("delta") or data.get("content") or "")
        if not delta:
            return
        if self._pending_tool_messages:
            self._flush_assistant()
        if self._current_assistant is None:
            self._current_assistant = delta
        else:
            self._current_assistant += delta

    def _on_tool_call(self, data: dict[str, Any]) -> None:
        # On many agents `tool_call` and `tool_started` are emitted together.
        # We dedupe by id; whichever arrives first wins.
        call_id = str(data.get("id") or data.get("call_id") or uuid.uuid4().hex[:8])
        name = str(data.get("name") or data.get("tool") or "")
        if not name:
            return
        if call_id in self._pending_calls:
            return
        args = data.get("arguments") or data.get("args") or {}
        if not isinstance(args, dict):
            args = {"_raw": args}
        if self.config.redact_tool_args:
            args = {"_redacted": True}
        self._pending_calls[call_id] = _PendingToolCall(
            id=call_id, name=name, arguments=dict(args)
        )

    def _on_tool_result(self, data: dict[str, Any]) -> None:
        call_id = str(data.get("id") or data.get("call_id") or "")
        tool_name = str(data.get("name") or data.get("tool") or "")
        result = data.get("result") or data.get("output") or ""
        if not isinstance(result, str):
            try:
                import json as _json

                result = _json.dumps(result, default=str)
            except Exception:
                result = str(result)
        if len(result) > self.config.max_tool_result_chars:
            result = result[: self.config.max_tool_result_chars] + "…"
        success = bool(data.get("success", True))
        error = data.get("error")
        duration_ms = float(data.get("duration_ms", 0.0))

        pending = self._pending_calls.pop(call_id, None)
        if pending is None and tool_name:
            # Re-pair by name when the event stream omitted call ids.
            for pid, p in list(self._pending_calls.items()):
                if p.name == tool_name:
                    pending = self._pending_calls.pop(pid, None)
                    call_id = call_id or pid
                    break

        if pending is None:
            pending = _PendingToolCall(
                id=call_id or uuid.uuid4().hex[:8],
                name=tool_name or "unknown",
                arguments={},
            )

        self._current_tool_calls.append(
            ToolCall(
                id=pending.id,
                name=pending.name,
                arguments=pending.arguments,
                result=result,
                success=success,
                error=str(error) if error else None,
                duration_ms=duration_ms,
            )
        )
        # Tool messages must follow the assistant turn that issued them — we
        # stash them here and drain in :meth:`_flush_assistant`.
        self._pending_tool_messages.append(
            (pending.id, pending.name, result, success)
        )

    def _on_turn_started(self, data: dict[str, Any]) -> None:
        self._flush_assistant()

    def _on_agent_done(self, data: dict[str, Any]) -> None:
        self._flush_assistant()

    def _on_final_content(self, data: dict[str, Any]) -> None:
        text = str(data.get("content") or data.get("text") or "")
        if not text:
            return
        if self._current_assistant is None:
            self._current_assistant = text
        else:
            self._current_assistant = (self._current_assistant + "\n" + text).strip()

    def _on_final_output(self, data: dict[str, Any]) -> None:
        out = data.get("output")
        text = ""
        if isinstance(out, str):
            text = out
        elif isinstance(out, dict):
            text = str(out.get("content") or out.get("text") or "")
        if text:
            self._on_final_content({"content": text})

    # ── internals ───────────────────────────────────────────────────

    def _flush_assistant(self) -> None:
        if (
            self._current_assistant is None
            and not self._current_tool_calls
            and not self._pending_tool_messages
        ):
            return
        if self._current_assistant is not None or self._current_tool_calls:
            self.trajectory.add_assistant(
                content=self._current_assistant or "",
                tool_calls=list(self._current_tool_calls),
            )
        for tc_id, name, result, success in self._pending_tool_messages:
            self.trajectory.add_tool(
                result=result,
                tool_call_id=tc_id,
                name=name,
                success=success,
            )
        self._current_assistant = None
        self._current_tool_calls = []
        self._pending_tool_messages = []

    _dispatch: dict[str, Any] = {}


# Wire the dispatch table after the class is fully defined.
RLRecorder._dispatch = {
    "assistant_message": RLRecorder._on_assistant_message,
    "assistant_delta": RLRecorder._on_assistant_delta,
    "tool_call": RLRecorder._on_tool_call,
    "tool_started": RLRecorder._on_tool_call,
    "tool_result": RLRecorder._on_tool_result,
    "turn_started": RLRecorder._on_turn_started,
    "agent_done": RLRecorder._on_agent_done,
    "final_content": RLRecorder._on_final_content,
    "final_output": RLRecorder._on_final_output,
}
