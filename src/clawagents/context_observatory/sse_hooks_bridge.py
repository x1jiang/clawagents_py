"""Bridge from SSE stream events to the EventStore.

When the agent runs inside the sidecar (not in-process), the original
``ContextObserverHooks`` (which rely on ``RunHooks`` callbacks) cannot work.
This module translates the ``HostToWebview``-equivalent dicts produced by
``SseClient.stream_chat()`` into ``EventStore`` entries so that the
Context Inspector, Token Analytics, and Event Timeline panels continue
to function.

Usage::

    bridge = SseEventBridge(store, context_window=128_000)
    async for event in client.stream_chat("hello"):
        bridge.ingest(event)
        handle_ui(event)  # update Streamlit chat panel
"""

from __future__ import annotations

import logging
import time
from typing import Any

from clawagents.context_observatory.events import (
    BudgetSnapshot,
    CompactionEvent,
    LLMCallEvent,
    MessageSnapshot,
    ToolCallSnapshot,
)
from clawagents.context_observatory.store import EventStore, _deserialize_event

logger = logging.getLogger(__name__)

_SYSTEM_FULL_CONTENT_LIMIT = 50_000
_MSG_FULL_CONTENT_LIMIT = 5_000


class SseEventBridge:
    """Ingest SSE events and record corresponding EventStore entries."""

    def __init__(
        self,
        store: EventStore,
        *,
        context_window: int = 128_000,
        model: str = "",
        user_text: str = "",
    ) -> None:
        self.store = store
        self.context_window = context_window
        self.model = model
        self._turn = store.max_turn
        self._user_text = user_text
        self._current_assistant_text = ""
        self._current_tools: list[dict[str, Any]] = []
        self._turn_started = False
        self._current_tool_calls: list[ToolCallSnapshot] = []
        self._cumulative_input = 0
        self._cumulative_output = 0

        # Buffer llm_context snapshots (one per LLM call within the turn).
        # The first matches call 0, the second call 1, etc.
        self._context_snapshots: list[dict[str, Any]] = []

        # Count streaming usage events (used only as fallback when done
        # event has no per_request data).
        self._usage_events: list[dict[str, Any]] = []
        self._received_observatory_events = False

    def ingest(self, event: dict[str, Any]) -> None:
        """Process a single SSE event and update the EventStore."""
        event_type = event.get("type", "")

        try:
            handler = getattr(self, f"_on_{event_type}", None)
            if handler:
                handler(event)
        except Exception:
            logger.debug("SseEventBridge.ingest failed for %s", event_type, exc_info=True)

    # -- Event handlers ----------------------------------------------------

    def _on_status(self, event: dict[str, Any]) -> None:
        msg = event.get("message", "")
        if msg == "Running…":
            self._turn += 1
            self._turn_started = True
            self._current_assistant_text = ""
            self._current_tools = []
            self._current_tool_calls = []
            self._context_snapshots = []
            self._usage_events = []

    def _on_assistant_delta(self, event: dict[str, Any]) -> None:
        self._current_assistant_text += event.get("delta", "")

    def _on_assistant_message(self, event: dict[str, Any]) -> None:
        text = event.get("text", "")
        if text:
            self._current_assistant_text = text

    def _on_tool_started(self, event: dict[str, Any]) -> None:
        tool_id = event.get("id", "")
        tool_name = event.get("name", "tool")
        args = event.get("args")
        args_str = ""
        if isinstance(args, dict):
            import json
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_str = str(args)
        elif isinstance(args, str):
            args_str = args

        self._current_tools.append({
            "id": tool_id,
            "name": tool_name,
            "started_at": time.time(),
        })
        self._current_tool_calls.append(ToolCallSnapshot(
            call_id=tool_id,
            tool_name=tool_name,
            args_preview=args_str[:2000],
            args_length=len(args_str),
        ))

    def _on_tool_completed(self, event: dict[str, Any]) -> None:
        tool_id = event.get("id", "")
        success = event.get("success", True)
        output = event.get("output", "")

        for tool in self._current_tools:
            if tool["id"] == tool_id:
                tool["completed"] = True
                tool["success"] = success
                tool["output"] = output
                tool["output_len"] = len(output)
                tool["completed_at"] = time.time()
                break

        for tc in self._current_tool_calls:
            if tc.call_id == tool_id:
                tc.success = success
                tc.output_preview = output[:2000]
                tc.output_length = len(output)
                started = next(
                    (t.get("started_at", 0) for t in self._current_tools if t["id"] == tool_id),
                    0,
                )
                if started:
                    tc.duration_ms = int((time.time() - started) * 1000)
                break

    def _on_llm_context(self, event: dict[str, Any]) -> None:
        """Buffer full context snapshot — will be consumed by _on_done."""
        self._context_snapshots.append(event)

    def _on_observatory_event(self, event: dict[str, Any]) -> None:
        raw = event.get("event")
        if not isinstance(raw, dict):
            return
        observed = _deserialize_event(raw)
        if observed is None:
            return
        self._received_observatory_events = True
        if observed.kind == "llm_call":
            for index, existing in enumerate(self.store._events):
                if existing.kind == "llm_call" and existing.turn == observed.turn:
                    self.store._events[index] = observed
                    return
        self.store.append(observed)

    def _on_usage(self, event: dict[str, Any]) -> None:
        """Buffer streaming usage events — will be consumed by _on_done."""
        self._usage_events.append(event)

    def _on_compact_progress(self, event: dict[str, Any]) -> None:
        phase = event.get("phase", "")
        turn = max(self._turn, 1)
        if phase == "start":
            self.store.append(CompactionEvent(
                turn=turn,
                phase="start",
                tokens_before=0,
                messages_before=0,
            ))
        elif phase == "end":
            self.store.append(CompactionEvent(
                turn=turn,
                phase="end",
                tokens_before=0,
                tokens_after=0,
                messages_before=0,
                messages_after=0,
                messages_dropped=0,
                savings_pct=0.0,
                summary_preview=event.get("message", ""),
            ))

    def _on_done(self, event: dict[str, Any]) -> None:
        """Final event — create LLMCallEvents from the done payload.

        The done event includes ``usage.per_request`` which is the
        authoritative list of LLM API calls for this turn.  We use that
        to emit one ``LLMCallEvent`` per API call, labelled
        "Call 1/N, Call 2/N, …".

        If ``per_request`` is missing, fall back to the buffered
        streaming ``usage`` events.
        """
        turn = max(self._turn, 1)

        if self._received_observatory_events:
            self.store.set_session_meta(
                model=self.model,
                context_window=self.context_window,
                completed_at=time.time(),
                iterations=event.get("iterations"),
                status=event.get("status"),
            )
            return

        usage = event.get("usage") or {}
        per_request = usage.get("per_request") or []

        if per_request:
            # Authoritative path: use per_request from the done event
            for i, pr in enumerate(per_request):
                self._emit_llm_call(
                    turn=turn,
                    call_index=i,
                    total_calls=len(per_request),
                    input_tokens=pr.get("input_tokens", 0),
                    output_tokens=pr.get("output_tokens", 0),
                    cached_tokens=pr.get("cached_input_tokens", 0),
                    cache_creation=pr.get("cache_creation_tokens", 0),
                    reasoning=pr.get("reasoning_tokens", 0),
                    model_name=pr.get("model", ""),
                )
        elif self._usage_events:
            # Fallback: use the buffered streaming usage events
            for i, ue in enumerate(self._usage_events):
                self._emit_llm_call(
                    turn=turn,
                    call_index=i,
                    total_calls=len(self._usage_events),
                    input_tokens=_num(ue.get("promptTokens") or ue.get("input_tokens")),
                    output_tokens=_num(ue.get("completionTokens") or ue.get("output_tokens")),
                    cached_tokens=_num(ue.get("cachedInputTokens") or ue.get("cached_input_tokens")),
                    cache_creation=_num(ue.get("cacheCreationTokens") or ue.get("cache_creation_tokens")),
                    reasoning=_num(ue.get("reasoningTokens") or ue.get("reasoning_tokens")),
                    model_name=ue.get("model", ""),
                )

        self.store.set_session_meta(
            model=self.model,
            context_window=self.context_window,
            completed_at=time.time(),
            iterations=event.get("iterations"),
            status=event.get("status"),
            session_cost_usd=event.get("sessionCostUsd"),
            run_cost_usd=event.get("runCostUsd"),
        )

        # Auto-save session to history
        chat_id = event.get("chatId") or event.get("chat_id")
        try:
            self.store.auto_save(chat_id=chat_id)
        except Exception:
            logger.debug("Auto-save failed", exc_info=True)

    def _emit_llm_call(
        self,
        *,
        turn: int,
        call_index: int,
        total_calls: int,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        cache_creation: int,
        reasoning: int,
        model_name: str,
    ) -> None:
        """Create and store one LLMCallEvent for a single API call."""
        self._cumulative_input += input_tokens
        self._cumulative_output += output_tokens

        utilization = (
            (input_tokens / self.context_window * 100.0)
            if self.context_window > 0 and input_tokens
            else 0.0
        )

        # Match context snapshot by call_index
        messages: list[MessageSnapshot] = []
        tokens_by_role: dict[str, int] = {}
        system_breakdown: dict[str, int] = {}

        if call_index < len(self._context_snapshots):
            ctx = self._context_snapshots[call_index]
            for m in ctx.get("messages", []):
                preview = m.get("content_preview", "")
                full = m.get("full_content")
                messages.append(MessageSnapshot(
                    role=m.get("role", "unknown"),
                    content_preview=preview,
                    content_length=m.get("content_length", len(preview)),
                    token_count=m.get("token_count", 0),
                    has_tool_calls=m.get("has_tool_calls", False),
                    tool_call_id=m.get("tool_call_id"),
                    full_content=full,
                ))
            system_breakdown = ctx.get("system_prompt_breakdown", {})
            tokens_by_role = ctx.get("tokens_by_role", {})

        # Only attach tool calls to the FIRST call (which decided to
        # call tools).  The second call processes tool results.
        tool_calls = list(self._current_tool_calls) if call_index == 0 else []

        # Build a label like "Call 1/2" for multi-call turns
        call_label = (
            f" (Call {call_index + 1}/{total_calls})"
            if total_calls > 1
            else ""
        )

        self.store.append(LLMCallEvent(
            turn=turn,
            model=model_name or self.model,
            messages=messages,
            system_prompt_breakdown=system_breakdown,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            cached_input_tokens=cached_tokens,
            cache_creation_tokens=cache_creation,
            reasoning_tokens=reasoning,
            context_window=self.context_window,
            utilization_pct=round(utilization, 2),
            tokens_by_role=tokens_by_role or {"assistant": output_tokens},
            tool_calls_made=tool_calls,
            response_text_preview=(
                self._current_assistant_text[:2000]
                if call_index == total_calls - 1
                else ""
            ),
            response_text_length=(
                len(self._current_assistant_text)
                if call_index == total_calls - 1
                else 0
            ),
            cumulative_input_tokens=self._cumulative_input,
            cumulative_output_tokens=self._cumulative_output,
            call_label=call_label,
        ))

        # Budget snapshot
        self.store.append(BudgetSnapshot(
            turn=turn,
            system_tokens=0,
            tool_tokens=0,
            user_assistant_tokens=input_tokens,
            image_tokens=0,
            budget_limits={
                "system": int(self.context_window * 0.15),
                "tools": int(self.context_window * 0.25),
                "user_assistant": int(self.context_window * 0.50),
                "images": int(self.context_window * 0.10),
            },
            actual_usage={"user_assistant": input_tokens},
        ))


def _num(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0
