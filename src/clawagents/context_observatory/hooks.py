"""RunHooks-based instrumentation for context management observability.

``ContextObserverHooks`` is a ``RunHooks`` subclass that intercepts key
lifecycle events (LLM calls, compaction, tool results) and records typed
``ContextEvent`` entries into an ``EventStore``.

Usage::

    from clawagents.context_observatory import ContextObserverHooks, EventStore

    store = EventStore()
    observer = ContextObserverHooks(store, context_window=128_000)

    agent = ClawAgent(
        model="gpt-4o",
        hooks=observer,
        context_window=128_000,
    )
    await agent.invoke("Do something")

    # Inspect recorded events
    for event in store.events:
        print(event.to_dict())
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from clawagents.context_observatory.analyzer import (
    analyze_system_prompt,
    compute_role_tokens,
)
from clawagents.context_observatory.events import (
    BudgetSnapshot,
    CompactionEvent,
    CrushEvent,
    LLMCallEvent,
    MessageSnapshot,
    TrimEvent,
)
from clawagents.context_observatory.store import EventStore
from clawagents.lifecycle import RunHooks
from clawagents.run_context import RunContext
from clawagents.tokenizer import count_tokens
from clawagents.usage import RequestUsage

logger = logging.getLogger(__name__)

# Preview length for message content snapshots
_PREVIEW_CHARS = 500


class ContextObserverHooks(RunHooks):
    """Non-invasive instrumentation hooks for context management analysis.

    Observes the agent loop through the standard ``RunHooks`` surface:
    - ``on_llm_start``: snapshot messages, compute per-role and per-component tokens
    - ``on_llm_end``: record output tokens and usage
    - ``on_pre_compact``: record pre-compaction state
    - ``on_post_compact``: record post-compaction state and savings
    - ``on_tool_end``: probe for crush-eligible outputs

    All observation is side-effect-free — nothing modifies the agent's behavior.
    """

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        context_window: int = 1_000_000,
        model: str | None = None,
        event_sink: Callable[[Any], None] | None = None,
    ) -> None:
        self.store = store if store is not None else EventStore()
        self.context_window = context_window
        self.model = model
        self._event_sink = event_sink
        self._turn = self.store.max_turn
        # State for compaction tracking
        self._pre_compact_tokens = 0
        self._pre_compact_msg_count = 0
        # State for trim detection (messages before vs after)
        self._last_messages_snapshot: list[tuple[str, int]] | None = None
        # Cumulative token accumulators
        self._cumulative_input = 0
        self._cumulative_output = 0

    def _record(self, event: Any) -> None:
        """Persist an event and optionally forward it to a live transport."""
        self.store.append(event)
        self._publish(event)

    def _publish(self, event: Any) -> None:
        if self._event_sink is not None:
            try:
                self._event_sink(event)
            except Exception:
                logger.debug("Context Observatory event sink failed", exc_info=True)

    # ── on_llm_start ─────────────────────────────────────────────────────

    async def on_llm_start(
        self,
        context: RunContext,
        model: str,
        messages: list[Any],
    ) -> None:
        """Snapshot the full message list sent to the LLM."""
        self._turn += 1
        turn = self._turn

        try:
            # Build message snapshots
            msg_snapshots: list[MessageSnapshot] = []
            total_input_tokens = 0
            tokens_by_role: dict[str, int] = {}

            for m in messages:
                role = getattr(m, "role", "unknown")
                content = getattr(m, "content", "")
                content_str = content if isinstance(content, str) else str(content) if content is not None else ""
                
                # If assistant only called tools, build a representative preview
                tool_calls_meta = getattr(m, "tool_calls_meta", None)
                if role == "assistant" and not content_str.strip() and tool_calls_meta:
                    calls_repr = []
                    for tc in tool_calls_meta:
                        name = tc.get("name", "")
                        args = tc.get("args", "")
                        calls_repr.append(f"🔧 Tool Call: {name}({args})")
                    content_str = "\n".join(calls_repr)

                content_len = len(content_str)
                token_count = count_tokens(content_str, self.model or model)
                has_tool_calls = bool(tool_calls_meta)
                tool_call_id = getattr(m, "tool_call_id", None)

                # System prompts get full content; others get a reasonable limit
                _full_limit = 50_000 if role == "system" else 5_000
                msg_snapshots.append(MessageSnapshot(
                    role=role,
                    content_preview=content_str[:_PREVIEW_CHARS],
                    content_length=content_len,
                    token_count=token_count,
                    has_tool_calls=has_tool_calls,
                    tool_call_id=tool_call_id,
                    full_content=content_str[:_full_limit],
                ))
                total_input_tokens += token_count
                tokens_by_role[role] = tokens_by_role.get(role, 0) + token_count


            # Analyze system prompt composition
            system_breakdown: dict[str, int] = {}
            system_msgs = [m for m in messages if getattr(m, "role", "") == "system"]
            if system_msgs:
                sys_content = getattr(system_msgs[0], "content", "")
                system_breakdown = analyze_system_prompt(
                    sys_content, self.model or model
                )

            utilization = (
                (total_input_tokens / self.context_window * 100.0)
                if self.context_window > 0
                else 0.0
            )

            self._record(LLMCallEvent(
                turn=turn,
                model=model or self.model or "",
                messages=msg_snapshots,
                system_prompt_breakdown=system_breakdown,
                total_input_tokens=total_input_tokens,
                context_window=self.context_window,
                utilization_pct=round(utilization, 2),
                tokens_by_role=tokens_by_role,
                cumulative_input_tokens=self._cumulative_input + total_input_tokens,
                cumulative_output_tokens=self._cumulative_output,
            ))

            # Budget snapshot
            self._emit_budget_snapshot(turn, tokens_by_role)

            # Save for trim detection
            self._last_messages_snapshot = [
                (getattr(m, "role", ""), len(getattr(m, "content", "") or ""))
                for m in messages
            ]

        except Exception:
            logger.debug("ContextObserverHooks.on_llm_start failed", exc_info=True)

    # ── on_llm_end ───────────────────────────────────────────────────────

    async def on_llm_end(
        self,
        context: RunContext,
        model: str,
        response_text: str,
        usage: RequestUsage | None,
    ) -> None:
        """Record output tokens from the LLM response."""
        try:
            # Update the last LLM call event with output tokens
            llm_calls = self.store.get_llm_calls()
            if llm_calls:
                last = llm_calls[-1]
                if usage:
                    last.total_output_tokens = usage.output_tokens
                    last.cached_input_tokens = usage.cached_input_tokens
                    last.cache_creation_tokens = getattr(usage, "cache_creation_tokens", 0)
                    last.reasoning_tokens = getattr(usage, "reasoning_tokens", 0)
                    # Update input tokens with actual from provider if available
                    if usage.input_tokens > 0:
                        last.total_input_tokens = usage.input_tokens
                        last.utilization_pct = round(
                            usage.input_tokens / self.context_window * 100.0
                            if self.context_window > 0
                            else 0.0,
                            2,
                        )
                else:
                    # Estimate from response text
                    last.total_output_tokens = count_tokens(
                        response_text, self.model or model
                    )
                # Update response preview
                last.response_text_preview = (response_text or "")[:2000]
                last.response_text_length = len(response_text or "")
                # Update cumulative totals
                self._cumulative_input += last.total_input_tokens
                self._cumulative_output += last.total_output_tokens
                last.cumulative_input_tokens = self._cumulative_input
                last.cumulative_output_tokens = self._cumulative_output
                self._publish(last)
        except Exception:
            logger.debug("ContextObserverHooks.on_llm_end failed", exc_info=True)

    # ── on_pre_compact ───────────────────────────────────────────────────

    async def on_pre_compact(
        self,
        context: RunContext,
        message_count: int,
        token_estimate: int,
    ) -> None:
        """Record pre-compaction state."""
        try:
            self._pre_compact_tokens = token_estimate
            self._pre_compact_msg_count = message_count

            self._record(CompactionEvent(
                turn=self._turn,
                phase="start",
                tokens_before=token_estimate,
                messages_before=message_count,
            ))
        except Exception:
            logger.debug("on_pre_compact failed", exc_info=True)

    # ── on_post_compact ──────────────────────────────────────────────────

    async def on_post_compact(
        self,
        context: RunContext,
        message_count_after: int,
        summary: str | None,
    ) -> None:
        """Record post-compaction state and calculate savings."""
        try:
            tokens_after = 0
            # Estimate from summary if available
            if summary:
                tokens_after = count_tokens(summary, self.model)

            messages_dropped = max(
                0, self._pre_compact_msg_count - message_count_after
            )
            savings = (
                (self._pre_compact_tokens - tokens_after) / self._pre_compact_tokens * 100.0
                if self._pre_compact_tokens > 0
                else 0.0
            )

            self._record(CompactionEvent(
                turn=self._turn,
                phase="end",
                tokens_before=self._pre_compact_tokens,
                tokens_after=tokens_after,
                messages_before=self._pre_compact_msg_count,
                messages_after=message_count_after,
                messages_dropped=messages_dropped,
                savings_pct=round(savings, 1),
                summary_preview=(summary or "")[:300],
            ))
        except Exception:
            logger.debug("on_post_compact failed", exc_info=True)

    # ── on_tool_end ──────────────────────────────────────────────────────

    async def on_tool_end(
        self,
        context: RunContext,
        tool_name: str,
        call_id: str,
        success: bool,
        output: str,
        error: str | None,
    ) -> None:
        """Probe tool output for crush-eligible content and record metrics."""
        if not output or not success:
            return

        try:
            from clawagents.memory.content_crush import (
                DEFAULT_CRUSH_THRESHOLD,
                crush_tool_output,
            )

            # Only probe outputs large enough to be crush candidates
            if len(output) < DEFAULT_CRUSH_THRESHOLD:
                return

            # Simulate crush to measure savings (non-destructive)
            result = crush_tool_output(output, tool_name=tool_name)
            if result.did_crush:
                self._record(CrushEvent(
                    turn=self._turn,
                    tool_name=tool_name,
                    content_kind=result.kind,
                    original_chars=result.original_chars,
                    crushed_chars=result.crushed_chars,
                    saved_chars=result.saved_chars,
                    original_tokens=count_tokens(output, self.model),
                    crushed_tokens=count_tokens(result.text, self.model),
                ))
        except Exception:
            logger.debug("crush probe failed", exc_info=True)

    # ── Helper: budget snapshot ──────────────────────────────────────────

    def _emit_budget_snapshot(
        self, turn: int, tokens_by_role: dict[str, int]
    ) -> None:
        """Emit a budget allocation snapshot based on ContentBudgets defaults."""
        try:
            from clawagents.memory.content_budgets import DEFAULT_CONTENT_BUDGETS

            budgets = DEFAULT_CONTENT_BUDGETS
            cw = self.context_window

            self._record(BudgetSnapshot(
                turn=turn,
                system_tokens=tokens_by_role.get("system", 0),
                tool_tokens=tokens_by_role.get("tool", 0),
                user_assistant_tokens=(
                    tokens_by_role.get("user", 0)
                    + tokens_by_role.get("assistant", 0)
                ),
                image_tokens=0,  # TODO: detect from multimodal content
                budget_limits={
                    "system": int(cw * budgets.system),
                    "tools": int(cw * budgets.tools),
                    "user_assistant": int(cw * budgets.user_assistant),
                    "images": int(cw * budgets.images),
                },
                actual_usage=tokens_by_role,
            ))
        except Exception:
            logger.debug("budget snapshot failed", exc_info=True)

    # ── Trim detection helper ────────────────────────────────────────────

    def record_trim_event(
        self,
        role: str,
        original_chars: int,
        trimmed_chars: int,
    ) -> None:
        """Manually record a trim event (called by external instrumentation)."""
        self._record(TrimEvent(
            turn=self._turn,
            role=role,
            original_chars=original_chars,
            trimmed_chars=trimmed_chars,
            saved_chars=max(0, original_chars - trimmed_chars),
        ))
