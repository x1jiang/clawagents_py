"""Context preparation and provider recovery for one graph turn."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from clawagents.providers.llm import LLMMessage

from .context_management import (
    _CONTEXT_BUDGET_RATIO,
    _MAX_OVERFLOW_RETRIES,
    _MICRO_COMPACT_KEEP_RECENT,
    _MICRO_COMPACT_MIN_USAGE_RATIO,
    _compact_if_needed,
    _micro_compact_tool_results,
    _soft_trim_messages,
    _wal_write,
)
from .message_repair import _patch_dangling_tool_calls
from .model_profiles import resolve_context_budget, resolve_long_context_threshold
from .run_runtime import RunEvents, SessionMessageJournal
from .turn_llm import TurnLLMCaller

logger = logging.getLogger(__name__)

# Soft-trim fires at 75% of the full compaction budget so the cheaper
# operation gets a chance before the expensive LLM compaction.
_SOFT_TRIM_BUDGET_FRACTION = 0.75


# ── Incremental token estimation ──────────────────────────────────────────


class IncrementalTokenLedger:
    """Provider-reported usage with incremental estimation between checkpoints.

    After an exact provider input-token count (or a full re-estimate), only
    newly appended messages are re-estimated — avoiding O(n) recounts every
    turn when the prefix is unchanged by identity.
    """

    def __init__(self, estimate_messages: Callable[[list[LLMMessage]], int]) -> None:
        self._estimate_messages = estimate_messages
        self._checkpoint: list[LLMMessage] = []
        self._checkpoint_tokens = 0

    def rebase(
        self, messages: list[LLMMessage], exact_tokens: int | None = None
    ) -> int:
        self._checkpoint = list(messages)
        self._checkpoint_tokens = (
            int(exact_tokens)
            if exact_tokens is not None
            else int(self._estimate_messages(messages))
        )
        return self._checkpoint_tokens

    def record_provider_usage(
        self, messages: list[LLMMessage], input_tokens: int
    ) -> int:
        return self.rebase(
            messages, input_tokens if input_tokens > 0 else None
        )

    def estimate(self, messages: list[LLMMessage]) -> int:
        has_prefix = len(self._checkpoint) <= len(messages) and all(
            messages[i] is self._checkpoint[i] for i in range(len(self._checkpoint))
        )
        if not has_prefix:
            return self.rebase(messages)
        if len(messages) == len(self._checkpoint):
            return self._checkpoint_tokens
        return self._checkpoint_tokens + int(
            self._estimate_messages(messages[len(self._checkpoint):])
        )


@dataclass(frozen=True)
class TurnCallOutcome:
    """Result of preparing and calling the model for a turn."""

    action: Literal["response", "retry", "stop"]
    messages: list[LLMMessage]
    response: Any | None = None


class TurnDriver:
    """Owns message preparation and context-overflow recovery state."""

    def __init__(
        self,
        *,
        llm: Any,
        caller: TurnLLMCaller,
        events: RunEvents,
        run_context: Any,
        session_journal: SessionMessageJournal,
        external_hooks: Any,
        before_llm: Any,
        fire_hook: Any,
        taxonomy_dispatcher: Any,
        native_schemas: Any,
        handoffs: list[Any],
        use_native_tools: bool,
        tools_supplied: bool,
        streaming: bool,
        output_type: type | None,
        context_window: int,
        resolved_model_name: str | None,
        cached_system_tokens: int,
        compaction_savings: list[float],
        token_ledger: IncrementalTokenLedger | None = None,
    ) -> None:
        self._llm = llm
        self._caller = caller
        self._events = events
        self._run_context = run_context
        self._session_journal = session_journal
        self._external_hooks = external_hooks
        self._before_llm = before_llm
        self._fire_hook = fire_hook
        self._taxonomy_dispatcher = taxonomy_dispatcher
        self._native_schemas = native_schemas
        self._handoffs = handoffs
        self._use_native_tools = use_native_tools
        self._tools_supplied = tools_supplied
        self._streaming = streaming
        self._output_type = output_type
        self._context_window = context_window
        self._resolved_model_name = resolved_model_name
        self._cached_system_tokens = cached_system_tokens
        self._compaction_savings = compaction_savings
        self._token_ledger = token_ledger
        self._token_multiplier = 1.0
        self._overflow_retries = 0

    async def call(
        self,
        messages: list[LLMMessage],
        *,
        state: Any,
        round_index: int,
        cancel_event: Any,
    ) -> TurnCallOutcome:
        """Prepare a request and return a response, retry request, or stop."""
        _wal_write(messages)
        self._session_journal.note(messages, durable=True)
        messages = await self._prepare_messages(messages)
        self._session_journal.note(messages, durable=False)

        try:
            result = await self._caller.call(
                messages,
                resolved_model_name=self._resolved_model_name,
                use_native_tools=self._use_native_tools,
                tools_supplied=self._tools_supplied,
                initial_schemas=self._native_schemas,
                handoffs=self._handoffs,
                streaming=self._streaming,
                cancel_event=cancel_event,
                run_context=self._run_context,
                output_type=self._output_type,
            )
        except Exception as exc:
            return await self._recover_from_error(messages, state, round_index, exc)

        self._resolved_model_name = result.resolved_model_name
        if result.response.partial and not result.response.content.strip():
            self._events.emit("warn", {"message": "interrupted — no content received"})
            state.status = "done"
            state.result = state.result or "[interrupted]"
            return TurnCallOutcome("stop", messages)
        return TurnCallOutcome("response", messages, result.response)

    async def _prepare_messages(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        messages = _patch_dangling_tool_calls(messages)
        # Use ledger for incremental estimation when available.
        current_tokens = (
            self._token_ledger.estimate(messages)
            if self._token_ledger is not None
            else self._budget_tokens(messages)
        )
        # Model-aware budget thresholds.
        context_budget_window, context_budget_ratio = (
            resolve_context_budget(self._resolved_model_name, self._context_window)
            if self._resolved_model_name
            else (self._context_window, _CONTEXT_BUDGET_RATIO)
        )
        compaction_budget = int(context_budget_window * context_budget_ratio)
        soft_trim_budget = int(compaction_budget * _SOFT_TRIM_BUDGET_FRACTION)

        messages, current_tokens = self._micro_compact(
            messages, current_tokens
        )
        if current_tokens > soft_trim_budget:
            trimmed = _soft_trim_messages(
                messages,
                self._context_window,
                self._token_multiplier,
                self._events.emit,
                self._resolved_model_name,
                current_tokens,
            )
            if trimmed is not messages:
                messages = trimmed
                current_tokens = self._rebase_ledger(messages)
        if current_tokens > compaction_budget:
            messages = await self._compact(messages)
            self._rebase_ledger(messages)
        # Compaction / trim can still leave pairs inconsistent — sanitize again.
        messages = _patch_dangling_tool_calls(messages)
        await self._apply_external_pre_llm(messages)
        return self._apply_before_llm(messages)

    def _rebase_ledger(self, messages: list[LLMMessage]) -> int:
        """Rebase the token ledger after a context mutation."""
        if self._token_ledger is not None:
            return self._token_ledger.rebase(messages)
        return self._budget_tokens(messages)

    def _micro_compact(
        self, messages: list[LLMMessage], current_tokens: int
    ) -> tuple[list[LLMMessage], int]:
        from clawagents.harness_profiles import resolve_harness_profile

        profile = resolve_harness_profile(self._resolved_model_name)
        keep_recent = (
            int(profile.clear_tool_keep)
            if profile and profile.clear_tool_keep is not None
            else _MICRO_COMPACT_KEEP_RECENT
        )
        ratio = (
            float(profile.clear_tool_trigger_ratio)
            if profile and profile.clear_tool_trigger_ratio is not None
            else _MICRO_COMPACT_MIN_USAGE_RATIO
        )
        economic_limit = resolve_long_context_threshold(self._resolved_model_name)
        economic_trigger = int(economic_limit * 0.90) if economic_limit else None
        if current_tokens > self._context_window * ratio or (
            economic_trigger is not None and current_tokens > economic_trigger
        ):
            compacted = _micro_compact_tool_results(messages, keep_recent=keep_recent)
            if compacted is not messages:
                messages = compacted
                current_tokens = self._rebase_ledger(messages)
        return messages, current_tokens

    async def _compact(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        result = await _compact_if_needed(
            messages,
            self._context_window,
            self._llm,
            self._events.emit,
            self._token_multiplier,
            self._resolved_model_name,
            self._run_context,
            fire_hook=self._fire_hook,
            savings_history=self._compaction_savings,
            taxonomy_dispatcher=self._taxonomy_dispatcher,
        )
        return result

    async def _apply_external_pre_llm(self, messages: list[LLMMessage]) -> None:
        if self._external_hooks is None:
            return
        try:
            history = [
                {
                    "role": message.role,
                    "content": message.content[:100]
                    if isinstance(message.content, str)
                    else "",
                }
                for message in messages[-3:]
            ]
            extra_messages = await self._external_hooks.pre_llm(history)
            for message in extra_messages or []:
                role = message.get("role", "user")
                if role not in ("system", "user", "assistant", "tool"):
                    self._events.emit(
                        "warn",
                        {
                            "message": (
                                "external pre_llm hook returned message with unknown role "
                                f"{role!r}; coercing to 'user'"
                            )
                        },
                    )
                    role = "user"
                messages.append(LLMMessage(role=role, content=message.get("content", "")))
        except Exception as exc:
            self._events.emit("warn", {"message": f"external pre_llm hook error: {exc}"})

    def _apply_before_llm(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        if self._before_llm is None:
            return messages
        try:
            transformed = self._before_llm(messages)
            if isinstance(transformed, list) and transformed:
                return transformed
            self._events.emit(
                "warn", {"message": "before_llm returned invalid value — ignored"}
            )
        except Exception as exc:
            self._events.emit("warn", {"message": f"before_llm hook error: {exc}"})
        return messages

    async def _recover_from_error(
        self,
        messages: list[LLMMessage],
        state: Any,
        round_index: int,
        error: Exception,
    ) -> TurnCallOutcome:
        from clawagents.errors.taxonomy import ErrorClass, classify_error

        descriptor = classify_error(error)
        self._events.emit(
            "error",
            {
                "phase": "llm_call",
                "message": str(error),
                "error_class": descriptor.error_class.value,
                "retryable": descriptor.retryable,
                "recovery_hint": descriptor.recovery_hint,
            },
        )
        if descriptor.error_class != ErrorClass.CONTEXT_WINDOW:
            logger.exception(
                "LLM call failed at round %d: [%s] %s",
                round_index,
                descriptor.error_class.value,
                error,
            )
            state.status = "error"
            state.result = f"[{descriptor.error_class.value}] {descriptor.recovery_hint}"
            return TurnCallOutcome("stop", messages)

        self._overflow_retries += 1
        if self._overflow_retries > _MAX_OVERFLOW_RETRIES:
            self._events.emit(
                "error",
                {
                    "phase": "llm_call",
                    "message": (
                        f"context overflow persists after {_MAX_OVERFLOW_RETRIES} retries. "
                        "Increase CONTEXT_WINDOW, reduce tools, or shorten your instruction."
                    ),
                },
            )
            state.status = "error"
            state.result = str(error)
            return TurnCallOutcome("stop", messages)

        observed_ratio = self._context_window / max(self._budget_tokens(messages, 1.0), 1)
        self._token_multiplier = min(observed_ratio * 1.1, 3.0)
        self._context_window = max(int(self._context_window * 0.5), 16_000)
        self._events.emit(
            "context",
            {
                "message": (
                    f"token overflow — calibrated multiplier to {self._token_multiplier:.2f}, "
                    f"shrunk effective window to {self._context_window} "
                    f"(retry {self._overflow_retries}/{_MAX_OVERFLOW_RETRIES})"
                )
            },
        )
        messages = _soft_trim_messages(
            messages,
            self._context_window,
            self._token_multiplier,
            self._events.emit,
            self._resolved_model_name,
        )
        messages = await self._compact(messages)
        self._rebase_ledger(messages)
        self._session_journal.note(messages, durable=False)
        return TurnCallOutcome("retry", messages)

    def _budget_tokens(self, messages: list[LLMMessage], multiplier: float | None = None) -> int:
        from .tool_observation import _estimate_messages_tokens

        return _estimate_messages_tokens(
            messages,
            multiplier=self._token_multiplier if multiplier is None else multiplier,
            model=self._resolved_model_name,
            cached_system_tokens=self._cached_system_tokens or None,
        )
