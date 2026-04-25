"""Class-based lifecycle hooks.

The existing ``before_llm`` / ``before_tool`` / ``after_tool`` callbacks are
kept untouched for backward compatibility. This module adds a richer,
class-based API inspired by openai-agents-python's ``RunHooks`` and
``AgentHooks`` so callers can override a few methods instead of wiring up
three separate callables.

Both legacy function-style hooks and class-style hooks are called by the
loop — class-style hooks always fire, and function-style hooks still get
the final word on blocking / modifications via :class:`HookResult`.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from clawagents.run_context import RunContext
from clawagents.usage import RequestUsage

TContext = TypeVar("TContext")


class RunHooks(Generic[TContext]):
    """Override any subset of these methods to observe the full run.

    Every method is ``async`` and a no-op by default. Exceptions raised by
    hooks are caught by the loop and logged — they never interrupt the run.
    """

    async def on_run_start(
        self,
        context: RunContext[TContext],
        task: str,
    ) -> None: ...

    async def on_run_end(
        self,
        context: RunContext[TContext],
        final_output: Any,
    ) -> None: ...

    async def on_agent_start(
        self,
        context: RunContext[TContext],
        agent_name: str,
    ) -> None: ...

    async def on_agent_end(
        self,
        context: RunContext[TContext],
        agent_name: str,
        final_output: Any,
    ) -> None: ...

    async def on_llm_start(
        self,
        context: RunContext[TContext],
        model: str,
        messages: list[Any],
    ) -> None: ...

    async def on_llm_end(
        self,
        context: RunContext[TContext],
        model: str,
        response_text: str,
        usage: RequestUsage | None,
    ) -> None: ...

    async def on_tool_start(
        self,
        context: RunContext[TContext],
        tool_name: str,
        call_id: str,
        args: dict[str, Any],
    ) -> None: ...

    async def on_tool_end(
        self,
        context: RunContext[TContext],
        tool_name: str,
        call_id: str,
        success: bool,
        output: str,
        error: str | None,
    ) -> None: ...

    async def on_handoff(
        self,
        context: RunContext[TContext],
        from_agent: str,
        to_agent: str,
    ) -> None: ...

    # ── v6.4: extended hook surface ───────────────────────────────────
    # All additive, default-noop. Existing 8 callbacks above are unchanged.

    async def on_pre_compact(
        self,
        context: RunContext[TContext],
        message_count: int,
        token_estimate: int,
    ) -> None:
        """Fires immediately before context compaction kicks in. Lets observers
        snapshot the pre-compaction transcript, pause writes, etc."""
        ...

    async def on_post_compact(
        self,
        context: RunContext[TContext],
        message_count_after: int,
        summary: str | None,
    ) -> None: ...

    async def on_subagent_start(
        self,
        context: RunContext[TContext],
        parent_agent: str,
        subagent_name: str,
        task: str,
    ) -> None: ...

    async def on_subagent_end(
        self,
        context: RunContext[TContext],
        parent_agent: str,
        subagent_name: str,
        output: Any,
    ) -> None: ...

    async def on_user_prompt_submit(
        self,
        context: RunContext[TContext],
        prompt: str,
    ) -> None:
        """Gateway-relevant: a new user message arrives mid-session.
        Lets observers gate / classify / annotate prompts at the boundary."""
        ...

    async def on_session_start(
        self,
        context: RunContext[TContext],
        session_id: str,
    ) -> None: ...

    async def on_session_end(
        self,
        context: RunContext[TContext],
        session_id: str,
    ) -> None: ...

    async def on_tool_failure(
        self,
        context: RunContext[TContext],
        tool_name: str,
        call_id: str,
        error: str,
    ) -> None:
        """Specialised handler for a tool that returned ``success=False``.

        Still receives ``on_tool_end`` (with ``success=False``) too — this
        method exists so observers can route failures to a separate sink
        without having to filter every ``on_tool_end`` call."""
        ...


class AgentHooks(RunHooks[TContext]):
    """Alias for per-agent hooks.

    Kept as a separate class for future divergence (e.g. an ``Agent`` that
    lives inside a multi-agent graph may want per-agent hooks *and* a
    top-level :class:`RunHooks`). Today the two APIs are identical so users
    can mix and match.
    """


_HOOK_METHODS: tuple[str, ...] = (
    "on_run_start", "on_run_end",
    "on_agent_start", "on_agent_end",
    "on_llm_start", "on_llm_end",
    "on_tool_start", "on_tool_end",
    "on_handoff",
    # v6.4 additive surface — keep in sync with RunHooks above
    "on_pre_compact", "on_post_compact",
    "on_subagent_start", "on_subagent_end",
    "on_user_prompt_submit",
    "on_session_start", "on_session_end",
    "on_tool_failure",
)


def composite_hooks(*hooks: RunHooks[TContext] | None) -> RunHooks[TContext]:
    """Combine multiple :class:`RunHooks` into a single composite.

    Callers can layer observability (tracing, metrics, logging) without
    wiring each one individually. Exceptions raised by any single hook are
    swallowed so one noisy observer can't break the others — matching the
    error semantics the loop already applies to class-based hooks.

    ``None`` entries in ``hooks`` are ignored, so it's safe to splat a list
    of optional observers without pre-filtering.
    """
    all_hooks: list[RunHooks[TContext]] = [h for h in hooks if h is not None]
    if not all_hooks:
        return RunHooks()
    if len(all_hooks) == 1:
        return all_hooks[0]

    composite: RunHooks[TContext] = RunHooks()

    def _make_dispatcher(method_name: str):
        async def _dispatch(*args: Any, **kwargs: Any) -> None:
            for h in all_hooks:
                try:
                    fn = getattr(h, method_name, None)
                    if fn is None:
                        continue
                    await fn(*args, **kwargs)
                except Exception:
                    # observation-only; swallow so one noisy observer can't
                    # break the others (same policy as the loop).
                    continue
        return _dispatch

    for method in _HOOK_METHODS:
        setattr(composite, method, _make_dispatcher(method))
    return composite
