"""Agent-to-agent handoffs.

A handoff is an LLM-visible tool (``transfer_to_<name>``) that, when called,
switches the running agent. The new agent takes over the conversation —
distinct from :func:`ClawAgent.as_tool`, where the parent calls the nested
agent and resumes after.

Inspired by openai-agents-python's ``agents.handoffs`` module but kept
backward-compatible with the existing ``run_agent_graph`` flow.

Public API
----------
- :class:`HandoffInputData` — what an :data:`InputFilter` receives
- :data:`InputFilter` — type alias for the filter callable
- :class:`Handoff` — runtime descriptor passed to the agent loop
- :func:`handoff` — convenience constructor (mirrors ``handoff()`` upstream)

Filters live in :mod:`clawagents.handoff_filters` (e.g. ``remove_all_tools``).
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from clawagents.providers.llm import LLMMessage
from clawagents.run_context import RunContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from clawagents.agent import ClawAgent


__all__ = [
    "HandoffInputData",
    "InputFilter",
    "Handoff",
    "handoff",
]


@dataclass
class HandoffInputData:
    """Snapshot of conversation state at the moment a handoff fires.

    ``input_history`` is the parent agent's full message list (system +
    user + any tool exchanges) up to the assistant turn that emitted the
    handoff tool call. ``pre_handoff_items`` and ``new_items`` mirror the
    upstream layout but, for the ClawAgent loop, we keep them as opaque
    lists so filters can stay forward-compatible if we add richer item
    types later. ``run_context`` is the live :class:`RunContext` for this
    run — filters may inspect it but should not mutate it in surprising
    ways.
    """

    input_history: list[LLMMessage]
    pre_handoff_items: list[Any] = field(default_factory=list)
    new_items: list[Any] = field(default_factory=list)
    run_context: Optional[RunContext] = None


InputFilter = Callable[[HandoffInputData], HandoffInputData]


@dataclass
class Handoff:
    """A handoff descriptor surfaced to the LLM as a synthetic tool.

    The agent loop turns each :class:`Handoff` on the running agent into
    a tool entry called ``name`` (default: ``transfer_to_<agent_name>``)
    with a single ``reason`` string parameter. When the LLM calls it, the
    loop runs the input filter, fires :meth:`RunHooks.on_handoff`, and
    re-enters the loop with ``target_agent_factory()`` as the active
    agent.

    ``target_agent_factory`` may be a zero-arg callable returning a
    :class:`ClawAgent`, or a :class:`ClawAgent` instance (which is
    wrapped in a constant factory).
    """

    name: str
    description: str
    target_agent_factory: Callable[[], "ClawAgent"]
    input_filter: Optional[InputFilter] = None
    on_handoff: Optional[Callable[[RunContext], Awaitable[None]]] = None

    def resolve_target(self) -> "ClawAgent":
        """Materialise the target agent. Memoised within a single instance."""
        return self.target_agent_factory()


def _default_handoff_name(agent_name: str) -> str:
    return f"transfer_to_{agent_name.replace(' ', '_')}"


def _default_handoff_description(agent_name: str) -> str:
    return (
        f"Hand off the conversation to the '{agent_name}' agent. "
        "Use this when the request is better handled by that agent. "
        "The other agent will take over the conversation; you will not "
        "resume after it finishes."
    )


def handoff(
    target: Union["ClawAgent", Callable[[], "ClawAgent"]],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    input_filter: Optional[InputFilter] = None,
    on_handoff: Optional[Callable[[RunContext], Awaitable[None]]] = None,
) -> Handoff:
    """Build a :class:`Handoff` from an agent or agent factory.

    The ``target`` may be either a :class:`ClawAgent` instance or a
    zero-argument callable returning one. The latter is preferred when
    the target agent needs lazy construction (avoids cyclic imports and
    delays heavy initialisation until the handoff fires).
    """
    # Local import to keep this module import-light and avoid cycles.
    from clawagents.agent import ClawAgent

    if isinstance(target, ClawAgent):
        agent_instance = target

        def _factory() -> ClawAgent:
            return agent_instance

        agent_name = _agent_display_name(agent_instance)
    elif callable(target):
        _factory = target
        # Best-effort name probe: invoke once to derive a default name.
        # Fall back to a generic identifier if probing fails.
        try:
            probe = target()
            agent_name = _agent_display_name(probe)
        except Exception:
            agent_name = "agent"
    else:
        raise TypeError(
            "handoff(target=...) expects a ClawAgent or zero-arg callable, "
            f"got {type(target).__name__}"
        )

    resolved_name = name or _default_handoff_name(agent_name)
    resolved_desc = description or _default_handoff_description(agent_name)

    return Handoff(
        name=resolved_name,
        description=resolved_desc,
        target_agent_factory=_factory,
        input_filter=input_filter,
        on_handoff=on_handoff,
    )


def _agent_display_name(agent: "ClawAgent") -> str:
    """Best-effort agent label for default tool names.

    ClawAgent has no ``.name`` field today, so we fall back to the system
    prompt's first non-empty line truncated to a slug-friendly length.
    """
    raw = getattr(agent, "name", None)
    if isinstance(raw, str) and raw:
        return raw
    sp = getattr(agent, "system_prompt", None) or ""
    if isinstance(sp, str) and sp.strip():
        first = sp.strip().splitlines()[0]
        # Crude slug: lowercase, replace whitespace runs with underscore.
        slug = "_".join(first.lower().split())[:32]
        if slug:
            return slug
    return "agent"
