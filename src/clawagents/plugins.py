"""Plugin manager for ClawAgents (learned from Hermes).

A *plugin* is a named bundle of hooks. The manager composes plugins in
priority order into the single hook slots accepted by ``run_agent_graph`` /
``Agent``. This lets callers register multiple cross-cutting concerns
(observability, redaction, policy, sandbox limits) without losing the
backward-compatible single-hook signature.

Hooks supported per plugin
--------------------------
* ``pre_tool`` — also called *pre_tool veto*. Same signature as
  :data:`clawagents.graph.agent_loop.BeforeToolHook`. Returning ``False`` or
  ``HookResult(allowed=False)`` blocks the call. Returning ``HookResult``
  with ``updated_args`` rewrites the call. The first plugin to deny wins.
* ``transform_tool_result`` — same signature as
  :data:`clawagents.graph.agent_loop.AfterToolHook`. Plugins compose like
  middleware: each receives the previous plugin's transformed result.
* ``before_llm`` — pre-flight message rewriter, composed left-to-right.

Plugin priority is ascending (lower number runs first). Two plugins with the
same priority retain registration order.
"""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from clawagents.graph.agent_loop import (
        BeforeLLMHook,
        BeforeToolHook,
        AfterToolHook,
        HookResult,
    )
    from clawagents.tools.registry import ToolResult


@dataclass(order=True)
class Plugin:
    """A bundle of optional hooks, identified by ``name``.

    ``transform_tool_result`` is an alias for ``after_tool``; both fields are
    accepted for ergonomic call sites. ``pre_tool`` is an alias for
    ``before_tool``.
    """

    sort_index: tuple[int, int] = field(init=False, repr=False)
    priority: int = 50
    name: str = ""
    pre_tool: Optional[Callable[..., Any]] = field(default=None, compare=False)
    before_tool: Optional[Callable[..., Any]] = field(default=None, compare=False)
    transform_tool_result: Optional[Callable[..., Any]] = field(default=None, compare=False)
    after_tool: Optional[Callable[..., Any]] = field(default=None, compare=False)
    before_llm: Optional[Callable[..., Any]] = field(default=None, compare=False)
    _seq: int = field(default=0, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.sort_index = (self.priority, self._seq)

    def resolved_before_tool(self) -> Optional[Callable[..., Any]]:
        return self.pre_tool or self.before_tool

    def resolved_after_tool(self) -> Optional[Callable[..., Any]]:
        return self.transform_tool_result or self.after_tool


class PluginManager:
    """Compose multiple plugins into single-hook adaptors.

    Usage::

        pm = PluginManager()
        pm.register(Plugin(name="audit", before_llm=...))
        pm.register(Plugin(name="sandbox", pre_tool=..., priority=10))

        agent = Agent(
            ...,
            before_tool=pm.composed_before_tool(),
            after_tool=pm.composed_after_tool(),
            before_llm=pm.composed_before_llm(),
        )
    """

    def __init__(self) -> None:
        self._plugins: List[Plugin] = []
        self._seq_counter = 0

    def register(self, plugin: Plugin) -> None:
        plugin._seq = self._seq_counter
        plugin.sort_index = (plugin.priority, plugin._seq)
        self._seq_counter += 1
        insort(self._plugins, plugin)

    def unregister(self, name: str) -> None:
        self._plugins = [p for p in self._plugins if p.name != name]

    def list_plugins(self) -> List[Plugin]:
        return list(self._plugins)

    def composed_before_tool(self) -> Optional[Callable[..., Any]]:
        raw_hooks = [p.resolved_before_tool() for p in self._plugins]
        hooks: list[Callable[..., Any]] = [h for h in raw_hooks if h is not None]
        if not hooks:
            return None

        def composed(tool_name: str, args: dict[str, Any]):
            from clawagents.graph.agent_loop import HookResult  # late import

            current_args = args
            for h in hooks:
                raw = h(tool_name, current_args)
                if raw is False:
                    return HookResult(allowed=False, reason="rejected by plugin")
                if isinstance(raw, HookResult):
                    if not raw.allowed:
                        return raw
                    if raw.updated_args is not None:
                        current_args = raw.updated_args
            if current_args is not args:
                return HookResult(allowed=True, updated_args=current_args)
            return HookResult(allowed=True)

        return composed

    def composed_after_tool(self) -> Optional[Callable[..., Any]]:
        raw_hooks = [p.resolved_after_tool() for p in self._plugins]
        hooks: list[Callable[..., Any]] = [h for h in raw_hooks if h is not None]
        if not hooks:
            return None

        def composed(tool_name: str, args: dict[str, Any], result: "ToolResult") -> "ToolResult":
            from clawagents.tools.registry import ToolResult as _TR  # late import

            current = result
            for h in hooks:
                try:
                    transformed = h(tool_name, args, current)
                except Exception:
                    continue
                if isinstance(transformed, _TR):
                    current = transformed
            return current

        return composed

    def composed_before_llm(self) -> Optional[Callable[..., Any]]:
        hooks = [p.before_llm for p in self._plugins if p.before_llm is not None]
        if not hooks:
            return None

        def composed(messages: list[Any]) -> list[Any]:
            current = messages
            for h in hooks:
                try:
                    out = h(current)
                except Exception:
                    continue
                if isinstance(out, list):
                    current = out
            return current

        return composed


__all__ = ["Plugin", "PluginManager"]
