"""Sub-agent delegation via the `task` tool.

Spawns an isolated ClawAgent.invoke() with a fresh context window.
Only the final result is returned to the parent agent.

Supports typed SubAgentSpec for per-agent configuration (name, prompt, etc.)

Hermes-derived guardrails:
  * Depth cap — recursive delegation is bounded by
    :data:`clawagents.run_context.MAX_SUBAGENT_DEPTH`. The tool refuses to
    spawn a child when the parent ``RunContext.depth`` is already at the cap.
  * Memory isolation — children always run with ``skip_memory=True`` so
    they cannot read the parent's memory directory, lessons, or skill state.
    Pass anything they need explicitly via the prompt.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from clawagents.providers.llm import LLMProvider
from clawagents.tools.registry import Tool, ToolRegistry, ToolResult
from clawagents.process.command_queue import enqueue_command_in_lane
from clawagents.process.lanes import CommandLane
from clawagents.config.features import is_enabled
from clawagents.run_context import MAX_SUBAGENT_DEPTH, RunContext

# Keys that must NOT be inherited by child agents — prevents parent context leakage.
EXCLUDED_STATE_KEYS: frozenset[str] = frozenset({
    "messages", "todos", "trajectory", "lessons", "session"
})

# Serialises the credential-proxy code path. Without this, two concurrent
# subagent invocations both mutate ``os.environ`` and the second one's "restore
# original" step picks up the FIRST one's overrides and stamps them back into
# place — leaving stale OPENAI_BASE_URL / ANTHROPIC_BASE_URL pointing at a
# proxy that has already stopped. The lock is only acquired when env mutation
# is actually happening (use_cred_proxy=True with API keys present); the
# common no-proxy path is unaffected.
_credential_proxy_env_lock = asyncio.Lock()


def _pin_llm_model(llm: LLMProvider, model: str) -> LLMProvider:
    """Return a provider clone that calls ``model`` instead of the parent's default."""
    if not model:
        return llm
    if getattr(llm, "model", None) == model:
        return llm
    try:
        from clawagents.providers.fallback import FallbackProvider

        if isinstance(llm, FallbackProvider):
            return FallbackProvider(
                _pin_llm_model(llm.primary, model),
                [_pin_llm_model(f, model) for f in llm.fallbacks],
                quarantine_threshold=llm.quarantine_threshold,
                health_check_interval_s=llm.health_check_interval_s,
                on_event=llm.on_event,
            )
    except Exception:
        pass
    if hasattr(llm, "model"):
        import copy

        pinned = copy.copy(llm)
        pinned.model = model
        return pinned
    from clawagents.config.config import load_config
    from clawagents.providers.llm import create_provider

    return create_provider(model, load_config())


async def _fire_parent_hook(
    run_context: Optional[RunContext],
    method_name: str,
    *args: Any,
) -> None:
    """Fire RunHooks stored on the parent run_context (best-effort)."""
    if run_context is None:
        return
    hooks = run_context._metadata.get("hooks") or []
    for h in hooks:
        fn = getattr(h, method_name, None)
        if fn is None:
            continue
        try:
            result = fn(run_context, *args)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass


@dataclass
class SubAgentSpec:
    """Specification for a named sub-agent with its own configuration.

    When the parent dispatches a task with a matching ``agent`` name,
    these settings override the defaults.
    """

    name: str
    """Unique name for this sub-agent type (e.g., 'researcher', 'coder')."""

    description: str
    """Human-readable description of what this sub-agent does."""

    system_prompt: Optional[str] = None
    """System prompt for this sub-agent."""

    max_iterations: int = 5
    """Max tool rounds. Default: 5."""

    use_native_tools: bool = True
    """Whether to use native tool calling for this sub-agent."""

    credential_proxy: bool = False
    """When True (and the ``credential_proxy`` feature flag is on), start a
    local credential proxy so the sub-agent never receives raw API keys.

    The sub-agent's environment will have ``OPENAI_BASE_URL`` /
    ``ANTHROPIC_BASE_URL`` pointed at the proxy, and real key env-vars
    will be stripped from its environment.
    """

    isolation: str = "none"
    """Filesystem isolation: ``none`` (default) or ``worktree``."""

    capability: str = "all"
    """Capability mode: ``read-only`` | ``read-write`` | ``execute`` | ``all``."""

    model: Optional[str] = None
    """Optional model pin for this sub-agent type."""

    persona: Optional[str] = None
    """Optional persona key looked up in TaskTool personas map."""

    tool_allowlist: Optional[List[str]] = None
    tool_denylist: Optional[List[str]] = None


def _registry_for_workspace(
    tools: ToolRegistry,
    workspace: str,
    *,
    deny_tools: frozenset[str] | None = None,
    allow_tools: frozenset[str] | None = None,
) -> ToolRegistry:
    """Shallow-retarget sandbox/workspace-bound tools onto ``workspace``."""
    import copy

    from clawagents.sandbox.local import LocalBackend

    sb = LocalBackend(root=workspace)
    child = ToolRegistry()
    # Inherit declarative permissions — worktree/capability clones must not
    # bypass deny/ask rules (rm -rf, credentials*, …).
    pe = getattr(tools, "_permission_engine", None)
    if pe is not None:
        child._permission_engine = pe  # type: ignore[attr-defined]
    ask_h = getattr(tools, "_permission_ask_handler", None)
    if ask_h is not None:
        child._permission_ask_handler = ask_h  # type: ignore[attr-defined]
    deny = deny_tools or frozenset()
    for tool in tools.list():
        name = getattr(tool, "name", "") or ""
        if name in deny or name == "task":
            continue
        if allow_tools is not None and name not in allow_tools and name not in ("think",):
            continue
        t = copy.copy(tool)
        if hasattr(t, "_sb"):
            t._sb = sb
        if hasattr(t, "_workspace"):
            t._workspace = workspace
        child.register(t)
    return child


class TaskTool:
    name = "task"

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        subagents: Optional[List[SubAgentSpec]] = None,
        use_queue: bool = False,
        personas: Optional[Dict[str, str]] = None,
        workspace: Optional[str] = None,
    ):
        self._llm = llm
        self._tools = tools
        self._subagents = subagents or []
        self._use_queue = use_queue
        self._personas = personas or {}
        self._workspace = workspace or os.getcwd()

        agent_names = [s.name for s in self._subagents]
        agent_list = f" Available specialized agents: {', '.join(agent_names)}." if agent_names else ""
        self.description = (
            "Delegate a task to a sub-agent with its own isolated context window. "
            "Use for complex sub-tasks that would clutter your main context. "
            "The sub-agent has access to the same tools but a fresh conversation. "
            "Set isolation=worktree for a git worktree-isolated child."
            + agent_list
        )
        self.parameters: Dict[str, Dict[str, Any]] = {
            "description": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
                "required": True,
            },
            "agent": {
                "type": "string",
                "description": f"Optional: name of a specialized sub-agent to use.{' Options: ' + ', '.join(agent_names) if agent_names else ''}",
            },
            "max_iterations": {
                "type": "number",
                "description": "Max tool rounds for the sub-agent. Default: 5",
            },
            "isolation": {
                "type": "string",
                "description": "none (default) or worktree — run child in a git worktree",
            },
            "capability": {
                "type": "string",
                "description": "read-only | read-write | execute | all",
            },
            "persona": {
                "type": "string",
                "description": "Optional persona overlay name",
            },
            "model": {
                "type": "string",
                "description": "Optional model pin for this child",
            },
        }

    async def execute(
        self,
        args: Dict[str, Any],
        run_context: Optional[RunContext] = None,
    ) -> ToolResult:
        from clawagents.graph.agent_loop import run_agent_graph

        description = str(args.get("description", ""))
        agent_name = args.get("agent")

        if not description:
            return ToolResult(success=False, output="", error="No task description provided")

        # ── Depth cap (Hermes parity) ──────────────────────────────────
        # A subagent must not itself spawn another subagent. The default
        # cap is 2 — top-level (0) → subagent (1) → would-be-grandchild (2)
        # is refused. We treat a missing run_context as depth=0 so this
        # only kicks in for genuinely nested calls.
        parent_depth = run_context.depth if run_context is not None else 0
        if parent_depth >= MAX_SUBAGENT_DEPTH:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Sub-agent delegation refused: depth cap of "
                    f"{MAX_SUBAGENT_DEPTH} reached "
                    f"(parent depth={parent_depth}). "
                    "Recursive delegation is disallowed; the parent "
                    "should perform the work directly or split it into "
                    "siblings rather than nesting another `task` call."
                ),
            )

        from clawagents.tools.subagent_resolve import resolve_subagent

        resolved = resolve_subagent(
            str(agent_name) if agent_name else None,
            specs=self._subagents,
            args=args,
            personas=self._personas,
        )
        spec = resolved.spec
        effective_max_iter = resolved.max_iterations
        effective_prompt = resolved.system_prompt
        effective_native_tools = resolved.use_native_tools
        use_cred_proxy = bool(resolved.credential_proxy and is_enabled("credential_proxy"))

        child_tools = self._tools
        child_workspace = self._workspace
        if resolved.isolation == "worktree" and is_enabled("task_worktree"):
            from clawagents.tools.worktree import ensure_task_worktree

            wt = ensure_task_worktree(
                workspace=self._workspace,
                name=f"{resolved.type}-{os.getpid()}",
            )
            if not wt.get("ok"):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"worktree isolation failed: {wt.get('error')}",
                )
            child_workspace = str(wt["path"])
            child_tools = _registry_for_workspace(
                self._tools,
                child_workspace,
                deny_tools=resolved.denied_tools(),
                allow_tools=resolved.tool_allowlist,
            )
        elif resolved.denied_tools() or resolved.tool_allowlist:
            child_tools = _registry_for_workspace(
                self._tools,
                child_workspace,
                deny_tools=resolved.denied_tools(),
                allow_tools=resolved.tool_allowlist,
            )

        async def do_run() -> ToolResult:
            from clawagents.sandbox.credential_proxy import CredentialProxy

            # Optionally wrap with credential proxy so the sub-agent never
            # sees raw API keys. The proxy injects credentials at the HTTP
            # transport layer; the sub-agent gets a safe localhost URL.
            proxy: Optional[CredentialProxy] = None
            proxy_env_overrides: Dict[str, str] = {}
            if use_cred_proxy:
                cred_headers: Dict[str, str] = {}
                for env_key, header_name in (
                    ("OPENAI_API_KEY", "Authorization"),
                    ("ANTHROPIC_API_KEY", "x-api-key"),
                ):
                    val = os.environ.get(env_key)
                    if val:
                        if env_key == "OPENAI_API_KEY":
                            cred_headers[header_name] = f"Bearer {val}"
                        else:
                            cred_headers[header_name] = val
                if cred_headers:
                    proxy = CredentialProxy(cred_headers)
                    proxy_url = proxy.start()
                    proxy_env_overrides = {
                        "OPENAI_BASE_URL": proxy_url,
                        "ANTHROPIC_BASE_URL": proxy_url,
                        # Strip real keys so the sub-agent can't use them directly
                        "OPENAI_API_KEY": "proxy",
                        "ANTHROPIC_API_KEY": "proxy",
                    }

            child_llm = (
                _pin_llm_model(self._llm, resolved.model)
                if resolved.model
                else self._llm
            )

            # Build kwargs, stripping any parent-context keys to keep the
            # child agent isolated (M1: subagent state isolation).
            run_kwargs: Dict[str, Any] = {
                k: v for k, v in {
                    "task": description,
                    "llm": child_llm,
                    "tools": child_tools,
                    "system_prompt": effective_prompt,
                    "max_iterations": effective_max_iter,
                    "streaming": False,
                    "use_native_tools": effective_native_tools,
                }.items() if k not in EXCLUDED_STATE_KEYS
            }

            # Fresh run-context for the child: bump depth, force memory
            # isolation, but keep the parent's permission_mode so a child
            # cannot escalate beyond what the user has authorised.
            #
            # Each subagent gets a *fresh* IterationBudget sized to its own
            # ``effective_max_iter`` (parity with Hermes' ``delegation.max_iterations``).
            # This means a runaway subagent cannot starve the parent's
            # remaining turns — the parent budget is left untouched.
            from clawagents.iteration_budget import IterationBudget as _IterBudget
            child_ctx: RunContext = RunContext(
                permission_mode=(
                    run_context.permission_mode
                    if run_context is not None
                    else RunContext().permission_mode
                ),
                depth=parent_depth + 1,
                skip_memory=True,
                iteration_budget=_IterBudget(max(1, int(effective_max_iter))),
                # Delegation is not a capability escape hatch: when a skill
                # allows the task tool, its tool boundary follows the child.
                active_skill_name=(
                    run_context.active_skill_name
                    if run_context is not None
                    else None
                ),
                active_skill_content_hash=(
                    run_context.active_skill_content_hash
                    if run_context is not None
                    else None
                ),
                active_skills=(
                    dict(run_context.active_skills)
                    if run_context is not None
                    else {}
                ),
                active_skill_allowed_tools=(
                    run_context.active_skill_allowed_tools
                    if run_context is not None
                    else None
                ),
            )
            child_ctx._metadata["workspace"] = child_workspace
            child_ctx._metadata["isolation"] = resolved.isolation
            child_ctx._metadata["subagent_type"] = resolved.type
            if run_context is not None:
                parent_on_event = getattr(run_context, "on_event", None)
                if callable(parent_on_event):
                    child_ctx.on_event = parent_on_event
                parent_session_id = getattr(run_context, "session_id", None) or (
                    run_context._metadata.get("session_id")
                    if isinstance(getattr(run_context, "_metadata", None), dict)
                    else None
                )
                if parent_session_id:
                    child_ctx.session_id = str(parent_session_id)
                    child_ctx._metadata["session_id"] = str(parent_session_id)
            # Forward parent gates so the child is not ungated.
            if run_context is not None and isinstance(run_context._metadata, dict):
                parent_meta = run_context._metadata
                if parent_meta.get("before_tool") is not None:
                    run_kwargs["before_tool"] = parent_meta["before_tool"]
                    child_ctx._metadata["before_tool"] = parent_meta["before_tool"]
                if parent_meta.get("permission_engine") is not None:
                    child_ctx._metadata["permission_engine"] = parent_meta[
                        "permission_engine"
                    ]
                    if getattr(child_tools, "_permission_engine", None) is None:
                        child_tools._permission_engine = parent_meta[  # type: ignore[attr-defined]
                            "permission_engine"
                        ]
                if parent_meta.get("approval_handler") is not None:
                    run_kwargs["approval_handler"] = parent_meta["approval_handler"]
                if parent_meta.get("taxonomy_dispatcher") is not None:
                    child_ctx._metadata["taxonomy_dispatcher"] = parent_meta[
                        "taxonomy_dispatcher"
                    ]
            run_kwargs["run_context"] = child_ctx
            # Subagent completion is not a session end: no session log, no dream.
            run_kwargs["session_end_tail"] = False

            parent_name = "ClawAgent"
            child_label = resolved.type
            if run_context is not None:
                parent_name = str(
                    run_context._metadata.get("agent_name") or parent_name
                )
            taxonomy_dispatcher = None
            if run_context is not None and isinstance(run_context._metadata, dict):
                taxonomy_dispatcher = run_context._metadata.get("taxonomy_dispatcher")
            if taxonomy_dispatcher is not None:
                try:
                    from clawagents.hooks.external import dispatch_taxonomy_hook
                    from clawagents.hooks.taxonomy import HookEvent

                    await dispatch_taxonomy_hook(
                        taxonomy_dispatcher,
                        HookEvent.SUBAGENT_START,
                        {
                            "parent": parent_name,
                            "subagent": child_label,
                            "description": description[:500],
                        },
                        blocking=False,
                    )
                except Exception:
                    pass
            await _fire_parent_hook(
                run_context,
                "on_subagent_start",
                parent_name,
                child_label,
                description,
            )

            state = None
            try:
                if proxy_env_overrides:
                    # Hold the lock across env mutate -> run -> env restore to
                    # serialise concurrent subagent runs that share os.environ.
                    async with _credential_proxy_env_lock:
                        _old_env: Dict[str, Optional[str]] = {}
                        for k, v in proxy_env_overrides.items():
                            _old_env[k] = os.environ.get(k)
                            os.environ[k] = v
                        try:
                            state = await run_agent_graph(**run_kwargs)
                        finally:
                            for k, orig in _old_env.items():
                                if orig is None:
                                    os.environ.pop(k, None)
                                else:
                                    os.environ[k] = orig
                            if proxy is not None:
                                proxy.stop()
                else:
                    # No proxy → no env mutation → no race → no lock needed.
                    state = await run_agent_graph(**run_kwargs)
                    if proxy is not None:
                        # Defensive: shouldn't happen (we only build proxy when
                        # there are overrides), but keep the cleanup symmetric.
                        proxy.stop()
            finally:
                if taxonomy_dispatcher is not None:
                    try:
                        from clawagents.hooks.external import dispatch_taxonomy_hook
                        from clawagents.hooks.taxonomy import HookEvent

                        await dispatch_taxonomy_hook(
                            taxonomy_dispatcher,
                            HookEvent.SUBAGENT_STOP,
                            {
                                "parent": parent_name,
                                "subagent": child_label,
                                "status": (
                                    "error"
                                    if state is None or state.status == "error"
                                    else state.status if state is not None else "error"
                                ),
                                "result_preview": (
                                    (state.result or "")[:500] if state is not None else ""
                                ),
                            },
                            blocking=False,
                        )
                    except Exception:
                        pass
                await _fire_parent_hook(
                    run_context,
                    "on_subagent_end",
                    parent_name,
                    child_label,
                    None if state is None else state.result,
                )

            if state is None:
                return ToolResult(
                    success=False, output="", error="Sub-agent failed to start",
                )

            if state.status == "error":
                return ToolResult(
                    success=False,
                    output=state.result or "",
                    error=f"Sub-agent failed: {state.result}",
                )

            agent_label = f"Sub-agent [{resolved.type}]"
            iso_note = f", isolation={resolved.isolation}" if resolved.isolation != "none" else ""
            return ToolResult(
                success=True,
                output=(
                    f"[{agent_label} completed: {state.tool_calls} tool calls, "
                    f"{state.iterations} iterations{iso_note}]\n\n{state.result}"
                ),
            )

        try:
            if self._use_queue:
                return await enqueue_command_in_lane(CommandLane.Subagent.value, do_run)
            return await do_run()
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Sub-agent error: {str(e)}")


def create_task_tool(
    llm: LLMProvider,
    tools: ToolRegistry,
    subagents: Optional[List[SubAgentSpec]] = None,
    use_queue: bool = False,
    personas: Optional[Dict[str, str]] = None,
    workspace: Optional[str] = None,
) -> Tool:
    """Factory function to create a TaskTool with the parent's LLM and tools."""
    return TaskTool(
        llm=llm,
        tools=tools,
        subagents=subagents,
        use_queue=use_queue,
        personas=personas,
        workspace=workspace,
    )
