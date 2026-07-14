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


class TaskTool:
    name = "task"

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        subagents: Optional[List[SubAgentSpec]] = None,
        use_queue: bool = False,
    ):
        self._llm = llm
        self._tools = tools
        self._subagents = subagents or []
        self._use_queue = use_queue

        agent_names = [s.name for s in self._subagents]
        agent_list = f" Available specialized agents: {', '.join(agent_names)}." if agent_names else ""
        self.description = (
            "Delegate a task to a sub-agent with its own isolated context window. "
            "Use for complex sub-tasks that would clutter your main context. "
            "The sub-agent has access to the same tools but a fresh conversation."
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
        }

    async def execute(
        self,
        args: Dict[str, Any],
        run_context: Optional[RunContext] = None,
    ) -> ToolResult:
        from clawagents.graph.agent_loop import run_agent_graph

        description = str(args.get("description", ""))
        agent_name = args.get("agent")
        try:
            max_iter = max(1, int(args.get("max_iterations", 5)))
        except (TypeError, ValueError):
            max_iter = 5

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

        spec: Optional[SubAgentSpec] = None
        if agent_name:
            spec = next((s for s in self._subagents if s.name == str(agent_name)), None)

        effective_max_iter = spec.max_iterations if spec else max_iter
        effective_prompt = spec.system_prompt if spec else None
        effective_native_tools = spec.use_native_tools if spec else True
        use_cred_proxy = bool(spec and spec.credential_proxy and is_enabled("credential_proxy"))

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

            # Build kwargs, stripping any parent-context keys to keep the
            # child agent isolated (M1: subagent state isolation).
            run_kwargs: Dict[str, Any] = {
                k: v for k, v in {
                    "task": description,
                    "llm": self._llm,
                    "tools": self._tools,
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
            run_kwargs["run_context"] = child_ctx

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

            if state.status == "error":
                return ToolResult(
                    success=False,
                    output=state.result or "",
                    error=f"Sub-agent failed: {state.result}",
                )

            agent_label = f"Sub-agent [{spec.name}]" if spec else "Sub-agent"
            return ToolResult(
                success=True,
                output=f"[{agent_label} completed: {state.tool_calls} tool calls, {state.iterations} iterations]\n\n{state.result}",
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
) -> Tool:
    """Factory function to create a TaskTool with the parent's LLM and tools."""
    return TaskTool(llm=llm, tools=tools, subagents=subagents, use_queue=use_queue)
