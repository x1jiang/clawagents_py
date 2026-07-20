import os
import re
import asyncio
import difflib
import unicodedata
import warnings
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Union

from clawagents.providers.llm import LLMProvider
from clawagents.tools.registry import ToolRegistry, Tool, ToolResult
from clawagents.graph.agent_loop import (
    run_agent_graph, AgentState, OnEvent,
    BeforeLLMHook, BeforeToolHook, AfterToolHook,
)
from clawagents.run_context import RunContext
from clawagents.lifecycle import RunHooks, AgentHooks
from clawagents.guardrails import InputGuardrail, OutputGuardrail
from clawagents.stream_events import StreamEvent
from clawagents.handoffs import Handoff


class LangChainToolAdapter:
    """
    Wraps a LangChain-style tool (with .ainvoke / .invoke) into a
    ClawAgent-compatible Tool with .execute().
    """
    def __init__(self, lc_tool):
        self.name = getattr(lc_tool, "name", type(lc_tool).__name__)
        self.description = getattr(lc_tool, "description", "")
        self.parameters = self._extract_params(lc_tool)
        self._lc_tool = lc_tool

    def _extract_params(self, lc_tool) -> Dict[str, Dict[str, Any]]:
        schema = getattr(lc_tool, "args_schema", None)
        if schema and hasattr(schema, "schema"):
            try:
                s = schema.schema()
                props = s.get("properties", {})
                required = s.get("required", [])
                return {
                    k: {
                        "type": v.get("type", "string"),
                        "description": v.get("description", ""),
                        "required": k in required,
                    }
                    for k, v in props.items()
                }
            except Exception:
                pass
        return {}

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        try:
            if hasattr(self._lc_tool, "ainvoke"):
                result = await self._lc_tool.ainvoke(args)
            elif hasattr(self._lc_tool, "invoke"):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self._lc_tool.invoke(args))
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self._lc_tool.run(**args))
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ClawAgent:
    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        streaming: bool = True,
        use_native_tools: bool = True,
        context_window: int = 1_000_000,
        on_event: Optional[OnEvent] = None,
        before_llm: Optional[BeforeLLMHook] = None,
        before_tool: Optional[BeforeToolHook] = None,
        after_tool: Optional[AfterToolHook] = None,
        trajectory: bool = False,
        rethink: bool = False,
        learn: bool = False,
        atlas: bool = False,
        atlas_config: Optional[Any] = None,
        max_iterations: int = 200,
        preview_chars: int = 120,
        response_chars: int = 500,
        features: Optional[dict[str, bool]] = None,
        workspace: Optional[Union[str, os.PathLike]] = None,
        advisor_llm: Optional[LLMProvider] = None,
        advisor_max_calls: int = 3,
        # ── New, backward-compatible surfaces (OpenAI-Agents-inspired) ──
        hooks: Optional[RunHooks] = None,
        agent_hooks: Optional[AgentHooks] = None,
        input_guardrails: Optional[list[InputGuardrail]] = None,
        output_guardrails: Optional[list[OutputGuardrail]] = None,
        output_type: Optional[type] = None,
        session: Any = None,
        session_preload_limit: int | None = 200,
        on_stream_event: Optional[Callable[[StreamEvent], None]] = None,
        # ── v6.4: Handoffs + Agent.as_tool ──
        handoffs: Optional[list[Handoff]] = None,
        name: Optional[str] = None,
        action_mode: str = "tools",
        approval_handler: Any = None,
        require_approval_tools: Optional[List[str]] = None,
    ):
        """
        Initialize a ClawAgent.

        Args:
            llm: The initialized LLM provider
            tools: The registry containing all available tools
            system_prompt: Optional base system instruction (alias: ``instruction``)
            instruction: Alias for ``system_prompt`` (factory-style naming)
            streaming: Whether to stream responses from the LLM
            use_native_tools: Instruct the LLM to use native structured tool calls (if supported)
            context_window: Maximum allowed tokens before oldest messages are compacted
            trajectory: Whether to log full trajectory data
            rethink: Enables logic to backtrack on consecutive execution failures
            learn: Whether to use post-trajectory reflection to extract permanent lessons
            atlas: Deprecated no-op (ATLAS removed; use goal_mode / start_goal)
            atlas_config: Deprecated no-op
            max_iterations: Maximum loop turns before returning early
            preview_chars: Number of characters to log in console output for tool results
            response_chars: Number of characters to log from LLM free-text response
            features: Dictionary to override global architectural variables (e.g. {"micro_compact": False, "wal": True})
            workspace: Project root for workspace-scoped side effects (``.clawagents/``,
                filesystem watcher, cwd-scoped tools). Does not ``chdir`` the process.
            advisor_llm: Optional stronger model for strategic guidance (consulted 2-3 times per task)
            advisor_max_calls: Maximum advisor consultations per task (default: 3)
            action_mode: ``tools`` (default) or ``code`` (CodeAct Python actions)
            approval_handler: ``None`` | ``"event"`` | callable — block tools that require approval
            require_approval_tools: tool names that always require approval when handler is set
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = _resolve_system_prompt(system_prompt, instruction)
        self.streaming = streaming
        self.use_native_tools = use_native_tools
        self.context_window = context_window
        self.on_event = on_event
        self.before_llm = before_llm
        self.before_tool = before_tool
        self.after_tool = after_tool
        self.trajectory = trajectory
        self.rethink = rethink
        self.learn = learn
        self.atlas = False  # ATLAS removed
        self.atlas_config = None
        self.max_iterations = max_iterations
        self.preview_chars = preview_chars
        self.response_chars = response_chars
        self.features = features
        self.workspace = (
            str(Path(workspace).expanduser().resolve()) if workspace is not None else None
        )
        self.advisor_llm = advisor_llm
        self.advisor_max_calls = advisor_max_calls
        self.hooks = hooks
        self.agent_hooks = agent_hooks
        self.input_guardrails = input_guardrails
        self.output_guardrails = output_guardrails
        self.output_type = output_type
        self.session = session
        self.session_preload_limit = session_preload_limit
        self.on_stream_event = on_stream_event
        self.handoffs: list[Handoff] = list(handoffs) if handoffs else []
        self.name = name
        self.action_mode = action_mode if action_mode in ("tools", "code") else "tools"
        self.approval_handler = approval_handler
        self.require_approval_tools = list(require_approval_tools or [])
        self.goal_mode = False

    async def invoke(
        self,
        task: str,
        max_iterations: Optional[int] = None,
        on_event: Optional[OnEvent] = None,
        timeout_s: float = 0,
        features: Optional[dict[str, bool]] = None,
        *,
        run_context: Optional[RunContext] = None,
        user_context: Any = None,
        hooks: Optional[RunHooks] = None,
        agent_hooks: Optional[AgentHooks] = None,
        input_guardrails: Optional[list[InputGuardrail]] = None,
        output_guardrails: Optional[list[OutputGuardrail]] = None,
        output_type: Optional[type] = None,
        session: Any = None,
        session_preload_limit: Optional[int] = None,
        on_stream_event: Optional[Callable[[StreamEvent], None]] = None,
        handoffs: Optional[list[Handoff]] = None,
        images: Optional[list[dict]] = None,
        files: Optional[list[dict]] = None,
        session_end_tail: bool = True,
    ) -> AgentState:
        """Start the ReAct agent loop for ``task``.

        All per-call keyword arguments (``hooks``, ``input_guardrails``,
        ``output_type``, ``session``, ``on_stream_event`` …) override the
        values supplied to ``ClawAgent.__init__`` for just this invocation.

        ``images`` attaches image content to the first user message so a vision
        model sees pixels, not a path. Each item is ``{"data": <base64 or
        data-URL>, "media_type": "image/png"}``; images are sanitized
        (resized/recompressed to provider limits) before sending.

        ``files`` attaches documents the same way. Each item is
        ``{"data": <base64 or data-URL>, "media_type": "application/pdf",
        "name": "report.pdf"}``. PDFs reach the model natively; DOCX is
        text-extracted; anything else degrades to a short text note.
        """
        image_blocks: Optional[list[dict]] = None
        if images:
            from clawagents.media.images import build_user_image_block

            image_blocks = []
            for img in images:
                if isinstance(img, str):
                    image_blocks.append(build_user_image_block(img))
                elif isinstance(img, dict):
                    data = img.get("data") or img.get("url") or ""
                    media_type = (
                        img.get("media_type") or img.get("mime_type") or "image/png"
                    )
                    if data:
                        image_blocks.append(build_user_image_block(data, media_type))

        file_blocks: Optional[list[dict]] = None
        if files:
            from clawagents.media.documents import build_user_file_block

            file_blocks = []
            for f in files:
                if not isinstance(f, dict):
                    continue
                data = f.get("data") or f.get("url") or ""
                media_type = (
                    f.get("media_type") or f.get("mime_type") or "application/pdf"
                )
                fname = f.get("name") or f.get("filename") or None
                if data:
                    file_blocks.append(
                        build_user_file_block(data, media_type, name=fname)
                    )

        if run_context is None:
            run_context = RunContext(context=user_context)
        elif user_context is not None and run_context.context is None:
            run_context.context = user_context
        ws = getattr(self, "workspace", None)
        if ws:
            if not isinstance(run_context._metadata, dict):
                run_context._metadata = {}
            run_context._metadata.setdefault("workspace", ws)
        default_pm = getattr(self, "_default_permission_mode", None)
        if default_pm is not None:
            run_context.permission_mode = default_pm
        store = getattr(self, "skill_store", None)
        if store is not None:
            run_context._metadata["skill_store"] = store
        # Gate Goal reminder + final verifier on this turn's mode (Act ≠ Goal).
        run_context._metadata["goal_mode"] = bool(getattr(self, "goal_mode", False))
        # OS sandbox contract — used by execute unsandboxed retry + env banner.
        if not isinstance(run_context._metadata, dict):
            run_context._metadata = {}
        run_context._metadata["sandbox_profile"] = getattr(
            self, "_sandbox_profile_name", "workspace"
        )
        run_context._metadata["chat_mode"] = getattr(self, "_chat_mode", None)
        run_context._metadata["allow_full_access"] = bool(
            getattr(self, "_allow_full_access", False)
        )
        run_context._metadata["allow_unsandboxed_exec"] = bool(
            getattr(self, "_allow_full_access", False)
            and getattr(self, "_chat_mode", None) == "full_access"
        )

        return await run_agent_graph(
            task=task,
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=max_iterations if max_iterations is not None else self.max_iterations,
            streaming=self.streaming,
            context_window=self.context_window,
            on_event=on_event or self.on_event,
            before_llm=self.before_llm,
            before_tool=self.before_tool,
            after_tool=self.after_tool,
            use_native_tools=self.use_native_tools,
            trajectory=self.trajectory,
            rethink=self.rethink,
            learn=self.learn,
            atlas=self.atlas,
            atlas_config=self.atlas_config,
            preview_chars=self.preview_chars,
            response_chars=self.response_chars,
            timeout_s=timeout_s,
            features=features if features is not None else self.features,
            advisor_llm=self.advisor_llm,
            advisor_max_calls=self.advisor_max_calls,
            run_context=run_context,
            user_context=user_context,
            hooks=hooks if hooks is not None else self.hooks,
            agent_hooks=agent_hooks if agent_hooks is not None else self.agent_hooks,
            input_guardrails=(
                input_guardrails if input_guardrails is not None else self.input_guardrails
            ),
            output_guardrails=(
                output_guardrails if output_guardrails is not None else self.output_guardrails
            ),
            output_type=output_type if output_type is not None else self.output_type,
            session=session if session is not None else self.session,
            session_preload_limit=(
                session_preload_limit
                if session_preload_limit is not None
                else self.session_preload_limit
            ),
            on_stream_event=(
                on_stream_event if on_stream_event is not None else self.on_stream_event
            ),
            handoffs=handoffs if handoffs is not None else self.handoffs,
            agent_name=self.name,
            action_mode=self.action_mode,
            approval_handler=self.approval_handler,
            require_approval_tools=self.require_approval_tools,
            image_blocks=image_blocks,
            file_blocks=file_blocks,
            session_end_tail=session_end_tail,
        )

    # ── Convenience hook methods ──────────────────────────────────────

    def _convenience_gate_base(self):
        """The before_tool gate to preserve underneath convenience filters.

        Captured once, on the first convenience-method call — by then
        ``create_claw_agent`` has already installed its permission + plan-mode
        gate as ``self.before_tool``. Subsequent convenience calls (e.g.
        ``block_tools`` then ``allow_only_tools``) replace each other's filter
        but always recompose against this same base, so the security gate is
        never dropped while the documented "last filter wins" behavior among
        the convenience methods is preserved. A bare ``ClawAgent`` with no gate
        captures ``None``; ``compose_before_tool`` then returns the lone filter
        unchanged (so it keeps returning plain booleans).
        """
        if not hasattr(self, "_before_tool_base"):
            self._before_tool_base = self.before_tool
        return self._before_tool_base

    def block_tools(self, *tool_names: str):
        """Block specific tools from being executed.

        Composes with (does not replace) the permission/plan-mode gate that
        ``create_claw_agent`` installs — blocking a tool must never widen what
        the other gates allow.

        Example: agent.block_tools("execute", "write_file")
        """
        from clawagents.modes import compose_before_tool

        base = self._convenience_gate_base()
        blocked = set(tool_names)

        def _block(name, args):
            return name not in blocked

        self.before_tool = compose_before_tool(_block, base)

    def allow_only_tools(self, *tool_names: str):
        """Only allow specific tools to be executed. All others blocked.

        Composes with the existing gate (see :meth:`block_tools`): an allowed
        tool still passes through the permission/plan-mode gate rather than
        bypassing it.

        Example: agent.allow_only_tools("read_file", "ls", "grep")
        """
        from clawagents.modes import compose_before_tool

        base = self._convenience_gate_base()
        allowed = set(tool_names)

        def _allow(name, args):
            return name in allowed

        self.before_tool = compose_before_tool(_allow, base)

    def inject_context(self, text: str):
        """Inject additional context into every LLM call.

        Example: agent.inject_context("Always respond in Spanish")
        """
        from clawagents.providers.llm import LLMMessage
        existing = self.before_llm

        marker = f"[Context] {text}"

        def hook(messages):
            if existing:
                messages = existing(messages)
            # Idempotent: before_llm runs every loop round; without this
            # check the context message accumulated one copy per round.
            if any(getattr(m, "content", None) == marker for m in messages):
                return messages
            return [*messages, LLMMessage(role="user", content=marker)]

        self.before_llm = hook

    async def compare(
        self,
        task: str,
        n_samples: int = 3,
        max_iterations: Optional[int] = None,
        on_event: Optional[OnEvent] = None,
    ) -> Dict[str, Any]:
        """Run the task N times and return the best result (GRPO-inspired).

        Runs the same task multiple times, scores each using deterministic
        signals from tool outputs, and returns the highest-scoring result.

        Example: result = await agent.compare("Fix the bug in app.py", n_samples=3)
        """
        from clawagents.trajectory.compare import compare_samples
        return await compare_samples(
            task=task,
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            n_samples=n_samples,
            max_iterations=max_iterations if max_iterations is not None else self.max_iterations,
            streaming=False,
            context_window=self.context_window,
            on_event=on_event or self.on_event,
            use_native_tools=self.use_native_tools,
            rethink=self.rethink,
            learn=self.learn,
            preview_chars=self.preview_chars,
            response_chars=self.response_chars,
        )

    def as_tool(
        self,
        *,
        tool_name: str,
        tool_description: str,
        custom_output_extractor: Optional[Callable[[AgentState], str]] = None,
        needs_approval: bool = False,
    ) -> Tool:
        """Expose this agent as a tool callable by another agent.

        Unlike a :class:`~clawagents.handoffs.Handoff`, the wrapped agent
        is invoked synchronously inside the parent's tool dispatch:
        parent calls, child runs, parent resumes. The default output is
        ``state.result`` from the child's terminal turn; provide
        ``custom_output_extractor`` to project anything from the final
        :class:`AgentState` (e.g. ``state.final_output`` after structured
        output coercion, or a specific trajectory turn).

        When ``needs_approval=True`` the tool emits an
        :class:`ApprovalRequiredEvent` and waits for the parent's
        :class:`RunContext` approval store before running the child —
        the same mechanism used by the permissions module's
        ``needs_approval`` policy.
        """
        wrapped_agent = self
        return _AgentAsTool(
            agent=wrapped_agent,
            tool_name=tool_name,
            tool_description=tool_description,
            custom_output_extractor=custom_output_extractor,
            needs_approval=needs_approval,
        )

    def truncate_output(self, max_chars: int = 5000):
        """Truncate tool outputs to a maximum character length.

        Example: agent.truncate_output(3000)
        """
        existing = self.after_tool

        def hook(name, args, result):
            # Chain any existing after_tool hook first, then truncate.
            if existing is not None:
                result = existing(name, args, result)
            if len(result.output) > max_chars:
                from clawagents.tools.registry import ToolResult
                return ToolResult(
                    success=result.success,
                    output=result.output[:max_chars] + f"\n...(truncated {len(result.output) - max_chars} chars)",
                    error=result.error,
                )
            return result

        self.after_tool = hook


class _AgentAsTool:
    """Tool adapter produced by :meth:`ClawAgent.as_tool`.

    Conforms to the :class:`~clawagents.tools.registry.Tool` protocol.
    Invokes the wrapped agent on ``args["task"]`` (free-text) and
    returns the extracted output as a :class:`ToolResult`.

    Approval gating is handled inside :meth:`execute` rather than via
    :class:`RunContext.is_tool_approved` so that the parent loop's
    standard approval-required emission still works — and so callers
    that don't pre-approve still see the typed event.
    """

    def __init__(
        self,
        agent: "ClawAgent",
        *,
        tool_name: str,
        tool_description: str,
        custom_output_extractor: Optional[Callable[[AgentState], str]] = None,
        needs_approval: bool = False,
    ):
        self._agent = agent
        self.name = tool_name
        self.description = tool_description
        self._extractor = custom_output_extractor
        self._needs_approval = needs_approval
        self.parameters: Dict[str, Dict[str, Any]] = {
            "task": {
                "type": "string",
                "description": (
                    "The task or question to send to the wrapped agent. "
                    "The agent receives this as its initial user input."
                ),
                "required": True,
            }
        }

    async def execute(
        self,
        args: Dict[str, Any],
        run_context: Optional[RunContext] = None,
    ) -> ToolResult:
        task = str(args.get("task", "")).strip()
        if not task:
            return ToolResult(
                success=False,
                output="",
                error=f"{self.name}: missing required 'task' argument",
            )

        # Optional approval gate. We don't block forever — if there's no
        # standing decision and no permission_mode that would auto-allow,
        # we surface a typed event and require the caller to pre-approve
        # via run_context. That mirrors the loop's existing semantics.
        if self._needs_approval and run_context is not None:
            decision = run_context.is_tool_approved(self.name, tool_name=self.name)
            if decision is False:
                rec = run_context.get_approval(self.name, tool_name=self.name)
                reason = (rec.reason if rec else None) or "approval rejected"
                return ToolResult(
                    success=False,
                    output="",
                    error=f"{self.name}: {reason}",
                )
            if decision is None:
                # Notify via on_stream_event if the parent passed one in.
                on_stream = getattr(run_context, "_metadata", {}).get("on_stream_event")
                if callable(on_stream):
                    try:
                        from clawagents.stream_events import ApprovalRequiredEvent
                        on_stream(ApprovalRequiredEvent(
                            tool_name=self.name,
                            call_id=self.name,
                            args=dict(args),
                        ))
                    except Exception:
                        pass
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"{self.name}: approval required (use "
                        "run_context.approve_tool() to allow)"
                    ),
                )

        try:
            # Forward the parent's run context so the child inherits its
            # permission_mode (and approvals). Without this the child ran with a
            # fresh DEFAULT context, letting an agent-as-tool execute write/exec
            # tools while the parent was in plan mode — a plan-mode escape.
            child_state = await self._agent.invoke(task, run_context=run_context)
        except Exception as e:
            return ToolResult(
                success=False, output="", error=f"{self.name} raised: {e}"
            )

        if self._extractor is not None:
            try:
                extracted = self._extractor(child_state)
            except Exception as e:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"{self.name}: output extractor raised: {e}",
                )
            return ToolResult(success=True, output=str(extracted))

        return ToolResult(success=True, output=str(child_state.result))


def create_claw_agent(
    model: Union[str, LLMProvider, None] = None,
    profile: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    instruction: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List] = None,
    skills: Union[str, List[Union[str, os.PathLike]], None] = None,
    skills_exclude: Optional[List[str]] = None,
    fallback_models: Optional[List[str]] = None,
    memory: Union[str, List[Union[str, os.PathLike]], None] = None,
    sandbox: Any = None,
    sandbox_profile: str | None = None,
    chat_mode: str | None = None,
    allow_full_access: bool = False,
    streaming: bool = True,
    context_window: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    wire_api: Optional[str] = None,
    ssl_verify: Optional[bool] = None,
    use_native_tools: bool = True,
    on_event: Optional[OnEvent] = None,
    trajectory: Optional[bool] = None,
    rethink: Optional[bool] = None,
    learn: Optional[bool] = None,
    atlas: Optional[bool] = None,
    atlas_config: Optional[Any] = None,
    max_iterations: Optional[int] = None,
    preview_chars: Optional[int] = None,
    response_chars: Optional[int] = None,
    advisor_model: Union[str, LLMProvider, None] = None,
    advisor_api_key: Optional[str] = None,
    advisor_max_calls: Optional[int] = None,
    mcp_servers: Optional[List[Any]] = None,
    handoffs: Optional[List[Handoff]] = None,
    name: Optional[str] = None,
    tool_discovery: bool = True,
    tool_discovery_max_results: int = 25,
    tool_discovery_max_profile: str = "full",
    mode: Optional[str] = None,
    action_mode: str = "tools",
    approval_handler: Any = None,
    require_approval_tools: Optional[List[str]] = None,
    on_exit_plan_mode: Any = None,
    permission_rules: list | None = None,
    goal_mode: bool = False,
    features: Optional[dict[str, bool]] = None,
    workspace: Optional[Union[str, os.PathLike]] = None,
) -> ClawAgent:
    """
    Create a ClawAgent with full-stack capabilities.

    Args:
        model:          Model name ("gpt-5", "gemini-3-flash") or LLMProvider.
                        None = auto-detect from env.
        profile:        Optional named provider profile. Explicit model/api_key/base_url
                        args override profile values.
        api_key:        API key for the model provider. Auto-routed based on model name.
                        Falls back to env vars (OPENAI_API_KEY / GEMINI_API_KEY) if omitted.
        base_url:       Custom base URL for OpenAI-compatible APIs. Enables Azure OpenAI,
                        AWS Bedrock, Ollama, vLLM, LM Studio, or any OpenAI-compatible endpoint.
                        Default: from OPENAI_BASE_URL env / None (uses api.openai.com).
        api_version:    API version string. Required for Azure OpenAI (e.g. "2024-12-01-preview").
                        Default: from OPENAI_API_VERSION env / None.
        instruction:    What the agent should do / how it should behave (alias: ``system_prompt``).
        system_prompt:  Alias for ``instruction`` (class-style naming).
        tools:          Additional tools. Built-in tools always included.
        skills:         Skill directories (default: auto-discovers ./skills). Bundled skills (e.g. OpenViking) are included when present.
        memory:         AGENTS.md paths (default: auto-discovers ./AGENTS.md, ./CLAWAGENTS.md).
        streaming:      Enable streaming output (default: True).
        context_window:  Max context window in tokens (default: from CONTEXT_WINDOW env / 1000000).
        max_tokens:     Max output tokens per call (default: from MAX_TOKENS env / 8192).
        temperature:    Sampling temperature (default: from TEMPERATURE env / 0.0).
        reasoning_effort:
                        OpenAI reasoning effort for o-series / GPT-5.5+ models
                        (``none``|``low``|``medium``|``high``|``xhigh``|``max``).
                        UI aliases: Light→low, Extra High→xhigh. Empty = provider
                        default. On Responses API, effort is kept with tools.
        wire_api:       OpenAI transport: ``auto`` | ``responses`` | ``chat_completions``.
                        Use ``responses`` for Responses-only OpenAI-compatible
                        proxies (e.g. Codex gateways that 404 ``/chat/completions``).
        ssl_verify:     TLS verify for custom ``base_url`` hosts. Set ``False`` for
                        corporate proxies with private CAs.
        trajectory:     Enable trajectory logging to .clawagents/trajectories/.
                        Records every turn for debugging and analysis.
                        Default: from CLAW_TRAJECTORY env / False.
        rethink:        Enable consecutive-failure detection. Injects a "rethink"
                        message after 3 consecutive tool failures.
                        Default: from CLAW_RETHINK env / False.
        learn:          Enable Prompt-Time Reinforcement Learning (PTRL).
                        After each run the agent self-analyzes its trajectory,
                        extracts lessons, and stores them in .clawagents/lessons.md.
                        On subsequent runs lessons are injected into the system
                        prompt and into rethink messages so the agent improves
                        over time — without model fine-tuning.
                        Automatically enables trajectory when True.
                        Default: from CLAW_LEARN env / False.
        atlas:          Deprecated no-op (ATLAS removed in 6.16). Use ``goal_mode``.
        atlas_config:   Deprecated no-op.
        max_iterations: Max tool rounds before the agent stops.
                        Default: from MAX_ITERATIONS env / 200.
        preview_chars:  Max chars for tool-output previews in trajectory logs.
                        Default: from CLAW_PREVIEW_CHARS env / 120.
        response_chars: Max chars for LLM response text in trajectory logs.
                        Default: from CLAW_RESPONSE_CHARS env / 500.
        advisor_model:  A stronger model for strategic guidance (consulted 2-3 times per task).
                        Cross-provider supported — e.g. use "claude-opus-4-6" to advise "gpt-5.4-nano".
                        Default: from ADVISOR_MODEL env / None (disabled).
        advisor_api_key: API key for the advisor model (only needed if it's a different provider).
                        Default: from ADVISOR_API_KEY env / None.
        advisor_max_calls: Maximum advisor consultations per task.
                        Default: from ADVISOR_MAX_CALLS env / 3.
        fallback_models: Ordered list of model name strings to try when the primary
                        provider fails. Wraps the primary in a FallbackProvider.
                        Also read from CLAWAGENTS_FALLBACK_MODELS env var
                        (comma-separated). Priority is controlled by
                        CLAWAGENTS_PROVIDER_CONFIG_MODE:
                          "env_override" — env var takes precedence (default),
                          "default"      — constructor argument takes precedence,
                          "fallback"     — env var is last resort.
        tool_discovery: Register compact tool discovery helpers by default.
        tool_discovery_max_results: Default result cap for tool_discover.
        tool_discovery_max_profile: Maximum profile exposed through discovery helpers.
        sandbox_profile: Named OS sandbox profile (``workspace``, ``strict``,
                        ``seatbelt``, ``bwrap``, …). Default: ``CLAW_SANDBOX_PROFILE``
                        or ``workspace`` (path-confined; upgrades to seatbelt/bwrap
                        when available). Pass ``sandbox=`` to inject a custom backend.
        chat_mode:      Host UI mode (``ask`` / ``read_only`` / ``auto`` /
                        ``full_access``). Coupled to the OS sandbox: ``full_access``
                        with ``allow_full_access`` → profile ``off``; ``read_only``
                        → ``read-only``. Explicit ``sandbox_profile`` still wins.
        allow_full_access: Settings gate required before ``chat_mode=full_access``
                        disables the OS sandbox (VS Code / Desktop parity).
        permission_rules: Extra declarative allow/deny/ask rules (deny wins).
                        Defaults load from ``.clawagents/permissions.json`` when
                        ``permission_rules`` feature is on.
        goal_mode:      Inject GOAL autopilot instruction (``start_goal`` /
                        ``update_goal`` nudge). Goal tools register when
                        ``goal_autopilot`` is enabled.

    Examples:
        # Zero-config (uses env vars)
        agent = create_claw_agent()

        # Explicit model + key
        agent = create_claw_agent("gpt-5-mini", api_key="sk-...")

        # Azure OpenAI
        agent = create_claw_agent("gpt-4o", api_key="...",
            base_url="https://myresource.openai.azure.com/",
            api_version="2024-12-01-preview")

        # Local model (Ollama / vLLM / LM Studio)
        agent = create_claw_agent("llama3.1", base_url="http://localhost:11434/v1")

        # AWS Bedrock native (IAM / HIPAA) — Claude on Bedrock
        agent = create_claw_agent("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        # or: create_claw_agent(profile="bedrock")

        # AWS Bedrock via OpenAI-compatible gateway (BAG / LiteLLM)
        agent = create_claw_agent("anthropic.claude-v3",
            base_url="http://localhost:8080/v1", api_key="bedrock")

        # With PTRL learning enabled
        agent = create_claw_agent("gpt-5-mini", learn=True, rethink=True)

        # With trajectory logging + higher limits
        agent = create_claw_agent("gpt-5-mini", trajectory=True, max_iterations=200,
                                  preview_chars=500, response_chars=2000)

    Advanced hooks (set after creation):
        agent.before_tool = lambda name, args: name != "execute"
    """
    # ── Resolve opt-in flags ────────────────────────────────────────────
    resolved_instruction = _resolve_system_prompt(system_prompt, instruction)
    from clawagents.paths import resolve_workspace_root

    workspace_root = str(resolve_workspace_root(workspace))

    if trajectory is None:
        trajectory = os.environ.get("CLAW_TRAJECTORY", "").lower() in ("1", "true", "yes")
    if rethink is None:
        rethink = os.environ.get("CLAW_RETHINK", "").lower() in ("1", "true", "yes")
    if learn is None:
        learn = os.environ.get("CLAW_LEARN", "").lower() in ("1", "true", "yes")
    # ATLAS removed (6.16+): kwargs / CLAW_ATLAS are ignored.
    if atlas or atlas_config or os.environ.get("CLAW_ATLAS", "").lower() in ("1", "true", "yes"):
        warnings.warn(
            "ATLAS was removed in clawagents 6.16; use goal_mode / start_goal instead. "
            "atlas= and CLAW_ATLAS are ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    atlas = False
    atlas_config = None
    if learn:
        trajectory = True
    if max_iterations is None:
        raw = os.environ.get("MAX_ITERATIONS", "")
        max_iterations = int(raw) if raw.isdigit() else 200
    if preview_chars is None:
        raw = os.environ.get("CLAW_PREVIEW_CHARS", "")
        preview_chars = int(raw) if raw.isdigit() else 120
    if response_chars is None:
        raw = os.environ.get("CLAW_RESPONSE_CHARS", "")
        response_chars = int(raw) if raw.isdigit() else 500

    # ── Resolve context_window from config if not provided ──────────────
    if context_window is None:
        from clawagents.config.config import load_config as _lc
        context_window = _lc().context_window  # default: 1_000_000

    # ── Resolve optional provider profile before model construction ─────
    provider_hint: Optional[str] = None
    if profile:
        from clawagents.provider_profiles import resolve_provider_profile
        resolved_profile = resolve_provider_profile(
            profile,
            model=model if isinstance(model, str) else None,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
        )
        if isinstance(model, str) or model is None:
            model = resolved_profile.model
        api_key = resolved_profile.api_key
        base_url = resolved_profile.base_url
        api_version = resolved_profile.api_version
        # Declared profile provider drives routing/key fields (not re-inferred
        # solely from the model string — aliases like internal-claude-* need this).
        provider_hint = (resolved_profile.provider or "").strip() or None

    # ── Resolve model → LLMProvider ────────────────────────────────────
    llm = _resolve_model(
        model, streaming, api_key, context_window, max_tokens, temperature,
        base_url, api_version, reasoning_effort, wire_api, ssl_verify,
        provider=provider_hint,
    )

    # ── Resolve fallback providers ──────────────────────────────────────
    llm = _apply_fallback_providers(
        llm,
        fallback_models=fallback_models,
        streaming=streaming,
        context_window=context_window,
        max_tokens=max_tokens,
        temperature=temperature,
        on_event=on_event,
    )

    # ── Resolve advisor model ─────────────────────────────────────────
    resolved_advisor_llm: Optional[LLMProvider] = None
    _adv_max_raw = os.environ.get("ADVISOR_MAX_CALLS", "")
    resolved_advisor_max_calls = advisor_max_calls if advisor_max_calls is not None else (int(_adv_max_raw) if _adv_max_raw.isdigit() else 3)
    advisor_spec = advisor_model if advisor_model is not None else (os.environ.get("ADVISOR_MODEL") or None)
    if advisor_spec:
        adv_key = advisor_api_key or (os.environ.get("ADVISOR_API_KEY") or None)
        resolved_advisor_llm = _resolve_model(advisor_spec, streaming, adv_key, context_window)

    # ── Resolve sandbox backend ────────────────────────────────────────
    # Chat UI mode and seatbelt are one contract — see sandbox_profile_for_chat_mode.
    sandbox_profile_name = "off"
    if sandbox is not None:
        sb = sandbox
        sandbox_profile_name = str(
            getattr(getattr(sb, "_profile", None), "name", None)
            or getattr(sb, "kind", "custom")
        )
    else:
        from clawagents.sandbox.profiles import (
            resolve_sandbox,
            sandbox_profile_for_chat_mode,
        )

        env_profile = (os.environ.get("CLAW_SANDBOX_PROFILE") or "").strip() or None
        chosen = sandbox_profile_for_chat_mode(
            chat_mode,
            allow_full_access=bool(allow_full_access),
            explicit=sandbox_profile,
            env_profile=env_profile,
        )
        sb = resolve_sandbox(
            chosen,
            workspace=workspace_root,
            default="workspace",
        )
        sandbox_profile_name = str(
            getattr(getattr(sb, "_profile", None), "name", None)
            or chosen
            or "workspace"
        )

    registry = ToolRegistry()

    # Declarative permission rules (deny wins) — attach to registry metadata.
    from clawagents.tools.permissions import PermissionRule, load_permission_engine

    _perm_engine = load_permission_engine(workspace_root)
    if permission_rules:
        for row in permission_rules:
            if isinstance(row, PermissionRule):
                _perm_engine.add_rule(row)
            elif isinstance(row, dict):
                _perm_engine.add_rule(PermissionRule(**{
                    k: v for k, v in row.items()
                    if k in PermissionRule.__dataclass_fields__
                }))
    registry._permission_engine = _perm_engine  # type: ignore[attr-defined]

    # ── Built-in tools (lazy where possible) ─────────────────────────
    # Eager: cheap, no sandbox dependency
    from clawagents.tools.todolist import todolist_tools
    from clawagents.tools.think import think_tools
    from clawagents.tools.interactive import interactive_tools

    for tool in [*todolist_tools, *think_tools, *interactive_tools]:
        registry.register(tool)

    # Lazy: sandbox-backed tools — schema available immediately,
    # module import + sandbox init deferred to first execute()
    from clawagents.tools.registry import LazyTool

    def _make_lazy_sb_tool(name, desc, params, module_path, factory_fn, sb_ref=sb, keywords=None):
        """Create a LazyTool that calls a factory function with the sandbox on first use."""
        class _SbLazyTool(LazyTool):
            def __init__(self):
                super().__init__(name, desc, params, module_path, "", keywords)
            async def execute(self, args):
                if self._resolved is None:
                    import importlib
                    mod = importlib.import_module(module_path)
                    factory = getattr(mod, factory_fn)
                    tools = factory(sb_ref)
                    self._resolved = next(t for t in tools if t.name == name)
                return await self._resolved.execute(args)
        return _SbLazyTool()

    # Schema is copied from the concrete tool implementation so lazy/native
    # schemas cannot drift from the backing tools.
    from clawagents.tools.filesystem import create_filesystem_tools
    for spec in create_filesystem_tools(sb):
        registry.register(_make_lazy_sb_tool(
            spec.name, spec.description, spec.parameters,
            "clawagents.tools.filesystem", "create_filesystem_tools",
            keywords=getattr(spec, "keywords", []),
        ))

    from clawagents.tools.exec import create_exec_tools
    for spec in create_exec_tools(sb):
        registry.register(_make_lazy_sb_tool(
            spec.name, spec.description, spec.parameters,
            "clawagents.tools.exec", "create_exec_tools",
            keywords=getattr(spec, "keywords", []),
        ))

    from clawagents.tools.advanced_fs import create_advanced_fs_tools
    for spec in create_advanced_fs_tools(sb):
        registry.register(_make_lazy_sb_tool(
            spec.name, spec.description, spec.parameters,
            "clawagents.tools.advanced_fs", "create_advanced_fs_tools",
            keywords=getattr(spec, "keywords", []),
        ))

    def _make_lazy_web_tool(tool_name: str):
        class _LazyWebTool(LazyTool):
            def __init__(self):
                from clawagents.tools.web import web_tools as _wt
                spec = next(t for t in _wt if t.name == tool_name)
                super().__init__(
                    spec.name, spec.description, spec.parameters,
                    "clawagents.tools.web", "", getattr(spec, "keywords", []),
                )
            async def execute(self, args):
                if self._resolved is None:
                    from clawagents.tools.web import web_tools as _wt
                    self._resolved = next(t for t in _wt if t.name == tool_name)
                return await self._resolved.execute(args)
        return _LazyWebTool()

    for _web_name in ("web_fetch", "web_search"):
        registry.register(_make_lazy_web_tool(_web_name))

    # ── Adapt and register user-provided tools ─────────────────────────
    if tools:
        for tool in tools:
            if hasattr(tool, "ainvoke") and not hasattr(tool, "execute"):
                registry.register(LangChainToolAdapter(tool))
            else:
                registry.register(tool)

    # ── Auto-discover skills from default locations ─────────────────────
    skill_summaries: Optional[str] = None
    skill_store = None
    base_skill_dirs = _to_list(skills) if skills is not None else _auto_discover_skills(workspace_root)
    _bundled = _get_bundled_skills_dir()
    # Bundled skills go FIRST: SkillStore gives later directories precedence on
    # name collisions, so user/workspace skills must override bundled ones
    # (openclaw/deepagents precedence order), not the other way around.
    skill_dirs = ([_bundled] + base_skill_dirs) if (_bundled and os.path.isdir(_bundled)) else base_skill_dirs
    if _bundled:
        try:
            from clawagents.skills.best_of_n import ensure_best_of_n_skill

            ensure_best_of_n_skill(Path(_bundled))
        except Exception:
            pass
    if skill_dirs:
        from clawagents.tools.skills import SkillStore, create_skill_tools

        skill_store = SkillStore()
        for d in skill_dirs:
            if os.path.exists(str(d)):
                skill_store.add_directory(d)

        # Support non-main threads (Streamlit, Jupyter) where asyncio.run()
        # fails due to set_wakeup_fd. Reuse caller's loop if available.
        try:
            _loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, skill_store.load_all()).result()
        except RuntimeError:
            asyncio.run(skill_store.load_all())

        if skills_exclude:
            from clawagents.tools.skills import _norm_skill_key

            _excluded_keys = {_norm_skill_key(str(n)) for n in skills_exclude if n}
            for _name in list(skill_store.skills):
                if _norm_skill_key(_name) in _excluded_keys:
                    skill_store.skills.pop(_name, None)
            for _name in list(skill_store.ineligible):
                if _norm_skill_key(_name) in _excluded_keys:
                    skill_store.ineligible.pop(_name, None)

        loaded_skills = skill_store.list()
        if loaded_skills:
            # Initial catalog (no user turn yet); before_llm re-ranks each round.
            skill_summaries = _build_skill_catalog_prompt(
                loaded_skills,
                context_window=context_window,
            )

        for skill_tool in create_skill_tools(
            skill_store,
            relevance_scorer=_skill_relevance_score,
            available_tool_names=lambda: {tool.name for tool in registry.list()},
        ):
            # Both tools: thin catalog in-prompt; list_skills for overflow;
            # use_skill loads complete instructions in verified pages.
            if skill_tool.name in ("use_skill", "list_skills"):
                registry.register(skill_tool)

    from clawagents.tools.skill_workshop import create_skill_workshop_tool

    registry.register(
        create_skill_workshop_tool(
            workspace=workspace_root,
            on_reload=(skill_store.reload if skill_store is not None else None),
        )
    )

    from clawagents.tools.search_history import create_search_history_tool

    registry.register(create_search_history_tool(workspace=workspace_root))

    from clawagents.tools.retrieve_tool_result import create_retrieve_tool_result_tool

    registry.register(create_retrieve_tool_result_tool(workspace=workspace_root))

    from clawagents.tools.context_tools import create_context_tools

    for t in create_context_tools(workspace=workspace_root):
        registry.register(t)

    from clawagents.tools.git_tools import create_git_tools

    for t in create_git_tools(workspace=workspace_root):
        registry.register(t)

    from clawagents.tools.plan_mode import create_plan_mode_tools

    for t in create_plan_mode_tools(on_exit_plan_mode=on_exit_plan_mode):
        if registry.get(t.name) is None:
            registry.register(t)

    from clawagents.tools.worktree_tools import create_worktree_tools

    for t in create_worktree_tools(workspace=workspace_root):
        registry.register(t)

    from clawagents.tools.hunk_review import create_hunk_review_tools

    for t in create_hunk_review_tools(workspace=workspace_root):
        if registry.get(t.name) is None:
            registry.register(t)

    from clawagents.tools.marketplace_tools import create_marketplace_tools

    for t in create_marketplace_tools(workspace=workspace_root):
        if registry.get(t.name) is None:
            registry.register(t)

    from clawagents.config.features import is_enabled as _feat_on
    from clawagents.goal.tools import create_goal_tools

    # Goal tools + verifier only when UI/API explicitly enables Goal mode.
    # Otherwise Act/Plan turns still saw start_goal / final-gate verify because
    # goal_autopilot defaults on and an active `.clawagents/goal/state.json`
    # from a prior Goal run kept hijacking the loop.
    if goal_mode and _feat_on("goal_autopilot"):
        for t in create_goal_tools():
            if registry.get(t.name) is None:
                registry.register(t)

    # ── Auto-discover memory from default locations ────────────────────
    memory_paths = _to_list(memory) if memory is not None else _auto_discover_memory()
    composed_before_llm = _compose_before_llm(
        memory_paths=memory_paths,
        skill_summaries=skill_summaries,
        skill_store=skill_store,
        context_window=context_window,
    )

    # ── Custom mode (instruction + tool gate + permission) ────────────
    mode_before: Optional[BeforeToolHook] = None
    permission_mode_override = None
    if mode:
        from clawagents.modes import (
            get_mode,
            make_mode_before_tool,
            resolve_permission_mode,
        )

        agent_mode = get_mode(mode)
        if agent_mode is None:
            raise ValueError(f"Unknown agent mode: {mode!r}")
        if agent_mode.instruction:
            if resolved_instruction:
                resolved_instruction = (
                    agent_mode.instruction.rstrip() + "\n\n" + resolved_instruction
                )
            else:
                resolved_instruction = agent_mode.instruction
        mode_before = make_mode_before_tool(agent_mode)
        permission_mode_override = resolve_permission_mode(agent_mode)
        if agent_mode.auto_approve and approval_handler is None:
            approval_handler = None  # CI mode: no blocking approvals

    action_mode_norm = action_mode if action_mode in ("tools", "code") else "tools"
    if action_mode_norm == "code":
        from clawagents.graph.codeact import CODEACT_SYSTEM_ADDENDUM

        resolved_instruction = (
            (resolved_instruction or "") + "\n\n" + CODEACT_SYSTEM_ADDENDUM
        ).strip() or CODEACT_SYSTEM_ADDENDUM

    if goal_mode:
        _goal_nudge = (
            "You are in GOAL mode (planner→verify→strategist autopilot). For any "
            "multi-step objective, call `start_goal` first to write a verifier "
            "contract, then work the plan and report via `update_goal`. Do not "
            "claim the goal is done until the verifier accepts."
        )
        if resolved_instruction and str(resolved_instruction).strip():
            resolved_instruction = (
                _goal_nudge.rstrip() + "\n\n" + str(resolved_instruction).lstrip()
            )
        else:
            resolved_instruction = _goal_nudge

    # Wire permission "ask" → approval_handler when the host provides one.
    if approval_handler is not None and callable(approval_handler):
        def _perm_ask(tool_name: str, args: dict, message: str) -> bool:
            try:
                result = approval_handler(tool_name, args, "")
                if hasattr(result, "__await__"):
                    # Sync gate cannot await; fall through as deny.
                    return False
                return bool(result)
            except Exception:
                return False

        _perm_engine.ask_handler = _perm_ask

    # Model harness: efficiency suffix / base prompt (e.g. GPT-5.6 Luna).
    # clear_tool_* knobs are consumed in the agent loop; this wires the prompt.
    # (excluded_tools is reserved — ToolRegistry has no unregister API yet.)
    try:
        from clawagents.harness_profiles import (
            apply_harness_profile_to_prompt,
            resolve_harness_profile,
        )

        _model_id = (
            model
            if isinstance(model, str)
            else getattr(llm, "model", None) or getattr(model, "model", None)
        )
        _harness = resolve_harness_profile(
            str(_model_id) if _model_id else None
        )
        if _harness is not None:
            resolved_instruction = apply_harness_profile_to_prompt(
                resolved_instruction or "", _harness
            )
    except Exception:
        pass

    agent = ClawAgent(
        llm=llm, tools=registry, system_prompt=resolved_instruction,
        streaming=streaming, use_native_tools=use_native_tools,
        context_window=context_window, on_event=on_event,
        before_llm=composed_before_llm, trajectory=trajectory,
        rethink=rethink, learn=learn, atlas=False,
        atlas_config=None, max_iterations=max_iterations,
        preview_chars=preview_chars, response_chars=response_chars,
        features=features,
        workspace=workspace_root,
        advisor_llm=resolved_advisor_llm, advisor_max_calls=resolved_advisor_max_calls,
        handoffs=handoffs, name=name,
        action_mode=action_mode_norm,
        approval_handler=approval_handler,
        require_approval_tools=require_approval_tools,
    )
    agent.goal_mode = bool(goal_mode)
    if skill_store is not None:
        agent.skill_store = skill_store  # type: ignore[attr-defined]
    agent._permission_engine = _perm_engine  # type: ignore[attr-defined]
    agent._sandbox_backend = sb  # type: ignore[attr-defined]
    agent._sandbox_profile_name = sandbox_profile_name  # type: ignore[attr-defined]
    agent._chat_mode = str(chat_mode or "").strip().lower() or None  # type: ignore[attr-defined]
    agent._allow_full_access = bool(allow_full_access)  # type: ignore[attr-defined]

    # Compose permission deny-gate with mode before_tool (HookResult-aware).
    from clawagents.graph.agent_loop import HookResult
    from clawagents.modes import compose_before_tool

    def _perm_before(tool_name: str, args: dict):
        ok, msg = _perm_engine.gate(tool_name, args if isinstance(args, dict) else {})
        return HookResult(allowed=ok, reason=msg or ("denied" if not ok else ""))

    agent.before_tool = compose_before_tool(_perm_before, mode_before)
    if permission_mode_override is not None:
        agent._default_permission_mode = permission_mode_override  # type: ignore[attr-defined]
    elif agent._chat_mode in ("read_only", "readonly", "plan"):
        # UI Plan tab → engine PLAN (Grok Build: explore + plan.md only).
        from clawagents.permissions.mode import PermissionMode as _PM

        agent._default_permission_mode = _PM.PLAN  # type: ignore[attr-defined]

    from clawagents.tools.background_task import create_background_task_tools
    for task_tool in create_background_task_tools():
        registry.register(task_tool)

    from clawagents.tools.tool_program import create_tool_program_tool
    registry.register(create_tool_program_tool(registry))

    # ── Sub-agent tool (always available) ──────────────────────────────
    from clawagents.tools.subagent import create_task_tool
    registry.register(create_task_tool(llm, registry, workspace=workspace_root))

    # v6.17: PTY sessions + rewind tools
    try:
        from clawagents.config.features import is_enabled as _feat617
        if _feat617("pty_sessions"):
            from clawagents.tools.pty_session import create_pty_tools
            for t in create_pty_tools():
                if registry.get(t.name) is None:
                    registry.register(t)
        if _feat617("session_rewind"):
            from clawagents.memory.hunk_watcher import create_rewind_tools, get_watcher
            for t in create_rewind_tools():
                if registry.get(t.name) is None:
                    registry.register(t)
            try:
                get_watcher(workspace_root).start()
            except Exception:
                pass
        if _feat617("hashline_tools"):
            from clawagents.tools.hashline import create_hashline_tools
            for t in create_hashline_tools(sb):
                if registry.get(t.name) is None:
                    registry.register(t)
    except Exception:
        pass

    # v6.17: hybrid smart-memory recall
    try:
        from clawagents.config.features import is_enabled as _feat_mem
        if _feat_mem("smart_memory") or _feat_mem("hybrid_memory_search"):
            from clawagents.tools.memory_search import create_memory_search_tool
            if registry.get("memory_search") is None:
                registry.register(create_memory_search_tool(workspace=workspace_root))
    except Exception:
        pass

    # ── MCP server integration (v6.4, optional) ────────────────────────
    if mcp_servers:
        from clawagents.mcp import (
            MCPServerManager,
            is_mcp_sdk_available,
        )
        if not is_mcp_sdk_available():
            raise ImportError(
                "create_claw_agent received mcp_servers= but the optional "
                "MCP SDK is not installed. Install with: pip install 'clawagents[mcp]' "
                "(or directly: pip install mcp)."
            )

        manager = MCPServerManager(mcp_servers)
        # Start the manager: connect each server, list tools, register them.
        # We run it in a fresh event loop the same way skill loading does, so
        # the call works from sync, Streamlit, and Jupyter contexts.
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, manager.start(registry)).result()
        except RuntimeError:
            asyncio.run(manager.start(registry))

        # Stash the manager on the agent so callers can shut it down. We also
        # register an atexit finaliser as a backstop for short-lived scripts.
        agent.mcp_manager = manager  # type: ignore[attr-defined]
        from clawagents.tools.mcp_auth import MCPAuthTool
        registry.register(MCPAuthTool(manager))
        import atexit

        def _shutdown_mcp():  # pragma: no cover — best-effort process exit hook
            try:
                asyncio.get_running_loop()
                # Already in a loop — let the user shut down explicitly.
                return
            except RuntimeError:
                try:
                    asyncio.run(manager.shutdown())
                except Exception:
                    pass

        atexit.register(_shutdown_mcp)

    # Compact discovery is registered last so it can see user, subagent, and MCP tools.
    if tool_discovery:
        from clawagents.tools.catalog import create_tool_discovery_tools
        for discovery_tool in create_tool_discovery_tools(
            registry,
            max_results=tool_discovery_max_results,
            max_profile=tool_discovery_max_profile,  # type: ignore[arg-type]
        ):
            if registry.get(discovery_tool.name) is None:
                registry.register(discovery_tool)

    # GPT-5.6 / Luna: shrink the advertised tool surface; optional groups stay
    # registered and unlock via activate_tool_group (see tool_groups.py).
    try:
        from clawagents.harness_profiles import resolve_harness_profile
        from clawagents.tools.tool_groups import (
            ActivateToolGroupTool,
            apply_mode_active_profile,
        )

        _mid = (
            model
            if isinstance(model, str)
            else getattr(llm, "model", None) or getattr(model, "model", None)
        )
        _hp = resolve_harness_profile(str(_mid) if _mid else None)
        if _hp is not None and _hp.name == "openai-gpt56":
            if registry.get("activate_tool_group") is None:
                registry.register(ActivateToolGroupTool(registry))
            apply_mode_active_profile(
                registry,
                chat_mode=str(chat_mode or "").strip().lower() or None,
                goal_mode=bool(goal_mode),
            )
    except Exception:
        pass

    return agent


# ─── Internal Helpers ─────────────────────────────────────────────────────

def _resolve_model(
    model: Union[str, LLMProvider, None],
    streaming: bool,
    api_key: Optional[str] = None,
    context_window: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    wire_api: Optional[str] = None,
    ssl_verify: Optional[bool] = None,
    provider: Optional[str] = None,
) -> LLMProvider:
    """Accept a model name string, an LLMProvider, or None (auto-detect)."""
    if isinstance(model, LLMProvider):
        return model

    from clawagents.config.config import load_config, get_default_model
    from clawagents.providers.llm import create_provider, normalize_reasoning_effort, _normalize_wire_api
    from clawagents.providers.model_classify import (
        api_key_field_for,
        classify_model,
        parse_model_ref,
    )

    config = load_config()
    config.streaming = streaming
    if context_window is not None:
        config.context_window = context_window
    if max_tokens is not None:
        config.max_tokens = max_tokens
    if temperature is not None:
        config.temperature = temperature
    if base_url is not None:
        config.openai_base_url = base_url
    if api_version is not None:
        config.openai_api_version = api_version
    if reasoning_effort is not None:
        config.reasoning_effort = normalize_reasoning_effort(reasoning_effort) or ""
    if wire_api is not None:
        config.openai_wire_api = _normalize_wire_api(wire_api)
    if ssl_verify is not None:
        config.openai_ssl_verify = bool(ssl_verify)

    active_model = model if isinstance(model, str) and model else get_default_model(config)
    kind = classify_model(
        active_model,
        base_url=config.openai_base_url,
        provider_hint=provider,
    )

    # Override the appropriate API key if provided — driven by the classifier
    # (and optional profile provider hint), not ad-hoc startswith checks.
    if api_key:
        field = api_key_field_for(kind, base_url=config.openai_base_url)
        if field == "gemini_api_key":
            config.gemini_api_key = api_key
        elif field == "anthropic_api_key":
            config.anthropic_api_key = api_key
        elif field == "openai_api_key":
            config.openai_api_key = api_key
        # field is None → native Bedrock IAM; leave key fields alone.

    # Strip litellm ``provider/`` before the factory (SDK must not see it).
    sdk_name = parse_model_ref(active_model).bare_id
    return create_provider(sdk_name, config, provider_hint=provider or kind)


def _resolve_system_prompt(
    system_prompt: Optional[str],
    instruction: Optional[str],
) -> Optional[str]:
    """Resolve ``system_prompt`` / ``instruction`` aliases (mutually compatible)."""
    if system_prompt is not None and instruction is not None and system_prompt != instruction:
        raise ValueError(
            "Cannot specify both system_prompt and instruction with different values"
        )
    if system_prompt is not None:
        return system_prompt
    return instruction


def _to_list(value) -> list:
    """Convert None, string, or list to a list."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    return list(value)


def _get_bundled_skills_dir() -> str:
    """Path to bundled skills (e.g. openviking)."""
    return str(Path(__file__).resolve().parent / "skills")


# Skill catalog: progressive disclosure (Claude Code / Codex pattern).
# Metadata only in-prompt; full bodies via use_skill. Budget scales with
# context window (~1.5%), with floor/ceiling so tiny and huge windows stay sane.
SKILL_LISTING_BUDGET_FRACTION = 0.015
SKILL_LISTING_BUDGET_FLOOR_CHARS = 4000
SKILL_LISTING_BUDGET_CEILING_CHARS = 16000
SKILL_LISTING_MAX_DESC_CHARS = 400
SKILL_LISTING_CHARS_PER_TOKEN = 4
# Soft cap on how many skills appear with descriptions before name-only overflow.
MAX_SKILLS_IN_PROMPT = 80
SKILL_CONFIDENT_MATCH_SCORE = 12.0
SKILL_POSSIBLE_MATCH_SCORE = 3.0
SKILL_RELEVANT_MAX_CANDIDATES = 12
# Back-compat alias (older tests / callers).
MAX_SKILLS_PROMPT_CHARS = SKILL_LISTING_BUDGET_FLOOR_CHARS

_SKILL_CATALOG_HEADER = (
    "## Available Skills\n"
    "If any skill below matches the user's task, call `use_skill` with that "
    "skill's exact name **before** improvising a multi-step workflow "
    "(project startup, cohort/SQL extraction, document formats, etc.). "
    "Load the minimal applicable set; multi-artifact tasks may require more than one. "
    "Call `list_skills` only if this list was truncated or you need a skill "
    "not shown.\n"
)


def _effective_listing_context_window(context_window: int | None) -> int:
    """Clamp config context_window for listing math.

    EngineConfig often defaults to 1_000_000 (soft ceiling), which would make
    a %-budget enormous. Cap at 256k for listing; floor at 32k.
    """
    cw = int(context_window or 128_000)
    return max(32_000, min(cw, 256_000))


def skill_listing_budget_chars(context_window: int | None = None) -> int:
    """Character budget for the in-prompt skills catalog.

    Overrides:
      CLAW_SKILL_LISTING_CHAR_BUDGET — absolute char budget
      CLAW_SKILL_LISTING_BUDGET_FRACTION — fraction of context (default 0.015)
    """
    fixed = (os.environ.get("CLAW_SKILL_LISTING_CHAR_BUDGET") or "").strip()
    if fixed.isdigit():
        return max(1000, int(fixed))
    frac_raw = (os.environ.get("CLAW_SKILL_LISTING_BUDGET_FRACTION") or "").strip()
    try:
        frac = float(frac_raw) if frac_raw else SKILL_LISTING_BUDGET_FRACTION
    except ValueError:
        frac = SKILL_LISTING_BUDGET_FRACTION
    frac = max(0.005, min(frac, 0.05))
    cw = _effective_listing_context_window(context_window)
    dynamic = int(cw * frac * SKILL_LISTING_CHARS_PER_TOKEN)
    return max(
        SKILL_LISTING_BUDGET_FLOOR_CHARS,
        min(SKILL_LISTING_BUDGET_CEILING_CHARS, dynamic),
    )


def _clamp_skill_description(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 1].rstrip() + "…"


def _format_skill_line(
    name: str,
    description: str,
    desc_cap: int,
    *,
    when_to_use: str = "",
) -> str:
    from clawagents.config.features import is_enabled
    from clawagents.skills.strategy import format_skill_catalog_line

    if is_enabled("skill_when_to_use") and when_to_use:
        return format_skill_catalog_line(
            name,
            description,
            when_to_use=when_to_use,
            desc_cap=desc_cap if desc_cap > 0 else 10_000,
        )
    desc = _clamp_skill_description(description, desc_cap)
    if desc:
        return f"- **{name}**: {desc}"
    return f"- **{name}**"


def _compact_skill_name_index(skills: list, max_chars: int = 2400) -> str:
    names = sorted(
        {
            str(getattr(skill, "name", "") or "").strip()
            for skill in skills
            if str(getattr(skill, "name", "") or "").strip()
        },
        key=str.casefold,
    )
    prefix = "Catalog names: "
    shown: list[str] = []
    for name in names:
        candidate = prefix + ", ".join([*shown, name])
        if len(candidate) > max_chars:
            break
        shown.append(name)
    text = prefix + ", ".join(shown)
    if len(shown) < len(names):
        text += f" (+{len(names) - len(shown)}; use list_skills)"
    return text


def _latest_user_text(messages: list) -> str:
    """Extract current intent, carrying prior substance through short follow-ups."""
    user_texts: list[str] = []
    for message in reversed(messages or []):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "user":
            continue
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, str):
            text = content
            if text.strip():
                user_texts.append(text)
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
            text = "\n".join(p for p in parts if p)
            if text.strip():
                user_texts.append(text)
        if len(user_texts) >= 4:
            break
    if not user_texts:
        return ""
    current = user_texts[0]
    if len(_skill_tokens(current)) >= 2:
        return current
    for prior in user_texts[1:]:
        if len(_skill_tokens(prior)) >= 2:
            return f"{current}\nPrior substantive request: {prior[-4000:]}"
    return current


_SKILL_STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "your", "have",
    "please", "using", "into", "based", "update", "first", "then", "when",
    "help", "want", "need", "make", "could", "would", "should",
    "can", "you", "me", "our", "we", "us", "about", "it",
    "yes", "do", "continue", "thanks", "thank", "okay", "ok", "good", "morning",
    "how", "are",
}


def _skill_stem(token: str) -> str:
    """Small dependency-free stemmer for retrieval, not linguistic display."""
    value = token.lower()
    irregular = {
        "analysis": "analys",
        "analyses": "analys",
        "analyze": "analys",
        "analyzing": "analys",
    }
    if value in irregular:
        return irregular[value]
    if len(value) > 6 and value.endswith("ation"):
        return value[:-5]
    if len(value) > 5 and value.endswith("ating"):
        return value[:-5]
    if len(value) > 4 and value.endswith("ate"):
        return value[:-3]
    if len(value) > 5 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
        if len(value) > 3 and value[-1:] == value[-2:-1]:
            value = value[:-1]
        return value
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith(("ses", "xes", "zes", "ches", "shes")):
        return value[:-2]
    if len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    if len(value) > 5 and value.endswith("up"):
        return value[:-2]
    return value


def _skill_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    out: set[str] = set()
    for token in re.findall(r"[a-z0-9+#]{2,}", normalized):
        if token in _SKILL_STOP_WORDS:
            continue
        out.add(token)
        stem = _skill_stem(token)
        out.add(stem)
        if token.endswith("ing") and stem and not stem.endswith("e"):
            out.add(stem + "e")
        if token.endswith("ed") and stem and not stem.endswith("e"):
            out.add(stem + "e")
        if token.endswith("ation") and stem:
            out.add(stem + "ate")
        if len(token) > 5 and token.endswith("up"):
            out.add(token[:-2])
    return out


def _skill_core_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return {
        _skill_stem(token)
        for token in re.findall(r"[a-z0-9+#]{2,}", normalized)
        if token not in _SKILL_STOP_WORDS
    }


def _skill_relevance_score(skill: Any, query: str) -> float:
    """High-recall local score using structure, phrases, and token coverage."""
    if not query:
        return 0.0
    q = unicodedata.normalize("NFKC", query).casefold()
    name = str(getattr(skill, "name", "") or "")
    desc = str(getattr(skill, "description", "") or "")
    path = str(getattr(skill, "path", "") or "")
    name_l = name.lower()
    desc_l = desc.lower()
    path_l = path.lower()
    score = 0.0

    def _phrases(field: str) -> list[str]:
        values = getattr(skill, field, []) or []
        return [str(value).strip().lower() for value in values if str(value).strip()]

    tokens = _skill_tokens(q)
    name_tokens = re.findall(r"[a-z0-9]+", name_l.replace("_", " ").replace("-", " "))
    name_pattern = (
        r"(?<![a-z0-9])"
        + r"[\s_-]+".join(re.escape(token) for token in name_tokens)
        + r"(?![a-z0-9])"
        if name_tokens
        else r"(?!)"
    )
    explicit_name = bool(re.search(name_pattern, q))

    aliases = _phrases("aliases")
    triggers = _phrases("triggers")
    anti_triggers = _phrases("anti_triggers")
    when_to_use = str(getattr(skill, "when_to_use", "") or "").strip().lower()

    def _phrase_matches(phrase: str) -> bool:
        phrase_tokens = _skill_core_tokens(phrase)
        return bool(phrase_tokens) and phrase_tokens <= _skill_core_tokens(q)

    def _phrase_is_negated(phrase: str) -> bool:
        words = re.findall(r"[a-z0-9]+", phrase.casefold())
        if not words:
            return False
        return bool(
            re.search(
                rf"\b(?:not|never|avoid|without|do\s+not|don't)\b"
                rf"(?:\W+\w+){{0,3}}\W+{re.escape(words[0])}\b",
                q,
            )
        )

    if not explicit_name and any(
        _phrase_matches(phrase) and not _phrase_is_negated(phrase)
        for phrase in anti_triggers
    ):
        return -160.0
    for phrase in aliases:
        if _phrase_matches(phrase):
            score += 80.0
    for phrase in triggers:
        if _phrase_matches(phrase):
            score += 70.0
    if when_to_use and _phrase_matches(when_to_use):
        score += 75.0
    elif when_to_use:
        when_tokens = _skill_core_tokens(when_to_use)
        when_overlap = _skill_core_tokens(q) & when_tokens
        score += 12.0 * len(when_overlap)

    if explicit_name:
        score += 120.0

    name_token_set = _skill_core_tokens(name_l.replace("_", " ").replace("-", " "))
    query_core_tokens = _skill_core_tokens(q)
    name_overlap = query_core_tokens & name_token_set
    score += 14.0 * len(name_overlap)
    if name_token_set and name_token_set <= query_core_tokens:
        score += 35.0

    query_words = re.findall(r"[a-z0-9+#]{4,}", q)
    searchable_words = re.findall(
        r"[a-z0-9+#]{4,}",
        " ".join([name_l, *aliases, *triggers, when_to_use]),
    )
    fuzzy = max(
        (
            difflib.SequenceMatcher(None, query_word, candidate).ratio()
            for query_word in query_words
            for candidate in searchable_words
        ),
        default=0.0,
    )
    if fuzzy >= 0.82:
        score += 8.0

    # Filename / folder stem matches (e.g. new_project_starting_instruction.md)
    path_obj = Path(path) if path else None
    stem = ""
    if path_obj is not None:
        if path_obj.name.lower() == "skill.md":
            stem = path_obj.parent.name.lower()
        else:
            stem = path_obj.stem.lower()
    if stem and len(stem) >= 4 and stem not in {"skill", "readme", "skills"}:
        stem_phrase = stem.replace("_", " ").replace("-", " ")
        stem_tokens = _skill_tokens(stem_phrase)
        if stem_tokens and stem_tokens <= tokens:
            score += 60.0
        elif any(tok in _skill_stem(stem) for tok in tokens if len(tok) >= 5):
            score += 8.0

    desc_tokens = _skill_tokens(desc_l)
    path_tokens = _skill_tokens(path_l.replace("_", " ").replace("-", " "))
    desc_overlap = tokens & desc_tokens
    path_overlap = tokens & path_tokens
    generic = {"workflow", "project", "file", "tool", "task", "analysis", "data", "general"}
    informative_overlap = desc_overlap - generic
    score += 6.0 * len(informative_overlap) + 1.0 * len(desc_overlap & generic)
    score += 2.0 * len(path_overlap)
    if tokens and desc_tokens:
        coverage = len(informative_overlap) / min(len(tokens), len(desc_tokens))
        if coverage >= 0.75:
            score += 18.0
        elif coverage >= 0.5:
            score += 10.0
    return score


def _rank_skills_for_query(loaded_skills: list, query: str | None) -> list:
    if not loaded_skills:
        return []
    if not (query or "").strip():
        return sorted(loaded_skills, key=lambda s: str(getattr(s, "name", "")).lower())
    scored = [(_skill_relevance_score(s, query or ""), s) for s in loaded_skills]
    scored.sort(key=lambda pair: (-pair[0], str(getattr(pair[1], "name", "")).lower()))
    return [s for _, s in scored]


def _build_skill_catalog_prompt(
    loaded_skills: list,
    *,
    context_window: int | None = None,
    query: str | None = None,
    touched_paths: list[str] | None = None,
) -> str:
    """Build a bounded name/description catalog (Claude/Codex-style).

    Strategy (quality-first):
    1. Rank with exact names, structured routing metadata, and token coverage.
    2. Include every confident match up to a generous ambiguity cap.
    3. Surface possible matches or an explicit discovery fallback instead of
       silently hiding the skill system.
    4. Treat the character budget as a safety ceiling, never the objective.
    """
    if not loaded_skills:
        return ""

    from clawagents.skills.strategy import auto_suggest_lines, filter_skills_for_catalog

    gated = filter_skills_for_catalog(loaded_skills, touched_paths=touched_paths)
    if not gated:
        return ""

    ranked = _rank_skills_for_query(gated, query)
    max_desc = SKILL_LISTING_MAX_DESC_CHARS
    env_desc = (os.environ.get("CLAW_SKILL_LISTING_MAX_DESC_CHARS") or "").strip()
    if env_desc.isdigit():
        max_desc = max(40, int(env_desc))

    budget = skill_listing_budget_chars(context_window)
    name_index = _compact_skill_name_index(
        gated,
        max_chars=max(800, min(2400, budget // 2)),
    )

    def _line(skill: Any, desc_cap: int) -> str:
        return _format_skill_line(
            str(getattr(skill, "name", "") or "skill"),
            str(getattr(skill, "description", "") or ""),
            desc_cap,
            when_to_use=str(getattr(skill, "when_to_use", "") or ""),
        )

    # Normal LLM rounds have a current user query. Prefer recall over saving a
    # small prompt suffix: include all confident candidates up to a generous
    # cap, and preserve a discovery route when confidence is low.
    if query and query.strip():
        scored = [
            (skill, _skill_relevance_score(skill, query))
            for skill in ranked
        ]
        confident = [pair for pair in scored if pair[1] >= SKILL_CONFIDENT_MATCH_SCORE]
        possible = [pair for pair in scored if pair[1] >= SKILL_POSSIBLE_MATCH_SCORE]
        meaningful_query = len(_skill_tokens(query)) >= 2
        if confident:
            relevant = list(confident[:SKILL_RELEVANT_MAX_CANDIDATES])
            selected_ids = {id(skill) for skill, _score in relevant}
            for candidate in possible:
                if len(relevant) >= SKILL_RELEVANT_MAX_CANDIDATES:
                    break
                if id(candidate[0]) not in selected_ids:
                    relevant.append(candidate)
                    selected_ids.add(id(candidate[0]))
            heading = "### Relevant skills for this turn"
        elif possible and meaningful_query:
            relevant = possible[: min(6, SKILL_RELEVANT_MAX_CANDIDATES)]
            heading = "### Possible skill matches — inspect before choosing"
        elif meaningful_query:
            return (
                "## Skill Discovery\n"
                "No confident catalog match was found. Before improvising a specialized "
                "multi-step workflow, call `list_skills` with a focused query; its ranked "
                "search covers aliases, triggers, descriptions, and inflected terms.\n"
                + name_index
            )
        else:
            return ""

        lines = [_SKILL_CATALOG_HEADER.rstrip(), name_index, heading]
        for skill, _score in relevant:
            lines.append(_line(skill, min(max_desc, 240)))
        for nudge in auto_suggest_lines(relevant):
            lines.append(nudge)
        lines.append(
            "Choose by task fit, not rank alone. Call `use_skill` and read every page "
            "before acting."
        )
        all_matches = possible
        if len(all_matches) > len(relevant):
            lines.append(
                f"{len(all_matches) - len(relevant)} additional matches are "
                "available through `list_skills` with the same query."
            )
        text = "\n".join(lines)
        if len(text) > budget:
            compact = [_SKILL_CATALOG_HEADER.rstrip(), name_index, heading]
            for skill, _score in relevant:
                line = _line(skill, 100)
                if len("\n".join([*compact, line])) > budget - 160:
                    break
                compact.append(line)
            compact.append("More matches: call `list_skills` with the user's task as query.")
            text = "\n".join(compact)
        return text

    footer_full = "\n\n({n} more skills available — use list_skills to see all)"
    # Reserve space for a worst-case overflow footer.
    reserve = len(footer_full.format(n=len(ranked))) + 8
    body_budget = max(200, budget - len(_SKILL_CATALOG_HEADER) - reserve)

    entries = [
        (
            str(getattr(s, "name", "") or "skill"),
            str(getattr(s, "description", "") or ""),
            str(getattr(s, "when_to_use", "") or ""),
        )
        for s in ranked
    ]
    entries = [(n, d, w) for n, d, w in entries if n.strip()]
    if not entries:
        return ""

    header = _SKILL_CATALOG_HEADER
    body_budget = max(200, budget - len(header) - reserve)

    # Try progressively shorter descriptions until everything fits, else pack
    # as many name(+desc) lines as fit under body_budget.
    desc_caps = [max_desc, 220, 140, 80, 40, 0]
    shown: list[str] = []
    omitted = 0
    for desc_cap in desc_caps:
        shown = []
        used = 0
        omitted = 0
        for i, (name, desc, when) in enumerate(entries):
            if i >= MAX_SKILLS_IN_PROMPT and desc_cap > 0:
                # Beyond soft count: name-only for the rest of this pass.
                line = _format_skill_line(name, desc, 0, when_to_use=when)
            else:
                line = _format_skill_line(name, desc, desc_cap, when_to_use=when)
            add = len(line) + (1 if shown else 0)
            if used + add > body_budget:
                omitted = len(entries) - len(shown)
                break
            shown.append(line)
            used += add
        else:
            omitted = 0
        if omitted == 0:
            break

    text = header + "\n".join(shown)
    if omitted:
        text += f"\n\n({omitted} more skills available — use list_skills to see all)"
    if len(text) > budget:
        text = text[: max(0, budget - 80)].rstrip() + (
            "\n\n...(skill list truncated — use list_skills to see all)"
        )
    return text


_DEFAULT_MEMORY_FILES = ["AGENTS.md", "CLAWAGENTS.md", "CLAUDE.md"]
_DEFAULT_SKILL_DIRS = ["skills", ".skills", "skill", ".skill", "Skills"]


def _auto_discover_memory() -> list:
    """Auto-discover memory + always-on rules files."""
    from clawagents.memory.rules import discover_rule_paths

    return [str(p) for p in discover_rule_paths()]


def _auto_discover_skills(workspace: str | os.PathLike | None = None) -> list:
    """Auto-discover skill directories in common locations.

    Returned lowest-precedence first (SkillStore gives later dirs precedence):
    personal skill homes, then workspace dirs. Personal homes
    (`~/.clawagents/skills`, `~/.agents/skills`) are opt-in via
    ``CLAW_USER_SKILL_HOMES=1`` so library consumers and tests stay hermetic;
    the VS Code extension resolves homes itself and passes them explicitly.
    """
    from clawagents.paths import resolve_workspace_root

    ws = str(resolve_workspace_root(workspace))
    found = []
    if (os.environ.get("CLAW_USER_SKILL_HOMES") or "").strip() == "1":
        from clawagents.paths import get_clawagents_home

        home_candidates = [
            str(get_clawagents_home(create=False) / "skills"),
            os.path.expanduser("~/.agents/skills"),
        ]
        for path in home_candidates:
            if os.path.isdir(path):
                found.append(path)
    for name in _DEFAULT_SKILL_DIRS:
        path = os.path.join(ws, name)
        if os.path.isdir(path):
            found.append(path)
    return found


def _apply_fallback_providers(
    primary: LLMProvider,
    fallback_models: Optional[List[str]],
    streaming: bool,
    context_window: Optional[int],
    max_tokens: Optional[int],
    temperature: Optional[float],
    on_event: Any,
) -> LLMProvider:
    """Resolve fallback model list respecting CLAWAGENTS_PROVIDER_CONFIG_MODE,
    build fallback LLMProvider instances, and wrap *primary* in a FallbackProvider.

    CLAWAGENTS_PROVIDER_CONFIG_MODE values:
      "env_override"  — env var CLAWAGENTS_FALLBACK_MODELS takes precedence (default)
      "default"       — constructor argument takes precedence
      "fallback"      — env var is last resort (appended after constructor list)
    """
    env_raw = os.environ.get("CLAWAGENTS_FALLBACK_MODELS", "")
    env_models: List[str] = [m.strip() for m in env_raw.split(",") if m.strip()] if env_raw else []
    config_mode = os.environ.get("CLAWAGENTS_PROVIDER_CONFIG_MODE", "env_override").lower()

    if config_mode == "env_override":
        resolved_fallbacks = env_models or (fallback_models or [])
    elif config_mode == "default":
        resolved_fallbacks = fallback_models or env_models
    else:  # "fallback"
        resolved_fallbacks = (fallback_models or []) + [m for m in env_models if m not in (fallback_models or [])]

    if not resolved_fallbacks:
        return primary

    from clawagents.providers.fallback import FallbackProvider

    fallback_providers: List[LLMProvider] = [
        _resolve_model(m, streaming, None, context_window, max_tokens, temperature, None, None)
        for m in resolved_fallbacks
    ]

    return FallbackProvider(
        primary=primary,
        fallbacks=fallback_providers,
        on_event=on_event,
    )


def _compose_before_llm(
    memory_paths: list,
    skill_summaries: Optional[str],
    skill_store: Any = None,
    context_window: Optional[int] = None,
) -> Optional[BeforeLLMHook]:
    """Compose memory/rules + skill injection into one before_llm hook.

    Reloads rule files every LLM round so always-on rules survive compaction.
    Re-ranks the skill catalog against the latest user turn when a store is set.
    """
    from clawagents.prompts import append_prompt_injection, build_prompt_injection

    def hook(messages: list) -> list:
        memory_content: Optional[str] = None
        if memory_paths:
            from clawagents.memory.rules import load_rules_text

            # Prefer rules loader (budget + CLAUDE.md / .clawagents/rules)
            memory_content = load_rules_text(paths=memory_paths)
            if memory_content is None:
                from clawagents.memory.loader import load_memory_files

                memory_content = load_memory_files(memory_paths)

        summaries = skill_summaries
        discovery = ""
        if skill_store is not None:
            try:
                skill_store.maybe_hot_reload()
            except Exception:
                pass
            loaded = skill_store.list()
            if loaded:
                touched = list(getattr(skill_store, "session_touched_paths", []) or [])
                summaries = _build_skill_catalog_prompt(
                    loaded,
                    context_window=context_window,
                    query=_latest_user_text(messages),
                    touched_paths=touched,
                )
            try:
                discovery = skill_store.consume_discovery_announcement() or ""
            except Exception:
                discovery = ""

        if discovery and summaries:
            summaries = discovery + "\n\n" + summaries
        elif discovery:
            summaries = discovery

        if not memory_content and not summaries:
            return messages
        injection = build_prompt_injection(memory_content, summaries)
        return list(append_prompt_injection(messages, injection))

    # Always return a hook when we have paths or skills so rounds re-read disk.
    if memory_paths or skill_summaries or skill_store is not None:
        return hook
    return None
