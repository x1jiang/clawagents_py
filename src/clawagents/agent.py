import os
import asyncio
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
        max_iterations: int = 200,
        preview_chars: int = 120,
        response_chars: int = 500,
        features: Optional[dict[str, bool]] = None,
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
    ):
        """
        Initialize a ClawAgent.

        Args:
            llm: The initialized LLM provider
            tools: The registry containing all available tools
            system_prompt: Optional base system instruction
            streaming: Whether to stream responses from the LLM
            use_native_tools: Instruct the LLM to use native structured tool calls (if supported)
            context_window: Maximum allowed tokens before oldest messages are compacted
            trajectory: Whether to log full trajectory data
            rethink: Enables logic to backtrack on consecutive execution failures
            learn: Whether to use post-trajectory reflection to extract permanent lessons
            max_iterations: Maximum loop turns before returning early
            preview_chars: Number of characters to log in console output for tool results
            response_chars: Number of characters to log from LLM free-text response
            features: Dictionary to override global architectural variables (e.g. {"micro_compact": False, "wal": True})
            advisor_llm: Optional stronger model for strategic guidance (consulted 2-3 times per task)
            advisor_max_calls: Maximum advisor consultations per task (default: 3)
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
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
        self.max_iterations = max_iterations
        self.preview_chars = preview_chars
        self.response_chars = response_chars
        self.features = features
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
    ) -> AgentState:
        """Start the ReAct agent loop for ``task``.

        All per-call keyword arguments (``hooks``, ``input_guardrails``,
        ``output_type``, ``session``, ``on_stream_event`` …) override the
        values supplied to ``ClawAgent.__init__`` for just this invocation.
        """
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
        )

    # ── Convenience hook methods ──────────────────────────────────────

    def block_tools(self, *tool_names: str):
        """Block specific tools from being executed.

        Example: agent.block_tools("execute", "write_file")
        """
        blocked = set(tool_names)
        self.before_tool = lambda name, args: name not in blocked

    def allow_only_tools(self, *tool_names: str):
        """Only allow specific tools to be executed. All others blocked.

        Example: agent.allow_only_tools("read_file", "ls", "grep")
        """
        allowed = set(tool_names)
        self.before_tool = lambda name, args: name in allowed

    def inject_context(self, text: str):
        """Inject additional context into every LLM call.

        Example: agent.inject_context("Always respond in Spanish")
        """
        from clawagents.providers.llm import LLMMessage
        existing = self.before_llm

        def hook(messages):
            if existing:
                messages = existing(messages)
            return [*messages, LLMMessage(role="user", content=f"[Context] {text}")]

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
        def hook(name, args, result):
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
            child_state = await self._agent.invoke(task)
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
    tools: Optional[List] = None,
    skills: Union[str, List[Union[str, os.PathLike]], None] = None,
    fallback_models: Optional[List[str]] = None,
    memory: Union[str, List[Union[str, os.PathLike]], None] = None,
    sandbox: Any = None,
    streaming: bool = True,
    context_window: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    use_native_tools: bool = True,
    on_event: Optional[OnEvent] = None,
    trajectory: Optional[bool] = None,
    rethink: Optional[bool] = None,
    learn: Optional[bool] = None,
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
        instruction:    What the agent should do / how it should behave.
        tools:          Additional tools. Built-in tools always included.
        skills:         Skill directories (default: auto-discovers ./skills). The built-in ByteRover skill is always included when present.
        memory:         AGENTS.md paths (default: auto-discovers ./AGENTS.md, ./CLAWAGENTS.md).
        streaming:      Enable streaming output (default: True).
        context_window:  Max context window in tokens (default: from CONTEXT_WINDOW env / 1000000).
        max_tokens:     Max output tokens per call (default: from MAX_TOKENS env / 8192).
        temperature:    Sampling temperature (default: from TEMPERATURE env / 0.0).
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

        # AWS Bedrock via gateway
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
    if trajectory is None:
        trajectory = os.environ.get("CLAW_TRAJECTORY", "").lower() in ("1", "true", "yes")
    if rethink is None:
        rethink = os.environ.get("CLAW_RETHINK", "").lower() in ("1", "true", "yes")
    if learn is None:
        learn = os.environ.get("CLAW_LEARN", "").lower() in ("1", "true", "yes")
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

    # ── Resolve model → LLMProvider ────────────────────────────────────
    llm = _resolve_model(model, streaming, api_key, context_window, max_tokens, temperature, base_url, api_version)

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
    if sandbox is None:
        from clawagents.sandbox.local import LocalBackend
        sb = LocalBackend()
    else:
        sb = sandbox

    registry = ToolRegistry()

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

    class _LazyWebFetch(LazyTool):
        def __init__(self):
            from clawagents.tools.web import web_tools as _wt
            spec = next(t for t in _wt if t.name == "web_fetch")
            super().__init__(spec.name, spec.description, spec.parameters, "clawagents.tools.web", "", getattr(spec, "keywords", []))
        async def execute(self, args):
            if self._resolved is None:
                from clawagents.tools.web import web_tools as _wt
                self._resolved = next(t for t in _wt if t.name == "web_fetch")
            return await self._resolved.execute(args)
    registry.register(_LazyWebFetch())

    # ── Adapt and register user-provided tools ─────────────────────────
    if tools:
        for tool in tools:
            if hasattr(tool, "ainvoke") and not hasattr(tool, "execute"):
                registry.register(LangChainToolAdapter(tool))
            else:
                registry.register(tool)

    # ── Auto-discover skills from default locations ─────────────────────
    skill_summaries: Optional[str] = None
    base_skill_dirs = _to_list(skills) if skills is not None else _auto_discover_skills()
    _bundled = _get_bundled_skills_dir()
    skill_dirs = (base_skill_dirs + [_bundled]) if (_bundled and os.path.isdir(_bundled)) else base_skill_dirs
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

        loaded_skills = skill_store.list()
        if loaded_skills:
            lines = [f"- **{s.name}**: {s.description or '(no description)'}" for s in loaded_skills]
            skill_summaries = "## Available Skills\nUse the `use_skill` tool to load full instructions.\n" + "\n".join(lines)

        # Skill prompt budget limits
        MAX_SKILLS_PROMPT_CHARS = 4000
        MAX_SKILLS_IN_PROMPT = 20

        if skill_summaries:
            skill_lines = [l for l in skill_summaries.split("\n") if l.startswith("- **")]
            if len(skill_lines) > MAX_SKILLS_IN_PROMPT:
                truncated = skill_lines[:MAX_SKILLS_IN_PROMPT]
                skill_summaries = ("## Available Skills\nUse the `use_skill` tool to load full instructions.\n"
                    + "\n".join(truncated)
                    + f"\n\n({len(skill_lines) - MAX_SKILLS_IN_PROMPT} more skills available — use list_skills to see all)")
            if len(skill_summaries) > MAX_SKILLS_PROMPT_CHARS:
                skill_summaries = (skill_summaries[:MAX_SKILLS_PROMPT_CHARS]
                    + "\n\n...(skill list truncated — use list_skills to see all)")

        for skill_tool in create_skill_tools(skill_store):
            if skill_tool.name == "use_skill":
                registry.register(skill_tool)

    # ── Auto-discover memory from default locations ────────────────────
    memory_paths = _to_list(memory) if memory is not None else _auto_discover_memory()
    composed_before_llm = _compose_before_llm(
        memory_paths=memory_paths,
        skill_summaries=skill_summaries,
    )

    agent = ClawAgent(
        llm=llm, tools=registry, system_prompt=instruction,
        streaming=streaming, use_native_tools=use_native_tools,
        context_window=context_window, on_event=on_event,
        before_llm=composed_before_llm, trajectory=trajectory,
        rethink=rethink, learn=learn, max_iterations=max_iterations,
        preview_chars=preview_chars, response_chars=response_chars,
        advisor_llm=resolved_advisor_llm, advisor_max_calls=resolved_advisor_max_calls,
        handoffs=handoffs, name=name,
    )

    from clawagents.tools.background_task import create_background_task_tools
    for task_tool in create_background_task_tools():
        registry.register(task_tool)

    from clawagents.tools.tool_program import create_tool_program_tool
    registry.register(create_tool_program_tool(registry))

    # ── Sub-agent tool (always available) ──────────────────────────────
    from clawagents.tools.subagent import create_task_tool
    registry.register(create_task_tool(llm, registry))

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
) -> LLMProvider:
    """Accept a model name string, an LLMProvider, or None (auto-detect)."""
    if isinstance(model, LLMProvider):
        return model

    from clawagents.config.config import load_config, get_default_model
    from clawagents.providers.llm import create_provider

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

    active_model = model if isinstance(model, str) and model else get_default_model(config)

    # Override the appropriate API key if provided.
    # Route by model family so a single ``api_key`` parameter targets the
    # correct provider config field. Without this, e.g. a Claude key
    # silently lands in ``openai_api_key`` and the Anthropic provider
    # falls back to the env var.
    if api_key:
        lower = active_model.lower()
        if lower.startswith("gemini"):
            config.gemini_api_key = api_key
        elif lower.startswith("claude") or lower.startswith("anthropic"):
            config.anthropic_api_key = api_key
        else:
            config.openai_api_key = api_key

    provider = create_provider(active_model, config)
    return provider


def _to_list(value) -> list:
    """Convert None, string, or list to a list."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    return list(value)


def _get_bundled_skills_dir() -> str:
    """Path to all bundled skills (byterover, openviking, etc.)."""
    return str(Path(__file__).resolve().parent / "skills")


# Default locations for auto-discovery
_DEFAULT_MEMORY_FILES = ["AGENTS.md", "CLAWAGENTS.md"]
_DEFAULT_SKILL_DIRS = ["skills", ".skills", "skill", ".skill", "Skills"]


def _auto_discover_memory() -> list:
    """Auto-discover memory files in common locations."""
    found = []
    for name in _DEFAULT_MEMORY_FILES:
        path = os.path.join(os.getcwd(), name)
        if os.path.isfile(path):
            found.append(path)
    return found


def _auto_discover_skills() -> list:
    """Auto-discover skill directories in common locations."""
    found = []
    for name in _DEFAULT_SKILL_DIRS:
        path = os.path.join(os.getcwd(), name)
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
) -> Optional[BeforeLLMHook]:
    """Compose memory loading + skill injection into one before_llm hook."""
    from clawagents.prompts import append_prompt_injection, build_prompt_injection

    memory_content: Optional[str] = None
    if memory_paths:
        from clawagents.memory.loader import load_memory_files
        memory_content = load_memory_files(memory_paths)

    if not memory_content and not skill_summaries:
        return None
    injection = build_prompt_injection(memory_content, skill_summaries)

    def hook(messages: list) -> list:
        return list(append_prompt_injection(messages, injection))

    return hook
