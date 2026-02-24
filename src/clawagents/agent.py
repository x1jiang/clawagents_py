import os
import asyncio
from typing import Optional, List, Dict, Any, Union

from clawagents.providers.llm import LLMProvider
from clawagents.tools.registry import ToolRegistry, Tool, ToolResult
from clawagents.graph.agent_loop import (
    run_agent_graph, AgentState, OnEvent,
    BeforeLLMHook, BeforeToolHook, AfterToolHook,
)


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
        context_window: int = 128_000,
        on_event: Optional[OnEvent] = None,
        before_llm: Optional[BeforeLLMHook] = None,
        before_tool: Optional[BeforeToolHook] = None,
        after_tool: Optional[AfterToolHook] = None,
    ):
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

    async def invoke(
        self,
        task: str,
        max_iterations: int = 200,
        on_event: Optional[OnEvent] = None,
    ) -> AgentState:
        return await run_agent_graph(
            task=task,
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=max_iterations,
            streaming=self.streaming,
            context_window=self.context_window,
            on_event=on_event or self.on_event,
            before_llm=self.before_llm,
            before_tool=self.before_tool,
            after_tool=self.after_tool,
            use_native_tools=self.use_native_tools,
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


def create_claw_agent(
    model: Union[str, LLMProvider, None] = None,
    api_key: Optional[str] = None,
    instruction: Optional[str] = None,
    tools: Optional[List] = None,
    skills: Union[str, List[Union[str, os.PathLike]], None] = None,
    memory: Union[str, List[Union[str, os.PathLike]], None] = None,
    sandbox: Any = None,
    streaming: bool = True,
    context_window: int = 128_000,
    max_tokens: int = 8192,
    use_native_tools: bool = True,
    on_event: Optional[OnEvent] = None,
) -> ClawAgent:
    """
    Create a ClawAgent with full-stack capabilities.

    Args:
        model:          Model name ("gpt-5", "gemini-3-flash") or LLMProvider.
                        None = auto-detect from env.
        api_key:        API key for the model provider. Auto-routed based on model name.
                        Falls back to env vars (OPENAI_API_KEY / GEMINI_API_KEY) if omitted.
        instruction:    What the agent should do / how it should behave.
        tools:          Additional tools. Built-in tools always included.
        skills:         Skill directories (default: auto-discovers ./skills).
        memory:         AGENTS.md paths (default: auto-discovers ./AGENTS.md, ./CLAWAGENTS.md).
        streaming:      Enable streaming output (default: True).
        context_window:  Max context window in tokens (default: 128000).
        max_tokens:     Max output tokens per call (default: 8192).

    Examples:
        # Zero-config (uses env vars)
        agent = create_claw_agent()

        # Explicit model + key
        agent = create_claw_agent("gpt-5-mini", api_key="sk-...")

        # Gemini with key
        agent = create_claw_agent("gemini-2.5-flash", api_key="AIza...")

    Advanced hooks (set after creation):
        agent.before_tool = lambda name, args: name != "execute"
    """
    # ── Resolve model → LLMProvider ────────────────────────────────────
    llm = _resolve_model(model, streaming, api_key, context_window, max_tokens)

    # ── Resolve sandbox backend ────────────────────────────────────────
    if sandbox is None:
        from clawagents.sandbox.local import LocalBackend
        sb = LocalBackend()
    else:
        sb = sandbox

    registry = ToolRegistry()

    # ── Built-in tools (backed by sandbox) ─────────────────────────────
    from clawagents.tools.filesystem import create_filesystem_tools
    from clawagents.tools.exec import create_exec_tools
    from clawagents.tools.advanced_fs import create_advanced_fs_tools
    from clawagents.tools.todolist import todolist_tools
    from clawagents.tools.think import think_tools
    from clawagents.tools.web import web_tools
    from clawagents.tools.interactive import interactive_tools

    for tool in [
        *create_filesystem_tools(sb), *create_exec_tools(sb), *todolist_tools,
        *think_tools, *web_tools, *create_advanced_fs_tools(sb), *interactive_tools,
    ]:
        registry.register(tool)

    # ── Adapt and register user-provided tools ─────────────────────────
    if tools:
        for tool in tools:
            if hasattr(tool, "ainvoke") and not hasattr(tool, "execute"):
                registry.register(LangChainToolAdapter(tool))
            else:
                registry.register(tool)

    # ── Auto-discover skills from default locations ─────────────────────
    skill_summaries: Optional[str] = None
    skill_dirs = _to_list(skills) if skills is not None else _auto_discover_skills()
    if skill_dirs:
        from clawagents.tools.skills import SkillStore, create_skill_tools

        skill_store = SkillStore()
        for d in skill_dirs:
            if os.path.exists(str(d)):
                skill_store.add_directory(d)

        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(lambda: asyncio.run(skill_store.load_all()))
                future.result(timeout=10)
        except RuntimeError:
            asyncio.run(skill_store.load_all())

        loaded_skills = skill_store.list()
        if loaded_skills:
            lines = [f"- **{s.name}**: {s.description or '(no description)'}" for s in loaded_skills]
            skill_summaries = "## Available Skills\nUse the `use_skill` tool to load full instructions.\n" + "\n".join(lines)

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
        before_llm=composed_before_llm,
    )

    # ── Sub-agent tool (always available) ──────────────────────────────
    from clawagents.tools.subagent import create_task_tool
    registry.register(create_task_tool(llm, registry))

    return agent


# ─── Internal Helpers ─────────────────────────────────────────────────────

def _resolve_model(
    model: Union[str, LLMProvider, None],
    streaming: bool,
    api_key: Optional[str] = None,
    context_window: Optional[int] = None,
    max_tokens: Optional[int] = None,
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

    active_model = model if isinstance(model, str) and model else get_default_model(config)

    # Override the appropriate API key if provided
    if api_key:
        if active_model.lower().startswith("gemini"):
            config.gemini_api_key = api_key
        else:
            config.openai_api_key = api_key

    return create_provider(active_model, config)


def _to_list(value) -> list:
    """Convert None, string, or list to a list."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    return list(value)


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


def _compose_before_llm(
    memory_paths: list,
    skill_summaries: Optional[str],
) -> Optional[BeforeLLMHook]:
    """Compose memory loading + skill injection into one before_llm hook."""
    from clawagents.providers.llm import LLMMessage

    memory_content: Optional[str] = None
    if memory_paths:
        from clawagents.memory.loader import load_memory_files
        memory_content = load_memory_files(memory_paths)

    if not memory_content and not skill_summaries:
        return None

    def hook(messages: list) -> list:
        inject_parts = []
        if memory_content:
            inject_parts.append(memory_content)
        if skill_summaries:
            inject_parts.append(skill_summaries)

        if inject_parts:
            joined = "\n\n".join(inject_parts)
            result = list(messages)
            for i, m in enumerate(result):
                role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
                if role == "system":
                    content = getattr(m, "content", "") if not isinstance(m, dict) else m.get("content", "")
                    result[i] = LLMMessage(role="system", content=content + "\n\n" + joined)
                    break
            return result
        return messages

    return hook
