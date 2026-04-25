__version__ = "6.4.1"

from clawagents.agent import ClawAgent, create_claw_agent
from clawagents.graph.agent_loop import (
    AgentState, OnEvent, EventKind,
    BeforeLLMHook, BeforeToolHook, AfterToolHook, HookResult,
)
from clawagents.trajectory import (
    TrajectoryRecorder, TurnRecord, RunSummary,
    extract_lessons, save_lessons, load_lessons,
    build_lesson_preamble, build_rethink_with_lessons,
)
from clawagents.context import (
    ContextEngine, ContextEngineConfig, DefaultContextEngine,
    register_context_engine, resolve_context_engine, list_context_engines,
)
from clawagents.channels import (
    ChannelMessage, ChannelAdapter, ChannelRouter, KeyedAsyncQueue,
)
from clawagents.errors import (
    ErrorClass, ErrorDescriptor, RecoveryRecipe,
    classify_error, get_recovery_recipe,
)
from clawagents.hooks import (
    HooksConfig, ExternalHookRunner, load_hooks_config,
)
from clawagents.hooks.prompt_hook import (
    PromptHook, PromptHookVerdict,
)
from clawagents.session import (
    SessionWriter, SessionReader, SessionInfo, list_sessions,
    Session, InMemorySession, JsonlFileSession, SQLiteSession,
)

# ── OpenAI-Agents-inspired APIs (additive) ─────────────────────────────
from clawagents.run_context import RunContext, ApprovalRecord
from clawagents.usage import Usage, RequestUsage
from clawagents.lifecycle import RunHooks, AgentHooks, composite_hooks
from clawagents.guardrails import (
    InputGuardrail, OutputGuardrail,
    GuardrailBehavior, GuardrailResult, GuardrailTripwireTriggered,
    input_guardrail, output_guardrail,
)
from clawagents.stream_events import (
    StreamEvent, TurnStartedEvent, AssistantTextEvent, AssistantDeltaEvent,
    ToolCallPlannedEvent, ToolStartedEvent, ToolResultEvent,
    ApprovalRequiredEvent, UsageEvent, GuardrailTrippedEvent,
    HandoffOccurredEvent,
    FinalOutputEvent, ErrorStreamEvent, ErrorEvent, stream_event_from_kind,
)
from clawagents.handoffs import (
    Handoff, HandoffInputData, InputFilter, handoff,
)
from clawagents.handoff_filters import remove_all_tools, nest_handoff_history
from clawagents.function_tool import function_tool
from clawagents.retry import RetryPolicy, DEFAULT_RETRY_POLICY

# ── Settings hierarchy (v6.4) ──────────────────────────────────────────
from clawagents.settings import (
    SettingsLayer, resolve_settings, get_setting,
)

# ── Structured HITL (v6.4) ─────────────────────────────────────────────
from clawagents.tools.ask_user_question import (
    AskUserQuestionTool,
    ask_user_question_tool,
)

# ── Multimodal helpers (v6.4) ──────────────────────────────────────────
from clawagents.media.images import (
    is_pillow_available,
    sanitize_image_block,
    sanitize_tool_output,
)

# ── Exec Safety v2 (v6.4) ─────────────────────────────────────────────
from clawagents.permissions import (
    PermissionMode, WRITE_CLASS_TOOLS,
    is_write_class_tool, permission_mode_from_string,
)
from clawagents.tools.bash_validator import (
    BashDecision, CommandCategory, Decision, validate_bash,
)
from clawagents.tools.exec_obfuscation import (
    ObfuscationFinding, detect_obfuscation,
)
from clawagents.tools.plan_mode import (
    EnterPlanModeTool, ExitPlanModeTool,
    enter_plan_mode_tool, exit_plan_mode_tool,
    create_plan_mode_tools,
)

# ── Tracing (v6.4) ─────────────────────────────────────────────────────
from clawagents.tracing import (
    Span, SpanKind, SpanStatus,
    TracingProcessor, TracingExporter,
    BatchTraceProcessor, NoopSpanExporter, ConsoleSpanExporter, JsonlSpanExporter,
    set_default_processor, get_default_processor, add_trace_processor,
    flush_traces, shutdown_tracing,
    agent_span, turn_span, generation_span, tool_span,
    handoff_span, guardrail_span, custom_span,
    current_span, current_trace_id,
)

# ── MCP (Model Context Protocol) integration (v6.4) ────────────────────
# The optional ``mcp`` SDK is imported lazily — these classes import without
# the SDK installed and only raise on ``connect()``.
from clawagents.mcp import (
    MCPServer,
    MCPServerStdio,
    MCPServerSse,
    MCPServerStreamableHttp,
    MCPServerManager,
    MCPLifecyclePhase,
    MCPToolDescriptor,
    MCPBridgedTool,
    is_mcp_sdk_available,
    require_mcp_sdk,
    mcp_tool_to_clawagents_tool,
)
