__version__ = "6.3.0.post1"

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
    FinalOutputEvent, ErrorStreamEvent, ErrorEvent, stream_event_from_kind,
)
from clawagents.function_tool import function_tool
from clawagents.retry import RetryPolicy, DEFAULT_RETRY_POLICY
