__version__ = "6.20.45"

from clawagents.agent import ClawAgent, create_claw_agent
from clawagents.run_result import RunResult
from clawagents.graph.agent_loop import (
    AgentState, OnEvent, EventKind,
    BeforeLLMHook, BeforeToolHook, AfterToolHook, HookResult,
)
from clawagents.graph.coordinator import (
    CoordinatorState,
    ForkedAgentWorkerBackend,
    SubprocessWorkerBackend,
    WorkerBackend,
    WorkerTask,
    run_coordinator,
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
    ChannelMessage, ChannelAttachment, ChannelCommand, ChannelAdapter,
    ChannelRouter, KeyedAsyncQueue, channel_message_to_agent_input,
    normalize_channel_attachments, parse_channel_command,
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
from clawagents.run_context import (
    MAX_SUBAGENT_DEPTH,
    ApprovalRecord,
    RunContext,
)
from clawagents.iteration_budget import (
    DEFAULT_DELEGATION_MAX_ITERATIONS,
    IterationBudget,
)
from clawagents.plugins import Plugin, PluginManager
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
    CompactProgressEvent,
    HandoffOccurredEvent,
    FinalOutputEvent, ErrorStreamEvent, ErrorEvent, stream_event_from_kind,
)
from clawagents.context.carryover import (
    CompactionCarryover,
    get_compaction_carryover,
    normalize_compaction_carryover,
    set_compaction_carryover,
)
from clawagents.handoffs import (
    Handoff, HandoffInputData, InputFilter, handoff,
)
from clawagents.handoff_filters import remove_all_tools, nest_handoff_history
from clawagents.function_tool import function_tool
from clawagents.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from clawagents.eval import (
    Message,
    AgentEnvironment,
    AgentResponder,
    TextEnvironment,
    TextEvaluationResult,
    TextEvaluationStep,
    run_agent_environment,
    run_text_environment,
)
from clawagents.tools.tool_program import (
    ToolProgramTool,
    create_tool_program_tool,
)
from clawagents.tools.cache import SqliteResultCacheManager
from clawagents.tools.catalog import create_tool_discovery_tools, names_for_tool_profile
from clawagents.explorer import create_explorer_tools
from clawagents.prompts import (
    PROMPT_CACHE_BOUNDARY,
    append_prompt_injection,
    build_prompt_injection,
    build_system_prompt,
)
from clawagents.sandbox import DockerBackend
from clawagents.sandbox.manifest import (
    SandboxManifest,
    SandboxManifestEntry,
    normalize_sandbox_manifest,
)

# ── Slash-command registry (v6.5) ──────────────────────────────────────
from clawagents.commands import (
    CommandDef, ResolvedCommand, COMMAND_REGISTRY,
    register_command, resolve_command, list_commands,
    format_help, all_command_names,
)

# ── Mid-run nudges (v6.5) ──────────────────────────────────────────────
from clawagents.interjection import (
    enqueue_interject, drain_interjects, take_stranded_interjects,
)
from clawagents.steer import (
    SteerMessage, SteerQueue, NextTurnQueue, SteerHook,
    steer, queue_message,
    drain_steer, drain_next_turn,
    peek_steer, peek_next_turn,
)

# ── Display-layer redaction (v6.5) ─────────────────────────────────────
from clawagents.redact import (
    redact, redact_obj, redact_env, add_pattern,
)

# ── Profile-aware filesystem paths (v6.5) ──────────────────────────────
from clawagents.paths import (
    DEFAULT_PROFILE,
    get_clawagents_home,
    get_clawagents_workspace_dir,
    get_trajectories_dir,
    get_sessions_dir,
    get_lessons_dir,
    display_clawagents_home,
    display_clawagents_workspace_dir,
    list_profiles,
)

# ── Auxiliary model registry (v6.5) ────────────────────────────────────
from clawagents.aux_models import (
    AuxModelTask, AuxModelSpec, AuxModelRegistry,
)

# ── Transport abstraction (v6.5) ───────────────────────────────────────
from clawagents.transport import (
    TransportRequest, TransportResponse,
    Transport, TransportRegistry, LegacyChatTransport,
)

# ── Background jobs (v6.5) ─────────────────────────────────────────────
from clawagents.background import (
    BackgroundJob, BackgroundJobManager, JobNotifier,
)
from clawagents.provider_profiles import (
    ProviderProfile, ResolvedProviderProfile,
    resolve_provider_profile, load_provider_profiles,
)
from clawagents.dry_run import build_dry_run_preview
from clawagents.plugin_compat import (
    LoadedCompatPlugin, PluginSkill, PluginCommand,
    load_plugin, discover_plugins,
)
from clawagents.tools.background_task import create_background_task_tools

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
from clawagents.permissions.plan_approval import (
    PlanApprovalAction, PlanApprovalDecision,
    await_plan_approval, normalize_plan_decision,
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

# ── Cron / scheduled jobs (v6.6) ───────────────────────────────────────
# In-process scheduler with persistent JSON job store. Cron expressions
# require the optional ``croniter`` package; interval and one-shot
# schedules work out of the box.
from clawagents.cron import (
    Scheduler,
    SchedulerError,
    Job,
    JobRunner,
    ParsedSchedule,
    parse_schedule,
    parse_duration,
    compute_next_run,
    create_job,
    get_job,
    list_jobs,
    update_job,
    pause_job,
    resume_job,
    trigger_job,
    remove_job,
    mark_job_run,
    advance_next_run,
    get_due_jobs,
    save_job_output,
    CRONITER_AVAILABLE,
)

# ── ACP adapter (v6.6) ─────────────────────────────────────────────────
# Wraps a ClawAgents agent so it can be served over Zed's Agent Client
# Protocol (JSON-RPC over stdio). The optional ``agent-client-protocol``
# package is only required to actually call ``AcpServer.serve()``.
from clawagents.acp import (
    AcpError,
    MissingAcpDependencyError,
    PromptRequest,
    SessionUpdate,
    AgentMessageChunk,
    AgentThoughtChunk,
    ToolCallStart,
    ToolCallComplete,
    PermissionRequest,
    PermissionDecision,
    StopReason,
    AgentSession,
    AcpServer,
    ACP_AVAILABLE,
)

# ── RL fine-tuning hooks (v6.6) ────────────────────────────────────────
# Capture agent runs as training-ready trajectories and export them to
# TRL / Atropos / SLIME / generic JSONL formats. The ``trl`` and
# ``atropos`` packages are optional — only required to actually drive a
# trainer or rollout collector.
from clawagents.rl import (
    RLError,
    MissingRLDependencyError,
    Trajectory,
    TrajectoryStep,
    ToolCall as RLToolCall,
    RLRecorder,
    RecorderConfig as RLRecorderConfig,
    RewardScorer,
    ContainsScorer,
    ExactMatchScorer,
    RegexScorer,
    LengthPenaltyScorer,
    CompositeScorer,
    export_jsonl as rl_export_jsonl,
    load_jsonl as rl_load_jsonl,
    to_chatml as rl_to_chatml,
    to_trl_sft,
    to_trl_dpo,
    to_atropos_rollout,
    to_next_state_transitions,
    TrlAdapter,
    AtroposAdapter,
    TRL_AVAILABLE,
    ATROPOS_AVAILABLE,
)

# run_goal is lazy via __getattr__ (see below) so goal.product is not imported
# at package import time.

# ── Grok-Build parity (v6.14+) ─────────────────────────────────────────
from clawagents.autopilot.loop import run_autopilot
from clawagents.marketplace import install_from_source, list_installed
from clawagents.sandbox.profiles import (
    OSSandboxProfile,
    get_profile,
    list_profiles,
    resolve_sandbox,
)
from clawagents.tools.subagent_resolve import resolve_subagent, ResolvedSubAgent
from clawagents.memory.attributed_hunks import (
    list_hunks,
    accept_hunk,
    reject_hunk,
    refresh_file_hunks,
)
from clawagents.memory.scope_graph import ScopeGraph, build_repo_map_incremental
from clawagents.memory.full_replace_compaction import (
    assemble_compacted_history,
    apply_full_replace_compaction,
    format_compact_summary,
    is_degenerate_summary,
)


def __getattr__(name: str):
    if name == "run_goal":
        from clawagents.goal.product import run_goal as _run_goal

        return _run_goal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
