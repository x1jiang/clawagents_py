__version__ = "6.1.1"

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
)
