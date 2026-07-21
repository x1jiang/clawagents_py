"""Stable parameter object for the public ReAct runner API.

``run_agent_graph`` remains backward compatible, but the implementation no
longer needs to manually forward the same expanding argument list through
multiple branches.  New per-run options have one home and one conversion to
the core runner's keyword interface.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class AgentRunConfig:
    task: str
    llm: Any
    tools: Any = None
    system_prompt: str | None = None
    max_iterations: int = 200
    streaming: bool = True
    context_window: int = 1_000_000
    on_event: Any = None
    before_llm: Any = None
    before_tool: Any = None
    after_tool: Any = None
    use_native_tools: bool = True
    trajectory: bool = False
    rethink: bool = False
    learn: bool = False
    atlas: bool = False
    atlas_config: Any = None
    preview_chars: int = 120
    response_chars: int = 500
    timeout_s: float = 0
    features: dict[str, bool] | None = None
    advisor_llm: Any = None
    advisor_max_calls: int = 3
    run_context: Any = None
    user_context: Any = None
    hooks: Any = None
    agent_hooks: Any = None
    input_guardrails: list[Any] | None = None
    output_guardrails: list[Any] | None = None
    output_type: type | None = None
    on_stream_event: Any = None
    session: Any = None
    session_preload_limit: int | None = 200
    handoffs: list[Any] | None = None
    agent_name: str | None = None
    action_mode: str = "tools"
    approval_handler: Any = None
    require_approval_tools: list[str] | None = None
    image_blocks: list[dict[str, Any]] | None = None
    file_blocks: list[dict[str, Any]] | None = None
    session_end_tail: bool = True

    def core_kwargs(self) -> dict[str, Any]:
        """Return a shallow copy safe to pass to the compatibility core."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
