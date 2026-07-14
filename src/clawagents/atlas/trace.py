"""Build and persist ATLAS GenerationTrace records from a ClawAgents run."""

from __future__ import annotations

from typing import Any

from clawagents.atlas.prompts import render_messages
from clawagents.atlas.session import AtlasRunSession
from clawagents.providers.llm import LLMMessage


def build_and_record_trace(
    run: AtlasRunSession,
    messages: list[LLMMessage],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    from atlas_runtime import GenerationTrace, record_trace, redact_trace

    meta = {
        "harness": "clawagents",
        "taxonomy_id": run.taxonomy_id,
        "checkpoint_count": run.checkpoint_count,
        "final_gate_allowed": run.final_gate_allowed,
    }
    if metadata:
        meta.update(metadata)

    trace = GenerationTrace(
        problem_id=run.run_id,
        task=run.task,
        raw_trajectory=render_messages(messages),
        metadata=meta,
    )
    if run.config.get("redact_traces", True):
        trace = redact_trace(trace)
    record_trace(run.atlas_session, trace)


def end_atlas_session(run: AtlasRunSession) -> Any:
    from atlas_runtime import end_session

    return end_session(run.atlas_session)


def abort_atlas_session(run: AtlasRunSession) -> None:
    """Best-effort cleanup when the run errors before a normal end_session."""
    session = run.atlas_session
    if getattr(session, "_ended", False):
        return
    try:
        session.workspace.finish_session(session.session_id)
        session._ended = True
    except Exception:
        pass
