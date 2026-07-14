"""Prompt helpers for ATLAS reflections inside the ClawAgents loop."""

from __future__ import annotations

from typing import Any

from clawagents.providers.llm import LLMMessage

STANDING_PROTOCOL_FALLBACK = """\
ATLAS runtime interaction is active.

Work on the user's task normally. Do not request or load the taxonomy at task
start. The harness may inject ATLAS reflection prompts after tool failures,
subagent completion, or before final submission. When that happens, respond
with the required Observe → Correlate → Map → Decide reflection shape (and
final-gate fields when asked). Mapping no codes ("none apply") is valid.

When the task itself is complete, return the proposed final answer. The
harness will run the mandatory final ATLAS gate before releasing that answer.
"""


def message_text(msg: LLMMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def render_messages(messages: list[LLMMessage]) -> str:
    return "\n\n".join(
        f"[{m.role.upper()}]\n{message_text(m)}" for m in messages
    )


def render_recent_messages(
    messages: list[LLMMessage],
    *,
    max_messages: int,
    max_chars: int,
    segment_start: int = 0,
) -> str:
    """Render a trajectory window for an ATLAS checkpoint."""
    window = messages[segment_start:] if segment_start else messages
    if not window:
        return ""
    selected = window[-max_messages:]
    first_user = next((m for m in messages if m.role == "user"), None)
    if first_user is not None and first_user not in selected:
        selected = [first_user, *selected]
    rendered = render_messages(selected)
    if len(rendered) <= max_chars:
        return rendered
    if first_user is not None:
        task_prefix = render_messages([first_user])
        if len(task_prefix) < max_chars:
            tail_budget = max_chars - len(task_prefix) - 24
            if tail_budget > 0:
                return task_prefix + "\n\n[...]\n" + rendered[-tail_budget:]
    return rendered[-max_chars:]


def standing_protocol(delivery_protocol: str | None) -> str:
    text = (delivery_protocol or "").strip()
    return text or STANDING_PROTOCOL_FALLBACK


def build_reflection_prompt(
    *,
    taxonomy_id: str,
    codes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    checkpoint_id: str,
    gate_label: str,
    recent_activity: str,
    full: bool,
    prompt_suffix: str = "",
) -> str:
    from atlas_runtime import render_reflection_prompt

    return (
        render_reflection_prompt(
            taxonomy_id=taxonomy_id,
            codes=codes,
            checkpoint_id=checkpoint_id,
            gate_label=gate_label,
            recent_activity=recent_activity,
            full=full,
        )
        + prompt_suffix
    )


def build_format_repair(
    *,
    checkpoint_id: str,
    issues: list[str] | tuple[str, ...],
    full: bool,
) -> str:
    from atlas_runtime import render_format_repair

    return render_format_repair(
        checkpoint_id=checkpoint_id,
        issues=issues,
        full=full,
    )
