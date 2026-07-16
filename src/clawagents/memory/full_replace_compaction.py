"""Grok Build–style full-replace compaction.

Algorithm (mirrors ``xai-grok-compaction`` code_compaction):

1. Keep the original system message(s) verbatim.
2. Summarize everything *before* the last real user query.
3. Rebuild history as::

       [system, user_prefix?, AGENTS.md?, <user_query>last</user_query>,
        recent…, continuation_summary(+carryover), assistant_ack?, reminder?]

4. Clean the model summary (strip ``<analysis>``, unwrap ``<summary>``,
   neutralize echoed control tags) and reject degenerate seeds.
5. Re-inject project instructions and active-state reminders outside the
   summarizer so policy state never depends on LLM fidelity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from clawagents.providers.llm import LLMMessage, LLMProvider

MIN_SUMMARY_SEED_CHARS = 500
DEFAULT_MAX_SUMMARY_ATTEMPTS = 3
CONTINUATION_PREAMBLE = (
    "This session is being continued from a previous conversation that ran out "
    "of context. The summary below covers the earlier portion of the conversation."
)

_CONTROL_TAGS = (
    ("</summary>", "<\u200b/summary>"),
    ("<summary>", "<\u200bsummary>"),
    ("</analysis>", "<\u200b/analysis>"),
    ("<analysis>", "<\u200banalysis>"),
    ("</summary_request>", "<\u200b/summary_request>"),
    ("<summary_request>", "<\u200bsummary_request>"),
)

FULL_REPLACE_SUMMARY_PROMPT = """\
Your task is to produce a faithful, concise summary of the conversation so far \
so that a successor assistant can continue the work seamlessly after the earlier \
turns are discarded. The successor will see the user's original query plus this \
summary. Capture what is needed to continue — the user's explicit requests, your \
most recent actions, key technical details, file paths, commands, configuration, \
and architectural decisions — but be economical.

CRITICAL: If earlier turns include a prior compaction summary (marked with \
"This session is being continued" or "[System — Compacted History]"), treat it \
as authoritative for the early history and carry still-relevant information forward.

Output the final summary inside a single <summary>...</summary> block with these \
numbered sections (write "None" if empty):

1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections
4. Errors and Fixes
5. Problem Solving
6. All User Messages
7. Pending Tasks
8. Current Work
9. Optional Next Step

Do NOT call tools. Respond with ONLY the <summary>...</summary> block.

## Original Task Context
{task_context}

## Conversation To Summarize
{conversation}
"""


# ─── Summary cleaning (Grok port) ─────────────────────────────────────────


def neutralize_compaction_control_tokens(text: str) -> str:
    out = text
    for src, dst in _CONTROL_TAGS:
        out = out.replace(src, dst)
    return out


def _strip_leading_scratchpad(inner: str) -> str:
    s = inner.strip()
    lead = s.lstrip("#*- >\t")
    if not (lead[:1].isdigit()) and "</analysis>" in s:
        pos = s.rfind("</analysis>")
        s = s[pos + len("</analysis>"):].lstrip()
    if s.startswith("<summary>"):
        s = s[len("<summary>"):].lstrip()
    return s


def format_compact_summary(summary: str) -> str:
    """Clean model output into a plain ``Summary:`` seed."""
    result = summary or ""

    # Peel leading <analysis>…</analysis> scratchpads
    while True:
        start = result.find("<analysis>")
        if start < 0:
            break
        summary_pos = result.find("<summary>")
        if summary_pos >= 0:
            is_leading = start < summary_pos or not result[summary_pos + len("<summary>"):start].strip()
        else:
            is_leading = not result[:start].strip()
        if not is_leading:
            break
        close = result.find("</analysis>", start)
        if close >= 0:
            end = close + len("</analysis>")
            result = result[:start] + result[end:]
        else:
            drop_to = result.find("<summary>", start)
            drop_to = len(result) if drop_to < 0 else drop_to
            result = result[:start] + result[drop_to:]
            break

    # Unwrap <summary>…</summary>
    start = result.find("<summary>")
    end = result.rfind("</summary>")
    if start >= 0 and end > start:
        before = result[:start]
        after = result[end + len("</summary>"):]
        inner = _strip_leading_scratchpad(result[start + len("<summary>"):end].strip())
        result = f"{before}Summary:\n{inner}{after}"

    result = neutralize_compaction_control_tokens(result)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


def is_degenerate_summary(raw_summary: str, *, min_chars: int = MIN_SUMMARY_SEED_CHARS) -> bool:
    return len(format_compact_summary(raw_summary)) < min_chars


def format_compact_summary_content(raw_summary: str) -> str:
    cleaned = format_compact_summary(raw_summary)
    if not cleaned.startswith("Summary:") and cleaned:
        cleaned = f"Summary:\n{cleaned}"
    return f"{CONTINUATION_PREAMBLE}\n\n{cleaned}"


def wrap_user_query(text: str) -> str:
    return f"<user_query>\n{text}\n</user_query>"


# ─── Assembly ─────────────────────────────────────────────────────────────


@dataclass
class CompactedHistoryParts:
    system_messages: list[LLMMessage]
    user_message_prefix: str = ""
    agents_md_reminder: str | None = None
    last_user_query: str | None = None
    recent_messages: list[LLMMessage] = field(default_factory=list)
    compaction_summary: str = ""
    system_reminder: str | None = None
    transcript_hint: str | None = None
    carryover_markdown: str | None = None


def assemble_compacted_history(parts: CompactedHistoryParts) -> list[LLMMessage]:
    """Build ``[SP, UP?, AGENTS?, UQ?, recent…, summary, ack?, reminder?]``."""
    out: list[LLMMessage] = list(parts.system_messages)

    if parts.user_message_prefix.strip():
        out.append(LLMMessage(role="user", content=parts.user_message_prefix.strip()))

    if parts.agents_md_reminder and parts.agents_md_reminder.strip():
        body = parts.agents_md_reminder.strip()
        if "<system-reminder>" not in body:
            body = f"<system-reminder>\n{body}\n</system-reminder>"
        out.append(LLMMessage(role="user", content=body))

    if parts.last_user_query and parts.last_user_query.strip():
        out.append(LLMMessage(role="user", content=wrap_user_query(parts.last_user_query.strip())))

    out.extend(parts.recent_messages)

    summary_body = format_compact_summary_content(parts.compaction_summary)
    if parts.carryover_markdown and parts.carryover_markdown.strip():
        summary_body = (
            f"{summary_body}\n\n[System — Compacted History]\n"
            f"{parts.carryover_markdown.strip()}"
        )
    else:
        # Keep Compacted History marker for existing test/event consumers
        summary_body = f"{summary_body}\n\n[System — Compacted History]"
    if parts.transcript_hint:
        summary_body = f"{summary_body}{parts.transcript_hint}"

    out.append(LLMMessage(role="user", content=summary_body))
    out.append(
        LLMMessage(
            role="assistant",
            content="Understood — continuing from the compacted handoff summary.",
        )
    )

    if parts.system_reminder and parts.system_reminder.strip():
        rem = parts.system_reminder.strip()
        if "<system-reminder>" not in rem:
            rem = f"<system-reminder>\n{rem}\n</system-reminder>"
        out.append(LLMMessage(role="user", content=rem))

    return out


# ─── Split helpers ─────────────────────────────────────────────────────────


def _is_real_user(msg: LLMMessage) -> bool:
    if msg.role != "user":
        return False
    content = msg.content if isinstance(msg.content, str) else ""
    if content.startswith("[Tool Result]"):
        return False
    if "Compacted History" in content:
        return False
    if content.startswith("This session is being continued"):
        return False
    return True


def find_last_real_user_index(messages: Sequence[LLMMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if _is_real_user(messages[i]):
            return i
    return -1


def snap_recent_tool_pair_safe(
    older: list[LLMMessage],
    recent: list[LLMMessage],
) -> tuple[list[LLMMessage], list[LLMMessage]]:
    """Ensure recent does not start with orphan tool results."""
    while recent and recent[0].role == "tool":
        # Pull the preceding assistant (with tool calls) from older if present
        if older and older[-1].role == "assistant":
            recent = [older[-1], *recent]
            older = older[:-1]
        else:
            # Drop orphan tool result
            recent = recent[1:]
    return older, recent


def split_for_full_replace(
    non_system: list[LLMMessage],
) -> tuple[list[LLMMessage], str | None, list[LLMMessage]] | None:
    """Return ``(to_summarize, last_query, recent)`` or None if nothing to compact."""
    last_idx = find_last_real_user_index(non_system)
    if last_idx < 0:
        return None
    last_query_msg = non_system[last_idx]
    last_query = (
        last_query_msg.content
        if isinstance(last_query_msg.content, str)
        else str(last_query_msg.content)
    )
    older = list(non_system[:last_idx])
    recent = list(non_system[last_idx + 1 :])
    older, recent = snap_recent_tool_pair_safe(older, recent)
    if not older:
        return None
    return older, last_query, recent


def render_conversation_for_summary(
    messages: Sequence[LLMMessage],
    *,
    lossy: bool = False,
    max_tool_chars: int = 400,
) -> str:
    parts: list[str] = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if m.role == "assistant" and getattr(m, "tool_calls_meta", None):
            names = ", ".join(tc.get("name", "?") for tc in (m.tool_calls_meta or []))
            blob = content[:200]
            parts.append(f"[TOOL CALLS: {names}] {blob}")
        elif m.role == "tool":
            if lossy:
                parts.append(f"[TOOL RESULT]: {content[:80]}…")
            else:
                parts.append(f"[TOOL RESULT]: {content[:max_tool_chars]}")
        else:
            parts.append(f"[{m.role.upper()}]: {content[:800]}")
    return "\n\n".join(parts)


def build_state_reminder(
    *,
    recent_files: Sequence[str] | None = None,
    plan_text: str | None = None,
    invoked_skills: Sequence[str] | None = None,
    active_workers: Sequence[str] | None = None,
    todos: Sequence[str] | None = None,
) -> str | None:
    lines: list[str] = []
    if recent_files:
        lines.append("Files touched this session:")
        lines.extend(f"- {p}" for p in list(recent_files)[:20])
    if invoked_skills:
        lines.append("Invoked skills: " + ", ".join(list(invoked_skills)[:12]))
    if active_workers:
        lines.append("Active workers: " + ", ".join(list(active_workers)[:12]))
    if todos:
        lines.append("Pending todos:")
        lines.extend(f"- {t}" for t in list(todos)[:12])
    if plan_text and plan_text.strip():
        lines.append("Active plan (preserved):")
        lines.append(plan_text.strip()[:2000])
    if not lines:
        return None
    return "\n".join(lines)


def load_agents_md_reminder(workspace: str | None = None) -> str | None:
    try:
        from pathlib import Path

        from clawagents.memory.rules import load_rules_text

        root = Path(workspace or Path.cwd())
        candidates = [
            root / "AGENTS.md",
            root / "CLAWAGENTS.md",
            root / "CLAUDE.md",
        ]
        paths = [str(p) for p in candidates if p.is_file()]
        if not paths:
            return None
        text = load_rules_text(paths=paths)
        if not text:
            from clawagents.memory.loader import load_memory_files

            text = load_memory_files(paths)
        if text and text.strip():
            return f"Project instructions (re-injected after compaction):\n\n{text.strip()}"
    except Exception:
        return None
    return None


async def sample_full_replace_summary(
    llm: LLMProvider,
    conversation_text: str,
    *,
    task_context: str = "",
    max_attempts: int = DEFAULT_MAX_SUMMARY_ATTEMPTS,
) -> str:
    """Sample a summary; retry on empty/degenerate output."""
    prompt = FULL_REPLACE_SUMMARY_PROMPT.format(
        task_context=task_context or "(none)",
        conversation=conversation_text,
    )
    last = ""
    for _ in range(max(1, max_attempts)):
        try:
            resp = await llm.chat([LLMMessage(role="user", content=prompt)])
            raw = (resp.content or "").strip()
        except Exception:
            raw = ""
        if raw and not is_degenerate_summary(raw):
            return raw
        last = raw or last
    # Accept whatever we got rather than failing the whole compact
    return last or "Summary:\nNone\n(Compaction produced an empty seed; continue from recent turns.)"


def sanitize_compacted_history(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Drop orphan tool results that lack a preceding assistant tool-call."""
    out: list[LLMMessage] = []
    pending_ids: set[str] = set()
    for m in messages:
        if m.role == "assistant" and getattr(m, "tool_calls_meta", None):
            pending_ids = {
                str(tc.get("id") or "")
                for tc in (m.tool_calls_meta or [])
                if tc.get("id")
            }
            out.append(m)
            continue
        if m.role == "tool":
            tcid = getattr(m, "tool_call_id", None) or ""
            if pending_ids and tcid and tcid not in pending_ids:
                continue
            if not pending_ids and not getattr(m, "tool_calls_meta", None):
                # orphan with no open assistant tool calls
                # keep if previous was assistant (best-effort) else drop
                if not out or out[-1].role != "assistant":
                    continue
            out.append(m)
            continue
        pending_ids = set()
        out.append(m)
    return out


def _rough_message_tokens(messages: Sequence[LLMMessage]) -> int:
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(content.encode("utf-8"))
    return max(1, total // 4)


async def apply_full_replace_compaction(
    messages: list[LLMMessage],
    llm: LLMProvider,
    *,
    workspace: str | None = None,
    carryover_markdown: str | None = None,
    agents_md: str | None = None,
    system_reminder: str | None = None,
    user_prefix: str = "",
    transcript_hint: str | None = None,
    lossy: bool = False,
    history_then_steps: bool | None = None,
) -> list[LLMMessage] | None:
    """Run full-replace compaction. Returns None when not applicable."""
    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    split = split_for_full_replace(non_system)
    if split is None:
        return None
    older, last_query, recent = split

    # HistoryThenSteps: graduated shedding of recent tool steps into history.
    # 15% → trim tool bodies; 30% → fold older half of recent; 50% → fold all.
    try:
        from clawagents.config.features import is_enabled
        from clawagents.memory.compaction_segments import should_compact_steps_after_history

        hts = (
            history_then_steps
            if history_then_steps is not None
            else is_enabled("history_then_steps")
        )
        if hts and recent:
            history_tokens = _rough_message_tokens(older)
            steps_tokens = _rough_message_tokens(recent)
            ratio = (steps_tokens / history_tokens) if history_tokens > 0 else 1.0
            if ratio > 0.50 or should_compact_steps_after_history(
                history_tokens, steps_tokens, steps_trigger_ratio=0.50
            ):
                older = list(older) + list(recent)
                recent = []
            elif ratio > 0.30 or should_compact_steps_after_history(
                history_tokens, steps_tokens, steps_trigger_ratio=0.30
            ):
                mid = max(1, len(recent) // 2)
                older = list(older) + list(recent[:mid])
                recent = list(recent[mid:])
            elif ratio > 0.15:
                trimmed: list[LLMMessage] = []
                for m in recent:
                    if getattr(m, "role", None) == "tool":
                        content = m.content if isinstance(m.content, str) else str(m.content)
                        if len(content) > 800:
                            content = content[:800] + "\n…[trimmed for HistoryThenSteps]"
                            trimmed.append(
                                LLMMessage(
                                    role="tool",
                                    content=content,
                                    tool_call_id=getattr(m, "tool_call_id", None),
                                )
                            )
                            continue
                    trimmed.append(m)
                recent = trimmed
    except Exception:
        pass

    task_context = last_query[:500] if last_query else ""
    conv = render_conversation_for_summary(older, lossy=lossy)
    if not conv.strip():
        return None

    raw_summary = await sample_full_replace_summary(
        llm, conv, task_context=task_context
    )
    agents = agents_md if agents_md is not None else load_agents_md_reminder(workspace)

    parts = CompactedHistoryParts(
        system_messages=system_msgs,
        user_message_prefix=user_prefix,
        agents_md_reminder=agents,
        last_user_query=last_query,
        recent_messages=recent,
        compaction_summary=raw_summary,
        system_reminder=system_reminder,
        transcript_hint=transcript_hint,
        carryover_markdown=carryover_markdown,
    )
    assembled = assemble_compacted_history(parts)
    return sanitize_compacted_history(assembled)


__all__ = [
    "MIN_SUMMARY_SEED_CHARS",
    "CONTINUATION_PREAMBLE",
    "format_compact_summary",
    "format_compact_summary_content",
    "is_degenerate_summary",
    "neutralize_compaction_control_tokens",
    "wrap_user_query",
    "CompactedHistoryParts",
    "assemble_compacted_history",
    "split_for_full_replace",
    "apply_full_replace_compaction",
    "build_state_reminder",
    "load_agents_md_reminder",
    "sanitize_compacted_history",
]
