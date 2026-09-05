"""Context window management for the agent loop.

Handles preflight payload checks, micro-compaction, soft-trimming,
full auto-compaction, history offloading, WAL, goal reminder sync,
interject draining, and miscellaneous helpers.

Extracted from ``agent_loop.py`` for modularity.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from clawagents.providers.llm import LLMProvider, LLMMessage, NativeToolSchema
from clawagents.run_context import RunContext
from clawagents.tools.registry import ToolRegistry
from clawagents.graph.model_profiles import (
    resolve_context_budget as _resolve_context_budget,
    resolve_long_context_threshold as _resolve_long_context_threshold,
)
from clawagents.graph.tool_observation import (
    _CHARS_PER_TOKEN,
    _estimate_tokens,
    _estimate_messages_tokens,
    _truncate_old_tool_args,
)

logger = logging.getLogger(__name__)

# Type alias matching the one in agent_loop.py (avoids circular import).
OnEvent = Callable[[str, dict[str, Any]], None]


# ─── Pre-flight Context Guard ─────────────────────────────────────────────
# Runs once before the main loop to ensure the initial payload fits in the
# context window. Applies graduated shedding when the system prompt + tool
# descriptions + user task already exceed the budget.

_MAX_OVERFLOW_RETRIES = 3


def _preflight_context_check(
    messages: list[LLMMessage],
    context_window: int,
    tool_desc: str,
    native_schemas: list[NativeToolSchema] | None,
    registry: ToolRegistry | None,
    emit: OnEvent,
    model_name: Optional[str] = None,
) -> tuple[list[LLMMessage], str, list[NativeToolSchema] | None]:
    """Ensure the initial payload fits in the context budget.

    Returns (messages, tool_desc, native_schemas) — possibly modified via
    graduated shedding.

    Tiers:
      1. Truncate verbose tool parameter descriptions
      2. Drop text-based tool descriptions if native schemas are available
      3. Truncate the system prompt itself, keeping the core behavior section
    """
    effective_window, ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    budget = int(effective_window * ratio)

    native_schema_tokens = 0
    if native_schemas:
        schema_text = json.dumps([
            {"name": s.name, "description": s.description, "parameters": s.parameters}
            for s in native_schemas
        ])
        native_schema_tokens = _estimate_tokens(schema_text)

    def _payload_tokens() -> int:
        return _estimate_messages_tokens(messages) + native_schema_tokens

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    emit("context", {
        "message": f"pre-flight: initial payload ~{_payload_tokens()} tokens exceeds budget {budget}"
    })

    # ── Tier 1: Truncate parameter descriptions in tool_desc ──────────
    if tool_desc and registry:
        short_parts = ["## Available Tools\n"]
        for tool in registry.list():
            short_parts.append(f"### {tool.name}\n{tool.description}")
            if tool.parameters:
                short_parts.append("Parameters: " + ", ".join(
                    f"`{k}` ({v.get('type', 'string')}{'*' if v.get('required') else ''})"
                    for k, v in tool.parameters.items()
                ))
            short_parts.append("")
        short_desc = "\n".join(short_parts)
        sys_msg = messages[0]
        if isinstance(sys_msg.content, str):
            messages = [
                LLMMessage(role="system", content=sys_msg.content.replace(tool_desc, short_desc)),
                *messages[1:],
            ]
            tool_desc = short_desc
            emit("context", {"message": f"tier-1: shortened tool descriptions -> ~{_payload_tokens()} tokens"})
        else:
            emit("warn", {
                "message": "tier-1 shedding skipped: system message has multimodal content (list), cannot string-replace"
            })

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    # ── Tier 2: Drop text tool descriptions if native schemas exist ───
    if tool_desc and native_schemas:
        sys_msg = messages[0]
        if isinstance(sys_msg.content, str):
            messages = [
                LLMMessage(role="system", content=sys_msg.content.replace(tool_desc, "").strip()),
                *messages[1:],
            ]
            tool_desc = ""
            emit("context", {"message": f"tier-2: removed text tool descriptions -> ~{_payload_tokens()} tokens"})
        else:
            emit("warn", {
                "message": "tier-2 shedding skipped: system message has multimodal content (list), cannot string-replace"
            })

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    # ── Tier 3: Truncate system prompt, preserving core behavior ──────
    sys_content = messages[0].content
    max_sys_chars = int((budget - native_schema_tokens - _estimate_tokens(messages[1].content if len(messages) > 1 else "")) * _CHARS_PER_TOKEN * 0.8)
    if isinstance(sys_content, str):
        if max_sys_chars > 200 and len(sys_content) > max_sys_chars:
            truncated = sys_content[:max_sys_chars] + "\n\n...(system prompt truncated to fit context window)"
            messages = [LLMMessage(role="system", content=truncated), *messages[1:]]
            emit("context", {"message": f"tier-3: truncated system prompt -> ~{_payload_tokens()} tokens"})
    else:
        emit("warn", {
            "message": "tier-3 shedding skipped: system message has multimodal content (list), cannot truncate as string"
        })

    if _payload_tokens() > budget:
        emit("warn", {
            "message": (
                f"pre-flight: payload still ~{_payload_tokens()} tokens after all shedding "
                f"(budget {budget}). Consider increasing CONTEXT_WINDOW or reducing tools/instruction."
            )
        })

    return messages, tool_desc, native_schemas


# ─── Micro-Compact: clear old tool results (learned from Claude Code) ─────
# Unlike soft-trim which truncates, micro-compact completely replaces old tool
# result content with a placeholder. The model still sees the tool_use →
# tool_result structure (knows *what* it did) but not the raw output.
# This can effectively double the usable context window with zero LLM overhead.

_COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "read_file", "execute", "execute_command", "bash", "run_command",
    "grep", "glob", "ls", "tree", "web_fetch", "web_search",
    "search_files", "list_dir", "find_files",
})

_MICRO_COMPACT_KEEP_RECENT = 3  # keep last N compactable tool results intact
# Only micro-compact once the transcript actually uses a meaningful share of
# the context window. Running it unconditionally blanked all but the last 3
# read/grep/exec results every round, degrading multi-file tasks at low usage.
_MICRO_COMPACT_MIN_USAGE_RATIO = 0.4
_ARTIFACT_ID_RE = re.compile(
    r"(?:Artifact id:\s*|id=)([A-Za-z0-9._-]{4,80})",
    re.IGNORECASE,
)


def _extract_artifact_id(content: str) -> str | None:
    if not content:
        return None
    m = _ARTIFACT_ID_RE.search(content)
    return m.group(1) if m else None


# (Tool approval + side effects imported above from tool_observation)


def _micro_compact_stub(content: str, *, tool_call_id: str | None = None) -> str:
    """Replace old tool bodies with a stub that still points at a recoverable artifact."""
    aid = _extract_artifact_id(content)
    if aid is None and isinstance(content, str) and len(content) > 500:
        try:
            from clawagents.tool_output_artifacts import store_tool_artifact

            aid, _ = store_tool_artifact(
                tool_name="micro_compact",
                tool_use_id=tool_call_id or f"micro-{abs(hash(content[:200])) % 10_000_000}",
                output=content,
                kind="prose",
                extra_meta={"source": "micro_compact"},
            )
        except Exception:
            logger.debug("micro-compact artifact store failed", exc_info=True)
            aid = None
    if aid:
        return (
            f"[Old tool result cleared to save context — artifact id={aid}. "
            f"Call retrieve_tool_result(id=\"{aid}\") to restore.]"
        )
    return "[Old tool result cleared to save context]"


def _micro_compact_tool_results(
    messages: list[LLMMessage],
    keep_recent: int = _MICRO_COMPACT_KEEP_RECENT,
) -> list[LLMMessage]:
    """Clear old tool result content for compactable tools (keep last N).

    The model still sees the tool_use → tool_result pairs, just not the raw
    50KB grep/file output. This preserves the agent's sense of *what* it did
    while freeing massive amounts of context. Stubs retain artifact ids when
    available so content remains recoverable via retrieve_tool_result.
    """
    from clawagents.config.features import is_enabled
    if not is_enabled("micro_compact"):
        return messages

    # Collect compactable tool call IDs in order
    compactable_ids: list[str] = []
    # For text-based tool calls, track by message index
    compactable_text_indices: list[int] = []

    for i, msg in enumerate(messages):
        if msg.role == "assistant":
            # Native tool calls
            if msg.tool_calls_meta:
                for tc in msg.tool_calls_meta:
                    if tc.get("name", "") in _COMPACTABLE_TOOLS:
                        compactable_ids.append(tc["id"])
            # Text-based tool calls
            elif isinstance(msg.content, str):
                try:
                    import json as _json
                    parsed = _json.loads(msg.content)
                    if isinstance(parsed, dict) and parsed.get("tool") in _COMPACTABLE_TOOLS:
                        compactable_text_indices.append(i)
                    elif isinstance(parsed, list):
                        if any(isinstance(item, dict) and item.get("tool") in _COMPACTABLE_TOOLS for item in parsed):
                            compactable_text_indices.append(i)
                except (ValueError, TypeError):
                    pass

    # Keep the most recent N compactable tool results
    keep_ids = set(compactable_ids[-keep_recent:])
    keep_text_indices = set(compactable_text_indices[-keep_recent:])

    # Clear old compactable tool results
    result: list[LLMMessage] = []
    cleared = 0
    for i, msg in enumerate(messages):
        # Native tool results
        if msg.role == "tool" and msg.tool_call_id:
            if msg.tool_call_id in compactable_ids and msg.tool_call_id not in keep_ids:
                body = msg.content if isinstance(msg.content, str) else str(msg.content)
                result.append(LLMMessage(
                    role="tool",
                    content=_micro_compact_stub(body, tool_call_id=msg.tool_call_id),
                    tool_call_id=msg.tool_call_id,
                ))
                cleared += 1
                continue
        # Text-based tool results (user message following assistant tool call)
        elif msg.role == "user" and isinstance(msg.content, str) and msg.content.startswith("[Tool Result]"):
            if i > 0 and (i - 1) in compactable_text_indices and (i - 1) not in keep_text_indices:
                stub = _micro_compact_stub(msg.content)
                result.append(LLMMessage(
                    role="user",
                    content=f"[Tool Result] {stub}",
                ))
                cleared += 1
                continue

        result.append(msg)

    return result


# ─── Soft-Trim: prune stale/low-value content before compaction ───────────

_SOFT_TRIM_BUDGET_FRACTION = 0.75  # soft-trim at 75% of the compaction budget_ratio
_SOFT_TRIM_RESULT_MAX_CHARS = 1000
_SOFT_TRIM_RESULT_KEEP_CHARS = 500
_SOFT_TRIM_RECENT_PROTECTED = 10

_IMAGE_DATA_RE = re.compile(r'^\[image\s*data?\]$', re.IGNORECASE)


def _soft_trim_messages(
    messages: list[LLMMessage],
    context_window: int,
    token_multiplier: float,
    emit: OnEvent,
    model_name: Optional[str] = None,
    current_tokens: Optional[int] = None,
) -> list[LLMMessage]:
    """Remove stale/low-value content from context before hitting compaction threshold."""
    effective_window, budget_ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    soft_budget = int(effective_window * budget_ratio * _SOFT_TRIM_BUDGET_FRACTION)
    # Cap soft-trim trigger by the model's pricing long-context cliff when set
    # (e.g. Luna 272K) so we shed stale tool dumps before the 2×/1.5× premium.
    long_ctx = _resolve_long_context_threshold(model_name)
    if long_ctx:
        soft_budget = min(soft_budget, max(8_000, int(long_ctx * 0.95)))
    if current_tokens is None:
        current_tokens = _estimate_messages_tokens(messages, token_multiplier)

    if current_tokens <= soft_budget:
        return messages

    protect_from = max(0, len(messages) - _SOFT_TRIM_RECENT_PROTECTED * 2)
    trim_count = 0

    # First pass: identify duplicate tool results and mark latest index
    seen: dict[str, int] = {}
    for i, m in enumerate(messages):
        if m.role == "tool" or (m.role == "user" and isinstance(m.content, str) and m.content.startswith("[Tool Result]")):
            if i > 0:
                prev = messages[i - 1]
                if prev.role == "assistant" and isinstance(prev.content, str):
                    content_str = m.content if isinstance(m.content, str) else ""
                    key = prev.content[:200] + "|" + content_str[:200]
                    seen[key] = i

    # Second pass: trim/prune
    result: list[LLMMessage] = []
    for i, m in enumerate(messages):
        if i >= protect_from:
            result.append(m)
            continue

        is_tool_result = (
            m.role == "tool"
            or (m.role == "user" and isinstance(m.content, str) and m.content.startswith("[Tool Result]"))
        )

        if is_tool_result and isinstance(m.content, str):
            # Prune image-only tool results from early turns
            trimmed_content = m.content.replace("[Tool Result]", "", 1).strip()
            if _IMAGE_DATA_RE.match(trimmed_content):
                result.append(LLMMessage(role=m.role, content="[Tool Result] [image data removed — stale]",
                                         tool_call_id=m.tool_call_id))
                trim_count += 1
                continue

            # Remove duplicate tool results (keep only the most recent)
            if i > 0:
                prev = messages[i - 1]
                if prev.role == "assistant" and isinstance(prev.content, str):
                    key = prev.content[:200] + "|" + m.content[:200]
                    latest_idx = seen.get(key)
                    if latest_idx is not None and latest_idx != i:
                        result.append(LLMMessage(role=m.role, content="[Tool Result] [duplicate — see later result]",
                                                 tool_call_id=m.tool_call_id))
                        trim_count += 1
                        continue

            # Trim large old tool results
            if len(m.content) > _SOFT_TRIM_RESULT_MAX_CHARS:
                half = _SOFT_TRIM_RESULT_KEEP_CHARS // 2
                trimmed = (
                    m.content[:half]
                    + f"\n...[soft-trimmed {len(m.content) - _SOFT_TRIM_RESULT_KEEP_CHARS} chars]...\n"
                    + m.content[-half:]
                )
                result.append(LLMMessage(role=m.role, content=trimmed, tool_call_id=m.tool_call_id))
                trim_count += 1
                continue

        result.append(m)

    if trim_count == 0:
        return messages
    emit("context", {"message": f"soft-trim: trimmed {trim_count} old tool results"})
    return result


# ─── Context Window Guard with Auto-Compaction ────────────────────────────

_CONTEXT_BUDGET_RATIO = 0.75
_RECENT_MESSAGES_TO_KEEP = 20
_COMPACTION_CHUNK_TOKENS = 30_000
_COMPACTION_MAX_RETRIES = 3

_IDENTIFIER_PRESERVATION = """
CRITICAL: Preserve these verbatim (do not paraphrase or omit):
- File paths (e.g., src/utils/auth.ts)
- Function/variable/class names (e.g., handleAuth, userToken)
- Error messages and stack traces
- Command-line commands that were run
- Configuration values and URLs"""


def _find_safe_split_index(non_system: list[LLMMessage], desired_recent: int) -> int:
    """Find a split index that doesn't break tool_call/tool_result pairs.

    Walks backward from the desired split point until we find a boundary
    that doesn't land between an assistant tool_call and its tool result.
    """
    split = max(0, len(non_system) - desired_recent)
    # Bound is < len(non_system), NOT len - 1: with the tighter bound a tail
    # run of ≥N tool messages left the last orphan tool result in `recent`
    # while its paired assistant tool_call got summarized away → provider 400.
    while split < len(non_system):
        msg = non_system[split]
        if msg.role == "tool" and msg.tool_call_id:
            split += 1
            continue
        break
    return split


async def _summarize_chunk(
    llm: LLMProvider,
    chunk_text: str,
    task_context: str,
) -> str:
    """Summarize a single chunk with retry and exponential backoff."""
    prompt = (
        "You are summarizing a chunk of an AI agent's conversation history.\n\n"
        f"## Original Task\n{task_context}\n\n"
        f"## Conversation Chunk\n{chunk_text}\n\n"
        "## Instructions\n"
        "Write a structured summary preserving:\n"
        "- What tools were called and their key results (file paths, data, errors)\n"
        "- What has been accomplished\n"
        "- Any critical facts, variable values, or decisions made\n"
        + _IDENTIFIER_PRESERVATION + "\n"
        "Be concise but preserve all actionable information."
    )

    last_error: BaseException | None = None
    for attempt in range(_COMPACTION_MAX_RETRIES):
        try:
            resp = await llm.chat([LLMMessage(role="user", content=prompt)])
            if resp.content.strip():
                return resp.content.strip()
        except Exception as e:
            last_error = e
        if attempt < _COMPACTION_MAX_RETRIES - 1:
            await asyncio.sleep(1.0 * (2 ** attempt))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Summarization returned empty")


def _content_key_text(content: Any) -> str:
    """Stable text stand-in for message content (compaction input + reuse keys).

    Multimodal list content must not go through ``str()`` — that dumps the
    full base64 data URL (megabytes) into the summarizer prompt. Join the
    real text parts and replace each image with a short digest placeholder;
    the digest keeps reuse keys distinct per distinct image so compaction's
    original-message reuse can't swap two same-text messages.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append(str(p.get("text", "") or ""))
        elif p.get("type") in ("image_url", "image"):
            digest = hashlib.sha1(
                json.dumps(p, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:8]
            parts.append(f"[image attachment #{digest}]")
        elif p.get("type") in ("file", "document"):
            digest = hashlib.sha1(
                json.dumps(p, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:8]
            parts.append(f"[file attachment #{digest}]")
    return "\n".join(parts)


def _is_compactable_user(msg: LLMMessage) -> bool:
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


_FILE_PATH_RE = re.compile(
    r"""(?:write_file|edit_file|apply_patch|read_file)[^\"']*[\"']([^\"']+)[\"']"""
    r"""|path[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']""",
    re.I,
)


def _extract_recent_files(messages: list[LLMMessage], *, limit: int = 12) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in messages:
        blob = ""
        if isinstance(m.content, str):
            blob = m.content
        if getattr(m, "tool_calls_meta", None):
            try:
                blob += " " + json.dumps(m.tool_calls_meta, default=str)
            except TypeError:
                pass
        for match in _FILE_PATH_RE.finditer(blob):
            path = next((g for g in match.groups() if g), None)
            if not path or path in seen:
                continue
            if any(ch in path for ch in ("\n", " ")):
                continue
            seen.add(path)
            found.append(path)
            if len(found) >= limit:
                return found
    return found


def _message_reuse_key(m: LLMMessage) -> tuple[Any, ...]:
    """Disambiguate reuse keys so empty-content tool calls cannot swap meta.

    Matching only on ``(role, content)`` swapped ``tool_calls_meta`` between
    unrelated assistants that both had ``content=None`` / ``""``.
    """
    role = m.role
    content = _content_key_text(m.content)
    if role == "tool":
        return (role, content, str(m.tool_call_id or ""), ())
    meta = getattr(m, "tool_calls_meta", None) or []
    if role == "assistant" and meta:
        ids = tuple(str(tc.get("id") or "") for tc in meta if isinstance(tc, dict))
        names = tuple(str(tc.get("name") or "") for tc in meta if isinstance(tc, dict))
        return (role, content, ids, names)
    return (role, content, "", ())


def _reuse_messages_where_possible(
    originals: list[LLMMessage],
    rebuilt: list[LLMMessage],
) -> list[LLMMessage]:
    """Prefer original LLMMessage object identity when reuse keys match.

    Required for session-persistence trackers that key off identity.
    """
    buckets: dict[tuple[Any, ...], list[LLMMessage]] = {}
    for om in originals:
        buckets.setdefault(_message_reuse_key(om), []).append(om)
    out: list[LLMMessage] = []
    for m in rebuilt:
        bucket = buckets.get(_message_reuse_key(m))
        if bucket:
            out.append(bucket.pop())
        else:
            out.append(m)
    return out


def _goal_llm_complete(run_context: Any, llm: LLMProvider):
    """Bind a prompt→text callable for goal planner/verifier/strategist."""

    async def _complete(prompt: str) -> str:
        meta = getattr(run_context, "_metadata", None) if run_context else None
        if isinstance(meta, dict) and callable(meta.get("goal_llm_complete")):
            return await meta["goal_llm_complete"](prompt)
        resp = await llm.chat([LLMMessage(role="user", content=prompt)])
        return str(getattr(resp, "content", "") or "")

    return _complete


def _drain_interject(run_context: Any) -> str | None:
    """Legacy single-string drain — prefer :func:`drain_interject_messages`."""
    from clawagents.interjection import drain_interjects

    parts = drain_interjects(run_context)
    if not parts:
        return None
    # Compat: join only if caller expects one blob (prefer multi-message path).
    return parts[0] if len(parts) == 1 else "\n\n".join(parts)


def _drain_interject_messages(run_context: Any) -> list[LLMMessage]:
    """Each pending interject → one standalone synthetic user turn (Grok parity)."""
    from clawagents.interjection import drain_interjects

    return [LLMMessage(role="user", content=text) for text in drain_interjects(run_context)]


_GOAL_REMINDER_START = "\n\n<!--claw:goal-reminder-->\n"
_GOAL_REMINDER_END = "\n<!--/claw:goal-reminder-->"


def _strip_goal_reminder(system_content: str) -> str:
    if not isinstance(system_content, str):
        return system_content
    start = system_content.find("<!--claw:goal-reminder-->")
    if start < 0:
        # Legacy unwrapped block from first-turn injection
        marker = "\n## Active Goal\n"
        idx = system_content.find(marker)
        if idx < 0:
            return system_content
        return system_content[:idx].rstrip()
    # Include any blank lines immediately before the marker
    while start > 0 and system_content[start - 1] == "\n":
        start -= 1
        if start > 0 and system_content[start - 1] == "\n":
            break
    end = system_content.find("<!--/claw:goal-reminder-->", start)
    if end < 0:
        return system_content[:start].rstrip()
    end += len("<!--/claw:goal-reminder-->")
    return (system_content[:start] + system_content[end:]).rstrip()


def _sync_goal_reminder_into_system(
    messages: list[LLMMessage],
    run_context: Any,
) -> None:
    """Keep Active Goal standing reminder fresh when start_goal runs mid-loop."""
    if not messages or getattr(messages[0], "role", None) != "system":
        return
    content = messages[0].content
    if not isinstance(content, str):
        return
    try:
        from clawagents.config.features import is_enabled as _feat_goal_sys
        from clawagents.goal import get_goal_tracker, goal_system_reminder

        meta = getattr(run_context, "_metadata", None)
        if not (isinstance(meta, dict) and meta.get("goal_mode")):
            return
        if not _feat_goal_sys("goal_autopilot"):
            return
        tracker = get_goal_tracker(run_context)
        rem = goal_system_reminder(tracker.state if tracker else None)
    except Exception:
        return
    base = _strip_goal_reminder(content)
    if rem:
        messages[0].content = base + _GOAL_REMINDER_START + rem + _GOAL_REMINDER_END
    else:
        messages[0].content = base


async def _compact_if_needed(
    messages: list[LLMMessage],
    context_window: int,
    llm: LLMProvider,
    emit: OnEvent,
    token_multiplier: float = 1.0,
    model_name: Optional[str] = None,
    run_context: Optional[RunContext] = None,
    fire_hook: Optional[Callable[..., Any]] = None,
    savings_history: list[float] | None = None,
    taxonomy_dispatcher: Any | None = None,
) -> list[LLMMessage]:
    messages = _truncate_old_tool_args(messages)

    # Soft-cap verbose assistant/user turns before heavier compaction.
    try:
        from clawagents.memory.output_trim import trim_verbose_messages

        messages, trimmed_n = trim_verbose_messages(messages)
        if trimmed_n:
            emit("context", {"message": f"trimmed {trimmed_n} verbose turn(s)"})
    except Exception:
        logger.debug("output trim failed", exc_info=True)

    # If recent compressions are thrashing, prefer artifact eviction only.
    if savings_history:
        try:
            from clawagents.memory.compaction import is_compression_thrashing

            if is_compression_thrashing(savings_history):
                emit("context", {
                    "message": "compaction thrashing detected — skipping LLM summarize; soft-trim only",
                })
                return messages
        except Exception:
            logger.debug("thrash check failed", exc_info=True)

    effective_window, ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    budget = int(effective_window * ratio)
    from clawagents.memory.compact_tool_results import compact_tool_results
    from clawagents.harness_profiles import resolve_harness_profile

    profile = resolve_harness_profile(model_name)
    headroom = (
        float(profile.compaction_headroom_ratio)
        if profile and profile.compaction_headroom_ratio is not None
        else 0.7
    )

    messages, compacted = compact_tool_results(
        messages,
        max_input_tokens=budget,
        token_multiplier=token_multiplier,
        headroom_ratio=headroom,
    )
    if compacted:
        emit("context", {"message": "compacted oversized tool results before summarization"})
    current_tokens = _estimate_messages_tokens(messages, token_multiplier)

    # Pre-compaction memory flush (Grok memory_flush)
    try:
        from clawagents.config.features import is_enabled as _feat_flush
        from clawagents.memory.memory_flush import should_flush, run_memory_flush

        cycle = 0
        if run_context is not None and isinstance(run_context._metadata, dict):
            cycle = int(run_context._metadata.get("compaction_cycle") or 0)
        ws = None
        if run_context is not None and isinstance(run_context._metadata, dict):
            ws = run_context._metadata.get("workspace")
        if _feat_flush("memory_flush") and should_flush(
            current_tokens, budget, compaction_cycle=cycle, workspace=ws
        ):
            async def _flush_llm(prompt: str) -> str:
                resp = await llm.chat([LLMMessage(role="user", content=prompt)])
                return str(getattr(resp, "content", "") or "")

            flush_out = await run_memory_flush(
                messages, _flush_llm, workspace=ws, compaction_cycle=cycle
            )
            emit(
                "context",
                {
                    "message": (
                        f"memory flush: {flush_out.status}"
                        + (f" ({flush_out.detail})" if flush_out.detail else "")
                    )
                },
            )
            if run_context is not None and isinstance(run_context._metadata, dict):
                run_context._metadata["compaction_cycle"] = cycle + 1
    except Exception:
        logger.debug("memory flush failed", exc_info=True)

    # Prefire / two-pass: summarize before the hard cliff (Grok two_pass).
    try:
        from clawagents.config.features import is_enabled as _feat_prefire

        prefire_ratio = 0.85
        if (
            _feat_prefire("prefire_compaction")
            and current_tokens > int(budget * prefire_ratio)
            and current_tokens <= budget
        ):
            emit(
                "context",
                {
                    "message": (
                        f"prefire compaction ~{current_tokens}/{budget} "
                        f"(>{int(prefire_ratio * 100)}% headroom)"
                    )
                },
            )
            # Force into the compaction path below by pretending we're over budget
            # only for the summarize stage — callers still see a successful shrink.
            current_tokens = budget + 1
    except Exception:
        logger.debug("prefire compaction probe failed", exc_info=True)

    if current_tokens <= budget:
        return messages

    emit("context", {"message": f"~{current_tokens} tokens exceeds budget {budget} — compacting"})
    emit("compact_progress", {
        "phase": "start",
        "message": "context budget exceeded; compacting older turns",
        "current_tokens": current_tokens,
        "budget": budget,
        "message_count": len(messages),
    })

    if fire_hook is not None:
        try:
            await fire_hook("on_pre_compact", len(messages), current_tokens)
        except Exception:
            logger.debug("on_pre_compact hook failed", exc_info=True)

    if taxonomy_dispatcher is not None:
        try:
            from clawagents.hooks.external import dispatch_taxonomy_hook
            from clawagents.hooks.taxonomy import HookEvent

            await dispatch_taxonomy_hook(
                taxonomy_dispatcher,
                HookEvent.PRE_COMPACT,
                {
                    "message_count": len(messages),
                    "current_tokens": current_tokens,
                    "budget": budget,
                },
                blocking=False,
            )
        except Exception:
            logger.debug("taxonomy pre_compact hook failed", exc_info=True)

    system_msgs: list[LLMMessage] = []
    non_system: list[LLMMessage] = []
    for m in messages:
        (system_msgs if m.role == "system" else non_system).append(m)

    if len(non_system) <= _RECENT_MESSAGES_TO_KEEP:
        return messages

    # ── Grok-style full-replace (preferred when enabled) ───────────────
    try:
        from clawagents.config.features import is_enabled as _feat

        if _feat("full_replace_compaction"):
            from clawagents.memory.full_replace_compaction import (
                apply_full_replace_compaction,
                build_state_reminder,
            )
            from clawagents.context.carryover import (
                get_compaction_carryover,
                set_compaction_carryover,
            )

            workspace = None
            if run_context is not None:
                ws = run_context._metadata.get("workspace")
                if isinstance(ws, str):
                    workspace = ws
            if not workspace:
                workspace = os.getcwd()

            # Auto-enrich carryover from transcript signals when host didn't set it
            try:
                task_focus = ""
                for m in non_system:
                    if m.role == "user" and isinstance(m.content, str) and _is_compactable_user(m):
                        task_focus = m.content[:500]
                        break
                recent_files = _extract_recent_files(non_system)
                existing = get_compaction_carryover(run_context, task_context=task_focus)
                active = list(getattr(run_context, "active_skills", {}) or {})
                invoked = list(existing.invoked_skills) or active
                for name in active:
                    if name not in invoked:
                        invoked.append(name)
                if run_context is not None and (
                    not existing.recent_files
                    or not existing.task_focus
                    or (active and not existing.invoked_skills)
                ):
                    set_compaction_carryover(
                        run_context,
                        task_focus=existing.task_focus or task_focus or None,
                        recent_files=existing.recent_files or recent_files,
                        recent_work_log=existing.recent_work_log,
                        invoked_skills=invoked,
                        active_workers=existing.active_workers,
                        channel_log=existing.channel_log,
                        plan_reminder=existing.plan_reminder,
                        metadata=existing.metadata,
                    )
                carryover = get_compaction_carryover(run_context, task_context=task_focus)
                carryover_md = carryover.to_markdown()
                reminder = build_state_reminder(
                    recent_files=carryover.recent_files,
                    plan_text=carryover.plan_reminder,
                    invoked_skills=carryover.invoked_skills,
                    active_workers=carryover.active_workers,
                )
            except Exception:
                logger.debug("full-replace carryover enrich failed", exc_info=True)
                carryover_md = ""
                reminder = None

            fr = await apply_full_replace_compaction(
                messages,
                llm,
                workspace=workspace,
                carryover_markdown=carryover_md or None,
                system_reminder=reminder,
                history_then_steps=_feat("history_then_steps"),
            )
            if fr is not None:
                # Prefer identity reuse for system + recent tails that survived
                fr = _reuse_messages_where_possible(messages, fr)
                fr_tokens = _estimate_messages_tokens(fr, token_multiplier)
                # Input ladder: if still over budget, retry lossy summarizer input
                if fr_tokens > budget:
                    fr_lossy = await apply_full_replace_compaction(
                        messages,
                        llm,
                        workspace=workspace,
                        carryover_markdown=carryover_md or None,
                        system_reminder=reminder,
                        lossy=True,
                        history_then_steps=_feat("history_then_steps"),
                    )
                    if fr_lossy is not None:
                        fr = _reuse_messages_where_possible(messages, fr_lossy)
                        fr_tokens = _estimate_messages_tokens(fr, token_multiplier)
                if fr_tokens <= budget or fr_tokens < current_tokens:
                    if savings_history is not None and current_tokens > 0:
                        saved = max(0, current_tokens - fr_tokens)
                        savings_history.append(saved / current_tokens * 100.0)
                    emit("context", {
                        "message": (
                            f"full-replace compaction rebuilt history "
                            f"(~{current_tokens} → ~{fr_tokens} tokens)"
                        ),
                    })
                    emit("compact_progress", {
                        "phase": "end",
                        "message": "compaction completed via full_replace",
                        "mode": "full_replace",
                        "before_tokens": current_tokens,
                        "after_tokens": fr_tokens,
                    })
                    if fire_hook is not None:
                        try:
                            summary_snip = next(
                                (
                                    m.content
                                    for m in fr
                                    if isinstance(m.content, str)
                                    and "being continued" in m.content
                                ),
                                None,
                            )
                            await fire_hook("on_post_compact", len(fr), summary_snip)
                        except Exception:
                            logger.debug("on_post_compact hook failed", exc_info=True)
                    if taxonomy_dispatcher is not None:
                        try:
                            from clawagents.hooks.external import dispatch_taxonomy_hook
                            from clawagents.hooks.taxonomy import HookEvent

                            await dispatch_taxonomy_hook(
                                taxonomy_dispatcher,
                                HookEvent.POST_COMPACT,
                                {
                                    "message_count": len(fr),
                                    "before_tokens": current_tokens,
                                    "after_tokens": fr_tokens,
                                    "mode": "full_replace",
                                },
                                blocking=False,
                            )
                        except Exception:
                            logger.debug("taxonomy post_compact hook failed", exc_info=True)
                    # Greppable compaction segment archive
                    try:
                        from clawagents.config.features import is_enabled as _feat_seg
                        from clawagents.memory.compaction_segments import (
                            write_segment,
                            segment_recovery_hint,
                        )

                        if _feat_seg("compaction_segments") and workspace:
                            archive = "\n".join(
                                f"[{m.role}] {str(m.content)[:500]}"
                                for m in messages
                                if getattr(m, "role", None) != "system"
                            )[:12000]
                            write_segment(
                                archive,
                                workspace=workspace,
                                turns=max(1, len(messages) - 1),
                            )
                            emit("context", {"message": segment_recovery_hint()})
                    except Exception:
                        logger.debug("compaction segment write failed", exc_info=True)
                    return fr
    except Exception:
        logger.debug("full_replace_compaction path failed; falling back", exc_info=True)

    # Prefer hardened compress_messages_safe when it yields meaningful savings.
    try:
        from clawagents.memory.compaction import AgentMessage, compress_messages_safe

        agent_msgs = [
            AgentMessage(
                role=m.role,
                content=_content_key_text(m.content),
            )
            for m in ([*system_msgs, *non_system])
        ]
        safe = await compress_messages_safe(
            llm,
            agent_msgs,
            context_window=effective_window,
            protect_first_n=max(1, len(system_msgs)),
            protect_last_n=_RECENT_MESSAGES_TO_KEEP,
        )
        if safe.get("effective"):
            # Rebuilding from AgentMessage would mint new objects for every
            # turn — breaking the identity-based session-persistence tracker
            # (unpersisted turns silently vanish) and stripping tool-call
            # metadata (tool_calls_meta / tool_call_id) that providers need
            # for transcript linkage. Reuse the original LLMMessage object
            # whenever (role, content) survived compression unchanged.
            # AgentMessage views only carry role+content — empty assistant/tool
            # bodies are ambiguous and must not reuse originals (that was the
            # tool_calls_meta swap). Non-empty content still reuses by text.
            _originals_by_key: dict[tuple[str, str], list[LLMMessage]] = {}
            for _om in (*system_msgs, *non_system):
                _text = _content_key_text(_om.content)
                if _om.role in ("assistant", "tool") and not str(_text or "").strip():
                    continue  # never offer empty bodies for reuse
                _originals_by_key.setdefault((_om.role, _text), []).append(_om)

            def _reuse_original(role: str, content: str) -> LLMMessage:
                text = content or ""
                if role in ("assistant", "tool") and not text.strip():
                    return LLMMessage(role=role, content=text)
                bucket = _originals_by_key.get((role, _content_key_text(text)))
                if bucket:
                    return bucket.pop()
                return LLMMessage(role=role, content=text)

            compact_out = [
                _reuse_original(m.role, m.content or "")
                for m in safe["messages"]
            ]
            summary_text = str(safe.get("summary") or "")
            if savings_history is not None:
                savings_history.append(float(safe.get("compression_savings_pct") or 0.0))
            # Preserve carryover, then normalize to user+assistant compaction pair.
            try:
                task_context = ""
                for m in non_system:
                    if m.role == "user" and not (
                        isinstance(m.content, str) and m.content.startswith("[Tool Result]")
                    ):
                        task_context = m.content[:500] if isinstance(m.content, str) else ""
                        break
                carryover = get_compaction_carryover(run_context, task_context=task_context)
                carryover_text = carryover.to_markdown()
            except Exception:
                logger.debug("carryover enrich after compress_messages_safe failed", exc_info=True)
                carryover_text = ""

            handoff = f"[System — Compacted History]\n{summary_text}"
            if carryover_text and summary_text:
                handoff = (
                    f"[System — Compacted History]\n{carryover_text}\n\n"
                    f"## Conversation Summary\n{summary_text}"
                )
            replaced = False
            for i, m in enumerate(compact_out):
                if m.role != "system" and (m.content or "") == summary_text:
                    compact_out[i] = LLMMessage(role="user", content=handoff)
                    replaced = True
                    break
            if not replaced and summary_text:
                insert_at = len([m for m in compact_out if m.role == "system"])
                compact_out.insert(insert_at, LLMMessage(role="user", content=handoff))
            # Assistant ack keeps providers that expect alternating roles happy.
            if summary_text and not any(
                m.role == "assistant"
                and isinstance(m.content, str)
                and "compacted handoff" in m.content.lower()
                for m in compact_out
            ):
                # Insert immediately after the handoff user message.
                for i, m in enumerate(compact_out):
                    if m.role == "user" and isinstance(m.content, str) and "Compacted History" in m.content:
                        compact_out.insert(
                            i + 1,
                            LLMMessage(
                                role="assistant",
                                content="Understood — continuing from the compacted handoff summary.",
                            ),
                        )
                        break
            compacted_tokens = _estimate_messages_tokens(compact_out, token_multiplier)
            if compacted_tokens <= budget:
                emit("context", {
                    "message": (
                        f"compress_messages_safe saved "
                        f"{safe.get('compression_savings_pct', 0):.1f}%"
                    ),
                })
                emit("compact_progress", {
                    "phase": "end",
                    "message": "compaction completed via compress_messages_safe",
                    "older_messages": len(safe.get("dropped_messages_list") or []),
                    "recent_messages": _RECENT_MESSAGES_TO_KEEP,
                })
                if fire_hook is not None:
                    try:
                        await fire_hook("on_post_compact", len(compact_out), summary_text or None)
                    except Exception:
                        logger.debug("on_post_compact hook failed", exc_info=True)
                return compact_out
            # "Effective" savings alone are not enough: the transcript is
            # still over budget, and returning here would hand the next LLM
            # call an oversized context. Keep the lossless savings and
            # escalate to the summarization tier below.
            emit("context", {
                "message": (
                    f"compress_messages_safe saved "
                    f"{safe.get('compression_savings_pct', 0):.1f}% but "
                    f"~{compacted_tokens} tokens still exceeds budget {budget} "
                    "— escalating to summarization"
                ),
            })
            messages = compact_out
            system_msgs = [m for m in compact_out if m.role == "system"]
            non_system = [m for m in compact_out if m.role != "system"]
    except Exception:
        logger.debug("compress_messages_safe path failed; falling back", exc_info=True)

    split_idx = _find_safe_split_index(non_system, _RECENT_MESSAGES_TO_KEEP)
    if split_idx <= 0:
        return messages

    older = non_system[:split_idx]
    recent = non_system[split_idx:]

    task_context = ""
    for m in non_system:
        if m.role == "user" and not (isinstance(m.content, str) and m.content.startswith("[Tool Result]")):
            task_context = m.content[:500] if isinstance(m.content, str) else ""
            break
    carryover = get_compaction_carryover(run_context, task_context=task_context)

    _archive_pre_compact_transcript(older, task_context)

    offload_path = _offload_history(older)
    if offload_path:
        emit("context", {"message": f"offloaded {len(older)} messages to {offload_path}"})

    text_parts: list[str] = []
    for m in older:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if m.role == "assistant" and m.tool_calls_meta:
            calls = ", ".join(tc["name"] for tc in m.tool_calls_meta)
            text_parts.append(f"[TOOL CALLS: {calls}] {content[:200]}")
        elif m.role == "tool":
            text_parts.append(f"[TOOL RESULT]: {content[:200]}")
        else:
            text_parts.append(f"[{m.role.upper()}]: {content[:500]}")

    total_tokens = _estimate_tokens("\n\n".join(text_parts), token_multiplier)

    try:
        emit("compact_progress", {
            "phase": "summarize",
            "message": "summarizing compacted turns",
            "older_messages": len(older),
            "recent_messages": len(recent),
            "carryover": carryover.to_dict(),
        })
        if total_tokens <= _COMPACTION_CHUNK_TOKENS:
            text_log = "\n\n".join(text_parts)
            summary_text = await _summarize_chunk(llm, text_log, task_context)
        else:
            chunks: list[str] = []
            current_chunk: list[str] = []
            current_chunk_tokens = 0

            for part in text_parts:
                part_tokens = _estimate_tokens(part, token_multiplier)
                if current_chunk_tokens + part_tokens > _COMPACTION_CHUNK_TOKENS and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_chunk_tokens = 0
                current_chunk.append(part)
                current_chunk_tokens += part_tokens
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            emit("context", {
                "message": f"splitting {len(text_parts)} parts into {len(chunks)} chunks for summarization",
            })
            emit("compact_progress", {
                "phase": "chunk",
                "message": "splitting older turns into summary chunks",
                "chunks": len(chunks),
                "older_messages": len(older),
                "recent_messages": len(recent),
            })

            chunk_summaries: list[str] = []
            for i, chunk in enumerate(chunks):
                chunk_summary = await _summarize_chunk(llm, chunk, task_context)
                chunk_summaries.append(f"### Chunk {i + 1}/{len(chunks)}\n{chunk_summary}")
            summary_text = "\n\n".join(chunk_summaries)

        if not summary_text.strip():
            emit("context", {"message": "compaction returned empty summary — dropping oldest"})
            emit("compact_progress", {
                "phase": "dropped",
                "message": "empty compaction summary; dropped older turns",
                "older_messages": len(older),
                "recent_messages": len(recent),
                "carryover": carryover.to_dict(),
            })
            out = [*system_msgs, *recent]
            if fire_hook is not None:
                try:
                    await fire_hook("on_post_compact", len(out), None)
                except Exception:
                    logger.debug("on_post_compact hook failed", exc_info=True)
            return out

        carryover_text = carryover.to_markdown()
        content = f"[System — Compacted History]\n{summary_text}"
        if carryover_text:
            content = f"[System — Compacted History]\n{carryover_text}\n\n## Conversation Summary\n{summary_text}"
        summary = LLMMessage(
            role="user",
            content=content,
        )
        emit("context", {"message": f"compacted {len(older)} messages into summary"})
        emit("compact_progress", {
            "phase": "end",
            "message": "compaction completed",
            "older_messages": len(older),
            "recent_messages": len(recent),
            "carryover": carryover.to_dict(),
        })
        out = [*system_msgs, summary, *recent]
        if fire_hook is not None:
            try:
                await fire_hook("on_post_compact", len(out), summary_text)
            except Exception:
                logger.debug("on_post_compact hook failed", exc_info=True)
        return out
    except Exception:
        logger.debug("Compaction LLM call failed", exc_info=True)
        emit("context", {"message": "compaction failed — dropping oldest messages"})
        emit("compact_progress", {
            "phase": "failed",
            "message": "compaction failed; dropped older turns",
            "older_messages": len(older),
            "recent_messages": len(recent),
            "carryover": carryover.to_dict(),
        })
        out = [*system_msgs, *recent]
        if fire_hook is not None:
            try:
                await fire_hook("on_post_compact", len(out), None)
            except Exception:
                logger.debug("on_post_compact hook failed", exc_info=True)
        return out


# ─── History Offloading ───────────────────────────────────────────────────


def _get_history_dir() -> Path:
    return Path.cwd() / ".clawagents" / "history"


def _archive_pre_compact_transcript(older_messages: list[LLMMessage], task_context: str) -> None:
    """Archive full messages to a markdown file before compaction (feature-gated)."""
    from clawagents.config.features import is_enabled
    if not is_enabled("transcript_archival"):
        return

    try:
        transcript_dir = Path.cwd() / ".clawagents" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = transcript_dir / f"pre_compact_{ts}_{len(older_messages)}msgs.md"

        lines: list[str] = [
            "## Pre-Compact Transcript\n",
            f"\nTask: {task_context}\n",
            "\n### Messages\n\n",
        ]
        for m in older_messages:
            content = _content_key_text(m.content)
            lines.append(f"**{m.role}**: {content[:2000]}\n\n")

        path.write_text("".join(lines), "utf-8")
    except Exception:
        logger.debug("Pre-compact transcript archival failed", exc_info=True)


def _offload_history(messages: list[LLMMessage]) -> str | None:
    """Save older messages to a JSON file before compaction.

    Content is passed through :func:`redact_obj` first — the offload file
    is a plain-text artifact on disk, so secrets the agent saw mid-run
    (bearer tokens, ``.env`` contents, …) must not be persisted verbatim,
    matching the redaction applied by every other persistence surface.
    """
    try:
        from clawagents.redact import redact_obj

        _get_history_dir().mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = _get_history_dir() / f"compacted_{ts}_{len(messages)}msgs.json"
        data = redact_obj([{"role": m.role, "content": m.content} for m in messages])
        path.write_text(json.dumps(data, indent=2), "utf-8")
        return str(path)
    except Exception:
        logger.debug("History offload failed", exc_info=True)
        return None


# ─── Write-Ahead Log (learned from Claude Code) ──────────────────────────
# Persist the latest message before each LLM API call so that if the process
# crashes mid-call, the user's last message isn't lost.


def _wal_write(messages: list[LLMMessage]) -> None:
    """Append the latest message to the WAL file for crash recovery."""
    from clawagents.config.features import is_enabled
    if not is_enabled("wal"):
        return

    try:
        wal_path = Path.cwd() / ".clawagents" / "wal.jsonl"
        wal_path.parent.mkdir(parents=True, exist_ok=True)
        last_msg = messages[-1] if messages else None
        if not last_msg:
            return
        content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        entry = json.dumps({
            "role": last_msg.role,
            "content": content[:500],
            "ts": time.time(),
            "msg_count": len(messages),
        })
        with open(wal_path, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # WAL failure should never block the agent loop


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_buffer():
    buf: list[str] = []
    def on_chunk(chunk: str) -> None:
        buf.append(chunk)
    return buf, on_chunk


# ─── Truncated JSON Detection ─────────────────────────────────────────────

_TRUNCATED_JSON_RE = re.compile(r'\{\s*"tool"\s*:', re.DOTALL)


def _looks_like_truncated_json(text: str) -> bool:
    """Detect if text looks like a JSON tool call that was cut off mid-output."""
    stripped = text.strip()
    if not stripped:
        return False
    if not _TRUNCATED_JSON_RE.search(stripped):
        return False
    # Has what looks like a tool call but doesn't parse as valid JSON
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, (dict, list)):
            return False  # Valid JSON — not truncated
    except json.JSONDecodeError:
        pass
    # Check for fence-wrapped truncated JSON
    for m in re.finditer(r'```(?:json)?\s*\n?(.*?)(?:```|$)', stripped, re.DOTALL):
        inner = m.group(1).strip()
        if _TRUNCATED_JSON_RE.search(inner):
            try:
                json.loads(inner)
                return False
            except json.JSONDecodeError:
                return True
    return True
