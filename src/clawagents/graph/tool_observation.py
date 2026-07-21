"""Tool observation formatting, eviction, side effects, and argument truncation.

Stateless utility functions consumed by the main agent loop. Everything
here is import-safe — no circular dependency on ``agent_loop.py``.

Extracted from ``agent_loop.py`` for modularity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from clawagents.providers.llm import LLMMessage
from clawagents.run_context import RunContext
from clawagents.tools.registry import ToolResult
from clawagents.tokenizer import (
    count_tokens_content,
    count_messages_tokens as _count_messages_tokens,
)

logger = logging.getLogger(__name__)


# ─── Tool Result Eviction (learned from deepagents) ───────────────────────
# When tool output exceeds a threshold, write the full result to a file and
# replace it with a head/tail preview + file path.

_EVICTION_CHARS_THRESHOLD = 80_000  # ~20K tokens
def _get_eviction_dir() -> Path:
    return Path.cwd() / ".clawagents" / "large_results"


_PREVIEW_MAX_CHARS = 2000

def _create_content_preview(content: str, head_lines: int = 5, tail_lines: int = 5) -> str:
    lines = content.split("\n")
    if len(lines) <= head_lines + tail_lines + 2 and len(content) <= _PREVIEW_MAX_CHARS:
        return content

    if len(lines) <= head_lines + tail_lines + 2:
        half = _PREVIEW_MAX_CHARS // 2
        return (content[:half]
                + f"\n... [{len(content) - _PREVIEW_MAX_CHARS} chars truncated] ...\n"
                + content[-half:])

    head = "\n".join(
        f"{i + 1}: {line}" for i, line in enumerate(lines[:head_lines])
    )
    total = len(lines)
    tail = "\n".join(
        f"{total - tail_lines + i + 1}: {line}"
        for i, line in enumerate(lines[-tail_lines:])
    )
    omitted = total - head_lines - tail_lines
    return f"{head}\n... [{omitted} lines truncated] ...\n{tail}"


def _evict_large_tool_result(tool_name: str, output: str) -> str:
    if len(output) < _EVICTION_CHARS_THRESHOLD:
        return output

    try:
        _get_eviction_dir().mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)
        file_path = _get_eviction_dir() / f"{sanitized}_{ts}.txt"
        file_path.write_text(output, "utf-8")

        preview = _create_content_preview(output)
        return (
            f"[Result too large ({len(output)} chars) — saved to {file_path}]\n"
            f"Use read_file to access the full result. Preview:\n\n{preview}"
        )
    except Exception:
        half = _EVICTION_CHARS_THRESHOLD // 2
        return (
            output[:half]
            + f"\n\n... [truncated {len(output) - _EVICTION_CHARS_THRESHOLD} chars] ...\n\n"
            + output[-half:]
        )


# Hosts (VS Code gateway) accept up to ~8KB of tool_completed text. Keep the
# model/console ``preview_chars`` small, but stream a longer UI-facing body.
UI_TOOL_RESULT_CHARS = 8_000


def _format_failed_exec_observation(payload: str) -> str | None:
    """Reorder execute failures: exit/stderr/stdout before the long command."""
    text = (payload or "").strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not data.get("command_executed"):
        return None
    parts: list[str] = [
        f"Command exited with code {data.get('exit_code', '?')}",
    ]
    stderr = str(data.get("stderr") or "").strip()
    stdout = str(data.get("stdout") or "").strip()
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    cmd = str(data.get("command") or "").strip()
    if cmd:
        if len(cmd) > 240:
            cmd = cmd[:240] + "…"
        parts.append(f"command: {cmd}")
    interp = str(data.get("interpretation") or "").strip()
    if interp:
        parts.append(interp)
    warning = str(data.get("warning") or "").strip()
    if warning:
        parts.append(warning)
    return "\n\n".join(parts)


def _tool_observation(result: ToolResult) -> str | list[dict[str, Any]]:
    """Observation text for the model — prefers full ``raw_output`` when present."""
    payload = getattr(result, "raw_output", None)
    if payload is None:
        payload = result.output
    if result.success:
        return payload
    if isinstance(payload, list):
        error = f"Error: {result.error}" if result.error else "Error: Tool failed"
        return [{"type": "text", "text": error}, *payload]
    output = str(payload or "").strip()
    exec_obs = _format_failed_exec_observation(output)
    if exec_obs is not None:
        return exec_obs
    error = f"Error: {result.error}" if result.error else "Error: Tool failed"
    return f"{error}\nOutput:\n{output}" if output else error


def _ui_tool_result_text(
    result: ToolResult,
    observation: str | list[dict[str, Any]],
    *,
    max_chars: int = UI_TOOL_RESULT_CHARS,
) -> str:
    """Sanitized tool text for the IDE UI (longer than model preview)."""
    if isinstance(observation, list):
        return "[Multimodal Array Content]"
    text = str(observation or "")
    if not text.strip() and not result.success and result.error:
        text = f"Error: {result.error}"
    if len(text) > max_chars:
        return text[: max(0, max_chars - 20)].rstrip() + "\n…[truncated]"
    return text


def _run_context_workspace(run_context: Any) -> str | None:
    """Workspace root from run_context metadata (never trust bare cwd alone)."""
    meta = getattr(run_context, "_metadata", None) if run_context is not None else None
    if isinstance(meta, dict):
        ws = meta.get("workspace")
        if isinstance(ws, str) and ws.strip():
            return ws.strip()
    return None


# ─── Adaptive Token Estimation (learned from deepagents) ──────────────────
# Now uses tiktoken for accurate BPE counting (with fallback to heuristic).

# Keep _CHARS_PER_TOKEN for the Tier-3 preflight char-budget calculation only
_CHARS_PER_TOKEN = 4


def _estimate_tokens(content: str | list[dict], multiplier: float = 1.0, model: str | None = None) -> int:
    return count_tokens_content(content, model=model, multiplier=multiplier)


def _estimate_messages_tokens(
    messages: list[LLMMessage],
    multiplier: float = 1.0,
    model: str | None = None,
    *,
    cached_system_tokens: int | None = None,
) -> int:
    return _count_messages_tokens(
        messages,
        model=model,
        multiplier=multiplier,
        cached_system_tokens=cached_system_tokens,
    )


# ─── Tool Argument Truncation in Old Messages (learned from deepagents) ───

_MAX_ARG_LENGTH = 2000
_ARG_TRUNCATION_MARKER = "...(argument truncated)"
_RECENT_PROTECTED_COUNT = 20
_TRUNCATABLE_RE = re.compile(
    r'\{"tool":\s*"(write_file|edit_file|create_file)".*?"args":\s*\{'
)


def _truncate_old_tool_args(
    messages: list[LLMMessage], protect_recent: int = _RECENT_PROTECTED_COUNT,
) -> list[LLMMessage]:
    if len(messages) <= protect_recent:
        return messages

    cutoff = len(messages) - protect_recent
    result: list[LLMMessage] = []

    for i, m in enumerate(messages):
        if (
            i < cutoff
            and m.role == "assistant"
            and isinstance(m.content, str)
            and _TRUNCATABLE_RE.search(m.content)
            and len(m.content) > _MAX_ARG_LENGTH
        ):
            result.append(LLMMessage(
                role=m.role,
                content=m.content[:_MAX_ARG_LENGTH] + _ARG_TRUNCATION_MARKER,
            ))
        else:
            result.append(m)

    return result


# ─── Tool Approval ────────────────────────────────────────────────────────

async def _wait_for_tool_approval(
    run_context: RunContext,
    call_id: str,
    tool_name: str,
    args: dict[str, Any],
    *,
    approval_handler: Any,
    emit: Callable,
    timeout_s: float = 300.0,
) -> bool:
    """Block until RunContext has an approval decision, or handler returns one.

    ``approval_handler`` may be:
      - ``"event"``: poll ``is_tool_approved`` until decided (host must call
        ``approve_tool`` / ``reject_tool``)
      - callable ``(tool_name, args, call_id) -> bool | Awaitable[bool]``
    """
    if callable(approval_handler):
        try:
            result = approval_handler(tool_name, args, call_id)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[misc]
            return bool(result)
        except Exception as exc:
            emit("warn", {"message": f"approval_handler error: {exc}"})
            return False

    # "event" (or any other truthy non-callable): poll RunContext
    deadline = asyncio.get_event_loop().time() + max(1.0, timeout_s)
    while asyncio.get_event_loop().time() < deadline:
        state = run_context.is_tool_approved(call_id, tool_name=tool_name)
        if state is not None:
            return bool(state)
        await asyncio.sleep(0.05)
    emit("warn", {"message": f"approval timed out for {tool_name}"})
    return False


# ─── Post-Tool Side Effects ───────────────────────────────────────────────

def _post_tool_side_effects(
    tool_name: str,
    args: dict[str, Any],
    success: bool,
    tool_output: str | list[dict[str, Any]],
    *,
    emit: Callable,
    run_context: RunContext | None = None,
) -> str | list[dict[str, Any]]:
    """Ledger / shadow-checkpoint / auto-verify after a tool completes."""
    from clawagents.config.features import is_enabled

    out = tool_output
    try:
        if success:
            path = None
            if isinstance(args, dict):
                path = args.get("path") or args.get("file_path") or args.get("file")
            if path:
                from clawagents.skills.strategy import note_touched_path

                note_touched_path(run_context, str(path))
                store = None
                if run_context is not None and isinstance(run_context._metadata, dict):
                    store = run_context._metadata.get("skill_store")
                if store is not None and hasattr(store, "note_touched_path"):
                    store.note_touched_path(str(path))
    except Exception:
        logger.debug("skill path touch tracking failed", exc_info=True)

    try:
        if success and is_enabled("context_ledger"):
            from clawagents.memory.context_ledger import maybe_record_from_tool_result

            text = out if isinstance(out, str) else str(out)
            entry = maybe_record_from_tool_result(tool_name, args, text)
            if entry is not None:
                emit("context", {"message": f"context ledger recorded {entry.sha[:12]}"})
                emit("checkpoint", {"kind": "ledger", "sha": entry.sha})
    except Exception:
        logger.debug("ledger record failed", exc_info=True)

    try:
        if success and is_enabled("shadow_checkpoints"):
            from clawagents.permissions.mode import is_write_class_tool
            from clawagents.memory.shadow_checkpoint import create_checkpoint

            # Post-success for execute/git_commit (pre-mutation already covers writes).
            if tool_name in {"execute", "git_commit"} or (
                is_write_class_tool(tool_name)
                and tool_name not in {
                    "write_file", "edit_file", "apply_patch", "create_file",
                    "replace_in_file", "insert_in_file", "insert_lines", "patch_file",
                }
            ):
                info = create_checkpoint(label=tool_name, tool=tool_name, phase="post")
                if info.get("ok") and info.get("sha"):
                    emit(
                        "checkpoint",
                        {
                            "kind": "shadow",
                            "sha": info["sha"],
                            "tool": tool_name,
                            "phase": "post",
                            "label": tool_name,
                            "ts": info.get("ts"),
                        },
                    )
                    if isinstance(out, str):
                        out = out + f"\n[checkpoint {info['sha'][:12]}]"
            elif is_write_class_tool(tool_name) or tool_name in {
                "write_file", "edit_file", "apply_patch",
            }:
                # Pre-mutation checkpoint already created in registry; emit latest HEAD for UI.
                from clawagents.memory.shadow_checkpoint import list_checkpoints

                rows = list_checkpoints(limit=1)
                if rows:
                    row = rows[0]
                    emit(
                        "checkpoint",
                        {
                            "kind": "shadow",
                            "sha": row.get("sha"),
                            "tool": tool_name,
                            "phase": "pre",
                            "label": row.get("label") or tool_name,
                            "ts": row.get("ts"),
                        },
                    )
                    if isinstance(out, str) and row.get("sha"):
                        out = out + f"\n[checkpoint {str(row['sha'])[:12]}]"
    except Exception:
        logger.debug("shadow checkpoint failed", exc_info=True)

    try:
        if is_enabled("auto_verify") and isinstance(out, str):
            from clawagents.tools.auto_verify import maybe_verify_after_edit

            extra = maybe_verify_after_edit(tool_name, success)
            if extra:
                out = out + "\n\n" + extra
    except Exception:
        logger.debug("auto_verify failed", exc_info=True)

    return out
