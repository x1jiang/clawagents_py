"""Configurable built-in base system prompt.

The base prompt is the text ClawAgents sends when the caller gives no
``instruction`` / ``system_prompt``. It can be overridden without editing the
package; resolution order (highest precedence first):

1. ``override`` argument (``base_prompt=`` on :func:`create_claw_agent`):
   inline text, or a path to a file
2. ``CLAW_BASE_PROMPT_FILE`` env var — path to a file
3. ``CLAW_BASE_PROMPT`` env var — inline text
4. ``<workspace>/.clawagents/base-prompt.md``
5. ``~/.clawagents/base-prompt.md``
6. :data:`DEFAULT_BASE_SYSTEM_PROMPT`

An *append* block is added after whichever base won, for callers who only
want extra rules. It resolves with the same shape (highest precedence first):

1. ``append`` argument (``base_prompt_append=``): inline text, or a file path
2. ``CLAW_BASE_PROMPT_APPEND_FILE`` env var — path to a file
3. ``CLAW_BASE_PROMPT_APPEND`` env var — inline text
4. ``<workspace>/.clawagents/base-prompt-append.md``
5. ``~/.clawagents/base-prompt-append.md``

A caller-supplied ``instruction`` *replaces* the base prompt entirely, but the
append block is still added after it; that composition lives in
:func:`clawagents.agent.create_claw_agent`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "DEFAULT_BASE_SYSTEM_PROMPT",
    "BASE_PROMPT_ENV",
    "BASE_PROMPT_FILE_ENV",
    "BASE_PROMPT_APPEND_ENV",
    "BASE_PROMPT_APPEND_FILE_ENV",
    "BASE_PROMPT_FILENAME",
    "BASE_PROMPT_APPEND_FILENAME",
    "resolve_base_system_prompt",
    "resolve_base_prompt_append",
]

BASE_PROMPT_ENV = "CLAW_BASE_PROMPT"
BASE_PROMPT_FILE_ENV = "CLAW_BASE_PROMPT_FILE"
BASE_PROMPT_APPEND_ENV = "CLAW_BASE_PROMPT_APPEND"
BASE_PROMPT_APPEND_FILE_ENV = "CLAW_BASE_PROMPT_APPEND_FILE"
BASE_PROMPT_FILENAME = "base-prompt.md"
BASE_PROMPT_APPEND_FILENAME = "base-prompt-append.md"

DEFAULT_BASE_SYSTEM_PROMPT = """You are a ClawAgent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls.

## Core Behavior
- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- If the request is ambiguous, ask questions before acting.

## Doing Tasks
When the user asks you to do something:
1. Think briefly about your approach, then act immediately using tools.
2. After getting tool results, continue using more tools or provide the final answer.
3. When done, provide the final answer directly. Do NOT ask if the user wants more.

Keep working until the task is fully complete.

## Efficiency Rules
- NEVER re-read a file you already have in context. Use the data from previous tool results.
- NEVER call the same tool with the same arguments twice. If you already have the result, use it.
- Batch independent tool calls into a single response when possible (use the array syntax).
- Prefer fewer, well-targeted tool calls over many exploratory ones.
- Use todo/planning tools only for broad or long-running tasks. Skip todo bookkeeping for bounded lookup, read, compare, or JSON-report tasks.
- Once tool results contain enough evidence to answer, stop calling tools and answer directly. Do not call tools only to mark progress complete."""


def _read_prompt_file(path: Union[str, os.PathLike]) -> Optional[str]:
    """Return the file's text (stripped) or ``None`` when unreadable."""
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _coerce_override(override: Union[str, os.PathLike]) -> str:
    """Inline text, or the contents of a file the string/PathLike points at."""
    if not isinstance(override, str):
        text = _read_prompt_file(override)
        if text is None:
            raise FileNotFoundError(f"base_prompt file not found: {override!s}")
        return text
    # A single-line string that names an existing file is treated as a path;
    # anything else is inline prompt text.
    if override and "\n" not in override:
        text = _read_prompt_file(override)
        if text is not None:
            return text
    return override


def _resolve_layer(
    override: Union[str, os.PathLike, None],
    *,
    file_env: str,
    inline_env: str,
    filename: str,
    workspace: Union[str, os.PathLike, None],
) -> Optional[str]:
    """Shared precedence walk: override > file env > inline env > files."""
    if override is not None:
        return _coerce_override(override)

    env_file = (os.environ.get(file_env) or "").strip()
    if env_file:
        text = _read_prompt_file(env_file)
        if text is not None:
            return text

    env_inline = os.environ.get(inline_env)
    if env_inline is not None and env_inline.strip():
        return env_inline.strip()

    root = Path(workspace) if workspace is not None else Path.cwd()
    for candidate in (
        root / ".clawagents" / filename,
        Path.home() / ".clawagents" / filename,
    ):
        text = _read_prompt_file(candidate)
        if text is not None:
            return text
    return None


def resolve_base_prompt_append(
    append: Union[str, os.PathLike, None] = None,
    *,
    workspace: Union[str, os.PathLike, None] = None,
) -> str:
    """Resolve the extra text appended after the system prompt (``""`` if none)."""
    text = _resolve_layer(
        append,
        file_env=BASE_PROMPT_APPEND_FILE_ENV,
        inline_env=BASE_PROMPT_APPEND_ENV,
        filename=BASE_PROMPT_APPEND_FILENAME,
        workspace=workspace,
    )
    return (text or "").strip()


def apply_append(prompt: str, extra: str) -> str:
    """Join ``extra`` after ``prompt`` with a blank line (either may be empty)."""
    extra = (extra or "").strip()
    if not extra:
        return prompt
    return f"{prompt.rstrip()}\n\n{extra}" if prompt else extra


def resolve_base_system_prompt(
    override: Union[str, os.PathLike, None] = None,
    *,
    append: Union[str, os.PathLike, None] = None,
    workspace: Union[str, os.PathLike, None] = None,
) -> str:
    """Resolve the built-in base prompt (plus append block) per module precedence."""
    base = _resolve_layer(
        override,
        file_env=BASE_PROMPT_FILE_ENV,
        inline_env=BASE_PROMPT_ENV,
        filename=BASE_PROMPT_FILENAME,
        workspace=workspace,
    )
    if base is None:
        base = DEFAULT_BASE_SYSTEM_PROMPT
    return apply_append(base, resolve_base_prompt_append(append, workspace=workspace))
