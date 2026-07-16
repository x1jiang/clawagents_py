"""External hook system for ClawAgents.

Hooks are shell commands configured in .clawagents/hooks.json or via
environment variables. They run before/after tool execution and LLM calls,
receiving JSON on stdin and returning JSON on stdout.

Security note — hooks are a TRUSTED-ONLY surface
------------------------------------------------
Anyone who can write to ``.clawagents/hooks.json`` or set ``CLAW_HOOK_*``
environment variables effectively has shell-level code execution inside the
agent process: the strings configured there are passed straight to
``asyncio.create_subprocess_exec`` and run with the agent's full privileges.

Hooks are off by default and only loaded when ``CLAW_FEATURE_EXTERNAL_HOOKS``
is set to ``1``/``true``/``yes``. Treat hook config the same way you treat
``.bashrc`` or a CI deploy key:

* never load hooks from untrusted ``cwd``s,
* never let untrusted users edit ``.clawagents/hooks.json``,
* don't echo or commit hook commands that contain secrets,
* prefer absolute paths to your hook scripts and pin them in version control.

Inspired by claw-code-main's hook system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HOOK_TIMEOUT_S = 10.0


@dataclass
class HooksConfig:
    """Configuration for external hooks."""
    pre_tool_use: str | None = None
    post_tool_use: str | None = None
    pre_llm: str | None = None
    post_llm: str | None = None

    @property
    def has_any(self) -> bool:
        return any([self.pre_tool_use, self.post_tool_use, self.pre_llm, self.post_llm])


def load_hooks_config() -> HooksConfig | None:
    """Load hooks config from .clawagents/hooks.json or env vars.

    Returns None if no hooks are configured.
    """
    config = HooksConfig()

    # 1. Try .clawagents/hooks.json
    hooks_file = Path.cwd() / ".clawagents" / "hooks.json"
    if hooks_file.exists():
        try:
            data = json.loads(hooks_file.read_text())
            config.pre_tool_use = data.get("pre_tool_use")
            config.post_tool_use = data.get("post_tool_use")
            config.pre_llm = data.get("pre_llm")
            config.post_llm = data.get("post_llm")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load hooks.json: %s", exc)

    # 2. Env var overrides
    for attr, env_key in [
        ("pre_tool_use", "CLAW_HOOK_PRE_TOOL_USE"),
        ("post_tool_use", "CLAW_HOOK_POST_TOOL_USE"),
        ("pre_llm", "CLAW_HOOK_PRE_LLM"),
        ("post_llm", "CLAW_HOOK_POST_LLM"),
    ]:
        val = os.environ.get(env_key)
        if val:
            setattr(config, attr, val)

    return config if config.has_any else None


async def run_hook(
    command: str,
    input_data: dict[str, Any],
    timeout: float = _HOOK_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Execute a hook command, passing input_data as JSON on stdin.

    Returns parsed JSON from stdout, or None on failure (fail-open).
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        input_bytes = json.dumps(input_data).encode()
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_bytes),
            timeout=timeout,
        )
        if proc.returncode != 0:
            logger.warning(
                "Hook %r exited %d: %s", command, proc.returncode,
                stderr.decode(errors="replace")[:200],
            )
            return None
        if not stdout.strip():
            return None
        return json.loads(stdout)
    except asyncio.TimeoutError:
        logger.warning("Hook %r timed out after %.1fs — proceeding without", command, timeout)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Hook %r failed: %s — proceeding without", command, exc)
        return None


def build_taxonomy_dispatcher(
    legacy: HooksConfig | None = None,
) -> Any | None:
    """Load expanded hook taxonomy dispatcher when explicitly opted in.

    Requires BOTH ``hook_taxonomy`` and ``external_hooks`` so a cloned
    ``.clawagents/hooks.json`` cannot execute ``bash -lc`` on SessionStart
    when the user never enabled external hooks.
    """
    from clawagents.config.features import is_enabled

    if not is_enabled("hook_taxonomy"):
        return None
    if not is_enabled("external_hooks"):
        return None

    from clawagents.hooks.taxonomy import (
        HookDispatcher,
        HookEvent,
        HookHandler,
        load_handlers_from_config,
        normalize_event,
    )

    handlers: list[HookHandler] = []
    hooks_file = Path.cwd() / ".clawagents" / "hooks.json"
    if hooks_file.exists():
        try:
            data = json.loads(hooks_file.read_text())
            handlers.extend(load_handlers_from_config(data))
            for key, ev_name in (
                ("pre_tool_use", "PreToolUse"),
                ("post_tool_use", "PostToolUse"),
                ("pre_llm", "UserPromptSubmit"),
                ("post_llm", "Notification"),
            ):
                cmd = data.get(key)
                ev = normalize_event(ev_name)
                if cmd and isinstance(cmd, str) and ev is not None:
                    if not any(h.event == ev and h.command for h in handlers):
                        handlers.append(
                            HookHandler(event=ev, command=["bash", "-lc", cmd])
                        )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load taxonomy hooks.json: %s", exc)

    if legacy is not None:
        for attr, ev_name in (
            ("pre_tool_use", "PreToolUse"),
            ("post_tool_use", "PostToolUse"),
            ("pre_llm", "UserPromptSubmit"),
            ("post_llm", "Notification"),
        ):
            cmd = getattr(legacy, attr, None)
            ev = normalize_event(ev_name)
            if cmd and ev is not None:
                if not any(h.event == ev and h.command for h in handlers):
                    handlers.append(
                        HookHandler(event=ev, command=["bash", "-lc", cmd])
                    )

    return HookDispatcher(handlers=handlers)


async def dispatch_taxonomy_hook(
    dispatcher: Any,
    event: Any,
    payload: dict[str, Any] | None = None,
    *,
    blocking: bool | None = None,
) -> tuple[bool, str]:
    """Run taxonomy hook dispatch off the event loop (subprocess/webhook I/O)."""
    if dispatcher is None:
        return True, ""
    decision = await asyncio.to_thread(
        dispatcher.dispatch,
        event,
        payload or {},
        blocking=blocking,
    )
    return bool(decision.allowed), str(decision.reason or "")


class ExternalHookRunner:
    """Runs external hooks at key points in the agent loop."""

    def __init__(self, config: HooksConfig):
        self.config = config

    async def pre_tool_use(
        self, tool_name: str, args: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Run pre_tool_use hook.

        Returns (allowed, possibly_modified_args).
        """
        if not self.config.pre_tool_use:
            return True, args

        result = await run_hook(self.config.pre_tool_use, {
            "event": "pre_tool_use",
            "tool": tool_name,
            "args": args,
        })
        if result is None:
            return True, args  # fail-open

        allowed = result.get("allowed", True)
        updated_args = result.get("updated_input", args)
        return bool(allowed), updated_args

    async def post_tool_use(
        self, tool_name: str, args: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run post_tool_use hook. Returns possibly modified result."""
        if not self.config.post_tool_use:
            return result

        hook_result = await run_hook(self.config.post_tool_use, {
            "event": "post_tool_use",
            "tool": tool_name,
            "args": args,
            "result": result,
        })
        if hook_result is None:
            return result  # fail-open

        return hook_result.get("updated_result", result)

    async def pre_llm(
        self, messages_summary: list[dict[str, str]],
    ) -> list[dict[str, str]] | None:
        """Run pre_llm hook. Returns additional messages to inject, or None."""
        if not self.config.pre_llm:
            return None

        result = await run_hook(self.config.pre_llm, {
            "event": "pre_llm",
            "message_count": len(messages_summary),
            "last_role": messages_summary[-1]["role"] if messages_summary else "",
        })
        if result is None:
            return None

        return result.get("messages")

    async def post_llm(self, response_preview: str, tool_calls_count: int) -> None:
        """Run post_llm hook (fire-and-forget logging)."""
        if not self.config.post_llm:
            return

        await run_hook(self.config.post_llm, {
            "event": "post_llm",
            "response_preview": response_preview[:500],
            "tool_calls_count": tool_calls_count,
        })
