"""Exec Tool — backed by a pluggable SandboxBackend.

Provides shell command execution with timeout and output capture.

The pre-execute pipeline is:

1. Obfuscation detector (``detect_obfuscation``) — refuses on hit.
2. Bash semantic validator (``validate_bash``) — BLOCK refuses, WARN
   prepends a notice (and refuses in PLAN mode for DESTRUCTIVE).
3. Legacy ``_is_dangerous_command`` denylist (kept for back-compat).
4. Optional RTK wrap / shell-session cwd wrap.
5. Sandbox exec — or local subprocess with Grok-style auto-background
   on foreground timeout.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from clawagents.permissions.mode import PermissionMode
from clawagents.tools.bash_validator import (
    BashDecision,
    CommandCategory,
    Decision,
    validate_bash,
)
from clawagents.tools.exec_obfuscation import detect_obfuscation
from clawagents.tools.registry import Tool, ToolResult
from clawagents.tracing import tool_span

DEFAULT_TIMEOUT_MS = 30000
MAX_OUTPUT_CHARS = 10000

BLOCKED_PATTERNS: list[str] = [
    ":(){ :|:& };:",
]

_DANGEROUS_RE = re.compile(
    r"(?:sudo\s+)?rm\s+(?:-\w*[rf]\w*\s+)*/\s*$"
    r"|>\s*['\"]?/dev/sd"
    r"|mkfs\."
    r"|dd\s+if="
    r"|:\(\)\s*\{",
    re.IGNORECASE,
)


def _is_dangerous_command(command: str) -> bool:
    if _DANGEROUS_RE.search(command):
        return True
    for pattern in BLOCKED_PATTERNS:
        if pattern in command:
            return True
    return False


def _truncate_exec_output(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    original_len = len(output)
    half = MAX_OUTPUT_CHARS // 2
    return (
        output[:half]
        + f"\n\n... [truncated {original_len - MAX_OUTPUT_CHARS} chars] ...\n\n"
        + output[-half:]
    )


def _format_nonzero_command_output(
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    warning_prefix: str,
) -> str:
    payload: dict[str, Any] = {
        "command_executed": True,
        "success": False,
        "exit_code": exit_code,
        "command": command,
        "stdout": _truncate_exec_output(stdout or ""),
        "stderr": _truncate_exec_output(stderr or ""),
        "interpretation": (
            "The command ran and exited nonzero. Treat stdout/stderr as "
            "diagnostic feedback, not as a tool transport failure."
        ),
    }
    warning = warning_prefix.strip()
    if warning:
        payload["warning"] = warning
    return json.dumps(payload, indent=2)


def _preflight_command(
    command: str,
    *,
    permission_mode: PermissionMode = PermissionMode.DEFAULT,
) -> tuple[str | None, str]:
    """Run exec safety pipeline. Returns ``(error|None, warning_prefix)``."""
    ob = detect_obfuscation(command)
    if ob is not None:
        return (
            "Refused: obfuscated/encoded command detected "
            f"({', '.join(ob.matched_patterns)}): {'; '.join(ob.reasons)}"
        ), ""

    decision: BashDecision = validate_bash(command)
    if decision.decision == Decision.BLOCK:
        return (
            f"Blocked by bash validator ({decision.category.value}): {decision.reason}"
        ), ""
    if (
        permission_mode == PermissionMode.PLAN
        and decision.category == CommandCategory.DESTRUCTIVE
    ):
        return (
            "Blocked: destructive command refused in plan mode "
            f"({decision.reason})"
        ), ""

    warning_prefix = ""
    if decision.decision == Decision.WARN:
        warning_prefix = (
            f"[bash_validator: WARN {decision.category.value} — "
            f"{decision.reason}]\n"
        )

    if _is_dangerous_command(command):
        return f"Blocked potentially destructive command: {command}", ""

    return None, warning_prefix


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _bg_manager(run_context: Any):
    from clawagents.background import BackgroundJobManager
    from clawagents.tools.background_task import create_background_task_tools

    mgr = getattr(run_context, "background_manager", None) if run_context else None
    if mgr is None:
        tools = create_background_task_tools()
        mgr = getattr(tools[0], "_manager", None)
    if mgr is None:
        mgr = BackgroundJobManager()
    if run_context is not None:
        try:
            setattr(run_context, "background_manager", mgr)
        except Exception:
            pass
    return mgr


def _shell_argv(command: str) -> list[str]:
    if sys.platform == "win32":
        return ["cmd.exe", "/c", command]
    return ["/bin/sh", "-c", command]


async def _exec_foreground_with_autobg(
    command: str,
    *,
    cwd: str,
    timeout_ms: int,
    mgr: Any,
) -> tuple[str, str, int, bool, Optional[str]]:
    """Run shell; on timeout adopt the process into ``mgr``.

    Returns ``(stdout, stderr, exit_code, timed_out_backgrounded, job_id|None)``.
    """
    import signal

    env = {**os.environ, "PAGER": "cat"}
    # Drop obvious secrets from child env (same spirit as LocalBackend).
    try:
        from clawagents.redact import is_secret_name

        env = {k: v for k, v in env.items() if not is_secret_name(k)}
    except Exception:
        pass

    timeout_s = max(0.1, timeout_ms / 1000.0)
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    comm = asyncio.create_task(proc.communicate())
    try:
        out_b, err_b = await asyncio.wait_for(asyncio.shield(comm), timeout=timeout_s)
        return (
            (out_b or b"").decode("utf-8", errors="replace"),
            (err_b or b"").decode("utf-8", errors="replace"),
            proc.returncode or 0,
            False,
            None,
        )
    except asyncio.TimeoutError:
        argv = _shell_argv(command)
        job = await mgr.adopt(proc, argv, cwd=cwd, communicate_task=comm)
        return (
            "",
            "",
            0,
            True,
            job.id,
        )
    except Exception:
        # Ensure we don't leak a running process on unexpected errors.
        if not comm.done():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await comm
            except Exception:
                pass
        raise


class ExecTool:
    name = "execute"
    keywords = ["shell", "bash", "command", "run script", "terminal"]
    description = (
        "Execute a shell command and return its output. Working directory "
        "persists across calls in this session (cd sticks). Noisy commands "
        "(pytest, git status/log/diff, ls, rg, …) may be auto-wrapped with rtk "
        "when installed. Set is_background=true for long-running commands; "
        "foreground timeouts may auto-background and return a job_id — use "
        "task_status / task_output / task_stop."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "command": {"type": "string", "description": "The shell command to execute", "required": True},
        "timeout": {"type": "number", "description": f"Timeout in milliseconds. Default: {DEFAULT_TIMEOUT_MS}"},
        "description": {
            "type": "string",
            "description": "One-sentence explanation of why this command is needed (recommended).",
        },
        "is_background": {
            "type": "boolean",
            "description": (
                "Run in the background and return job_id immediately "
                "(for long-running commands). Default: false."
            ),
        },
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any], run_context: Any = None) -> ToolResult:
        from clawagents.config.features import is_enabled
        from clawagents.tools.shell_session import session_for

        sb = self._sb
        command = str(args.get("command", ""))
        try:
            timeout_ms = max(100, int(args.get("timeout", DEFAULT_TIMEOUT_MS)))
        except (TypeError, ValueError):
            timeout_ms = DEFAULT_TIMEOUT_MS

        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        permission_mode = getattr(run_context, "permission_mode", PermissionMode.DEFAULT)
        is_background = _truthy(args.get("is_background"))
        desc = str(args.get("description") or "").strip()

        with tool_span("exec.validate", command=command):
            err, warning_prefix = _preflight_command(
                command, permission_mode=permission_mode
            )
            if err is not None:
                return ToolResult(success=False, output="", error=err)

        # Loop-side RTK wrap (token-efficient shell) — not hooks.
        try:
            from clawagents.tools.rtk_wrap import maybe_wrap_with_rtk

            wrapped, wrap_reason = maybe_wrap_with_rtk(command)
            if wrap_reason and wrapped != command:
                warning_prefix += f"[rtk_wrap: {wrap_reason}]\n"
                command = wrapped
        except Exception:
            pass

        session = None
        run_cwd = getattr(sb, "cwd", None) or os.getcwd()
        # Session + auto-bg adopt need a real local shell. Other backends
        # (in-memory / docker / test doubles) keep the classic sb.exec path.
        is_local_sb = getattr(sb, "kind", None) == "local"
        if is_enabled("execute_shell_session") and is_local_sb:
            session = session_for(run_context, sb)
            run_cwd = session.cwd
            command = session.wrap(command)
            warning_prefix += f"[shell_session: cwd={run_cwd}]\n"

        if is_background:
            if not is_enabled("execute_background"):
                return ToolResult(
                    success=False,
                    output="",
                    error="is_background requires CLAW_FEATURE_EXECUTE_BACKGROUND=1",
                )

            mgr = _bg_manager(run_context)
            with tool_span("exec.background", command=command):
                try:
                    job = await mgr.start(_shell_argv(command), cwd=run_cwd)
                except Exception as e:
                    return ToolResult(
                        success=False, output="", error=f"Background start failed: {e}"
                    )

            payload = {
                "backgrounded": True,
                "job_id": job.id,
                "pid": job.pid,
                "command": str(args.get("command", "")),
                "cwd": run_cwd,
                "description": desc or None,
                "hint": "Use task_status / task_output / task_stop with this job_id.",
            }
            return ToolResult(
                success=True,
                output=warning_prefix + json.dumps(payload, indent=2),
            )

        # Grok-style auto-background on FG timeout (adopt live process).
        use_autobg = (
            is_local_sb
            and is_enabled("execute_auto_background")
            and is_enabled("execute_background")
        )
        if use_autobg:
            mgr = _bg_manager(run_context)
            with tool_span("exec.run", command=command, timeout_ms=timeout_ms):
                try:
                    stdout, stderr, exit_code, bgd, job_id = await _exec_foreground_with_autobg(
                        command,
                        cwd=run_cwd,
                        timeout_ms=timeout_ms,
                        mgr=mgr,
                    )
                except Exception as e:
                    return ToolResult(
                        success=False, output="", error=f"Command failed: {str(e)}"
                    )

            if bgd and job_id:
                payload = {
                    "backgrounded": True,
                    "auto_background_on_timeout": True,
                    "job_id": job_id,
                    "timeout_ms": timeout_ms,
                    "command": str(args.get("command", "")),
                    "cwd": run_cwd,
                    "description": desc or None,
                    "hint": (
                        "Foreground wait timed out; process kept running in the "
                        "background. Use task_status / task_output / task_stop."
                    ),
                }
                return ToolResult(
                    success=True,
                    output=warning_prefix + json.dumps(payload, indent=2),
                )

            if session is not None:
                stdout = session.consume_stdout(stdout)

            success = exit_code == 0
            if not success:
                return ToolResult(
                    success=False,
                    output=_format_nonzero_command_output(
                        str(args.get("command", command)),
                        exit_code,
                        stdout or "",
                        stderr or "",
                        warning_prefix,
                    ),
                    error=(
                        f"Command exited with code {exit_code}: "
                        f"{args.get('command', command)}"
                    ),
                )

            output = stdout or ""
            if stderr:
                output += ("\n" if output else "") + f"[stderr] {stderr}"
            output = _truncate_exec_output(output)
            if session is not None:
                warning_prefix += f"[shell_session: cwd now {session.cwd}]\n"
            return ToolResult(
                success=True, output=warning_prefix + (output or "(no output)")
            )

        # Legacy sandbox path (kill-on-timeout).
        with tool_span("exec.run", command=command, timeout_ms=timeout_ms):
            try:
                result = await sb.exec(command, timeout=timeout_ms, cwd=run_cwd)
            except TypeError:
                # Backends that don't accept cwd=
                try:
                    result = await sb.exec(command, timeout=timeout_ms)
                except Exception as e:
                    return ToolResult(
                        success=False, output="", error=f"Command failed: {str(e)}"
                    )
            except Exception as e:
                return ToolResult(
                    success=False, output="", error=f"Command failed: {str(e)}"
                )

            if result.killed:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"Command timed out after {timeout_ms}ms: "
                        f"{args.get('command', command)}"
                    ),
                )

            stdout = result.stdout or ""
            if session is not None:
                stdout = session.consume_stdout(stdout)

            success = result.exit_code == 0
            if not success:
                return ToolResult(
                    success=False,
                    output=_format_nonzero_command_output(
                        str(args.get("command", command)),
                        result.exit_code,
                        stdout,
                        result.stderr or "",
                        warning_prefix,
                    ),
                    error=(
                        f"Command exited with code {result.exit_code}: "
                        f"{args.get('command', command)}"
                    ),
                )

            output = stdout
            if result.stderr:
                output += ("\n" if output else "") + f"[stderr] {result.stderr}"
            output = _truncate_exec_output(output)
            if session is not None:
                warning_prefix += f"[shell_session: cwd now {session.cwd}]\n"
            return ToolResult(
                success=True, output=warning_prefix + (output or "(no output)")
            )


# ─── Public API ──────────────────────────────────────────────────────────────

def create_exec_tools(backend: Any) -> List[Tool]:
    """Create exec tools backed by a specific SandboxBackend."""
    return [ExecTool(backend)]


def _default_backend() -> Any:
    from clawagents.sandbox.local import LocalBackend
    return LocalBackend()


class _LazyExecTools(list):
    """Lazy list that populates itself on first access."""
    _initialized = False

    def _ensure(self):
        if not self._initialized:
            self._initialized = True
            self.extend(create_exec_tools(_default_backend()))

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def __len__(self):
        self._ensure()
        return super().__len__()

    def __getitem__(self, idx):
        self._ensure()
        return super().__getitem__(idx)

    def __contains__(self, item):
        self._ensure()
        return super().__contains__(item)


exec_tools: List[Tool] = _LazyExecTools()
