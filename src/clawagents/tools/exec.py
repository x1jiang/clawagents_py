"""Exec Tool — backed by a pluggable SandboxBackend.

Provides shell command execution with timeout and output capture.

The pre-execute pipeline is:

1. Obfuscation detector (``detect_obfuscation``) — refuses on hit.
2. Bash semantic validator (``validate_bash``) — BLOCK refuses, WARN
   prepends a notice (and refuses in PLAN mode for DESTRUCTIVE).
3. Legacy ``_is_dangerous_command`` denylist (kept for back-compat).
4. Sandbox exec.

Each phase runs inside its own ``tool_span`` so traces show where time
went.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

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

# Legacy substring backstop — must never widen policy beyond the bash
# validator. Substring match, so anything added here that overlaps with
# a valid command (e.g. ``"curl http"`` matching ``https://``) breaks
# real workloads.
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


def _ensure_brv_command(command: str) -> str:
    """Run ByteRover CLI via npx so it works without a global install."""
    s = command.strip()
    if s == "brv":
        return "npx byterover-cli"
    if s.startswith("brv "):
        return "npx byterover-cli " + s[4:].strip()
    return command


def _truncate_exec_output(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    original_len = len(output)
    half = MAX_OUTPUT_CHARS // 2
    return output[:half] + f"\n\n... [truncated {original_len - MAX_OUTPUT_CHARS} chars] ...\n\n" + output[-half:]


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


class ExecTool:
    name = "execute"
    keywords = ["shell", "bash", "command", "run script", "terminal"]
    description = (
        "Execute a shell command and return its output. Use for running scripts, "
        "installing packages, checking system state, etc. Commands run in the current working directory."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "command": {"type": "string", "description": "The shell command to execute", "required": True},
        "timeout": {"type": "number", "description": f"Timeout in milliseconds. Default: {DEFAULT_TIMEOUT_MS}"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any], run_context: Any = None) -> ToolResult:
        sb = self._sb
        command = str(args.get("command", ""))
        try:
            timeout_ms = max(100, int(args.get("timeout", DEFAULT_TIMEOUT_MS)))
        except (TypeError, ValueError):
            timeout_ms = DEFAULT_TIMEOUT_MS

        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        # Ensure ByteRover CLI is available: run via npx if command is brv and not on PATH
        command = _ensure_brv_command(command)

        warning_prefix = ""
        permission_mode = getattr(run_context, "permission_mode", PermissionMode.DEFAULT)

        with tool_span("exec.validate", command=command):
            # 1. Obfuscation detector
            ob = detect_obfuscation(command)
            if ob is not None:
                return ToolResult(
                    success=False, output="",
                    error=(
                        "Refused: obfuscated/encoded command detected "
                        f"({', '.join(ob.matched_patterns)}): {'; '.join(ob.reasons)}"
                    ),
                )

            # 2. Bash semantic validator
            decision: BashDecision = validate_bash(command)
            if decision.decision == Decision.BLOCK:
                return ToolResult(
                    success=False, output="",
                    error=(
                        f"Blocked by bash validator ({decision.category.value}): "
                        f"{decision.reason}"
                    ),
                )
            if (
                permission_mode == PermissionMode.PLAN
                and decision.category == CommandCategory.DESTRUCTIVE
            ):
                return ToolResult(
                    success=False, output="",
                    error=(
                        "Blocked: destructive command refused in plan mode "
                        f"({decision.reason})"
                    ),
                )
            if decision.decision == Decision.WARN:
                warning_prefix = (
                    f"[bash_validator: WARN {decision.category.value} — "
                    f"{decision.reason}]\n"
                )

            # 3. Legacy denylist (back-compat)
            if _is_dangerous_command(command):
                return ToolResult(
                    success=False, output="",
                    error=f"Blocked potentially destructive command: {command}",
                )

        with tool_span("exec.run", command=command, timeout_ms=timeout_ms):
            try:
                result = await sb.exec(command, timeout=timeout_ms)
            except Exception as e:
                return ToolResult(success=False, output="", error=f"Command failed: {str(e)}")

            if result.killed:
                return ToolResult(
                    success=False, output="",
                    error=f"Command timed out after {timeout_ms}ms: {command}",
                )

            success = result.exit_code == 0
            if not success:
                return ToolResult(
                    success=False,
                    output=_format_nonzero_command_output(
                        command,
                        result.exit_code,
                        result.stdout or "",
                        result.stderr or "",
                        warning_prefix,
                    ),
                    error=f"Command exited with code {result.exit_code}: {command}",
                )

            output = result.stdout or ""
            if result.stderr:
                output += ("\n" if output else "") + f"[stderr] {result.stderr}"
            output = _truncate_exec_output(output)

            return ToolResult(success=success, output=warning_prefix + (output or "(no output)"))


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
