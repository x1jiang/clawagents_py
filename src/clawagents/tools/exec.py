"""Exec Tool — backed by a pluggable SandboxBackend.

Provides shell command execution with timeout and output capture.
"""

from __future__ import annotations

from typing import Any, Dict, List

from clawagents.tools.registry import Tool, ToolResult

DEFAULT_TIMEOUT_MS = 30000
MAX_OUTPUT_CHARS = 10000

BLOCKED_PATTERNS = ["rm -rf /", "mkfs", "dd if=", "> /dev/sd"]


class ExecTool:
    name = "execute"
    description = (
        "Execute a shell command and return its output. Use for running scripts, "
        "installing packages, checking system state, etc. Commands run in the current working directory."
    )
    parameters = {
        "command": {"type": "string", "description": "The shell command to execute", "required": True},
        "timeout": {"type": "number", "description": f"Timeout in milliseconds. Default: {DEFAULT_TIMEOUT_MS}"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        command = str(args.get("command", ""))
        try:
            timeout_ms = max(100, int(args.get("timeout", DEFAULT_TIMEOUT_MS)))
        except (TypeError, ValueError):
            timeout_ms = DEFAULT_TIMEOUT_MS

        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        for pattern in BLOCKED_PATTERNS:
            if pattern in command:
                return ToolResult(success=False, output="", error=f"Blocked potentially destructive command: {command}")

        try:
            result = await sb.exec(command, timeout=timeout_ms)

            if result.killed:
                return ToolResult(success=False, output="", error=f"Command timed out after {timeout_ms}ms: {command}")

            output = result.stdout or ""
            if result.stderr:
                output += ("\n" if output else "") + f"[stderr] {result.stderr}"

            if len(output) > MAX_OUTPUT_CHARS:
                original_len = len(output)
                half = MAX_OUTPUT_CHARS // 2
                output = output[:half] + f"\n\n... [truncated {original_len - MAX_OUTPUT_CHARS} chars] ...\n\n" + output[-half:]

            success = result.exit_code == 0
            if not success and not output:
                return ToolResult(
                    success=False,
                    output=result.stderr or "",
                    error=f"Command failed with exit code {result.exit_code}: {command}",
                )

            return ToolResult(success=success, output=output or "(no output)")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Command failed: {str(e)}")


# ─── Public API ──────────────────────────────────────────────────────────────

def create_exec_tools(backend: Any) -> List[Tool]:
    """Create exec tools backed by a specific SandboxBackend."""
    return [ExecTool(backend)]


def _default_backend() -> Any:
    from clawagents.sandbox.local import LocalBackend
    return LocalBackend()


exec_tools: List[Tool] = create_exec_tools(_default_backend())
