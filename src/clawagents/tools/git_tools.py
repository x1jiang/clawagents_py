"""First-class git tools (status/diff/commit/undo) — safer than free-form shell."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from clawagents.tools.registry import Tool, ToolResult


def _git(args: list[str], cwd: str, *, timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


class GitStatusTool:
    name = "git_status"
    description = "Show git status (short) for the workspace."
    parameters: dict[str, Any] = {}

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        code, out, err = _git(["status", "-sb"], self._workspace)
        if code != 0:
            return ToolResult(success=False, output="", error=err or out)
        return ToolResult(success=True, output=out or "(clean)")


class GitDiffTool:
    name = "git_diff"
    description = "Show git diff (unstaged + staged summary)."
    parameters = {
        "staged": {"type": "boolean", "description": "If true, show --staged only"},
        "path": {"type": "string", "description": "Optional path limiter"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        cmd = ["diff"]
        if args.get("staged"):
            cmd.append("--staged")
        path = str(args.get("path") or "").strip()
        if path:
            cmd.extend(["--", path])
        code, out, err = _git(cmd, self._workspace)
        if code != 0:
            return ToolResult(success=False, output="", error=err or out)
        text = out or "(no diff)"
        if len(text) > 60_000:
            text = text[:60_000] + "\n… [diff truncated] …"
        return ToolResult(success=True, output=text)


class GitCommitTool:
    name = "git_commit"
    description = (
        "Stage listed paths (or all tracked changes with all=true) and create a commit. "
        "Records a context-ledger entry on success."
    )
    parameters = {
        "message": {"type": "string", "description": "Commit message", "required": True},
        "paths": {
            "type": "array",
            "description": "Paths to stage (relative). Ignored if all=true.",
            "items": {"type": "string"},
        },
        "all": {"type": "boolean", "description": "Stage all modified tracked files (-a)"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        msg = str(args.get("message") or "").strip()
        if not msg:
            return ToolResult(success=False, output="", error="message is required")
        if args.get("all"):
            code, out, err = _git(["add", "-u"], self._workspace)
            if code != 0:
                return ToolResult(success=False, output="", error=err or out)
        else:
            paths = args.get("paths") or []
            if not isinstance(paths, list) or not paths:
                return ToolResult(
                    success=False, output="", error="provide paths=[] or all=true"
                )
            for p in paths:
                rel = str(p)
                # refuse path escape
                full = Path(self._workspace) / rel
                try:
                    full.resolve().relative_to(Path(self._workspace).resolve())
                except ValueError:
                    return ToolResult(success=False, output="", error=f"path escapes workspace: {rel}")
                code, out, err = _git(["add", "--", rel], self._workspace)
                if code != 0:
                    return ToolResult(success=False, output="", error=err or out)
        code, out, err = _git(["commit", "-m", msg], self._workspace)
        if code != 0:
            return ToolResult(success=False, output="", error=err or out or "commit failed")
        try:
            from clawagents.memory.context_ledger import record_commit_ledger

            record_commit_ledger(workspace=self._workspace)
        except Exception:
            pass
        sha_code, sha, _ = _git(["rev-parse", "--short", "HEAD"], self._workspace)
        return ToolResult(
            success=True,
            output=f"Committed {(sha if sha_code == 0 else '')}: {msg}\n{out}".strip(),
        )


class GitUndoAiTool:
    name = "git_undo_ai"
    description = (
        "Undo the last commit if it was created in this workspace session style "
        "(soft reset by default). Refuses if the commit is not HEAD or already pushed "
        "when require_unpushed=true."
    )
    parameters = {
        "hard": {"type": "boolean", "description": "Use reset --hard (destructive)"},
        "require_unpushed": {
            "type": "boolean",
            "description": "Refuse if HEAD is on remote (default true)",
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        require_unpushed = args.get("require_unpushed", True)
        if require_unpushed:
            code, out, err = _git(["status", "-sb"], self._workspace)
            if code == 0 and "ahead" not in out and "..." in out:
                # on tracking branch and not ahead → likely pushed
                if "ahead" not in out:
                    return ToolResult(
                        success=False,
                        output="",
                        error="HEAD does not appear unpushed; refusing undo",
                    )
        mode = "--hard" if args.get("hard") else "--soft"
        code, out, err = _git(["reset", mode, "HEAD~1"], self._workspace)
        if code != 0:
            return ToolResult(success=False, output="", error=err or out)
        return ToolResult(success=True, output=f"Reset {mode} HEAD~1\n{out}".strip())


def create_git_tools(workspace: str | None = None) -> list[Tool]:
    ws = workspace or os.getcwd()
    return [
        GitStatusTool(ws),
        GitDiffTool(ws),
        GitCommitTool(ws),
        GitUndoAiTool(ws),
    ]
