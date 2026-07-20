"""Auto lint/test after successful file edits (Aider-inspired)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


_EDIT_TOOLS = frozenset({
    "write_file", "edit_file", "apply_patch", "insert_lines",
})


def _python_exe() -> str:
    """Prefer python3 / current interpreter — bare ``python`` is often missing."""
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found
    return sys.executable or "python3"


def detect_verify_commands(workspace: str | Path | None = None) -> list[list[str]]:
    root = Path(workspace or Path.cwd())
    cmds: list[list[str]] = []
    pkg = root / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            scripts = data.get("scripts") or {}
            if "lint" in scripts:
                cmds.append(["npm", "run", "lint", "--if-present"])
            if "test" in scripts:
                cmds.append(["npm", "test", "--if-present"])
        except (OSError, json.JSONDecodeError):
            pass
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        py = _python_exe()
        if "[tool.ruff]" in text or "ruff" in text:
            cmds.append([py, "-m", "ruff", "check", "."])
        if "[tool.pytest" in text or "pytest" in text:
            cmds.append([py, "-m", "pytest", "-q", "--tb=no"])
    return cmds[:3]


def run_verify(
    workspace: str | Path | None = None,
    *,
    timeout: int = 90,
) -> str:
    root = Path(workspace or Path.cwd())
    cmds = detect_verify_commands(root)
    if not cmds:
        return ""
    parts: list[str] = ["[auto_verify]"]
    for cmd in cmds:
        try:
            p = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            blob = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
            blob = blob.strip()
            if len(blob) > 4_000:
                from clawagents.memory.content_crush import crush_tool_output

                blob = crush_tool_output(blob, tool_name="execute").text
            status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
            parts.append(f"$ {' '.join(cmd)} → {status}\n{blob}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            parts.append(f"$ {' '.join(cmd)} → error: {exc}")
    return "\n\n".join(parts)


def maybe_verify_after_edit(
    tool_name: str,
    success: bool,
    *,
    workspace: str | Path | None = None,
    enabled: bool = True,
) -> str:
    if not enabled or not success or tool_name not in _EDIT_TOOLS:
        return ""
    return run_verify(workspace)
