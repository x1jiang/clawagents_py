"""Post-edit syntax gate for write-class tools.

After a successful edit to ``.js`` / ``.mjs`` / ``.cjs`` / ``.py`` / ``.sh``,
run the ~50ms language checker and return a short note to append to the tool
result. Converts "caught if the model happens to check" into "caught same round".
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

# Keep this list tight — only checkers that are fast and local.
_CHECKERS: dict[str, tuple[str, list[str]]] = {
    ".js": ("node", ["node", "--check"]),
    ".mjs": ("node", ["node", "--check"]),
    ".cjs": ("node", ["node", "--check"]),
    ".py": ("python", ["python3", "-m", "py_compile"]),
    ".sh": ("bash", ["bash", "-n"]),
}

_WRITE_TOOLS_FOR_GATE = frozenset({
    "write_file", "edit_file", "apply_patch", "hashline_edit", "create_file",
    "replace_in_file", "insert_in_file", "insert_lines", "patch_file",
})


def _resolve_path(path_str: str, workspace: str | Path | None) -> Path | None:
    p = Path(path_str).expanduser()
    if not p.is_absolute() and workspace:
        p = Path(workspace) / p
    try:
        p = p.resolve()
    except OSError:
        return None
    if not p.is_file():
        return None
    return p


def _looks_like_esm(file_path: Path) -> bool:
    """Heuristic: ``.js`` with top-level import/export needs module-mode check.

    Plain ``node --check file.js`` silently returns 0 for some broken ESM
    ``.js`` files (Node treats them as modules but skips a real parse).
    """
    try:
        head = file_path.read_text(encoding="utf-8", errors="replace")[:8000]
    except OSError:
        return False
    for line in head.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        if s.startswith("import ") or s.startswith("export ") or s.startswith("import{"):
            return True
        break
    return "\nimport " in head or "\nexport " in head


def _run_cmd(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def run_syntax_gate(path: str | Path, *, timeout_s: float = 2.0) -> str | None:
    """Return a status line for *path*, or ``None`` if no checker applies."""
    file_path = Path(path)
    ext = file_path.suffix.lower()
    spec = _CHECKERS.get(ext)
    if spec is None:
        return None
    binary, argv_prefix = spec
    # Prefer python3; fall back to python for py_compile.
    if binary == "python":
        exe = shutil.which("python3") or shutil.which("python")
        if not exe:
            return None
        cmds = [[exe, "-m", "py_compile", str(file_path)]]
    elif binary == "node":
        if shutil.which("node") is None:
            return None
        cmds = [[*argv_prefix, str(file_path)]]
        # ESM-in-.js: re-check with module default type so syntax errors surface.
        if ext == ".js" and _looks_like_esm(file_path):
            cmds.append(
                ["node", "--experimental-default-type=module", "--check", str(file_path)]
            )
    else:
        if shutil.which(binary) is None:
            return None
        cmds = [[*argv_prefix, str(file_path)]]

    last_ok_cmd = cmds[0]
    try:
        for cmd in cmds:
            proc = _run_cmd(cmd, timeout_s=timeout_s)
            last_ok_cmd = cmd
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                if len(err) > 800:
                    err = err[:800] + "…"
                return (
                    f"[syntax_gate] {file_path.name}: FAILED (exit {proc.returncode})\n"
                    f"{err or '(no checker output)'}"
                )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[syntax_gate] checker skipped for {file_path.name}: {exc}"

    return f"[syntax_gate] {file_path.name}: ok ({' '.join(last_ok_cmd[:2])}…)"


def append_syntax_gate(
    tool_name: str,
    args: dict[str, Any] | None,
    output: str,
    *,
    workspace: str | Path | None = None,
) -> str:
    """Append syntax-gate note to a successful write-tool output string."""
    if tool_name not in _WRITE_TOOLS_FOR_GATE:
        return output
    if not isinstance(args, dict):
        return output
    path_str = (
        args.get("path")
        or args.get("file_path")
        or args.get("target_path")
        or args.get("file")
        or ""
    )
    if not isinstance(path_str, str) or not path_str.strip():
        return output
    resolved = _resolve_path(path_str.strip(), workspace)
    if resolved is None:
        return output
    note = run_syntax_gate(resolved)
    if not note:
        return output
    if not isinstance(output, str):
        output = str(output)
    if not output:
        return note
    return f"{output.rstrip()}\n\n{note}"
