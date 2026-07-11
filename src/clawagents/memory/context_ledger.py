"""Commit-boundary context ledger (context-ledger inspired, local-only).

After a successful ``git commit``, record a compact restorable entry keyed by
SHA. Later sessions inject the ledger; agents rehydrate via ``git show``.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LedgerEntry:
    sha: str
    subject: str
    files: list[str] = field(default_factory=list)
    signatures: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    chars_saved_est: int = 0

    def to_markdown(self) -> str:
        files = ", ".join(self.files[:20]) + ("…" if len(self.files) > 20 else "")
        sigs = "\n".join(f"  - `{s}`" for s in self.signatures[:30]) or "  - (none)"
        decs = "\n".join(f"  - {d}" for d in self.decisions[:12]) or "  - (see commit message)"
        return (
            f"### {self.sha[:12]} — {self.subject}\n"
            f"- files: {files or '(none)'}\n"
            f"- signatures:\n{sigs}\n"
            f"- decisions:\n{decs}\n"
            f"- rehydrate: `rehydrate_ledger(sha=\"{self.sha}\")` or `git show {self.sha}`\n"
        )


def ledger_path(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or Path.cwd()) / ".clawagents"
    root.mkdir(parents=True, exist_ok=True)
    return root / "context-ledger.md"


def _run_git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


_SIG_RE = re.compile(
    r"(?m)^\s*(?:def |async def |class |export (?:default )?(?:async )?function |"
    r"fn |pub (?:async )?fn |func |interface |struct |type )\s*[\w.]+"
)


def _extract_signatures(diff_text: str, *, limit: int = 40) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in _SIG_RE.finditer(diff_text):
        line = m.group(0).strip()
        # Prefer added lines from unified diff
        key = line.lstrip("+").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        found.append(key[:160])
        if len(found) >= limit:
            break
    return found


def _decision_bullets(subject: str, body: str) -> list[str]:
    bullets: list[str] = []
    if subject:
        bullets.append(subject.strip())
    for line in (body or "").splitlines():
        s = line.strip()
        if s.startswith(("-", "*")) and len(s) > 3:
            bullets.append(s.lstrip("-* ").strip())
        elif s.lower().startswith(("fix", "add", "feat", "breaking")):
            bullets.append(s)
        if len(bullets) >= 8:
            break
    return bullets


def record_commit_ledger(
    *,
    workspace: str | Path | None = None,
    sha: str | None = None,
) -> LedgerEntry | None:
    """Record HEAD (or ``sha``) into the project ledger. Returns entry or None."""
    cwd = Path(workspace or Path.cwd())
    target = sha or _run_git(["rev-parse", "HEAD"], cwd)
    if not target:
        return None

    # Skip if already recorded
    path = ledger_path(cwd)
    if path.is_file() and target[:12] in path.read_text(encoding="utf-8", errors="replace"):
        return None

    subject = _run_git(["log", "-1", "--format=%s", target], cwd) or "(no subject)"
    body = _run_git(["log", "-1", "--format=%b", target], cwd)
    files_raw = _run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", target], cwd)
    files = [f for f in files_raw.splitlines() if f.strip()]
    diff = _run_git(["show", "--format=", "--unified=0", target], cwd)
    entry = LedgerEntry(
        sha=target,
        subject=subject[:200],
        files=files[:80],
        signatures=_extract_signatures(diff),
        decisions=_decision_bullets(subject, body),
        chars_saved_est=max(0, len(diff) - 400),
    )

    header = "# Context Ledger\n\nRestorable feature memory. Rehydrate with `rehydrate_ledger`.\n\n"
    existing = ""
    if path.is_file():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if not existing.startswith("#"):
            existing = header + existing
    else:
        existing = header
    path.write_text(existing.rstrip() + "\n\n" + entry.to_markdown(), encoding="utf-8")

    # Machine index for fast lookup
    index = cwd / ".clawagents" / "context-ledger.jsonl"
    with index.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry)) + "\n")
    return entry


def load_ledger_preamble(
    *,
    workspace: str | Path | None = None,
    max_chars: int = 6_000,
) -> str:
    path = ledger_path(workspace)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 40] + "\n… [ledger truncated] …\n"
    return f"## Context Ledger\n\n{text}\n"


def rehydrate_from_git(
    sha: str,
    *,
    workspace: str | Path | None = None,
    path: str | None = None,
    max_chars: int = 80_000,
) -> tuple[bool, str]:
    cwd = Path(workspace or Path.cwd())
    sha = (sha or "").strip()
    if not sha:
        return False, "sha is required"
    if path:
        out = _run_git(["show", f"{sha}:{path}"], cwd)
        if not out and out != "":
            # empty file ok; missing path fails with empty from _run_git on error
            probe = _run_git(["cat-file", "-e", f"{sha}:{path}"], cwd)
            if probe is None:
                pass
        # Prefer show with path
        try:
            proc = subprocess.run(
                ["git", "show", f"{sha}:{path}"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                return False, proc.stderr.strip() or f"cannot show {sha}:{path}"
            text = proc.stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
    else:
        text = _run_git(["show", "--stat", "--format=fuller", sha], cwd)
        if not text:
            return False, f"unknown commit {sha}"
        # Also include a capped patch
        patch = _run_git(["show", "--format=", sha], cwd)
        if patch:
            text = text + "\n\n" + patch
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n… [truncated at {max_chars} chars] …"
    return True, text


def maybe_record_from_tool_result(
    tool_name: str,
    args: dict[str, Any],
    output: str,
    *,
    workspace: str | Path | None = None,
) -> LedgerEntry | None:
    """If execute/git commit succeeded, append a ledger entry."""
    name = (tool_name or "").lower()
    if name not in {"execute", "execute_command", "bash", "run_command", "git_commit"}:
        return None
    cmd = str(args.get("command") or args.get("cmd") or "")
    if name == "git_commit" or re.search(r"\bgit\s+commit\b", cmd):
        if "error" in (output or "").lower() and "committed" not in (output or "").lower():
            # Heuristic: still try HEAD — commit may have succeeded
            pass
        return record_commit_ledger(workspace=workspace)
    return None
