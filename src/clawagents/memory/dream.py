"""Dream consolidation — merge session logs into durable MEMORY.md.

Grok Build parity (xai-grok-memory dream.rs): gated on elapsed time + session
count. Writes under ``.clawagents/MEMORY.md`` only — never overwrites a
human-authored workspace-root MEMORY.md. Cleans processed session files.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional


LLMComplete = Callable[[str], Awaitable[str]]


@dataclass
class DreamConfig:
    enabled: bool = True
    min_hours: float = 4.0
    min_sessions: int = 3
    max_chars: int = 16_000
    prompt_cap: int = 32_000
    stale_session_min_age_secs: float = 300.0


@dataclass
class DreamGateOpen:
    sessions: list[str]


@dataclass
class DreamResult:
    ok: bool
    reason: str = ""
    memory_path: str | None = None
    sessions_cleared: list[str] | None = None


def _sessions_dir(workspace: Path) -> Path:
    d = workspace / ".clawagents" / "memory-sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock_path(workspace: Path) -> Path:
    return workspace / ".clawagents" / "dream.lock"


def _state_path(workspace: Path) -> Path:
    return workspace / ".clawagents" / "dream_state.json"


def _load_state(workspace: Path) -> dict:
    p = _state_path(workspace)
    if not p.is_file():
        return {"last_dream_at": 0.0, "processed": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"last_dream_at": 0.0, "processed": []}


def _save_state(workspace: Path, state: dict) -> None:
    p = _state_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def append_session_log(
    text: str,
    *,
    workspace: str | Path | None = None,
    stem: str | None = None,
) -> Path | None:
    """Append a session memory log for later dream consolidation."""
    from clawagents.config.features import is_enabled

    if not is_enabled("memory_dream") and not is_enabled("smart_memory"):
        return None
    body = (text or "").strip()
    if not body:
        return None
    ws = Path(workspace or os.getcwd()).resolve()
    d = _sessions_dir(ws)
    name = stem or f"sess_{int(time.time())}"
    path = d / f"{name}.md"
    with path.open("a", encoding="utf-8") as f:
        f.write(body.rstrip() + "\n\n")
    return path


def check_dream_gates(
    workspace: str | Path | None = None,
    config: DreamConfig | None = None,
) -> DreamGateOpen | str:
    """Return DreamGateOpen or a closed reason string."""
    cfg = config or DreamConfig()
    if not cfg.enabled:
        return "Disabled"
    ws = Path(workspace or os.getcwd()).resolve()
    state = _load_state(ws)
    hours = (time.time() - float(state.get("last_dream_at") or 0)) / 3600.0
    if hours < cfg.min_hours:
        return "TooSoon"
    sessions = sorted(
        p.stem
        for p in _sessions_dir(ws).glob("*.md")
        if p.is_file() and p.stat().st_size > 0
    )
    if len(sessions) < cfg.min_sessions:
        return "TooFewSessions"
    return DreamGateOpen(sessions=sessions)


def build_dream_user_message(
    workspace: Path,
    stems: list[str],
    existing_memory: str | None,
    *,
    prompt_cap: int = 32_000,
) -> str | None:
    parts: list[str] = [
        "You are consolidating agent session memories into a durable MEMORY.md.",
        "Merge facts, resolve contradictions (prefer newer sessions), drop noise.",
        "Output markdown with clear ## headers. If nothing durable, reply NO_REPLY.",
        "",
    ]
    if existing_memory and existing_memory.strip():
        # Skip tiny scaffold templates
        if len(existing_memory.encode("utf-8")) >= 500 or "TODO" not in existing_memory:
            parts.append("## Existing MEMORY.md\n" + existing_memory[:8000])
            parts.append("")
    sess_dir = _sessions_dir(workspace)
    for stem in stems:
        path = sess_dir / f"{stem}.md"
        if not path.is_file():
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        parts.append(f"## Session `{stem}`\n{body[:6000]}")
    msg = "\n\n".join(parts)
    if len(msg) > prompt_cap:
        msg = msg[:prompt_cap]
    return msg if stems else None


def process_dream_response(response: str, *, max_chars: int = 16_000) -> str | None:
    text = (response or "").strip()
    if not text or text.upper() == "NO_REPLY":
        return None
    if "##" not in text and "# " not in text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


async def run_dream(
    llm_complete: LLMComplete,
    *,
    workspace: str | Path | None = None,
    config: DreamConfig | None = None,
) -> DreamResult:
    """Execute dream consolidation when gates open."""
    from clawagents.config.features import is_enabled

    if not is_enabled("memory_dream"):
        return DreamResult(ok=False, reason="feature_disabled")

    cfg = config or DreamConfig()
    ws = Path(workspace or os.getcwd()).resolve()
    gate = check_dream_gates(ws, cfg)
    if isinstance(gate, str):
        return DreamResult(ok=False, reason=gate)

    lock = _lock_path(ws)
    if lock.exists():
        # Stale lock > 1h → take over
        try:
            if time.time() - lock.stat().st_mtime < 3600:
                return DreamResult(ok=False, reason="locked")
        except OSError:
            pass
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        return DreamResult(ok=False, reason=f"lock_failed:{exc}")

    # Always release the lock — including CancelledError from wait_for timeout.
    try:
        memory_path = ws / ".clawagents" / "MEMORY.md"
        # Never overwrite a human-authored workspace-root MEMORY.md.
        legacy_root = ws / "MEMORY.md"
        existing = ""
        if memory_path.is_file():
            try:
                existing = memory_path.read_text(encoding="utf-8")
            except OSError:
                existing = ""
        elif legacy_root.is_file():
            try:
                existing = legacy_root.read_text(encoding="utf-8")
            except OSError:
                existing = ""

        prompt = build_dream_user_message(
            ws, gate.sessions, existing, prompt_cap=cfg.prompt_cap
        )
        if not prompt:
            return DreamResult(ok=False, reason="empty_prompt")

        try:
            raw = await llm_complete(prompt)
        except Exception as exc:  # noqa: BLE001
            return DreamResult(ok=False, reason=f"llm_error:{exc}")

        consolidated = process_dream_response(raw, max_chars=cfg.max_chars)
        if not consolidated:
            return DreamResult(ok=False, reason="nothing_to_store")

        try:
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory_path.write_text(consolidated.rstrip() + "\n", encoding="utf-8")
        except OSError as exc:
            return DreamResult(ok=False, reason=f"write_failed:{exc}")

        cleared: list[str] = []
        now = time.time()
        sess_dir = _sessions_dir(ws)
        for stem in gate.sessions:
            path = sess_dir / f"{stem}.md"
            try:
                if path.is_file() and now - path.stat().st_mtime >= cfg.stale_session_min_age_secs:
                    path.unlink(missing_ok=True)
                    cleared.append(stem)
            except OSError:
                continue

        state = _load_state(ws)
        state["last_dream_at"] = time.time()
        processed = list(state.get("processed") or [])
        processed.extend(cleared)
        state["processed"] = processed[-200:]
        _save_state(ws, state)

        try:
            from clawagents.memory.smart_store import ingest_text

            ingest_text(
                consolidated,
                path=".clawagents/MEMORY.md",
                source="curated",
                workspace=ws,
                chunk_id="memory_md",
            )
        except Exception:
            pass

        return DreamResult(
            ok=True,
            reason="consolidated",
            memory_path=str(memory_path),
            sessions_cleared=cleared,
        )
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "DreamConfig",
    "DreamGateOpen",
    "DreamResult",
    "append_session_log",
    "check_dream_gates",
    "build_dream_user_message",
    "process_dream_response",
    "run_dream",
]
