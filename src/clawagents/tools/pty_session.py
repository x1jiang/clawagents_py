"""Interactive PTY shell sessions — Grok ptyctl parity via pexpect+pyte.

Optional deps: ``pip install 'clawagents[pty]'`` (pexpect, pyte).
Without them, tools return a clear install hint.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401 — Any used by PtyStartTool


def _pty_available() -> bool:
    try:
        import pexpect  # noqa: F401
        import pyte  # noqa: F401

        return True
    except ImportError:
        return False


def _sanitized_pty_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Scrub secrets from the PTY environment (same floor as LocalBackend.exec)."""
    from clawagents.redact import is_secret_name
    from clawagents.sandbox.local import LocalBackend

    base = {
        k: v
        for k, v in os.environ.items()
        if k not in LocalBackend._SENSITIVE_ENV_KEYS and not is_secret_name(k)
    }
    if extra:
        for k, v in extra.items():
            if k in LocalBackend._SENSITIVE_ENV_KEYS or is_secret_name(k):
                continue
            base[k] = v
    return base


@dataclass
class WaitDiagnostics:
    screen: str
    cursor_row: int
    cursor_col: int
    ended: bool
    raw_tail: str = ""


@dataclass
class WaitOutcome:
    matched: bool
    elapsed_ms: int
    diagnostics: WaitDiagnostics | None = None


class PtySession:
    """Headless terminal session with screen grid + vim-notation keys."""

    def __init__(
        self,
        command: str | list[str] | None = None,
        *,
        cols: int = 120,
        rows: int = 40,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        if not _pty_available():
            raise ImportError(
                "PTY sessions need pexpect+pyte. Install: pip install 'clawagents[pty]'"
            )
        import pexpect
        import pyte

        self.session_id = f"pty_{uuid.uuid4().hex[:10]}"
        self.cols = cols
        self.rows = rows
        self._ended = False
        self._raw_tail = bytearray()
        self._lock = threading.Lock()
        self._generation = 0

        cmd = command or os.environ.get("SHELL", "/bin/bash")
        if isinstance(cmd, list):
            cmd = " ".join(cmd)

        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.Stream(self._screen)
        self._child = pexpect.spawn(
            cmd,
            cwd=cwd or os.getcwd(),
            env=_sanitized_pty_env(env),
            encoding=None,
            dimensions=(rows, cols),
            timeout=None,
        )
        self._last_used = time.time()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        time.sleep(0.05)
        self._drain()

    def _read_loop(self) -> None:
        while not self._ended:
            try:
                data = self._child.read_nonblocking(size=4096, timeout=0.2)
            except Exception:
                if not self._child.isalive():
                    self._ended = True
                    break
                continue
            if not data:
                continue
            with self._lock:
                self._raw_tail.extend(data[-2048:])
                if len(self._raw_tail) > 4096:
                    self._raw_tail = self._raw_tail[-2048:]
                try:
                    text = data.decode("utf-8", errors="replace")
                except Exception:
                    text = str(data)
                self._stream.feed(text)
                self._generation += 1

    def _drain(self) -> None:
        time.sleep(0.02)

    def screen_text(self, *, include_empty: bool = False) -> str:
        with self._lock:
            lines = []
            for row in self._screen.display:
                line = "".join(row).rstrip()
                if include_empty or line.strip():
                    lines.append(line)
            return "\n".join(lines)

    def cursor(self) -> tuple[int, int]:
        with self._lock:
            # pyte is 0-indexed; expose 1-indexed
            return (self._screen.cursor.y + 1, self._screen.cursor.x + 1)

    @staticmethod
    def parse_keys(notation: str) -> bytes:
        """Parse vim-notation keys: ``hello<CR>``, ``<Esc>:wq<CR>``, ``<C-c>``."""
        specials = {
            "cr": b"\r",
            "enter": b"\r",
            "lf": b"\n",
            "esc": b"\x1b",
            "escape": b"\x1b",
            "tab": b"\t",
            "space": b" ",
            "bs": b"\x7f",
            "backspace": b"\x7f",
            "up": b"\x1b[A",
            "down": b"\x1b[B",
            "right": b"\x1b[C",
            "left": b"\x1b[D",
            "home": b"\x1b[H",
            "end": b"\x1b[F",
        }
        out = bytearray()
        i = 0
        s = notation or ""
        while i < len(s):
            if s[i] == "<":
                j = s.find(">", i)
                if j < 0:
                    out.extend(s[i:].encode())
                    break
                token = s[i + 1 : j].strip()
                low = token.lower()
                if low in specials:
                    out.extend(specials[low])
                elif low.startswith("c-") and len(low) == 3:
                    ch = low[2]
                    out.append(ord(ch) & 0x1F)
                elif low.startswith("m-") or low.startswith("a-"):
                    ch = token.split("-", 1)[-1]
                    out.extend(b"\x1b" + ch.encode()[:1])
                elif low.startswith("s-") and len(token) >= 3:
                    out.extend(token.split("-", 1)[-1].upper().encode())
                else:
                    out.extend(token.encode())
                i = j + 1
            else:
                out.append(ord(s[i]))
                i += 1
        return bytes(out)

    def send_keys(self, notation: str) -> None:
        data = self.parse_keys(notation)
        if self._ended:
            raise RuntimeError("PTY session has ended")
        self._child.write(data)
        self._drain()

    def send_bytes(self, data: bytes) -> None:
        if self._ended:
            raise RuntimeError("PTY session has ended")
        self._child.write(data)
        self._drain()

    def wait_for(
        self,
        *,
        text: str | None = None,
        regex: str | None = None,
        gone: str | None = None,
        stable_ms: int | None = None,
        timeout_ms: int = 10_000,
    ) -> WaitOutcome:
        conditions = [text, regex, gone, stable_ms]
        if sum(c is not None for c in conditions) != 1:
            raise ValueError("Provide exactly one of text/regex/gone/stable_ms")
        timeout_ms = max(1, min(int(timeout_ms), 120_000))
        deadline = time.time() + timeout_ms / 1000.0
        start = time.time()
        pattern = re.compile(regex) if regex else None
        last_gen = -1
        stable_since = time.time()

        while time.time() < deadline:
            if self._ended:
                break
            with self._lock:
                gen = self._generation
            screen = self.screen_text(include_empty=True)
            matched = False
            if text is not None:
                matched = text in screen
            elif pattern is not None:
                matched = pattern.search(screen) is not None
            elif gone is not None:
                matched = gone not in screen
            elif stable_ms is not None:
                if gen != last_gen:
                    last_gen = gen
                    stable_since = time.time()
                matched = (time.time() - stable_since) * 1000 >= stable_ms
            if matched:
                return WaitOutcome(
                    matched=True,
                    elapsed_ms=int((time.time() - start) * 1000),
                )
            time.sleep(0.05)

        row, col = self.cursor()
        with self._lock:
            raw = bytes(self._raw_tail[-2048:]).decode("utf-8", errors="replace")
        return WaitOutcome(
            matched=False,
            elapsed_ms=int((time.time() - start) * 1000),
            diagnostics=WaitDiagnostics(
                screen=self.screen_text(include_empty=True),
                cursor_row=row,
                cursor_col=col,
                ended=self._ended,
                raw_tail=raw,
            ),
        )

    def status(self) -> dict[str, Any]:
        row, col = self.cursor()
        return {
            "session_id": self.session_id,
            "alive": not self._ended and bool(self._child.isalive()),
            "cols": self.cols,
            "rows": self.rows,
            "cursor": {"row": row, "col": col},
            "generation": self._generation,
        }

    def stop(self) -> None:
        self._ended = True
        try:
            self._child.terminate(force=True)
        except Exception:
            pass


_SESSIONS: Dict[str, PtySession] = {}
_SESSIONS_LOCK = threading.Lock()
_SESSION_TTL_S = 30 * 60  # idle reap


def _reap_idle_sessions() -> None:
    """Drop ended or long-idle PTY sessions so they cannot leak forever."""
    now = time.time()
    dead: list[str] = []
    with _SESSIONS_LOCK:
        for sid, sess in list(_SESSIONS.items()):
            idle = now - float(getattr(sess, "_last_used", now) or now)
            ended = bool(getattr(sess, "_ended", False))
            alive = False
            try:
                alive = (not ended) and bool(sess._child.isalive())
            except Exception:
                alive = False
            if ended or not alive or idle > _SESSION_TTL_S:
                dead.append(sid)
        for sid in dead:
            sess = _SESSIONS.pop(sid, None)
            if sess is not None:
                try:
                    sess.stop()
                except Exception:
                    pass


def get_session(session_id: str) -> PtySession | None:
    _reap_idle_sessions()
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(session_id)
        if sess is not None:
            sess._last_used = time.time()
        return sess


def create_pty_tools():
    """Registerable tool objects for PTY sessions."""
    from clawagents.tools.registry import Tool, ToolResult

    class PtyStartTool:
        name = "pty_start"
        description = (
            "Start an interactive PTY shell session for TTY apps (dev servers, "
            "REPLs, vim, ssh prompts). Returns session_id. Requires clawagents[pty]. "
            "For non-interactive scripts and builds, prefer execute instead."
        )
        parameters = {
            "command": {"type": "string", "description": "Shell command (default $SHELL)"},
            "cols": {"type": "number", "description": "Columns (default 120)"},
            "rows": {"type": "number", "description": "Rows (default 40)"},
            "cwd": {
                "type": "string",
                "description": "Working directory (default: execute shell_session cwd when available)",
            },
        }
        keywords = ["pty", "shell", "terminal", "repl"]

        async def execute(self, args: dict, run_context: Any = None) -> ToolResult:
            from clawagents.config.features import is_enabled

            if not is_enabled("pty_sessions"):
                return ToolResult(success=False, output="", error="pty_sessions feature disabled")
            if not _pty_available():
                return ToolResult(
                    success=False,
                    output="",
                    error="Install PTY deps: pip install 'clawagents[pty]'",
                )
            cwd = str(args.get("cwd") or "").strip() or None
            if cwd is None and run_context is not None:
                sess_state = getattr(run_context, "shell_session", None)
                if sess_state is not None and getattr(sess_state, "cwd", None):
                    cwd = str(sess_state.cwd)
            if cwd is not None:
                try:
                    cwd = str(Path(cwd).expanduser().resolve())
                except OSError as exc:
                    return ToolResult(
                        success=False, output="", error=f"pty_start invalid cwd: {exc}"
                    )
                if not os.path.isdir(cwd):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"pty_start failed: cwd is not a directory: {cwd}",
                    )
            try:
                cols = max(20, min(500, int(args.get("cols") or 120)))
                rows = max(5, min(200, int(args.get("rows") or 40)))
            except (TypeError, ValueError):
                cols, rows = 120, 40
            try:
                sess = PtySession(
                    args.get("command"),
                    cols=cols,
                    rows=rows,
                    cwd=cwd,
                )
            except Exception as exc:  # noqa: BLE001
                return ToolResult(success=False, output="", error=str(exc))
            sess._last_used = time.time()
            with _SESSIONS_LOCK:
                _SESSIONS[sess.session_id] = sess
            _reap_idle_sessions()
            screen = sess.screen_text()
            return ToolResult(
                success=True,
                output=f"session_id={sess.session_id}\ncwd={cwd or ''}\n{screen[-2000:]}",
            )

    class PtyKeysTool:
        name = "pty_keys"
        description = "Send vim-notation keys to a PTY session (e.g. '<Esc>:wq<CR>', '<C-c>')."
        parameters = {
            "session_id": {"type": "string", "required": True},
            "keys": {"type": "string", "required": True, "description": "Vim-notation key sequence"},
        }

        async def execute(self, args: dict) -> ToolResult:
            sess = get_session(str(args.get("session_id") or ""))
            if sess is None:
                return ToolResult(success=False, output="", error="unknown session_id")
            try:
                sess.send_keys(str(args.get("keys") or ""))
            except Exception as exc:  # noqa: BLE001
                return ToolResult(success=False, output="", error=str(exc))
            return ToolResult(success=True, output=sess.screen_text()[-3000:])

    class PtyScreenTool:
        name = "pty_screen"
        description = "Read the rendered PTY screen grid (+ cursor)."
        parameters = {
            "session_id": {"type": "string", "required": True},
        }

        async def execute(self, args: dict) -> ToolResult:
            sess = get_session(str(args.get("session_id") or ""))
            if sess is None:
                return ToolResult(success=False, output="", error="unknown session_id")
            row, col = sess.cursor()
            st = sess.status()
            return ToolResult(
                success=True,
                output=f"cursor=({row},{col}) alive={st['alive']}\n{sess.screen_text(include_empty=True)}",
            )

    class PtyWaitTool:
        name = "pty_wait"
        description = (
            "Wait for Text/Regex/Gone/StableMs on a PTY screen. "
            "On timeout returns full screen+cursor diagnostics (no follow-up needed)."
        )
        parameters = {
            "session_id": {"type": "string", "required": True},
            "text": {"type": "string"},
            "regex": {"type": "string"},
            "gone": {"type": "string"},
            "stable_ms": {"type": "number"},
            "timeout_ms": {"type": "number", "description": "Default 10000, max 120000"},
        }

        async def execute(self, args: dict) -> ToolResult:
            import asyncio

            _reap_idle_sessions()
            sess = get_session(str(args.get("session_id") or ""))
            if sess is None:
                return ToolResult(success=False, output="", error="unknown session_id")
            try:
                sess._last_used = time.time()
                outcome = await asyncio.to_thread(
                    sess.wait_for,
                    text=args.get("text"),
                    regex=args.get("regex"),
                    gone=args.get("gone"),
                    stable_ms=int(args["stable_ms"]) if args.get("stable_ms") is not None else None,
                    timeout_ms=int(args.get("timeout_ms") or 10_000),
                )
            except Exception as exc:  # noqa: BLE001
                return ToolResult(success=False, output="", error=str(exc))
            if outcome.matched:
                return ToolResult(
                    success=True,
                    output=f"matched in {outcome.elapsed_ms}ms\n{sess.screen_text()[-2000:]}",
                )
            diag = outcome.diagnostics
            body = (
                f"timeout after {outcome.elapsed_ms}ms\n"
                f"cursor=({diag.cursor_row},{diag.cursor_col}) ended={diag.ended}\n"
                f"{diag.screen}"
            )
            return ToolResult(success=True, output=body)

    class PtyStopTool:
        name = "pty_stop"
        description = "Terminate a PTY session."
        parameters = {"session_id": {"type": "string", "required": True}}

        async def execute(self, args: dict) -> ToolResult:
            sid = str(args.get("session_id") or "")
            with _SESSIONS_LOCK:
                sess = _SESSIONS.pop(sid, None)
            if sess is None:
                return ToolResult(success=False, output="", error="unknown session_id")
            sess.stop()
            return ToolResult(success=True, output=f"stopped {sid}")

    return [PtyStartTool(), PtyKeysTool(), PtyScreenTool(), PtyWaitTool(), PtyStopTool()]


__all__ = [
    "PtySession",
    "WaitOutcome",
    "WaitDiagnostics",
    "create_pty_tools",
    "get_session",
    "_pty_available",
]
