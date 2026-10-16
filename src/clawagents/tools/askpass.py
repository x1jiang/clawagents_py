"""Interactive AskPass bridge for terminal commands (SSH, sudo, git, etc.).

Routes password, passphrase, and host-key confirmation prompts from child
processes (`SSH_ASKPASS`, `SUDO_ASKPASS`, `GIT_ASKPASS`) directly to the user
via the active `ask_prompt_fn` callback on `RunContext` (e.g., the VS Code
agent chat window or CLI prompt) without ever exposing secrets to the LLM
context, tool output, or session logs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import shlex
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# Prompts that ask for a non-secret confirmation (such as SSH host key checks)
# should be shown as normal text inputs rather than masked password inputs.
_NON_SECRET_PROMPT_RE = re.compile(
    r"(?i)(?:\(yes/no(?:/\[fingerprint\])?\)"
    r"|continue connecting"
    r"|authenticity of host"
    r"|key fingerprint is)",
)

# Prompts that ask for a secret/credential (used to auto-detect secret mode on
# general ask_user calls when the caller did not pass secret=True explicitly).
_SECRET_QUESTION_RE = re.compile(
    r"(?i)\b(?:password|passphrase|secret|api[_\s-]?key|access[_\s-]?token|"
    r"auth[_\s-]?token|bearer[_\s-]?token|private[_\s-]?key|pin|otp|"
    r"verification\s+code|two-factor|2fa|sudo\s+password)\b",
)

# Match `-o BatchMode=yes` / `-oBatchMode=yes` / `-o BatchMode=true` so SSH can
# use SSH_ASKPASS when interactive askpass is available.
_SSH_BATCHMODE_YES_RE = re.compile(
    r"(-o\s*BatchMode\s*=\s*)(?:yes|true)\b",
    re.IGNORECASE,
)

# Match `sudo` at command boundary when not already using -A/-S/-n/-k/-K/-V/-h.
_SUDO_BARE_RE = re.compile(
    r"(^|(?:&&|\|\||[;|])\s*)(sudo)(\s+)(?!-(?:[A-Za-z]*[ASnkKVh]|-askpass|-stdin|-non-interactive|-help|-version)\b)",
)

_CLIENT_SCRIPT = """\
import json
import os
import socket
import sys


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() or "Password:"
    nonce = os.environ.get("CLAW_ASKPASS_NONCE", "")
    sock_path = os.environ.get("CLAW_ASKPASS_SOCK", "")
    port_str = os.environ.get("CLAW_ASKPASS_PORT", "")

    payload = (json.dumps({"nonce": nonce, "prompt": prompt}) + "\\n").encode("utf-8")
    sock = None
    try:
        if sock_path and hasattr(socket, "AF_UNIX") and os.path.exists(sock_path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(3600.0)
                s.connect(sock_path)
                sock = s
            except Exception:
                sock = None
        if sock is None and port_str:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3600.0)
            s.connect(("127.0.0.1", int(port_str)))
            sock = s
        if sock is None:
            return 1

        sock.sendall(payload)
        buf = bytearray()
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
            if b"\\n" in chunk:
                break
        line = bytes(buf).split(b"\\n", 1)[0].decode("utf-8", errors="replace").strip()
        if not line:
            return 1
        data = json.loads(line)
        if not data.get("ok"):
            return 1
        value = data.get("value")
        if value is None:
            return 1
        sys.stdout.write(str(value) + "\\n")
        sys.stdout.flush()
        return 0
    except Exception:
        return 1
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
"""


def is_secret_prompt(prompt: str) -> bool:
    """Return True if an askpass/terminal prompt should mask user input."""
    text = (prompt or "").strip()
    if not text:
        return True
    if _NON_SECRET_PROMPT_RE.search(text):
        return False
    return True


def looks_like_secret_question(question: str) -> bool:
    """Return True if a general question asks for a password, passphrase, or secret."""
    text = (question or "").strip()
    if not text or _NON_SECRET_PROMPT_RE.search(text):
        return False
    return bool(_SECRET_QUESTION_RE.search(text))


def resolve_ask_prompt_fn(run_context: Any) -> Callable[..., Any] | None:
    """Extract the interactive prompt callback from `run_context` if available."""
    if run_context is None:
        return None
    fn = getattr(run_context, "ask_prompt_fn", None)
    if callable(fn):
        return fn
    meta = getattr(run_context, "_metadata", None)
    if isinstance(meta, dict):
        fn = meta.get("ask_prompt_fn")
        if callable(fn):
            return fn
    return None


def rewrite_command_for_askpass(command: str, *, has_askpass: bool) -> str:
    """Normalize SSH BatchMode and bare sudo so interactive askpass can trigger."""
    if not has_askpass or not command:
        return command
    if os.environ.get("CLAW_KEEP_SSH_BATCHMODE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        command = _SSH_BATCHMODE_YES_RE.sub(r"\1no", command)
    if os.environ.get("CLAW_DISABLE_SUDO_ASKPASS_REWRITE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        command = _SUDO_BARE_RE.sub(r"\1\2 -A\3", command)
    return command


ASKPASS_HUMAN_TIMEOUT_S: float = 300.0

_INTERACTIVE_SECRET_CMD_RE = re.compile(
    r"\b(?:ssh|scp|sftp|rsync|sudo|su|doas|ssh-add|git\s+(?:push|pull|fetch|clone))\b",
    re.IGNORECASE,
)


def command_may_prompt_for_secret(command: str) -> bool:
    """Return True if ``command`` invokes a tool that may trigger SSH/SUDO/GIT_ASKPASS."""
    text = (command or "").strip()
    if not text:
        return False
    return bool(_INTERACTIVE_SECRET_CMD_RE.search(text))


class AskpassSession:
    """Local IPC server + helper scripts for `SSH_ASKPASS` / `SUDO_ASKPASS` / `GIT_ASKPASS`."""

    def __init__(
        self,
        ask_prompt_fn: Callable[..., Any] | None,
        *,
        command: str = "",
    ) -> None:
        self._ask_prompt_fn = ask_prompt_fn
        self.command = command
        self.nonce = uuid.uuid4().hex
        self.tmpdir: str | None = None
        self.sock_path: str = ""
        self.port: int = 0
        self.script_path: str = ""
        self._tcp_sock: socket.socket | None = None
        self._unix_sock: socket.socket | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._active_prompts = 0
        self._prompt_done_cond = threading.Condition(self._lock)

    @property
    def enabled(self) -> bool:
        return callable(self._ask_prompt_fn)

    @property
    def active_prompts(self) -> int:
        with self._lock:
            return self._active_prompts

    def start(self) -> "AskpassSession":
        if not self.enabled or self.tmpdir is not None:
            return self
        # Short prefix in /tmp so AF_UNIX path stays well under macOS 104-byte limit.
        base_tmp = "/tmp" if sys.platform != "win32" and os.path.isdir("/tmp") else None
        self.tmpdir = tempfile.mkdtemp(prefix="claw-ap-", dir=base_tmp)
        os.chmod(self.tmpdir, stat.S_IRWXU)

        client_py = os.path.join(self.tmpdir, "client.py")
        Path(client_py).write_text(_CLIENT_SCRIPT, encoding="utf-8")

        if sys.platform == "win32":
            launcher = os.path.join(self.tmpdir, "askpass.bat")
            Path(launcher).write_text(
                f'@echo off\r\n"{sys.executable}" -E "{client_py}" %*\r\n',
                encoding="utf-8",
            )
            self.script_path = launcher
        else:
            launcher = os.path.join(self.tmpdir, "askpass")
            py_quoted = shlex.quote(sys.executable)
            client_quoted = shlex.quote(client_py)
            Path(launcher).write_text(
                f"#!/bin/sh\nexec {py_quoted} -E {client_quoted} \"$@\"\n",
                encoding="utf-8",
            )
            os.chmod(launcher, stat.S_IRWXU)
            self.script_path = launcher

        # Start TCP loopback listener (works across namespaces/sandboxes with loopback).
        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_sock.bind(("127.0.0.1", 0))
        tcp_sock.listen(8)
        self.port = int(tcp_sock.getsockname()[1])
        self._tcp_sock = tcp_sock
        threading.Thread(
            target=self._accept_loop,
            args=(tcp_sock,),
            name="claw-askpass-tcp",
            daemon=True,
        ).start()

        # Start AF_UNIX listener on POSIX (works even when network is unshared).
        if hasattr(socket, "AF_UNIX") and sys.platform != "win32":
            sock_path = os.path.join(self.tmpdir, "s")
            try:
                unix_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                unix_sock.bind(sock_path)
                os.chmod(sock_path, stat.S_IRUSR | stat.S_IWUSR)
                unix_sock.listen(8)
                self.sock_path = sock_path
                self._unix_sock = unix_sock
                threading.Thread(
                    target=self._accept_loop,
                    args=(unix_sock,),
                    name="claw-askpass-unix",
                    daemon=True,
                ).start()
            except Exception:
                self.sock_path = ""

        return self

    def env_overlay(self) -> dict[str, str]:
        if not self.enabled or not self.script_path:
            return {}
        overlay = {
            "SSH_ASKPASS": self.script_path,
            "SSH_ASKPASS_REQUIRE": "force",
            "GIT_ASKPASS": self.script_path,
            "SUDO_ASKPASS": self.script_path,
            "GIT_TERMINAL_PROMPT": "0",
            "CLAW_ASKPASS_PORT": str(self.port),
            "CLAW_ASKPASS_NONCE": self.nonce,
        }
        if self.sock_path:
            overlay["CLAW_ASKPASS_SOCK"] = self.sock_path
        if not os.environ.get("DISPLAY"):
            overlay["DISPLAY"] = "clawagents:0"
        return overlay

    def _accept_loop(self, srv: socket.socket) -> None:
        while not self._closed:
            try:
                conn, _ = srv.accept()
            except Exception:
                break
            threading.Thread(
                target=self._handle_client,
                args=(conn,),
                name="claw-askpass-client",
                daemon=True,
            ).start()

    def _invoke_callback(self, prompt: str, *, secret: bool) -> str | None:
        fn = self._ask_prompt_fn
        if fn is None:
            return None
        try:
            sig = inspect.signature(fn)
            kwargs: dict[str, Any] = {}
            if "secret" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["secret"] = secret
            if "command" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["command"] = self.command
            res = fn(prompt, **kwargs)
        except TypeError:
            res = fn(prompt)
        if inspect.isawaitable(res):
            return asyncio.run(res)
        return res

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            try:
                conn.settimeout(30.0)
                buf = bytearray()
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if b"\n" in chunk:
                        break
                line = bytes(buf).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
                if not line:
                    return
                req = json.loads(line)
                if req.get("nonce") != self.nonce:
                    conn.sendall(b'{"ok": false, "error": "bad nonce"}\n')
                    return
                prompt = str(req.get("prompt") or "Password:").strip()
                secret = is_secret_prompt(prompt)
                with self._lock:
                    self._active_prompts += 1
                try:
                    answer = self._invoke_callback(prompt, secret=secret)
                finally:
                    with self._lock:
                        self._active_prompts = max(0, self._active_prompts - 1)
                        self._prompt_done_cond.notify_all()
                if answer is None:
                    conn.sendall(b'{"ok": false}\n')
                else:
                    resp = json.dumps({"ok": True, "value": str(answer)}) + "\n"
                    conn.sendall(resp.encode("utf-8"))
            except Exception:
                try:
                    conn.sendall(b'{"ok": false}\n')
                except Exception:
                    pass

    async def wait_while_prompting(self, max_wait_s: float = 3600.0) -> None:
        """Wait asynchronously while at least one interactive prompt is active."""
        deadline = time.monotonic() + max_wait_s
        while self.active_prompts > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.1)

    async def wait_process(self, wait_task: "asyncio.Future[Any]", timeout_s: float) -> Any:
        """Wait for `wait_task` with `timeout_s`, pausing timeout while user is prompted."""
        while True:
            try:
                return await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout_s)
            except asyncio.TimeoutError:
                if self.active_prompts > 0:
                    await self.wait_while_prompting()
                    continue
                raise

    def close(self) -> None:
        self._closed = True
        for s in (self._tcp_sock, self._unix_sock):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        self._tcp_sock = None
        self._unix_sock = None
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def __enter__(self) -> "AskpassSession":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
