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
import shlex
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from clawagents.permissions.mode import PermissionMode
from clawagents.sandbox.backend import ExecResult
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
    # Execute output crosses every display/persistence boundary. Scrub here so
    # failures cannot leak a provider key or a password that malformed shell
    # interpolation accidentally treated as a command name.
    from clawagents.redact import redact

    output = redact(output)
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    original_len = len(output)
    half = MAX_OUTPUT_CHARS // 2
    return (
        output[:half]
        + f"\n\n... [truncated {original_len - MAX_OUTPUT_CHARS} chars] ...\n\n"
        + output[-half:]
    )


def _git_not_a_repo_signal(command: str, exit_code: int, stdout: str, stderr: str) -> bool:
    if exit_code != 128:
        return False
    blob = f"{stdout}\n{stderr}".lower()
    if "not a git repository" in blob:
        return True
    # Common when agents chain ``node --check … && git diff`` outside a repo.
    cmd = (command or "").lower()
    return "git " in f" {cmd} " or cmd.strip().startswith("git")


def _sandbox_eperm_signal(stdout: str, stderr: str) -> bool:
    blob = f"{stdout}\n{stderr}"
    return "Operation not permitted" in blob or "EPERM" in blob


def _sandbox_write_hint(stdout: str, stderr: str) -> str | None:
    if not _sandbox_eperm_signal(stdout, stderr):
        return None
    blob = f"{stdout}\n{stderr}"
    try:
        import tempfile

        scratch = tempfile.gettempdir()
    except Exception:
        scratch = "<system temp>"
    # Scripts often do `cmd >/dev/null || echo fail` — the redirect fails, not cmd.
    if "/dev/null" in blob and (
        "credentials" not in blob.lower() and ".config/" not in blob
    ):
        return (
            "OS sandbox denied the redirect target /dev/null (not necessarily "
            "your command). On current clawagents this should be allowed — "
            "upgrade the package, or retry with execute(unsandboxed=true) when "
            "Full access is on, or set CLAW_SANDBOX_PROFILE=off."
        )
    home_config = (
        "gcloud" in blob.lower()
        or ".config/" in blob
        or "credentials.db" in blob
    )
    if home_config:
        return (
            "OS sandbox blocked a write outside the workspace (home config / "
            f"credentials). Workspace + {scratch} + /tmp are allowed by default. "
            "gcloud/aws/docker need ~/.config — enable Full access (chat_mode="
            "full_access with Allow Full Access; disables OS sandbox), retry "
            "with execute(unsandboxed=true) under that mode, or run in a "
            "normal macOS Terminal."
        )
    return (
        f"Sandbox write denied. Prefer the workspace or session scratch "
        f"({scratch}); /tmp and /private/tmp are also allowed when the OS "
        f"sandbox profile is active. Avoid writing outside those roots. "
        "Under Full access you may retry with execute(unsandboxed=true)."
    )


def _may_run_unsandboxed(run_context: Any, args: Dict[str, Any]) -> bool:
    """True when this call is allowed to skip seatbelt/bwrap wrap."""
    if not _truthy(args.get("unsandboxed")):
        return False
    meta = getattr(run_context, "_metadata", None)
    if not isinstance(meta, dict):
        return False
    return bool(meta.get("allow_unsandboxed_exec"))


def _short_exit_error(exit_code: int) -> str:
    """UI/error field: exit code only — full command lives in structured output."""
    return f"Command exited with code {exit_code}"


def _is_empty_search_result(command: str, exit_code: int, stdout: str, stderr: str) -> bool:
    """Recognize grep/rg's documented exit-1 meaning: no selected lines."""
    if exit_code != 1 or stdout.strip() or stderr.strip():
        return False
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return False
    if not tokens or any(token in {"&&", "||", ";", "|", "&"} for token in tokens):
        return False
    index = 0
    while index < len(tokens) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]):
        index += 1
    if index >= len(tokens):
        return False
    program = os.path.basename(tokens[index].lstrip("\\"))
    return program in {"grep", "egrep", "fgrep", "rg"}


def _failure_diagnostic_hint(stdout: str, stderr: str) -> str:
    """Classify failures that require diagnosis/user action, not tool churn."""
    blob = f"{stdout}\n{stderr}"
    if re.search(
        r"NT_STATUS_(?:LOGON_FAILURE|ACCESS_DENIED)|STATUS_LOGON_FAILURE|"
        r"authentication (?:failed|failure)|invalid credentials",
        blob,
        re.IGNORECASE,
    ):
        return (
            "Remote authentication was rejected. Stop changing runtimes, clients, "
            "or transport code: they reached the service. Report the exact server "
            "response and ask the user to verify/rotate the credential and the "
            "domain-qualified username before retrying."
        )
    if re.search(r"PackagesNotFoundError|No matching distribution found", blob):
        return (
            "The requested package is unavailable from the configured repository. "
            "Do not switch package managers or install unrelated tooling unless the "
            "user requests it; verify the package name/channel or use an already "
            "available runtime."
        )
    missing_modules = list(
        dict.fromkeys(
            re.findall(
                r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]",
                blob,
            )
        )
    )
    if missing_modules:
        shown = ", ".join(f"`{name}`" for name in missing_modules)
        probe = missing_modules[0].split(".", 1)[0]
        return (
            f"The selected Python interpreter is missing module {shown}; execution "
            "stopped at import and did not reach the application or data logic. "
            "First use the intended project interpreter (for example "
            "`.venv/bin/python`) and check with that same interpreter using "
            f"`.venv/bin/python -m pip show {probe}`. Do not use a bare pip or "
            "global install. If it is truly absent, install/add the project-declared "
            "dependency in that environment."
        )
    if re.search(
        r"(?m)(?<=: )"
        r"(?=[A-Za-z0-9_+./=\-]{10,}:(?: command)? not found$)"
        r"(?=[^:\n]*[A-Z])(?=[^:\n]*[a-z])(?=[^:\n]*\d)"
        r"[A-Za-z0-9_+./=\-]+(?=:(?: command)? not found$)",
        blob,
    ):
        return (
            "A high-entropy value was interpreted as a shell command. Treat this as "
            "unsafe secret interpolation: stop, do not print the value, avoid "
            "sourcing .env, and pass credentials through an env/auth file."
        )
    missing_inputs = list(
        dict.fromkeys(
            match.strip()
            for match in re.findall(
                r"Input file does not exist:\s*([^\r\n]+)", blob
            )
            if match.strip()
        )
    )
    empty_json = re.search(
        r"JSONDecodeError:\s*Expecting value:\s*line 1 column 1",
        blob,
    )
    if missing_inputs and empty_json:
        shown = ", ".join(f"`{path}`" for path in missing_inputs[:3])
        if len(missing_inputs) > 3:
            shown += f", and {len(missing_inputs) - 3} more"
        partial = ""
        if re.search(r"(?m)^pages\s+\d+\s+packets\s+\d+", stdout):
            partial = (
                " Stdout shows at least one iteration succeeded; preserve that "
                "partial success when reporting results."
            )
        return (
            f"Primary failures are missing input files: {shown}. The JSON decode "
            "errors are secondary: shell redirection created empty output files "
            "before the producer failed, then the consumer tried to parse them. "
            "Do not debug the JSON parser. Preflight input paths and guard each "
            "producer result before running its consumer."
            f"{partial}"
        )
    if "subprocess.py" in blob and ("_execute_child" in blob or "Popen" in blob):
        missing = list(
            dict.fromkeys(
                re.findall(
                    r"FileNotFoundError: \[Errno 2\] No such file or directory: "
                    r"['\"]([^'\"]+)['\"]",
                    blob,
                )
            )
        )
        if missing:
            primary = missing[0]
            later = missing[1:]
            detail = ""
            if "kdestroy" in later:
                detail = (
                    " The later missing `kdestroy` occurred during cleanup and is "
                    "secondary; guarding cleanup can remove that traceback but cannot "
                    "fix the primary dependency."
                )
            elif later:
                names = ", ".join(f"`{name}`" for name in later)
                detail = f" Additional missing executables: {names}."
            reach = "the external program"
            if primary in {"kinit", "klist", "kdestroy"}:
                reach = "Kerberos authentication or remote publish"
            checks = " ".join(f"command -v {name};" for name in missing).rstrip(";")
            return (
                f"The runtime is missing required external executable `{primary}`; "
                f"this attempt did not reach {reach}.{detail} Stop retrying or changing "
                "credentials. Verify the same runtime/container PATH with "
                f"`{checks}`, then install or provide the required client there."
            )
    if "Not enough '\\' characters in service" in blob or "Usage: smbclient" in blob:
        return (
            "The client rejected its command-line syntax before authentication. "
            "Fix the service/option quoting using the installed client's --help; "
            "do not infer a credential failure from this attempt."
        )
    if re.search(
        r"(?im)^\s*(?:QUARANTINED:|Quarantined runs:\s*[1-9]\d*)",
        blob,
    ):
        return (
            "The application completed processing and deliberately quarantined "
            "the input; this is an application validation outcome, not an execute "
            "transport failure. Do not rerun unchanged. Inspect the quarantine "
            "manifest and logs for the validation errors/warnings."
        )
    return ""


def _redirect_targets(command: str) -> list[str]:
    """Extract explicit shell redirection targets without reading them."""
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return []
    targets: list[str] = []
    redirect = re.compile(r"^(?:[012]?>{1,2}|&>)")
    for index, token in enumerate(tokens):
        match = redirect.match(token)
        if not match:
            continue
        target = token[match.end() :]
        if not target and index + 1 < len(tokens):
            target = tokens[index + 1]
        if target and not target.startswith("&") and target not in targets:
            targets.append(target)
    return targets


def _command_failure_hint(
    command: str,
    stdout: str,
    stderr: str,
    warning_prefix: str,
) -> str:
    """Explain shell structure that otherwise makes an empty failure opaque."""
    hints: list[str] = []
    if "bash_validator: WARN" in warning_prefix:
        hints.append(
            "The bash-validator warning was advisory and did not cause the "
            "nonzero exit; execution proceeded."
        )
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        tokens = []
    if "&&" in tokens:
        hints.append(
            "This was an `&&` chain: the shell stopped at the first failing "
            "stage, so any later stages after it did not run."
        )
    if "for" in tokens and "do" in tokens and "done" in tokens:
        hints.append(
            "A shell `for` loop normally continues after a failed iteration; "
            "guard each producer/consumer pair explicitly so one failure does not "
            "generate cascading errors while other items may still succeed."
        )
    redirects = _redirect_targets(command)
    if redirects and not stdout.strip() and not stderr.strip():
        rendered = ", ".join(f"`{target}`" for target in redirects)
        hints.append(
            "Captured stdout/stderr are empty; useful output may have been "
            f"redirected to {rendered}. Inspect a bounded, non-sensitive tail "
            "or rerun the stages separately without redirection to identify the "
            "failing stage."
        )
    return " ".join(hints)


def _format_nonzero_command_output(
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    warning_prefix: str,
) -> str:
    interpretation = (
        "The command ran and exited nonzero. Treat stdout/stderr as "
        "diagnostic feedback, not as a tool transport failure."
    )
    if _git_not_a_repo_signal(command, exit_code, stdout, stderr):
        interpretation = (
            "Git exited 128 because this working directory is not a git "
            "repository. Do not retry git here. Run syntax/tests in a "
            "separate execute call without chaining `&& git …` "
            "(e.g. `node --check file.js` alone). Prefer snapshot_diff to "
            "review edits when git is unavailable."
        )
    hint = _sandbox_write_hint(stdout, stderr)
    if hint:
        interpretation = f"{interpretation} {hint}"
    diagnostic = _failure_diagnostic_hint(stdout, stderr)
    if diagnostic:
        interpretation = f"{interpretation} {diagnostic}"
    command_hint = _command_failure_hint(
        command, stdout, stderr, warning_prefix
    )
    if command_hint:
        interpretation = f"{interpretation} {command_hint}"
    payload: dict[str, Any] = {
        "command_executed": True,
        "success": False,
        "exit_code": exit_code,
        "command": _truncate_exec_output(command),
        "stdout": _truncate_exec_output(stdout or ""),
        "stderr": _truncate_exec_output(stderr or ""),
        "interpretation": interpretation,
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


def _child_env() -> dict[str, str]:
    """Sanitized subprocess env — same floor as ``LocalBackend._sanitized_env``."""
    try:
        from clawagents.redact import is_secret_name
        from clawagents.sandbox.local import LocalBackend

        deny = LocalBackend._SENSITIVE_ENV_KEYS
        return {
            k: v
            for k, v in os.environ.items()
            if k not in deny and not is_secret_name(k)
        } | {"PAGER": "cat"}
    except Exception:
        return {**os.environ, "PAGER": "cat"}


def _drain_profile_warnings(sb: Any) -> str:
    """Return new profile warnings and clear them so they aren't re-emitted."""
    warns = getattr(sb, "profile_warnings", None)
    if not warns:
        return ""
    batch = list(warns)[-8:]
    try:
        warns.clear()
    except Exception:
        pass
    return "".join(f"[sandbox_profile: {w}]\n" for w in batch)


def _discard_exec_spills(result: Any) -> None:
    for value in (
        getattr(result, "stdout_path", None),
        getattr(result, "stderr_path", None),
    ):
        if value:
            try:
                os.unlink(value)
            except FileNotFoundError:
                pass


def _archive_exec_spills(
    result: Any,
    *,
    stdout: str,
    stderr: str,
    workspace: str,
    command: str,
) -> str:
    """Adopt complete spilled streams and return a compact retrieval header."""
    stdout_path = getattr(result, "stdout_path", None)
    stderr_path = getattr(result, "stderr_path", None)
    if not stdout_path and not stderr_path:
        return ""
    from clawagents.tool_output_artifacts import store_exec_artifact_from_spills

    tool_use_id = f"execute-{uuid.uuid4().hex[:16]}"
    try:
        artifact_id, _path, chars = store_exec_artifact_from_spills(
            tool_use_id=tool_use_id,
            stdout=stdout,
            stderr=stderr,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            workspace=workspace,
            extra_meta={"command": command[:1_000]},
        )
    except Exception as exc:
        _discard_exec_spills(result)
        return f"[warning: failed to archive complete command output: {exc}]\n"
    return (
        f"[Complete command output archived id={artifact_id}; {chars} chars. "
        f"Retrieve with retrieve_tool_result(id=\"{artifact_id}\").]\n"
    )


def _maybe_wrap_for_profile(sb: Any, command: str, *, cwd: str | None) -> str:
    """Apply seatbelt/bwrap wrap when ``sb`` is a ProfileBackend."""
    wrap = getattr(sb, "wrap_command", None)
    if callable(wrap):
        return str(wrap(command, cwd=cwd))
    return command


def _unsandboxed_backend(sb: Any) -> Any:
    """Prefer the inner LocalBackend when ``sb`` is a ProfileBackend."""
    inner = getattr(sb, "_inner", None)
    return inner if inner is not None else sb


def _resolve_block_until_ms(args: Dict[str, Any]) -> tuple[int, bool]:
    """Return ``(block_until_ms, immediate_background)``.

    ``block_until_ms`` aliases ``timeout``. ``0`` means immediate background.
    Negative / unparseable values fall back to the default timeout.
    """
    if "block_until_ms" in args and args.get("block_until_ms") is not None:
        try:
            raw = int(args.get("block_until_ms"))
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT_MS, False
        if raw == 0:
            return 0, True
        if raw < 0:
            return DEFAULT_TIMEOUT_MS, False
        return max(100, raw), False
    try:
        timeout_ms = int(args.get("timeout", DEFAULT_TIMEOUT_MS))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_MS, False
    if timeout_ms < 0:
        return DEFAULT_TIMEOUT_MS, False
    return max(100, timeout_ms), False


async def _exec_foreground_with_autobg(
    command: str,
    *,
    cwd: str,
    timeout_ms: int,
    mgr: Any,
    on_chunk: Any | None = None,
    streaming: bool = False,
) -> tuple[ExecResult, bool, Optional[str]]:
    """Run shell; on timeout adopt the process into ``mgr``.

    Returns ``(result, timed_out_backgrounded, job_id|None)``.
    """
    import signal

    env = _child_env()
    timeout_s = max(0.1, timeout_ms / 1000.0)
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )

    # Always pump incrementally, even when live events are disabled. This keeps
    # memory bounded for the non-streaming configuration too.
    from clawagents.utils.bounded_output import SpoolingTextAccumulator

    total = 0
    out_acc = SpoolingTextAccumulator(MAX_OUTPUT_CHARS)
    err_acc = SpoolingTextAccumulator(MAX_OUTPUT_CHARS)

    async def _pump(stream: Any, acc: SpoolingTextAccumulator) -> str:
        nonlocal total
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            total += len(text)
            acc.append(text)
            if streaming and on_chunk is not None:
                try:
                    # Bound event payload size for hosts that forward progress live.
                    on_chunk(text[:2000], total)
                except Exception:
                    pass
        return str(acc)

    out_t = asyncio.create_task(_pump(proc.stdout, out_acc))
    err_t = asyncio.create_task(_pump(proc.stderr, err_acc))
    wait_t = asyncio.create_task(proc.wait())

    async def _finish_comm() -> tuple[bytes, bytes]:
        out_s = await out_t
        err_s = await err_t
        if not wait_t.done():
            await wait_t
        for path in (out_acc.close(), err_acc.close()):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        return out_s.encode("utf-8", errors="replace"), err_s.encode(
            "utf-8", errors="replace"
        )

    try:
        await asyncio.wait_for(asyncio.shield(wait_t), timeout=timeout_s)
        stdout = await out_t
        stderr = await err_t
        return (
            ExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
                stdout_path=out_acc.close(),
                stderr_path=err_acc.close(),
            ),
            False,
            None,
        )
    except asyncio.TimeoutError:
        argv = _shell_argv(command)
        comm = asyncio.create_task(_finish_comm())
        job = await mgr.adopt(proc, argv, cwd=cwd, communicate_task=comm)
        return (ExecResult(stdout="", stderr="", exit_code=0), True, job.id)
    except (asyncio.CancelledError, Exception):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        for t in (out_t, err_t, wait_t):
            if not t.done():
                t.cancel()
        for path in (out_acc.close(), err_acc.close()):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        raise


class _ToolProgressEmitter:
    """Coalesce command deltas into bounded updates emitted at most every 100ms."""

    def __init__(self, run_context: Any, interval_s: float = 0.1) -> None:
        try:
            from clawagents.config.features import is_enabled

            enabled = is_enabled("execute_streaming")
        except Exception:
            enabled = False
        callback = getattr(run_context, "on_event", None) if run_context else None
        self._callback = callback if enabled and callable(callback) else None
        metadata = getattr(run_context, "_metadata", None) if run_context else None
        typed_callback = (
            metadata.get("_emit_typed_event") if isinstance(metadata, dict) else None
        )
        self._typed_callback = typed_callback if callable(typed_callback) else None
        self._interval_s = interval_s
        self._last_emit = 0.0
        self._pending = ""
        self._total = 0
        self._stream = "stdout"
        self._handle: Any | None = None

    def feed(self, stream: str, delta: str) -> None:
        if self._callback is None or not delta:
            return
        self._total += len(delta)
        self._stream = stream
        self._pending = (self._pending + delta)[-4_000:]
        elapsed = time.monotonic() - self._last_emit
        if elapsed >= self._interval_s:
            self.flush()
            return
        if self._handle is None:
            try:
                loop = asyncio.get_running_loop()
                self._handle = loop.call_later(
                    self._interval_s - elapsed, self._scheduled_flush
                )
            except RuntimeError:
                pass

    def _scheduled_flush(self) -> None:
        self._handle = None
        self.flush()

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if self._callback is None or not self._pending:
            return
        delta = self._pending[-2_000:]
        self._pending = ""
        self._last_emit = time.monotonic()
        data = {
            "tool_name": "execute",
            "stream": self._stream,
            "delta": delta,
            "total_bytes": self._total,
        }
        try:
            self._callback("tool_progress", data)
        except Exception:
            pass
        if self._typed_callback is not None:
            try:
                self._typed_callback("tool_progress", data)
            except Exception:
                pass


class ExecTool:
    name = "execute"
    keywords = ["shell", "bash", "command", "run script", "terminal"]
    description = (
        "Execute a non-interactive shell command and return its output. "
        "When ctx_execute or ctx_batch_execute is available, prefer those for "
        "broad reads or output that needs filtering; use execute for bounded "
        "commands, mutations, builds, and tests. "
        "Working directory and (when enabled) env exports persist across calls "
        "in this session. Noisy commands (pytest, git status/log/diff, ls, rg, …) "
        "may be auto-wrapped with rtk when installed. "
        "Do not chain git with other checks via `&&` when the workspace may not "
        "be a git repo — run `node --check` / tests alone, and prefer git_status / "
        "git_diff tools (they report clearly when there is no .git). "
        "Use block_until_ms (alias of timeout) for the foreground wait; "
        "block_until_ms=0 or is_background=true returns a job_id immediately. "
        "Foreground deadlines may auto-background — use task_status / task_output / "
        "task_stop. Not for interactive TTY apps (vim, ssh prompts, REPLs that need "
        "a screen) — use pty_start / pty_keys / pty_wait / pty_screen / pty_stop."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "command": {"type": "string", "description": "The shell command to execute", "required": True},
        "timeout": {"type": "number", "description": f"Timeout in milliseconds. Default: {DEFAULT_TIMEOUT_MS}"},
        "block_until_ms": {
            "type": "number",
            "description": (
                "Foreground wait budget in ms (alias of timeout). "
                "0 = immediate background when execute_background is on. "
                f"Default: timeout / {DEFAULT_TIMEOUT_MS}."
            ),
        },
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
        "unsandboxed": {
            "type": "boolean",
            "description": (
                "Skip OS sandbox wrap (seatbelt/bwrap) for this command. "
                "Only honored when chat mode is full_access with Allow Full "
                "Access. Use after a sandbox EPERM on home-config CLIs "
                "(gcloud/aws/docker)."
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
        timeout_ms, immediate_bg = _resolve_block_until_ms(args)

        if not command:
            return ToolResult(
                success=False,
                output="No command provided",
                error="No command provided",
            )

        permission_mode = getattr(run_context, "permission_mode", PermissionMode.DEFAULT)
        is_background = _truthy(args.get("is_background")) or immediate_bg
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
        sticky_env = is_enabled("execute_shell_env")
        if is_enabled("execute_shell_session") and is_local_sb:
            session = session_for(run_context, sb)
            run_cwd = session.cwd
            # Explicit background: use session cwd/env start point but do not
            # inject PWD/ENV trailers — those would never be consumed here and
            # would leave later foreground calls with a stale session.
            if not is_background:
                command = session.wrap(command, sticky_env=sticky_env)
            env_n = len(session.env) if sticky_env else 0
            warning_prefix += f"[shell_session: cwd={run_cwd}"
            if sticky_env:
                warning_prefix += f" env_keys={env_n}"
            warning_prefix += "]\n"

        if is_background:
            if not is_enabled("execute_background"):
                return ToolResult(
                    success=False,
                    output="",
                    error="is_background requires CLAW_FEATURE_EXECUTE_BACKGROUND=1",
                )

            # Match foreground isolation: wrap for seatbelt/bwrap + scrub env
            # unless Full access authorized unsandboxed=true.
            try:
                if _may_run_unsandboxed(run_context, args):
                    bg_command = command
                    warning_prefix += "[sandbox: off for this command (unsandboxed)]\n"
                else:
                    bg_command = _maybe_wrap_for_profile(sb, command, cwd=run_cwd)
            except Exception as e:
                return ToolResult(
                    success=False, output="", error=f"Background sandbox wrap failed: {e}"
                )
            warning_prefix += _drain_profile_warnings(sb)
            mgr = _bg_manager(run_context)
            with tool_span("exec.background", command=bg_command):
                try:
                    job = await mgr.start(
                        _shell_argv(bg_command),
                        cwd=run_cwd,
                        env=_child_env(),
                    )
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
            on_chunk = None
            streaming = is_enabled("execute_streaming")
            progress = _ToolProgressEmitter(run_context)
            if streaming:

                def on_chunk(delta: str, total_bytes: int) -> None:
                    _ = total_bytes
                    progress.feed("combined", delta)

            with tool_span("exec.run", command=command, timeout_ms=timeout_ms):
                try:
                    result, bgd, job_id = await _exec_foreground_with_autobg(
                        command,
                        cwd=run_cwd,
                        timeout_ms=timeout_ms,
                        mgr=mgr,
                        on_chunk=on_chunk,
                        streaming=streaming,
                    )
                except Exception as e:
                    progress.flush()
                    return ToolResult(
                        success=False, output="", error=f"Command failed: {str(e)}"
                    )

            if bgd and job_id:
                progress.flush()
                # Trailers were already injected via session.wrap — sync cwd/env
                # when the adopted job finishes so later FG execute is not stale.
                if session is not None:
                    sess = session
                    sticky = sticky_env

                    def _sync_session_from_job(job: Any) -> None:
                        try:
                            sess.consume_stdout(job.stdout or "", sticky_env=sticky)
                        except Exception:
                            pass

                    try:
                        job_rec = mgr.status(job_id)
                        # Re-adopt path already started watcher; attach via
                        # a one-shot waiter so we don't require adopt() API change.
                        async def _wait_and_sync() -> None:
                            try:
                                done = await mgr.await_complete(job_id)
                                _sync_session_from_job(done)
                            except Exception:
                                pass

                        asyncio.create_task(
                            _wait_and_sync(), name=f"shell-session-sync-{job_id}"
                        )
                        _ = job_rec
                    except Exception:
                        pass
                payload = {
                    "backgrounded": True,
                    "auto_background_on_timeout": True,
                    "job_id": job_id,
                    "timeout_ms": timeout_ms,
                    "block_until_ms": timeout_ms,
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

            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.exit_code

            if session is not None:
                stdout_path = getattr(result, "stdout_path", None)
                if stdout_path:
                    session.consume_stdout_file(stdout_path, sticky_env=sticky_env)
                stdout = session.consume_stdout(stdout, sticky_env=sticky_env)

            progress.flush()
            archive_header = _archive_exec_spills(
                result,
                stdout=stdout,
                stderr=stderr,
                workspace=run_cwd,
                command=str(args.get("command", command)),
            )

            success = exit_code == 0
            if not success:
                if _is_empty_search_result(
                    str(args.get("command", command)),
                    exit_code,
                    stdout or "",
                    stderr or "",
                ):
                    return ToolResult(
                        success=True,
                        output=warning_prefix + archive_header + "Search completed: no matches found.",
                    )
                return ToolResult(
                    success=False,
                    output=archive_header + _format_nonzero_command_output(
                        str(args.get("command", command)),
                        exit_code,
                        stdout or "",
                        stderr or "",
                        warning_prefix,
                    ),
                    error=_short_exit_error(exit_code),
                )

            output = stdout or ""
            if stderr:
                output += ("\n" if output else "") + f"[stderr] {stderr}"
            output = _truncate_exec_output(output)
            if session is not None:
                warning_prefix += f"[shell_session: cwd now {session.cwd}]\n"
            return ToolResult(
                success=True,
                output=warning_prefix + archive_header + (output or "(no output)"),
            )

        # Legacy sandbox path (kill-on-timeout). Profile backends land here
        # because kind is ``profile:*:local`` — keep that so seatbelt/bwrap run.
        exec_sb = sb
        if _may_run_unsandboxed(run_context, args):
            exec_sb = _unsandboxed_backend(sb)
            warning_prefix += "[sandbox: off for this command (unsandboxed)]\n"

        progress = _ToolProgressEmitter(run_context)

        async def _run_once(target: Any):
            try:
                return await target.exec(
                    command,
                    timeout=timeout_ms,
                    cwd=run_cwd,
                    max_output_chars=MAX_OUTPUT_CHARS,
                    on_output=progress.feed,
                )
            except TypeError as e:
                msg = str(e).lower()
                if "unexpected keyword" not in msg and "cwd" not in msg and "max_output" not in msg:
                    raise
                try:
                    return await target.exec(
                        command,
                        timeout=timeout_ms,
                        cwd=run_cwd,
                    )
                except TypeError:
                    return await target.exec(command, timeout=timeout_ms)

        with tool_span("exec.run", command=command, timeout_ms=timeout_ms):
            try:
                result = await _run_once(exec_sb)
            except Exception as e:
                return ToolResult(
                    success=False, output="", error=f"Command failed: {e}"
                )

            # Soft sandbox fallback is otherwise invisible to the agent.
            warning_prefix += _drain_profile_warnings(sb)

            if result.killed:
                stdout = result.stdout or ""
                stderr = getattr(result, "stderr", None) or ""
                if session is not None:
                    stdout_path = getattr(result, "stdout_path", None)
                    if stdout_path:
                        session.consume_stdout_file(stdout_path, sticky_env=sticky_env)
                    stdout = session.consume_stdout(stdout, sticky_env=sticky_env)
                archive_header = _archive_exec_spills(
                    result,
                    stdout=stdout,
                    stderr=stderr,
                    workspace=run_cwd,
                    command=str(args.get("command", command)),
                )
                progress.flush()
                return ToolResult(
                    success=False,
                    output=archive_header + _truncate_exec_output(stdout + stderr),
                    error=(
                        f"Command timed out after {timeout_ms}ms: "
                        f"{args.get('command', command)}"
                    ),
                )

            stdout = result.stdout or ""
            stderr = getattr(result, "stderr", None) or ""
            # Escalation ladder: sandbox EPERM → one automatic unsandboxed
            # retry when Full access authorized this turn.
            meta = getattr(run_context, "_metadata", None) if run_context else None
            can_escalate = (
                isinstance(meta, dict)
                and bool(meta.get("allow_unsandboxed_exec"))
                and exec_sb is sb
                and result.exit_code != 0
                and _sandbox_eperm_signal(stdout, stderr)
            )
            if can_escalate:
                warning_prefix += (
                    "[sandbox: EPERM — auto-retrying once without OS sandbox]\n"
                )
                _discard_exec_spills(result)
                try:
                    result = await _run_once(_unsandboxed_backend(sb))
                    stdout = result.stdout or ""
                    stderr = getattr(result, "stderr", None) or ""
                except Exception as e:
                    return ToolResult(
                        success=False,
                        output=warning_prefix,
                        error=f"Unsandboxed retry failed: {e}",
                    )

            if session is not None:
                stdout_path = getattr(result, "stdout_path", None)
                if stdout_path:
                    session.consume_stdout_file(stdout_path, sticky_env=sticky_env)
                stdout = session.consume_stdout(stdout, sticky_env=sticky_env)

            archive_header = _archive_exec_spills(
                result,
                stdout=stdout,
                stderr=stderr,
                workspace=run_cwd,
                command=str(args.get("command", command)),
            )
            progress.flush()

            success = result.exit_code == 0
            if not success:
                if _is_empty_search_result(
                    str(args.get("command", command)),
                    result.exit_code,
                    stdout,
                    stderr,
                ):
                    return ToolResult(
                        success=True,
                        output=warning_prefix + archive_header + "Search completed: no matches found.",
                    )
                return ToolResult(
                    success=False,
                    output=archive_header + _format_nonzero_command_output(
                        str(args.get("command", command)),
                        result.exit_code,
                        stdout,
                        stderr,
                        warning_prefix,
                    ),
                    error=_short_exit_error(result.exit_code),
                )

            output = stdout
            if result.stderr:
                output += ("\n" if output else "") + f"[stderr] {result.stderr}"
            output = _truncate_exec_output(output)
            if session is not None:
                warning_prefix += f"[shell_session: cwd now {session.cwd}]\n"
            return ToolResult(
                success=True,
                output=warning_prefix + archive_header + (output or "(no output)"),
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
