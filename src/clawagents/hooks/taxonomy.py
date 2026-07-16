"""Expanded hook taxonomy + HTTPS webhook runners with SSRF guard.

Grok Build parity (xai-grok-hooks): 14 events, exit-2/JSON deny for PreToolUse,
Claude-hooks name aliases, HTTPS-only webhooks with private-IP blocklist.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from urllib.parse import urlparse


DENY_EXIT_CODE = 2


class HookEvent(str, Enum):
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    STOP = "Stop"
    STOP_FAILURE = "StopFailure"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    PERMISSION_DENIED = "PermissionDenied"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    NOTIFICATION = "Notification"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"


# Claude / wire aliases → HookEvent
_EVENT_ALIASES: dict[str, HookEvent] = {
    "sessionstart": HookEvent.SESSION_START,
    "session_start": HookEvent.SESSION_START,
    "sessionend": HookEvent.SESSION_END,
    "session_end": HookEvent.SESSION_END,
    "stop": HookEvent.STOP,
    "stopfailure": HookEvent.STOP_FAILURE,
    "stop_failure": HookEvent.STOP_FAILURE,
    "pretooluse": HookEvent.PRE_TOOL_USE,
    "pre_tool_use": HookEvent.PRE_TOOL_USE,
    "beforeshellexecution": HookEvent.PRE_TOOL_USE,
    "beforetoolcall": HookEvent.PRE_TOOL_USE,
    "posttooluse": HookEvent.POST_TOOL_USE,
    "post_tool_use": HookEvent.POST_TOOL_USE,
    "aftertoolcall": HookEvent.POST_TOOL_USE,
    "posttoolusefailure": HookEvent.POST_TOOL_USE_FAILURE,
    "permissiondenied": HookEvent.PERMISSION_DENIED,
    "userpromptsubmit": HookEvent.USER_PROMPT_SUBMIT,
    "notification": HookEvent.NOTIFICATION,
    "subagentstart": HookEvent.SUBAGENT_START,
    "subagentstop": HookEvent.SUBAGENT_STOP,
    "subagentend": HookEvent.SUBAGENT_STOP,
    "precompact": HookEvent.PRE_COMPACT,
    "postcompact": HookEvent.POST_COMPACT,
}


def normalize_event(name: str) -> HookEvent | None:
    key = (name or "").strip()
    if not key:
        return None
    try:
        return HookEvent(key)
    except ValueError:
        return _EVENT_ALIASES.get(key.lower().replace("-", ""))


@dataclass
class HookDecision:
    allowed: bool
    reason: str = ""
    source: str = ""


@dataclass
class HookHandler:
    event: HookEvent
    command: list[str] | None = None
    url: str | None = None
    timeout_s: float = 10.0
    matcher: str | None = None  # optional tool-name glob


def is_blocked_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if addr.is_loopback:
        return False  # loopback allowed (local sidecars)
    if (
        addr.is_private
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        return True
    # CGNAT 100.64/10
    if isinstance(addr, ipaddress.IPv4Address):
        if ipaddress.IPv4Address("100.64.0.0") <= addr <= ipaddress.IPv4Address("100.127.255.255"):
            return True
        if str(addr).startswith("169.254."):
            return True
    return False


def validate_hook_url(url: str) -> tuple[bool, str]:
    """HTTPS-only + SSRF blocklist. Returns (ok, reason)."""
    try:
        parsed = urlparse(url)
    except Exception as exc:  # noqa: BLE001
        return False, f"bad_url:{exc}"
    if parsed.scheme != "https":
        return False, "https_only"
    host = parsed.hostname
    if not host:
        return False, "missing_host"
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return False, f"dns:{exc}"
    for info in infos:
        ip = info[4][0]
        if is_blocked_ip(ip):
            return False, f"ssrf_blocked:{ip}"
    return True, "ok"


def parse_blocking_result(stdout: str, exit_code: int) -> HookDecision:
    """JSON decision wins; else exit 2 = deny, 0 = allow, else fail-open."""
    text = (stdout or "").strip()
    if text:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "decision" in data:
                dec = str(data.get("decision") or "").lower()
                reason = str(data.get("reason") or "")
                if dec == "deny":
                    return HookDecision(allowed=False, reason=reason, source="json")
                if dec == "allow":
                    return HookDecision(allowed=True, reason=reason, source="json")
        except json.JSONDecodeError:
            pass
    if exit_code == DENY_EXIT_CODE:
        return HookDecision(allowed=False, reason="exit_2", source="exit")
    if exit_code == 0:
        return HookDecision(allowed=True, reason="", source="exit")
    # fail-open
    return HookDecision(allowed=True, reason=f"fail_open_exit_{exit_code}", source="fail_open")


def _run_command(handler: HookHandler, payload: dict[str, Any]) -> HookDecision:
    if not handler.command:
        return HookDecision(allowed=True, reason="no_command", source="skip")
    try:
        proc = subprocess.run(
            handler.command,
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            timeout=max(0.5, handler.timeout_s),
            check=False,
        )
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        return parse_blocking_result(out, int(proc.returncode))
    except Exception as exc:  # noqa: BLE001
        return HookDecision(allowed=True, reason=f"crash:{exc}", source="fail_open")


def _run_webhook(handler: HookHandler, payload: dict[str, Any]) -> HookDecision:
    if not handler.url:
        return HookDecision(allowed=True, reason="no_url", source="skip")
    ok, reason = validate_hook_url(handler.url)
    if not ok:
        # SSRF block = fail-open (do not deny the tool)
        return HookDecision(allowed=True, reason=reason, source="ssrf_fail_open")
    data = json.dumps({"event": payload.get("event"), "payload": payload}).encode("utf-8")
    req = urllib.request.Request(
        handler.url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "clawagents-hooks/6.17"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(0.5, handler.timeout_s)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return parse_blocking_result(body, 0)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        if body.strip().startswith("{"):
            return parse_blocking_result(body, DENY_EXIT_CODE if exc.code >= 400 else 0)
        return HookDecision(allowed=True, reason=f"http_{exc.code}", source="fail_open")
    except Exception as exc:  # noqa: BLE001
        return HookDecision(allowed=True, reason=f"webhook_error:{exc}", source="fail_open")


@dataclass
class HookDispatcher:
    handlers: list[HookHandler] = field(default_factory=list)

    def add(self, handler: HookHandler) -> None:
        self.handlers.append(handler)

    def dispatch(
        self,
        event: HookEvent | str,
        payload: dict[str, Any] | None = None,
        *,
        blocking: bool | None = None,
    ) -> HookDecision:
        from clawagents.config.features import is_enabled

        if not is_enabled("hook_taxonomy"):
            return HookDecision(allowed=True, reason="feature_disabled")

        ev = event if isinstance(event, HookEvent) else normalize_event(str(event))
        if ev is None:
            return HookDecision(allowed=True, reason="unknown_event")
        body = dict(payload or {})
        body["event"] = ev.value
        is_blocking = blocking if blocking is not None else (ev == HookEvent.PRE_TOOL_USE)

        for handler in self.handlers:
            if handler.event != ev:
                continue
            if handler.matcher and body.get("tool"):
                from fnmatch import fnmatch

                if not fnmatch(str(body.get("tool")), handler.matcher):
                    continue
            if handler.url:
                decision = _run_webhook(handler, body)
            else:
                decision = _run_command(handler, body)
            if is_blocking and not decision.allowed:
                return decision  # first deny wins
        return HookDecision(allowed=True, reason="all_allow")


def load_handlers_from_config(raw: dict[str, Any] | list[Any]) -> list[HookHandler]:
    """Parse hooks.json style config into handlers."""
    rows = raw if isinstance(raw, list) else (raw.get("hooks") or raw.get("handlers") or [])
    out: list[HookHandler] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        ev = normalize_event(str(row.get("event") or row.get("type") or ""))
        if ev is None:
            continue
        cmd = row.get("command")
        if isinstance(cmd, str):
            command = ["bash", "-lc", cmd]
        elif isinstance(cmd, list):
            command = [str(x) for x in cmd]
        else:
            command = None
        out.append(
            HookHandler(
                event=ev,
                command=command,
                url=str(row["url"]) if row.get("url") else None,
                timeout_s=float(row.get("timeout_s") or row.get("timeout") or 10),
                matcher=str(row["matcher"]) if row.get("matcher") else None,
            )
        )
    return out


__all__ = [
    "DENY_EXIT_CODE",
    "HookEvent",
    "HookDecision",
    "HookHandler",
    "HookDispatcher",
    "normalize_event",
    "is_blocked_ip",
    "validate_hook_url",
    "parse_blocking_result",
    "load_handlers_from_config",
]
