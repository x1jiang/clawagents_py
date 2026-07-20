"""Display-layer secret redaction for ClawAgents.

This module strips API-key-like patterns from text before it reaches:

* the terminal (CLI streaming output, banners, error traces),
* gateway / channel chats (Telegram, Signal, WhatsApp, gateway WS),
* trajectory NDJSON files,
* diagnostic and debug logs.

Inspired by ``hermes-agent``'s ``agent/redact.py``. Redaction is applied at the
**display / persistence layer only** — underlying values still flow through the
agent loop unchanged so tools that legitimately need a secret (e.g. an
authenticated API call) keep working.

Redaction is **enabled by default**. Operators may opt out per-process via the
``CLAW_REDACT`` environment variable:

* ``CLAW_REDACT=0`` / ``false`` / ``no`` → redaction disabled.
* ``CLAW_REDACT=warn`` → redaction disabled but a single warning is logged the
  first time a secret-like pattern would have been redacted.

Custom patterns can be added at runtime via :func:`add_pattern`. Built-in
coverage:

* OpenAI ``sk-...`` / ``sk-proj-...`` keys
* Anthropic ``sk-ant-...`` keys
* Google AI / GCP ``AIza...`` keys
* GitHub ``ghp_`` / ``gho_`` / ``ghu_`` / ``ghs_`` / ``ghr_`` / ``github_pat_``
* AWS access keys (``AKIA``/``ASIA``)
* JWT-shaped tokens (``eyJ...``)
* Bearer tokens, generic ``api_key=``/``password=`` assignments
* Slack ``xoxa-`` / ``xoxb-`` / ``xoxp-`` / ``xoxr-`` / ``xoxs-`` tokens

Performance: the patterns are compiled once and applied as a single ``sub``
pass per call; for short strings (~hundreds of chars) the cost is well below
1 µs. Long strings are still safe — patterns are anchored on
``\\b``/non-word boundaries so they don't backtrack.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


# ─── Built-in patterns ────────────────────────────────────────────────────


def _compile(label: str, pattern: str) -> tuple[str, re.Pattern[str]]:
    return (label, re.compile(pattern))


_BUILTIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # OpenAI / OpenAI Project keys
    _compile("OPENAI_KEY", r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"),
    # Anthropic keys
    _compile("ANTHROPIC_KEY", r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    # Google AI / GCP API keys
    _compile("GOOGLE_KEY", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    # GitHub PAT / fine-grained tokens / OAuth tokens
    _compile(
        "GITHUB_TOKEN",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
    ),
    # AWS access key IDs
    _compile("AWS_ACCESS_KEY_ID", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # AWS secret access keys (40 chars, base64-ish, paired with KEY/SECRET name)
    _compile(
        "AWS_SECRET_KEY",
        r"(?i)aws[_-]?(?:secret[_-]?access[_-]?key|secret)\s*[:=]\s*['\"]?"
        r"([A-Za-z0-9/+=]{40})['\"]?",
    ),
    # Slack tokens
    _compile("SLACK_TOKEN", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    # Three-segment JWTs (header.payload.signature)
    _compile(
        "JWT",
        r"\beyJ[A-Za-z0-9_=\-]{4,}\.[A-Za-z0-9_=\-]{4,}\.[A-Za-z0-9_.+/=\-]{4,}\b",
    ),
    # PEM private-key blocks (RSA, EC, DSA, OpenSSH, generic).
    _compile(
        "PRIVATE_KEY_PEM",
        r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY-----"
        r"[\s\S]*?-----END (?:RSA |DSA |EC |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY-----",
    ),
    # ``Authorization: Bearer <token>`` and ``Authorization: Basic <token>``.
    # Also catches ``X-Api-Key: <token>`` style header lines.
    _compile(
        "BEARER",
        r"(?i)(?:authorization|proxy-authorization|x[_-]?api[_-]?key)\s*[:=]\s*"
        r"(?:bearer\s+|basic\s+|token\s+)?['\"]?[A-Za-z0-9_\-.~+/=]{16,}['\"]?",
    ),
    # URL basic-auth: ``https://user:pass@host``.
    _compile(
        "URL_BASIC_AUTH",
        r"\b(?:https?|ftp|ssh|git|mongodb|postgres(?:ql)?|mysql|redis|amqp|amqps)"
        r"://[^\s/@]+:[^\s/@]+@",
        # Note: anchored on the scheme so we don't flag random "user:pass@…" text.
    ),
    # Generic key=value assignments for known secret-looking names. The
    # alternative patterns are ordered most-specific first; the field
    # name is matched anywhere in the line (no \\b — so ``AWS_SECRET=…``
    # works even though ``_`` is a word character).
    _compile(
        "GENERIC_SECRET",
        r"(?i)(?:api[_-]?key|api[_-]?secret|client[_-]?secret|"
        r"access[_-]?token|refresh[_-]?token|session[_-]?token|"
        r"private[_-]?key|x[_-]?api[_-]?key|"
        r"password|passwd|pwd|secret|credential)"
        r"\s*[:=]\s*['\"]?([A-Za-z0-9_\-+/=.~]{6,})['\"]?",
    ),
    # A malformed shell interpolation can turn a password into a command name,
    # e.g. ``bash: line 1: vP7Vf5uipuaO: command not found``.  Restrict this to
    # mixed-case, digit-bearing, high-entropy-looking tokens so ordinary missing
    # commands such as ``smbclient`` remain useful diagnostic text.
    _compile(
        "SHELL_SECRET",
        r"(?m)(?<=: )"
        r"(?=[A-Za-z0-9_+./=\-]{10,}:(?: command)? not found$)"
        r"(?=[^:\n]*[A-Z])(?=[^:\n]*[a-z])(?=[^:\n]*\d)"
        r"[A-Za-z0-9_+./=\-]+(?=:(?: command)? not found$)",
    ),
]


_user_patterns: list[tuple[str, re.Pattern[str]]] = []
_warned_once = False


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _redact_mode() -> str:
    """Return one of ``"on"``, ``"warn"``, ``"off"`` based on env."""
    raw = (os.environ.get("CLAW_REDACT") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"warn", "warning"}:
        return "warn"
    return "on"


def add_pattern(label: str, pattern: str | re.Pattern[str]) -> None:
    """Register a custom regex pattern. Matched substrings are replaced with
    ``[REDACTED:<label>]`` on every subsequent :func:`redact` call.

    ``pattern`` is compiled if a string is given. The label should be a short
    identifier like ``"INTERNAL_TOKEN"``.
    """
    compiled = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
    _user_patterns.append((label, compiled))


def reset_patterns() -> None:
    """Drop all user-registered patterns. Built-in patterns are unaffected."""
    _user_patterns.clear()


# ─── Public API ───────────────────────────────────────────────────────────


def redact(text: str | None, *, label: bool = True) -> str:
    """Return ``text`` with API-key-like substrings replaced.

    If ``label`` is True (default), each match is replaced with
    ``[REDACTED:<KIND>]``; if False, with a fixed ``[REDACTED]``.

    Returns ``""`` for ``None`` to keep callsites simple.
    """
    if text is None:
        return ""
    if not isinstance(text, str):  # pragma: no cover — defensive
        text = str(text)
    if not text:
        return text

    mode = _redact_mode()
    if mode == "off":
        return text

    out = text
    matched_any = False
    for kind, pat in _BUILTIN_PATTERNS + _user_patterns:
        replacement = f"[REDACTED:{kind}]" if label else "[REDACTED]"
        new_out = pat.sub(replacement, out)
        if new_out != out:
            matched_any = True
            out = new_out

    if matched_any and mode == "warn":
        global _warned_once
        if not _warned_once:
            logger.warning(
                "clawagents.redact: detected secret-like content but CLAW_REDACT=warn "
                "(redaction disabled). Set CLAW_REDACT=1 to enable."
            )
            _warned_once = True
        return text  # warn-only mode: leave text untouched

    return out


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings inside dicts / lists / tuples / sets.

    Non-string scalars (int, float, bool, None) pass through unchanged. The
    structure shape is preserved (sets stay sets, tuples stay tuples).
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, Mapping):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(redact_obj(v) for v in obj)
    if isinstance(obj, set):
        return {redact_obj(v) for v in obj}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj


# Names whose values should always be replaced when redacting environment
# dictionaries / config dumps, even if the value itself doesn't match a known
# secret pattern. Keep this conservative — false positives only mean a config
# value gets ``***`` in a log line.
_SECRET_NAME_HINTS: tuple[str, ...] = (
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "pwd",
    "token",
    "auth",
    "credential",
    "private_key",
    "access_key",
    "session_key",
    "bearer",
)


def is_secret_name(name: str) -> bool:
    """Return True if ``name`` looks like an env var or config key holding a
    secret, e.g. ``"OPENAI_API_KEY"`` or ``"db_password"``."""
    if not name:
        return False
    lower = name.lower()
    return any(hint in lower for hint in _SECRET_NAME_HINTS)


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` with values for secret-named keys masked.

    The mask is a fixed ``[REDACTED]`` regardless of whether the value looks
    like a recognized provider key; both presence and length are hidden.
    """
    out: dict[str, str] = {}
    for k, v in env.items():
        if is_secret_name(k):
            out[k] = "[REDACTED]"
        else:
            out[k] = redact(v) if isinstance(v, str) else v
    return out


# ─── Convenience: filter for the stdlib ``logging`` module ────────────────


class RedactingLogFilter(logging.Filter):
    """A ``logging.Filter`` that redacts ``record.msg`` (after formatting).

    Install with::

        for handler in logging.getLogger().handlers:
            handler.addFilter(RedactingLogFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Render ``msg % args`` once so we don't try to redact the format
            # string ahead of arg substitution.
            rendered = record.getMessage()
            redacted = redact(rendered)
            if redacted != rendered:
                record.msg = redacted
                record.args = None
        except Exception:  # pragma: no cover — logging must never crash
            pass
        return True


def install_logging_filter(logger_obj: logging.Logger | None = None) -> None:
    """Attach :class:`RedactingLogFilter` to every handler on ``logger_obj``
    (or the root logger if not given). Idempotent."""
    target = logger_obj or logging.getLogger()
    flt = RedactingLogFilter()
    for h in target.handlers:
        existing = [f for f in h.filters if isinstance(f, RedactingLogFilter)]
        if not existing:
            h.addFilter(flt)


__all__ = [
    "add_pattern",
    "redact",
    "redact_obj",
    "redact_env",
    "is_secret_name",
    "reset_patterns",
    "RedactingLogFilter",
    "install_logging_filter",
]
