"""Error taxonomy and recovery recipes for ClawAgents.

Classifies errors from LLM providers and tool execution into discrete
failure classes, each with a retryable flag, recovery hint, and optional
failover model suggestion.

Inspired by claw-code-main's error.rs taxonomy.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ErrorClass(str, enum.Enum):
    """Discrete error failure classes."""
    CONTEXT_WINDOW = "context_window"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_RETRY_EXHAUSTED = "provider_retry_exhausted"
    PROVIDER_INTERNAL = "provider_internal"
    PROVIDER_TRANSPORT = "provider_transport"
    RUNTIME_IO = "runtime_io"
    UNKNOWN = "unknown"


@dataclass
class ErrorDescriptor:
    """Structured error classification result."""
    error_class: ErrorClass
    retryable: bool
    recovery_hint: str
    max_retries: int = 3
    failover_model: str | None = None
    original: BaseException | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_class": self.error_class.value,
            "retryable": self.retryable,
            "recovery_hint": self.recovery_hint,
            "max_retries": self.max_retries,
            "failover_model": self.failover_model,
        }


@dataclass
class RecoveryRecipe:
    """Recovery strategy for an error class."""
    retryable: bool
    max_retries: int
    recovery_hint: str
    failover_model: str | None = None
    backoff_base_s: float = 1.0
    compact_on_retry: bool = False


# ─── Recovery Recipes ────────────────────────────────────────────────────

RECOVERY_RECIPES: dict[ErrorClass, RecoveryRecipe] = {
    ErrorClass.CONTEXT_WINDOW: RecoveryRecipe(
        retryable=True,
        max_retries=2,
        recovery_hint="Context window exceeded. Compacting messages and retrying with shorter context.",
        compact_on_retry=True,
    ),
    ErrorClass.PROVIDER_AUTH: RecoveryRecipe(
        retryable=False,
        max_retries=0,
        recovery_hint="Authentication failed. Check your API key is set correctly in .env (OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY).",
    ),
    ErrorClass.PROVIDER_RATE_LIMIT: RecoveryRecipe(
        retryable=True,
        max_retries=5,
        recovery_hint="Rate limited by provider. Backing off and retrying.",
        backoff_base_s=2.0,
    ),
    ErrorClass.PROVIDER_RETRY_EXHAUSTED: RecoveryRecipe(
        retryable=False,
        max_retries=0,
        recovery_hint="Max retries exhausted. The provider may be experiencing an outage. Try again later or switch models.",
        failover_model="gpt-5-nano",
    ),
    ErrorClass.PROVIDER_INTERNAL: RecoveryRecipe(
        retryable=True,
        max_retries=3,
        recovery_hint="Provider internal error (5xx). Retrying with backoff.",
        backoff_base_s=2.0,
    ),
    ErrorClass.PROVIDER_TRANSPORT: RecoveryRecipe(
        retryable=True,
        max_retries=3,
        recovery_hint="Network/transport error. Check your internet connection. Retrying.",
        backoff_base_s=1.0,
    ),
    ErrorClass.RUNTIME_IO: RecoveryRecipe(
        retryable=False,
        max_retries=0,
        recovery_hint="Local I/O error (file not found, permission denied, JSON decode failure).",
    ),
    ErrorClass.UNKNOWN: RecoveryRecipe(
        retryable=False,
        max_retries=0,
        recovery_hint="An unexpected error occurred.",
    ),
}


# ─── Classification ──────────────────────────────────────────────────────

def classify_error(err: BaseException, provider: str = "") -> ErrorDescriptor:
    """Classify an exception into a structured ErrorDescriptor.

    Accepts ``BaseException`` (not just ``Exception``) because asyncio task
    cancellation can surface as ``CancelledError`` which inherits from
    ``BaseException`` directly. Uses string-based inspection to avoid
    importing provider-specific SDK types at module level.
    """
    msg = str(err).lower()
    err_type = type(err).__name__.lower()

    # 1. Context window / token overflow
    if any(tok in msg for tok in (
        "context_length_exceeded", "context window", "token limit",
        "maximum context length", "too many tokens",
        "prompt is too long", "request too large",
        "max_tokens", "context_window_exceeded",
    )):
        recipe = RECOVERY_RECIPES[ErrorClass.CONTEXT_WINDOW]
        return ErrorDescriptor(
            error_class=ErrorClass.CONTEXT_WINDOW,
            retryable=recipe.retryable,
            recovery_hint=recipe.recovery_hint,
            max_retries=recipe.max_retries,
            original=err,
        )

    # 2. Auth errors
    status = _extract_status(err)
    if status in (401, 403) or any(tok in msg for tok in (
        "unauthorized", "forbidden", "invalid api key", "invalid_api_key",
        "authentication", "invalid x-api-key", "permission denied",
        "incorrect api key", "invalid auth",
    )):
        recipe = RECOVERY_RECIPES[ErrorClass.PROVIDER_AUTH]
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_AUTH,
            retryable=recipe.retryable,
            recovery_hint=recipe.recovery_hint,
            max_retries=recipe.max_retries,
            original=err,
        )

    # 3. Rate limit
    if status == 429 or any(tok in msg for tok in (
        "rate limit", "too many requests", "rate_limit_exceeded",
        "quota exceeded", "resource_exhausted",
    )):
        recipe = RECOVERY_RECIPES[ErrorClass.PROVIDER_RATE_LIMIT]
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_RATE_LIMIT,
            retryable=recipe.retryable,
            recovery_hint=recipe.recovery_hint,
            max_retries=recipe.max_retries,
            original=err,
        )

    # 4. Provider internal (5xx)
    if status and 500 <= status <= 504:
        recipe = RECOVERY_RECIPES[ErrorClass.PROVIDER_INTERNAL]
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_INTERNAL,
            retryable=recipe.retryable,
            recovery_hint=recipe.recovery_hint,
            max_retries=recipe.max_retries,
            original=err,
        )

    # 5. Transport / network
    if any(tok in msg for tok in (
        "econnreset", "connection", "timeout", "network",
        "socket hang up", "fetch failed", "stream stalled",
        "dns", "ssl", "tls",
    )) or any(tok in err_type for tok in (
        "connection", "timeout",
    )):
        recipe = RECOVERY_RECIPES[ErrorClass.PROVIDER_TRANSPORT]
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_TRANSPORT,
            retryable=recipe.retryable,
            recovery_hint=recipe.recovery_hint,
            max_retries=recipe.max_retries,
            original=err,
        )

    # 6. Runtime I/O
    if isinstance(err, (FileNotFoundError, PermissionError, IsADirectoryError, OSError)):
        recipe = RECOVERY_RECIPES[ErrorClass.RUNTIME_IO]
        return ErrorDescriptor(
            error_class=ErrorClass.RUNTIME_IO,
            retryable=recipe.retryable,
            recovery_hint=recipe.recovery_hint,
            max_retries=recipe.max_retries,
            original=err,
        )
    if any(tok in err_type for tok in ("json", "decode", "parse")):
        recipe = RECOVERY_RECIPES[ErrorClass.RUNTIME_IO]
        return ErrorDescriptor(
            error_class=ErrorClass.RUNTIME_IO,
            retryable=recipe.retryable,
            recovery_hint=f"JSON/parse error: {str(err)[:200]}",
            max_retries=recipe.max_retries,
            original=err,
        )

    # 7. Unknown
    recipe = RECOVERY_RECIPES[ErrorClass.UNKNOWN]
    return ErrorDescriptor(
        error_class=ErrorClass.UNKNOWN,
        retryable=False,
        recovery_hint=f"Unexpected error: {str(err)[:200]}",
        max_retries=0,
        original=err,
    )


def get_recovery_recipe(error_class: ErrorClass) -> RecoveryRecipe:
    """Get the recovery recipe for an error class."""
    return RECOVERY_RECIPES.get(error_class, RECOVERY_RECIPES[ErrorClass.UNKNOWN])


def _extract_status(err: BaseException) -> int | None:
    """Extract HTTP status code from exception (provider-agnostic)."""
    # OpenAI SDK
    if hasattr(err, "status_code"):
        return getattr(err, "status_code", None)
    # Anthropic SDK
    if hasattr(err, "status"):
        return getattr(err, "status", None)
    # Generic HTTP
    if hasattr(err, "response") and hasattr(err.response, "status_code"):
        return err.response.status_code
    # String-based fallback
    msg = str(err)
    for code in (401, 403, 429, 500, 502, 503, 504):
        if str(code) in msg:
            return code
    return None
