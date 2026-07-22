"""Error taxonomy and recovery recipes for ClawAgents.

Classifies errors from LLM providers and tool execution into discrete
failure classes, each with a retryable flag, recovery hint, and optional
failover model suggestion.

Inspired by claw-code-main's error.rs taxonomy.
"""

from __future__ import annotations

import enum
import re
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
    # Note: no bare "max_tokens" here — it appears in unrelated validation
    # errors ("max_tokens must be at least 1"), which would trigger a useless
    # compact-and-retry loop instead of surfacing the real problem.
    if any(tok in msg for tok in (
        "context_length_exceeded", "context window", "token limit",
        "maximum context length", "too many tokens",
        "prompt is too long", "request too large",
        "context_window_exceeded", "exceeds the maximum number of tokens",
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
    # Mantle frontier mis-route (xAI Grok on …/v1 instead of …/openai/v1).
    if "berm is not enabled" in msg or (
        "access_denied" in msg and "berm" in msg
    ):
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_AUTH,
            retryable=False,
            recovery_hint=(
                "Mantle frontier model used the wrong base path "
                "(…/v1 instead of …/openai/v1). For xai.grok-4.3, ClawAgents "
                "should rewrite the Mantle URL automatically — upgrade/restart "
                "the sidecar if this persists."
            ),
            max_retries=0,
            original=err,
        )
    # Mantle Claude with plain Anthropic client (X-Api-Key) instead of Bearer.
    if "bedrock-mantle" in msg and any(
        tok in msg
        for tok in (
            "invalid x-api-key",
            "x-api-key",
            "authentication_error",
            "invalid api key",
        )
    ):
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_AUTH,
            retryable=False,
            recovery_hint=(
                "Mantle Claude needs Authorization Bearer (AsyncAnthropicBedrockMantle), "
                "not X-Api-Key. Upgrade clawagents>=6.20.45 and restart the sidecar."
            ),
            max_retries=0,
            original=err,
        )
    if status in (401, 403) or any(tok in msg for tok in (
        "unauthorized", "forbidden", "invalid api key", "invalid_api_key",
        "authentication", "invalid x-api-key", "permission denied",
        "incorrect api key", "invalid auth",
        # google-genai: 400 INVALID_ARGUMENT / API_KEY_INVALID
        "api key not valid", "api_key_invalid",
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

    # 4a. Claude Fable / Mythos — account must opt into provider_data_share.
    if "data retention mode" in msg and (
        "provider_data_share" in msg or "not available for this model" in msg
    ):
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_AUTH,
            retryable=False,
            recovery_hint=(
                "This Mantle model (e.g. Claude Fable 5) requires account data "
                "retention mode provider_data_share. Set it once via "
                "PUT …/v1/data_retention {\"mode\":\"provider_data_share\"} with your "
                "Bedrock API key (AWS docs). Not a ClawAgents bug."
            ),
            max_retries=0,
            original=err,
        )

    # 4b. Mantle / Bedrock model or path 404 (often region or wrong …/openai vs …/openai/v1)
    if status == 404 or "does not exist" in msg or "error code: 404" in msg:
        return ErrorDescriptor(
            error_class=ErrorClass.UNKNOWN,
            retryable=False,
            recovery_hint=(
                "HTTP 404 from the model host. For Mantle GPT-5.x / Grok, the "
                "base must be …/openai/v1 (Responses). GPT-5.6 Sol is only in "
                "us-east-1 and us-east-2 — switch AWS region if you are on "
                "us-west-2. Chat-only Mantle models (gpt-oss, DeepSeek) stay on …/v1."
            ),
            max_retries=0,
            original=err,
        )

    # 5. Provider internal (5xx)
    if status and 500 <= status <= 504:
        recipe = RECOVERY_RECIPES[ErrorClass.PROVIDER_INTERNAL]
        return ErrorDescriptor(
            error_class=ErrorClass.PROVIDER_INTERNAL,
            retryable=recipe.retryable,
            recovery_hint=recipe.recovery_hint,
            max_retries=recipe.max_retries,
            original=err,
        )

    # 6. Transport / network
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

    # 7. Runtime I/O
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

    # 8. Unknown
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


def _coerce_status(value: object) -> int | None:
    """Normalize a status attribute to an int HTTP code, if it is one.

    google-genai's ClientError exposes ``status`` as a string enum name
    ("INVALID_ARGUMENT"); other SDKs use ints or numeric strings.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _extract_status(err: BaseException) -> int | None:
    """Extract HTTP status code from exception (provider-agnostic)."""
    # OpenAI SDK
    status = _coerce_status(getattr(err, "status_code", None))
    if status is not None:
        return status
    # google-genai (code) / Anthropic SDK (status)
    status = _coerce_status(getattr(err, "code", None))
    if status is not None:
        return status
    status = _coerce_status(getattr(err, "status", None))
    if status is not None:
        return status
    # Generic HTTP
    if hasattr(err, "response") and hasattr(err.response, "status_code"):
        status = _coerce_status(err.response.status_code)
        if status is not None:
            return status
    # String-based fallback. Word-boundary match so "429" doesn't fire on
    # "1429 tokens", "file_429.txt", or a request id containing the digits.
    msg = str(err)
    for code in (401, 403, 429, 500, 502, 503, 504):
        if re.search(rf"(?<!\d){code}(?!\d)", msg):
            return code
    return None
