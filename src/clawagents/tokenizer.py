"""Accurate token counting with tiktoken (optional) and graceful fallback.

When tiktoken is installed, uses BPE encoding matched to the model for
precise counts.  When it isn't, falls back to the legacy 4-chars-per-token
heuristic and logs a one-time warning.
"""

from __future__ import annotations

import logging
import math
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawagents.providers.llm import LLMMessage

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_CHARS_PER_TOKEN_FALLBACK = 4

# Per-message overhead tokens (role, delimiters, etc.)
# OpenAI documents ~4 tokens per message for chat models.
_PER_MESSAGE_OVERHEAD = 4

# ── Encoding resolution ──────────────────────────────────────────────────

# Mapping from model-name prefix → tiktoken encoding name.
# Order matters: more specific prefixes first.
_MODEL_TO_ENCODING: list[tuple[str, str]] = [
    # GPT-5 series uses o200k_base
    ("gpt-5", "o200k_base"),
    # GPT-4o series uses o200k_base
    ("gpt-4o", "o200k_base"),
    # o-series reasoning models
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4", "o200k_base"),
    # Legacy GPT-4 / GPT-3.5
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
]

_DEFAULT_ENCODING = "o200k_base"

_fallback_warned = False


def _encoding_for_model(model: str | None) -> str:
    """Return the tiktoken encoding name for a model string."""
    if model:
        lower = model.lower()
        for prefix, enc in _MODEL_TO_ENCODING:
            if lower.startswith(prefix):
                return enc
    return _DEFAULT_ENCODING


@lru_cache(maxsize=8)
def _get_encoder(encoding_name: str):  # type: ignore[return]
    """Lazy-load and cache a tiktoken encoder.  Returns None if unavailable."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except ImportError:
        global _fallback_warned
        if not _fallback_warned:
            _fallback_warned = True
            warnings.warn(
                "tiktoken is not installed — falling back to rough "
                "4-chars-per-token estimation.  Install it for accurate "
                "token counts:  pip install tiktoken",
                stacklevel=3,
            )
        return None
    except Exception as exc:
        logger.debug("tiktoken encoder load failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────


def count_tokens(text: str, model: str | None = None) -> int:
    """Return the token count for *text*.

    Uses tiktoken when available; otherwise falls back to ``len(text) / 4``.
    """
    if not text:
        return 0

    enc_name = _encoding_for_model(model)
    encoder = _get_encoder(enc_name)

    if encoder is not None:
        return len(encoder.encode(text))

    # Fallback
    return math.ceil(len(text) / _CHARS_PER_TOKEN_FALLBACK)


def count_tokens_content(
    content: str | list[dict[str, Any]],
    model: str | None = None,
    multiplier: float = 1.0,
) -> int:
    """Count tokens for a message content field (str or multimodal list)."""
    if isinstance(content, str):
        base = count_tokens(content, model)
    else:
        # Multimodal: text parts + ~500 tokens per image part
        text_chars = sum(len(p.get("text", "")) for p in content)
        image_count = sum(1 for p in content if p.get("type") == "image_url")
        base = count_tokens("x" * text_chars, model) + image_count * 500
    return math.ceil(base * multiplier)


def count_messages_tokens(
    messages: list["LLMMessage"],
    model: str | None = None,
    multiplier: float = 1.0,
) -> int:
    """Count total tokens across a list of LLMMessage objects.

    Adds per-message overhead (~4 tokens for role/delimiters).
    """
    total = 0
    for m in messages:
        total += count_tokens_content(m.content, model, multiplier) + _PER_MESSAGE_OVERHEAD
    return total
