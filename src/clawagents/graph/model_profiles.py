"""Model-aware context window profiles for the agent loop."""

from __future__ import annotations

# NOTE: Order matters for prefix matching. List the *most specific*
# keys first so e.g. "gpt-5.4-medium" resolves to the "gpt-5.4" profile
# rather than falling back to "gpt-5".
MODEL_PROFILES: dict[str, dict[str, int | float]] = {
    # ── OpenAI — GPT-5.6 (~1.05M context) ──────────────────────────────
    # long_context_threshold: official pricing cliff (>272K → 2× input / 1.5× output).
    # Economic micro-compact / soft-trim start below this so agent loops stay
    # out of the premium tier when possible (distinct from the 892.5K safety budget).
    "gpt-5.6-sol": {
        "max_input_tokens": 1_050_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 272_000,
    },
    "gpt-5.6-terra": {
        "max_input_tokens": 1_050_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 272_000,
    },
    "gpt-5.6-luna": {
        "max_input_tokens": 1_050_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 272_000,
    },
    "gpt-5.6": {
        "max_input_tokens": 1_050_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 272_000,
    },
    # ── OpenAI — GPT-5.5 / 5.4 family (400K context) ───────────────────
    "gpt-5.5": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.4-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.4-nano": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.4": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.3-codex": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.3-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.3": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.2-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.2": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.1-codex": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.1-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.1": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5-codex": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5-nano": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    # ── OpenAI — GPT-4.1 (1M context) ──────────────────────────────────
    "gpt-4.1-mini": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "gpt-4.1-nano": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "gpt-4.1": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    # ── OpenAI — GPT-4o (128K context) ─────────────────────────────────
    "gpt-4o-mini": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gpt-4o": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    # ── OpenAI — reasoning (o-series) ──────────────────────────────────
    "o4-mini": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o3-mini": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o3": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o1-pro": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o1-mini": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "o1": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    # ── Google — Gemini 3.x (1M–2M context) ────────────────────────────
    "gemini-3.6-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.6": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.5-flash-lite": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.5-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.1-pro": {"max_input_tokens": 2_000_000, "budget_ratio": 0.90},
    "gemini-3.1-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.1": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-pro": {"max_input_tokens": 2_000_000, "budget_ratio": 0.90},
    "gemini-3-flash-preview": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    # ── Google — Gemini 2.5 ────────────────────────────────────────────
    "gemini-2.5-pro": {"max_input_tokens": 2_000_000, "budget_ratio": 0.90},
    "gemini-2.5-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    # ── Anthropic — Claude 4.x ─────────────────────────────────────────
    "claude-opus-4-7": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-opus-4-5": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-opus-4": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-4.6-sonnet": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-4.5-sonnet": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-sonnet-4-5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-sonnet-4": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    # ── Anthropic — Claude 3.x ─────────────────────────────────────────
    "claude-3-7-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-3-5-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-3-5-haiku": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    # ── Ollama / local OpenAI-compatible models ────────────────────────
    # NOTE: prefix-matching walks in insertion order. Put specific tags
    # (``gemma4:e4b``) before generic families (``gemma4``) before legacy
    # prefixes (``gemma3``/``gemma``) so "gemma4:e4b" doesn't collapse to
    # the 8K Gemma-1 default.
    # ── Google — Gemma 4 (released 2026-04-02; Apache-2.0) ─────────────
    "gemma4:e2b": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gemma4:e4b": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gemma4:26b": {"max_input_tokens": 256_000, "budget_ratio": 0.85},
    "gemma4:31b": {"max_input_tokens": 256_000, "budget_ratio": 0.85},
    "gemma4": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    # ── Google — Gemma 3n (edge/mobile 32K) ────────────────────────────
    "gemma3n:e4b": {"max_input_tokens": 32_000, "budget_ratio": 0.80},
    "gemma3n:e2b": {"max_input_tokens": 32_000, "budget_ratio": 0.80},
    "gemma3n": {"max_input_tokens": 32_000, "budget_ratio": 0.80},
    # ── Google — Gemma 3 / 2 / 1 ───────────────────────────────────────
    "gemma3": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gemma2": {"max_input_tokens": 8_192, "budget_ratio": 0.75},
    "gemma": {"max_input_tokens": 8_192, "budget_ratio": 0.75},
    "llama3.3": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "llama3.2": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "llama3.1": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "qwen2.5-coder": {"max_input_tokens": 32_768, "budget_ratio": 0.80},
    "qwen2.5": {"max_input_tokens": 32_768, "budget_ratio": 0.80},
    "deepseek-r1": {"max_input_tokens": 64_000, "budget_ratio": 0.75},
    "mistral": {"max_input_tokens": 32_768, "budget_ratio": 0.80},
    "phi4": {"max_input_tokens": 16_384, "budget_ratio": 0.75},
}


def resolve_model_profile(model_name: str | None) -> dict[str, int | float] | None:
    """Return the best-matching MODEL_PROFILES entry, or None."""
    if not model_name:
        return None
    name = str(model_name).strip().lower()
    if not name:
        return None
    # Strip common Bedrock / provider prefixes so openai.gpt-5.6-luna matches.
    for prefix in (
        "bedrock/",
        "global.",
        "us.",
        "eu.",
        "apac.",
        "ap.",
        "openai.",
        "anthropic.",
        "amazon.",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    profile = MODEL_PROFILES.get(name)
    if profile:
        return profile
    for key, value in MODEL_PROFILES.items():
        if name.startswith(key):
            return value
    return None


def resolve_context_budget(model_name: str, context_window: int) -> tuple[int, float]:
    """Return (effective_window, budget_ratio) based on model profile."""
    profile = resolve_model_profile(model_name)
    if profile:
        return int(profile["max_input_tokens"]), float(profile["budget_ratio"])
    return context_window, 0.75


def resolve_long_context_threshold(model_name: str | None) -> int | None:
    """Pricing long-context cliff in tokens, if the model has one (e.g. Luna 272K)."""
    profile = resolve_model_profile(model_name)
    if not profile:
        return None
    raw = profile.get("long_context_threshold")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
