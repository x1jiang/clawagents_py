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
    # ── xAI — Grok (OpenAI-compatible API at https://api.x.ai/v1) ──────
    # Context windows + long-context pricing cliff (≥200K prompt → 2× all
    # token rates) from https://docs.x.ai/developers/pricing (Jul 2026).
    "grok-4.20-multi-agent-0309": {
        "max_input_tokens": 1_000_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-4.20-0309-reasoning": {
        "max_input_tokens": 1_000_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-4.20-0309-non-reasoning": {
        "max_input_tokens": 1_000_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-4.20": {
        "max_input_tokens": 1_000_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-build-0.1": {
        "max_input_tokens": 256_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-build": {
        "max_input_tokens": 256_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-4.5": {
        "max_input_tokens": 500_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-4.3": {
        "max_input_tokens": 1_000_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok-4": {
        "max_input_tokens": 256_000,
        "budget_ratio": 0.85,
        "long_context_threshold": 200_000,
    },
    "grok": {"max_input_tokens": 131_072, "budget_ratio": 0.85},
    # ── Google — Gemini 3.x (1M–2M context) ────────────────────────────
    "gemini-3.8-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.8": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.7-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.7": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.6-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.6": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.5-flash-lite": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.5-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.1-pro": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.1-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.1": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-pro": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-flash-preview": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    # ── Google — Gemini 2.5 ────────────────────────────────────────────
    "gemini-2.5-pro": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-2.5-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    # ── Anthropic — Claude 5 / 4.x ─────────────────────────────────────
    # Context windows per https://platform.claude.com/docs/en/build-with-claude/context-windows
    # (Sep 2026): Fable 5/5.1, Mythos 5/5.1, Opus 5, Opus 4.8/4.7/4.6,
    # Sonnet 5 and Sonnet 4.6 are 1M by default (no beta header). Every
    # other Claude model — including Sonnet 4.5, Haiku 4.5 and Opus 4.5 —
    # is 200K. Order matters: dotted-minor ids ("opus-4-8") must precede
    # the bare "claude-opus-4" family fallback.
    "claude-fable-5-1": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-fable-5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-mythos-5-1": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-mythos-5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-opus-5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-opus-4-8": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-opus-4-7": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-opus-4-6": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-opus-4-5": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-opus-4": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-sonnet-5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-sonnet-4-6": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-4.6-sonnet": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-4.5-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-sonnet-4-5": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-sonnet-4": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-haiku-4-5": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-haiku-4": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    # ── Anthropic — Claude 3.x ─────────────────────────────────────────
    "claude-3-7-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-3-5-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-3-5-haiku": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    # ── Bedrock Mantle third-party models ──────────────────────────────
    # Context windows from the AWS Bedrock model cards (Sep 2026). Mantle
    # ids keep the vendor dot for DeepSeek ("deepseek.v3.2") but drop it for
    # the others ("zai.glm-5" → "glm-5") — see resolve_model_profile.
    "deepseek.v3.2": {"max_input_tokens": 164_000, "budget_ratio": 0.85},
    "deepseek.v3.1": {"max_input_tokens": 128_000, "budget_ratio": 0.85},
    "kimi-k2.5": {"max_input_tokens": 256_000, "budget_ratio": 0.85},
    "kimi-k2-thinking": {"max_input_tokens": 256_000, "budget_ratio": 0.85},
    "glm-5": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "glm-4.7": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "gpt-oss-safeguard": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gpt-oss": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
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


_GEO_PREFIXES = ("global.", "us.", "eu.", "apac.", "ap.", "af.", "me.", "ca.", "sa.")
_VENDOR_PREFIXES = (
    "openai.",
    "anthropic.",
    "amazon.",
    "meta.",
    "mistral.",
    "cohere.",
    "ai21.",
    "xai.",
    "moonshot.",
    "moonshotai.",
    "zai.",
)
# Mantle ids where the vendor dot is part of the model name we key on.
_KEEP_VENDOR_DOT = ("deepseek.",)


def normalize_model_id(model_name: str) -> str:
    """Reduce a provider-qualified id to the bare model key used in MODEL_PROFILES.

    Handles the Bedrock / Mantle spellings in one pass each:
    ``bedrock/us.anthropic.claude-opus-4-8-20260301-v1:0`` →
    ``claude-opus-4-8-20260301-v1``. Mirrors ``normalizeModelId`` in the
    VS Code webview so both sides agree on which profile a model hits.
    """
    name = str(model_name or "").strip().lower()
    if not name:
        return name
    if name.startswith("bedrock/"):
        name = name[len("bedrock/") :]
    for prefix in _GEO_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if not name.startswith(_KEEP_VENDOR_DOT):
        for prefix in _VENDOR_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
    # Drop the Bedrock ":0" revision suffix so prefix matching sees the model.
    if ":" in name:
        name = name.split(":", 1)[0]
    return name


def resolve_model_profile(model_name: str | None) -> dict[str, int | float] | None:
    """Return the best-matching MODEL_PROFILES entry, or None."""
    if not model_name:
        return None
    name = normalize_model_id(model_name)
    if not name:
        return None
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
