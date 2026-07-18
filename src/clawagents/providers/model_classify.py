"""Canonical model-string → provider classification.

All host/runtime routing should go through this module so geo prefixes,
LiteLLM ``provider/model`` refs, and key-field assignment cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# AWS Bedrock cross-region / global inference profile prefixes.
# NOTE: the APAC prefix is ``apac.`` (not ``ap.``).
BEDROCK_GEO_PREFIXES: tuple[str, ...] = ("us.", "eu.", "apac.", "global.")

BEDROCK_FM_PREFIXES: tuple[str, ...] = (
    "anthropic.",
    "amazon.",
    "meta.",
    "cohere.",
    "mistral.",
    "ai21.",
    "deepseek.",
    "openai.",  # Bedrock GPT-OSS / Mantle OpenAI catalog
    "qwen.",
)

# LiteLLM-style routing prefixes stripped before the SDK sees the model id.
LITELLM_PROVIDER_PREFIXES: tuple[str, ...] = (
    "bedrock/",
    "ollama/",
    "anthropic/",
    "openai/",
    "gemini/",
    "azure/",
)

ProviderKind = Literal[
    "gemini",
    "anthropic",
    "bedrock",
    "ollama",
    "openai",
]

ApiKeyField = Literal[
    "gemini_api_key",
    "anthropic_api_key",
    "openai_api_key",
]


@dataclass(frozen=True)
class ModelRef:
    """Parsed model reference."""

    raw: str
    bare_id: str
    prefix_hint: Optional[str]  # from litellm ``provider/`` if present


def parse_model_ref(model: str) -> ModelRef:
    """Split optional ``provider/model`` prefix from the bare model id."""
    raw = (model or "").strip()
    lower = raw.lower()
    for prefix in LITELLM_PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            hint = prefix.rstrip("/")
            return ModelRef(raw=raw, bare_id=raw[len(prefix) :], prefix_hint=hint)
    return ModelRef(raw=raw, bare_id=raw, prefix_hint=None)


def strip_bedrock_geo_prefix(model: str) -> str:
    """Strip ``us.`` / ``eu.`` / ``apac.`` / ``global.`` when present."""
    m = (model or "").strip()
    lower = m.lower()
    for prefix in BEDROCK_GEO_PREFIXES:
        if lower.startswith(prefix):
            return m[len(prefix) :]
    return m


def is_gemini_model(model: str) -> bool:
    ref = parse_model_ref(model)
    if ref.prefix_hint == "gemini":
        return True
    return ref.bare_id.lower().startswith("gemini")


def is_anthropic_model(model: str) -> bool:
    """True for Anthropic API model names (not Bedrock FM ids alone).

    Accepts ``claude…``, LiteLLM ``anthropic/…``, and bare Anthropic names.
    Bedrock ``anthropic.claude-…`` / ``us.anthropic.…`` are Bedrock IDs —
    callers should check :func:`is_bedrock_model_id` first when routing.
    """
    ref = parse_model_ref(model)
    if ref.prefix_hint == "anthropic":
        return True
    bare = ref.bare_id.lower()
    if bare.startswith("claude"):
        return True
    # LiteLLM left the slash form; bare may still start with anthropic/
    if bare.startswith("anthropic/") or bare.startswith("anthropic."):
        return True
    return False


def is_bedrock_model_id(model: str) -> bool:
    """True for Amazon Bedrock model / inference-profile IDs."""
    ref = parse_model_ref(model)
    if ref.prefix_hint == "bedrock":
        return True
    lower = ref.bare_id.lower().strip()
    if not lower:
        return False
    if lower.startswith(BEDROCK_GEO_PREFIXES):
        return True
    for prefix in BEDROCK_FM_PREFIXES:
        if lower.startswith(prefix):
            return True
    return False


def strip_bedrock_prefix(model: str) -> str:
    """Strip explicit ``bedrock/`` routing prefix only."""
    ref = parse_model_ref(model)
    if ref.prefix_hint == "bedrock":
        return ref.bare_id
    return (model or "").strip()


def is_ollama_model(model: str) -> bool:
    ref = parse_model_ref(model)
    if ref.prefix_hint == "ollama":
        return True
    return False  # bare tags need local heuristics elsewhere


def normalize_provider_hint(hint: str | None) -> Optional[str]:
    if not hint:
        return None
    h = hint.strip().lower()
    if h in ("aws", "amazon", "amazon-bedrock"):
        return "bedrock"
    if h in ("google", "google-genai"):
        return "gemini"
    if h in ("azure", "azure-openai"):
        return "openai"
    if h in ("openai", "anthropic", "gemini", "bedrock", "ollama"):
        return h
    return None


def classify_model(
    model: str,
    *,
    base_url: str | None = None,
    provider_hint: str | None = None,
) -> ProviderKind:
    """Return the provider kind that should handle ``model``.

    Precedence:
    1. Explicit ``provider_hint`` (profile / settings)
    2. LiteLLM ``provider/`` prefix on the model string
    3. Model-id shape (Bedrock geo/FM, gemini*, claude*, …)
    4. Default: openai (incl. OpenAI-compatible gateways via ``base_url``)
    """
    hint = normalize_provider_hint(provider_hint)
    ref = parse_model_ref(model)
    if hint:
        return hint  # type: ignore[return-value]
    if ref.prefix_hint:
        mapped = normalize_provider_hint(ref.prefix_hint)
        if mapped:
            return mapped  # type: ignore[return-value]

    bare = ref.bare_id
    if is_gemini_model(bare):
        return "gemini"
    # Native Bedrock only when no OpenAI-compatible gateway URL is set.
    if is_bedrock_model_id(bare) and not (base_url or "").strip():
        return "bedrock"
    # Bedrock-shaped IDs with a gateway URL → OpenAI-compatible client.
    if is_bedrock_model_id(bare) and (base_url or "").strip():
        return "openai"
    if is_anthropic_model(bare) and not (base_url or "").strip():
        # Exclude pure Bedrock FM ids already handled above.
        if not is_bedrock_model_id(bare):
            return "anthropic"
        # anthropic.claude-* without base_url is Bedrock (caught above).
        # anthropic/claude-* after strip is bare claude → anthropic.
        return "anthropic"
    if ref.prefix_hint == "ollama" or is_ollama_model(model):
        return "ollama"
    # Heuristic: bare claude* without base_url
    if bare.lower().startswith("claude") and not (base_url or "").strip():
        return "anthropic"
    return "openai"


def api_key_field_for(
    kind: ProviderKind,
    *,
    base_url: str | None = None,
) -> Optional[ApiKeyField]:
    """Which ``EngineConfig`` field should receive an explicit ``api_key``.

    Returns ``None`` for native Bedrock (IAM credential chain — do not stash
    the key into anthropic/openai fields).
    """
    if kind == "gemini":
        return "gemini_api_key"
    if kind == "anthropic":
        return "anthropic_api_key"
    if kind == "bedrock":
        # Native IAM — ignore explicit api_key for field stashing.
        if not (base_url or "").strip():
            return None
        return "openai_api_key"
    # openai / ollama / gateway
    return "openai_api_key"


def sdk_model_id(model: str, *, kind: ProviderKind | None = None) -> str:
    """Model string safe to send to the provider SDK (prefixes stripped)."""
    ref = parse_model_ref(model)
    if kind == "bedrock" or ref.prefix_hint == "bedrock":
        return strip_bedrock_prefix(ref.bare_id) if ref.prefix_hint != "bedrock" else ref.bare_id
    return ref.bare_id
