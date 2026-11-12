"""Provider selection: endpoint/model classification helpers and ``create_provider``.

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import os
from typing import Any
try:
    from google import genai
    from google.genai import types
    _HAS_GEMINI = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GEMINI = False
from clawagents.config.config import EngineConfig
from clawagents.errors.taxonomy import MantleCredentialsError
from clawagents.providers.base import (
    LLMMessage,
    LLMProvider,
)
from clawagents.providers.openai_provider import (
    OpenAIProvider,
    _bare_openai_model_id,
    _normalize_wire_api,
)
from clawagents.providers.gemini_provider import (
    GeminiProvider,
)
from clawagents.providers.anthropic_provider import (
    AnthropicProvider,
)
from clawagents.providers.bedrock_provider import (
    BedrockConverseProvider,
    BedrockProvider,
    MantleAnthropicProvider,
    _is_bedrock_claude_model,
    _mantle_gateway_key,
    _mantle_origin,
    _mantle_region_from_url,
    mantle_anthropic_base_url,
)

# ─── Factory ──────────────────────────────────────────────────────────────

# Prefixes that indicate a model is served by Ollama's OpenAI-compatible
# endpoint (http://localhost:11434/v1). Matching is case-insensitive.
# Users can always force-route by prefixing with ``ollama/``.
_OLLAMA_PREFIXES: tuple[str, ...] = (
    "ollama/",
    "gemma4:",
    "gemma3n:",
    "gemma3:",
    "gemma2:",
    "gemma:",
    "gemma4",
    "gemma3n",
    "gemma3",
    "gemma2",
    "gemma",
    "llama3",
    "llama2",
    "llama",
    "qwen2",
    "qwen",
    "mistral",
    "mixtral",
    "phi4",
    "phi3",
    "phi",
    "deepseek-r1",
    "deepseek",
    "codellama",
)
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _looks_like_ollama(model_name: str) -> bool:
    """Return True if *model_name* looks like an Ollama/local model tag.

    Use the explicit ``ollama/<tag>`` form to bypass heuristics.
    """
    lower = model_name.lower()
    return any(lower.startswith(p) for p in _OLLAMA_PREFIXES)


def _is_mantle_url(url: str | None) -> bool:
    """True for Amazon Bedrock Mantle (OneHUB) OpenAI-compatible hosts."""
    return bool(url) and "bedrock-mantle." in (url or "").lower()


def mantle_openai_base_url(url: str) -> str:
    """Mantle OpenAI frontier root (SDK appends ``/responses`` → ``…/openai/v1/responses``).

    AWS docs require ``https://bedrock-mantle.{region}.api.aws/openai/v1``.
    Passing only ``…/openai`` makes the OpenAI client call ``…/openai/responses``
    (missing ``/v1``) and Mantle returns HTTP 404.
    """
    origin = _mantle_origin(url)
    return f"{origin}/openai/v1" if origin else ""


def is_mantle_anthropic_model(model: str) -> bool:
    """Claude IDs that must use Mantle ``/anthropic/v1/messages`` (not chat)."""
    from clawagents.providers.model_classify import (
        parse_model_ref,
        strip_bedrock_geo_prefix,
    )

    m = parse_model_ref(model or "").bare_id.strip().lower()
    m = strip_bedrock_geo_prefix(m).lower()
    return m.startswith("anthropic.") or m.startswith("claude")


def is_mantle_openai_responses_model(model: str) -> bool:
    """Frontier OpenAI IDs on Mantle that need ``/openai/v1/responses``.

    ``openai.gpt-oss-*`` stays on chat completions; GPT-5.3/5.4/5.5/5.6 do not.
    Accepts bare ``gpt-5.6-luna`` as well as ``openai.gpt-5.6-luna``.
    """
    m = (model or "").strip().lower()
    if m.startswith("openai."):
        m = m[len("openai.") :]
    if "gpt-oss" in m:
        return False
    return any(token in m for token in ("gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6-astra"))


def is_mantle_xai_model(model: str) -> bool:
    """xAI Grok IDs on Mantle that must use the ``/openai/v1`` frontier path.

    Hitting plain ``…/v1/chat/completions`` returns
    ``Berm is not enabled for this account`` (access_denied). Keep the catalog
    id as ``xai.grok-*`` (do not rewrite to ``openai.xai.*``).
    """
    from clawagents.providers.model_classify import (
        parse_model_ref,
        strip_bedrock_geo_prefix,
    )

    m = parse_model_ref(model or "").bare_id.strip().lower()
    m = strip_bedrock_geo_prefix(m).lower()
    return m.startswith("xai.") or m.startswith("grok-")


def needs_mantle_openai_base(model: str) -> bool:
    """True when Mantle must rewrite ``…/v1`` → ``…/openai`` for this model."""
    return is_mantle_openai_responses_model(model) or is_mantle_xai_model(model)


def _mantle_openai_model_id(model: str) -> str:
    """Ensure Mantle OpenAI-catalog Responses models use the ``openai.`` prefix.

    xAI frontier models keep their ``xai.`` catalog ids.
    """
    m = (model or "").strip()
    if not m:
        return m
    if m.lower() == "grok-4.6":
        return "xai.grok-4.6"
    if m.lower().startswith("openai.") or is_mantle_xai_model(m):
        return m
    if is_mantle_openai_responses_model(m):
        return f"openai.{m}"
    return m


def create_provider(
    model_name: str,
    config: EngineConfig,
    *,
    provider_hint: str | None = None,
) -> LLMProvider:
    """Create a single LLM provider inferred from model name.

    Clones ``config`` before mutating provider-specific fields so callers
    can safely reuse one ``EngineConfig`` across providers (e.g. main +
    advisor) or in concurrent flows without cross-talk.

    LiteLLM-style ``provider/model`` prefixes are stripped before the SDK
    sees the model id. Optional ``provider_hint`` (profile / settings)
    overrides id-shape inference.
    """
    from clawagents.providers.model_classify import (
        classify_model,
        is_bedrock_model_id,
        parse_model_ref,
        strip_bedrock_prefix,
    )

    config = config.model_copy()
    ref = parse_model_ref(model_name)
    # Never send ``anthropic/…`` / ``openai/…`` / ``bedrock/…`` literals to SDKs.
    model_name = ref.bare_id
    # Mantle catalog uses ``moonshotai.*`` (not ``moonshot.*``).
    if model_name.lower().startswith("moonshot."):
        model_name = "moonshotai." + model_name.split(".", 1)[1]
    lower = model_name.lower()
    kind = classify_model(
        ref.raw or model_name,
        base_url=config.openai_base_url,
        provider_hint=provider_hint or ref.prefix_hint,
    )

    # ── xAI Grok (OpenAI-compatible wire at https://api.x.ai/v1) ────────
    # Handled before the generic OpenAI path so a bare ``grok-4.5`` works with
    # no base_url configured. An explicit base_url still wins (proxies/gateways).
    from clawagents.providers.model_classify import XAI_BASE_URL, is_grok_model

    if is_grok_model(ref.raw or model_name) and not _is_mantle_url(config.openai_base_url):
        if not (config.openai_base_url or "").strip():
            config.openai_base_url = XAI_BASE_URL
        if not (config.openai_api_key or "").strip():
            config.openai_api_key = (
                os.environ.get("XAI_API_KEY")
                or os.environ.get("GROK_API_KEY")
                or config.openai_api_key
            )
        # xAI's documented surface is /v1/chat/completions. Left on "auto" the
        # unknown-gateway heuristic can pick Responses, which api.x.ai does not
        # serve the same way; an explicit wire_api still wins.
        if _normalize_wire_api(config.openai_wire_api) == "auto":
            config.openai_wire_api = "chat_completions"
        config.openai_model = model_name
        return OpenAIProvider(config)

    if kind == "gemini" or lower.startswith("gemini"):
        if not _HAS_GEMINI:
            raise ImportError(
                "google-genai package not installed. Install with: pip install clawagents[gemini]"
            )
        config.gemini_model = model_name
        return GeminiProvider(config)

    # ── Amazon Bedrock (native IAM) ─────────────────────────────────────
    # Prefer OpenAI-compatible gateway when openai_base_url is set (BAG / LiteLLM).
    # Otherwise route Bedrock model IDs to AsyncAnthropicBedrock (Claude) or
    # Converse (Nova / Llama / GPT-OSS / …).
    bedrock_id = kind == "bedrock" or is_bedrock_model_id(model_name)
    if bedrock_id and not config.openai_base_url:
        model_id = strip_bedrock_prefix(model_name)
        config.bedrock_model = model_id
        if _is_bedrock_claude_model(model_id):
            config.anthropic_model = model_id
            return BedrockProvider(config)
        return BedrockConverseProvider(config)

    # ── Mantle (OneHUB): multi-path host, not one OpenAI /v1 for all ───
    # anthropic.* → /anthropic/v1/messages;
    # openai.gpt-5.* / xai.grok-* → /openai/v1 (responses);
    # chat-ok catalog (gpt-oss, deepseek, qwen, …) → /v1/chat/completions.
    if config.openai_base_url and _is_mantle_url(config.openai_base_url):
        # AWS Astra model card: Mantle is available only in Oregon. Do not
        # silently move requests (or regional credentials) to another region.
        if (
            _bare_openai_model_id(model_name).startswith("gpt-6-astra")
            and _mantle_region_from_url(config.openai_base_url) != "us-west-2"
        ):
            raise ValueError(
                "GPT-6 Astra on Bedrock Mantle requires us-west-2 (Oregon). "
                "Set the AWS region and Mantle base URL to us-west-2."
            )
        if (
            _bare_openai_model_id(model_name) == "grok-4.6"
            and _mantle_region_from_url(config.openai_base_url) != "us-west-2"
        ):
            raise ValueError(
                "Grok 4.6 on Bedrock Mantle requires us-west-2 (Oregon). "
                "Set the AWS region and Mantle base URL to us-west-2. "
                "Your configured region has not been changed."
            )
        # Every Mantle path is Bearer-authenticated; none accept a placeholder
        # or a vendor key, so fail here instead of on an opaque 401.
        mantle_key = _mantle_gateway_key(config)
        if not mantle_key:
            raise MantleCredentialsError(
                "Bedrock Mantle requires an API key. Set BEDROCK_API_KEY (or "
                "MANTLE_API_KEY) to a Bedrock API key for the Mantle region, then "
                "restart. Native AWS IAM credentials are not used for Mantle, and "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY are not valid there."
            )
        config.openai_api_key = mantle_key
        if is_mantle_anthropic_model(model_name):
            config.anthropic_api_key = mantle_key
            config.anthropic_model = model_name
            config.anthropic_base_url = mantle_anthropic_base_url(config.openai_base_url)
            # Must use Mantle client (Bearer), not plain AsyncAnthropic (X-Api-Key).
            return MantleAnthropicProvider(config)
        if needs_mantle_openai_base(model_name):
            rewritten = mantle_openai_base_url(config.openai_base_url)
            if rewritten:
                config.openai_base_url = rewritten
            # Frontier Mantle models reject plain …/v1 (Berm access_denied for
            # xAI; 404/validation for GPT-5.x) even when wire_api was saved as
            # chat_completions from an older Mantle default.
            config.openai_wire_api = "responses"
            # GPT-5.x needs ``openai.`` prefix; xAI keeps ``xai.grok-*``.
            config.openai_model = _mantle_openai_model_id(model_name)
            return OpenAIProvider(config)
        # These verified catalog models only serve Chat Completions. A model
        # switch can retain Astra's /openai/v1 + responses settings, so reset
        # both while keeping the configured regional origin and credentials.
        if model_name.lower() in {
            "minimax.minimax-m2.5", "mistral.devstral-2-123b",
            "qwen.qwen3-coder-next", "nvidia.nemotron-super-3-120b",
            "mistral.mistral-large-3-675b-instruct",
        }:
            config.openai_base_url = _mantle_origin(config.openai_base_url) + "/v1"
            config.openai_wire_api = "chat_completions"
        # Unknown/custom models keep an explicitly chosen transport.
        elif _normalize_wire_api(config.openai_wire_api) == "auto":
            config.openai_wire_api = "chat_completions"

    if kind == "anthropic" or lower.startswith("claude") or lower.startswith("anthropic"):
        # Bedrock Access Gateway / LiteLLM / other OpenAI-compatible proxies set
        # openai_base_url and speak the OpenAI protocol — do not send those
        # requests to Anthropic's native API (model IDs look like
        # anthropic.claude-… or us.anthropic.claude-…).
        if config.openai_base_url:
            if not config.openai_api_key:
                config.openai_api_key = "bedrock"
            config.openai_model = model_name
            return OpenAIProvider(config)
        # Plain "claude-sonnet-4-5" (no Bedrock ID shape) → Anthropic API.
        # Fully-qualified Bedrock IDs without base_url already handled above.
        config.anthropic_model = model_name
        return AnthropicProvider(config)
    if kind == "ollama" or _looks_like_ollama(model_name) or ref.prefix_hint == "ollama":
        # ``ollama/`` already stripped via parse_model_ref.
        tag = model_name
        if not config.openai_base_url:
            config.openai_base_url = _OLLAMA_DEFAULT_BASE_URL
        if not config.openai_api_key:
            # Ollama ignores the API key but the OpenAI client refuses an empty string.
            config.openai_api_key = "ollama"
        config.openai_model = tag
        return OpenAIProvider(config)
    config.openai_model = model_name
    return OpenAIProvider(config)


def has_active_anthropic_tool_turn(messages: list[LLMMessage], provider: Any) -> bool:
    """Protect signed reasoning until the model consumes its tool results.

    A later tool-free assistant response closes the round. Restrict the guard
    to the Anthropic transport; histories switched to another provider need
    not retain Anthropic's encrypted reasoning prefix.
    """
    from clawagents.providers.fallback import FallbackProvider

    if isinstance(provider, FallbackProvider):
        return any(has_active_anthropic_tool_turn(messages, candidate)
                   for candidate in [provider.primary, *provider.fallbacks])
    if not isinstance(provider, AnthropicProvider):
        return False
    for message in reversed(messages):
        blocks = getattr(message, "anthropic_blocks", None) or []
        if message.role == "assistant" and blocks:
            return bool(message.tool_calls_meta) and any(
                block.get("type") in ("thinking", "redacted_thinking") for block in blocks
            )
    return False
