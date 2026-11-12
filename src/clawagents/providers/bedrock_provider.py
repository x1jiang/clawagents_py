"""Amazon Bedrock providers: Claude on Bedrock, Claude on Mantle, and the Converse API.

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from clawagents.config.config import EngineConfig
from clawagents.errors.taxonomy import MantleCredentialsError
logger = logging.getLogger("clawagents.providers.llm")
# ─── Anthropic Provider ───────────────────────────────────────────────────

try:
    import anthropic as _anthropic_mod
    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic_mod = None  # type: ignore
    _HAS_ANTHROPIC = False
from clawagents.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    NativeToolCall,
    NativeToolSchema,
    OnChunkCallback,
)
from clawagents.providers.stream_retry import (
    _invoke_callback,
)
from clawagents.providers.tool_convert import (
    _fire_first_token,
)
from clawagents.providers.openai_provider import (
    normalize_reasoning_effort,
)
from clawagents.providers.anthropic_provider import (
    AnthropicProvider,
    _current_anthropic_model,
)

# Stand-ins callers pass when a gateway ignores the key (Ollama, local BAG).
# Mantle does not ignore it — treat these as "no key".
_GATEWAY_KEY_PLACEHOLDERS = frozenset({"bedrock", "mantle", "ollama", "none", "n/a"})


def _mantle_gateway_key(config: EngineConfig) -> str:
    """Bearer token usable on Mantle, or ``""`` when only unusable ones exist.

    Mantle takes a Bedrock API key. Gateway placeholders and vendor keys
    (``sk-…`` for OpenAI, ``sk-ant-…`` for Anthropic) all fail there with an
    opaque 401, so they are rejected here rather than sent. Each field is
    judged on its own so an ambient ``OPENAI_API_KEY`` cannot shadow a Mantle
    key the caller passed explicitly.
    """
    for candidate in (config.openai_api_key, config.anthropic_api_key):
        key = (candidate or "").strip()
        if not key or key.lower() in _GATEWAY_KEY_PLACEHOLDERS or key.startswith("sk-"):
            continue
        return key
    return ""


def _aws_region(config: EngineConfig) -> str:
    return (
        (config.aws_region or "").strip()
        or os.environ.get("AWS_REGION", "").strip()
        or os.environ.get("AWS_DEFAULT_REGION", "").strip()
        or "us-east-1"
    )


def _bedrock_client_kwargs(config: EngineConfig) -> dict[str, Any]:
    """Kwargs for ``AsyncAnthropicBedrock`` / boto3 session."""
    kwargs: dict[str, Any] = {"aws_region": _aws_region(config)}
    if (config.aws_profile or "").strip():
        kwargs["aws_profile"] = config.aws_profile.strip()
    if (config.aws_access_key_id or "").strip():
        kwargs["aws_access_key"] = config.aws_access_key_id.strip()
    if (config.aws_secret_access_key or "").strip():
        kwargs["aws_secret_key"] = config.aws_secret_access_key.strip()
    if (config.aws_session_token or "").strip():
        kwargs["aws_session_token"] = config.aws_session_token.strip()
    return kwargs


def _is_bedrock_claude_model(model_name: str) -> bool:
    lower = model_name.lower()
    return "anthropic." in lower or lower.startswith("claude")


class BedrockProvider(AnthropicProvider):
    """Claude on Amazon Bedrock via ``AsyncAnthropicBedrock`` (Messages API).

    Auth uses the standard AWS credential chain (env keys, shared credentials,
    instance/task role) — suitable for HIPAA workloads on AWS. Requires
    ``pip install 'clawagents[bedrock]'``.
    """

    name = "bedrock"

    def __init__(self, config: EngineConfig):
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Install with: pip install 'clawagents[bedrock]'"
            )
        try:
            bedrock_cls = getattr(_anthropic_mod, "AsyncAnthropicBedrock", None)
        except Exception:  # noqa: BLE001
            bedrock_cls = None
        if bedrock_cls is None:
            raise ImportError(
                "AsyncAnthropicBedrock is unavailable. Upgrade anthropic "
                "(pip install 'clawagents[bedrock]') — boto3 is also required."
            )
        # Do not call AnthropicProvider.__init__ (that builds AsyncAnthropic).
        self.client = bedrock_cls(**_bedrock_client_kwargs(config))
        self.model = (
            (config.bedrock_model or "").strip()
            or (config.anthropic_model or "").strip()
            or "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._reasoning_effort = normalize_reasoning_effort(config.reasoning_effort)


class MantleAnthropicProvider(AnthropicProvider):
    """Claude on Bedrock Mantle via ``AsyncAnthropicBedrockMantle``.

    Plain ``AsyncAnthropic`` sends ``X-Api-Key``; Mantle expects
    ``Authorization: Bearer <Bedrock API key>`` (same as OpenAI Mantle chat).
    Official AWS samples use ``AnthropicBedrockMantle`` / async twin.
    """

    name = "anthropic"

    def __init__(self, config: EngineConfig):
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Install with: pip install clawagents[anthropic]"
            )
        # Sending a placeholder or vendor key as the bearer earns a bare 401
        # "Invalid bearer token" that reads like a bad key, not a missing one.
        key = _mantle_gateway_key(config)
        if not key:
            raise MantleCredentialsError(
                "Bedrock Mantle requires an API key. Set BEDROCK_API_KEY (or "
                "MANTLE_API_KEY) to a Bedrock API key for the Mantle region, then "
                "restart. Native AWS IAM credentials are not used for Mantle, and "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY are not valid there."
            )
        base = (getattr(config, "anthropic_base_url", None) or "").strip().rstrip("/")
        if not base and config.openai_base_url:
            base = mantle_anthropic_base_url(config.openai_base_url)
        region = _mantle_region_from_url(base or config.openai_base_url)
        if _current_anthropic_model(config.anthropic_model) in ("opus-5", "sonnet-5"):
            # AWS model cards list these Mantle regions. Never silently move a
            # configured workload across regions to make a model available.
            supported = ("us-east-1", "eu-north-1", "eu-west-1", "ap-southeast-4", "us-gov-west-1")
            if region not in supported:
                raise ValueError(
                    f"{config.anthropic_model} is unavailable through Bedrock Mantle in {region}. "
                    f"Choose a supported Mantle region: {', '.join(supported)}. "
                    "Your configured region has not been changed."
                )
        mantle_cls = getattr(_anthropic_mod, "AsyncAnthropicBedrockMantle", None)
        if mantle_cls is not None:
            client_kwargs: dict[str, Any] = {"aws_region": region, "api_key": key}
            if base:
                client_kwargs["base_url"] = base
            self.client = mantle_cls(**client_kwargs)
        else:
            # anthropic < 0.95: no Mantle client. Bearer via auth_token so the
            # SDK does not fall back to X-Api-Key, which Mantle rejects.
            logger.warning(
                "anthropic %s has no AsyncAnthropicBedrockMantle; using the legacy "
                "auth_token fallback. Upgrade with: pip install -U 'anthropic>=0.95.0'",
                getattr(_anthropic_mod, "__version__", "?"),
            )
            client_kwargs = {"auth_token": key}
            if base:
                client_kwargs["base_url"] = base
            client = _anthropic_mod.AsyncAnthropic(**client_kwargs)
            # auth_headers merges X-Api-Key with the Bearer, and api_key
            # defaults to ANTHROPIC_API_KEY from the environment. Mantle must
            # see the Bearer alone.
            client.api_key = None
            self.client = client
        self.model = (config.anthropic_model or "").strip()
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._reasoning_effort = normalize_reasoning_effort(config.reasoning_effort)


def _converse_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert message content into Bedrock Converse content blocks.

    Strings become a single text block. List content (the internal
    OpenAI-style multimodal shape) maps text parts to text blocks and
    data-URL ``image_url`` parts to native Converse image blocks with
    decoded bytes. Unconvertible parts are dropped — letting the old
    ``str(list)`` fallback run would dump megabytes of base64 into the
    prompt as literal text.
    """
    if content is None or isinstance(content, str):
        return [{"text": content or ""}]
    if not isinstance(content, list):
        return [{"text": str(content)}]
    import base64 as _b64
    import binascii as _binascii

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text") or ""
            if text:
                blocks.append({"text": text})
        elif ptype == "image_url":
            url = ((part.get("image_url") or {}).get("url")) or ""
            if not (isinstance(url, str) and url.startswith("data:") and ";base64," in url):
                continue
            header, b64 = url[5:].split(";base64,", 1)
            fmt = header.split(";", 1)[0].strip().lower().removeprefix("image/")
            if fmt == "jpg":
                fmt = "jpeg"
            if fmt not in ("png", "jpeg", "gif", "webp"):
                continue
            try:
                raw = _b64.b64decode(b64, validate=True)
            except (_binascii.Error, ValueError):
                continue
            blocks.append({"image": {"format": fmt, "source": {"bytes": raw}}})
        elif ptype == "file":
            f = part.get("file") or {}
            fd = f.get("file_data") or ""
            if not (isinstance(fd, str) and fd.startswith("data:") and ";base64," in fd):
                continue
            header, b64 = fd[5:].split(";base64,", 1)
            mime = header.split(";", 1)[0].strip().lower()
            if mime != "application/pdf":
                continue
            try:
                raw = _b64.b64decode(b64, validate=True)
            except (_binascii.Error, ValueError):
                continue
            blocks.append(
                {
                    "document": {
                        "format": "pdf",
                        "name": _converse_doc_name(f.get("filename")),
                        "source": {"bytes": raw},
                    }
                }
            )
    return blocks or [{"text": ""}]


def _converse_doc_name(name: Any) -> str:
    """Sanitize a filename into a Converse document name (ASCII alphanumerics,
    single spaces, hyphens, parentheses, brackets — per the Converse API)."""
    cleaned = "".join(
        c if ((c.isalnum() and c.isascii()) or c in " -()[]") else "-"
        for c in str(name or "")
    )
    cleaned = " ".join(cleaned.split())
    return cleaned[:60] or "document"


class BedrockConverseProvider(LLMProvider):
    """Non-Claude Bedrock models (Nova, Llama, GPT-OSS, …) via Converse API.

    Text chat + tools. Requires ``boto3`` (``pip install 'clawagents[bedrock]'``).
    """

    name = "bedrock-converse"

    def __init__(self, config: EngineConfig):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for Amazon Nova / non-Claude Bedrock models. "
                "Install with: pip install 'clawagents[bedrock]'"
            ) from exc
        region = _aws_region(config)
        session_kwargs: dict[str, Any] = {}
        if (config.aws_profile or "").strip():
            session_kwargs["profile_name"] = config.aws_profile.strip()
        if (config.aws_access_key_id or "").strip():
            session_kwargs["aws_access_key_id"] = config.aws_access_key_id.strip()
        if (config.aws_secret_access_key or "").strip():
            session_kwargs["aws_secret_access_key"] = config.aws_secret_access_key.strip()
        if (config.aws_session_token or "").strip():
            session_kwargs["aws_session_token"] = config.aws_session_token.strip()
        session = boto3.Session(**session_kwargs) if session_kwargs else boto3.Session()
        self._client = session.client("bedrock-runtime", region_name=region)
        self.model = (
            (config.bedrock_model or "").strip()
            or "amazon.nova-pro-v1:0"
        )
        from clawagents.graph.model_profiles import resolve_model_profile

        profile = resolve_model_profile(self.model) or {}
        self._max_tokens = min(
            config.max_tokens, int(profile.get("max_output_tokens", config.max_tokens))
        )
        self._temperature = config.temperature

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        _ = session_id
        system_parts: list[str] = []
        converse_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content if isinstance(m.content, str) else str(m.content))
                continue
            if m.role == "tool":
                text = m.content if isinstance(m.content, str) else str(m.content)
                converse_messages.append(
                    {"role": "user", "content": [{"text": f"[tool {m.tool_call_id or ''}] {text}"}]}
                )
                continue
            role = "assistant" if m.role == "assistant" else "user"
            converse_messages.append(
                {"role": role, "content": _converse_content_blocks(m.content)}
            )

        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": self._max_tokens,
                "temperature": float(self._temperature or 0),
            },
        }
        if system_parts:
            kwargs["system"] = [{"text": "\n".join(system_parts)}]
        if tools:
            from clawagents.providers.tool_schema import emit_openai_schema_node

            # Best-effort tool schemas for Converse toolConfig.
            tool_specs = []
            for s in tools:
                props = {
                    k: emit_openai_schema_node(v)
                    for k, v in s.parameters.items()
                    if isinstance(v, dict)
                }
                required = [k for k, v in s.parameters.items() if v.get("required")]
                tool_specs.append(
                    {
                        "toolSpec": {
                            "name": s.name,
                            "description": s.description or s.name,
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": props,
                                    "required": required,
                                }
                            },
                        }
                    }
                )
            kwargs["toolConfig"] = {"tools": tool_specs}

        loop = asyncio.get_running_loop()

        def _call() -> dict[str, Any]:
            return self._client.converse(**kwargs)

        raw = await loop.run_in_executor(None, _call)
        output = (raw or {}).get("output") or {}
        message = output.get("message") or {}
        content_blocks = message.get("content") or []
        texts: list[str] = []
        tool_calls: list[NativeToolCall] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if "text" in block and block["text"]:
                texts.append(str(block["text"]))
            tool_use = block.get("toolUse")
            if isinstance(tool_use, dict):
                tool_calls.append(
                    NativeToolCall(
                        tool_name=str(tool_use.get("name") or ""),
                        args=tool_use.get("input") or {},
                        tool_call_id=str(tool_use.get("toolUseId") or ""),
                    )
                )
        content = "".join(texts)
        if content or tool_calls:
            _fire_first_token(on_first_token)
        if on_chunk and content:
            # on_chunk may be sync or async; awaiting a sync callback's None raised TypeError.
            await _invoke_callback(on_chunk, content)
        usage = (raw or {}).get("usage") or {}
        prompt_tokens = int(usage.get("inputTokens") or 0)
        cache_read = int(usage.get("cacheReadInputTokens") or 0)
        cache_creation = int(usage.get("cacheWriteInputTokens") or 0)
        tokens = int(usage.get("totalTokens") or 0) or (
            prompt_tokens + int(usage.get("outputTokens") or 0)
        )
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=tokens,
            prompt_tokens=prompt_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            tool_calls=tool_calls or None,
        )


def _mantle_origin(url: str) -> str:
    """``https://bedrock-mantle.{region}.api.aws`` from any Mantle path."""
    from urllib.parse import urlparse

    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    if not parsed.netloc:
        return ""
    scheme = parsed.scheme or "https"
    return f"{scheme}://{parsed.netloc}"


def mantle_anthropic_base_url(url: str) -> str:
    """Mantle Anthropic Messages root (SDK appends ``/v1/messages``)."""
    origin = _mantle_origin(url)
    return f"{origin}/anthropic" if origin else ""


def _mantle_region_from_url(url: str | None) -> str:
    """Extract ``us-east-1`` from ``https://bedrock-mantle.us-east-1.api.aws/…``."""
    from urllib.parse import urlparse

    host = (urlparse((url or "").strip()).hostname or "").lower()
    parts = host.split(".")
    if len(parts) >= 4 and parts[0] == "bedrock-mantle":
        return parts[1]
    return "us-east-1"
