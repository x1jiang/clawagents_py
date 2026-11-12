"""OpenAI and OpenAI-compatible provider (Chat Completions and Responses), with the GPT request-contract helpers.

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any
from openai import AsyncOpenAI, APIStatusError
from clawagents.config.config import EngineConfig
logger = logging.getLogger("clawagents.providers.llm")
# ─── Feature H: Thinking Token Preservation ────────────────────────────────

import re as _re
from clawagents.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    NativeToolCall,
    NativeToolSchema,
    OnChunkCallback,
    _ThinkStreamFilter,
)
from clawagents.providers.stream_retry import (
    _CHUNK_STALL_S,
    _MAX_RETRIES,
    _StreamReplayGuard,
    _admit_stream_breaker,
    _get_stream_breaker,
    _import_http_client_module,
    _invoke_callback,
    _is_retryable,
    _jittered_delay,
    _record_stream_breaker,
    _stall_guarded_stream,
    _with_retry,
)
from clawagents.providers.tool_convert import (
    _args_json_clean,
    _coerce_args_dict,
    _fire_first_token,
    _openai_affinity,
    _openai_cache_tokens,
    _openai_reasoning_text,
    _openai_reasoning_tokens,
    _parse_openai_tool_calls,
    _repair_json,
    _synth_tool_call_id,
    _to_openai_tools,
    _tool_args_text,
)

# ─── OpenAI Provider ──────────────────────────────────────────────────────
#
# Auto-selects Chat Completions (``chat.completions.create``) vs Responses
# (``responses.create``) from model + endpoint. GPT-5.5 / GPT-5.6 / Codex and
# reasoning+tools on official OpenAI prefer Responses so effort works with
# tools; Ollama/BAG/Azure stay on Chat Completions. If Responses is missing on
# the endpoint, the provider falls back to Chat Completions for the session.


# o-series reasoning models require temperature=1 (API restriction).
# GPT-5 models accept any temperature — do NOT include them here.
_FIXED_TEMPERATURE_MODELS: dict[str, float] = {
    "o1": 1.0,
    "o1-mini": 1.0,
    "o1-preview": 1.0,
    "o3": 1.0,
    "o3-mini": 1.0,
    "o4-mini": 1.0,
    "gpt-5-nano": 1.0,
    "gpt-5-mini": 1.0,
    "gpt-5-turbo": 1.0,
}

_NON_REASONING_MODELS: set[str] = {
    "gpt-5-micro", "gpt-4o", "gpt-4o-mini",
}


def _resolve_temperature(model: str, requested: float) -> float:
    """Return the fixed temperature if the model requires it, else the requested value."""
    if model in _NON_REASONING_MODELS:
        return requested
    for prefix, fixed in _FIXED_TEMPERATURE_MODELS.items():
        if model == prefix or model.startswith(prefix + "-"):
            return fixed
    if model == "gpt-5" or model.startswith("gpt-5-2") or model.startswith("gpt-5."):
        return 1.0
    return requested


def _bare_openai_model_id(model: str) -> str:
    """Strip vendor prefixes so classifiers agree on ``gpt-5.6-luna`` etc.

    Mantle / catalog ids often look like ``openai.gpt-5.6-luna``. Sibling
    helpers must all strip the same way or routing/reasoning drift (400s).
    """
    m = (model or "").strip().lower()
    for prefix in ("openai.", "azure.", "mantle.", "xai."):
        if m.startswith(prefix):
            m = m[len(prefix) :]
    return m


def openai_model_rejects_temperature(model: str) -> bool:
    """True when OpenAI / Mantle Responses reject an explicit ``temperature``.

    GPT-5.5 / 5.6 (incl. ``openai.gpt-5.6-luna``), o-series, xAI Grok, and
    similar reasoning models return 400 if ``temperature`` is present. Omit it.
    """
    raw = (model or "").strip().lower()
    if raw.startswith("xai.") or "grok-" in raw or raw.startswith("grok-"):
        return True
    m = _bare_openai_model_id(model)
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith(("gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6", *_GPT6_MODELS)):
        return True
    if m.startswith("gpt-5") and "codex" in m:
        return True
    if anthropic_model_rejects_sampling_params(model):
        return True
    return False


def _with_temperature(
    kwargs: dict[str, Any], model: str, temperature: float | None
) -> dict[str, Any]:
    """Attach ``temperature`` only when the model accepts it."""
    if temperature is None:
        return kwargs
    if openai_model_rejects_temperature(model):
        kwargs.pop("temperature", None)
        return kwargs
    kwargs["temperature"] = temperature
    return kwargs


def anthropic_model_rejects_sampling_params(model: str) -> bool:
    """True when the Anthropic API rejects ``temperature`` / ``top_p`` / ``top_k``.

    Claude Opus 4.7+ (including Mantle ``anthropic.claude-opus-4-8``) and
    Claude 5+ (Opus 5/5.5, Sonnet 5/5.5, Haiku 5/5.5, Fable 5/5.1) return HTTP
    400 ``temperature is deprecated for this model`` if those fields are present.
    Omit them and guide behavior via prompting instead.
    """
    import re

    m = (model or "").strip().lower().replace("_", "-")
    # Collapse dotted minors: opus-4.8 → opus-4-8; keep geo/FM prefixes intact.
    m = re.sub(r"((?:opus|sonnet|haiku|fable)-\d+)\.(\d+)", r"\1-\2", m)
    hit = re.search(r"opus-4-(\d+)", m)
    if hit:
        return int(hit.group(1)) >= 7
    # Claude 5+ generations across Opus, Sonnet, Haiku, and Fable reject sampling params.
    if re.search(r"(?:opus|sonnet|haiku|fable)-([5-9]|[1-9]\d+)(?:-|\b)", m):
        return True
    return False


# Every GPT-6 id this module special-cases. Keep the request-contract helpers
# below keyed on these tuples so a new model is added in one place.
_GPT6_MODELS: tuple[str, ...] = (
    "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6-cyber",
)
# Chat Completions rejects function tools on these at every effort.
# GPT-6 Cyber: its model page lists tools under the Responses API only and
# does not document Chat Completions tool calling, so it takes the same
# conservative route (Responses is documented for it either way).
_GPT6_RESPONSES_ONLY_TOOLS: tuple[str, ...] = (
    "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-astra", "gpt-6-cyber",
)

# Hosted Muse models on the Meta Model API (https://dev.meta.ai/docs/models.md).
# ``muse-glimmer-*`` is the self-hosted open-weight model and is not included.
_HOSTED_MUSE_PREFIXES: tuple[str, ...] = ("muse-spark-", "muse-wonder-", "muse-ember-")


def is_hosted_muse_model(model: str) -> bool:
    """True for a Meta Model API Muse id (``muse-spark-1.3``, ``meta/muse-wonder-1.0``…)."""
    m = (model or "").strip().lower()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    if m.startswith("meta."):
        m = m[len("meta."):]
    return m.startswith(_HOSTED_MUSE_PREFIXES)


def _requires_responses_for_tools(model: str) -> bool:
    """True when Chat Completions rejects function tools at every effort.

    GPT-6.1 Sol and GPT-6 Astra accept only ``low`` and above, and Chat
    Completions rejects tools with any ``reasoning_effort`` (verified live
    2026-10-07), so tool calls must use Responses.
    """
    return _bare_openai_model_id(model).startswith(_GPT6_RESPONSES_ONLY_TOOLS)


def _responses_only_model_label(model: str) -> str:
    bare = _bare_openai_model_id(model)
    return ("GPT-6 Astra" if bare.startswith("gpt-6-astra")
            else "GPT-6.1 Terra" if bare.startswith("gpt-6.1-terra")
            else "GPT-6 Cyber" if bare.startswith("gpt-6-cyber")
            else "GPT-6.1 Sol")


def _chat_completions_needs_reasoning_none(model: str) -> bool:
    """True when Chat Completions rejects tools + default reasoning_effort.

    GPT-5.5 / GPT-5.6 / GPT-6 Sol and Luna default to reasoning. On
    ``/v1/chat/completions``, that combination with function tools returns
    HTTP 400 ("use /v1/responses or set reasoning_effort to 'none'"). Prefer
    Responses for those models; this force-none is only for the Chat Completions
    fallback path.
    """
    m = _bare_openai_model_id(model)
    return m.startswith(("gpt-5.5", "gpt-5.6", "gpt-6-sol", "gpt-6-luna"))


_REASONING_EFFORT_VALUES = frozenset({
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
})


def normalize_reasoning_effort(value: str | None) -> str | None:
    """Return a valid effort string, or None to omit the parameter."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    # UI labels → API values
    aliases = {
        "light": "low",
        "extra high": "xhigh",
        "extra_high": "xhigh",
        "extrahigh": "xhigh",
    }
    v = aliases.get(v, v)
    return v if v in _REASONING_EFFORT_VALUES else None


# xAI Grok offers exactly these levels; `none` / `minimal` are rejected by the
# server (xai-grok-pager/src/slash/commands/effort_levels.rs), so an unsupported
# level must be mapped rather than forwarded.
_GROK_EFFORT_LEVELS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh"})


def clamp_reasoning_effort_for_model(model: str, effort: str | None) -> str | None:
    """Map an effort level onto what ``model`` actually accepts."""
    if not effort:
        return effort
    from clawagents.providers.model_classify import is_grok_model

    bare = _bare_openai_model_id(model)
    if bare.startswith(("gpt-6-astra", "gpt-6.1-sol", "gpt-6-cyber")):
        return "low" if effort in ("none", "minimal") else effort
    if is_hosted_muse_model(bare):
        # Meta Model API: "none" is a 400 on reasoning models, and the
        # Contributor tier does not offer "max".
        if effort == "none":
            return "minimal"
        if effort == "max" and bare.endswith("-contributor"):
            return "xhigh"
        return effort
    if _bare_openai_model_id(model).startswith(("gpt-6-sol", "gpt-6-luna", "gpt-6.1-terra")):
        return "low" if effort == "minimal" else effort
    if not is_grok_model(_bare_openai_model_id(model)):
        return effort
    if effort in _GROK_EFFORT_LEVELS:
        return effort
    # `none`/`minimal` would 400; `max` is not offered. Pick the nearest level.
    return "low" if effort in ("none", "minimal") else "xhigh"


def model_supports_reasoning_effort(model: str) -> bool:
    """Heuristic: models that accept ``reasoning_effort`` / Responses ``reasoning``."""
    m = _bare_openai_model_id(model)
    if not m:
        return False
    if m.startswith("grok"):
        return True
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith(("gpt-5.5", "gpt-5.6", *_GPT6_MODELS)):
        return True
    if is_hosted_muse_model(m):
        return True
    # Bare gpt-5 / gpt-5-codex family (not gpt-5-nano/micro as primary chat)
    if m == "gpt-5" or m.startswith("gpt-5-"):
        return True
    return False


def _normalize_wire_api(value: str | None) -> str:
    """Return ``auto``, ``responses``, or ``chat_completions``."""
    v = (value or "auto").strip().lower().replace("-", "_")
    if v in ("responses", "response"):
        return "responses"
    if v in ("chat", "chat_completions", "completions", "chatcompletions"):
        return "chat_completions"
    return "auto"


def supports_openai_fast_mode(model: str) -> bool:
    """Models with documented Fast processing on the direct OpenAI API."""
    m = model.strip().lower()
    return bool(_re.fullmatch(
        r"(?:gpt-6-(?:astra|sol|luna)|gpt-6\.1-(?:sol|terra)|gpt-5\.6(?:-(?:sol|terra|luna))?)"
        r"(?:-\d{4}-\d{2}-\d{2})?", m,
    ))


def _responses_endpoint_likely(base_url: str | None, api_type: str = "") -> bool:
    """True when Responses is a reasonable default for this host.

    Official OpenAI and unknown OpenAI-compatible gateways (corporate
    Responses-only proxies) are allowed. Azure stays on Chat Completions
    unless ``wire_api=responses`` forces it.
    """
    if (api_type or "").lower() == "azure":
        return False
    return True


def prefers_responses_api(
    model: str,
    *,
    base_url: str | None = None,
    api_type: str = "",
    has_tools: bool = False,
    reasoning_effort: str | None = None,
    wire_api: str | None = None,
) -> bool:
    """Pick Responses vs Chat Completions from model + endpoint + wire_api."""
    wire = _normalize_wire_api(wire_api)
    if wire == "chat_completions":
        return False
    if wire == "responses":
        return True
    # auto
    if not _responses_endpoint_likely(base_url, api_type):
        return False
    m = _bare_openai_model_id(model)
    if not m:
        return False
    if "codex" in m:
        return True
    if m.startswith(("gpt-5.5", "gpt-5.6", *_GPT6_MODELS)):
        return True
    # Other GPT-5 / o-series: Responses when tools + non-none effort so the
    # API accepts both (Chat Completions often forces effort=none).
    effort = normalize_reasoning_effort(reasoning_effort)
    if has_tools and effort and effort != "none":
        if m.startswith(("o1", "o3", "o4")) or m == "gpt-5" or m.startswith("gpt-5"):
            return True
    return False


def _is_responses_unsupported(exc: BaseException) -> bool:
    """True when the endpoint does not implement Responses (safe to fall back)."""
    if isinstance(exc, APIStatusError) and exc.status_code in (404, 405, 501):
        return True
    msg = str(exc).lower()
    needles = (
        "unrecognized request url",
        "invalid pathname",
        "unknown route",
        "not implemented",
        "/v1/responses",
        "does not support responses",
        "no such endpoint",
    )
    if isinstance(exc, APIStatusError) and exc.status_code in (400, 404, 405, 501):
        return any(n in msg for n in needles)
    return "unrecognized request url" in msg or "does not support responses" in msg


def _apply_tool_reasoning_compat(
    kwargs: dict[str, Any],
    *,
    model: str,
    has_tools: bool,
    preferred: str | None = None,
) -> None:
    """Chat Completions reasoning_effort (forces none for GPT-5.5/5.6 and GPT-6 Sol/Luna + tools)."""
    if has_tools and _requires_responses_for_tools(model):
        raise ValueError(
            f"{_responses_only_model_label(model)} requires the Responses API for tool calling. "
            "Set openai_wire_api to 'responses' or 'auto'."
        )
    effort = clamp_reasoning_effort_for_model(
        model, normalize_reasoning_effort(preferred)
    )
    if effort:
        kwargs["reasoning_effort"] = effort
    # Chat Completions tools on these models require none.
    if has_tools and _chat_completions_needs_reasoning_none(model):
        kwargs["reasoning_effort"] = "none"


def _apply_responses_reasoning(
    kwargs: dict[str, Any],
    *,
    preferred: str | None = None,
) -> None:
    """Responses API uses ``reasoning={"effort": ...}`` (tools keep effort)."""
    effort = clamp_reasoning_effort_for_model(
        str(kwargs.get("model") or ""), normalize_reasoning_effort(preferred)
    )
    if effort:
        kwargs["reasoning"] = {"effort": effort}



def _chat_tools_to_responses_tools(
    oai_tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Chat Completions nested ``function`` tools → Responses flat tools."""
    if not oai_tools:
        return None
    out: list[dict[str, Any]] = []
    for t in oai_tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            item: dict[str, Any] = {
                "type": "function",
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                # Explicit non-strict: our schemas may omit additionalProperties /
                # full required lists that Responses strict mode expects.
                "strict": False,
            }
            out.append(item)
        else:
            out.append(t)
    return out or None


def _content_to_responses_parts(content: list[Any], role: str) -> list[dict[str, Any]]:
    """Convert Chat Completions-style content parts to Responses API parts.

    User-image attachments travel internally as ``{"type": "image_url"}``
    blocks; the Responses endpoint rejects that type — it wants
    ``input_text`` / ``input_image`` parts with ``image_url`` as a plain
    string. Assistant history text maps to ``output_text``. Unconvertible
    parts are dropped so one stray block can't 400 the whole request.
    """
    text_type = "output_text" if role == "assistant" else "input_text"
    parts: list[dict[str, Any]] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        if ptype in ("text", "input_text", "output_text"):
            parts.append({"type": text_type, "text": p.get("text", "") or ""})
        elif ptype == "input_image" and role != "assistant":
            parts.append(p)
        elif ptype == "image_url" and role != "assistant":
            url = ((p.get("image_url") or {}).get("url")) or ""
            if isinstance(url, str) and url:
                parts.append({"type": "input_image", "image_url": url})
        elif ptype == "input_file" and role != "assistant":
            parts.append(p)
        elif ptype == "file" and role != "assistant":
            f = p.get("file") or {}
            fd = f.get("file_data") or ""
            if isinstance(fd, str) and fd:
                parts.append(
                    {
                        "type": "input_file",
                        "filename": f.get("filename") or "attachment",
                        "file_data": fd,
                    }
                )
    return parts


def _messages_to_responses_input(
    messages: list[dict[str, Any]],
    deferred_by_call_id: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert Chat Completions-style messages to Responses ``instructions`` + ``input``."""
    instructions_parts: list[str] = []
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                instructions_parts.append(content)
            continue
        if role == "tool":
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content)
            call_id = str(m.get("tool_call_id") or "")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content or "",
                }
            )
            # Cache-preserving activation: introduce tools this result unlocked
            # from the transcript instead of appending them to the tools array
            # (which lives in the cached prefix). Responses expresses that as a
            # client-executed tool_search pair rather than Anthropic's
            # tool_reference block.
            unlocked = (deferred_by_call_id or {}).get(call_id) or []
            if unlocked:
                names = [str(t.get("name") or "") for t in unlocked]
                digest = hashlib.sha256(
                    f"{call_id}:{','.join(names)}".encode()
                ).hexdigest()[:16]
                search_id = f"claw_tool_load_{digest}"
                items.append(
                    {
                        "type": "tool_search_call",
                        "call_id": search_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": " ".join(names), "limit": len(names)},
                    }
                )
                items.append(
                    {
                        "type": "tool_search_output",
                        "call_id": search_id,
                        "execution": "client",
                        "status": "completed",
                        "tools": unlocked,
                    }
                )
            continue
        if role == "assistant" and m.get("tool_calls"):
            content = m.get("content")
            if isinstance(content, str) and content:
                items.append({"role": "assistant", "content": content})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(tc.get("id") or ""),
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue
        if role in ("user", "assistant", "developer"):
            content = m.get("content")
            if content is None:
                content = ""
            if isinstance(content, list):
                content = _content_to_responses_parts(content, role)
                if not content:
                    # Nothing convertible survived (unknown block types); an
                    # empty content list would 400 the request — drop the message.
                    continue
            items.append({"role": role, "content": content})
            continue
        # Unknown roles: pass through if they look like Responses items.
        if isinstance(m.get("type"), str):
            items.append(m)
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, items


def _parse_responses_result(
    resp: Any,
) -> tuple[str, list[NativeToolCall] | None, int, int, int, int]:
    """Return (text, tool_calls, total, prompt, cache_read, cache_write)."""
    content_parts: list[str] = []
    calls: list[NativeToolCall] = []
    for item in getattr(resp, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "message":
            for part in getattr(item, "content", None) or []:
                ptype = getattr(part, "type", None)
                if ptype in ("output_text", "text"):
                    content_parts.append(getattr(part, "text", "") or "")
        elif itype == "function_call":
            calls.append(
                NativeToolCall(
                    tool_name=getattr(item, "name", "") or "",
                    args=_repair_json(getattr(item, "arguments", None) or "{}"),
                    tool_call_id=(
                        getattr(item, "call_id", "") or getattr(item, "id", "") or ""
                    ),
                )
            )
    text = "".join(content_parts)
    if not text:
        text = getattr(resp, "output_text", None) or ""
    usage = getattr(resp, "usage", None)
    total = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
    prompt = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    cached, cache_write = _openai_cache_tokens(usage)
    return text, (calls or None), total, prompt, cached, cache_write


def _parse_responses_extras(resp: Any) -> tuple[str | None, int, str | None]:
    """Return ``(finish_reason, reasoning_tokens, thinking)`` from a Responses object."""
    if resp is None:
        return None, 0, None
    status = str(getattr(resp, "status", "") or "").strip().lower()
    inc = getattr(resp, "incomplete_details", None)
    inc_reason = ""
    if isinstance(inc, dict):
        inc_reason = str(inc.get("reason") or "").strip().lower()
    elif inc is not None:
        inc_reason = str(getattr(inc, "reason", "") or "").strip().lower()
    finish_reason: str | None = None
    if inc_reason == "content_filter":
        finish_reason = "content_filter"
    elif inc_reason in ("max_output_tokens", "max_tokens", "length") or status == "incomplete":
        finish_reason = "length"
    elif status == "completed":
        finish_reason = "stop"
    elif status:
        finish_reason = status

    usage = getattr(resp, "usage", None)
    reasoning_tokens = _openai_reasoning_tokens(usage)

    thinking_parts: list[str] = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for block in (getattr(item, "summary", None) or []) + (
            getattr(item, "content", None) or []
        ):
            btext = (
                block.get("text")
                if isinstance(block, dict)
                else getattr(block, "text", None)
            )
            if isinstance(btext, str) and btext:
                thinking_parts.append(btext)
    thinking = "\n".join(thinking_parts).strip() or None
    return finish_reason, reasoning_tokens, thinking



def _sanitize_openai_tool_pairs(formatted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop orphan tool messages; ensure every tool_calls id has a result.

    OpenAI rejects transcripts where role=tool appears without a preceding
    assistant message that declared that tool_call_id.
    """
    declared: set[str] = set()
    for m in formatted:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tc_id = tc.get("id")
                if tc_id:
                    declared.add(str(tc_id))

    out: list[dict[str, Any]] = []
    for m in formatted:
        if m.get("role") == "tool":
            tc_id = str(m.get("tool_call_id") or "")
            if not tc_id or tc_id not in declared:
                continue
            out.append(m)
            continue
        out.append(m)

    responded = {
        str(m.get("tool_call_id"))
        for m in out
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    final: list[dict[str, Any]] = []
    for m in out:
        final.append(m)
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            tc_id = tc.get("id")
            if not tc_id:
                continue
            tc_id = str(tc_id)
            if tc_id in responded:
                continue
            final.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": (
                        "Tool call was cancelled — the agent was interrupted "
                        "before it could complete."
                    ),
                }
            )
            responded.add(tc_id)
    return final


def _openai_chat_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Format LLMMessages for the OpenAI wire (Chat Completions directly;
    the Responses path converts further via ``_messages_to_responses_input``).

    Multimodal user content — the canonical ``text`` / ``image_url`` /
    ``file`` parts — passes through verbatim: Chat Completions accepts those
    shapes natively. The ``__CACHE_BOUNDARY__`` marker is an Anthropic-only
    prompt-cache hint; strip it here so OpenAI never receives the stray
    internal token at the tail of its system prompt.
    """
    formatted: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool" and m.tool_call_id:
            formatted.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
            )
        elif m.role == "assistant" and m.tool_calls_meta:
            formatted.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                    for tc in m.tool_calls_meta
                ],
            })
        else:
            content = m.content
            if isinstance(content, str) and "__CACHE_BOUNDARY__" in content:
                content = content.replace("__CACHE_BOUNDARY__", "").strip()
            formatted.append({"role": m.role, "content": content})
    return formatted


class _ResponsesDeferredMixin:
    """Deferred-tool bookkeeping for the OpenAI Responses wire.

    Responses expresses mid-run activation as a client-executed
    ``tool_search_call`` / ``tool_search_output`` pair rather than Anthropic's
    ``tool_reference`` block, and the deferred schema travels *inside* that
    output instead of staying in the ``tools`` array.
    """

    _deferred_names: set[str] | None = None
    _deferred_by_call_id: dict[str, list[dict[str, Any]]] | None = None
    _deferred_disabled: bool = False

    def _deferred_tools_enabled(self) -> bool:
        if self._deferred_disabled:
            return False
        try:
            from clawagents.config.features import is_enabled

            return bool(is_enabled("deferred_tool_loading"))
        except Exception:
            return False

    def _prepare_deferred_tools(self, messages, tools, oai_tools) -> None:
        self._deferred_names = None
        self._deferred_by_call_id = None
        if not tools or not oai_tools or not self._deferred_tools_enabled():
            return
        from clawagents.providers.deferred_tools import split_deferred_tools

        _immediate, deferred = split_deferred_tools(messages, tools)
        if not deferred:
            return
        by_name = {t.get("name"): t for t in (_chat_tools_to_responses_tools(oai_tools) or [])}
        introduced: set[str] = set()
        by_call: dict[str, list[dict[str, Any]]] = {}
        for m in messages:
            if getattr(m, "role", "") != "tool" or not getattr(m, "tool_call_id", None):
                continue
            payload = []
            for raw in getattr(m, "added_tool_names", None) or []:
                name = str(raw)
                if name not in deferred or name in introduced:
                    continue
                schema = by_name.get(name)
                if schema is None:
                    continue
                introduced.add(name)
                payload.append({**schema, "defer_loading": True})
            if payload:
                by_call[str(m.tool_call_id)] = payload
        if by_call:
            self._deferred_names = introduced
            self._deferred_by_call_id = by_call

    def _disable_deferred_tools(self, exc: BaseException) -> bool:
        from clawagents.providers.deferred_tools import is_deferred_tool_rejection

        if self._deferred_disabled or not is_deferred_tool_rejection(exc):
            return False
        self._deferred_disabled = True
        self._deferred_names = None
        self._deferred_by_call_id = None
        logger.warning(
            "  [openai-responses] deferred tool loading rejected (%s) — falling "
            "back to the full tool list for the rest of this session",
            type(exc).__name__,
        )
        return True


class OpenAIProvider(_ResponsesDeferredMixin, LLMProvider):
    name = "openai"

    def __init__(self, config: EngineConfig):
        base_url = config.openai_base_url or None
        api_version = config.openai_api_version or None
        api_key = config.openai_api_key or ("not-needed" if base_url else "")

        # ``self.client`` may be either ``AsyncAzureOpenAI`` or ``AsyncOpenAI``.
        # Annotate up front so mypy doesn't pin the variable to the type of the
        # first assignment branch and complain when the fallback assigns the
        # other concrete type.
        self.client: Any
        self._http_client: Any = None
        api_type = (config.openai_api_type or "").lower()
        is_azure = api_type == "azure" or (api_version and base_url and "azure" in base_url.lower())
        ssl_verify = bool(getattr(config, "openai_ssl_verify", True))
        # Custom hosts with private CAs often need verify=False; keep verify
        # for official OpenAI / empty base_url.
        if base_url and "api.openai.com" not in base_url.lower() and not ssl_verify:
            httpx = _import_http_client_module()
            if httpx is None:
                # Don't fall back silently: the user asked for verify=False
                # and would otherwise just see certificate errors.
                logger.warning(
                    "openai_ssl_verify=False ignored: neither httpx nor httpx2 "
                    "is importable, so TLS verification stays on"
                )
            else:
                # Match the SDK's default read budget: a non-streaming
                # reasoning turn on a self-hosted 30B model can exceed 120s
                # and each ReadTimeout re-ran the whole generation.
                self._http_client = httpx.AsyncClient(
                    verify=False,
                    timeout=httpx.Timeout(600.0, connect=10.0),
                )

        if is_azure and api_version and base_url:
            try:
                from openai import AsyncAzureOpenAI
                azure_kwargs: dict[str, Any] = {
                    "api_key": api_key,
                    "azure_endpoint": base_url,
                    "api_version": api_version,
                    "max_retries": 0,
                }
                if self._http_client is not None:
                    azure_kwargs["http_client"] = self._http_client
                self.client = AsyncAzureOpenAI(**azure_kwargs)
            except ImportError:
                client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url, "max_retries": 0}
                if self._http_client is not None:
                    client_kwargs["http_client"] = self._http_client
                self.client = AsyncOpenAI(**client_kwargs)
        else:
            client_kwargs = {"api_key": api_key, "max_retries": 0}
            if base_url:
                client_kwargs["base_url"] = base_url
            if self._http_client is not None:
                client_kwargs["http_client"] = self._http_client
            self.client = AsyncOpenAI(**client_kwargs)

        self.model = config.openai_model
        from clawagents.graph.model_profiles import resolve_model_profile

        profile = resolve_model_profile(self.model) or {}
        self._max_tokens = min(
            config.max_tokens, int(profile.get("max_output_tokens", config.max_tokens))
        )
        self._temperature = _resolve_temperature(config.openai_model, config.temperature)
        self._reasoning_effort = normalize_reasoning_effort(
            getattr(config, "reasoning_effort", None) or None
        )
        self._base_url = base_url
        self._api_type = "azure" if is_azure else api_type
        self._wire_api = _normalize_wire_api(getattr(config, "openai_wire_api", None))
        self._fast_mode = bool(getattr(config, "openai_fast_mode", False))
        self._fast_eligible = (
            not self._api_type
            and (not base_url or base_url.rstrip("/").lower() == "https://api.openai.com/v1")
            and supports_openai_fast_mode(self.model)
        )
        if self._fast_mode and not self._fast_eligible:
            raise ValueError("Fast mode requires a supported model on the direct OpenAI API")
        # Sticky fallback when Responses is missing — never when wire_api forces it.
        self._force_chat_completions = False
        self._warned_responses_bridge = False

    def _should_use_responses(self, has_tools: bool) -> bool:
        # Bridge: these models reject Chat Completions tools at every effort,
        # so tool turns use Responses even when wire_api asks for chat. Azure
        # deployments keep their explicit wire and get the clear error.
        if has_tools and _requires_responses_for_tools(self.model) and self._api_type != "azure":
            if self._wire_api == "chat_completions" and not self._warned_responses_bridge:
                logger.warning(
                    "  [openai] %s tool calls require Responses; routing to /v1/responses "
                    "although wire_api=chat_completions",
                    self.model,
                )
                self._warned_responses_bridge = True
            return True
        if self._force_chat_completions and self._wire_api != "responses":
            return False
        return prefers_responses_api(
            self.model,
            base_url=self._base_url,
            api_type=self._api_type,
            has_tools=has_tools,
            reasoning_effort=self._reasoning_effort,
            wire_api=self._wire_api,
        )

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
        formatted = _sanitize_openai_tool_pairs(_openai_chat_messages(messages))
        from clawagents.providers.glimmer import is_glimmer_model, reasoning_strength_messages

        if is_glimmer_model(self.model):
            formatted = reasoning_strength_messages(formatted, self._reasoning_effort)
        oai_tools = _to_openai_tools(tools) if tools else None
        # Deferred-tool bookkeeping is computed from the ORIGINAL LLMMessages
        # (the chat-format dicts drop added_tool_names) and kept off the wire
        # dicts entirely, so nothing extra can ever reach Chat Completions.
        self._prepare_deferred_tools(messages, tools, oai_tools)

        try:
            from clawagents.circuit_breaker import breaker_key as _bk

            _bkey = _bk("openai", base_url=self._base_url, model=self.model)
            _bkey_resp = _bk("openai-responses", base_url=self._base_url, model=self.model)
        except Exception:
            _bkey = "openai"
            _bkey_resp = "openai-responses"

        return await self._chat_dispatch(
            formatted,
            on_chunk,
            cancel_event,
            oai_tools,
            _bkey,
            _bkey_resp,
            session_id=session_id,
            on_first_token=on_first_token,
        )

    async def _chat_dispatch(
        self,
        formatted: list[dict[str, Any]],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        oai_tools: list[dict[str, Any]] | None,
        _bkey: str,
        _bkey_resp: str,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        if self._should_use_responses(bool(oai_tools)):
            try:
                # Always use the streaming collector — including non-stream
                # callers. ``_request_once_responses`` already delegates there;
                # wrapping it in ``_with_retry`` multiplied attempts (4×4=16).
                return await self._stream_with_retry_responses(
                    formatted,
                    on_chunk,
                    cancel_event,
                    oai_tools,
                    session_id=session_id,
                    on_first_token=on_first_token,
                )
            except Exception as exc:
                # Unverified deferred-tool wire shape must never cost a turn:
                # drop deferral and replay with the ordinary tool list.
                if self._disable_deferred_tools(exc):
                    return await self._stream_with_retry_responses(
                        formatted,
                        on_chunk,
                        cancel_event,
                        oai_tools,
                        session_id=session_id,
                        on_first_token=on_first_token,
                    )
                if (
                    _is_responses_unsupported(exc)
                    and self._wire_api != "responses"
                    # GPT-6.1 Sol / GPT-6 Astra have no Chat Completions tool fallback.
                    and not (oai_tools and _requires_responses_for_tools(self.model))
                ):
                    logger.warning(
                        "  [openai] Responses API unavailable (%s) — "
                        "falling back to Chat Completions",
                        type(exc).__name__,
                    )
                    self._force_chat_completions = True
                elif (
                    _is_responses_unsupported(exc)
                    and oai_tools
                    and _requires_responses_for_tools(self.model)
                ):
                    raise ValueError(
                        f"{_responses_only_model_label(self.model)} tool calls require the "
                        "Responses API, which this endpoint does not support. Use an "
                        "endpoint that serves /v1/responses."
                    ) from exc
                else:
                    raise

        if not on_chunk:
            return await _with_retry(
                "openai",
                lambda: self._request_once(formatted, oai_tools, session_id=session_id),
                policy=getattr(self, "retry_policy", None),
                breaker_tag=_bkey,
            )
        return await self._stream_with_retry(
            formatted,
            on_chunk,
            cancel_event,
            oai_tools,
            session_id=session_id,
            on_first_token=on_first_token,
        )

    async def _request_once(
        self, messages: list[dict[str, Any]],
        oai_tools: list[dict[str, Any]] | None = None,
        *,
        session_id: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
        }
        if getattr(self, "_fast_eligible", False):
            kwargs["service_tier"] = "fast" if self._fast_mode else "default"
        _with_temperature(kwargs, self.model, self._temperature)
        if oai_tools:
            kwargs["tools"] = oai_tools
        schema = getattr(self, "_structured_json_schema", None)
        if isinstance(schema, dict) and schema:
            try:
                from clawagents.structured_output import openai_chat_response_format

                kwargs["response_format"] = openai_chat_response_format(schema)
            except Exception:
                pass
        _apply_tool_reasoning_compat(
            kwargs,
            model=self.model,
            has_tools=bool(oai_tools),
            preferred=self._reasoning_effort,
        )
        affinity = _openai_affinity(self._base_url, session_id)
        if affinity.get("prompt_cache_key"):
            kwargs["prompt_cache_key"] = affinity["prompt_cache_key"]
        from clawagents.providers.gemma import apply_chat_options
        apply_chat_options(kwargs, self.model, profile_model=getattr(self, "_gemma_agentic_model", ""))
        create_kwargs: dict[str, Any] = dict(kwargs)
        if affinity.get("extra_headers"):
            create_kwargs["extra_headers"] = affinity["extra_headers"]
        resp = await self.client.chat.completions.create(**create_kwargs)
        _prompt_tokens = (resp.usage.prompt_tokens or 0) if resp.usage else 0
        _cached_tokens, _cache_write_tokens = _openai_cache_tokens(resp.usage)
        _reasoning_tokens = _openai_reasoning_tokens(resp.usage)
        if not resp.choices:
            # Azure content filters and some OpenAI-compatible proxies return
            # 200 with an empty ``choices`` array; don't IndexError on it.
            return LLMResponse(content="", model=self.model,
                               tokens_used=resp.usage.total_tokens if resp.usage else 0,
                               prompt_tokens=_prompt_tokens,
                               cache_read_tokens=_cached_tokens,
                               cache_creation_tokens=_cache_write_tokens,
                               reasoning_tokens=_reasoning_tokens,
                               partial=True)
        choice = resp.choices[0]
        msg = choice.message
        _finish = getattr(choice, "finish_reason", None)
        native_calls = _parse_openai_tool_calls(
            getattr(msg, "tool_calls", None),
            finish_reason=str(_finish) if _finish else None,
        )
        return LLMResponse(
            content=msg.content or "",
            model=self.model,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
            prompt_tokens=_prompt_tokens,
            cache_read_tokens=_cached_tokens,
            cache_creation_tokens=_cache_write_tokens,
            tool_calls=native_calls,
            thinking=_openai_reasoning_text(msg) or None,
            finish_reason=str(_finish) if _finish else None,
            reasoning_tokens=_reasoning_tokens,
        )

    def _responses_kwargs(
        self,
        messages: list[dict[str, Any]],
        oai_tools: list[dict[str, Any]] | None,
        *,
        stream: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        instructions, input_items = _messages_to_responses_input(
            messages, getattr(self, "_deferred_by_call_id", None)
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self._max_tokens,
            "store": False,
        }
        if getattr(self, "_fast_eligible", False):
            kwargs["service_tier"] = "fast" if self._fast_mode else "default"
        _with_temperature(kwargs, self.model, self._temperature)
        if instructions:
            kwargs["instructions"] = instructions
        resp_tools = _chat_tools_to_responses_tools(oai_tools)
        deferred_names = getattr(self, "_deferred_names", None) or set()
        if deferred_names:
            # Unlike Anthropic (deferred stays in `tools` with defer_loading),
            # Responses carries the schema inside tool_search_output — leaving
            # it in `tools` too would rewrite the prefix we are protecting.
            resp_tools = [t for t in (resp_tools or []) if t.get("name") not in deferred_names]
        if resp_tools:
            kwargs["tools"] = resp_tools
        schema = getattr(self, "_structured_json_schema", None)
        if isinstance(schema, dict) and schema:
            try:
                from clawagents.structured_output import openai_responses_text_format

                kwargs["text"] = openai_responses_text_format(schema)
            except Exception:
                pass
        _apply_responses_reasoning(kwargs, preferred=self._reasoning_effort)
        if stream:
            kwargs["stream"] = True
        affinity = _openai_affinity(self._base_url, session_id)
        if affinity.get("prompt_cache_key"):
            kwargs["prompt_cache_key"] = affinity["prompt_cache_key"]
        if affinity.get("extra_headers"):
            kwargs["extra_headers"] = affinity["extra_headers"]
        return kwargs

    async def _request_once_responses(
        self,
        messages: list[dict[str, Any]],
        oai_tools: list[dict[str, Any]] | None = None,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        # Some OpenAI-compatible gateways (Codex Responses proxies) ignore
        # stream=false and return raw SSE text the SDK cannot parse. Always
        # collect via the streaming path.
        return await self._stream_with_retry_responses(
            messages,
            on_chunk=None,
            cancel_event=None,
            oai_tools=oai_tools,
            session_id=session_id,
            on_first_token=on_first_token,
        )

    async def _stream_with_retry_responses(
        self,
        messages: list[dict[str, Any]],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        oai_tools: list[dict[str, Any]] | None = None,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None
        breaker = _get_stream_breaker(
            "openai-responses", base_url=self._base_url, model=self.model
        )

        # De-duplicate text across retry attempts (see _StreamReplayGuard).
        replay_guard = _StreamReplayGuard(on_chunk)
        if on_chunk is not None:
            on_chunk = replay_guard.feed

        attempt = 0
        while attempt <= _MAX_RETRIES:
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [openai-responses] Stream retry %d/%d after %.1fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            try:
                await _admit_stream_breaker(breaker)
            except Exception as exc:
                last_error = exc
                try:
                    from clawagents.circuit_breaker import BreakerOpen as _BO

                    if isinstance(exc, _BO):
                        await asyncio.sleep(max(0.05, float(exc.retry_after) or 0.05))
                        continue
                except Exception:
                    pass
                raise

            replay_guard.begin_attempt()
            chunks: list[str] = []
            reasoning_chunks: list[str] = []
            final_tokens = 0
            final_prompt_tokens = 0
            final_cached_tokens = 0
            final_cache_write_tokens = 0
            final_reasoning_tokens = 0
            final_finish_reason: str | None = None
            final_thinking: str | None = None
            first_token_fired = False
            # Key by call_id when present so a duplicated/colliding output_index
            # cannot merge two tool calls' arguments. Fall back to idx:* keys.
            tools_accumulation: dict[str, dict[str, Any]] = {}
            index_to_key: dict[int, str] = {}

            def _remember_tool(
                idx: int,
                *,
                call_id: str = "",
                name: str = "",
                arguments: str | None = None,
                replace_args: bool = False,
            ) -> str:
                cid = (call_id or "").strip()
                key = f"id:{cid}" if cid else (index_to_key.get(idx) or f"idx:{idx}")
                prev_key = index_to_key.get(idx)
                prev = tools_accumulation.get(key) or (
                    tools_accumulation.get(prev_key, {}) if prev_key else {}
                )
                if cid:
                    # Migrate any idx-keyed stub into the call_id key.
                    if prev_key and prev_key != key and prev_key in tools_accumulation:
                        prev = {**tools_accumulation.pop(prev_key), **prev}
                    index_to_key[idx] = key
                else:
                    index_to_key.setdefault(idx, key)
                if replace_args:
                    args = (
                        arguments
                        if arguments is not None
                        else (prev.get("arguments", "") or "")
                    )
                else:
                    args = (prev.get("arguments", "") or "") + (arguments or "")
                tools_accumulation[key] = {
                    "id": cid or prev.get("id", "") or "",
                    "name": name or prev.get("name", "") or "",
                    "arguments": args,
                }
                return key

            def _accumulated_calls() -> list[NativeToolCall] | None:
                if not tools_accumulation:
                    return None
                calls: list[NativeToolCall] = []
                for _pos, _key in enumerate(sorted(tools_accumulation.keys())):
                    _fn = tools_accumulation[_key]
                    if not _fn.get("name"):
                        continue
                    _raw_args = _tool_args_text(_fn.get("arguments")) or "{}"
                    if final_finish_reason == "length" and not _args_json_clean(_raw_args):
                        logger.warning(
                            "  [openai-responses] tool call %s truncated by max_output_tokens — dropping it for recovery",
                            _fn["name"],
                        )
                        continue
                    calls.append(NativeToolCall(
                        tool_name=_fn["name"],
                        args=_coerce_args_dict(_repair_json(_raw_args)),
                        tool_call_id=_fn.get("id", "") or _synth_tool_call_id(_fn["name"], _pos),
                    ))
                return calls or None

            try:
                kwargs = self._responses_kwargs(
                    messages, oai_tools, stream=True, session_id=session_id
                )
                stream = await self.client.responses.create(**kwargs)

                async for event in _stall_guarded_stream(stream, _CHUNK_STALL_S):
                    if cancel_event and cancel_event.is_set():
                        close = getattr(stream, "close", None)
                        if callable(close):
                            maybe = close()
                            if asyncio.iscoroutine(maybe):
                                await maybe
                        return LLMResponse(
                            content="".join(chunks),
                            model=self.model,
                            tokens_used=final_tokens,
                            prompt_tokens=final_prompt_tokens,
                            cache_read_tokens=final_cached_tokens,
                            cache_creation_tokens=final_cache_write_tokens,
                            reasoning_tokens=final_reasoning_tokens,
                            thinking="".join(reasoning_chunks).strip() or final_thinking,
                            partial=True,
                            tool_calls=_accumulated_calls(),
                        )

                    try:
                        etype = getattr(event, "type", "") or ""
                        if etype == "response.output_text.delta":
                            text = getattr(event, "delta", None) or ""
                            if text:
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                chunks.append(text)
                                await _invoke_callback(on_chunk, text)
                        elif etype in (
                            "response.reasoning_summary_text.delta",
                            "response.reasoning_text.delta",
                        ):
                            rdelta = getattr(event, "delta", None) or ""
                            if rdelta:
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                reasoning_chunks.append(rdelta)
                        elif etype == "response.output_item.added":
                            item = getattr(event, "item", None)
                            idx = int(getattr(event, "output_index", 0) or 0)
                            if item is not None and getattr(item, "type", None) == "function_call":
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                _remember_tool(
                                    idx,
                                    call_id=getattr(item, "call_id", "")
                                    or getattr(item, "id", "")
                                    or "",
                                    name=getattr(item, "name", "") or "",
                                    arguments=getattr(item, "arguments", "") or "",
                                    replace_args=True,
                                )
                        elif etype == "response.function_call_arguments.delta":
                            idx = int(getattr(event, "output_index", 0) or 0)
                            delta = getattr(event, "delta", None) or ""
                            if delta and not first_token_fired:
                                first_token_fired = True
                                _fire_first_token(on_first_token)
                            _remember_tool(idx, arguments=delta, replace_args=False)
                        elif etype == "response.output_item.done":
                            item = getattr(event, "item", None)
                            idx = int(getattr(event, "output_index", 0) or 0)
                            if item is not None and getattr(item, "type", None) == "function_call":
                                _remember_tool(
                                    idx,
                                    call_id=getattr(item, "call_id", "")
                                    or getattr(item, "id", "")
                                    or "",
                                    name=getattr(item, "name", "") or "",
                                    arguments=getattr(item, "arguments", None),
                                    replace_args=True,
                                )
                        elif etype in ("response.completed", "response.incomplete"):
                            resp = getattr(event, "response", None)
                            if etype == "response.incomplete":
                                final_finish_reason = "length"
                            if resp is not None:
                                _t, _c, total, prompt, cached, cache_write = (
                                    _parse_responses_result(resp)
                                )
                                _fr, _rtok, _think = _parse_responses_extras(resp)
                                final_tokens = total
                                final_prompt_tokens = prompt
                                final_cached_tokens = cached
                                final_cache_write_tokens = cache_write
                                final_reasoning_tokens = _rtok
                                if _fr:
                                    final_finish_reason = _fr
                                elif etype == "response.incomplete":
                                    final_finish_reason = "length"
                                if _think:
                                    final_thinking = _think
                                if _t and not chunks:
                                    if not first_token_fired:
                                        first_token_fired = True
                                        _fire_first_token(on_first_token)
                                    chunks.append(_t)
                                    await _invoke_callback(on_chunk, _t)
                                if _c and not tools_accumulation:
                                    if not first_token_fired:
                                        first_token_fired = True
                                        _fire_first_token(on_first_token)
                                    for i, call in enumerate(_c):
                                        _remember_tool(
                                            i,
                                            call_id=call.tool_call_id,
                                            name=call.tool_name,
                                            arguments=json.dumps(call.args),
                                            replace_args=True,
                                        )
                        elif etype in ("response.failed", "error"):
                            raise RuntimeError(f"Responses stream failed: {event}")
                    except RuntimeError:
                        raise
                    except Exception:
                        pass  # malformed event — skip

                _record_stream_breaker(breaker, success=True)
                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    prompt_tokens=final_prompt_tokens,
                    cache_read_tokens=final_cached_tokens,
                    cache_creation_tokens=final_cache_write_tokens,
                    reasoning_tokens=final_reasoning_tokens,
                    thinking="".join(reasoning_chunks).strip() or final_thinking,
                    finish_reason=final_finish_reason,
                    tool_calls=_accumulated_calls(),
                )

            except Exception as exc:
                last_error = exc
                if _is_responses_unsupported(exc):
                    raise
                if _is_retryable(exc):
                    _record_stream_breaker(breaker, success=False)
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [openai-responses] Stream interrupted after %d chars — retrying (%s: %.200s)",
                        len("".join(chunks)), type(exc).__name__, exc,
                    )
                    attempt += 1
                    continue
                partial = "".join(chunks)
                acc_calls = _accumulated_calls()
                if partial.strip() or acc_calls:
                    logger.warning(
                        "  [openai-responses] Stream interrupted after %d chars — returning partial",
                        len(partial),
                    )
                    return LLMResponse(
                        content=partial,
                        model=self.model,
                        tokens_used=final_tokens,
                        prompt_tokens=final_prompt_tokens,
                        cache_read_tokens=final_cached_tokens,
                        cache_creation_tokens=final_cache_write_tokens,
                        partial=True,
                        tool_calls=acc_calls,
                    )
                break

        raise last_error  # type: ignore[misc]

    async def _stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        oai_tools: list[dict[str, Any]] | None = None,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None
        breaker = _get_stream_breaker(
            "openai", base_url=self._base_url, model=self.model
        )

        # De-duplicate text across retry attempts (see _StreamReplayGuard).
        replay_guard = _StreamReplayGuard(on_chunk)
        if on_chunk is not None:
            on_chunk = replay_guard.feed

        attempt = 0
        while attempt <= _MAX_RETRIES:
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [openai] Stream retry %d/%d after %.1fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            try:
                await _admit_stream_breaker(breaker)
            except Exception as exc:
                last_error = exc
                try:
                    from clawagents.circuit_breaker import BreakerOpen as _BO

                    if isinstance(exc, _BO):
                        await asyncio.sleep(max(0.05, float(exc.retry_after) or 0.05))
                        continue
                except Exception:
                    pass
                raise

            replay_guard.begin_attempt()
            chunks: list[str] = []
            thinking_chunks: list[str] = []
            finish_reason: str | None = None
            final_tokens = 0
            final_prompt_tokens = 0
            final_cached_tokens = 0
            final_cache_write_tokens = 0
            final_reasoning_tokens = 0
            tools_accumulation: dict[int, dict[str, Any]] = {}

            think_filter = _ThinkStreamFilter()
            stream: Any = None

            def _accumulated_calls() -> list[NativeToolCall] | None:
                if not tools_accumulation:
                    return None
                calls: list[NativeToolCall] = []
                for position, _idx in enumerate(sorted(tools_accumulation.keys(), key=str)):
                    _fn = tools_accumulation[_idx]
                    name = _fn.get("name") or ""
                    if not name:
                        continue
                    raw_args = _fn.get("arguments") or "{}"
                    if finish_reason == "length" and not _args_json_clean(raw_args):
                        logger.warning(
                            "  [openai] streamed tool call %s truncated by max_tokens — "
                            "dropping it for recovery",
                            name,
                        )
                        continue
                    calls.append(NativeToolCall(
                        tool_name=name,
                        args=_coerce_args_dict(_repair_json(raw_args)),
                        tool_call_id=_fn.get("id") or _synth_tool_call_id(name, position),
                    ))
                return calls or None

            def _finalize_text() -> None:
                # Drain any held-back partial tag; an unclosed <think> is thinking.
                visible, reasoning = think_filter.flush()
                if visible:
                    chunks.append(visible)
                if reasoning:
                    thinking_chunks.append(reasoning)

            def _thinking_text() -> str | None:
                return "".join(thinking_chunks) or None

            async def _close_stream() -> None:
                # Abandoning an SGLang/llama.cpp stream without closing leaves
                # the server generating for a request nobody reads.
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        maybe = close()
                        if asyncio.iscoroutine(maybe):
                            await maybe
                    except Exception:
                        pass

            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": self._max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if getattr(self, "_fast_eligible", False):
                    kwargs["service_tier"] = "fast" if self._fast_mode else "default"
                _with_temperature(kwargs, self.model, self._temperature)
                if oai_tools:
                    kwargs["tools"] = oai_tools
                _apply_tool_reasoning_compat(
                    kwargs,
                    model=self.model,
                    has_tools=bool(oai_tools),
                    preferred=self._reasoning_effort,
                )
                affinity = _openai_affinity(
                    self._base_url, session_id
                )
                if affinity.get("prompt_cache_key"):
                    kwargs["prompt_cache_key"] = affinity["prompt_cache_key"]
                from clawagents.providers.gemma import apply_chat_options
                apply_chat_options(kwargs, self.model, profile_model=getattr(self, "_gemma_agentic_model", ""))
                create_kwargs = dict(kwargs)
                if affinity.get("extra_headers"):
                    create_kwargs["extra_headers"] = affinity["extra_headers"]
                stream = await self.client.chat.completions.create(**create_kwargs)
                first_token_fired = False

                async for chunk in _stall_guarded_stream(stream, _CHUNK_STALL_S):
                    if cancel_event and cancel_event.is_set():
                        await _close_stream()
                        _finalize_text()
                        return LLMResponse(
                            content="".join(chunks),
                            model=self.model,
                            tokens_used=final_tokens,
                            prompt_tokens=final_prompt_tokens,
                            cache_read_tokens=final_cached_tokens,
                            cache_creation_tokens=final_cache_write_tokens,
                            partial=True,
                            tool_calls=_accumulated_calls(),
                            thinking=_thinking_text(),
                            finish_reason=finish_reason,
                            reasoning_tokens=final_reasoning_tokens,
                        )

                    try:
                        if chunk.choices:
                            _fr = getattr(chunk.choices[0], "finish_reason", None)
                            if _fr:
                                finish_reason = str(_fr)
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            # Reasoning models stream chain-of-thought on a
                            # separate channel; it is the first real token
                            # (TTFT) but never part of the visible answer.
                            reasoning_delta = _openai_reasoning_text(delta)
                            if reasoning_delta:
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                thinking_chunks.append(reasoning_delta)
                            if delta.content:
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                # Inline <think> (Qwen/DeepSeek-style) is kept
                                # out of the visible stream, like the final
                                # content after strip_thinking_tokens.
                                text, inline_think = think_filter.feed(delta.content)
                                if inline_think:
                                    thinking_chunks.append(inline_think)
                                if text:
                                    chunks.append(text)
                                    await _invoke_callback(on_chunk, text)

                            if getattr(delta, "tool_calls", None):
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                for tc in delta.tool_calls:
                                    idx = getattr(tc, "index", None)
                                    if idx is None:
                                        # Servers that omit ``index`` still send
                                        # ``id`` on the first delta; without a
                                        # key every call collapsed into one.
                                        idx = getattr(tc, "id", None) or len(tools_accumulation)
                                    if idx not in tools_accumulation:
                                        tools_accumulation[idx] = {"id": "", "name": "", "arguments": ""}
                                    if getattr(tc, "id", None):
                                        tools_accumulation[idx]["id"] = tc.id
                                    if getattr(tc, "function", None):
                                        if tc.function.name:
                                            tools_accumulation[idx]["name"] += tc.function.name
                                        _arg_delta = _tool_args_text(tc.function.arguments)
                                        if _arg_delta:
                                            tools_accumulation[idx]["arguments"] += _arg_delta

                        if chunk.usage:
                            final_tokens = chunk.usage.total_tokens
                            final_prompt_tokens = chunk.usage.prompt_tokens or 0
                            final_cached_tokens, final_cache_write_tokens = (
                                _openai_cache_tokens(chunk.usage)
                            )
                            final_reasoning_tokens = _openai_reasoning_tokens(chunk.usage)
                    except Exception:
                        pass  # malformed chunk — skip

                _record_stream_breaker(breaker, success=True)
                _finalize_text()
                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    prompt_tokens=final_prompt_tokens,
                    cache_read_tokens=final_cached_tokens,
                    cache_creation_tokens=final_cache_write_tokens,
                    tool_calls=_accumulated_calls(),
                    thinking=_thinking_text(),
                    finish_reason=finish_reason,
                    reasoning_tokens=final_reasoning_tokens,
                )

            except Exception as exc:
                last_error = exc
                await _close_stream()
                if _is_retryable(exc):
                    _record_stream_breaker(breaker, success=False)
                # A mid-stream exception used to return the truncated text as a
                # non-retried "final" answer. Retry retryable errors first; only
                # surface a partial (now including any accumulated tool calls)
                # when retries are exhausted or the error is not retryable.
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [openai] Stream interrupted after %d chars — retrying (%s: %.200s)",
                        len("".join(chunks)), type(exc).__name__, exc,
                    )
                    attempt += 1
                    continue
                _finalize_text()
                partial = "".join(chunks)
                acc_calls = _accumulated_calls()
                if partial.strip() or acc_calls:
                    logger.warning(
                        "  [openai] Stream interrupted after %d chars — returning partial",
                        len(partial),
                    )
                    return LLMResponse(
                        content=partial,
                        model=self.model,
                        tokens_used=final_tokens,
                        prompt_tokens=final_prompt_tokens,
                        cache_read_tokens=final_cached_tokens,
                        cache_creation_tokens=final_cache_write_tokens,
                        partial=True,
                        tool_calls=acc_calls,
                        thinking=_thinking_text(),
                        finish_reason=finish_reason,
                        reasoning_tokens=final_reasoning_tokens,
                    )
                break

        raise last_error  # type: ignore[misc]
