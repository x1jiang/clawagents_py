"""Truncated-JSON repair and conversion of native tool schemas to each provider's wire format.

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
logger = logging.getLogger("clawagents.providers.llm")
from clawagents.providers.base import (
    NativeToolCall,
    NativeToolSchema,
)

# ─── Truncated JSON Repair ─────────────────────────────────────────────────


def _repair_json(text: str) -> Any:
    """Best-effort parse of possibly-truncated JSON from an LLM tool call.

    Strategy:
      1. Try normal json.loads.
      2. Try closing open braces/brackets from the end.
      3. Fall back to empty dict.
    """
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to close unclosed braces/brackets
    closers = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in closers:
            stack.append(closers[ch])
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    # If truncation happened mid-string ({"path": "/tmp/fi…), terminate the
    # dangling string literal before appending the structural closers —
    # otherwise the recoverable prefix was thrown away entirely.
    repaired = text + ('"' if in_string else "") + "".join(reversed(stack))
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract a partial object up to the last complete key-value
    try:
        # Truncate to last comma or colon and close
        for i in range(len(text) - 1, 0, -1):
            if text[i] in (",", ":"):
                candidate = text[:i].rstrip(",: \t\n") + "".join(reversed(stack))
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    logger.warning("JSON repair failed for tool call arguments (input: %s) — using empty args", text[:200])
    return {}


# ─── Native Tool Schema Converters ────────────────────────────────────────


def _to_openai_tools(schemas: list[NativeToolSchema]) -> list[dict[str, Any]]:
    """Convert NativeToolSchema list → OpenAI Chat Completions `tools` param."""
    from clawagents.providers.tool_schema import emit_openai_schema_node

    result = []
    for s in schemas:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for k, v in s.parameters.items():
            if not isinstance(v, dict):
                continue
            prop = emit_openai_schema_node(v)
            if v.get("required"):
                required.append(k)
            properties[k] = prop
        fn_def: dict[str, Any] = {
            "name": s.name,
            "description": s.description,
            "parameters": {"type": "object", "properties": properties},
        }
        if required:
            fn_def["parameters"]["required"] = required
        result.append({"type": "function", "function": fn_def})
    return result


def _to_gemini_tools(schemas: list[NativeToolSchema]) -> list[dict[str, Any]]:
    """Convert NativeToolSchema list → Gemini FunctionDeclaration format."""
    from clawagents.providers.tool_schema import emit_gemini_schema_node

    declarations = []
    for s in schemas:
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for k, v in s.parameters.items():
            if not isinstance(v, dict):
                continue
            prop = emit_gemini_schema_node(v)
            if v.get("required"):
                required.append(k)
            properties[k] = prop
        decl: dict[str, Any] = {
            "name": s.name,
            "description": s.description,
            "parameters": {"type": "OBJECT", "properties": properties},
        }
        if required:
            decl["parameters"]["required"] = required
        declarations.append(decl)
    return [{"function_declarations": declarations}]


def _hashed_session_key(session_id: str) -> str:
    """Stable opaque key derived from a logical session identity."""
    import hashlib

    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return f"claw-{digest}"


def _openai_affinity(base_url: str | None, session_id: str | None) -> dict[str, Any]:
    """OpenAI prompt_cache_key + sticky headers; OpenRouter session header."""
    if not session_id:
        return {}
    key = _hashed_session_key(session_id)
    lower = (base_url or "https://api.openai.com/v1").lower()
    if "api.openai.com" in lower:
        return {
            "prompt_cache_key": key,
            "extra_headers": {
                "session_id": key,
                "x-client-request-id": key,
            },
        }
    if "openrouter.ai" in lower:
        return {"extra_headers": {"x-session-id": key}}
    return {}


def _fire_first_token(cb: Any) -> None:
    if cb is None:
        return
    try:
        cb()
    except Exception:
        pass


def _usage_detail_int(details: Any, *names: str) -> int:
    """Read the first present int field from a usage-details object or dict."""
    if details is None:
        return 0
    for name in names:
        if isinstance(details, dict):
            raw = details.get(name)
        else:
            raw = getattr(details, name, None)
        if raw is None:
            continue
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _openai_reasoning_tokens(usage: Any) -> int:
    """Reasoning tokens from ``completion_tokens_details`` (OpenAI) or the
    top-level ``usage.reasoning_tokens`` that SGLang / vLLM-style servers report."""
    if not usage:
        return 0
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details") or usage.get(
            "output_tokens_details"
        )
        top = usage.get("reasoning_tokens")
    else:
        details = getattr(usage, "completion_tokens_details", None) or getattr(
            usage, "output_tokens_details", None
        )
        top = getattr(usage, "reasoning_tokens", None)
    value = _usage_detail_int(details, "reasoning_tokens")
    if not value:
        try:
            value = int(top or 0)
        except (TypeError, ValueError):
            value = 0
    return max(0, value)


def _openai_reasoning_text(obj: Any) -> str:
    """Separate reasoning channel of a chat message or streaming delta.

    SGLang / vLLM / DeepSeek-style servers put chain-of-thought in
    ``reasoning_content``; some proxies use ``reasoning``. The openai SDK keeps
    unknown fields as attributes (``extra="allow"``) and in ``model_extra``.
    """
    if obj is None:
        return ""
    for attr in ("reasoning_content", "reasoning"):
        value = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        for attr in ("reasoning_content", "reasoning"):
            value = extra.get(attr)
            if isinstance(value, str) and value:
                return value
    return ""


def _openai_cache_tokens(usage: Any) -> tuple[int, int]:
    """Return ``(cache_read_tokens, cache_write_tokens)`` from OpenAI usage.

    Chat Completions expose ``usage.prompt_tokens_details``; Responses API
    exposes ``usage.input_tokens_details``. Both may include:

    - ``cached_tokens`` — prompt-cache *hits* (typically ~10% of input rate)
    - ``cache_write_tokens`` — prompt-cache *loads/writes* (typically 1.25×)

    Alternate names (``cache_creation_tokens``, etc.) are accepted for
    OpenAI-compatible proxies. Missing details → ``(0, 0)``.
    """
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details") or usage.get(
            "input_tokens_details"
        )
    else:
        details = getattr(usage, "prompt_tokens_details", None) or getattr(
            usage, "input_tokens_details", None
        )
    read = _usage_detail_int(details, "cached_tokens", "cache_read_tokens")
    write = _usage_detail_int(
        details,
        "cache_write_tokens",
        "cache_creation_tokens",
        "cache_creation_input_tokens",
    )
    return read, write


def _openai_cached_tokens(usage: Any) -> int:
    """Read prompt-cache hits from an OpenAI usage object, defaulting to 0."""
    return _openai_cache_tokens(usage)[0]


def _tool_args_text(value: Any) -> str:
    """Tool-call ``arguments`` as JSON text.

    OpenAI sends a JSON string; some OpenAI-compatible servers (llama.cpp,
    older vLLM, proxies) send an already-parsed object. Concatenating a dict
    onto the streaming accumulator used to raise and be swallowed, so the tool
    ran with ``{}``.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return "{}"
    return str(value)


def _args_json_clean(text: str) -> bool:
    try:
        return isinstance(json.loads(text or "{}"), dict)
    except (TypeError, ValueError):
        return False


def _coerce_args_dict(value: Any) -> dict[str, Any]:
    """``_repair_json`` may salvage a list/scalar; tools only take a mapping."""
    return value if isinstance(value, dict) else {}


def _synth_tool_call_id(name: str, position: int) -> str:
    """Servers that omit ``id`` still need a stable pair id for the next request;
    an empty id makes the tool result orphaned on replay."""
    return f"call_{position}_{hashlib.sha1(f'{name}:{position}'.encode()).hexdigest()[:8]}"


def _parse_openai_tool_calls(
    tool_calls: Any,
    *,
    finish_reason: str | None = None,
) -> list[NativeToolCall] | None:
    """Extract NativeToolCall list from OpenAI response tool_calls (handles function vs custom union).

    With ``finish_reason="length"`` a call whose arguments were cut mid-JSON is
    dropped instead of executed on repaired-but-wrong arguments; the loop's
    output-limit recovery then asks the model to continue.
    """
    if not tool_calls:
        return None
    result: list[NativeToolCall] = []
    for position, tc in enumerate(tool_calls):
        fn = getattr(tc, "function", None)
        tc_type = getattr(tc, "type", None)
        if fn is None or (tc_type and tc_type != "function"):
            continue
        name = getattr(fn, "name", "") or ""
        if not name:
            continue
        raw_args = _tool_args_text(getattr(fn, "arguments", None)) or "{}"
        if finish_reason == "length" and not _args_json_clean(raw_args):
            logger.warning(
                "  [openai] tool call %s truncated by max_tokens — dropping it for recovery",
                name,
            )
            continue
        result.append(NativeToolCall(
            tool_name=name,
            args=_coerce_args_dict(_repair_json(raw_args)),
            tool_call_id=getattr(tc, "id", "") or _synth_tool_call_id(name, position),
        ))
    return result if result else None
