"""Google Gemini provider.

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import asyncio
import logging
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
)
from clawagents.providers.stream_retry import (
    _CHUNK_STALL_S,
    _MAX_RETRIES,
    _StreamReplayGuard,
    _admit_stream_breaker,
    _get_stream_breaker,
    _invoke_callback,
    _is_retryable,
    _jittered_delay,
    _record_stream_breaker,
    _stall_guarded_stream,
    _with_retry,
)
from clawagents.providers.tool_convert import (
    _fire_first_token,
    _to_gemini_tools,
)
from clawagents.providers.openai_provider import (
    normalize_reasoning_effort,
)

# ─── Gemini Provider ──────────────────────────────────────────────────────


def _serialize_gemini_parts(parts: Any) -> list[dict[str, Any]] | None:
    """Serialize Gemini Part objects to dicts, preserving thought/thought_signature/id."""
    if not parts:
        return None
    import base64

    serialized = []
    for p in parts:
        d: dict[str, Any] = {}
        fc = getattr(p, "function_call", None)
        text = getattr(p, "text", None)
        # Skip empty text on function_call parts — empty-only text can make
        # history curators drop the model turn, leaving an orphan FR → 400.
        if text is not None and not (fc and text == ""):
            d["text"] = text
        if getattr(p, "thought", None):
            d["thought"] = True
        sig = getattr(p, "thought_signature", None)
        if sig is not None:
            # Session JSON can't store raw bytes — round-trip via base64.
            if isinstance(sig, (bytes, bytearray)):
                d["thought_signature"] = base64.b64encode(bytes(sig)).decode("ascii")
                d["_thought_signature_b64"] = True
            else:
                d["thought_signature"] = sig
        if fc:
            fc_dict: dict[str, Any] = {
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            }
            fc_id = getattr(fc, "id", None)
            if fc_id:
                fc_dict["id"] = str(fc_id)
            d["function_call"] = fc_dict
        if d:
            serialized.append(d)

    # Gemini 3 parallel calls: signature lives only on the first FC part.
    # Do not copy onto siblings — replay parts as received.
    return serialized if serialized else None


def _restore_gemini_parts_for_api(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode base64 thought signatures before sending contents back to Gemini."""
    import base64

    out: list[dict[str, Any]] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        q = dict(p)
        if q.pop("_thought_signature_b64", False) and isinstance(q.get("thought_signature"), str):
            try:
                q["thought_signature"] = base64.b64decode(q["thought_signature"])
            except Exception:  # noqa: BLE001
                pass
        out.append(q)
    return out


def _stamp_function_call_ids(
    raw_parts: list[dict[str, Any]] | None,
    fn_calls: list[NativeToolCall] | None,
) -> list[dict[str, Any]] | None:
    """Ensure serialized FC parts carry the same ids as NativeToolCall."""
    if not raw_parts or not fn_calls:
        return raw_parts
    fc_dicts = [d for d in raw_parts if isinstance(d, dict) and "function_call" in d]
    for d, tc in zip(fc_dicts, fn_calls):
        if tc.tool_call_id:
            d["function_call"]["id"] = tc.tool_call_id
    return raw_parts


def _part_has_function_call(part: dict[str, Any]) -> bool:
    return isinstance(part, dict) and ("function_call" in part or "functionCall" in part)


def _part_has_function_response(part: dict[str, Any]) -> bool:
    return isinstance(part, dict) and (
        "function_response" in part or "functionResponse" in part
    )


def _parts_have_function_call(parts: list[Any]) -> bool:
    return any(_part_has_function_call(p) for p in parts)


def _parts_have_function_response(parts: list[Any]) -> bool:
    return any(_part_has_function_response(p) for p in parts)


def _model_parts_from_tool_meta(
    content: Any,
    tool_calls_meta: list[dict[str, Any]],
    *,
    gemini_parts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build model parts, preferring preserved gemini_parts (thought_signature)."""
    if gemini_parts and _parts_have_function_call(list(gemini_parts)):
        return _restore_gemini_parts_for_api(list(gemini_parts))

    # Fall back: copy signatures from any prior gemini_parts onto rebuilt FCs.
    sig = None
    sig_b64 = False
    if gemini_parts:
        for p in gemini_parts:
            if isinstance(p, dict) and p.get("thought_signature") is not None:
                sig = p["thought_signature"]
                sig_b64 = bool(p.get("_thought_signature_b64"))
                break

    parts: list[dict[str, Any]] = []
    # Gemini 3: do not put free-text ahead of function_call in history when we
    # lack the original parts — FC-only is safer for FR pairing.
    for tc in tool_calls_meta:
        fc: dict[str, Any] = {
            "name": tc.get("name") or "unknown",
            "args": tc.get("args") or {},
        }
        if tc.get("id"):
            fc["id"] = str(tc["id"])
        part: dict[str, Any] = {"function_call": fc}
        if sig is not None:
            part["thought_signature"] = sig
            if sig_b64:
                part["_thought_signature_b64"] = True
        parts.append(part)
    return _restore_gemini_parts_for_api(parts)


# google-genai FunctionResponse / FunctionCall / Part allow-lists.
# Extra keys (e.g. call_id) raise pydantic extra_forbidden and abort the turn.
_GEMINI_FR_KEYS = frozenset(
    {"id", "name", "response", "will_continue", "scheduling", "parts"}
)
_GEMINI_FC_KEYS = frozenset({"id", "name", "args", "partial_args", "will_continue"})
_GEMINI_PART_KEYS = frozenset(
    {
        "text",
        "thought",
        "thought_signature",
        "function_call",
        "function_response",
        "inline_data",
        "file_data",
        "executable_code",
        "code_execution_result",
        "video_metadata",
        "media_resolution",
        "tool_call",
        "tool_response",
        "part_metadata",
    }
)


def _is_gemini_sdk_shape_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "extra_forbidden" in msg or (
        "validation error" in msg and "generatecontentparameters" in msg
    )


def _scrub_gemini_function_response(fr: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fr.items() if k in _GEMINI_FR_KEYS}


def _scrub_gemini_function_call(fc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fc.items() if k in _GEMINI_FC_KEYS}


def _scrub_gemini_part(part: dict[str, Any]) -> dict[str, Any] | None:
    q = {k: v for k, v in part.items() if k in _GEMINI_PART_KEYS}
    if isinstance(q.get("function_response"), dict):
        q["function_response"] = _scrub_gemini_function_response(q["function_response"])
    if isinstance(q.get("function_call"), dict):
        q["function_call"] = _scrub_gemini_function_call(q["function_call"])
    sig = q.get("thought_signature")
    if isinstance(sig, str):
        import base64

        try:
            q["thought_signature"] = base64.b64decode(sig)
        except Exception:  # noqa: BLE001
            q.pop("thought_signature", None)
    if q.get("text") == "" and ("function_call" in q or "function_response" in q):
        q.pop("text", None)
    if set(q.keys()) <= {"text"} and not str(q.get("text") or "").strip():
        return None
    return q or None


def _scrub_gemini_contents(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop SDK-illegal keys so generate_content cannot fail extra_forbidden."""
    out: list[dict[str, Any]] = []
    for turn in contents:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        parts: list[dict[str, Any]] = []
        for p in turn.get("parts") or []:
            if not isinstance(p, dict):
                continue
            scrubbed = _scrub_gemini_part(p)
            if scrubbed:
                parts.append(scrubbed)
        if role in ("user", "model") and parts:
            out.append({"role": role, "parts": parts})
    return out


def _sanitize_gemini_contents(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild a Gemini-legal transcript: strict alternation + FC→FR pairs.

    Rules enforced:
    - Start with user
    - Alternate user / model
    - Every model function_call turn is followed by exactly one user turn that
      contains only function_response parts (count/ids aligned when possible)
    - Plain user text never shares a turn with function_response
    """
    # First pass: normalize parts
    normalized: list[dict[str, Any]] = []
    for turn in contents:
        role = turn.get("role")
        parts = [p for p in list(turn.get("parts") or []) if isinstance(p, dict)]
        if role not in ("user", "model") or not parts:
            continue
        if role == "model" and _parts_have_function_call(parts):
            parts = _restore_gemini_parts_for_api(parts)
        normalized.append({"role": role, "parts": parts})

    out: list[dict[str, Any]] = []
    i = 0
    while i < len(normalized):
        turn = normalized[i]
        role = turn["role"]
        parts = turn["parts"]

        if role == "model" and _parts_have_function_call(parts):
            # Emit FC model turn (keep original parts incl. thought_signature).
            out.append({"role": "model", "parts": parts})
            fcs = [p for p in parts if _part_has_function_call(p)]
            # Collect following FR-only / tool-response user turns.
            fr_parts: list[dict[str, Any]] = []
            j = i + 1
            while j < len(normalized):
                nxt = normalized[j]
                if nxt["role"] != "user":
                    break
                nparts = nxt["parts"]
                if _parts_have_function_response(nparts):
                    for p in nparts:
                        if _part_has_function_response(p):
                            fr_parts.append(p)
                    j += 1
                    continue
                break
            # Align FR count to FC count
            if len(fr_parts) < len(fcs):
                have_names = {
                    (p.get("function_response") or p.get("functionResponse") or {}).get("name")
                    for p in fr_parts
                }
                for p in fcs:
                    fc = p.get("function_call") or p.get("functionCall") or {}
                    name = fc.get("name") or "unknown"
                    if name in have_names and len(fr_parts) >= len(fcs):
                        continue
                    fr = {
                        "function_response": _gemini_function_response_body(
                            name,
                            "[tool call cancelled or skipped before a result was recorded]",
                            fc.get("id"),
                        )
                    }
                    fr_parts.append(fr)
                    have_names.add(name)
            elif len(fr_parts) > len(fcs):
                fr_parts = fr_parts[: len(fcs)]
            # Ensure FR ids match FC ids by position when FC has id
            for idx, fr in enumerate(fr_parts):
                if idx >= len(fcs):
                    break
                fc = fcs[idx].get("function_call") or fcs[idx].get("functionCall") or {}
                if fc.get("id"):
                    body = fr.get("function_response") or fr.get("functionResponse") or {}
                    fr["function_response"] = _stamp_function_response_ids(body, fc["id"])
                    fr.pop("functionResponse", None)
            if fr_parts:
                out.append({"role": "user", "parts": fr_parts})
            i = j
            continue

        if role == "user" and _parts_have_function_response(parts):
            # Orphan FR (no preceding FC model turn) — drop.
            i += 1
            continue

        if role == "model" and not _parts_have_function_call(parts):
            # Plain model text — only if we don't create model,model
            if out and out[-1]["role"] == "model":
                # Merge text into previous only if previous has no FC
                if not _parts_have_function_call(out[-1]["parts"]):
                    out[-1]["parts"].extend(parts)
                else:
                    # Skip trailing text after FC (FR already handled above)
                    pass
            else:
                out.append({"role": "model", "parts": parts})
            i += 1
            continue

        # Plain user text
        if role == "user":
            plain = [p for p in parts if not _part_has_function_response(p)]
            if not plain:
                i += 1
                continue
            if out and out[-1]["role"] == "user":
                if not _parts_have_function_response(out[-1]["parts"]):
                    out[-1]["parts"].extend(plain)
                else:
                    # After FR: Gemini requires alternation — spacer model, then user.
                    out.append({"role": "model", "parts": [{"text": "…"}]})
                    out.append({"role": "user", "parts": plain})
            elif out and out[-1]["role"] == "model" and _parts_have_function_call(out[-1]["parts"]):
                # Should be unreachable (FC branch synthesizes FR); keep safe.
                out.append({"role": "user", "parts": plain})
            else:
                out.append({"role": "user", "parts": plain})
        i += 1

    while out and out[0]["role"] == "model":
        out.pop(0)
    # Ensure ends with user (Gemini wants last content to be user/function)
    if out and out[-1]["role"] == "model" and not _parts_have_function_call(out[-1]["parts"]):
        # trailing model text without following user is ok for generateContent
        pass
    return _scrub_gemini_contents(out)


def _is_gemini_history_400(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "400" in msg
        or "invalid_argument" in msg
        or "invalid argument" in msg
    ) and (
        "function response" in msg
        or "function call" in msg
        or "thought_signature" in msg
        or "thought signature" in msg
    )


GEMINI_ANSWER_NUDGE = (
    "The tools already ran. Answer the user's question from those results in "
    "plain language. Do not reprint [called …] / [used …] commands, and do not "
    "rewrite files unless asked."
)

GEMINI_SUMMARIZE_MARKER = "[Harness] Please answer from the tool results"
GEMINI_EVIDENCE_MARKER = "[Gemini] No tool ran this turn — do not invent results"

_GEMINI_DUMP_LINE_RE = _re.compile(r"^\[(?:called|used|result)\s+", _re.IGNORECASE)


def looks_like_gemini_command_dump(text: str | None) -> bool:
    """True when a model reply is only flatten-style ``[called …]`` / ``[used …]``.

    Gemini 3.7 often echoes the flatten fallback as its "answer". That is never
    a user-facing result.
    """
    if not text or not str(text).strip():
        return False
    s = str(text).strip()
    if s.startswith("[called ") or s.startswith("[used "):
        parts = _re.split(r"\n\s*\n", s, maxsplit=1)
        if (
            len(parts) == 2
            and len(parts[1].strip()) >= 40
            and not parts[1].lstrip().startswith("[")
        ):
            return False
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return bool(lines) and all(_GEMINI_DUMP_LINE_RE.match(ln) for ln in lines)


def _gemini_function_response_body(
    name: str,
    result: Any,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build a FunctionResponse dict. Gemini 3 pairs by ``id`` only.

    Do not send ``call_id`` — google-genai's Pydantic FunctionResponse
    rejects it (``extra_forbidden``), which aborted the turn.
    """
    body: dict[str, Any] = {
        "name": name,
        "response": {"result": result},
    }
    if call_id:
        body["id"] = str(call_id)
    return body


def _stamp_function_response_ids(
    body: dict[str, Any],
    call_id: str | None,
) -> dict[str, Any]:
    out = dict(body)
    out.pop("call_id", None)
    if call_id:
        out["id"] = str(call_id)
    return out


def _flatten_gemini_fc_text(fc: dict[str, Any]) -> str:
    # Never dump args — write_file/execute payloads become the visible "answer".
    return f"[used {fc.get('name') or 'tool'}]"


def _flatten_gemini_fr_text(fr: dict[str, Any], *, limit: int = 4000) -> str:
    name = fr.get("name") or "tool"
    resp = fr.get("response")
    result = resp.get("result") if isinstance(resp, dict) else resp
    text = str(result)
    if len(text) > limit:
        text = text[:limit] + "…"
    return f"[result {name}: {text}]"


def _upsert_gemini_stream_function_call(
    fn_calls: list[NativeToolCall],
    fc: Any,
) -> NativeToolCall:
    """Keep one NativeToolCall per streamed function_call (latest args win)."""
    import uuid

    name = getattr(fc, "name", None) or "unknown"
    args = dict(fc.args) if getattr(fc, "args", None) else {}
    fc_id = getattr(fc, "id", None)
    if fc_id:
        sid = str(fc_id)
        for i, prev in enumerate(fn_calls):
            if prev.tool_call_id == sid or (
                prev.tool_name == name and str(prev.tool_call_id).startswith("gemini_")
            ):
                fn_calls[i] = NativeToolCall(name, args, sid)
                return fn_calls[i]
        call = NativeToolCall(name, args, sid)
        fn_calls.append(call)
        return call
    for i, prev in enumerate(fn_calls):
        if prev.tool_name == name:
            fn_calls[i] = NativeToolCall(name, args, prev.tool_call_id)
            return fn_calls[i]
    call = NativeToolCall(name, args, f"gemini_{uuid.uuid4().hex[:8]}")
    fn_calls.append(call)
    return call


def _absorb_gemini_stream_parts(all_parts: list[Any], new_parts: Any) -> None:
    """Replace a streamed function_call part in place; append text/thought."""
    for p in new_parts:
        fc = getattr(p, "function_call", None)
        if not fc:
            all_parts.append(p)
            continue
        fc_id = getattr(fc, "id", None)
        name = getattr(fc, "name", None)
        replaced = False
        for i, prev in enumerate(all_parts):
            pfc = getattr(prev, "function_call", None)
            if not pfc:
                continue
            pid = getattr(pfc, "id", None)
            same_id = bool(fc_id and pid and str(pid) == str(fc_id))
            same_name_pending_id = bool(
                name
                and getattr(pfc, "name", None) == name
                and (not fc_id or not pid)
            )
            if same_id or same_name_pending_id:
                all_parts[i] = p
                replaced = True
                break
        if not replaced:
            all_parts.append(p)


def _flatten_gemini_tool_history(
    contents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert FC/FR turns into plain text so a poisoned transcript can recover.

    Used as a one-shot retry when Gemini rejects tool-turn ordering / signatures.
    Function-call args are omitted so Gemini 3.7 cannot echo ``[called write_file(…)]``
    as the user-visible answer.
    """
    flat: list[dict[str, Any]] = []
    flattened_tool = False
    for turn in contents:
        role = turn.get("role")
        parts = list(turn.get("parts") or [])
        texts: list[str] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if "text" in p and p["text"]:
                texts.append(str(p["text"]))
            elif _part_has_function_call(p):
                flattened_tool = True
                fc = p.get("function_call") or p.get("functionCall") or {}
                texts.append(_flatten_gemini_fc_text(fc))
            elif _part_has_function_response(p):
                flattened_tool = True
                fr = p.get("function_response") or p.get("functionResponse") or {}
                texts.append(_flatten_gemini_fr_text(fr))
        if not texts:
            continue
        # Map model→model, user/FR→user
        out_role = "model" if role == "model" else "user"
        blob = "\n".join(texts)
        if flat and flat[-1]["role"] == out_role:
            flat[-1]["parts"][0]["text"] += "\n" + blob
        else:
            flat.append({"role": out_role, "parts": [{"text": blob}]})
    while flat and flat[0]["role"] == "model":
        flat.pop(0)
    if flattened_tool:
        if flat and flat[-1]["role"] == "user":
            flat[-1]["parts"][0]["text"] += "\n\n" + GEMINI_ANSWER_NUDGE
        else:
            flat.append({"role": "user", "parts": [{"text": GEMINI_ANSWER_NUDGE}]})
    return flat


def _gemini_part_from_block(part: Any) -> dict[str, Any] | None:
    """Convert one canonical content part into a Gemini ``parts`` entry.

    ``text`` → ``{"text"}``; data-URL ``image_url``/``file`` parts →
    ``{"inline_data"}`` with decoded bytes (Gemini accepts image and PDF
    mime types inline). Returns ``None`` for unconvertible parts so callers
    drop them instead of sending an invalid entry.
    """
    if not isinstance(part, dict):
        return None
    ptype = part.get("type")
    if ptype == "text":
        return {"text": part.get("text", "")}
    if ptype == "image_url":
        url = ((part.get("image_url") or {}).get("url")) or ""
    elif ptype == "file":
        url = ((part.get("file") or {}).get("file_data")) or ""
    else:
        return None
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        import base64
        import binascii

        header, b64_str = url[5:].split(";base64,", 1)
        mime = header.split(";", 1)[0].strip() or "application/octet-stream"
        try:
            decoded = base64.b64decode(b64_str)
        except (binascii.Error, ValueError):
            return None
        return {"inline_data": {"mime_type": mime, "data": decoded}}
    return None


def _gemini_38_model(model: str) -> bool:
    # Keep the migration scoped to 3.8 and 4.x (Argon, Carbon); 3.7 and older retain their sampling API.
    return bool(_re.search(r"(?:^|/)gemini-(?:3\.8|4(?:-(?:argon|carbon))?)(?:-|$)", (model or "").lower()))


_GEMINI_BLOCK_FINISH_REASONS = (
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "RECITATION",
    "IMAGE_SAFETY",
)


def _gemini_finish_reason_name(fr: Any) -> str:
    if fr is None:
        return ""
    name = getattr(fr, "name", None)
    if isinstance(name, str) and name:
        return name
    raw = str(fr)
    return raw.split(".", 1)[-1] if "." in raw else raw


def _gemini_prompt_block_reason(resp_or_chunk: Any) -> str | None:
    pf = getattr(resp_or_chunk, "prompt_feedback", None)
    br = getattr(pf, "block_reason", None) if pf is not None else None
    if br is None:
        return None
    raw_name = getattr(br, "name", None)
    if isinstance(raw_name, str):
        name = raw_name
    elif isinstance(br, str):
        name = br
    elif type(br).__module__.startswith("unittest.mock"):
        return None
    else:
        name = str(br)
    cleaned = name.split(".", 1)[-1] if "." in name else name
    if cleaned.upper() in ("", "NONE", "BLOCKED_REASON_UNSPECIFIED", "UNSPECIFIED"):
        return None
    return cleaned


def _gemini_4_fallback_model(model: str) -> str | None:
    m = (model or "").strip().lower()
    if _re.search(r"(?:^|/)gemini-4(?:-(?:argon|carbon))?(?:-|$)", m):
        return "gemini-3.8-flash"
    return None


def _gemini_function_retry_config(config: Any) -> Any:
    """Keep generation/structured-output settings when recovering a tool call."""
    options = {}
    for attr in ("max_output_tokens", "temperature", "system_instruction", "tools",
                 "thinking_config", "response_mime_type", "response_schema", "response_json_schema"):
        value = getattr(config, attr, None)
        if value is not None:
            options[attr] = value
    options["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
    return types.GenerateContentConfig(**options)


def _gemini_thought_tokens(usage: Any) -> int:
    value = getattr(usage, "thoughts_token_count", 0)
    return max(0, value) if isinstance(value, int) else 0


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: EngineConfig):
        if not _HAS_GEMINI:
            raise ImportError("google-genai not installed. Install with: pip install clawagents[gemini]")
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.gemini_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._reasoning_effort = normalize_reasoning_effort(config.reasoning_effort)

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
        _ = session_id  # Gemini has no OpenAI-style prompt_cache_key header.
        # Build a toolCallId → toolName lookup from all assistant messages
        tc_id_to_name: dict[str, str] = {}
        for m in messages:
            if m.role == "assistant" and m.tool_calls_meta:
                for tc in m.tool_calls_meta:
                    tc_id_to_name[tc["id"]] = tc["name"]

        system_parts: list[str] = []
        user_contents: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                elif isinstance(m.content, list):
                    system_parts.extend([p.get("text", "") for p in m.content if p.get("type") == "text"])
            elif m.role == "tool" and m.tool_call_id:
                tool_name = tc_id_to_name.get(m.tool_call_id, "unknown")
                fr_body = _gemini_function_response_body(
                    tool_name, m.content, m.tool_call_id
                )
                user_contents.append({"role": "user", "parts": [{"function_response": fr_body}]})
            elif m.role == "assistant" and m.tool_calls_meta:
                # Prefer preserved gemini_parts (thought_signature + FC ids).
                user_contents.append({
                    "role": "model",
                    "parts": _model_parts_from_tool_meta(
                        m.content,
                        m.tool_calls_meta,
                        gemini_parts=m.gemini_parts,
                    ),
                })
            elif m.role == "assistant" and m.gemini_parts:
                user_contents.append({
                    "role": "model",
                    "parts": _restore_gemini_parts_for_api(list(m.gemini_parts)),
                })
            else:
                role_name = "model" if m.role == "assistant" else "user"
                if isinstance(m.content, str):
                    user_contents.append({"role": role_name, "parts": [{"text": m.content}]})
                elif isinstance(m.content, list):
                    parts2: list[dict[str, Any]] = []
                    for part in m.content:
                        converted = _gemini_part_from_block(part)
                        if converted is not None:
                            parts2.append(converted)
                    if parts2:
                        user_contents.append({"role": role_name, "parts": parts2})

        user_contents = _sanitize_gemini_contents(user_contents)

        # ``__CACHE_BOUNDARY__`` is an Anthropic-only prompt-cache hint; strip
        # it so Gemini never receives the stray internal marker.
        system_instruction = "\n".join(system_parts).replace("__CACHE_BOUNDARY__", "").strip()

        config_opts: dict[str, Any] = {"max_output_tokens": self._max_tokens}
        if _gemini_38_model(self.model):
            # 3.8 rejects sampling controls and minimal thinking. Omitting the
            # level preserves its medium default; shared UI effort maps safely.
            config_opts["max_output_tokens"] = min(self._max_tokens, 65_536)
            effort = getattr(self, "_reasoning_effort", None)
            if effort:
                level = "low" if effort in ("none", "minimal", "low") else (
                    "medium" if effort == "medium" else "high"
                )
                config_opts["thinking_config"] = {"thinking_level": level}
        else:
            config_opts["temperature"] = self._temperature
        if system_instruction:
            config_opts["system_instruction"] = system_instruction
        if tools:
            config_opts["tools"] = _to_gemini_tools(tools)
        schema = getattr(self, "_structured_json_schema", None)
        if isinstance(schema, dict) and schema and not tools:
            try:
                from clawagents.structured_output import gemini_response_schema

                config_opts["response_mime_type"] = "application/json"
                config_opts["response_schema"] = gemini_response_schema(schema)
            except Exception:
                pass
        gemini_config = types.GenerateContentConfig(**config_opts)

        async def _call(contents: list[dict[str, Any]]) -> LLMResponse:
            contents = _scrub_gemini_contents(contents)
            if not on_chunk:
                from clawagents.circuit_breaker import breaker_key as _bk

                return await _with_retry(
                    "gemini",
                    lambda: self._request_once(contents, gemini_config),
                    policy=getattr(self, "retry_policy", None),
                    breaker_tag=_bk("gemini", model=self.model),
                )
            return await self._stream_with_retry(
                contents,
                gemini_config,
                on_chunk,
                cancel_event,
                on_first_token=on_first_token,
            )

        try:
            return await _call(user_contents)
        except Exception as exc:
            if not _is_gemini_history_400(exc):
                raise
            flat = _flatten_gemini_tool_history(user_contents)
            if flat == user_contents or not flat:
                raise
            logger.warning(
                "  [gemini] history 400 (%s) — retrying with flattened tool turns",
                type(exc).__name__,
            )
            return await _call(flat)

    async def _request_once(
        self,
        user_contents: list[dict[str, Any]],
        gemini_config: types.GenerateContentConfig,
        *,
        _malformed_retry: bool = False,
        _block_fallback: bool = False,
    ) -> LLMResponse:
        user_contents = _scrub_gemini_contents(user_contents)
        try:
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_contents,
                config=gemini_config,
            )
        except Exception as exc:
            if not _is_gemini_sdk_shape_error(exc):
                raise
            cleaned = _scrub_gemini_contents(user_contents)
            if cleaned == user_contents:
                raise
            logger.warning("  [gemini] SDK shape error — retrying after scrub")
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=cleaned,
                config=gemini_config,
            )
        fn_calls: list[NativeToolCall] | None = None
        raw_parts = None
        finish_reason = None
        parts = None
        candidates = getattr(resp, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            parts = getattr(candidates[0].content, "parts", None) if candidates[0].content else None
            if parts:
                raw_parts = _serialize_gemini_parts(parts)
                fn_calls = []
                for p in parts:
                    fc = getattr(p, "function_call", None)
                    if fc:
                        import uuid
                        fc_id = getattr(fc, "id", None) or f"gemini_{uuid.uuid4().hex[:8]}"
                        fn_calls.append(NativeToolCall(
                            tool_name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                            tool_call_id=str(fc_id),
                        ))
                if raw_parts and fn_calls:
                    _stamp_function_call_ids(raw_parts, fn_calls)
                if not fn_calls:
                    fn_calls = None
        extracted_text = ""
        if candidates and parts:
            extracted_text = "".join(
                getattr(p, "text", "") for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            )

        fr_str = _gemini_finish_reason_name(finish_reason)
        if not _malformed_retry and "MALFORMED_FUNCTION_CALL" in fr_str and not fn_calls:
            logger.warning("  [gemini] MALFORMED_FUNCTION_CALL detected — retrying with mode=ANY")
            retry_config = _gemini_function_retry_config(gemini_config)
            return await self._request_once(
                user_contents, retry_config, _malformed_retry=True, _block_fallback=_block_fallback
            )

        prompt_block = _gemini_prompt_block_reason(resp)
        is_candidate_block = any(tok in fr_str.upper() for tok in _GEMINI_BLOCK_FINISH_REASONS)
        if not extracted_text and not fn_calls and (prompt_block or is_candidate_block):
            block_label = prompt_block or fr_str
            fallback_model = _gemini_4_fallback_model(self.model)
            if fallback_model and not _block_fallback:
                logger.warning(
                    "  [gemini] %s blocked request (%s) — falling back to %s",
                    self.model, block_label, fallback_model,
                )
                self.model = fallback_model
                return await self._request_once(
                    user_contents, gemini_config, _malformed_retry=_malformed_retry, _block_fallback=True
                )
            raise RuntimeError(
                f"Gemini blocked the request (reason={block_label}). "
                "Try rephrasing sensitive keywords or switching to another model."
            )

        _um = resp.usage_metadata
        _prompt_tokens = (getattr(_um, "prompt_token_count", 0) or 0) if _um else 0
        _thought_tokens = _gemini_thought_tokens(_um)
        _output_tokens = ((getattr(_um, "candidates_token_count", 0) or 0) if _um else 0) + _thought_tokens
        _cache_read = (
            int(getattr(_um, "cached_content_token_count", 0) or 0) if _um else 0
        )
        return LLMResponse(
            content=extracted_text,
            model=self.model,
            # ``tokens_used`` is input+output everywhere else; Gemini used to
            # record output-only (and no prompt), garbling usage accounting.
            tokens_used=_prompt_tokens + _output_tokens,
            prompt_tokens=_prompt_tokens,
            cache_read_tokens=_cache_read,
            reasoning_tokens=_thought_tokens,
            tool_calls=fn_calls,
            gemini_parts=raw_parts,
            finish_reason=fr_str or None,
        )

    async def _stream_with_retry(
        self,
        user_contents: list[dict[str, Any]],
        gemini_config: types.GenerateContentConfig,
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        *,
        on_first_token: Any | None = None,
        _block_fallback: bool = False,
    ) -> LLMResponse:
        user_contents = _scrub_gemini_contents(user_contents)
        last_error: BaseException | None = None
        breaker = _get_stream_breaker("gemini", model=self.model)

        # De-duplicate text across retry attempts (see _StreamReplayGuard).
        replay_guard = _StreamReplayGuard(on_chunk)
        if on_chunk is not None:
            on_chunk = replay_guard.feed

        attempt = 0
        while attempt <= _MAX_RETRIES:
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [gemini] Stream retry %d/%d after %.1fs",
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
            final_tokens = 0
            final_prompt_tokens = 0
            final_thought_tokens = 0
            final_cache_read = 0
            fn_calls: list[NativeToolCall] = []
            all_stream_parts: list[Any] = []
            last_finish_reason: Any = None
            prompt_block_reason: str | None = None
            first_token_fired = False

            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=user_contents,
                    config=gemini_config,
                )

                async for chunk in _stall_guarded_stream(stream, _CHUNK_STALL_S):
                    if cancel_event and cancel_event.is_set():
                        return LLMResponse(
                            content="".join(chunks),
                            model=self.model,
                            tokens_used=final_tokens,
                            prompt_tokens=final_prompt_tokens,
                            reasoning_tokens=final_thought_tokens,
                            cache_read_tokens=final_cache_read,
                            partial=True,
                            tool_calls=fn_calls if fn_calls else None,
                            gemini_parts=_stamp_function_call_ids(
                                _serialize_gemini_parts(all_stream_parts),
                                fn_calls if fn_calls else None,
                            ),
                        )

                    try:
                        _pb = _gemini_prompt_block_reason(chunk)
                        if _pb:
                            prompt_block_reason = _pb
                        _chunk_text_parts: list[str] = []
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for _cand in chunk.candidates:
                                _cand_parts = getattr(getattr(_cand, "content", None), "parts", None)
                                if _cand_parts:
                                    for _p in _cand_parts:
                                        _text = getattr(_p, "text", None)
                                        if _text and not getattr(_p, "thought", False):
                                            _chunk_text_parts.append(_text)
                        if _chunk_text_parts:
                            _joined = "".join(_chunk_text_parts)
                            if not first_token_fired:
                                first_token_fired = True
                                _fire_first_token(on_first_token)
                            chunks.append(_joined)
                            await _invoke_callback(on_chunk, _joined)
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for candidate in chunk.candidates:
                                fr = getattr(candidate, "finish_reason", None)
                                if fr is not None:
                                    last_finish_reason = fr
                                _cand_parts2 = getattr(getattr(candidate, "content", None), "parts", None)
                                if _cand_parts2:
                                    _absorb_gemini_stream_parts(all_stream_parts, _cand_parts2)
                                    for p in _cand_parts2:
                                        fc = getattr(p, "function_call", None)
                                        if fc:
                                            if not first_token_fired:
                                                first_token_fired = True
                                                _fire_first_token(on_first_token)
                                            _upsert_gemini_stream_function_call(fn_calls, fc)
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            _um = chunk.usage_metadata
                            final_prompt_tokens = getattr(_um, "prompt_token_count", 0) or 0
                            final_thought_tokens = _gemini_thought_tokens(_um)
                            final_tokens = final_prompt_tokens + (
                                getattr(_um, "candidates_token_count", 0) or 0
                            ) + final_thought_tokens
                            final_cache_read = int(
                                getattr(_um, "cached_content_token_count", 0) or 0
                            )
                    except Exception:
                        pass  # malformed chunk — skip

                fr_str = _gemini_finish_reason_name(last_finish_reason)
                if "MALFORMED_FUNCTION_CALL" in fr_str and not fn_calls:
                    logger.warning("  [gemini] MALFORMED_FUNCTION_CALL in stream — retrying with mode=ANY (non-stream)")
                    retry_config = _gemini_function_retry_config(gemini_config)
                    return await self._request_once(
                        user_contents, retry_config, _malformed_retry=True, _block_fallback=_block_fallback
                    )

                is_candidate_block = any(tok in fr_str.upper() for tok in _GEMINI_BLOCK_FINISH_REASONS)
                if not chunks and not fn_calls and (prompt_block_reason or is_candidate_block):
                    block_label = prompt_block_reason or fr_str
                    fallback_model = _gemini_4_fallback_model(self.model)
                    if fallback_model and not _block_fallback:
                        logger.warning(
                            "  [gemini] %s blocked stream (%s) — falling back to %s",
                            self.model, block_label, fallback_model,
                        )
                        _record_stream_breaker(breaker, success=True)
                        self.model = fallback_model
                        return await self._stream_with_retry(
                            user_contents,
                            gemini_config,
                            on_chunk,
                            cancel_event,
                            on_first_token=on_first_token,
                            _block_fallback=True,
                        )
                    _record_stream_breaker(breaker, success=True)
                    raise RuntimeError(
                        f"Gemini blocked the request (reason={block_label}). "
                        "Try rephrasing sensitive keywords or switching to another model."
                    )

                _record_stream_breaker(breaker, success=True)
                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    prompt_tokens=final_prompt_tokens,
                    reasoning_tokens=final_thought_tokens,
                    cache_read_tokens=final_cache_read,
                    tool_calls=fn_calls if fn_calls else None,
                    gemini_parts=_stamp_function_call_ids(
                        _serialize_gemini_parts(all_stream_parts),
                        fn_calls if fn_calls else None,
                    ),
                    finish_reason=fr_str or None,
                )

            except Exception as exc:
                last_error = exc
                # Retry retryable mid-stream failures before surfacing a
                # truncated partial; include accumulated tool calls when we do.
                if _is_retryable(exc):
                    _record_stream_breaker(breaker, success=False)
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [gemini] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
                    attempt += 1
                    continue
                if chunks or fn_calls:
                    partial = "".join(chunks)
                    logger.warning(
                        "  [gemini] Stream interrupted after %d chars — returning partial",
                        len(partial),
                    )
                    return LLMResponse(
                        content=partial,
                        model=self.model,
                        tokens_used=final_tokens,
                        prompt_tokens=final_prompt_tokens,
                        reasoning_tokens=final_thought_tokens,
                        partial=True,
                        tool_calls=fn_calls if fn_calls else None,
                        gemini_parts=_stamp_function_call_ids(
                            _serialize_gemini_parts(all_stream_parts),
                            fn_calls if fn_calls else None,
                        ),
                    )
                break

        raise last_error  # type: ignore[misc]
