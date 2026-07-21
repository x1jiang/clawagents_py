from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, TypeVar

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError
try:
    from google import genai
    from google.genai import types
    _HAS_GEMINI = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GEMINI = False

from clawagents.config.config import EngineConfig

logger = logging.getLogger(__name__)

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

T = TypeVar("T")

# ─── Public Types ──────────────────────────────────────────────────────────


class LLMMessage:
    def __init__(
        self,
        role: Literal["system", "user", "assistant", "tool"],
        content: str | list[dict[str, Any]],
        tool_call_id: str | None = None,
        tool_calls_meta: list[dict[str, Any]] | None = None,
        gemini_parts: list[dict[str, Any]] | None = None,
        thinking: str | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id          # For role="tool": the ID this result belongs to
        self.tool_calls_meta = tool_calls_meta    # For role="assistant": list of {id, name, args}
        self.gemini_parts = gemini_parts          # Preserved Gemini response parts (thought/thought_signature)
        self.thinking = thinking                  # Feature H: preserved <think> block content


class NativeToolSchema:
    """Schema for a tool that can be passed to LLM native function calling."""
    __slots__ = ("name", "description", "parameters")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict[str, Any]],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters


class NativeToolCall:
    """A structured tool call returned by the LLM's native function calling."""
    __slots__ = ("tool_name", "args", "tool_call_id")

    def __init__(self, tool_name: str, args: dict[str, Any], tool_call_id: str = ""):
        self.tool_name = tool_name
        self.args = args
        self.tool_call_id = tool_call_id


class LLMResponse:
    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int,
        partial: bool = False,
        tool_calls: list[NativeToolCall] | None = None,
        gemini_parts: list[dict[str, Any]] | None = None,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        prompt_tokens: int = 0,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.partial = partial
        self.tool_calls = tool_calls
        self.gemini_parts = gemini_parts          # Preserved Gemini response parts (thought/thought_signature)
        # Prompt cache tracking (Claude Code pattern)
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens
        self.prompt_tokens = prompt_tokens


OnChunkCallback = (
    Callable[[str], Coroutine[Any, Any, None]] | Callable[[str], None] | None
)


# ─── Feature H: Thinking Token Preservation ────────────────────────────────

import re as _re

_THINK_BLOCK_RE = _re.compile(r"<think>(.*?)</think>", _re.DOTALL)


def strip_thinking_tokens(content: str) -> tuple[str, str | None]:
    """Extract <think>...</think> blocks and return (clean_content, thinking).

    Handles models like Qwen3, DeepSeek that wrap chain-of-thought in <think> tags.
    Returns the content with thinking removed, and the thinking text separately.
    """
    if not content or "<think>" not in content:
        return content, None
    thinking_parts: list[str] = []
    for m in _THINK_BLOCK_RE.finditer(content):
        thinking_parts.append(m.group(1).strip())
    clean = _THINK_BLOCK_RE.sub("", content).strip()
    thinking = "\n".join(thinking_parts) if thinking_parts else None
    return clean, thinking


def rebuild_thinking_content(content: str, thinking: str | None) -> str:
    """Re-attach thinking tokens for models that expect them in conversation history."""
    if not thinking:
        return content
    return f"<think>{thinking}</think>\n{content}"


class LLMProvider(ABC):
    name: str
    # Per-provider RetryPolicy override. When ``None``, the module-level
    # default is used. Set on an instance or subclass to customise behaviour
    # without touching the concrete provider class.
    retry_policy: Any = None

    @abstractmethod
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
        pass

    async def complete(
        self,
        messages: list[LLMMessage],
        *args: Any,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-tool convenience alias for ``chat``.

        Goal / memory-flush / dream helpers historically called ``complete``.
        Providers only implement ``chat``; this default forwards and accepts
        (then ignores) ``stream=`` so those call sites don't AttributeError.
        Non-streaming is the default when ``on_chunk`` is omitted.
        """
        return await self.chat(messages)


# ─── Streaming Robustness Internals ───────────────────────────────────────

_MAX_RETRIES = 3
_INITIAL_DELAY_S = 1.0
_MAX_DELAY_S = 16.0
_CHUNK_STALL_S = 60.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_retryable(err: BaseException) -> bool:
    if isinstance(err, APIStatusError):
        return err.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(err, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(err, Exception):
        msg = str(err).lower()
        return any(
            tok in msg
            for tok in (
                "econnreset", "network", "timeout", "stream stalled",
                "rate limit", "too many requests", "service unavailable",
                "429", "500", "502", "503", "504",
            )
        )
    return False


def _jittered_delay(attempt: int) -> float:
    base = _INITIAL_DELAY_S * (2 ** attempt)
    return min(base + random.random() * base * 0.1, _MAX_DELAY_S)


async def _stall_guarded_stream(
    aiter: AsyncIterator[T],
    timeout_s: float,
) -> AsyncIterator[T]:
    """Yield items from *aiter*, raising TimeoutError if no item arrives
    within *timeout_s* seconds (stall detection)."""
    ait = aiter.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(ait.__anext__(), timeout=timeout_s)
            yield chunk
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            # Re-raise with a message: a bare TimeoutError stringifies to ""
            # so callers logged blank errors and the taxonomy had nothing to
            # classify.
            raise TimeoutError(
                f"LLM stream stalled: no chunk received for {timeout_s:.0f}s"
            ) from None


async def _invoke_callback(
    cb: OnChunkCallback,
    text: str,
) -> None:
    """Call *cb* with *text*, isolating errors so a broken callback
    never kills the stream."""
    if cb is None:
        return
    try:
        if asyncio.iscoroutinefunction(cb):
            await cb(text)
        else:
            cb(text)
    except Exception:
        logger.debug("onChunk callback raised — isolated", exc_info=True)


async def _with_retry(
    tag: str,
    fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    policy: Any = None,
    breaker_tag: str | None = None,
) -> T:
    """Retry ``fn`` with either the legacy heuristic or a :class:`RetryPolicy`.

    When ``policy`` is ``None``, behaviour matches the pre-existing
    ``_is_retryable`` heuristic so providers that don't opt-in keep working.

    When ``provider_circuit_breaker`` is enabled, a per-endpoint breaker admits
    half-open probes. ``BreakerOpen`` waits without burning retry budget.
    ``breaker_tag`` should be a :func:`breaker_key` identity (base_url+model).
    """
    last_error: BaseException | None = None
    breaker = None
    bkey = breaker_tag or tag
    try:
        from clawagents.config.features import is_enabled as _feat_cb

        if _feat_cb("provider_circuit_breaker"):
            from clawagents.circuit_breaker import (
                BreakerOpen,
                Outcome,
                get_provider_breaker,
            )

            breaker = get_provider_breaker(bkey)
    except Exception:
        breaker = None

    async def _admit() -> None:
        if breaker is None:
            return
        from clawagents.circuit_breaker import BreakerOpen

        # Wait for half-open slots without consuming provider retries.
        for _ in range(24):
            try:
                breaker.check()
                return
            except BreakerOpen as open_exc:
                delay = max(0.05, float(open_exc.retry_after) or 0.05)
                logger.warning(
                    "  [%s] circuit breaker open — backing off %.2fs",
                    bkey,
                    delay,
                )
                await asyncio.sleep(delay)
        breaker.check()

    async def _guarded() -> T:
        await _admit()
        try:
            result = await fn()
        except Exception as exc:
            if breaker is not None and _is_retryable(exc):
                from clawagents.circuit_breaker import Outcome

                breaker.record(Outcome.FAILURE)
            raise
        if breaker is not None:
            from clawagents.circuit_breaker import Outcome

            breaker.record(Outcome.SUCCESS)
        return result

    if policy is None:
        attempt = 0
        while attempt <= _MAX_RETRIES:
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [%s] Retry %d/%d after %.1fs",
                    tag, attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            try:
                return await _guarded()
            except Exception as exc:
                last_error = exc
                if breaker is not None:
                    try:
                        from clawagents.circuit_breaker import BreakerOpen as _BO

                        if isinstance(exc, _BO):
                            # Already waited in _admit; do not burn a retry slot.
                            await asyncio.sleep(max(0.05, float(exc.retry_after) or 0.05))
                            continue
                    except Exception:
                        pass
                if not _is_retryable(exc):
                    break
                attempt += 1
        if last_error is not None:
            try:
                from clawagents.circuit_breaker import BreakerOpen as _BO

                if isinstance(last_error, _BO):
                    raise RuntimeError(
                        f"Provider circuit breaker open for {bkey} — "
                        f"endpoint failing repeatedly "
                        f"(retry after ~{float(last_error.retry_after) or 0:.1f}s)."
                    ) from last_error
            except RuntimeError:
                raise
            except Exception:
                pass
        raise last_error  # type: ignore[misc]

    # Policy-driven path.
    max_attempts = max(1, int(getattr(policy, "max_retries", _MAX_RETRIES)) + 1)
    attempt = 0
    while attempt < max_attempts:
        try:
            return await _guarded()
        except Exception as exc:
            last_error = exc
            try:
                from clawagents.circuit_breaker import BreakerOpen as _BO

                if isinstance(exc, _BO):
                    await asyncio.sleep(max(0.05, float(exc.retry_after) or 0.05))
                    continue  # do not burn attempt
            except Exception:
                pass
            attempt += 1
            try:
                descriptor = policy.classify(exc)
                should = policy.should_retry(exc, attempt, descriptor=descriptor)
            except Exception:
                descriptor = None
                should = _is_retryable(exc)
            if not should or attempt >= max_attempts:
                break
            try:
                delay = policy.compute_delay(
                    attempt,
                    retry_after=getattr(descriptor, "retry_after", None),
                )
            except Exception:
                delay = _jittered_delay(attempt - 1)
            logger.warning(
                "  [%s] Retry %d/%d after %.1fs (policy=%s)",
                tag, attempt, max_attempts - 1, delay,
                getattr(descriptor, "error_class", "?"),
            )
            await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


def _get_stream_breaker(
    tag: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
) -> Any | None:
    """Return a per-endpoint breaker for streaming paths, or None when disabled."""
    try:
        from clawagents.config.features import is_enabled as _feat_cb
        from clawagents.circuit_breaker import breaker_key, get_provider_breaker

        if not _feat_cb("provider_circuit_breaker"):
            return None
        return get_provider_breaker(breaker_key(tag, base_url=base_url, model=model))
    except Exception:
        return None


async def _admit_stream_breaker(breaker: Any) -> None:
    if breaker is None:
        return
    from clawagents.circuit_breaker import BreakerOpen

    for _ in range(24):
        try:
            breaker.check()
            return
        except BreakerOpen as open_exc:
            await asyncio.sleep(max(0.05, float(open_exc.retry_after) or 0.05))
    try:
        breaker.check()
    except BreakerOpen as open_exc:
        raise RuntimeError(
            f"Provider circuit breaker open — pausing requests "
            f"(retry after ~{float(open_exc.retry_after) or 0:.1f}s). "
            f"Underlying endpoint has been failing repeatedly."
        ) from open_exc


def _record_stream_breaker(breaker: Any, *, success: bool) -> None:
    if breaker is None:
        return
    try:
        from clawagents.circuit_breaker import Outcome

        breaker.record(Outcome.SUCCESS if success else Outcome.FAILURE)
    except Exception:
        pass


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


def _openai_cached_tokens(usage: Any) -> int:
    """Read prompt-cache hits from an OpenAI usage object, defaulting to 0.

    ``usage.prompt_tokens_details.cached_tokens`` reports the portion of the
    prompt served from OpenAI's automatic prompt cache (billed at a discount).
    """
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    try:
        return int(getattr(details, "cached_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _parse_openai_tool_calls(
    tool_calls: Any,
) -> list[NativeToolCall] | None:
    """Extract NativeToolCall list from OpenAI response tool_calls (handles function vs custom union)."""
    if not tool_calls:
        return None
    result: list[NativeToolCall] = []
    for tc in tool_calls:
        if getattr(tc, "type", None) == "function":
            fn = tc.function
            result.append(NativeToolCall(
                tool_name=fn.name,
                args=_repair_json(fn.arguments or "{}"),
                tool_call_id=getattr(tc, "id", "") or "",
            ))
    return result if result else None


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
    for prefix in ("openai.", "azure.", "mantle."):
        if m.startswith(prefix):
            m = m[len(prefix) :]
    return m


def openai_model_rejects_temperature(model: str) -> bool:
    """True when OpenAI / Mantle Responses reject an explicit ``temperature``.

    GPT-5.5 / 5.6 (incl. ``openai.gpt-5.6-luna``), o-series, and similar
    reasoning models return 400 if ``temperature`` is present. Omit the field.
    """
    m = _bare_openai_model_id(model)
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith(("gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6")):
        return True
    if m.startswith("gpt-5") and "codex" in m:
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

    Claude Opus 4.7+ (including Mantle ``anthropic.claude-opus-4-8``) return
    HTTP 400 ``temperature is deprecated for this model`` if those fields are
    present. Omit them and guide behavior via prompting instead.
    """
    import re

    m = (model or "").strip().lower().replace("_", "-")
    # Collapse dotted minors: opus-4.8 → opus-4-8; keep geo/FM prefixes intact.
    m = re.sub(r"(opus-4)\.(\d+)", r"\1-\2", m)
    hit = re.search(r"opus-4-(\d+)", m)
    if hit:
        return int(hit.group(1)) >= 7
    # Future Opus 5+ generations inherit the same restriction.
    if re.search(r"opus-([5-9]|[1-9]\d+)(?:-|\b)", m):
        return True
    return False


def _chat_completions_needs_reasoning_none(model: str) -> bool:
    """True when Chat Completions rejects tools + default reasoning_effort.

    GPT-5.5 / GPT-5.6 default to a non-``none`` reasoning effort. On
    ``/v1/chat/completions``, that combination with function tools returns
    HTTP 400 ("use /v1/responses or set reasoning_effort to 'none'"). Prefer
    Responses for those models; this force-none is only for the Chat Completions
    fallback path.
    """
    m = _bare_openai_model_id(model)
    return m.startswith("gpt-5.5") or m.startswith("gpt-5.6")


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


def model_supports_reasoning_effort(model: str) -> bool:
    """Heuristic: models that accept ``reasoning_effort`` / Responses ``reasoning``."""
    m = _bare_openai_model_id(model)
    if not m:
        return False
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith(("gpt-5.5", "gpt-5.6")):
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
    if m.startswith(("gpt-5.5", "gpt-5.6")):
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
    """Chat Completions reasoning_effort (forces none for GPT-5.5/5.6 + tools)."""
    effort = normalize_reasoning_effort(preferred)
    if effort:
        kwargs["reasoning_effort"] = effort
    # Chat Completions + tools on GPT-5.5/5.6 still requires none.
    if has_tools and _chat_completions_needs_reasoning_none(model):
        kwargs["reasoning_effort"] = "none"


def _apply_responses_reasoning(
    kwargs: dict[str, Any],
    *,
    preferred: str | None = None,
) -> None:
    """Responses API uses ``reasoning={"effort": ...}`` (tools keep effort)."""
    effort = normalize_reasoning_effort(preferred)
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
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(m.get("tool_call_id") or ""),
                    "output": content or "",
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
) -> tuple[str, list[NativeToolCall] | None, int, int, int]:
    """Return (text, tool_calls, total_tokens, prompt_tokens, cached_tokens)."""
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
    cached = 0
    if usage:
        details = getattr(usage, "input_tokens_details", None)
        try:
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        except (TypeError, ValueError):
            cached = 0
    return text, (calls or None), total, prompt, cached


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


class OpenAIProvider(LLMProvider):
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
            try:
                import httpx
                self._http_client = httpx.AsyncClient(verify=False, timeout=120.0)
            except ImportError:
                self._http_client = None

        if is_azure and api_version and base_url:
            try:
                from openai import AsyncAzureOpenAI
                azure_kwargs: dict[str, Any] = {
                    "api_key": api_key,
                    "azure_endpoint": base_url,
                    "api_version": api_version,
                }
                if self._http_client is not None:
                    azure_kwargs["http_client"] = self._http_client
                self.client = AsyncAzureOpenAI(**azure_kwargs)
            except ImportError:
                client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
                if self._http_client is not None:
                    client_kwargs["http_client"] = self._http_client
                self.client = AsyncOpenAI(**client_kwargs)
        else:
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            if self._http_client is not None:
                client_kwargs["http_client"] = self._http_client
            self.client = AsyncOpenAI(**client_kwargs)

        self.model = config.openai_model
        self._max_tokens = config.max_tokens
        self._temperature = _resolve_temperature(config.openai_model, config.temperature)
        self._reasoning_effort = normalize_reasoning_effort(
            getattr(config, "reasoning_effort", None) or None
        )
        self._base_url = base_url
        self._api_type = "azure" if is_azure else api_type
        self._wire_api = _normalize_wire_api(getattr(config, "openai_wire_api", None))
        # Sticky fallback when Responses is missing — never when wire_api forces it.
        self._force_chat_completions = False

    def _should_use_responses(self, has_tools: bool) -> bool:
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
        oai_tools = _to_openai_tools(tools) if tools else None

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
                if (
                    _is_responses_unsupported(exc)
                    and self._wire_api != "responses"
                ):
                    logger.warning(
                        "  [openai] Responses API unavailable (%s) — "
                        "falling back to Chat Completions",
                        type(exc).__name__,
                    )
                    self._force_chat_completions = True
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
        create_kwargs: dict[str, Any] = dict(kwargs)
        if affinity.get("extra_headers"):
            create_kwargs["extra_headers"] = affinity["extra_headers"]
        resp = await self.client.chat.completions.create(**create_kwargs)
        _prompt_tokens = (resp.usage.prompt_tokens or 0) if resp.usage else 0
        _cached_tokens = _openai_cached_tokens(resp.usage)
        if not resp.choices:
            # Azure content filters and some OpenAI-compatible proxies return
            # 200 with an empty ``choices`` array; don't IndexError on it.
            return LLMResponse(content="", model=self.model,
                               tokens_used=resp.usage.total_tokens if resp.usage else 0,
                               prompt_tokens=_prompt_tokens,
                               cache_read_tokens=_cached_tokens,
                               partial=True)
        msg = resp.choices[0].message
        native_calls = _parse_openai_tool_calls(getattr(msg, "tool_calls", None))
        return LLMResponse(
            content=msg.content or "",
            model=self.model,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
            prompt_tokens=_prompt_tokens,
            cache_read_tokens=_cached_tokens,
            tool_calls=native_calls,
        )

    def _responses_kwargs(
        self,
        messages: list[dict[str, Any]],
        oai_tools: list[dict[str, Any]] | None,
        *,
        stream: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        instructions, input_items = _messages_to_responses_input(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self._max_tokens,
            "store": False,
        }
        _with_temperature(kwargs, self.model, self._temperature)
        if instructions:
            kwargs["instructions"] = instructions
        resp_tools = _chat_tools_to_responses_tools(oai_tools)
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

            chunks: list[str] = []
            final_tokens = 0
            final_prompt_tokens = 0
            final_cached_tokens = 0
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
                for _key in sorted(tools_accumulation.keys()):
                    _fn = tools_accumulation[_key]
                    if not _fn.get("name"):
                        continue
                    calls.append(NativeToolCall(
                        tool_name=_fn["name"],
                        args=_repair_json(_fn["arguments"] or "{}"),
                        tool_call_id=_fn.get("id", ""),
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
                        elif etype == "response.completed":
                            resp = getattr(event, "response", None)
                            if resp is not None:
                                _t, _c, total, prompt, cached = _parse_responses_result(resp)
                                final_tokens = total
                                final_prompt_tokens = prompt
                                final_cached_tokens = cached
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
                        "  [openai-responses] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
                    attempt += 1
                    continue
                if chunks or tools_accumulation:
                    partial = "".join(chunks)
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
                        partial=True,
                        tool_calls=_accumulated_calls(),
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

            chunks: list[str] = []
            final_tokens = 0
            final_prompt_tokens = 0
            final_cached_tokens = 0
            tools_accumulation: dict[int, dict[str, Any]] = {}

            def _accumulated_calls() -> list[NativeToolCall] | None:
                if not tools_accumulation:
                    return None
                calls: list[NativeToolCall] = []
                for _idx in sorted(tools_accumulation.keys()):
                    _fn = tools_accumulation[_idx]
                    calls.append(NativeToolCall(
                        tool_name=_fn["name"],
                        args=_repair_json(_fn["arguments"] or "{}"),
                        tool_call_id=_fn.get("id", ""),
                    ))
                return calls

            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": self._max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
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
                create_kwargs = dict(kwargs)
                if affinity.get("extra_headers"):
                    create_kwargs["extra_headers"] = affinity["extra_headers"]
                stream = await self.client.chat.completions.create(**create_kwargs)
                first_token_fired = False

                async for chunk in _stall_guarded_stream(stream, _CHUNK_STALL_S):
                    if cancel_event and cancel_event.is_set():
                        await stream.close()
                        return LLMResponse(
                            content="".join(chunks),
                            model=self.model,
                            tokens_used=final_tokens,
                            prompt_tokens=final_prompt_tokens,
                            cache_read_tokens=final_cached_tokens,
                            partial=True,
                            tool_calls=_accumulated_calls(),
                        )

                    try:
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                text = delta.content
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                chunks.append(text)
                                await _invoke_callback(on_chunk, text)
                            
                            if getattr(delta, "tool_calls", None):
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                for tc in delta.tool_calls:
                                    idx = tc.index
                                    if idx not in tools_accumulation:
                                        tools_accumulation[idx] = {"id": "", "name": "", "arguments": ""}
                                    if getattr(tc, "id", None):
                                        tools_accumulation[idx]["id"] = tc.id
                                    if getattr(tc, "function", None):
                                        if tc.function.name:
                                            tools_accumulation[idx]["name"] += tc.function.name
                                        if tc.function.arguments:
                                            tools_accumulation[idx]["arguments"] += tc.function.arguments

                        if chunk.usage:
                            final_tokens = chunk.usage.total_tokens
                            final_prompt_tokens = chunk.usage.prompt_tokens or 0
                            final_cached_tokens = _openai_cached_tokens(chunk.usage)
                    except Exception:
                        pass  # malformed chunk — skip

                _record_stream_breaker(breaker, success=True)
                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    prompt_tokens=final_prompt_tokens,
                    cache_read_tokens=final_cached_tokens,
                    tool_calls=_accumulated_calls(),
                )

            except Exception as exc:
                last_error = exc
                if _is_retryable(exc):
                    _record_stream_breaker(breaker, success=False)
                # A mid-stream exception used to return the truncated text as a
                # non-retried "final" answer. Retry retryable errors first; only
                # surface a partial (now including any accumulated tool calls)
                # when retries are exhausted or the error is not retryable.
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [openai] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
                    attempt += 1
                    continue
                if chunks or tools_accumulation:
                    partial = "".join(chunks)
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
                        partial=True,
                        tool_calls=_accumulated_calls(),
                    )
                break

        raise last_error  # type: ignore[misc]


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
                    fr: dict[str, Any] = {
                        "function_response": {
                            "name": name,
                            "response": {
                                "result": "[tool call cancelled or skipped before a result was recorded]",
                            },
                        }
                    }
                    if fc.get("id"):
                        fr["function_response"]["id"] = fc["id"]
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
                    body = dict(body)
                    body["id"] = fc["id"]
                    fr["function_response"] = body
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
    return out


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


def _flatten_gemini_tool_history(
    contents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert FC/FR turns into plain text so a poisoned transcript can recover.

    Used as a one-shot retry when Gemini rejects tool-turn ordering / signatures.
    """
    flat: list[dict[str, Any]] = []
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
                fc = p.get("function_call") or p.get("functionCall") or {}
                texts.append(f"[called {fc.get('name', 'tool')}({fc.get('args') or {}})]")
            elif _part_has_function_response(p):
                fr = p.get("function_response") or p.get("functionResponse") or {}
                resp = fr.get("response")
                texts.append(f"[result {fr.get('name', 'tool')}: {resp}]")
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


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: EngineConfig):
        if not _HAS_GEMINI:
            raise ImportError("google-genai not installed. Install with: pip install clawagents[gemini]")
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.gemini_model
        self._max_tokens = config.max_tokens
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
                fr_body: dict[str, Any] = {
                    "name": tool_name,
                    "response": {"result": m.content},
                    # Gemini 3 pairs FR to FC by id — must echo the call id.
                    "id": m.tool_call_id,
                }
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

        config_opts: dict[str, Any] = {
            "max_output_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
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
    ) -> LLMResponse:
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_contents,
            config=gemini_config,
        )
        fn_calls: list[NativeToolCall] | None = None
        raw_parts = None
        finish_reason = None
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

        fr_str = str(finish_reason) if finish_reason else ""
        if not _malformed_retry and "MALFORMED_FUNCTION_CALL" in fr_str and not fn_calls:
            logger.warning("  [gemini] MALFORMED_FUNCTION_CALL detected — retrying with mode=ANY")
            retry_opts: dict[str, Any] = {}
            for attr in ("max_output_tokens", "temperature", "system_instruction", "tools"):
                val = getattr(gemini_config, attr, None)
                if val is not None:
                    retry_opts[attr] = val
            retry_opts["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
            retry_config = types.GenerateContentConfig(**retry_opts)
            return await self._request_once(user_contents, retry_config, _malformed_retry=True)

        _um = resp.usage_metadata
        _prompt_tokens = (getattr(_um, "prompt_token_count", 0) or 0) if _um else 0
        _output_tokens = (getattr(_um, "candidates_token_count", 0) or 0) if _um else 0
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
            tool_calls=fn_calls,
            gemini_parts=raw_parts,
        )

    async def _stream_with_retry(
        self,
        user_contents: list[dict[str, Any]],
        gemini_config: types.GenerateContentConfig,
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        *,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None
        breaker = _get_stream_breaker("gemini", model=self.model)

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

            chunks: list[str] = []
            final_tokens = 0
            final_prompt_tokens = 0
            final_cache_read = 0
            fn_calls: list[NativeToolCall] = []
            all_stream_parts: list[Any] = []
            last_finish_reason: Any = None
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
                            cache_read_tokens=final_cache_read,
                            partial=True,
                            tool_calls=fn_calls if fn_calls else None,
                            gemini_parts=_stamp_function_call_ids(
                                _serialize_gemini_parts(all_stream_parts),
                                fn_calls if fn_calls else None,
                            ),
                        )

                    try:
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
                                    for p in _cand_parts2:
                                        all_stream_parts.append(p)
                                        fc = getattr(p, "function_call", None)
                                        if fc:
                                            import uuid
                                            if not first_token_fired:
                                                first_token_fired = True
                                                _fire_first_token(on_first_token)
                                            fc_id = getattr(fc, "id", None) or f"gemini_{uuid.uuid4().hex[:8]}"
                                            fn_calls.append(NativeToolCall(
                                                tool_name=fc.name,
                                                args=dict(fc.args) if fc.args else {},
                                                tool_call_id=str(fc_id),
                                            ))
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            _um = chunk.usage_metadata
                            final_prompt_tokens = getattr(_um, "prompt_token_count", 0) or 0
                            final_tokens = final_prompt_tokens + (
                                getattr(_um, "candidates_token_count", 0) or 0
                            )
                            final_cache_read = int(
                                getattr(_um, "cached_content_token_count", 0) or 0
                            )
                    except Exception:
                        pass  # malformed chunk — skip

                fr_str = str(last_finish_reason) if last_finish_reason else ""
                if "MALFORMED_FUNCTION_CALL" in fr_str and not fn_calls:
                    logger.warning("  [gemini] MALFORMED_FUNCTION_CALL in stream — retrying with mode=ANY (non-stream)")
                    retry_opts: dict[str, Any] = {}
                    for attr in ("max_output_tokens", "temperature", "system_instruction", "tools"):
                        val = getattr(gemini_config, attr, None)
                        if val is not None:
                            retry_opts[attr] = val
                    retry_opts["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
                    retry_config = types.GenerateContentConfig(**retry_opts)
                    return await self._request_once(user_contents, retry_config, _malformed_retry=True)

                _record_stream_breaker(breaker, success=True)
                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    prompt_tokens=final_prompt_tokens,
                    cache_read_tokens=final_cache_read,
                    tool_calls=fn_calls if fn_calls else None,
                    gemini_parts=_stamp_function_call_ids(
                        _serialize_gemini_parts(all_stream_parts),
                        fn_calls if fn_calls else None,
                    ),
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
                        partial=True,
                        tool_calls=fn_calls if fn_calls else None,
                        gemini_parts=_stamp_function_call_ids(
                            _serialize_gemini_parts(all_stream_parts),
                            fn_calls if fn_calls else None,
                        ),
                    )
                break

        raise last_error  # type: ignore[misc]


# ─── Anthropic Provider ───────────────────────────────────────────────────

try:
    import anthropic as _anthropic_mod
    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic_mod = None  # type: ignore
    _HAS_ANTHROPIC = False


def _anthropic_message_content(content: Any) -> Any:
    """Shape a user/assistant message's content for Anthropic's Messages API.

    Anthropic has no ``image_url`` content type — convert the canonical
    OpenAI-style image blocks (used for user-message images) into
    ``image``/``source`` blocks. Without this an attached image reaches Claude
    as an invalid block and the request 400s. Non-list content and non-image
    blocks pass through unchanged; malformed image_url parts are dropped.
    """
    if not isinstance(content, list):
        return content
    if not any(
        isinstance(p, dict) and p.get("type") in ("image_url", "file") for p in content
    ):
        return content
    from clawagents.media.documents import file_part_to_anthropic_block
    from clawagents.media.images import image_url_to_anthropic_block

    converted: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            block = image_url_to_anthropic_block(part)
            if block is not None:
                converted.append(block)
        elif isinstance(part, dict) and part.get("type") == "file":
            block = file_part_to_anthropic_block(part)
            if block is not None:
                converted.append(block)
        else:
            converted.append(part)
    return converted


def _apply_conversation_cache_breakpoints(api_messages: list[dict[str, Any]]) -> None:
    """Mark the stable conversation prefix for Anthropic ephemeral prompt caching.

    Places a ``cache_control`` breakpoint on the last content block immediately
    before the final substantive user turn (typically the volatile tail).
    """
    if len(api_messages) < 2:
        return

    last_user_idx: int | None = None
    for i in range(len(api_messages) - 1, -1, -1):
        msg = api_messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            last_user_idx = i
            break
        if isinstance(content, list):
            has_text = any(
                isinstance(block, dict)
                and block.get("type") == "text"
                and str(block.get("text", "")).strip()
                for block in content
            )
            if has_text:
                last_user_idx = i
                break

    if last_user_idx is None or last_user_idx <= 0:
        return

    boundary_idx = last_user_idx - 1
    msg = api_messages[boundary_idx]
    content = msg.get("content")
    marker = {"cache_control": {"type": "ephemeral"}}

    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, **marker}]
    elif isinstance(content, list) and content:
        blocks = [dict(block) if isinstance(block, dict) else block for block in content]
        last = blocks[-1]
        if isinstance(last, dict):
            last = dict(last)
            last["cache_control"] = {"type": "ephemeral"}
            blocks[-1] = last
        msg["content"] = blocks


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: EngineConfig):
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Install with: pip install clawagents[anthropic]"
            )
        client_kwargs: dict[str, Any] = {"api_key": config.anthropic_api_key}
        base = (getattr(config, "anthropic_base_url", None) or "").strip()
        if base:
            # SDK appends /v1/messages — Mantle expects …/anthropic/v1/messages.
            client_kwargs["base_url"] = base.rstrip("/")
        self.client = _anthropic_mod.AsyncAnthropic(**client_kwargs)
        self.model = config.anthropic_model
        self._max_tokens = config.max_tokens
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
        system_parts = []
        api_messages = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content if isinstance(m.content, str) else str(m.content))
            elif m.role == "tool" and m.tool_call_id:
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                # Coalesce consecutive tool results into ONE user message —
                # Anthropic requires every tool_result block answering a single
                # assistant turn (parallel tool calls) to be in the same user
                # message, otherwise the API rejects the transcript.
                prev = api_messages[-1] if api_messages else None
                if (
                    prev is not None
                    and prev.get("role") == "user"
                    and isinstance(prev.get("content"), list)
                    and prev["content"]
                    and isinstance(prev["content"][0], dict)
                    and prev["content"][0].get("type") == "tool_result"
                ):
                    prev["content"].append(block)
                else:
                    api_messages.append({"role": "user", "content": [block]})
            elif m.role == "assistant" and m.tool_calls_meta:
                content_blocks = []
                if m.content:
                    content_blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls_meta:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
            else:
                role = "assistant" if m.role == "assistant" else "user"
                api_messages.append(
                    {"role": role, "content": _anthropic_message_content(m.content)}
                )

        try:
            from clawagents.config.features import is_enabled as _feat_cache

            if _feat_cache("cache_boundary"):
                _apply_conversation_cache_breakpoints(api_messages)
        except Exception:
            pass

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_parts:
            joined = "\n".join(system_parts)
            # Feature: Cache boundary optimization
            # Split on __CACHE_BOUNDARY__ marker to create static (cached) + dynamic blocks
            if "__CACHE_BOUNDARY__" in joined:
                static_part, dynamic_part = joined.split("__CACHE_BOUNDARY__", 1)
                system_blocks = [
                    {"type": "text", "text": static_part.strip(), "cache_control": {"type": "ephemeral"}},
                ]
                if dynamic_part.strip():
                    system_blocks.append({"type": "text", "text": dynamic_part.strip()})
                kwargs["system"] = system_blocks
            else:
                kwargs["system"] = joined
        # Send temperature whenever it's set (including 0). Gating on ``> 0``
        # dropped ``temperature=0`` — the config default — so Anthropic silently
        # sampled at the API default of 1.0 while OpenAI/Gemini honoured 0,
        # making "temperature: 0" runs non-deterministic only on Claude.
        # Opus 4.7+ reject the field entirely (400) — omit it for those models.
        if (
            not anthropic_model_rejects_sampling_params(self.model)
            and self._temperature is not None
            and self._temperature >= 0
        ):
            kwargs["temperature"] = self._temperature
        schema = getattr(self, "_structured_json_schema", None)
        if isinstance(schema, dict) and schema and not tools:
            # Anthropic structured output suppresses tools — only apply when
            # this turn is tool-free (matches Grok shell StructuredOutput tool path).
            try:
                from clawagents.structured_output import anthropic_output_format

                kwargs["output_config"] = {"format": anthropic_output_format(schema)}
            except Exception:
                pass
        if tools:
            from clawagents.providers.tool_schema import emit_openai_schema_node

            kwargs["tools"] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            k: emit_openai_schema_node(v)
                            for k, v in s.parameters.items()
                            if isinstance(v, dict)
                        },
                        "required": [
                            k for k, v in s.parameters.items() if v.get("required")
                        ],
                    },
                }
                for s in tools
            ]

        if not on_chunk:
            from clawagents.circuit_breaker import breaker_key as _bk

            return await _with_retry(
                "anthropic",
                lambda: self._request_once(kwargs),
                policy=getattr(self, "retry_policy", None),
                breaker_tag=_bk("anthropic", model=self.model),
            )
        return await self._stream_with_retry(
            kwargs,
            on_chunk,
            cancel_event,
            on_first_token=on_first_token,
        )

    async def _request_once(self, kwargs: dict[str, Any]) -> LLMResponse:
        resp = await self.client.messages.create(**kwargs)
        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(NativeToolCall(
                    tool_name=block.name,
                    args=dict(block.input) if block.input else {},
                    tool_call_id=block.id,
                ))

        usage = resp.usage
        uncached = getattr(usage, "input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        prompt_total = uncached + cache_creation + cache_read
        return LLMResponse(
            content="".join(text_parts),
            model=self.model,
            tokens_used=(prompt_total + usage.output_tokens) if usage else 0,
            tool_calls=tool_calls if tool_calls else None,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            prompt_tokens=prompt_total,
        )

    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        *,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None
        breaker = _get_stream_breaker(
            "anthropic",
            model=getattr(self, "model", None) or kwargs.get("model"),
        )

        attempt = 0
        while attempt <= _MAX_RETRIES:
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning("  [anthropic] Retry %d/%d after %.1fs", attempt, _MAX_RETRIES, delay)
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

            chunks: list[str] = []
            tool_calls: list[NativeToolCall] = []
            current_tool: dict[str, Any] | None = None
            output_tokens = 0
            cache_creation = 0
            cache_read = 0
            prompt_tokens = 0
            first_token_fired = False

            try:
                async with self.client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if cancel_event and cancel_event.is_set():
                            return LLMResponse(
                                content="".join(chunks), model=self.model,
                                tokens_used=prompt_tokens + cache_creation + cache_read + output_tokens,
                                prompt_tokens=prompt_tokens + cache_creation + cache_read,
                                partial=True,
                                tool_calls=tool_calls if tool_calls else None,
                                cache_creation_tokens=cache_creation,
                                cache_read_tokens=cache_read,
                            )

                        if event.type == "message_start" and hasattr(event, "message"):
                            u = getattr(event.message, "usage", None)
                            if u:
                                prompt_tokens = getattr(u, "input_tokens", 0) or 0
                                cache_creation = getattr(u, "cache_creation_input_tokens", 0) or 0
                                cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                        elif event.type == "content_block_start":
                            if hasattr(event.content_block, "type"):
                                if event.content_block.type == "tool_use":
                                    if not first_token_fired:
                                        first_token_fired = True
                                        _fire_first_token(on_first_token)
                                    current_tool = {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "input_json": "",
                                    }
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                chunks.append(event.delta.text)
                                await _invoke_callback(on_chunk, event.delta.text)
                            elif hasattr(event.delta, "partial_json") and current_tool:
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                current_tool["input_json"] += event.delta.partial_json
                        elif event.type == "content_block_stop":
                            if current_tool:
                                tool_calls.append(NativeToolCall(
                                    tool_name=current_tool["name"],
                                    args=_repair_json(current_tool["input_json"] or "{}"),
                                    tool_call_id=current_tool["id"],
                                ))
                                current_tool = None
                        elif event.type == "message_delta":
                            if hasattr(event.usage, "output_tokens"):
                                output_tokens = event.usage.output_tokens

                _record_stream_breaker(breaker, success=True)
                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    # input+output, matching the non-streaming path (it used to
                    # report output-only, understating usage by the prompt size).
                    tokens_used=prompt_tokens + cache_creation + cache_read + output_tokens,
                    tool_calls=tool_calls if tool_calls else None,
                    cache_creation_tokens=cache_creation,
                    cache_read_tokens=cache_read,
                    prompt_tokens=prompt_tokens + cache_creation + cache_read,
                )

            except Exception as exc:
                last_error = exc
                # Retry retryable mid-stream failures before surfacing a
                # truncated partial; include accumulated tool calls when we do.
                if _is_retryable(exc):
                    _record_stream_breaker(breaker, success=False)
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [anthropic] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
                    attempt += 1
                    continue
                if chunks or tool_calls:
                    return LLMResponse(
                        content="".join(chunks), model=self.model,
                        tokens_used=prompt_tokens + cache_creation + cache_read + output_tokens,
                        prompt_tokens=prompt_tokens + cache_creation + cache_read,
                        partial=True,
                        tool_calls=tool_calls if tool_calls else None,
                        cache_creation_tokens=cache_creation,
                        cache_read_tokens=cache_read,
                    )
                break

        raise last_error  # type: ignore[misc]


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
            import boto3  # type: ignore
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
        self._max_tokens = config.max_tokens
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
            await on_chunk(content)
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


def _looks_like_ollama(model_name: str) -> bool:
    """Return True if *model_name* looks like an Ollama/local model tag.

    Use the explicit ``ollama/<tag>`` form to bypass heuristics.
    """
    lower = model_name.lower()
    return any(lower.startswith(p) for p in _OLLAMA_PREFIXES)


def _is_mantle_url(url: str | None) -> bool:
    """True for Amazon Bedrock Mantle (OneHUB) OpenAI-compatible hosts."""
    return bool(url) and "bedrock-mantle." in (url or "").lower()


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


def mantle_openai_base_url(url: str) -> str:
    """Mantle OpenAI Responses root (client appends ``/v1/responses``)."""
    origin = _mantle_origin(url)
    return f"{origin}/openai" if origin else ""


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
    return any(token in m for token in ("gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6"))


def _mantle_openai_model_id(model: str) -> str:
    """Ensure Mantle Responses models use the ``openai.`` catalog prefix."""
    m = (model or "").strip()
    if not m:
        return m
    if m.lower().startswith("openai."):
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
    lower = model_name.lower()
    kind = classify_model(
        ref.raw or model_name,
        base_url=config.openai_base_url,
        provider_hint=provider_hint or ref.prefix_hint,
    )

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
    # anthropic.* → /anthropic/v1/messages; openai.gpt-5.* → /openai/v1/responses;
    # chat-ok catalog (gpt-oss, deepseek, qwen, …) → /v1/chat/completions.
    if config.openai_base_url and _is_mantle_url(config.openai_base_url):
        mantle_key = config.openai_api_key or config.anthropic_api_key
        if is_mantle_anthropic_model(model_name):
            if mantle_key:
                config.anthropic_api_key = mantle_key
            config.anthropic_model = model_name
            config.anthropic_base_url = mantle_anthropic_base_url(config.openai_base_url)
            return AnthropicProvider(config)
        if is_mantle_openai_responses_model(model_name):
            rewritten = mantle_openai_base_url(config.openai_base_url)
            if rewritten:
                config.openai_base_url = rewritten
            if not config.openai_api_key and mantle_key:
                config.openai_api_key = mantle_key
            if not config.openai_api_key:
                config.openai_api_key = "bedrock"
            # These models reject /v1/chat/completions even when wire_api was
            # saved as chat_completions from an older Mantle default.
            config.openai_wire_api = "responses"
            # Mantle catalog requires ``openai.gpt-5.6-*`` (bare ids 404 /v1).
            config.openai_model = _mantle_openai_model_id(model_name)
            return OpenAIProvider(config)
        # Chat-completions catalog: keep …/v1 (or normalize to it).
        if _normalize_wire_api(config.openai_wire_api) == "auto":
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
