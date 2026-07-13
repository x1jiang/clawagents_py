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
    ) -> LLMResponse:
        pass


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
) -> T:
    """Retry ``fn`` with either the legacy heuristic or a :class:`RetryPolicy`.

    When ``policy`` is ``None``, behaviour matches the pre-existing
    ``_is_retryable`` heuristic so providers that don't opt-in keep working.
    """
    last_error: BaseException | None = None

    if policy is None:
        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [%s] Retry %d/%d after %.1fs",
                    tag, attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            try:
                return await fn()
            except Exception as exc:
                last_error = exc
                if not _is_retryable(exc):
                    break
        raise last_error  # type: ignore[misc]

    # Policy-driven path.
    max_attempts = max(1, int(getattr(policy, "max_retries", _MAX_RETRIES)) + 1)
    attempt = 0
    while attempt < max_attempts:
        try:
            return await fn()
        except Exception as exc:
            last_error = exc
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
    result = []
    for s in schemas:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for k, v in s.parameters.items():
            ptype = v.get("type", "string")
            prop: dict[str, Any] = {"type": ptype, "description": v.get("description", "")}
            if ptype == "array":
                items = v.get("items") if isinstance(v.get("items"), dict) else {"type": "string"}
                prop["items"] = {"type": items.get("type", "string")}
            elif "items" in v:
                prop["items"] = v["items"]
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
    declarations = []
    for s in schemas:
        # Inner dict carries str values plus nested dict for "items" — use Any.
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for k, v in s.parameters.items():
            ptype = str(v.get("type", "string")).upper()
            prop: dict[str, Any] = {
                "type": ptype,
                "description": v.get("description", ""),
            }
            # Gemini requires ARRAY properties to declare items.type.
            if ptype == "ARRAY":
                items = v.get("items") if isinstance(v.get("items"), dict) else {}
                item_type = str(items.get("type", "string")).upper()
                prop["items"] = {"type": item_type}
            elif "items" in v and isinstance(v["items"], dict):
                prop["items"] = {"type": str(v["items"].get("type", "string")).upper()}
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
# Uses the Chat Completions API (chat.completions.create). Supports native
# function calling via the `tools` parameter for models like GPT-4o, GPT-5,
# GPT-5-nano, GPT-5.1, and GPT-5.2 (non-Codex).
#
# NOTE: GPT-5.2-Codex and similar models use the Responses API
# (client.responses.create) which has a different tool-calling interface.
# Those would need a separate ResponsesAPIProvider.


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


def _chat_completions_needs_reasoning_none(model: str) -> bool:
    """True when Chat Completions rejects tools + default reasoning_effort.

    GPT-5.5 / GPT-5.6 default to a non-``none`` reasoning effort. On
    ``/v1/chat/completions``, that combination with function tools returns
    HTTP 400 ("use /v1/responses or set reasoning_effort to 'none'"). Until
    we speak Responses API, force ``none`` whenever tools are attached.
    """
    m = (model or "").strip().lower()
    return m.startswith("gpt-5.5") or m.startswith("gpt-5.6")


def _apply_tool_reasoning_compat(
    kwargs: dict[str, Any],
    *,
    model: str,
    has_tools: bool,
) -> None:
    if has_tools and _chat_completions_needs_reasoning_none(model):
        kwargs["reasoning_effort"] = "none"


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
        api_type = (config.openai_api_type or "").lower()
        is_azure = api_type == "azure" or (api_version and base_url and "azure" in base_url.lower())
        if is_azure and api_version and base_url:
            try:
                from openai import AsyncAzureOpenAI
                self.client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=api_version,
                )
            except ImportError:
                self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = AsyncOpenAI(**client_kwargs)

        self.model = config.openai_model
        self._max_tokens = config.max_tokens
        self._temperature = _resolve_temperature(config.openai_model, config.temperature)

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        formatted: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool" and m.tool_call_id:
                formatted.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
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
                # The ``__CACHE_BOUNDARY__`` marker is an Anthropic-only prompt
                # cache hint; strip it here so OpenAI never receives the stray
                # internal token at the tail of its system prompt.
                if isinstance(content, str) and "__CACHE_BOUNDARY__" in content:
                    content = content.replace("__CACHE_BOUNDARY__", "").strip()
                formatted.append({"role": m.role, "content": content})
        formatted = _sanitize_openai_tool_pairs(formatted)
        oai_tools = _to_openai_tools(tools) if tools else None

        if not on_chunk:
            return await _with_retry(
                "openai",
                lambda: self._request_once(formatted, oai_tools),
                policy=getattr(self, "retry_policy", None),
            )
        return await self._stream_with_retry(formatted, on_chunk, cancel_event, oai_tools)

    async def _request_once(
        self, messages: list[dict[str, Any]],
        oai_tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
        _apply_tool_reasoning_compat(
            kwargs, model=self.model, has_tools=bool(oai_tools),
        )
        resp = await self.client.chat.completions.create(**kwargs)
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

    async def _stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        oai_tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [openai] Stream retry %d/%d after %.1fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

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
                    "temperature": self._temperature,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if oai_tools:
                    kwargs["tools"] = oai_tools
                _apply_tool_reasoning_compat(
                    kwargs, model=self.model, has_tools=bool(oai_tools),
                )
                stream = await self.client.chat.completions.create(**kwargs)

                async for chunk in _stall_guarded_stream(stream, _CHUNK_STALL_S):
                    if cancel_event and cancel_event.is_set():
                        await stream.close()
                        return LLMResponse(
                            content="".join(chunks),
                            model=self.model,
                            tokens_used=final_tokens,
                            prompt_tokens=final_prompt_tokens,
                            partial=True,
                            tool_calls=_accumulated_calls(),
                        )

                    try:
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                text = delta.content
                                chunks.append(text)
                                await _invoke_callback(on_chunk, text)
                            
                            if getattr(delta, "tool_calls", None):
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
                # A mid-stream exception used to return the truncated text as a
                # non-retried "final" answer. Retry retryable errors first; only
                # surface a partial (now including any accumulated tool calls)
                # when retries are exhausted or the error is not retryable.
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [openai] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
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
    ) -> LLMResponse:
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
                        if part.get("type") == "text":
                            parts2.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            import base64
                            import binascii

                            url = ((part.get("image_url") or {}).get("url")) or ""
                            if url.startswith("data:") and ";base64," in url:
                                mime, b64_str = url[5:].split(";base64,", 1)
                                try:
                                    decoded = base64.b64decode(b64_str)
                                except (binascii.Error, ValueError):
                                    continue
                                parts2.append({"inline_data": {"mime_type": mime, "data": decoded}})
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
        gemini_config = types.GenerateContentConfig(**config_opts)

        async def _call(contents: list[dict[str, Any]]) -> LLMResponse:
            if not on_chunk:
                return await _with_retry(
                    "gemini",
                    lambda: self._request_once(contents, gemini_config),
                    policy=getattr(self, "retry_policy", None),
                )
            return await self._stream_with_retry(
                contents, gemini_config, on_chunk, cancel_event,
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
        return LLMResponse(
            content=extracted_text,
            model=self.model,
            # ``tokens_used`` is input+output everywhere else; Gemini used to
            # record output-only (and no prompt), garbling usage accounting.
            tokens_used=_prompt_tokens + _output_tokens,
            prompt_tokens=_prompt_tokens,
            tool_calls=fn_calls,
            gemini_parts=raw_parts,
        )

    async def _stream_with_retry(
        self,
        user_contents: list[dict[str, Any]],
        gemini_config: types.GenerateContentConfig,
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
    ) -> LLMResponse:
        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [gemini] Stream retry %d/%d after %.1fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            chunks: list[str] = []
            final_tokens = 0
            final_prompt_tokens = 0
            fn_calls: list[NativeToolCall] = []
            all_stream_parts: list[Any] = []
            last_finish_reason: Any = None

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

                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    prompt_tokens=final_prompt_tokens,
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
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [gemini] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
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


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: EngineConfig):
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Install with: pip install clawagents[anthropic]"
            )
        self.client = _anthropic_mod.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.model = config.anthropic_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
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
                api_messages.append({"role": role, "content": m.content})

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
        if self._temperature is not None and self._temperature >= 0:
            kwargs["temperature"] = self._temperature
        if tools:
            def _anthropic_prop(v: dict[str, Any]) -> dict[str, Any]:
                prop: dict[str, Any] = {
                    "type": v.get("type", "string"),
                    "description": v.get("description", ""),
                }
                # Array parameters must carry their ``items`` schema — dropping
                # it made Claude guess element types (or the API reject the tool).
                if prop["type"] == "array":
                    prop["items"] = v.get("items") or {"type": "string"}
                return prop

            kwargs["tools"] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            k: _anthropic_prop(v) for k, v in s.parameters.items()
                        },
                        "required": [k for k, v in s.parameters.items() if v.get("required")],
                    },
                }
                for s in tools
            ]

        if not on_chunk:
            return await _with_retry(
                "anthropic",
                lambda: self._request_once(kwargs),
                policy=getattr(self, "retry_policy", None),
            )
        return await self._stream_with_retry(kwargs, on_chunk, cancel_event)

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
        return LLMResponse(
            content="".join(text_parts),
            model=self.model,
            tokens_used=(usage.input_tokens + usage.output_tokens) if usage else 0,
            tool_calls=tool_calls if tool_calls else None,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        )

    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
    ) -> LLMResponse:
        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning("  [anthropic] Retry %d/%d after %.1fs", attempt, _MAX_RETRIES, delay)
                await asyncio.sleep(delay)

            chunks: list[str] = []
            tool_calls: list[NativeToolCall] = []
            current_tool: dict[str, Any] | None = None
            output_tokens = 0
            cache_creation = 0
            cache_read = 0
            prompt_tokens = 0

            try:
                async with self.client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if cancel_event and cancel_event.is_set():
                            return LLMResponse(
                                content="".join(chunks), model=self.model,
                                tokens_used=prompt_tokens + output_tokens,
                                prompt_tokens=prompt_tokens,
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
                                    current_tool = {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "input_json": "",
                                    }
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                chunks.append(event.delta.text)
                                await _invoke_callback(on_chunk, event.delta.text)
                            elif hasattr(event.delta, "partial_json") and current_tool:
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

                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    # input+output, matching the non-streaming path (it used to
                    # report output-only, understating usage by the prompt size).
                    tokens_used=prompt_tokens + output_tokens,
                    tool_calls=tool_calls if tool_calls else None,
                    cache_creation_tokens=cache_creation,
                    cache_read_tokens=cache_read,
                    prompt_tokens=prompt_tokens,
                )

            except Exception as exc:
                last_error = exc
                # Retry retryable mid-stream failures before surfacing a
                # truncated partial; include accumulated tool calls when we do.
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [anthropic] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
                    continue
                if chunks or tool_calls:
                    return LLMResponse(
                        content="".join(chunks), model=self.model,
                        tokens_used=prompt_tokens + output_tokens,
                        prompt_tokens=prompt_tokens,
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
    ) -> LLMResponse:
        system_parts: list[str] = []
        converse_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content if isinstance(m.content, str) else str(m.content))
                continue
            role = "assistant" if m.role == "assistant" else "user"
            text = m.content if isinstance(m.content, str) else str(m.content)
            if m.role == "tool":
                text = f"[tool {m.tool_call_id or ''}] {text}"
                role = "user"
            converse_messages.append(
                {"role": role, "content": [{"text": text or ""}]}
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
            # Best-effort tool schemas for Converse toolConfig.
            tool_specs = []
            for s in tools:
                props = {
                    k: {
                        "type": v.get("type", "string"),
                        "description": v.get("description", ""),
                    }
                    for k, v in s.parameters.items()
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
        if on_chunk and content:
            await on_chunk(content)
        usage = (raw or {}).get("usage") or {}
        tokens = int(usage.get("totalTokens") or 0) or (
            int(usage.get("inputTokens") or 0) + int(usage.get("outputTokens") or 0)
        )
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=tokens,
            tool_calls=tool_calls or None,
        )


def _looks_like_ollama(model_name: str) -> bool:
    """Return True if *model_name* looks like an Ollama/local model tag.

    Use the explicit ``ollama/<tag>`` form to bypass heuristics.
    """
    lower = model_name.lower()
    return any(lower.startswith(p) for p in _OLLAMA_PREFIXES)


def create_provider(model_name: str, config: EngineConfig) -> LLMProvider:
    """Create a single LLM provider inferred from model name.

    Clones ``config`` before mutating provider-specific fields so callers
    can safely reuse one ``EngineConfig`` across providers (e.g. main +
    advisor) or in concurrent flows without cross-talk.
    """
    from clawagents.config.config import is_bedrock_model_id, strip_bedrock_prefix

    config = config.model_copy()
    lower = model_name.lower()
    if lower.startswith("gemini"):
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
    explicit_bedrock = lower.startswith("bedrock/")
    bedrock_id = is_bedrock_model_id(model_name)
    if (explicit_bedrock or bedrock_id) and not config.openai_base_url:
        model_id = strip_bedrock_prefix(model_name)
        config.bedrock_model = model_id
        if _is_bedrock_claude_model(model_id):
            config.anthropic_model = model_id
            return BedrockProvider(config)
        return BedrockConverseProvider(config)

    if lower.startswith("claude") or lower.startswith("anthropic"):
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
    if _looks_like_ollama(model_name):
        # Strip the explicit ``ollama/`` routing prefix; Ollama serves the bare tag.
        tag = model_name[len("ollama/"):] if lower.startswith("ollama/") else model_name
        if not config.openai_base_url:
            config.openai_base_url = _OLLAMA_DEFAULT_BASE_URL
        if not config.openai_api_key:
            # Ollama ignores the API key but the OpenAI client refuses an empty string.
            config.openai_api_key = "ollama"
        config.openai_model = tag
        return OpenAIProvider(config)
    config.openai_model = model_name
    return OpenAIProvider(config)
