"""Retry, circuit-breaker and stall-guard plumbing shared by every streaming provider.

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import random
from typing import Any, AsyncIterator, Callable, Coroutine, TypeVar
from openai import APIStatusError, APIConnectionError, APITimeoutError
logger = logging.getLogger("clawagents.providers.llm")
T = TypeVar("T")
# ─── Feature H: Thinking Token Preservation ────────────────────────────────

import re as _re
from clawagents.providers.base import (
    OnChunkCallback,
)

# ─── Streaming Robustness Internals ───────────────────────────────────────

_MAX_RETRIES = 3
_INITIAL_DELAY_S = 1.0
_MAX_DELAY_S = 16.0
_CHUNK_STALL_S = 60.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


# Status codes in free text only count when they look like a status code
# ("Error code: 502", "HTTP 503", "status 429"), not as any digit run — a 400
# whose message says "column 500" or "15024 tokens" used to be retried 3×.
_STATUS_IN_TEXT_RE = _re.compile(
    r"(?:error code|status(?: code)?|http)\s*[:=]?\s*(429|50[0-4])\b"
)


def _is_retryable(err: BaseException) -> bool:
    if isinstance(err, APIStatusError):
        return err.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(err, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(err, Exception):
        status = getattr(err, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status in _RETRYABLE_STATUS_CODES
        msg = str(err).lower()
        # Responses streaming can send overload errors as an event without an
        # HTTP status. Treat the provider's explicit overload markers like 503.
        if "server_is_overloaded" in msg or any(
            phrase in msg
            for phrase in ("currently overloaded", "temporarily overloaded")
        ):
            return True
        if any(
            tok in msg
            for tok in (
                "econnreset", "network", "timeout", "stream stalled",
                "rate limit", "too many requests", "service unavailable",
            )
        ):
            return True
        return _STATUS_IN_TEXT_RE.search(msg) is not None
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
        # asyncio.iscoroutinefunction is deprecated (removed in 3.16).
        if inspect.iscoroutinefunction(cb):
            await cb(text)
        else:
            cb(text)
    except Exception:
        logger.debug("onChunk callback raised — isolated", exc_info=True)


def _import_http_client_module() -> Any:
    """Return ``httpx`` or, failing that, ``httpx2`` (``None`` if neither).

    Newer OpenAI SDKs depend on ``httpx2`` rather than ``httpx``, so a clean
    install may have only one of them. The two share the client API used here.
    """
    for name in ("httpx", "httpx2"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


class _StreamReplayGuard:
    """Keep a retried stream from showing the same text twice.

    A stream that fails part-way has already delivered text through
    ``on_chunk``. The retry starts the response again from the beginning, so
    forwarding it as-is made consumers show the first part twice.

    Per attempt after the first, incoming text is held back while it merely
    repeats what was already delivered. Once the new attempt goes past that
    point only the new part is forwarded. If the new attempt *differs* from
    what was shown (the usual case — generation is not deterministic), the
    consumer is told to discard its partial text via the callback's optional
    ``reset()`` attribute, and the new attempt is delivered from its start.
    Callbacks without ``reset`` keep the old behaviour for that case.
    """

    def __init__(self, callback: OnChunkCallback) -> None:
        self._callback = callback
        self._delivered = ""  # text the consumer currently shows
        self._attempt = ""  # text of the current retry attempt so far
        self._replaying = False

    def begin_attempt(self) -> None:
        self._replaying = bool(self._delivered)
        self._attempt = ""

    async def feed(self, text: str) -> None:
        if not text:
            return
        if not self._replaying:
            self._delivered += text
            await _invoke_callback(self._callback, text)
            return
        self._attempt += text
        if self._delivered.startswith(self._attempt):
            return  # still repeating text the consumer already has
        if self._attempt.startswith(self._delivered):
            fresh = self._attempt[len(self._delivered):]
        else:
            await self._reset()
            fresh = self._attempt
        self._delivered = self._attempt
        self._replaying = False
        await _invoke_callback(self._callback, fresh)

    async def _reset(self) -> None:
        reset = getattr(self._callback, "reset", None)
        if not callable(reset):
            return
        try:
            result = reset()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("onChunk reset raised — isolated", exc_info=True)


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
