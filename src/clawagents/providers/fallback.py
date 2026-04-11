"""Three-tier provider fallback with quarantine.

Chain: primary → named fallback → global fallback list.
Providers that fail consecutively get quarantined.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine

from clawagents.providers.llm import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    NativeToolCall,
    NativeToolSchema,
    OnChunkCallback,
)

logger = logging.getLogger(__name__)

_HEALTH_CHECK_MESSAGE = LLMMessage(role="user", content="ping")


class ProviderState:
    """Tracks consecutive failure count and quarantine status for a single provider."""

    def __init__(self) -> None:
        self.consecutive_failures: int = 0
        self.quarantined: bool = False
        self.quarantine_start: float = 0.0

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.quarantined = False

    def quarantine(self) -> None:
        self.quarantined = True
        self.quarantine_start = time.monotonic()

    def health_check_due(self, interval_s: float) -> bool:
        return self.quarantined and (time.monotonic() - self.quarantine_start) >= interval_s


class FallbackProvider(LLMProvider):
    """Wraps an LLMProvider with a three-tier fallback chain and quarantine logic.

    Chain: primary → named fallbacks (in order) → skip quarantined providers.
    A provider that fails ``quarantine_threshold`` consecutive times is quarantined
    and excluded from the active pool until it passes a lightweight health check
    run every ``health_check_interval_s`` seconds.
    """

    name: str = "fallback"

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider],
        quarantine_threshold: int = 3,
        health_check_interval_s: float = 60.0,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.primary = primary
        self.fallbacks = fallbacks
        self.quarantine_threshold = quarantine_threshold
        self.health_check_interval_s = health_check_interval_s
        self.on_event = on_event

        # Per-provider state keyed by provider object id
        self._states: dict[int, ProviderState] = {}
        for p in [primary, *fallbacks]:
            self._states[id(p)] = ProviderState()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _state(self, provider: LLMProvider) -> ProviderState:
        return self._states[id(provider)]

    def _emit(self, level: str, message: str) -> None:
        logger.warning(message) if level == "warn" else logger.info(message)
        if self.on_event:
            try:
                self.on_event(level, {"message": message})
            except Exception:
                pass

    def _is_active(self, provider: LLMProvider) -> bool:
        return not self._state(provider).quarantined

    def _maybe_quarantine(self, provider: LLMProvider) -> None:
        state = self._state(provider)
        if state.consecutive_failures >= self.quarantine_threshold and not state.quarantined:
            state.quarantine()
            self._emit(
                "warn",
                f"Provider '{provider.name}' quarantined after "
                f"{state.consecutive_failures} consecutive failures.",
            )

    async def _try_health_check(self, provider: LLMProvider) -> bool:
        """Attempt a lightweight ping to check if a quarantined provider has recovered."""
        try:
            await provider.chat([_HEALTH_CHECK_MESSAGE])
            self._state(provider).record_success()
            self._emit("warn", f"Provider '{provider.name}' passed health check — restored to active pool.")
            return True
        except Exception:
            # Reset the quarantine timer so we check again after the interval
            self._state(provider).quarantine_start = time.monotonic()
            return False

    async def _health_check_quarantined(self) -> None:
        """Run health checks for all quarantined providers whose interval has elapsed."""
        for provider in [self.primary, *self.fallbacks]:
            state = self._state(provider)
            if state.health_check_due(self.health_check_interval_s):
                await self._try_health_check(provider)

    # ── LLMProvider interface ─────────────────────────────────────────────

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        # Run health checks for any quarantined providers before attempting calls
        await self._health_check_quarantined()

        all_providers = [self.primary, *self.fallbacks]
        last_exc: Exception | None = None

        for i, provider in enumerate(all_providers):
            state = self._state(provider)

            if state.quarantined:
                self._emit(
                    "warn",
                    f"Skipping quarantined provider '{provider.name}'.",
                )
                continue

            try:
                response = await provider.chat(
                    messages,
                    on_chunk=on_chunk,
                    cancel_event=cancel_event,
                    tools=tools,
                )
                state.record_success()
                return response

            except Exception as exc:
                last_exc = exc
                state.record_failure()
                self._maybe_quarantine(provider)

                # Determine next active provider for the warning message
                remaining = [
                    p for p in all_providers[i + 1:]
                    if not self._state(p).quarantined
                ]
                if remaining:
                    self._emit(
                        "warn",
                        f"Provider '{provider.name}' failed ({exc!r}), "
                        f"falling back to '{remaining[0].name}'.",
                    )
                else:
                    self._emit(
                        "warn",
                        f"Provider '{provider.name}' failed ({exc!r}). "
                        "No active fallback providers remaining.",
                    )

        raise RuntimeError(
            f"All providers failed. Last error: {last_exc!r}"
        ) from last_exc
