"""Interpret a provider response into an explicit ReAct turn decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clawagents.providers.llm import LLMResponse, NativeToolCall, strip_thinking_tokens
from clawagents.tools.registry import ParsedToolCall

from .run_runtime import RunEvents


@dataclass(frozen=True)
class ParsedTurn:
    response: LLMResponse
    thinking: str | None
    tool_calls: list[ParsedToolCall]
    native_tool_calls: list[NativeToolCall] | None
    should_resample: bool = False


class TurnResponseInterpreter:
    """Normalises reasoning channels and tool-call dialects for the loop."""

    def __init__(self, *, llm: Any, registry: Any, events: RunEvents) -> None:
        self._llm = llm
        self._registry = registry
        self._events = events

    def parse(
        self,
        response: LLMResponse,
        *,
        use_native_tools: bool,
        run_context: Any,
    ) -> ParsedTurn:
        response, thinking = self._extract_thinking(response)
        if self._detect_doom_loop(response, thinking, run_context):
            return ParsedTurn(response, thinking, [], None, should_resample=True)
        native_calls: list[NativeToolCall] | None = None
        if use_native_tools:
            native_calls = response.tool_calls or []
            calls = [
                ParsedToolCall(tool_name=call.tool_name, args=call.args)
                for call in native_calls
            ]
        else:
            calls = self._registry.parse_tool_calls(response.content)
        return ParsedTurn(response, thinking, calls, native_calls)

    @staticmethod
    def _extract_thinking(response: LLMResponse) -> tuple[LLMResponse, str | None]:
        thinking: str | None = None
        channel = getattr(response, "thinking", None)
        if response.content and "<think>" in response.content:
            clean_content, thinking = strip_thinking_tokens(response.content)
            if channel and thinking:
                thinking = f"{channel}\n{thinking}"
            elif channel:
                thinking = str(channel)
            # Rebuild with every field: dropping finish_reason here used to
            # hide a max_tokens cut from the output-limit recovery, so an
            # unclosed <think> blob became the "final answer".
            response = LLMResponse(
                content=clean_content,
                model=response.model,
                tokens_used=response.tokens_used,
                partial=response.partial,
                tool_calls=response.tool_calls,
                gemini_parts=response.gemini_parts,
                cache_creation_tokens=getattr(response, "cache_creation_tokens", 0),
                cache_read_tokens=getattr(response, "cache_read_tokens", 0),
                prompt_tokens=getattr(response, "prompt_tokens", 0),
                thinking=thinking,
                finish_reason=getattr(response, "finish_reason", None),
                reasoning_tokens=getattr(response, "reasoning_tokens", 0),
            )
        if not thinking and channel:
            thinking = str(channel)
        return response, thinking

    def _detect_doom_loop(
        self,
        response: LLMResponse,
        thinking: str | None,
        run_context: Any,
    ) -> bool:
        try:
            from clawagents.config.features import is_enabled
            from clawagents.doom_loop import (
                DoomLoopRecoveryPolicy,
                DoomLoopState,
                detect_tail_repetition,
                note_trigger,
                should_resample,
            )

            if not is_enabled("doom_loop"):
                return False
            signal = (
                detect_tail_repetition(thinking, channel="thinking")
                if thinking
                else None
            )
            if signal is None and response.content:
                signal = detect_tail_repetition(str(response.content), channel="response")
            if signal is None:
                self._clear_force_response(run_context)
                return False
            metadata = getattr(run_context, "_metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            state = metadata.get("doom_loop_state")
            if not isinstance(state, DoomLoopState):
                state = DoomLoopState()
                metadata["doom_loop_state"] = state
            note_trigger(state, signal)
            policy = DoomLoopRecoveryPolicy()
            if should_resample(signal, state, policy):
                state.retry_count += 1
                try:
                    temperature = float(getattr(self._llm, "_temperature", 0.0) or 0.0)
                    setattr(self._llm, "_temperature", min(1.0, max(0.4, temperature + 0.4)))
                except Exception:
                    pass
                metadata["doom_force_response"] = True
                self._events.emit(
                    "warn",
                    {
                        "message": (
                            f"doom-loop {signal.label} — resampling "
                            f"({state.retry_count}/{policy.max_retries}, force response channel)"
                        )
                    },
                )
                return True
            if metadata.get("doom_force_response") and signal.channel == "thinking":
                metadata["doom_force_response"] = True
        except Exception:
            # Detection is a safety net; it must never mask a usable response.
            return False
        self._clear_force_response(run_context)
        return False

    @staticmethod
    def _clear_force_response(run_context: Any) -> None:
        metadata = getattr(run_context, "_metadata", None)
        if isinstance(metadata, dict):
            metadata.pop("doom_force_response", None)
