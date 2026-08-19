"""Terminal-response handling for an agent turn."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from clawagents.providers.llm import (
    GEMINI_EVIDENCE_MARKER,
    GEMINI_SUMMARIZE_MARKER,
    LLMMessage,
    looks_like_gemini_command_dump,
)

logger = logging.getLogger(__name__)

_MAX_GEMINI_ANSWER_NUDGES = 2
_MAX_GEMINI_EVIDENCE_NUDGES = 2

_HARNESS_USER_MARKERS = (GEMINI_SUMMARIZE_MARKER, GEMINI_EVIDENCE_MARKER)

_MD_TABLE_HEADER_RE = re.compile(r"(?m)^\s*\|.+\|\s*$\n^\s*\|[-:| ]+\|")
_CLAIMED_QUERY_RE = re.compile(
    r"(?i)(executed sql|ran (?:the )?sql|sql (?:was )?executed|"
    r"query returned|database output|from the database|"
    r"qualifying encounters|after executing|real (?:intraday|sql))"
)


def _is_harness_user(content: str) -> bool:
    return any(marker in content for marker in _HARNESS_USER_MARKERS)


def _this_turn_has_tool_work(messages: list[LLMMessage]) -> bool:
    """True when a tool ran after the latest real user message."""
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if msg.role != "user" or not isinstance(msg.content, str):
            continue
        if _is_harness_user(msg.content):
            continue
        start = index + 1
        break
    for msg in messages[start:]:
        if msg.role == "tool":
            return True
        if msg.role == "assistant" and msg.tool_calls_meta:
            return True
        content = msg.content
        if isinstance(content, str) and (
            content.startswith("[used ")
            or content.startswith("[called ")
            or "[result " in content
        ):
            return True
    return False


def _looks_like_ungrounded_query_result(text: str) -> bool:
    """True when the reply presents query counts without this-turn tool output."""
    blob = (text or "").strip()
    if not blob:
        return False
    if _MD_TABLE_HEADER_RE.search(blob) and len(re.findall(r"\b\d{1,6}\b", blob)) >= 4:
        return True
    if _CLAIMED_QUERY_RE.search(blob) and re.search(r"\b\d{2,}\b", blob):
        return True
    return False


def _transcript_has_tool_work(messages: list[LLMMessage]) -> bool:
    for msg in messages:
        if msg.role == "tool":
            return True
        if msg.role == "assistant" and msg.tool_calls_meta:
            return True
        content = msg.content
        if isinstance(content, str) and (
            content.startswith("[used ")
            or content.startswith("[called ")
            or "[result " in content
        ):
            return True
    return False


def _gemini_nudge_count(messages: list[LLMMessage]) -> int:
    return sum(
        1
        for msg in messages
        if msg.role == "user"
        and isinstance(msg.content, str)
        and GEMINI_SUMMARIZE_MARKER in msg.content
    )


def _gemini_evidence_nudge_count(messages: list[LLMMessage]) -> int:
    return sum(
        1
        for msg in messages
        if msg.role == "user"
        and isinstance(msg.content, str)
        and GEMINI_EVIDENCE_MARKER in msg.content
    )


@dataclass(frozen=True)
class CompletionDecision:
    action: Literal["continue", "done"]


class CompletionHandler:
    """Turns a no-tool model response into a retry or terminal state."""

    def __init__(
        self,
        *,
        registry: Any,
        run_context: Any,
        events: Any,
        recorder: Any,
        llm: Any,
        before_tool: Any,
        action_mode: str,
        looks_like_truncated_json: Callable[[str], bool],
        sanitize_assistant_text: Callable[[str], str],
        goal_llm_complete: Callable[[Any, Any], Callable[[str], Awaitable[str]]],
    ) -> None:
        self._registry = registry
        self._run_context = run_context
        self._events = events
        self._recorder = recorder
        self._llm = llm
        self._before_tool = before_tool
        self._action_mode = action_mode
        self._looks_like_truncated_json = looks_like_truncated_json
        self._sanitize_assistant_text = sanitize_assistant_text
        self._goal_llm_complete = goal_llm_complete

    async def handle(
        self,
        *,
        state: Any,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
        use_native_tools: bool,
        consult_advisor: Callable[[list[LLMMessage], str], Awaitable[None]],
        should_final_check: bool,
    ) -> CompletionDecision:
        """Handle a response without tool calls."""
        if not use_native_tools and self._looks_like_truncated_json(response.content):
            self._events.emit(
                "warn", {"message": "truncated JSON tool call detected — asking LLM to retry"}
            )
            messages.extend(
                [
                    LLMMessage(
                        role="assistant", content=response.content, thinking=thinking
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            "Your previous response was cut off mid-JSON. "
                            "Please resend the complete tool call as valid JSON."
                        ),
                    ),
                ]
            )
            return CompletionDecision("continue")

        if use_native_tools and self._should_retry_ungrounded_query_result(
            messages, response
        ):
            self._events.emit(
                "warn",
                {
                    "message": (
                        "Model reported query counts without running a tool — asking it to execute"
                    )
                },
            )
            messages.extend(
                [
                    LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        thinking=thinking,
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            f"{GEMINI_EVIDENCE_MARKER}. "
                            "Call `execute` or `use_skill` now. "
                            "Quote only that tool output. "
                            "Do not invent a table, matrix, or SQL result."
                        ),
                    ),
                ]
            )
            return CompletionDecision("continue")

        if use_native_tools and self._should_retry_empty_or_command_dump(
            messages, response
        ):
            self._events.emit(
                "warn",
                {
                    "message": (
                        "Gemini returned no answer after tools — asking it to summarize"
                    )
                },
            )
            messages.extend(
                [
                    LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        thinking=thinking,
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            f"{GEMINI_SUMMARIZE_MARKER} already collected. "
                            "Write the answer in plain language. "
                            "Do not print [called …] commands."
                        ),
                    ),
                ]
            )
            return CompletionDecision("continue")

        codeact = await self._try_codeact(state, messages, response, thinking)
        if codeact is not None:
            return codeact

        assistant_appended = await self._run_final_advisor_check(
            state,
            messages,
            response,
            thinking,
            consult_advisor,
            should_final_check,
        )
        if assistant_appended is None:
            return CompletionDecision("continue")
        assistant_appended = await self._verify_goal(
            messages,
            response,
            thinking,
            assistant_appended,
        )
        if assistant_appended is None:
            return CompletionDecision("continue")

        # Act-invariant reconciliation gate: block final answer while
        # external state remains uncertain.
        block_reason = self._completion_block_reason()
        if block_reason:
            messages.append(
                LLMMessage(role="assistant", content=response.content, thinking=thinking)
            )
            messages.append(LLMMessage(role="user", content=block_reason))
            return CompletionDecision("continue")

        if self._recorder:
            self._recorder.record_turn(
                response_text=response.content or "",
                model=response.model,
                tokens_used=response.tokens_used,
                thinking=thinking,
            )
        state.result = self._sanitize_assistant_text(response.content)
        state.status = "done"
        self._events.emit("final_content", {"content": state.result})
        self._events.typed(
            "assistant_message", {"content": state.result, "thinking": thinking}
        )
        if not assistant_appended:
            messages.append(
                LLMMessage(role="assistant", content=response.content, thinking=thinking)
            )
        return CompletionDecision("done")

    @staticmethod
    def _should_retry_empty_or_command_dump(messages: list[LLMMessage], response: Any) -> bool:
        if _gemini_nudge_count(messages) >= _MAX_GEMINI_ANSWER_NUDGES:
            return False
        content = (getattr(response, "content", None) or "").strip()
        if looks_like_gemini_command_dump(content):
            return True
        if not content:
            return _transcript_has_tool_work(messages)
        return False

    @staticmethod
    def _should_retry_ungrounded_query_result(
        messages: list[LLMMessage], response: Any
    ) -> bool:
        if _gemini_evidence_nudge_count(messages) >= _MAX_GEMINI_EVIDENCE_NUDGES:
            return False
        if _this_turn_has_tool_work(messages):
            return False
        content = (getattr(response, "content", None) or "").strip()
        return _looks_like_ungrounded_query_result(content)

    async def _try_codeact(
        self,
        state: Any,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
    ) -> CompletionDecision | None:
        if self._action_mode != "code":
            return None
        from clawagents.graph.codeact import extract_code_action, run_code_action

        code = extract_code_action(response.content or "")
        if not code:
            return None
        messages.append(
            LLMMessage(role="assistant", content=response.content, thinking=thinking)
        )
        self._events.emit("tool_call", {"name": "codeact", "args": {"code": code[:500]}})
        result = run_code_action(
            code,
            self._registry,
            before_tool=self._before_tool,
            run_context=self._run_context,
            run_async=self._run_async,
        )
        state.tool_calls += len(result.get("tool_calls") or []) or 1
        observation = str(result.get("observation") or "")
        self._events.emit(
            "tool_result",
            {
                "name": "codeact",
                "success": not result.get("error"),
                "output": observation[:2000],
            },
        )
        if result.get("done"):
            # Act-invariant reconciliation gate: block done while
            # external state remains uncertain.
            block_reason = self._completion_block_reason()
            if block_reason:
                messages.append(LLMMessage(role="user", content=block_reason))
                return CompletionDecision("continue")
            state.result = observation
            state.status = "done"
            self._events.emit("final_content", {"content": state.result})
            self._events.typed(
                "assistant_message", {"content": state.result, "thinking": thinking}
            )
            return CompletionDecision("done")
        messages.append(
            LLMMessage(role="user", content=f"[CodeAct Observation]\n{observation}")
        )
        return CompletionDecision("continue")

    @staticmethod
    def _run_async(coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coroutine).result()

    async def _run_final_advisor_check(
        self,
        state: Any,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
        consult_advisor: Callable[[list[LLMMessage], str], Awaitable[None]],
        should_final_check: bool,
    ) -> bool | None:
        if not should_final_check:
            return False
        messages.append(LLMMessage(role="assistant", content=response.content, thinking=thinking))
        await consult_advisor(messages, "final-check")
        last_message = messages[-1] if messages else None
        if (
            last_message
            and isinstance(last_message.content, str)
            and last_message.content.startswith("[Advisor Guidance]")
        ):
            return None
        return True

    async def _verify_goal(
        self,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
        assistant_appended: bool,
    ) -> bool | None:
        try:
            from clawagents.config.features import is_enabled
            from clawagents.goal import GoalOrchestrator, get_goal_tracker

            metadata = getattr(self._run_context, "_metadata", None)
            goal_mode = isinstance(metadata, dict) and metadata.get("goal_mode")
            tracker = get_goal_tracker(self._run_context) if goal_mode else None
            if not (
                goal_mode
                and is_enabled("goal_autopilot")
                and tracker is not None
                and tracker.is_active()
                and tracker.state is not None
                and tracker.state.status.value not in ("done", "failed", "paused")
            ):
                return assistant_appended
            if not assistant_appended:
                messages.append(
                    LLMMessage(role="assistant", content=response.content, thinking=thinking)
                )
                assistant_appended = True
            orchestrator = GoalOrchestrator(
                tracker,
                self._goal_llm_complete(self._run_context, self._llm),
            )
            accepted, goal_state = await orchestrator.verify((response.content or "")[:6000])
            if accepted:
                self._events.emit("context", {"message": "goal verifier accepted — DONE"})
                return assistant_appended
            injection = (
                "[Goal Verifier] Completion rejected. Continue the plan.\n"
                f"Consecutive misses: {goal_state.consecutive_not_achieved}.\n"
            )
            if goal_state.strategy_text:
                injection += f"Strategy note:\n{goal_state.strategy_text[:2000]}\n"
            messages.append(LLMMessage(role="user", content=injection))
            self._events.emit("context", {"message": "goal verifier rejected completion"})
            return None
        except Exception:
            logger.debug("goal final gate failed", exc_info=True)
            return assistant_appended

    def _completion_block_reason(self) -> str | None:
        """Check act-invariant reconciliation gate."""
        try:
            from clawagents.permissions.act_invariants import completion_block_reason

            return completion_block_reason(self._run_context)
        except Exception:
            logger.debug("completion_block_reason check failed", exc_info=True)
            return None
