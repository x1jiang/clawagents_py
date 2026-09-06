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

# Output-limit recovery. Reasoning models (Muse-Glimmer on SGLang, Gemma with
# thinking enabled, DeepSeek, …) can spend the whole ``max_tokens`` budget on
# chain-of-thought and stop with finish_reason="length", empty content and no
# tool call. Treating that as "the answer" ends the run mid-task; routing it
# into the Gemini "write the answer now" nudge tells the model to stop working.
# Instead, ask it to continue, briefly, and count the attempts.
TRUNCATION_MARKER = "[System] Output limit reached"
_MAX_TRUNCATION_NUDGES = 2
_TRUNCATION_PLACEHOLDER = "[output truncated at the token limit]"

_HARNESS_USER_MARKERS = (GEMINI_SUMMARIZE_MARKER, GEMINI_EVIDENCE_MARKER, TRUNCATION_MARKER)

_MD_TABLE_HEADER_RE = re.compile(r"(?m)^\s*\|.+\|\s*$\n^\s*\|[-:| ]+\|")
_MD_TABLE_SEP_RE = re.compile(r"^\|[-:| ]+\|$")
_CLAIMED_QUERY_RE = re.compile(
    r"(?i)(executed sql|ran (?:the )?sql|sql (?:was )?executed|"
    r"query returned|database output|from the database|"
    r"qualifying encounters|after executing|real (?:intraday|sql))"
)
_FLATTEN_EXECUTE_RESULT_RE = re.compile(
    r"\[result\s+execute:\s*(.*?)\]",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"```[\w-]*\n?")
_COHORT_COUNT_RE = re.compile(
    r"(?i)\b(\d{2,6})\s+(patients?|encounters?|cases?|rows?|records?)\b"
)
_DAY_COUNT_RE = re.compile(
    r"(?i)\b(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|"
    r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b[^.\n]{0,16}?(\d{1,6})"
)
_HTML_NUM_CELL_RE = re.compile(r"(?i)<t[dh][^>]*>\s*(\d{1,6})\s*</t[dh]>")
_QUERY_EVIDENCE_TOOLS = frozenset({"execute"})
_UNGROUNDED_REFUSAL = (
    "Harness blocked an ungrounded count result. "
    "This-turn `execute` output does not support the numbers in the draft, "
    "so the table was not published. Re-run the query or use a stronger model."
)


def _is_harness_user(content: str) -> bool:
    return any(marker in content for marker in _HARNESS_USER_MARKERS)


def _this_turn_start(messages: list[LLMMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if msg.role != "user" or not isinstance(msg.content, str):
            continue
        if _is_harness_user(msg.content):
            continue
        return index + 1
    return 0


def _this_turn_has_tool_work(messages: list[LLMMessage]) -> bool:
    """True when a tool ran after the latest real user message."""
    for msg in messages[_this_turn_start(messages) :]:
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


def _this_turn_execute_output(messages: list[LLMMessage]) -> str:
    """Verbatim this-turn `execute` results. `use_skill` is not query evidence."""
    start = _this_turn_start(messages)
    id_to_name: dict[str, str] = {}
    parts: list[str] = []
    execute_called = False
    for msg in messages[start:]:
        if msg.role == "assistant" and msg.tool_calls_meta:
            for tc in msg.tool_calls_meta:
                cid = str(tc.get("id") or "")
                name = str(tc.get("name") or "")
                if cid and name:
                    id_to_name[cid] = name
                if name in _QUERY_EVIDENCE_TOOLS:
                    execute_called = True
        if msg.role == "tool" and isinstance(msg.content, str):
            name = id_to_name.get(str(msg.tool_call_id or ""), "")
            if name in _QUERY_EVIDENCE_TOOLS:
                parts.append(msg.content)
            continue
        content = msg.content if isinstance(msg.content, str) else ""
        if "[used execute" in content or "[called execute" in content:
            execute_called = True
        parts.extend(_FLATTEN_EXECUTE_RESULT_RE.findall(content))
    if not parts and execute_called:
        for msg in messages[start:]:
            if msg.role == "tool" and isinstance(msg.content, str):
                parts.append(msg.content)
    return "\n".join(part for part in parts if str(part).strip())


def _plain_answer(text: str) -> str:
    return _FENCE_RE.sub("\n", text or "")


def _looks_like_ungrounded_query_result(text: str) -> bool:
    """True when the reply presents query counts that need execute evidence."""
    blob = _plain_answer(text).strip()
    if not blob:
        return False
    nums = re.findall(r"\b\d{1,6}\b", blob)
    if _MD_TABLE_HEADER_RE.search(blob) and (
        len(nums) >= 3 or any(int(item) >= 10 for item in nums)
    ):
        return True
    if len(_HTML_NUM_CELL_RE.findall(blob)) >= 2:
        return True
    if len(_DAY_COUNT_RE.findall(blob)) >= 3:
        return True
    if _COHORT_COUNT_RE.search(blob):
        return True
    if _CLAIMED_QUERY_RE.search(blob) and re.search(r"\b\d{2,}\b", blob):
        return True
    return False


def _evidence_numbers(text: str) -> set[int]:
    found: set[int] = set()
    for raw in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{1,6}\b", text or ""):
        found.add(int(raw.replace(",", "")))
    return found


def _table_data_number_rows(text: str) -> list[list[int]]:
    rows: list[list[int]] = []
    seen_sep = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if _MD_TABLE_SEP_RE.match(stripped):
            seen_sep = True
            continue
        if not seen_sep:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        nums: list[int] = []
        for cell in cells[1:]:
            raw = cell.replace(",", "")
            if re.fullmatch(r"\d{1,6}", raw):
                nums.append(int(raw))
        if nums:
            rows.append(nums)
    return rows


def _ungrounded_count_tokens(answer: str, evidence: str) -> list[int]:
    """Counts in the reply that are not in execute output and not a row sum."""
    ev = _evidence_numbers(evidence)
    bad: list[int] = []
    rows = _table_data_number_rows(_plain_answer(answer))
    if rows:
        for row in rows:
            for number in row:
                if number in ev:
                    continue
                others = [item for item in row if item != number]
                if others and all(item in ev for item in others) and number == sum(others):
                    continue
                bad.append(number)
        return bad
    for number in _evidence_numbers(answer):
        if 1900 <= number <= 2100:
            continue
        if number not in ev:
            bad.append(number)
    return bad


def _should_reject_ungrounded_counts(bad: list[int]) -> bool:
    if any(number >= 10 for number in bad):
        return True
    return len(bad) >= 2


def _ungrounded_query_reason(
    messages: list[LLMMessage],
    content: str,
) -> str | None:
    if not _looks_like_ungrounded_query_result(content):
        return None
    evidence = _this_turn_execute_output(messages)
    if not evidence.strip():
        return (
            "Call `execute` now (`use_skill` is instructions, not a query). "
            "Quote only that tool output. "
            "Do not invent a table, matrix, or SQL result."
        )
    bad = _ungrounded_count_tokens(content, evidence)
    if not _should_reject_ungrounded_counts(bad):
        return None
    shown = ", ".join(str(number) for number in bad[:8])
    return (
        f"This-turn `execute` output is missing counts you reported ({shown}). "
        "Re-run `execute` or quote only numbers from that output. "
        "Do not invent hour/day cells from a daily total."
    )


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


def _truncation_nudge_count(messages: list[LLMMessage]) -> int:
    return sum(
        1
        for msg in messages
        if msg.role == "user"
        and isinstance(msg.content, str)
        and msg.content.startswith(TRUNCATION_MARKER)
    )


def response_hit_output_limit(response: Any) -> bool:
    """True when the turn was cut short with no tool call to act on.

    Either the provider stopped for ``max_tokens`` (``finish_reason="length"``)
    or the stream was interrupted after retries (``partial`` with content —
    the empty-content case is handled as a cancellation by the turn driver).
    """
    if getattr(response, "tool_calls", None):
        return False
    if str(getattr(response, "finish_reason", "") or "").lower() == "length":
        return True
    return bool(getattr(response, "partial", False)) and bool(
        (getattr(response, "content", "") or "").strip()
    )


_MAX_TOKENS_BUMP_CAP = 65_536


def _grow_output_budget(llm: Any) -> int | None:
    """Give a nudged retry more room (×1.5, capped) so the same think does not
    hit the same wall; explicit provider state only, mirrors the doom-loop
    temperature bump in turn_response."""
    target = getattr(llm, "primary", None) or llm  # FallbackProvider wrapper
    current = getattr(target, "_max_tokens", None)
    if not isinstance(current, int) or isinstance(current, bool) or current <= 0:
        return None
    grown = min(_MAX_TOKENS_BUMP_CAP, int(current * 1.5))
    if grown <= current:
        return None
    try:
        setattr(target, "_max_tokens", grown)
    except Exception:
        return None
    return grown


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

        if response_hit_output_limit(response):
            nudges = _truncation_nudge_count(messages)
            if nudges < _MAX_TRUNCATION_NUDGES:
                grown = _grow_output_budget(self._llm)
                self._events.emit(
                    "warn",
                    {
                        "message": (
                            "output limit reached before a tool call or answer — "
                            f"asking the model to continue ({nudges + 1}/{_MAX_TRUNCATION_NUDGES})"
                            + (f", max_tokens → {grown}" if grown else "")
                        )
                    },
                )
                # Keep a non-empty assistant turn so strict alternating chat
                # templates (Gemma/llama.cpp) still accept the transcript. The
                # truncated text itself is not replayed: it is a partial
                # thought, and re-sending 2-6K tokens of it twice only grows
                # the context.
                messages.extend(
                    [
                        LLMMessage(
                            role="assistant",
                            content=_TRUNCATION_PLACEHOLDER,
                            thinking=thinking,
                        ),
                        LLMMessage(
                            role="user",
                            content=(
                                f"{TRUNCATION_MARKER} before you produced a tool call or a "
                                "final answer, so that turn was discarded. Continue the task "
                                "now: keep any reasoning to a few sentences, then emit the "
                                "next tool call or the final answer in this response."
                            ),
                        ),
                    ]
                )
                return CompletionDecision("continue")
            self._events.emit(
                "warn",
                {
                    "message": (
                        "output limit reached repeatedly — treating the partial "
                        "text as the final answer"
                    )
                },
            )

        ungrounded = None
        if use_native_tools:
            ungrounded = _ungrounded_query_reason(
                messages, (getattr(response, "content", None) or "").strip()
            )
        if ungrounded and _gemini_evidence_nudge_count(messages) >= (
            _MAX_GEMINI_EVIDENCE_NUDGES
        ):
            self._events.emit(
                "warn",
                {
                    "message": (
                        "Model still reported ungrounded query counts after "
                        "evidence nudges — blocking the table"
                    )
                },
            )
            state.result = _UNGROUNDED_REFUSAL
            state.status = "done"
            if self._recorder:
                self._recorder.record_turn(
                    response_text=state.result,
                    model=getattr(response, "model", None),
                    tokens_used=getattr(response, "tokens_used", 0),
                    thinking=thinking,
                )
            self._events.emit("final_content", {"content": state.result})
            self._events.typed(
                "assistant_message", {"content": state.result, "thinking": thinking}
            )
            messages.append(
                LLMMessage(role="assistant", content=state.result, thinking=thinking)
            )
            return CompletionDecision("done")
        if ungrounded:
            self._events.emit(
                "warn",
                {
                    "message": (
                        "Model reported query counts that are not in this-turn "
                        "execute output — asking it to execute"
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
                        content=f"{GEMINI_EVIDENCE_MARKER}. {ungrounded}",
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
        content = (getattr(response, "content", None) or "").strip()
        return _ungrounded_query_reason(messages, content) is not None

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
