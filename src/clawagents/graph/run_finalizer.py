"""End-of-run lifecycle for the graph executor."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Awaitable, Callable

from clawagents.guardrails import (
    GuardrailBehavior,
    GuardrailTripwireTriggered,
)
from clawagents.providers.llm import LLMMessage

from .run_runtime import HookDispatcher, RunEvents, SessionMessageJournal

logger = logging.getLogger(__name__)


class RunFinalizer:
    """Owns durable run completion side effects and output normalization."""

    def __init__(
        self,
        *,
        events: RunEvents,
        hooks: HookDispatcher,
        run_context: Any,
        session_journal: SessionMessageJournal,
        session_writer: Any,
        recorder: Any,
        llm: Any,
        task: str,
        learn: bool,
        output_guardrails: list[Any],
        output_type: Any,
        run_output_guardrails: Callable[..., Awaitable[tuple[str, str | None]]],
        coerce_output_type: Callable[[str, Any], Any],
        accumulate_usage: Callable[[Any], Any],
        taxonomy_dispatcher: Any,
        session_end_tail: bool,
    ) -> None:
        self._events = events
        self._hooks = hooks
        self._run_context = run_context
        self._session_journal = session_journal
        self._session_writer = session_writer
        self._recorder = recorder
        self._llm = llm
        self._task = task
        self._learn = learn
        self._output_guardrails = output_guardrails
        self._output_type = output_type
        self._run_output_guardrails = run_output_guardrails
        self._coerce_output_type = coerce_output_type
        self._accumulate_usage = accumulate_usage
        self._taxonomy_dispatcher = taxonomy_dispatcher
        self._session_end_tail = session_end_tail

    async def finalize(self, state: Any, messages: list[LLMMessage], *, elapsed: float) -> Any:
        """Finalize a completed run without changing its terminal decision."""
        self._write_session_completion(state)
        run_summary = self._finalize_trajectory(state)
        await self._judge_and_learn(state, run_summary)
        await self._apply_output_guardrails(state)
        self._coerce_final_output(state)
        await self._persist_messages(messages)
        self._emit_final_output(state)
        await self._finish_hooks(state)
        await self._run_dream(state)
        await self._notify_taxonomy(state)
        self._emit_cache_waste()
        self._events.emit(
            "agent_done",
            {
                "tool_calls": state.tool_calls,
                "iterations": state.iterations,
                "elapsed": elapsed,
                "usage": self._run_context.usage.to_dict(),
            },
        )
        self._flush_stranded_interjects()
        return state

    def _emit_cache_waste(self) -> None:
        """Report prompt-cache re-billing, with a cause, when it is material.

        Silent by default: emits nothing on a healthy run, on a provider that
        does not report cache reads, or when the shortfall is under the block
        granularity floor.
        """
        try:
            from clawagents.cache_waste import analyze_cache_waste

            meta = getattr(self._run_context, "_metadata", None)
            exempt = (
                meta.get("cache_context_change_rounds") or ()
                if isinstance(meta, dict)
                else ()
            )
            report = analyze_cache_waste(
                self._run_context.usage.per_request,
                context_change_rounds=exempt,
            )
            if report.significant:
                self._events.emit(
                    "cache_waste",
                    {"message": report.summary(), **report.to_dict()},
                )
        except Exception:
            logger.debug("cache waste analysis failed", exc_info=True)

    def _last_persisted_assistant(self) -> str | None:
        """Content of the last assistant_message already in the session log."""
        path = getattr(self._session_writer, "path", None)
        if path is None:
            return None
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            if '"assistant_message"' not in line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if event.get("type") == "assistant_message":
                return str(event.get("content") or "")
        return None

    def _write_session_completion(self, state: Any) -> None:
        if self._session_writer is None:
            return
        # Only the tool-calling path writes an assistant_message, so a run that
        # ends in plain prose — the common shape — persisted no answer at all.
        # Record it here, skipping the case where the run stopped straight
        # after a tool round whose content was already stored.
        final = getattr(state, "result", "")
        if isinstance(final, str) and final.strip():
            if self._last_persisted_assistant() != final:
                self._session_writer.write_assistant_message(final)
        self._session_writer.write_turn_completed(
            state.iterations,
            state.tool_calls,
            state.status,
        )
        state.session_file = str(self._session_writer.path)

    def _finalize_trajectory(self, state: Any) -> Any | None:
        if self._recorder is None:
            return None
        outcome = state.status if state.status != "running" else "success"
        summary = self._recorder.finalize(outcome)
        state.trajectory_file = summary.trajectory_file
        self._events.emit(
            "context", {"message": f"trajectory saved to {summary.trajectory_file}"}
        )
        return summary

    async def _judge_and_learn(self, state: Any, run_summary: Any | None) -> None:
        if not (self._learn and self._recorder and run_summary):
            return
        await self._judge_run(state, run_summary)
        if getattr(self._run_context, "skip_memory", False):
            return
        await self._extract_lessons(run_summary)

    async def _judge_run(self, state: Any, run_summary: Any) -> None:
        try:
            from clawagents.trajectory.judge import judge_run

            judge_result = await judge_run(
                self._llm,
                self._task,
                asdict(run_summary),
                state.result,
                [asdict(turn) for turn in self._recorder.turns],
            )
            judge_response = judge_result.pop("_llm_response", None)
            if judge_response is not None:
                try:
                    self._accumulate_usage(judge_response)
                except Exception:  # noqa: BLE001
                    pass
            run_summary.judge_score = judge_result.get("judge_score")
            run_summary.judge_justification = judge_result.get("judge_justification", "")
            self._events.emit(
                "context",
                {
                    "message": (
                        f"LLM Judge: score={run_summary.judge_score}/3 — "
                        f"{run_summary.judge_justification[:80]}"
                    )
                },
            )
        except Exception:
            logger.debug("LLM-as-Judge failed", exc_info=True)

    async def _extract_lessons(self, run_summary: Any) -> None:
        try:
            from clawagents.trajectory.lessons import (
                extract_lessons,
                save_lessons,
                should_extract_lessons,
            )

            summary_dict = asdict(run_summary)
            if not should_extract_lessons(summary_dict):
                self._events.emit(
                    "context",
                    {
                        "message": (
                            "PTRL: skipped lesson extraction "
                            f"(quality={run_summary.quality}, "
                            f"mixed={run_summary.has_mixed_outcomes}, "
                            f"score={run_summary.run_score})"
                        )
                    },
                )
                return
            lessons_text = await extract_lessons(
                self._llm,
                summary_dict,
                [asdict(turn) for turn in self._recorder.turns],
            )
            if not lessons_text:
                return
            save_lessons(
                lessons_text,
                run_summary.task,
                run_summary.outcome,
                model=run_summary.model,
            )
            self._events.emit(
                "context", {"message": "PTRL: extracted and saved lessons from this run"}
            )
            self._promote_failure_lessons(lessons_text)
            self._promote_fact_store(lessons_text)
            self._promote_recurring_lessons(lessons_text, run_summary.task)
        except Exception:
            logger.debug("PTRL: post-run self-analysis failed", exc_info=True)

    def _promote_failure_lessons(self, lessons_text: str) -> None:
        try:
            from clawagents.trajectory.failure_learn import (
                append_failure_lessons_to_agents_md,
            )

            promoted = append_failure_lessons_to_agents_md(lessons_text)
            if promoted:
                self._events.emit(
                    "context",
                    {
                        "message": (
                            f"PTRL: appended {len(promoted)} failure lesson(s) to AGENTS.md"
                        )
                    },
                )
        except Exception:
            logger.debug("PTRL: AGENTS.md failure-learn append failed", exc_info=True)

    def _promote_fact_store(self, lessons_text: str) -> None:
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("fact_store"):
                return
            from clawagents.memory.facts import promote_lesson_bullets_to_facts

            facts = promote_lesson_bullets_to_facts(lessons_text)
            if facts:
                self._events.emit(
                    "context", {"message": f"PTRL: promoted {len(facts)} live fact(s)"}
                )
        except Exception:
            logger.debug("PTRL: fact promotion failed", exc_info=True)

    def _promote_recurring_lessons(self, lessons_text: str, task: str) -> None:
        try:
            from clawagents.trajectory.lesson_promotion import maybe_promote_recurring_lessons

            promoted = maybe_promote_recurring_lessons(lessons_text, task=task)
            if promoted:
                self._events.emit(
                    "context",
                    {
                        "message": (
                            f"PTRL: promoted {len(promoted)} recurring lesson(s) to skill_workshop"
                        )
                    },
                )
        except Exception:
            logger.debug("PTRL: lesson promotion failed", exc_info=True)

    async def _apply_output_guardrails(self, state: Any) -> None:
        if not self._output_guardrails or not state.result:
            return
        try:
            rewritten, tripped = await self._run_output_guardrails(
                self._output_guardrails,
                self._run_context,
                state.result,
            )
            if tripped:
                state.guardrail_triggered = tripped
                state.result = str(rewritten)
                self._events.typed(
                    "guardrail_tripped",
                    {
                        "guardrail_name": tripped,
                        "where": "output",
                        "behavior": GuardrailBehavior.REJECT_CONTENT.value,
                        "message": state.result,
                    },
                )
                self._events.emit(
                    "warn", {"message": f"output guardrail tripped: {tripped}"}
                )
        except GuardrailTripwireTriggered as tripwire:
            state.guardrail_triggered = tripwire.guardrail_name
            self._events.typed(
                "guardrail_tripped",
                {
                    "guardrail_name": tripwire.guardrail_name,
                    "where": "output",
                    "behavior": tripwire.result.behavior.value,
                    "message": tripwire.result.message or "",
                },
            )
            self._events.emit(
                "warn",
                {"message": f"output guardrail raised: {tripwire.guardrail_name}"},
            )

    def _coerce_final_output(self, state: Any) -> None:
        if self._output_type is not None and state.status == "done" and state.result:
            try:
                state.final_output = self._coerce_output_type(state.result, self._output_type)
            except Exception as exc:
                self._events.emit(
                    "warn", {"message": f"output_type coercion failed: {exc}"}
                )
                state.final_output = state.result
        elif state.status == "done":
            state.final_output = state.result

    async def _persist_messages(self, messages: list[LLMMessage]) -> None:
        if not self._session_journal.enabled:
            return
        try:
            await self._session_journal.persist(messages)
        except Exception as exc:
            self._events.emit("warn", {"message": f"session save failed: {exc}"})

    def _emit_final_output(self, state: Any) -> None:
        self._events.typed(
            "final_output",
            {
                "output": state.final_output if state.final_output is not None else state.result,
                "raw": state.result if isinstance(state.result, str) else "",
                "usage": self._run_context.usage.to_dict(),
            },
        )

    async def _finish_hooks(self, state: Any) -> None:
        if self._hooks.hooks:
            await self._hooks.fire("on_run_end", state.result)

    async def _run_dream(self, state: Any) -> None:
        try:
            from clawagents.config.features import is_enabled

            workspace = self._workspace()
            if self._session_end_tail and (
                is_enabled("memory_dream") or is_enabled("smart_memory")
            ):
                from clawagents.memory.dream import append_session_log

                stem = getattr(self._session_writer, "session_id", None)
                log_body = (
                    f"## Task\n{(self._task or '')[:4000]}\n\n"
                    f"## Outcome\n{state.status}\n\n"
                    f"## Result\n{(state.result or '')[:8000]}"
                )
                append_session_log(log_body, workspace=workspace, stem=stem)

            if self._session_end_tail and is_enabled("memory_dream"):
                from clawagents.memory.dream import check_dream_gates, run_dream

                if isinstance(check_dream_gates(workspace), str):
                    return

                async def dream_llm(prompt: str) -> str:
                    response = await self._llm.chat([LLMMessage(role="user", content=prompt)])
                    return str(getattr(response, "content", "") or "")

                try:
                    dream_output = await asyncio.wait_for(
                        run_dream(dream_llm, workspace=workspace),
                        timeout=90.0,
                    )
                    message = (
                        f"dream: {dream_output.reason}"
                        if dream_output.ok
                        else f"dream skipped: {dream_output.reason}"
                    )
                    self._events.emit("context", {"message": message})
                except asyncio.TimeoutError:
                    self._events.emit(
                        "context", {"message": "dream: timed out (lock released)"}
                    )
        except Exception:
            logger.debug("dream scheduling failed", exc_info=True)

    async def _notify_taxonomy(self, state: Any) -> None:
        if self._taxonomy_dispatcher is None:
            return
        try:
            from clawagents.hooks.external import dispatch_taxonomy_hook
            from clawagents.hooks.taxonomy import HookEvent

            await dispatch_taxonomy_hook(
                self._taxonomy_dispatcher,
                HookEvent.SESSION_END,
                {
                    "status": state.status,
                    "result_preview": (state.result or "")[:500],
                    "tool_calls": state.tool_calls,
                },
                blocking=False,
            )
            result_text = state.result or ""
            stopped_unsuccessfully = (
                state.status in ("error", "max_iterations")
                or result_text.startswith("[cancelled]")
                or result_text.startswith("[interrupted]")
            )
            if stopped_unsuccessfully:
                await dispatch_taxonomy_hook(
                    self._taxonomy_dispatcher,
                    HookEvent.STOP_FAILURE,
                    {"status": state.status, "message": result_text or state.status},
                    blocking=False,
                )
                await dispatch_taxonomy_hook(
                    self._taxonomy_dispatcher,
                    HookEvent.NOTIFICATION,
                    {
                        "message": result_text or state.status,
                        "kind": "stop_failure",
                    },
                    blocking=False,
                )
            await dispatch_taxonomy_hook(
                self._taxonomy_dispatcher,
                HookEvent.STOP,
                {"status": state.status},
                blocking=False,
            )
        except Exception:
            logger.debug("taxonomy session_end hook failed", exc_info=True)

    def _flush_stranded_interjects(self) -> None:
        try:
            from clawagents.interjection import take_stranded_interjects

            stranded = take_stranded_interjects(self._run_context)
            if not stranded:
                return
            metadata = getattr(self._run_context, "_metadata", None)
            if isinstance(metadata, dict):
                metadata["stranded_interjects"] = list(stranded)
            self._events.emit(
                "stranded_interject", {"prompts": stranded, "count": len(stranded)}
            )
        except Exception:
            logger.debug("stranded interject flush failed", exc_info=True)

    def _workspace(self) -> str:
        metadata = getattr(self._run_context, "_metadata", None)
        if isinstance(metadata, dict) and metadata.get("workspace"):
            return str(metadata["workspace"])
        return os.getcwd()
