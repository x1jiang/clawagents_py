"""ClawAgents harness adapter for the ATLAS runtime lifecycle."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from clawagents.atlas._runtime import require_atlas_runtime
from clawagents.atlas.config import resolve_atlas_config
from clawagents.atlas.gates import (
    FinalGateOutcome,
    evaluate_final_gate,
    harvest_pending,
    make_pending,
    record_evidence,
)
from clawagents.atlas.prompts import (
    build_reflection_prompt,
    render_recent_messages,
    standing_protocol,
)
from clawagents.atlas.session import AtlasRunSession, PendingKind
from clawagents.atlas.trace import abort_atlas_session, build_and_record_trace, end_atlas_session
from clawagents.providers.llm import LLMMessage

# Re-export for callers / tests
__all__ = ["AtlasAdapter", "BoundaryAction", "require_atlas_runtime"]


@dataclass(frozen=True)
class BoundaryAction:
    """Instruction for the agent loop after an ATLAS boundary event."""

    inject: str | None = None
    skip_rethink: bool = False
    continue_loop: bool = False
    allow_done: bool = False
    raise_error: str | None = None


class AtlasAdapter:
    """Owns one ATLAS session for a parent ClawAgents run."""

    def __init__(self, config: dict[str, Any] | None = None, *, atlas_config: Any = None):
        require_atlas_runtime()
        self.config = config if config is not None else resolve_atlas_config(atlas_config)
        self.run: AtlasRunSession | None = None

    @classmethod
    def maybe_create(
        cls,
        *,
        atlas: bool | None,
        atlas_config: Any = None,
    ) -> AtlasAdapter | None:
        from clawagents.atlas.config import resolve_atlas_enabled

        if not resolve_atlas_enabled(atlas):
            return None
        return cls(atlas_config=atlas_config)

    def start(self, task: str, *, session_id: str | None = None) -> AtlasRunSession:
        from atlas_runtime import start_session

        try:
            from finding import resolver as finding_resolver

            absent = finding_resolver.ABSENT
        except Exception:
            absent = None

        inherit = self.config.get("inherit", absent)
        if inherit is None:
            inherit = absent

        run_id = session_id or f"clawagents:{uuid.uuid4().hex}"
        kwargs: dict[str, Any] = {
            "trace_output": self.config["trace_output"],
            "session_id": run_id,
            "dashboard": bool(self.config.get("dashboard", False)),
            "generation_threshold": int(self.config.get("generation_threshold") or 5),
            "generation_stops": bool(self.config.get("generation_stops", False)),
            "skip_judge": bool(self.config.get("skip_judge", False)),
            "k_init": int(self.config.get("k_init") or 10),
            "k": int(self.config.get("k") or 20),
            "refinement_stops": bool(self.config.get("refinement_stops", False)),
            "advanced_refinement": bool(self.config.get("advanced_refinement", False)),
            "freeze": bool(self.config.get("freeze", False)),
            "max_retries": int(self.config.get("repair_rounds") or 3),
        }
        if self.config.get("atlas_model"):
            kwargs["atlas_model"] = self.config["atlas_model"]
        if self.config.get("store_dir"):
            kwargs["store_dir"] = self.config["store_dir"]
        if self.config.get("trace_root"):
            kwargs["trace_root"] = self.config["trace_root"]
        if self.config.get("evidence_export"):
            kwargs["evidence_export"] = self.config["evidence_export"]
        if self.config.get("repo"):
            kwargs["repo"] = self.config["repo"]
        if self.config.get("repo_path"):
            kwargs["repo_path"] = self.config["repo_path"]

        atlas_session = start_session(inherit, **kwargs)
        self.run = AtlasRunSession(
            atlas_session=atlas_session,
            config=self.config,
            run_id=run_id,
            task=task,
        )
        return self.run

    def protocol_for_system(self) -> str:
        if self.run is None:
            return standing_protocol(None)
        return standing_protocol(self.run.runtime_protocol)

    def note_tool_call(self) -> None:
        if self.run:
            self.run.note_tool_call()

    def on_tool_failure(
        self,
        messages: list[LLMMessage],
        *,
        tool_name: str,
        error: str | None,
    ) -> BoundaryAction:
        """Advisory checkpoint after a failed tool (throttled)."""
        run = self.run
        if run is None or run.pending is not None:
            return BoundaryAction()
        if not run.failure_throttle_ok():
            return BoundaryAction()
        prompt = self._build_advisory_prompt(
            messages,
            gate_label=f"tool failure ({tool_name})",
        )
        run.mark_failure_nudge()
        run.skip_rethink_this_round = True
        return BoundaryAction(inject=prompt, skip_rethink=True)

    def on_subagent_end(self, messages: list[LLMMessage], *, name: str = "task") -> BoundaryAction:
        run = self.run
        if run is None or run.pending is not None:
            return BoundaryAction()
        prompt = self._build_advisory_prompt(
            messages,
            gate_label=f"subagent stop ({name})",
        )
        return BoundaryAction(inject=prompt)

    def begin_final_gate(self, messages: list[LLMMessage]) -> BoundaryAction:
        """Inject the final submission gate before releasing the answer."""
        run = self.run
        if run is None:
            return BoundaryAction(allow_done=True)
        if run.final_gate_allowed is not None:
            return BoundaryAction(allow_done=True)
        if run.pending is not None and run.pending.kind == PendingKind.FINAL:
            return BoundaryAction()  # already waiting on reflection

        checkpoint_id = run.new_checkpoint_id()
        recent = render_recent_messages(
            messages,
            max_messages=int(run.config.get("recent_activity_messages") or 8),
            max_chars=int(run.config.get("recent_activity_chars") or 12_000),
        )
        suffix = (
            "\nThe runtime-counted value for `Repair attempts used:` is "
            f"{run.repair_attempts}. Emit that exact integer."
        )
        prompt = build_reflection_prompt(
            taxonomy_id=run.taxonomy_id,
            codes=run.taxonomy.get("codes") or [],
            checkpoint_id=checkpoint_id,
            gate_label="final submission gate",
            recent_activity=recent,
            full=True,
            prompt_suffix=suffix,
        )
        run.pending = make_pending(
            kind=PendingKind.FINAL,
            checkpoint_id=checkpoint_id,
            gate_label="final submission gate",
            full=True,
        )
        run.checkpoint_count += 1
        return BoundaryAction(inject=prompt, continue_loop=True)

    def process_assistant_text(
        self,
        messages: list[LLMMessage],
        text: str,
    ) -> BoundaryAction:
        """Harvest a pending reflection from the latest assistant text."""
        run = self.run
        if run is None or run.pending is None:
            return BoundaryAction()

        outcome = harvest_pending(run, text or "")
        if outcome.status == "format_retry" and outcome.repair_prompt:
            return BoundaryAction(inject=outcome.repair_prompt, continue_loop=True)

        if outcome.status == "give_up":
            # Advisory: drop and continue. Final: treat as format exhaustion.
            if run.pending.kind == PendingKind.ADVISORY:
                run.pending = None
                run.segment_start = max(1, len(messages) - 1)
                return BoundaryAction()
            run.pending = None
            policy = str(run.config.get("gate_exhaustion_policy") or "release")
            run.final_gate_allowed = False
            if policy == "raise":
                return BoundaryAction(
                    raise_error=outcome.error or "ATLAS reflection remained invalid",
                )
            return BoundaryAction(allow_done=True)

        # ok
        assert outcome.reflection is not None
        gate = (
            "clawagents_stop"
            if run.pending.kind == PendingKind.FINAL
            else "clawagents_checkpoint"
        )
        try:
            record_evidence(run, outcome.reflection, gate=gate)
        except Exception:
            pass

        if run.pending.kind == PendingKind.FINAL:
            run.pending = None
            gate_outcome = evaluate_final_gate(
                run,
                outcome.gate_text or text,
                pinned_status=outcome.pinned_status,
            )
            return self._map_final_outcome(gate_outcome)

        # Advisory accepted
        run.pending = None
        run.segment_start = max(1, len(messages) - 1)
        return BoundaryAction(
            inject=(
                "ATLAS checkpoint accepted. Apply the reflected change only if "
                "Decide required one, then continue the original task."
            ),
        )

    def consume_skip_rethink(self) -> bool:
        run = self.run
        if run is None or not run.skip_rethink_this_round:
            return False
        run.skip_rethink_this_round = False
        return True

    def finalize(
        self,
        messages: list[LLMMessage],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any | None:
        run = self.run
        if run is None or run.finalized:
            return None
        try:
            build_and_record_trace(run, messages, metadata=metadata)
            ended = end_atlas_session(run)
            run.finalized = True
            return ended
        except Exception:
            abort_atlas_session(run)
            run.finalized = True
            raise

    def abort(self) -> None:
        if self.run is not None and not self.run.finalized:
            abort_atlas_session(self.run)
            self.run.finalized = True

    # ── internals ─────────────────────────────────────────────────────

    def _build_advisory_prompt(
        self,
        messages: list[LLMMessage],
        *,
        gate_label: str,
    ) -> str:
        run = self.run
        assert run is not None
        checkpoint_id = run.new_checkpoint_id()
        recent = render_recent_messages(
            messages,
            max_messages=int(run.config.get("recent_activity_messages") or 8),
            max_chars=int(run.config.get("recent_activity_chars") or 12_000),
            segment_start=run.segment_start,
        )
        prompt = build_reflection_prompt(
            taxonomy_id=run.taxonomy_id,
            codes=run.taxonomy.get("codes") or [],
            checkpoint_id=checkpoint_id,
            gate_label=gate_label,
            recent_activity=recent,
            full=False,
        )
        run.pending = make_pending(
            kind=PendingKind.ADVISORY,
            checkpoint_id=checkpoint_id,
            gate_label=gate_label,
            full=False,
        )
        run.checkpoint_count += 1
        return prompt

    def _map_final_outcome(self, outcome: FinalGateOutcome) -> BoundaryAction:
        if outcome.status == "allow":
            return BoundaryAction(allow_done=True)
        if outcome.status == "repair" and outcome.prompt:
            return BoundaryAction(inject=outcome.prompt, continue_loop=True)
        if outcome.status == "raise":
            return BoundaryAction(raise_error=outcome.reason or "ATLAS gate exhausted")
        # release
        return BoundaryAction(allow_done=True)
