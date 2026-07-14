"""Harvest ATLAS reflections and evaluate final-submission gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from clawagents.atlas.prompts import build_format_repair
from clawagents.atlas.session import AtlasRunSession, PendingKind, PendingReflection


@dataclass(frozen=True)
class HarvestOutcome:
    """Result of trying to parse a pending reflection from assistant text."""

    status: Literal["ok", "format_retry", "give_up"]
    reflection: Any | None = None
    repair_prompt: str | None = None
    error: str | None = None
    gate_text: str | None = None
    pinned_status: str | None = None


@dataclass(frozen=True)
class FinalGateOutcome:
    status: Literal["allow", "repair", "format_retry", "release", "raise"]
    reason: str = ""
    prompt: str | None = None
    gate_text: str | None = None


def harvest_pending(
    run: AtlasRunSession,
    text: str,
) -> HarvestOutcome:
    """Parse the assistant's response against the pending reflection request."""
    from atlas_runtime import harvest_reflection

    pending = run.pending
    if pending is None:
        return HarvestOutcome(status="give_up", error="no pending reflection")

    format_retries = int(run.config.get("format_retries") or 2)
    harvest = harvest_reflection(
        text,
        checkpoint_id=pending.checkpoint_id,
        known_code_ids=run.known_code_ids,
    )
    if harvest.result is not None:
        return HarvestOutcome(
            status="ok",
            reflection=harvest.result,
            gate_text=text,
            pinned_status=pending.pinned_status,
        )

    partial = harvest.partial
    pinned = pending.pinned_status
    if (
        pending.full
        and partial is not None
        and getattr(partial, "has_block", False)
        and getattr(partial, "status", None)
        and pinned is None
    ):
        pinned = partial.status
        pending.pinned_status = pinned

    if pending.format_attempt >= format_retries:
        return HarvestOutcome(
            status="give_up",
            error=str(getattr(harvest, "error", None) or "invalid reflection"),
            pinned_status=pinned,
        )

    pending.format_attempt += 1
    if partial is not None and getattr(partial, "has_block", False):
        repair = build_format_repair(
            checkpoint_id=pending.checkpoint_id,
            issues=list(getattr(partial, "issues", None) or []),
            full=pending.full,
        )
    else:
        repair = (
            f"ATLAS reflection was invalid: {getattr(harvest, 'error', 'unknown')}. "
            f"Re-emit the complete reflection for Checkpoint ID "
            f"{pending.checkpoint_id} in the exact required shape."
        )
    return HarvestOutcome(
        status="format_retry",
        repair_prompt=repair,
        error=str(getattr(harvest, "error", None) or ""),
        pinned_status=pinned,
    )


def evaluate_final_gate(
    run: AtlasRunSession,
    gate_text: str,
    *,
    pinned_status: str | None,
) -> FinalGateOutcome:
    """Run ATLAS ``pre_submission`` and map the decision to harness actions."""
    from atlas_runtime import pin_gate_decision, pre_submission

    repair_rounds = int(run.config.get("repair_rounds") or 3)
    decision, _flipped = pin_gate_decision(
        pre_submission(run.atlas_session, gate_text),
        pinned_status,
        max_retries=repair_rounds,
    )
    if decision.allow:
        run.final_gate_allowed = True
        return FinalGateOutcome(status="allow", gate_text=gate_text)

    run.repair_attempts += 1
    if run.repair_attempts > repair_rounds:
        policy = str(run.config.get("gate_exhaustion_policy") or "release")
        run.final_gate_allowed = False
        if policy == "raise":
            return FinalGateOutcome(
                status="raise",
                reason=str(decision.reason or "ATLAS final repair limit exceeded"),
                gate_text=gate_text,
            )
        return FinalGateOutcome(
            status="release",
            reason=str(decision.reason or "repair budget exhausted"),
            gate_text=gate_text,
        )

    prompt = (
        f"ATLAS blocked completion: {decision.reason}. "
        "Perform the focused repair from Decide, verify it, "
        "and return a corrected proposed final answer."
    )
    return FinalGateOutcome(
        status="repair",
        reason=str(decision.reason or ""),
        prompt=prompt,
        gate_text=gate_text,
    )


def record_evidence(
    run: AtlasRunSession,
    reflection: Any,
    *,
    gate: str,
) -> None:
    from atlas_runtime import record_reflection
    from pathlib import Path

    trace_output = Path(run.config["trace_output"])
    record_reflection(
        trace_output,
        {
            "taxonomy_id": run.taxonomy_id,
            "session_id": run.run_id,
        },
        reflection,
        gate=gate,
        task_id=run.run_id,
    )


def make_pending(
    *,
    kind: PendingKind,
    checkpoint_id: str,
    gate_label: str,
    full: bool,
) -> PendingReflection:
    return PendingReflection(
        kind=kind,
        checkpoint_id=checkpoint_id,
        gate_label=gate_label,
        full=full,
    )
