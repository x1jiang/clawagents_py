"""Task-level Propose–Probe–Compose (PPC) over a harness tree.

``harness_surface`` holds the pieces Harness² (arXiv:2610.14224,
github.com/google-research/harness2) is built from — the editable surface and
its validation, the INV/DOM and compose prompts, blind selection, and the
experience library. This module is the loop that connects them:

* **Propose** — from the same base evidence, an ``INV`` editor (how the agent
  works) and a ``DOM`` editor (what a practitioner knows) each propose harness
  edits. Edits are "omitted is inherited": a proposal returns only the files
  it changes, and ``KEEP_BASE`` is a valid answer.
* **Probe** — every candidate harness is rolled out once on the task, so each
  edit is judged by what it did rather than by how plausible it sounds.
* **Compose** — a synthesiser sees every candidate with its probe and keeps
  only what the evidence supports; the composed harness is rolled out too.

Two budgets, both ``4k`` rollouts and ``3k`` edit stages for width ``k``:

* ``parallel`` — ``k`` base rollouts, ``2k`` candidates, ``k`` independent
  compositions. Wide: more independent draws from the same evidence.
* ``sequential`` — ``k`` rounds; each round's composition becomes the next
  round's checkpoint. Deep: later rounds build on verified edits.

The final harness is picked blind from the ``k`` compositions (never from
scores — nothing here sees a grader). The loop is deliberately independent of
how a harness is executed or edited: callers supply ``rollout`` and
``propose`` callables, which keeps it testable without a model and usable
with any agent runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

from clawagents.trajectory.harness_context import (
    DeliverablesMode,
    RolloutEvidence,
    materialize_evidence_workspace,
)
from clawagents.trajectory.harness_ledger import BudgetExceededError, HarnessLedger
from clawagents.trajectory.harness_runmark import (
    inputs_fingerprint,
    prepare_clean_run_dir,
    tree_fingerprint,
    write_runmark,
)
from clawagents.trajectory.harness_surface import (
    DEFAULT_SURFACE,
    REJECTION_TEXT_MAX,
    ExperienceEntry,
    ExperienceLibrary,
    SurfaceSpec,
    Tree,
    accept_or_repair_tree,
    assign_anon_ids,
    build_blind_selector_prompt,
    build_compose_prompt,
    build_propose_prompt,
    diff_summary,
    lint_task_prompt,
    parse_choice_reply,
    parse_experience_reply,
    tree_diversity,
)

Mode = Literal["parallel", "sequential"]

# One proposal session plus at most one retry carrying the rejection text.
STAGE_ATTEMPTS = 2
# How much of a rollout transcript a later stage is shown.
TRAJECTORY_EXCERPT_MAX = 6_000

_INCOMPLETE_RE = re.compile(
    r"\b(?:TODO|FIXME|TBD|PLACEHOLDER)\b|\[interrupted\]|\[cancelled\]|Reached maximum",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rollout:
    """What one execution of the task under a harness produced."""

    ok: bool
    delivered: bool = False
    deliverable: str = ""
    """Text of the produced deliverable (what blind selection compares)."""
    trajectory: str = ""
    """Transcript (or a summary of it) shown to later edit stages."""
    error: str = ""
    usd: float = 0.0
    deliverable_files: Mapping[str, str] = field(default_factory=dict)
    confidence: float | None = None

    @property
    def selectable(self) -> bool:
        return self.ok and self.delivered and bool(self.deliverable.strip())


def score_rollout_confidence(rollout: Rollout | None) -> float:
    """DSpark-inspired deterministic confidence score ``[0.0, 1.0]`` for a rollout.

    Used when ``adaptive_probe=True`` to short-circuit speculative candidate/compose
    rounds when all base rollouts already produce complete, clean deliverables.
    """
    if rollout is None or not rollout.selectable:
        return 0.0
    if rollout.confidence is not None:
        return max(0.0, min(1.0, float(rollout.confidence)))
    body = rollout.deliverable.strip()
    score = 0.85 if len(body) >= 40 else 0.60
    if _INCOMPLETE_RE.search(body):
        score -= 0.45
    if rollout.error.strip():
        score -= 0.40
    if len(body) >= 80 and not _INCOMPLETE_RE.search(body):
        score += 0.08
    traj = rollout.trajectory.strip()
    if traj and "Traceback (most recent call last)" not in traj[-800:]:
        score += 0.05
    return round(max(0.0, min(1.0, score)), 4)


@dataclass(frozen=True)
class Proposal:
    """A harness edit: only the files that change. Omitted files are inherited."""

    edits: Mapping[str, str] = field(default_factory=dict)
    removed: tuple[str, ...] = ()
    notes: str = ""
    keep_base: bool = False
    usd: float = 0.0

    @property
    def is_noop(self) -> bool:
        return self.keep_base or (not self.edits and not self.removed)


RolloutFn = Callable[[Tree, str], Awaitable[Rollout]]
"""``(harness_tree, label) -> Rollout``. Runs the task once under that harness."""
ProposeFn = Callable[[str, str], Awaitable[Proposal]]
"""``(prompt, label) -> Proposal``. One edit session (INV, DOM or compose)."""
SelectFn = Callable[[str], Awaitable[str]]
"""``selector_prompt -> reply`` ending in a ```` ```choice ```` block or JSON choice."""


@dataclass
class StageRecord:
    label: str
    kind: str  # base | INV | DOM | compose
    tree: Tree
    rollout: Rollout | None = None
    parent: str = ""
    kept_base: bool = False
    fell_back: bool = False
    error: str = ""
    notes: str = ""
    attempts: int = 0
    autotriggered: tuple[str, ...] = ()
    cached: bool = False
    reused_from: str = ""
    skipped: bool = False
    skip_reason: str = ""
    evidence_workspace: str = ""


@dataclass
class PPCResult:
    task_id: str
    mode: Mode
    k: int
    stages: list[StageRecord]
    finals: list[str]
    chosen: str
    selection: dict[str, Any]
    ledger: dict[str, int]
    diversity: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    harness_ledger: HarnessLedger | None = None

    def stage(self, label: str) -> StageRecord:
        for s in self.stages:
            if s.label == label:
                return s
        raise KeyError(label)

    @property
    def adopted(self) -> StageRecord:
        return self.stage(self.chosen)

    @property
    def adopted_tree(self) -> Tree:
        return dict(self.adopted.tree)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "k": self.k,
            "finals": list(self.finals),
            "chosen": self.chosen,
            "selection": self.selection,
            "ledger": dict(self.ledger),
            "harness_ledger": self.harness_ledger.to_dict() if self.harness_ledger else None,
            "diversity": self.diversity,
            "warnings": list(self.warnings),
            "stages": [
                {
                    "label": s.label,
                    "kind": s.kind,
                    "parent": s.parent,
                    "kept_base": s.kept_base,
                    "fell_back": s.fell_back,
                    "error": s.error,
                    "attempts": s.attempts,
                    "autotriggered": list(s.autotriggered),
                    "cached": s.cached,
                    "reused_from": s.reused_from,
                    "skipped": s.skipped,
                    "skip_reason": s.skip_reason,
                    "evidence_workspace": s.evidence_workspace,
                    "files": sorted(s.tree),
                    "rollout": None
                    if s.rollout is None
                    else {
                        "ok": s.rollout.ok,
                        "delivered": s.rollout.delivered,
                        "deliverable_chars": len(s.rollout.deliverable),
                        "error": s.rollout.error,
                        "usd": s.rollout.usd,
                        "confidence": score_rollout_confidence(s.rollout),
                    },
                }
                for s in self.stages
            ],
        }


def apply_edits(parent: Mapping[str, str], proposal: Proposal) -> Tree:
    """Overlay a proposal on its parent tree (omitted files are inherited)."""
    tree = dict(parent)
    for rel in proposal.removed:
        tree.pop(rel, None)
    for rel, content in proposal.edits.items():
        tree[rel] = content
    return tree


class ProposerCache:
    """Content-addressed store of proposal replies.

    The key covers the stage label and the full prompt, which already embeds
    every piece of evidence the session saw. A re-run with identical evidence
    reuses the reply; any change in the evidence is a different key. Each
    entry also keeps the prompt, so a run can be audited after the fact.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @staticmethod
    def key(label: str, prompt: str) -> str:
        return hashlib.sha256(f"{label}\0{prompt}".encode("utf-8")).hexdigest()[:16]

    def _path(self, label: str, prompt: str) -> Path:
        return self.root / f"{self.key(label, prompt)}.json"

    def get(self, label: str, prompt: str) -> Proposal | None:
        path = self._path(label, prompt)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        reply = raw.get("proposal") if isinstance(raw, dict) else None
        if not isinstance(reply, dict):
            return None
        return Proposal(
            edits={str(k): str(v) for k, v in (reply.get("edits") or {}).items()},
            removed=tuple(str(r) for r in reply.get("removed") or ()),
            notes=str(reply.get("notes") or ""),
            keep_base=bool(reply.get("keep_base")),
        )

    def put(self, label: str, prompt: str, proposal: Proposal) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": label,
            "prompt": prompt,
            "proposal": {
                "edits": dict(proposal.edits),
                "removed": list(proposal.removed),
                "notes": proposal.notes,
                "keep_base": proposal.keep_base,
            },
        }
        tmp = self._path(label, prompt).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path(label, prompt))

    def wrap(self, propose: ProposeFn) -> ProposeFn:
        async def cached(prompt: str, label: str) -> Proposal:
            hit = self.get(label, prompt)
            if hit is not None:
                return hit
            proposal = await propose(prompt, label)
            self.put(label, prompt, proposal)
            return proposal

        return cached


def _excerpt(text: str, limit: int = TRAJECTORY_EXCERPT_MAX) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}\n[... {len(text) - limit:,} chars omitted ...]\n{text[-(limit - head):]}"


def _rejection_feedback(prompt: str, error: str) -> str:
    return (
        f"{prompt}\n\n# YOUR PREVIOUS ATTEMPT WAS REJECTED\n\n"
        f"{error[:REJECTION_TEXT_MAX]}\n\n"
        "Fix exactly these problems and return the corrected edit, or answer `KEEP_BASE` "
        "if the edit cannot be made valid."
    )


def _candidate_summary(surface: SurfaceSpec, parent: Tree, stage: StageRecord) -> str:
    """What a compose stage is told about one probed candidate."""
    if stage.fell_back:
        status = f"REJECTED — fell back to its parent ({stage.error or 'invalid edit'})"
    elif stage.kept_base:
        status = "KEEP_BASE — no change proposed"
    else:
        d = diff_summary(surface, parent, stage.tree)
        status = (
            f"added {d['added'] or '[]'}, modified {d['modified'] or '[]'}, "
            f"removed {d['removed'] or '[]'}, always-on {d['always_on_char_delta']:+d} chars"
        )
    r = stage.rollout
    if r is None:
        probe = "not rolled out"
    elif not r.ok:
        probe = f"rollout FAILED ({r.error or 'no detail'})"
    else:
        probe = (
            f"rollout ok, deliverable {'present' if r.delivered else 'MISSING'} "
            f"({len(r.deliverable):,} chars)"
        )
    parts = [f"- dimension: {stage.kind}", f"- edit: {status}", f"- probe: {probe}"]
    if stage.notes.strip():
        parts.append(f"- editor notes: {stage.notes.strip()[:1_500]}")
    if r is not None and r.trajectory.strip():
        parts.append(f"\nProbe transcript:\n{_excerpt(r.trajectory, 3_000)}")
    return "\n".join(parts)


def _save_rollout_payload(stage_dir: Path, r: Rollout) -> None:
    payload = {
        "ok": r.ok,
        "delivered": r.delivered,
        "deliverable": r.deliverable,
        "trajectory": r.trajectory,
        "error": r.error,
        "usd": r.usd,
        "deliverable_files": dict(r.deliverable_files),
        "confidence": r.confidence,
    }
    tmp = stage_dir / "rollout.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(stage_dir / "rollout.json")


def _load_rollout_payload(stage_dir: Path) -> Rollout | None:
    p = stage_dir / "rollout.json"
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    raw_files = raw.get("deliverable_files")
    files_map = (
        {str(k): str(v) for k, v in raw_files.items()}
        if isinstance(raw_files, dict)
        else {}
    )
    conf_raw = raw.get("confidence")
    return Rollout(
        ok=bool(raw.get("ok")),
        delivered=bool(raw.get("delivered")),
        deliverable=str(raw.get("deliverable") or ""),
        trajectory=str(raw.get("trajectory") or ""),
        error=str(raw.get("error") or ""),
        usd=float(raw.get("usd") or 0.0),
        deliverable_files=files_map,
        confidence=float(conf_raw) if isinstance(conf_raw, (int, float)) else None,
    )


def _to_rollout_evidence(records: Sequence[StageRecord]) -> list[RolloutEvidence]:
    out: list[RolloutEvidence] = []
    for rec in records:
        r = rec.rollout
        if r is None:
            continue
        out.append(
            RolloutEvidence(
                label=rec.label,
                ok=r.ok,
                delivered=r.delivered,
                trajectory=r.trajectory,
                deliverable=r.deliverable,
                deliverable_files=r.deliverable_files,
                error=r.error,
            )
        )
    return out


async def propose_probe_compose(
    task: str,
    base_tree: Mapping[str, str],
    *,
    rollout: RolloutFn,
    propose: ProposeFn,
    select: SelectFn | None = None,
    k: int = 1,
    mode: Mode = "parallel",
    task_id: str = "task",
    surface: SurfaceSpec = DEFAULT_SURFACE,
    library: ExperienceLibrary | None = None,
    cache: ProposerCache | None = None,
    concurrency: int = 2,
    stage_attempts: int = STAGE_ATTEMPTS,
    allow_subagents: bool = True,
    adaptive_probe: bool = False,
    confidence_threshold: float = 0.90,
    max_usd: float | None = None,
    run_dir: Path | str | None = None,
    deliverables_mode: DeliverablesMode = "contents",
    source_dir: Path | str | None = None,
    on_stage_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> PPCResult:
    """Run one task through Propose–Probe–Compose and pick a harness blind.

    ``rollout`` and ``propose`` are the only places a model is involved, so a
    caller decides how a harness is executed and edited. Failures are
    contained: a proposal that cannot be made valid falls back to its parent
    tree, a failed rollout is recorded rather than raised, and selection falls
    back to a delivered base when no composition is usable.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if mode not in ("parallel", "sequential"):
        raise ValueError(f"mode must be 'parallel' or 'sequential', got {mode!r}")

    base = dict(base_tree)
    stages: list[StageRecord] = []
    ledger = {"rollouts": 0, "stages": 0}
    h_ledger = HarnessLedger(
        mode=mode,
        task_id=task_id,
        k=k,
        max_usd=max_usd,
        adaptive_probe=adaptive_probe,
    )
    run_root = Path(run_dir) if run_dir is not None else None
    in_digest = inputs_fingerprint(task, source_dir) if run_root is not None else None
    probed_by_fp: dict[str, StageRecord] = {}

    warnings: list[str] = []
    leaked = lint_task_prompt(task)
    if leaked:
        warnings.append(
            "task text mentions measurement words "
            f"{leaked}: a model told it is being measured tends to optimise for the measurement"
        )
    gate = asyncio.Semaphore(max(1, concurrency))
    library_index = ""
    library_ctx_dir = library.context_dir() if library is not None else None
    if library is not None and library_ctx_dir is not None:
        try:
            library_index = library.index_path.read_text(encoding="utf-8")
        except OSError:
            library_index = ""

    def _emit_stage(event_kind: str, payload: dict[str, Any]) -> None:
        if on_stage_event is not None:
            try:
                on_stage_event(event_kind, payload)
            except Exception:
                pass

    async def probe(stage: StageRecord) -> StageRecord:
        fp = tree_fingerprint(stage.tree)
        if adaptive_probe and stage.kind != "base" and fp in probed_by_fp:
            prior = probed_by_fp[fp]
            if prior.rollout is not None and prior.rollout.ok:
                stage.rollout = prior.rollout
                stage.reused_from = prior.label
                h_ledger.record_rollout(stage.label, reused_fingerprint=fp)
                _emit_stage(
                    "stage_completed",
                    {"label": stage.label, "kind": stage.kind, "reused_from": prior.label},
                )
                return stage

        if run_root is not None:
            stage_dir = run_root / "rollouts" / stage.label
            if prepare_clean_run_dir(
                stage_dir,
                expected_harness=stage.tree,
                expected_inputs_digest=in_digest,
            ):
                loaded = _load_rollout_payload(stage_dir)
                if loaded is not None:
                    stage.rollout = loaded
                    stage.cached = True
                    h_ledger.record_rollout(stage.label, cached=True)
                    if loaded.ok:
                        probed_by_fp.setdefault(fp, stage)
                    _emit_stage(
                        "stage_completed",
                        {"label": stage.label, "kind": stage.kind, "cached": True},
                    )
                    return stage

        async with gate:
            h_ledger.check_can_proceed()
            _emit_stage("stage_started", {"label": stage.label, "kind": stage.kind, "phase": "probe"})
            ledger["rollouts"] += 1
            try:
                stage.rollout = await rollout(dict(stage.tree), stage.label)
            except (BudgetExceededError, asyncio.CancelledError):
                raise
            except Exception as exc:  # a crashed rollout is evidence, not a crash
                stage.rollout = Rollout(
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}"[:REJECTION_TEXT_MAX],
                )
            r_usd = float(stage.rollout.usd) if stage.rollout is not None else 0.0
            h_ledger.record_rollout(stage.label, usd=r_usd)
            if stage.rollout is not None and stage.rollout.ok:
                probed_by_fp.setdefault(fp, stage)
                if run_root is not None:
                    stage_dir = run_root / "rollouts" / stage.label
                    _save_rollout_payload(stage_dir, stage.rollout)
                    write_runmark(
                        stage_dir,
                        run_id=stage.label,
                        harness=stage.tree,
                        inputs_digest=in_digest,
                    )
            _emit_stage(
                "stage_completed",
                {
                    "label": stage.label,
                    "kind": stage.kind,
                    "ok": bool(stage.rollout and stage.rollout.ok),
                    "usd": h_ledger.usd,
                },
            )
        return stage

    async def edit_stage(
        label: str,
        kind: str,
        prompt: str,
        parent_label: str,
        parent: Tree,
        evidence_records: Sequence[StageRecord] = (),
        decisions: Mapping[str, str] | None = None,
    ) -> StageRecord:
        """One edit session with accept-or-repair; falls back to ``parent``."""
        ledger["stages"] += 1
        record = StageRecord(label=label, kind=kind, tree=dict(parent), parent=parent_label)
        if run_root is not None:
            ws_dir = run_root / "sessions" / label
            ev_ws = materialize_evidence_workspace(
                ws_dir,
                task=task,
                parent_harness=parent,
                rollouts=_to_rollout_evidence(evidence_records),
                deliverables_mode=deliverables_mode,
                source_dir=source_dir,
                decisions=decisions,
                library_dir=library_ctx_dir,
            )
            record.evidence_workspace = str(ev_ws.root)

        current_prompt = prompt
        session_usd = 0.0
        all_cached = True
        for attempt in range(1, max(1, stage_attempts) + 1):
            record.attempts = attempt
            proposal: Proposal | None = None
            hit = cache.get(label, current_prompt) if cache is not None else None
            if hit is not None:
                proposal = hit
            else:
                all_cached = False
                try:
                    async with gate:
                        h_ledger.check_can_proceed()
                        _emit_stage(
                            "stage_started",
                            {"label": label, "kind": kind, "phase": "propose", "attempt": attempt},
                        )
                        proposal = await propose(current_prompt, label)
                        session_usd += float(proposal.usd or 0.0)
                        if cache is not None:
                            cache.put(label, current_prompt, proposal)
                except (BudgetExceededError, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    record.error = f"proposer failed: {type(exc).__name__}: {exc}"[:REJECTION_TEXT_MAX]
                    continue

            record.notes = proposal.notes
            if proposal.is_noop:
                record.tree, record.kept_base, record.fell_back, record.error = (
                    dict(parent),
                    True,
                    False,
                    "",
                )
                record.cached = all_cached and cache is not None
                h_ledger.record_session(
                    label,
                    usd=session_usd,
                    cached=record.cached,
                    attempts=attempt,
                )
                return record
            accepted, kept_base, fell_back, error, triggered = accept_or_repair_tree(
                apply_edits(parent, proposal),
                parent,
                surface,
                allow_subagents=allow_subagents,
                attempts=1,
            )
            record.autotriggered = tuple(triggered)
            if not fell_back:
                record.tree, record.kept_base, record.fell_back, record.error = (
                    accepted,
                    kept_base,
                    False,
                    "",
                )
                record.cached = all_cached and cache is not None
                h_ledger.record_session(
                    label,
                    usd=session_usd,
                    cached=record.cached,
                    attempts=attempt,
                )
                return record
            record.error = error
            current_prompt = _rejection_feedback(prompt, error)
        record.tree, record.kept_base, record.fell_back = dict(parent), False, True
        record.cached = all_cached and cache is not None
        h_ledger.record_session(
            label,
            usd=session_usd,
            cached=record.cached,
            attempts=record.attempts,
        )
        return record

    def evidence(records: Sequence[StageRecord]) -> list[str]:
        return [
            _excerpt(r.rollout.trajectory) if r.rollout and r.rollout.trajectory else "(no transcript)"
            for r in records
        ]

    def propose_prompt(dim: str, records: Sequence[StageRecord]) -> str:
        return build_propose_prompt(task, dim, evidence(records), library_index=library_index)

    def compose_prompt(parent: Tree, candidates: Sequence[StageRecord]) -> str:
        return build_compose_prompt(
            task, {c.label: _candidate_summary(surface, parent, c) for c in candidates}
        )

    # ── shared base rollouts ────────────────────────────────────────────
    bases = [StageRecord(label=f"base{i}", kind="base", tree=dict(base)) for i in range(1, k + 1)]
    await asyncio.gather(*(probe(b) for b in bases))
    stages.extend(bases)

    finals: list[StageRecord] = []
    base_confident = adaptive_probe and all(
        b.rollout is not None
        and b.rollout.selectable
        and score_rollout_confidence(b.rollout) >= confidence_threshold
        for b in bases
    )

    if base_confident:
        # DSpark-inspired confidence short-circuit: record skipped edit/probe
        # stages so the 4k/3k accounting identity still holds while saving 3k
        # edit sessions and 3k rollouts.
        for i in range(1, k + 1):
            for dim in ("INV", "DOM"):
                lbl = f"{dim}{i}"
                cand_rec = StageRecord(
                    label=lbl,
                    kind=dim,
                    tree=dict(base),
                    rollout=bases[i - 1].rollout,
                    parent="base",
                    kept_base=True,
                    skipped=True,
                    skip_reason="base_confident",
                )
                h_ledger.record_session(lbl, skipped=True, skip_reason="base_confident")
                h_ledger.record_rollout(lbl, skipped=True, skip_reason="base_confident")
                stages.append(cand_rec)
            fin_lbl = f"agg{i}" if mode == "parallel" else f"c{i}"
            fin_rec = StageRecord(
                label=fin_lbl,
                kind="compose",
                tree=dict(base),
                rollout=bases[i - 1].rollout,
                parent="base",
                kept_base=True,
                skipped=True,
                skip_reason="base_confident",
            )
            h_ledger.record_session(fin_lbl, skipped=True, skip_reason="base_confident")
            h_ledger.record_rollout(fin_lbl, skipped=True, skip_reason="base_confident")
            stages.append(fin_rec)
            finals.append(fin_rec)
    elif mode == "parallel":
        candidates = await asyncio.gather(*(
            edit_stage(f"{dim}{i}", dim, propose_prompt(dim, bases), "base", base, bases)
            for i in range(1, k + 1)
            for dim in ("INV", "DOM")
        ))
        await asyncio.gather(*(probe(c) for c in candidates))
        stages.extend(candidates)
        prompt = compose_prompt(base, candidates)
        cand_decisions = {c.label: c.notes for c in candidates if c.notes.strip()}
        finals = list(await asyncio.gather(*(
            edit_stage(
                f"agg{i}",
                "compose",
                prompt,
                "base",
                base,
                [*bases, *candidates],
                cand_decisions,
            )
            for i in range(1, k + 1)
        )))
        await asyncio.gather(*(probe(f) for f in finals))
        stages.extend(finals)
    else:
        checkpoint, checkpoint_label, seen = base, "base", list(bases)
        prior_decisions: dict[str, str] = {}
        for t in range(1, k + 1):
            round_candidates = list(await asyncio.gather(*(
                edit_stage(
                    f"{dim}{t}",
                    dim,
                    propose_prompt(dim, seen),
                    checkpoint_label,
                    checkpoint,
                    seen,
                    prior_decisions,
                )
                for dim in ("INV", "DOM")
            )))
            await asyncio.gather(*(probe(c) for c in round_candidates))
            stages.extend(round_candidates)
            for rc in round_candidates:
                if rc.notes.strip():
                    prior_decisions[rc.label] = rc.notes
            composed = await edit_stage(
                f"c{t}",
                "compose",
                compose_prompt(checkpoint, round_candidates),
                checkpoint_label,
                checkpoint,
                [*seen, *round_candidates],
                prior_decisions,
            )
            if composed.notes.strip():
                prior_decisions[composed.label] = composed.notes
            await probe(composed)
            stages.append(composed)
            finals.append(composed)
            checkpoint, checkpoint_label, seen = dict(composed.tree), composed.label, [composed]

    h_ledger.assert_identity()
    selection = await _select_final(task, task_id, k, mode, finals, bases, select)
    _emit_stage("selection_completed", {"chosen": selection["chosen"], "selection": selection})
    return PPCResult(
        task_id=task_id,
        mode=mode,
        k=k,
        stages=stages,
        finals=[f.label for f in finals],
        chosen=selection["chosen"],
        selection=selection,
        ledger=ledger,
        diversity=tree_diversity({f.label: f.tree for f in finals}),
        warnings=warnings,
        harness_ledger=h_ledger,
    )


async def _select_final(
    task: str,
    task_id: str,
    k: int,
    mode: str,
    finals: Sequence[StageRecord],
    bases: Sequence[StageRecord],
    select: SelectFn | None,
) -> dict[str, Any]:
    """Blind pick among the finals; base rollouts are never in the pool."""
    pool = [f for f in finals if f.rollout is not None and f.rollout.selectable]
    dropped = [f.label for f in finals if f not in pool]

    def fallback(reason: str) -> dict[str, Any]:
        delivered_base = next((b for b in bases if b.rollout is not None and b.rollout.selectable), None)
        pick = pool[0] if pool else (delivered_base or finals[0])
        return {"chosen": pick.label, "fallback": True, "trivial": False,
                "why": reason, "dropped": dropped, "id_map": {}}

    if not pool:
        return fallback("no composition produced a usable deliverable")
    if len(pool) == 1:
        return {"chosen": pool[0].label, "fallback": False, "trivial": True,
                "why": "single usable composition; selected without a selector call",
                "dropped": dropped, "id_map": {}}
    if select is None:
        return fallback("no selector supplied; kept the first usable composition")

    by_label = {f.label: f for f in pool}
    id_map = assign_anon_ids(list(by_label), task_id=task_id, k=k, mode=mode)
    artifacts = {anon: by_label[label].rollout.deliverable for anon, label in id_map.items()}  # type: ignore[union-attr]
    try:
        reply = await select(build_blind_selector_prompt(task, artifacts))
        anon, label, why = parse_choice_reply(reply, id_map)
    except Exception as exc:
        out = fallback(f"selector failed ({type(exc).__name__}: {exc})"[:REJECTION_TEXT_MAX])
        out["id_map"] = id_map
        return out
    return {"chosen": label, "chosen_id": anon, "fallback": False, "trivial": False,
            "why": why, "dropped": dropped, "id_map": id_map}


def build_reflection_prompt(task: str, result: PPCResult, surface: SurfaceSpec = DEFAULT_SURFACE) -> str:
    """Post-adoption reflection over the whole record, rejected candidates included.

    Asks for an ``<edits>`` block (one causal verdict per changed component)
    and an ``<experience>`` block, in the format ``parse_experience_reply``
    reads. Adoption alone is not evidence: with several components changed at
    once or a single rollout behind a claim, the honest verdict is
    ``confounded``.
    """
    base = result.stage("base1").tree
    blocks = []
    for stage in result.stages:
        if stage.kind == "base":
            continue
        tag = " (ADOPTED)" if stage.label == result.chosen else ""
        blocks.append(f"### `{stage.label}`{tag}\n{_candidate_summary(surface, base, stage)}")
    why = str(result.selection.get("why") or "").strip() or "(none recorded)"
    return (
        "A harness for the task below was just adopted. Work out what each edit actually did.\n\n"
        f"# THE TASK\n\n{task.strip()}\n\n"
        f"# THE RECORD ({result.mode}, k={result.k})\n\n" + "\n\n".join(blocks) + "\n\n"
        f"# WHY THE ADOPTED ONE WAS PICKED\n\n{why}\n\n"
        "# OUTPUT\n\n"
        "<edits>\n"
        "One record per changed component, fields on separate lines:\n"
        "component: <path>\ncandidates: <labels that carried it>\nchange: <what changed>\n"
        "observed: <what differed in the rollouts that had it>\n"
        "verdict: helped | hurt | no observed effect | confounded\n"
        "Use `confounded` whenever several components changed together or one rollout is the "
        "only support.\n"
        "</edits>\n\n"
        "<experience>\n"
        "cue: <the situation in which this applies>\n"
        "lesson: <what to do, stated so it transfers to a different task>\n"
        "evidence: <the rollout moment that supports it>\n"
        "confidence: low | medium | high\n"
        "dimension: INV | DOM\n"
        "</experience>\n\n"
        "A lesson that rests only on `confounded` rows must be narrowed or dropped: answer "
        "`NO_ENTRY` instead of the two blocks.\n"
    )


def record_experience(
    result: PPCResult,
    library: ExperienceLibrary,
    reflection_reply: str,
    *,
    run_id: str = "",
) -> ExperienceEntry | None:
    """Store the adopted harness and its lesson, if the reflection earned one.

    Returns ``None`` (and writes nothing) when the reply is ``NO_ENTRY``, lacks
    a cue/lesson/evidence, or its edits are all confounded or without effect.
    """
    entry = parse_experience_reply(reflection_reply)
    if entry is None:
        return None
    library.add(
        result.task_id,
        entry,
        why=str(result.selection.get("why") or ""),
        label=result.chosen,
        run_id=run_id,
        harness=result.adopted_tree,
    )
    return entry


def read_tree(workspace: Path | str, surface: SurfaceSpec = DEFAULT_SURFACE) -> Tree:
    """Collect the editable harness surface of a workspace into a tree."""
    ws = Path(workspace)
    tree: Tree = {}
    singles = list(surface.always_on) + ([surface.hooks_file] if surface.hooks_file else [])
    for rel in singles:
        path = ws / rel
        if path.is_file():
            tree[rel] = path.read_text(encoding="utf-8", errors="replace")
    for top, pattern in ((surface.skill_dir, "**/*"), ("sub_agents", "**/AGENT.md"), ("scripts", "*.py")):
        root = ws / top
        if not root.is_dir():
            continue
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                tree[path.relative_to(ws).as_posix()] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable asset: not part of the text surface
    return tree


def write_tree(workspace: Path | str, tree: Mapping[str, str]) -> list[str]:
    """Write a harness tree into a workspace; returns the paths written.

    Paths must be relative and stay inside the workspace.
    """
    ws = Path(workspace).resolve()
    written: list[str] = []
    for rel, content in sorted(tree.items()):
        target = (ws / rel).resolve()
        if Path(rel).is_absolute() or (target != ws and ws not in target.parents):
            raise ValueError(f"harness path escapes the workspace: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)
    return written


__all__ = [
    "PPCResult",
    "Proposal",
    "ProposerCache",
    "Rollout",
    "StageRecord",
    "apply_edits",
    "build_reflection_prompt",
    "propose_probe_compose",
    "read_tree",
    "record_experience",
    "score_rollout_confidence",
    "write_tree",
]
