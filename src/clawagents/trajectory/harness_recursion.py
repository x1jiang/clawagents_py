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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

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

    @property
    def selectable(self) -> bool:
        return self.ok and self.delivered and bool(self.deliverable.strip())


@dataclass(frozen=True)
class Proposal:
    """A harness edit: only the files that change. Omitted files are inherited."""

    edits: Mapping[str, str] = field(default_factory=dict)
    removed: tuple[str, ...] = ()
    notes: str = ""
    keep_base: bool = False

    @property
    def is_noop(self) -> bool:
        return self.keep_base or (not self.edits and not self.removed)


RolloutFn = Callable[[Tree, str], Awaitable[Rollout]]
"""``(harness_tree, label) -> Rollout``. Runs the task once under that harness."""
ProposeFn = Callable[[str, str], Awaitable[Proposal]]
"""``(prompt, label) -> Proposal``. One edit session (INV, DOM or compose)."""
SelectFn = Callable[[str], Awaitable[str]]
"""``selector_prompt -> reply`` ending in a ```` ```choice ```` block."""


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
                    "files": sorted(s.tree),
                    "rollout": None
                    if s.rollout is None
                    else {
                        "ok": s.rollout.ok,
                        "delivered": s.rollout.delivered,
                        "deliverable_chars": len(s.rollout.deliverable),
                        "error": s.rollout.error,
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
    warnings: list[str] = []
    leaked = lint_task_prompt(task)
    if leaked:
        warnings.append(
            "task text mentions measurement words "
            f"{leaked}: a model told it is being measured tends to optimise for the measurement"
        )
    gate = asyncio.Semaphore(max(1, concurrency))
    propose_fn = cache.wrap(propose) if cache is not None else propose
    library_index = ""
    if library is not None and library.context_dir() is not None:
        try:
            library_index = library.index_path.read_text(encoding="utf-8")
        except OSError:
            library_index = ""

    async def probe(stage: StageRecord) -> StageRecord:
        async with gate:
            ledger["rollouts"] += 1
            try:
                stage.rollout = await rollout(dict(stage.tree), stage.label)
            except Exception as exc:  # a crashed rollout is evidence, not a crash
                stage.rollout = Rollout(ok=False, error=f"{type(exc).__name__}: {exc}"[:REJECTION_TEXT_MAX])
        return stage

    async def edit_stage(label: str, kind: str, prompt: str, parent_label: str, parent: Tree) -> StageRecord:
        """One edit session with accept-or-repair; falls back to ``parent``."""
        ledger["stages"] += 1
        record = StageRecord(label=label, kind=kind, tree=dict(parent), parent=parent_label)
        current_prompt = prompt
        for attempt in range(1, max(1, stage_attempts) + 1):
            record.attempts = attempt
            try:
                async with gate:
                    proposal = await propose_fn(current_prompt, label)
            except Exception as exc:
                record.error = f"proposer failed: {type(exc).__name__}: {exc}"[:REJECTION_TEXT_MAX]
                continue
            record.notes = proposal.notes
            if proposal.is_noop:
                record.tree, record.kept_base, record.fell_back, record.error = dict(parent), True, False, ""
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
                record.tree, record.kept_base, record.fell_back, record.error = accepted, kept_base, False, ""
                return record
            record.error = error
            current_prompt = _rejection_feedback(prompt, error)
        record.tree, record.kept_base, record.fell_back = dict(parent), False, True
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
    if mode == "parallel":
        candidates = await asyncio.gather(*(
            edit_stage(f"{dim}{i}", dim, propose_prompt(dim, bases), "base", base)
            for i in range(1, k + 1)
            for dim in ("INV", "DOM")
        ))
        await asyncio.gather(*(probe(c) for c in candidates))
        stages.extend(candidates)
        prompt = compose_prompt(base, candidates)
        finals = list(await asyncio.gather(*(
            edit_stage(f"agg{i}", "compose", prompt, "base", base) for i in range(1, k + 1)
        )))
        await asyncio.gather(*(probe(f) for f in finals))
        stages.extend(finals)
    else:
        checkpoint, checkpoint_label, seen = base, "base", list(bases)
        for t in range(1, k + 1):
            round_candidates = list(await asyncio.gather(*(
                edit_stage(f"{dim}{t}", dim, propose_prompt(dim, seen), checkpoint_label, checkpoint)
                for dim in ("INV", "DOM")
            )))
            await asyncio.gather(*(probe(c) for c in round_candidates))
            stages.extend(round_candidates)
            composed = await edit_stage(
                f"c{t}", "compose", compose_prompt(checkpoint, round_candidates),
                checkpoint_label, checkpoint,
            )
            await probe(composed)
            stages.append(composed)
            finals.append(composed)
            checkpoint, checkpoint_label, seen = dict(composed.tree), composed.label, [composed]

    selection = await _select_final(task, task_id, k, mode, finals, bases, select)
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
    "write_tree",
]
