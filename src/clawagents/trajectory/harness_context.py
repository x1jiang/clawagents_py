"""Structured proposer evidence workspace (Harness² ``core/context.py``).

Instead of pasting a heavily truncated 6K-char string excerpt into a single prompt,
``materialize_evidence_workspace`` stages an inspectable directory tree that an
agentic proposer can browse selectively with ``read_file`` / ``grep``:

* ``context/task.md`` — model-visible task text
* ``context/source/`` — optional task input materials
* ``context/decisions/`` — prior edit notes and composition verdicts
* ``context/rollouts/<label>.md`` — full rollout trajectories
* ``context/deliverables/<label>/INDEX.md`` — produced artifacts (either full
  contents or ``deliverables_mode="names"`` so proposers cannot overfit to or
  copy raw task outputs into portable skills)
* ``context/harness/`` — the editable parent harness files
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

DeliverablesMode = Literal["contents", "names"]


@dataclass(frozen=True)
class RolloutEvidence:
    """Evidence bundle captured from one probed rollout."""

    label: str
    ok: bool
    delivered: bool
    trajectory: str = ""
    deliverable: str = ""
    deliverable_files: Mapping[str, str] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class EvidenceWorkspace:
    """Metadata and prompt layout summary for a materialized proposer workspace."""

    root: Path
    rollouts: tuple[str, ...]
    deliverables: tuple[str, ...]
    harness_files: int
    has_library: bool = False
    deliverables_mode: DeliverablesMode = "contents"
    source_files: int = 0
    decisions: int = 0
    failed: tuple[str, ...] = ()

    @property
    def context_dir(self) -> Path:
        return self.root / "context"

    @property
    def layout(self) -> str:
        """Render the mission evidence-layout block describing only present paths."""
        lines = ["context/task.md                  the task as the agent received it"]
        if self.source_files:
            lines.append(
                f"context/source/                 task source materials ({self.source_files} files)"
            )
        if self.decisions:
            lines.append(
                "context/decisions/              earlier edits and composition decisions"
            )
        if self.rollouts:
            shown = ", ".join(f"{r}.md" for r in self.rollouts[:6])
            more = f" (+{len(self.rollouts) - 6} more)" if len(self.rollouts) > 6 else ""
            lines.append(f"context/rollouts/               FULL trajectories: {shown}{more}")
        if self.deliverables:
            if self.deliverables_mode == "names":
                lines.append(
                    "context/deliverables/<label>/    INDEX.md lists WHICH files each "
                    "rollout produced -- names only."
                )
                lines.append(
                    "                                 File contents are deliberately "
                    "not provided; work from the trajectories and the file set itself."
                )
            else:
                lines.append(
                    "context/deliverables/<label>/    the files each rollout produced, "
                    "one per artifact (browse INDEX.md for the set)"
                )
        if self.harness_files:
            lines.append(
                f"context/harness/                the harness you are editing "
                f"({self.harness_files} files)"
            )
        if self.has_library:
            lines.append(
                "context/library/INDEX.md        prior cross-task harness experiences"
            )
        return "\n".join(lines)


def _safe_write_rel(base: Path, rel: str, text: str) -> None:
    resolved_base = base.resolve()
    target = (resolved_base / rel).resolve()
    if Path(rel).is_absolute() or (target != resolved_base and resolved_base not in target.parents):
        raise ValueError(f"evidence path escapes context root: {rel!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def materialize_evidence_workspace(
    workspace_dir: Path | str,
    *,
    task: str,
    parent_harness: Mapping[str, str],
    rollouts: Sequence[RolloutEvidence],
    deliverables_mode: DeliverablesMode = "contents",
    source_dir: Path | str | None = None,
    decisions: Mapping[str, str] | None = None,
    library_dir: Path | str | None = None,
) -> EvidenceWorkspace:
    """Populate ``<workspace_dir>/context/`` for an agentic proposer session."""
    if deliverables_mode not in ("contents", "names"):
        raise ValueError(
            f"deliverables_mode must be 'contents' or 'names', got {deliverables_mode!r}"
        )
    root = Path(workspace_dir)
    ctx = root / "context"
    if ctx.exists():
        shutil.rmtree(ctx)
    ctx.mkdir(parents=True, exist_ok=True)

    (ctx / "task.md").write_text(task.strip() + "\n", encoding="utf-8")

    # Optional task source materials
    source_count = 0
    if source_dir is not None:
        src_root = Path(source_dir)
        if src_root.is_dir():
            dst_src = ctx / "source"
            for p in sorted(src_root.rglob("*")):
                if not p.is_file() or "__pycache__" in p.parts or p.name == ".DS_Store":
                    continue
                rel = p.relative_to(src_root)
                target = dst_src / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                source_count += 1

    # Optional prior decision notes
    decision_count = 0
    if decisions:
        dec_dir = ctx / "decisions"
        for name, body in sorted(decisions.items()):
            safe_name = Path(name).name
            if not safe_name.endswith(".md"):
                safe_name = f"{safe_name}.md"
            _safe_write_rel(dec_dir, safe_name, body.rstrip() + "\n")
            decision_count += 1

    # Rollout trajectories & deliverables
    rollout_labels: list[str] = []
    deliverable_labels: list[str] = []
    failed_labels: list[str] = []

    rollouts_dir = ctx / "rollouts"
    deliverables_root = ctx / "deliverables"
    for r in rollouts:
        safe_label = Path(r.label).name
        if not safe_label:
            continue
        rollout_labels.append(safe_label)
        if not r.ok:
            failed_labels.append(safe_label)
        header = [
            f"# Rollout `{safe_label}`",
            f"- ok: {r.ok}",
            f"- delivered: {r.delivered}",
        ]
        if r.error:
            header.append(f"- error: {r.error}")
        body = "\n".join(header) + "\n\n" + (r.trajectory.strip() or "(no transcript)") + "\n"
        _safe_write_rel(rollouts_dir, f"{safe_label}.md", body)

        files_map: dict[str, str] = dict(r.deliverable_files)
        if not files_map and r.deliverable.strip():
            files_map = {" DELIVERABLE.md".strip(): r.deliverable}
        if files_map:
            deliverable_labels.append(safe_label)
            label_dir = deliverables_root / safe_label
            index_lines = [
                f"# Deliverables for `{safe_label}`",
                "",
                f"Mode: `{deliverables_mode}`",
                "",
            ]
            for rel_path, content in sorted(files_map.items()):
                char_len = len(content)
                index_lines.append(f"- `{rel_path}` ({char_len:,} chars)")
                if deliverables_mode == "contents":
                    _safe_write_rel(label_dir, rel_path, content)
            _safe_write_rel(label_dir, "INDEX.md", "\n".join(index_lines) + "\n")

    # Parent harness files
    harness_dir = ctx / "harness"
    harness_count = 0
    for rel_path, content in sorted(parent_harness.items()):
        _safe_write_rel(harness_dir, rel_path, content)
        harness_count += 1

    # Optional cross-task experience library
    has_library = False
    if library_dir is not None:
        lib_src = Path(library_dir)
        if (lib_src / "INDEX.md").is_file():
            shutil.copytree(lib_src, ctx / "library", dirs_exist_ok=True)
            has_library = True

    return EvidenceWorkspace(
        root=root,
        rollouts=tuple(rollout_labels),
        deliverables=tuple(deliverable_labels),
        harness_files=harness_count,
        has_library=has_library,
        deliverables_mode=deliverables_mode,
        source_files=source_count,
        decisions=decision_count,
        failed=tuple(failed_labels),
    )


__all__ = [
    "DeliverablesMode",
    "EvidenceWorkspace",
    "RolloutEvidence",
    "materialize_evidence_workspace",
]
