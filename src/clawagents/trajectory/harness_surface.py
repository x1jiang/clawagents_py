"""Harness²-inspired recursive harness surface, validation, blind selection, and experience library.

Distills the core mechanisms from Google Research's Harness² (arXiv:2610.14224):
  1. Editable 8-component harness surface (`SurfaceSpec`) & component cost hierarchy:
     skill (on-demand) > script (deterministic) > subagent (large-in/small-out) >
     hook (in-loop enforcement) > always-on text (hard-capped, cut-to-pay).
  2. Pre-rollout static validation (`validate_harness_tree`, `autotrigger_skills`,
     `check_script_references`, `check_python_compiles`, `assert_no_grader_verdicts`).
  3. Orthogonal `INV` (Invariant machinery) vs. `DOM` (Domain practitioner) proposal
     and `AGG`/`COMPOSE` synthesis prompts (`Propose -> Probe -> Compose`).
  4. Blind anonymized deliverable selection (`assign_anon_ids`, `parse_choice_reply`,
     `COVERAGE -> CORRECTNESS -> FIDELITY`).
  5. Causal edit attribution (`helped | hurt | no observed effect | confounded`) and
     per-domain `ExperienceLibrary` (`INDEX.md` + `EXPERIENCE.md`).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

Tree = dict[str, str]

VALID_VERDICTS = frozenset({"helped", "hurt", "no observed effect", "confounded"})
VALID_DIMENSIONS = frozenset({"INV", "DOM", "AGG", "COMPOSE"})

RESERVED_SCRIPTS: frozenset[str] = frozenset({"read_doc.py", "pack.py", "unpack.py"})

GRADER_VERDICT_MARKERS: tuple[str, ...] = (
    "deterministic_eval_summary",
    "deterministic_eval_result",
    "judge_scope_and_deterministic_status",
    "Judge findings on the rendered result",
    '"passed_count"',
    '"pass_rate"',
    '"test_pass_rate"',
)

AUTOTRIGGER_DESC_MAX = 600
# How much of a validation error is fed back to the proposer / kept on record.
REJECTION_TEXT_MAX = 400
_NAME_RE = r"[a-z0-9][a-z0-9_\-]*"
_FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)
_CHOICE_RE = re.compile(r"```choice\s*\n?\s*([A-Za-z0-9_-]+)\s*\n?\s*```", re.DOTALL)
_WHY_RE = re.compile(r"<why>(.*?)</why>", re.DOTALL)
_EXP_RE = re.compile(r"<experience>(.*?)</experience>", re.DOTALL)
_EDITS_RE = re.compile(r"<edits>(.*?)</edits>", re.DOTALL)


@dataclass(frozen=True)
class SurfaceSpec:
    """Editable surface rules and caps for a ClawAgents workspace harness."""

    name: str = "clawagents"
    always_on: tuple[str, ...] = ("AGENTS.md", ".clawagents/lessons.md", ".clawagents/skill-triggers.md")
    always_on_max: int = 16_000
    always_on_cap_reason: str = (
        "always-on prose competes with the model's output budget on every turn; "
        "move situation-specific guidance into an on-demand skill or trim existing rules"
    )
    skill_dir: str = "skills"
    skill_name_rule: str = "must_match_if_present"
    autotrigger_file: str = ".clawagents/skill-triggers.md"
    trigger_line: str = "- When {desc}, load the `{name}` skill."
    subagent_requires_mode: bool = False
    hooks_file: str | None = ".clawagents/hooks.json"
    hook_events: frozenset[str] = frozenset({"PreToolUse", "PostToolUse", "SessionStart", "Stop"})
    compile_skill_scripts: bool = True
    reject_empty_always_on: bool = False
    # When True, any path that is not part of the editable surface (always-on
    # files, skills, sub_agents, scripts, the hooks file) fails validation.
    # Harness² rejects those and reports them back to the proposer; off by
    # default here because existing callers pass whole-workspace trees.
    reject_off_surface: bool = False


DEFAULT_SURFACE = SurfaceSpec()


def extract_frontmatter(markdown: str) -> str:
    """Return YAML frontmatter block content if present at the top of `markdown`, else ''."""
    m = _FRONTMATTER_RE.match(markdown or "")
    return m.group(1) if m else ""


def parse_frontmatter_fields(markdown: str) -> dict[str, str]:
    """Parse simple single-line and folded scalar `key: value` pairs from YAML frontmatter."""
    fm = extract_frontmatter(markdown)
    if not fm:
        return {}
    fields: dict[str, str] = {}
    lines = fm.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if val in (">", "|", ">-", "|-", ""):
            block: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t") or not lines[i].strip()):
                if lines[i].strip():
                    block.append(lines[i].strip())
                i += 1
            fields[key] = " ".join(block).strip()
            continue
        fields[key] = val.strip("\"'")
        i += 1
    return fields


def check_python_compiles(rel_path: str, source: str) -> list[str]:
    """Verify Python source compiles cleanly; return actionable error strings."""
    try:
        compile(source, rel_path, "exec")
        return []
    except (SyntaxError, ValueError, TypeError) as exc:
        if isinstance(exc, SyntaxError):
            detail = f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})"
        else:
            detail = f"{type(exc).__name__}: {exc}"
        return [f"{rel_path} does not compile: {detail[:160]}"]


def find_script_references(text: str, skill_dir: str = "skills") -> list[str]:
    """Extract `scripts/<file>.py` paths invoked via `python` / `python3` in markdown text."""
    sd = re.escape(skill_dir)
    rx = re.compile(rf"(?:python3?\s+)(?:-[A-Za-z]+\s+)*[\"'`]?((?:{sd}/[\w\-]+/)?scripts/[\w.\-]+\.py)\b")
    return list(dict.fromkeys(rx.findall(text or "")))


def check_script_references(
    doc_path: str,
    doc_text: str,
    available_paths: set[str] | frozenset[str],
    *,
    skill_name: str | None = None,
    skill_dir: str = "skills",
    reserved_scripts: frozenset[str] | set[str] = RESERVED_SCRIPTS,
) -> list[str]:
    """Verify every `python scripts/<file>.py` reference in `doc_text` exists in `available_paths`."""
    errs: list[str] = []
    for ref in find_script_references(doc_text, skill_dir=skill_dir):
        basename = ref.rsplit("/", 1)[-1]
        if basename in reserved_scripts:
            continue
        candidates = {ref}
        if skill_name:
            # Inside a skill proposal, `support_files` paths are relative to the skill root
            # (e.g. `scripts/helper.py`), while the body may invoke either `scripts/helper.py`
            # or `skills/<skill_name>/scripts/helper.py`.
            prefix = f"{skill_dir}/{skill_name}/"
            if ref.startswith(prefix):
                candidates.add(ref[len(prefix) :])
            else:
                candidates.add(f"{prefix}{ref}")
        if not (candidates & set(available_paths)):
            errs.append(f"{doc_path} invokes `{ref}` but that file is not in the skill/harness")
    return errs


def check_hooks(rel_path: str, source: str, events: frozenset[str] | set[str]) -> list[str]:
    """Validate a hooks file's shape at proposal time.

    A malformed hook does not fail loudly at run time — it is simply never
    fired, so an edit that "adds enforcement" silently adds nothing. Expected
    shape: ``{"hooks": {"<Event>": [{"type": "command", "command": "..."}]}}``
    where each event is one of ``events``. An entry may also be a matcher
    group ``{"matcher": "...", "hooks": [ ... ]}``.
    """
    errs: list[str] = []
    try:
        data = json.loads(source)
    except (json.JSONDecodeError, TypeError) as exc:
        return [f"{rel_path} is not valid JSON: {exc}"]
    hooks = data.get("hooks") if isinstance(data, dict) else None
    if not isinstance(hooks, dict):
        return [f'{rel_path} needs a top-level "hooks" object mapping event names to hook lists']
    for event, entries in hooks.items():
        if event not in events:
            errs.append(f"{rel_path}: unknown hook event {event!r} (allowed: {sorted(events)})")
            continue
        if not isinstance(entries, list):
            errs.append(f"{rel_path}: hooks.{event} must be a list")
            continue
        for i, entry in enumerate(entries):
            where = f"{rel_path}: hooks.{event}[{i}]"
            if not isinstance(entry, dict):
                errs.append(f"{where} must be an object")
                continue
            # Matcher group: validate the nested command hooks.
            nested = entry.get("hooks")
            commands = nested if isinstance(nested, list) else [entry]
            if isinstance(nested, list) and not nested:
                errs.append(f"{where} has an empty hooks list")
            for j, hook in enumerate(commands):
                sub_where = f"{where}.hooks[{j}]" if isinstance(nested, list) else where
                if not isinstance(hook, dict):
                    errs.append(f"{sub_where} must be an object")
                elif hook.get("type") != "command":
                    errs.append(f'{sub_where} needs "type": "command"')
                elif not str(hook.get("command") or "").strip():
                    errs.append(f"{sub_where} needs a non-empty command")
    return errs


def autotrigger_skills(
    surface: SurfaceSpec,
    tree: Tree,
    *,
    always_on_max: int | None = None,
) -> list[str]:
    """Ensure every skill in `tree` is named in always-on text; return skill names auto-wired."""
    cap = surface.always_on_max if always_on_max is None else always_on_max
    sd = re.escape(surface.skill_dir)
    skill_rx = re.compile(rf"{sd}/({_NAME_RE})/SKILL\.md")
    skills = sorted({m.group(1) for rel in tree if (m := skill_rx.fullmatch(rel))})
    body = "\n\n".join(tree.get(a, "") for a in surface.always_on)
    fixed: list[str] = []

    for name in skills:
        if name in body:
            continue
        skill_md = tree.get(f"{surface.skill_dir}/{name}/SKILL.md", "")
        fm = parse_frontmatter_fields(skill_md)
        desc = (fm.get("when-to-use") or fm.get("description") or f"the {name} workflow applies").strip()
        # A trigger line is always-on text; a long description would spend
        # the cap on one skill.
        desc = desc[:AUTOTRIGGER_DESC_MAX].rstrip().rstrip(".")
        line = surface.trigger_line.format(name=name, desc=desc)
        cur = tree.get(surface.autotrigger_file, "")
        if len(body) + len(line) + 2 > cap:
            continue
        tree[surface.autotrigger_file] = (cur.rstrip() + "\n" + line) if cur.strip() else line
        body = "\n\n".join(tree.get(a, "") for a in surface.always_on)
        fixed.append(name)
    return fixed


def ensure_workspace_skill_trigger(
    workspace: Path | str,
    skill_name: str,
    description: str,
    *,
    surface: SurfaceSpec = DEFAULT_SURFACE,
) -> bool:
    """Register a concise trigger line in `.clawagents/skill-triggers.md` when a skill is applied."""
    ws = Path(workspace)
    trigger_path = ws / surface.autotrigger_file
    existing = trigger_path.read_text(encoding="utf-8") if trigger_path.is_file() else ""
    if f"`{skill_name}`" in existing or re.search(rf"\b{re.escape(skill_name)}\b", existing):
        return False
    desc = (description or f"the {skill_name} workflow applies").strip().rstrip(".")
    line = surface.trigger_line.format(name=skill_name, desc=desc)
    if len(existing) + len(line) + 2 > surface.always_on_max:
        return False
    trigger_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Skill Triggers\n\n" if not existing.strip() else ""
    updated = (existing.rstrip() + "\n" + line + "\n") if existing.strip() else (header + line + "\n")
    trigger_path.write_text(updated, encoding="utf-8")
    return True


def assert_no_grader_verdicts(
    tree_or_text: Mapping[str, str] | str,
    markers: Sequence[str] = GRADER_VERDICT_MARKERS,
) -> None:
    """Tripwire preventing benchmark/grader verdict strings from entering lessons or skills."""
    items: list[tuple[str, str]]
    if isinstance(tree_or_text, str):
        items = [("<text>", tree_or_text)]
    else:
        items = [(str(k), str(v)) for k, v in sorted(tree_or_text.items())]
    for rel, body in items:
        for m in markers:
            if m in rel:
                raise ValueError(
                    f"grader verdict marker {m!r} found in path {rel!r}; "
                    "evaluation outputs must never enter harness memory or skills"
                )
            if m in body:
                where = body.index(m)
                snippet = body[max(0, where - 40) : where + 80]
                raise ValueError(
                    f"grader verdict marker {m!r} found in {rel!r} at offset {where} ({snippet!r}); "
                    "evaluation outputs must never enter harness memory or skills"
                )


def validate_harness_tree(
    tree: Tree,
    surface: SurfaceSpec = DEFAULT_SURFACE,
    *,
    allow_subagents: bool = True,
    always_on_max: int | None = None,
    reserved_scripts: frozenset[str] | set[str] = RESERVED_SCRIPTS,
) -> dict[str, Any]:
    """Validate an entire harness tree (always-on cap, skill frontmatter/triggers, scripts, hooks)."""
    errs: list[str] = []
    cap = surface.always_on_max if always_on_max is None else always_on_max
    body = "\n\n".join(tree.get(a, "") for a in surface.always_on)

    if len(body) > cap:
        errs.append(
            f"always-on content is {len(body)} chars, over the {cap} cap; "
            f"{surface.always_on_cap_reason}"
        )
    if surface.reject_empty_always_on and not body.strip():
        errs.append(f"{surface.always_on[0]} is empty")

    sd = re.escape(surface.skill_dir)
    skill_rx = re.compile(rf"{sd}/({_NAME_RE})/SKILL\.md")
    skill_script_rx = re.compile(rf"{sd}/{_NAME_RE}/scripts/[A-Za-z0-9_.\-]+")
    script_rx = re.compile(r"scripts/[a-z0-9_][a-z0-9_]*\.py")
    subagent_rx = re.compile(rf"sub_agents/({_NAME_RE})/AGENT\.md")

    skills = sorted({m.group(1) for rel in tree if (m := skill_rx.fullmatch(rel))})
    for name in skills:
        rel = f"{surface.skill_dir}/{name}/SKILL.md"
        fm = parse_frontmatter_fields(tree[rel])
        if not fm.get("description"):
            errs.append(f"{rel} needs a `description:` in its frontmatter")
        fm_name = fm.get("name")
        if surface.skill_name_rule == "must_match" and fm_name != name:
            errs.append(f"{rel} needs frontmatter `name: {name}` matching its directory")
        elif fm_name and fm_name != name:
            errs.append(
                f"{rel} frontmatter `name: {fm_name}` overrides directory name `{name}` at load; "
                f"use `name: {name}`"
            )
        if name not in body:
            errs.append(
                f"skill '{name}' is never named in always-on text; add a trigger line in "
                f"`{surface.autotrigger_file}`"
            )

    subs = sorted({m.group(1) for rel in tree if (m := subagent_rx.fullmatch(rel))})
    if subs and not allow_subagents:
        errs.append(f"sub_agents are not enabled: {subs}")
    for name in subs:
        rel = f"sub_agents/{name}/AGENT.md"
        fm = parse_frontmatter_fields(tree[rel])
        if surface.subagent_requires_mode and fm.get("mode") != "subagent":
            errs.append(f"{rel} needs frontmatter `mode: subagent`")
        if not fm.get("description"):
            errs.append(f"{rel} needs a `description:` in its frontmatter")

    have_scripts = {
        rel for rel in tree if script_rx.fullmatch(rel) or skill_script_rx.fullmatch(rel)
    }
    for rel, src in tree.items():
        if script_rx.fullmatch(rel) or (
            surface.compile_skill_scripts and skill_script_rx.fullmatch(rel) and rel.endswith(".py")
        ):
            errs.extend(check_python_compiles(rel, src))
        m_skill = skill_rx.fullmatch(rel)
        if m_skill or rel in surface.always_on:
            errs.extend(
                check_script_references(
                    rel,
                    src,
                    have_scripts,
                    skill_name=m_skill.group(1) if m_skill else None,
                    skill_dir=surface.skill_dir,
                    reserved_scripts=reserved_scripts,
                )
            )

    if surface.hooks_file and surface.hooks_file in tree:
        errs.extend(check_hooks(surface.hooks_file, tree[surface.hooks_file], surface.hook_events))

    if surface.reject_off_surface:
        allowed_exact = set(surface.always_on) | ({surface.hooks_file} if surface.hooks_file else set())
        for rel in sorted(tree):
            if rel in allowed_exact or skill_rx.fullmatch(rel) or skill_script_rx.fullmatch(rel):
                continue
            if script_rx.fullmatch(rel) or subagent_rx.fullmatch(rel):
                continue
            errs.append(f"{rel} is not part of the editable harness surface")

    try:
        assert_no_grader_verdicts(tree)
    except ValueError as exc:
        errs.append(str(exc))

    if errs:
        raise ValueError(" | ".join(errs))

    return {
        "always_on_chars": len(body),
        "n_files": len(tree),
        "skills": len(skills),
        "sub_agents": len(subs),
        "scripts": sum(1 for rel in tree if script_rx.fullmatch(rel)),
    }


def diff_summary(surface: SurfaceSpec, before: Tree, after: Tree) -> dict[str, Any]:
    """Summarize added/removed/modified files and always-on char delta between two harness trees."""
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(rel for rel in set(before) & set(after) if before[rel] != after[rel])
    before_on = "\n\n".join(before.get(a, "") for a in surface.always_on)
    after_on = "\n\n".join(after.get(a, "") for a in surface.always_on)
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "always_on_char_delta": len(after_on) - len(before_on),
    }


def tree_diversity(trees: Mapping[str, Tree]) -> dict[str, Any]:
    """Pairwise line-Jaccard similarity over candidate trees so mode collapse is visible."""
    names = sorted(trees)
    if len(names) < 2:
        return {"n": len(names), "n_distinct": len(names), "mean_line_jaccard": 1.0}

    def _lines(t: Tree) -> set[str]:
        return {f"{rel}:{ln}" for rel, c in t.items() for ln in c.splitlines() if ln.strip()}

    sets = {n: _lines(trees[n]) for n in names}
    sims: list[float] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            u = sets[a] | sets[b]
            sims.append(len(sets[a] & sets[b]) / len(u) if u else 1.0)
    distinct = len({tuple(sorted(t.items())) for t in trees.values()})
    return {
        "n": len(names),
        "n_distinct": distinct,
        "mean_line_jaccard": round(sum(sims) / len(sims), 4),
    }


def accept_or_repair_tree(
    candidate: Tree,
    fallback: Tree,
    surface: SurfaceSpec = DEFAULT_SURFACE,
    *,
    allow_subagents: bool = True,
    reserved_scripts: frozenset[str] | set[str] = RESERVED_SCRIPTS,
    repair: "Callable[[str, Tree], Tree | None] | None" = None,
    attempts: int = 2,
) -> tuple[Tree, bool, bool, str, list[str]]:
    """Auto-repair missing skill triggers, validate `candidate`, and fall back on error.

    Deterministic fixes come first (autotrigger), then validation. When
    validation fails and ``repair`` is given, it is called with the rejection
    text and the rejected tree and may return a corrected tree; that is tried
    again, up to ``attempts`` validations in total (Harness² uses 2). Only when
    every attempt fails does the stage fall back to ``fallback``.

    Returns `(accepted_tree, kept_base, fell_back, error_msg, autotriggered_skills)`.
    """
    working = dict(candidate)
    triggered: list[str] = []
    error = ""
    for attempt in range(max(1, attempts)):
        triggered = autotrigger_skills(surface, working)
        try:
            validate_harness_tree(
                working,
                surface,
                allow_subagents=allow_subagents,
                reserved_scripts=reserved_scripts,
            )
        except ValueError as exc:
            error = str(exc)[:REJECTION_TEXT_MAX]
            if repair is None or attempt + 1 >= max(1, attempts):
                break
            repaired = repair(error, dict(working))
            if not repaired:
                break
            working = dict(repaired)
            continue
        changed = diff_summary(surface, fallback, working)
        kept_base = not (changed["added"] or changed["removed"] or changed["modified"])
        return working, kept_base, False, "", triggered
    return dict(fallback), False, True, error, triggered


def build_propose_prompt(
    task: str,
    dimension: str,
    base_trajectories: Sequence[str],
    *,
    deliverable_word: str = "deliverable",
    domain_word: str = "domain",
    library_index: str = "",
) -> str:
    """Build an orthogonal INV or DOM harness proposal prompt (Harness² Propose stage)."""
    dim = dimension.upper().strip()
    if dim not in ("INV", "DOM"):
        raise ValueError(f"dimension must be 'INV' or 'DOM', got {dimension!r}")
    role_block = (
        f"# YOUR DIMENSION: INV — How the Agent WORKS\n\n"
        "You own procedure, tool error recovery, deterministic helper scripts (`scripts/<name>.py`), "
        f"bounded subagents, and completion checks — not subject matter. Reopen or mechanically check "
        f"the {deliverable_word} before finishing; a successful write call does not establish usability.\n"
        "Do not add domain rules or matter facts; a parallel DOM candidate owns them."
        if dim == "INV"
        else f"# YOUR DIMENSION: DOM — What a Practitioner in This {domain_word.title()} Knows\n\n"
        f"You own the practitioner model for the {deliverable_word}: required sections/columns, "
        "cross-file reconciliation rules, and situation-triggered skills (`skills/<name>/SKILL.md`).\n"
        "Always-on text is expensive; put situation-specific domain rules in an on-demand skill "
        "and name its trigger situation narrowly. If no change is supported by the trajectories, "
        "respond with `KEEP_BASE`."
    )
    traj_text = "\n\n---\n\n".join(
        f"### Base Rollout {i + 1}\n{t}" for i, t in enumerate(base_trajectories)
    )
    lib_section = f"\n\n# DOMAIN LIBRARY INDEX\n\n{library_index.strip()}" if library_index.strip() else ""
    return (
        f"You are the {dim} harness editor for a task in this {domain_word}.\n\n"
        f"# THE TASK\n\n{task.strip()}\n\n"
        f"{role_block}\n\n"
        f"# BASE TRAJECTORIES\n\n{traj_text}"
        f"{lib_section}\n\n"
        "# OUTPUT\n\n"
        "Cite the exact trajectory moment motivating each edit in `<notes>...</notes>`, or write `KEEP_BASE`.\n"
    )


def build_compose_prompt(
    task: str,
    candidate_summaries: Mapping[str, str],
    *,
    deliverable_word: str = "deliverable",
) -> str:
    """Build the Harness² AGG / COMPOSE synthesizer prompt."""
    cand_lines = "\n\n".join(
        f"### Candidate `{label}`\n{summary}" for label, summary in sorted(candidate_summaries.items())
    )
    return (
        "You are the integration judge and harness synthesizer.\n\n"
        f"# THE TASK\n\n{task.strip()}\n\n"
        "# CANDIDATE PROBES\n\n"
        f"{cand_lines}\n\n"
        "# YOUR JOB\n\n"
        f"1. Judge each candidate edit from its probe rollout and produced {deliverable_word}: "
        "did it help, hurt, or have no observed effect versus base?\n"
        "2. Synthesize ONE final harness containing only what the evidence supports: merge the "
        "verified parts of INV and DOM, resolve overlaps rather than concatenating, keep always-on "
        "text tiny, and keep every retained skill's trigger line narrow. If no candidate demonstrably "
        "improved on base, say `KEEP_BASE`.\n\n"
        "# NOTES FORMAT\n\n"
        "In `<notes>...</notes>`, write one line per decision:\n"
        "`kept|revised|dropped -> <which edit> -> <trajectory/deliverable evidence>`\n"
    )


# ─── Causal Edit Attribution & Experience Distillation ───────────────────────


@dataclass(frozen=True)
class EditAttribution:
    """One row from `<edits>` recording what a harness/skill edit actually did."""

    component: str
    candidates: str
    change: str
    observed: str
    verdict: str  # helped | hurt | no observed effect | confounded

    @property
    def is_actionable_evidence(self) -> bool:
        """Only non-confounded rows with an observed effect (`helped` or `hurt`) support a lesson."""
        return self.verdict.strip().lower() in ("helped", "hurt")


@dataclass(frozen=True)
class ExperienceEntry:
    """A validated transferable lesson or anchored workspace fact."""

    cue: str
    lesson: str
    evidence: str
    confidence: str = "medium"
    dimension: str = "INV"  # INV (machinery/procedure) or DOM (practitioner/domain)
    kind: str = "procedure"  # procedure | workspace_fact
    edits: tuple[EditAttribution, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue": self.cue,
            "lesson": self.lesson,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "dimension": self.dimension,
            "kind": self.kind,
            "edits": [e.__dict__ for e in self.edits],
        }


def parse_edits_block(text: str) -> list[EditAttribution]:
    """Parse `<edits>...</edits>` blocks into `EditAttribution` records."""
    m = _EDITS_RE.search(text or "")
    raw = m.group(1).strip() if m else ""
    if not raw:
        return []
    out: list[EditAttribution] = []
    current: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        key = k.strip().lower()
        val = v.strip()
        if key == "component" and current.get("component"):
            out.append(_build_edit_attr(current))
            current = {}
        if key in ("component", "candidates", "change", "observed", "verdict"):
            current[key] = val
    if current.get("component"):
        out.append(_build_edit_attr(current))
    return out


def _build_edit_attr(d: dict[str, str]) -> EditAttribution:
    verdict = d.get("verdict", "confounded").strip().lower()
    if verdict not in VALID_VERDICTS:
        verdict = "confounded"
    return EditAttribution(
        component=d.get("component", "").strip(),
        candidates=d.get("candidates", "").strip(),
        change=d.get("change", "").strip(),
        observed=d.get("observed", "").strip(),
        verdict=verdict,
    )


def parse_experience_reply(text: str) -> ExperienceEntry | None:
    """Parse `<edits>` and `<experience>` from an LLM reflection reply.

    Enforces Harness²'s causal evidence gate:
      - `NO_ENTRY` or missing `cue`/`lesson`/`evidence` returns `None`.
      - If `<edits>` rows are present and every row is `confounded` or `no observed effect`,
        returns `None` (adoption alone never establishes that an edit helped).
    """
    if not text or "NO_ENTRY" in text:
        return None
    edits = parse_edits_block(text)
    if edits and not any(e.is_actionable_evidence for e in edits):
        return None
    m = _EXP_RE.search(text)
    if not m:
        return None
    fields: dict[str, str] = {}
    for line in m.group(1).strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            key = k.strip().lower()
            if key in ("cue", "lesson", "evidence", "confidence", "dimension", "kind"):
                fields[key] = v.strip()
    if not all(fields.get(req) for req in ("cue", "lesson", "evidence")):
        return None
    dim = fields.get("dimension", "INV").upper()
    if dim not in ("INV", "DOM"):
        dim = "INV"
    kind = fields.get("kind", "procedure").lower()
    if kind not in ("procedure", "workspace_fact"):
        kind = "workspace_fact" if "in this workspace" in fields["lesson"].lower() else "procedure"
    entry = ExperienceEntry(
        cue=fields["cue"],
        lesson=fields["lesson"],
        evidence=fields["evidence"],
        confidence=fields.get("confidence", "medium"),
        dimension=dim,
        kind=kind,
        edits=tuple(edits),
    )
    assert_no_grader_verdicts({"EXPERIENCE.md": json.dumps(entry.to_dict())})
    return entry


# ─── Per-Domain Experience Library ───────────────────────────────────────────


def _slug(value: str) -> str:
    cleaned = re.sub(r"\s*[/\\]\s*", "__", (value or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", cleaned).strip("_")
    return cleaned or "general"


@dataclass
class ExperienceLibrary:
    """Per-domain experience library (`INDEX.md` + `harnesses/<task>/EXPERIENCE.md`)."""

    root: Path
    domain: str = "general"

    @property
    def dir(self) -> Path:
        return Path(self.root) / _slug(self.domain)

    @property
    def index_path(self) -> Path:
        return self.dir / "INDEX.md"

    def entry_dir(self, task_id: str) -> Path:
        return self.dir / "harnesses" / _slug(task_id)

    def entries(self) -> list[str]:
        d = self.dir / "harnesses"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def add(
        self,
        task_id: str,
        entry: ExperienceEntry,
        *,
        why: str = "",
        label: str = "",
        run_id: str = "",
        harness: Tree | None = None,
    ) -> Path:
        """Persist one validated `ExperienceEntry` (and optional harness snapshot) and rebuild `INDEX.md`."""
        assert_no_grader_verdicts({"EXPERIENCE.md": json.dumps(entry.to_dict()) + why})
        if harness:
            assert_no_grader_verdicts(harness)
        d = self.entry_dir(task_id)
        d.mkdir(parents=True, exist_ok=True)
        if harness:
            hdir = d / "harness"
            shutil.rmtree(hdir, ignore_errors=True)
            for rel, content in harness.items():
                dest = hdir / rel.lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

        lines = [
            f"# {task_id}",
            "",
            f"- adopted: `{label or 'run'}`" + (f" (run `{run_id}`)" if run_id else ""),
            f"- dimension: `{entry.dimension}`",
            f"- kind: `{entry.kind}`",
            "",
        ]
        if why.strip():
            lines += ["## Why this was chosen", why.strip(), ""]
        if entry.edits:
            lines += ["## What each edit did"]
            for e in entry.edits:
                lines.append(
                    f"- **{e.component}** (`{e.verdict}` in {e.candidates or 'run'}): "
                    f"{e.change} -> {e.observed}"
                )
            lines.append("")
        lines += [
            "## Transferable lesson",
            f"- **cue**: {entry.cue}",
            f"- **lesson**: {entry.lesson}",
            f"- **evidence**: {entry.evidence}",
            f"- **confidence**: {entry.confidence}",
            f"- **dimension**: {entry.dimension}",
            f"- **kind**: {entry.kind}",
            "",
        ]
        (d / "EXPERIENCE.md").write_text("\n".join(lines), encoding="utf-8")
        self.rebuild_index()
        return d

    def retain_before(self, task_id: str, stream: Sequence[str]) -> list[str]:
        """Keep only entries for tasks that precede ``task_id`` in ``stream``.

        A task must learn only from earlier tasks: an entry written by a later
        task (from a previous pass over the same stream) or by the task itself
        would leak its own outcome into the run. Returns the removed task ids.
        Entries for tasks not in ``stream`` are left alone.
        """
        order = [_slug(t) for t in stream]
        current = _slug(task_id)
        if current not in order:
            return []
        later = set(order[order.index(current):])
        doomed = [name for name in self.entries() if name in later]
        if doomed:
            self.forget(doomed)
        return doomed

    def context_dir(self) -> Path | None:
        """The library directory to show a proposer, or ``None`` while it is empty."""
        return self.dir if self.entries() else None

    def forget(self, task_ids: Sequence[str]) -> None:
        removed = False
        for tid in task_ids:
            d = self.entry_dir(tid)
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed = True
        if removed:
            self.rebuild_index()

    def rebuild_index(self) -> Path:
        rows = [
            f"# Experience Library — {self.domain}",
            "",
            "Each entry records a validated lesson from an earlier task in this domain.",
            "",
        ]
        names = self.entries()
        if not names:
            rows.append("_(empty — no validated entries yet)_")
        for name in names:
            exp_file = self.dir / "harnesses" / name / "EXPERIENCE.md"
            cue = _extract_md_bullet(exp_file, "cue")
            lesson = _extract_md_bullet(exp_file, "lesson")
            dim = _extract_md_bullet(exp_file, "dimension")
            rows.append(f"## `harnesses/{name}/`")
            rows.append(f"- experience: `harnesses/{name}/EXPERIENCE.md`")
            inventory = _harness_inventory(self.dir / "harnesses" / name / "harness")
            if inventory:
                rows.append(f"- harness: {inventory}")
            if dim:
                rows.append(f"- dimension: `{dim}`")
            if cue:
                rows.append(f"- cue: {cue}")
            if lesson:
                rows.append(f"- lesson: {lesson}")
            rows.append("")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text("\n".join(rows), encoding="utf-8")
        return self.index_path


def _harness_inventory(hdir: Path) -> str:
    """One-line inventory of a stored harness snapshot (skills, sub_agents, scripts, size)."""
    if not hdir.is_dir():
        return ""
    files = [p for p in hdir.rglob("*") if p.is_file()]
    rels = [p.relative_to(hdir).as_posix() for p in files]
    skills = sum(1 for r in rels if r.endswith("/SKILL.md"))
    subs = sum(1 for r in rels if r.endswith("/AGENT.md"))
    scripts = sum(1 for r in rels if r.endswith(".py"))
    chars = sum(p.stat().st_size for p in files)
    return f"{skills} skill(s), {subs} sub_agent(s), {scripts} script(s), {chars:,} chars in {len(files)} file(s)"


# Words a task-facing prompt must not contain: they tell the model it is being
# measured, which invites optimising for the measurement (Harness² rejects
# adapter prompts containing them at construction).
MEASUREMENT_WORDS: tuple[str, ...] = ("benchmark", "evaluation", "grader", "rubric", "score")


def lint_task_prompt(text: str, words: Sequence[str] = MEASUREMENT_WORDS) -> list[str]:
    """Return the measurement words found in a task-facing prompt (empty when clean)."""
    found = []
    for word in words:
        if re.search(rf"\b{re.escape(word)}s?\b", text or "", re.IGNORECASE):
            found.append(word)
    return found


def _extract_md_bullet(path: Path, field_name: str) -> str:
    if not path.is_file():
        return ""
    prefix = f"- **{field_name}**:"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


# ─── Blind Anonymized Selection (COVERAGE -> CORRECTNESS -> FIDELITY) ────────


def _anon_id(index: int) -> str:
    """Map 0-based index to A..Z, AA..AZ, BA..."""
    out = ""
    n = int(index) + 1
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def assign_anon_ids(
    labels: Sequence[str],
    *,
    task_id: str = "task",
    k: int = 1,
    mode: str = "bon",
) -> dict[str, str]:
    """Deterministic seeded shuffle mapping anonymous ID (`A`, `B`, ...) -> candidate label."""
    seed = int(hashlib.sha1(f"{task_id}|{k}|{mode}".encode("utf-8")).hexdigest()[:12], 16)
    ordered = sorted(labels)
    random.Random(seed).shuffle(ordered)
    return {_anon_id(i): lbl for i, lbl in enumerate(ordered)}


def build_blind_selector_prompt(
    task: str,
    anon_candidates: Mapping[str, str],
    *,
    deliverable_word: str = "deliverable",
) -> str:
    """Build Harness²'s blind artifact selector prompt (COVERAGE -> CORRECTNESS -> FIDELITY)."""
    ids = ", ".join(sorted(anon_candidates))
    candidate_blocks = []
    for anon_id in sorted(anon_candidates):
        body = anon_candidates[anon_id]
        candidate_blocks.append(
            f"## Candidate {anon_id} ({len(body):,} chars)\n\n{body}"
        )
    joined_candidates = "\n\n---\n\n".join(candidate_blocks)
    return (
        f"You are choosing which of {len(anon_candidates)} anonymous candidate ANSWERS to a task "
        "should be delivered to the user.\n\n"
        "# THE TASK\n\n"
        f"{task.strip()}\n\n"
        "# THE CANDIDATES ARE ANONYMOUS ON PURPOSE\n\n"
        "The IDs are arbitrary and shuffled, and tell you nothing about how an output was "
        "produced. Judge each candidate strictly on what it produced against the request — "
        "never on how elaborate or verbose it sounds.\n\n"
        "# HOW TO DECIDE, IN THIS ORDER\n\n"
        f"1. COVERAGE. Is every {deliverable_word} the instructions name present, and does each "
        "contain every section, field, value, or analysis asked for? Prefer the candidate that "
        "answers the most of those points.\n\n"
        "2. CORRECTNESS. Are the values internally consistent and grounded in the workspace "
        "evidence? Reject invented specifics; where evidence is silent, a candidate that says so "
        "explicitly is safer than one that guesses confidently.\n\n"
        "3. FIDELITY. Required formats, file names, column headers, and structure, with no "
        "placeholder, TODO, or uncomputed formula. The deliverable's shape belongs to the request.\n\n"
        "# CANDIDATE ARTIFACTS\n\n"
        f"{joined_candidates}\n\n"
        "# OUTPUT FORMAT\n\n"
        "First your reasoning, then your choice as the very last block:\n\n"
        "<why>\n"
        "At most 5 lines. Name the criterion (COVERAGE, CORRECTNESS, or FIDELITY) that decided it, "
        "and the closest runner-up with the specific defect that cost it the pick.\n"
        "</why>\n\n"
        "```choice\n"
        f"<one of: {ids}>\n"
        "```\n"
    )


def selector_json_schema(choices: Sequence[str]) -> dict[str, Any]:
    """Strict JSON Schema for blind artifact selection (grammar-constrained decoding)."""
    allowed = sorted({str(c).strip() for c in choices if str(c).strip()})
    choice_prop: dict[str, Any] = {"type": "string"}
    if allowed:
        choice_prop["enum"] = allowed
    return {
        "type": "object",
        "properties": {
            "why": {
                "type": "string",
                "description": (
                    "Concise rationale naming the deciding criterion "
                    "(COVERAGE, CORRECTNESS, or FIDELITY) and runner-up defect."
                ),
            },
            "choice": choice_prop,
        },
        "required": ["why", "choice"],
        "additionalProperties": False,
    }


def proposal_json_schema() -> dict[str, Any]:
    """Strict JSON Schema for single-shot harness edit proposals."""
    return {
        "type": "object",
        "properties": {
            "keep_base": {"type": "boolean"},
            "notes": {"type": "string"},
            "edits": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "removed": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["keep_base", "notes", "edits", "removed"],
        "additionalProperties": False,
    }


def parse_choice_reply(reply_text: str, id_map: Mapping[str, str]) -> tuple[str, str, str]:
    """Parse `(chosen_anon_id, chosen_label, why)` from a blind selector reply.

    Accepts either:
    * Schema-constrained JSON ``{"choice": "<id>", "why": "..."}`` (bare or fenced), or
    * Harness²'s ``<why>...</why>`` + ```` ```choice ```` block.
    """
    raw = (reply_text or "").strip()
    m = _CHOICE_RE.search(raw)
    if m:
        anon = m.group(1).strip()
        if anon not in id_map:
            raise ValueError(f"selector chose {anon!r}, which is not in {sorted(id_map)}")
        w = _WHY_RE.search(raw)
        why = (w.group(1).strip() if w else "")[:2000]
        return anon, id_map[anon], why

    # Fallback / primary path for JSON-schema constrained replies:
    candidate = raw
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```\s*$", "", candidate).strip()
    obj_match = re.search(r"\{[\s\S]*\}", candidate)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("choice"), str):
            anon = parsed["choice"].strip()
            if anon not in id_map:
                raise ValueError(f"selector chose {anon!r}, which is not in {sorted(id_map)}")
            why = str(parsed.get("why") or "").strip()[:2000]
            return anon, id_map[anon], why

    raise ValueError("no ```choice block or valid JSON choice in selector reply")


_NOTES_RE = re.compile(r"<notes>(.*?)</notes>", re.DOTALL)


def parse_proposal_reply(
    reply_text: str,
) -> tuple[dict[str, str], tuple[str, ...], str, bool]:
    """Parse ``(edits, removed, notes, keep_base)`` from an LLM proposal reply."""
    raw = (reply_text or "").strip()
    if not raw:
        return {}, (), "", True

    notes_match = _NOTES_RE.search(raw)
    extracted_notes = notes_match.group(1).strip() if notes_match else ""

    candidate = raw
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```\s*$", "", candidate).strip()

    # Explicit KEEP_BASE token outside JSON
    if candidate == "KEEP_BASE" or (
        "KEEP_BASE" in raw and "{" not in raw
    ):
        return {}, (), extracted_notes, True

    obj_match = re.search(r"\{[\s\S]*\}", candidate)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict):
            keep_base = bool(data.get("keep_base", False))
            raw_edits = data.get("edits")
            edits = (
                {str(k): str(v) for k, v in raw_edits.items() if str(k).strip()}
                if isinstance(raw_edits, dict)
                else {}
            )
            raw_removed = data.get("removed")
            removed = (
                tuple(str(r) for r in raw_removed if str(r).strip())
                if isinstance(raw_removed, list)
                else ()
            )
            notes = str(data.get("notes") or extracted_notes).strip()
            if keep_base or (not edits and not removed):
                return {}, (), notes, True
            return edits, removed, notes, False

    if "KEEP_BASE" in raw:
        return {}, (), extracted_notes, True
    raise ValueError("proposal reply did not contain valid JSON edits or KEEP_BASE")
