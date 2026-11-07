"""Promote recurring PTRL lessons into governed skill_workshop proposals."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from clawagents.trajectory.harness_surface import ExperienceEntry, parse_experience_reply
from clawagents.trajectory.lessons import (
    classify_lesson_dimension,
    lesson_key,
    parse_lesson_bullets,
    slugify_lesson_name,
    strip_lesson_tag,
)

logger = logging.getLogger(__name__)

_INDEX_FILE = "lesson-index.json"
_DEFAULT_MIN_OCCURRENCES = 3


def _clawagents_dir(workspace: Path) -> Path:
    return workspace / ".clawagents"


def _index_path(workspace: Path) -> Path:
    return _clawagents_dir(workspace) / _INDEX_FILE


def _load_index(workspace: Path) -> dict[str, Any]:
    path = _index_path(workspace)
    if not path.is_file():
        return {"lessons": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("lessons"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        logger.debug("lesson promotion: corrupt index", exc_info=True)
    return {"lessons": {}}


def _save_index(workspace: Path, data: dict[str, Any]) -> None:
    path = _index_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_lessons_in_index(new_lessons_md: str, workspace: Path) -> dict[str, dict[str, Any]]:
    """Increment occurrence counts for lesson bullets; return entries updated this call."""
    index = _load_index(workspace)
    lessons: dict[str, dict[str, Any]] = index.setdefault("lessons", {})
    now = int(time.time())
    updated: dict[str, dict[str, Any]] = {}

    for bullet in parse_lesson_bullets(new_lessons_md):
        dim, kind = classify_lesson_dimension(bullet)
        if kind == "workspace_fact":
            # Harness² rule: workspace-specific facts stay anchored in workspace memory,
            # never promoted into portable domain skills.
            continue
        clean_text = strip_lesson_tag(bullet)
        key = lesson_key(clean_text)
        entry = lessons.get(key)
        if entry is None:
            entry = {
                "text": clean_text,
                "dimension": dim,
                "kind": kind,
                "count": 0,
                "first_seen": now,
                "last_seen": now,
                "promoted_proposal_id": None,
            }
            lessons[key] = entry
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_seen"] = now
        entry["text"] = clean_text
        entry["dimension"] = dim
        updated[key] = entry

    exp: ExperienceEntry | None = parse_experience_reply(new_lessons_md)
    if exp is not None and exp.kind == "procedure":
        exp_text = f"When {exp.cue}: {exp.lesson}"
        key = lesson_key(exp_text)
        entry = lessons.get(key)
        if entry is None:
            entry = {
                "text": exp_text,
                "cue": exp.cue,
                "lesson": exp.lesson,
                "evidence": exp.evidence,
                "confidence": exp.confidence,
                "dimension": exp.dimension,
                "kind": exp.kind,
                "count": 0,
                "first_seen": now,
                "last_seen": now,
                "promoted_proposal_id": None,
            }
            lessons[key] = entry
        # High-confidence causal <experience> entries with non-confounded evidence
        # earn immediate promotion without waiting for 3 blind repetitions.
        bump = _DEFAULT_MIN_OCCURRENCES if exp.confidence.lower().startswith("high") else 1
        entry["count"] = int(entry.get("count", 0)) + bump
        entry["last_seen"] = now
        updated[key] = entry

    _save_index(workspace, index)
    return updated


def maybe_promote_recurring_lessons(
    new_lessons_md: str,
    *,
    task: str,
    workspace: str | Path | None = None,
    min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
    skills_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Create skill_workshop proposals for lessons seen >= min_occurrences."""
    ws = Path(workspace or Path.cwd())
    updated = record_lessons_in_index(new_lessons_md, ws)
    created: list[dict[str, Any]] = []

    try:
        from clawagents.skills.workshop.service import SkillWorkshopService

        service = SkillWorkshopService(ws, skills_dir)
        existing_names = {p.get("name", "") for p in service.list()}
    except Exception:
        logger.debug("lesson promotion: workshop unavailable", exc_info=True)
        return created

    index = _load_index(ws)
    index.setdefault("lessons", {})  # ensure the key persists on save

    for key, entry in updated.items():
        count = int(entry.get("count", 0))
        if count < min_occurrences:
            continue
        if entry.get("promoted_proposal_id"):
            continue

        text = str(entry.get("text", ""))
        cue = str(entry.get("cue", "")).strip()
        dim = str(entry.get("dimension", "INV")).strip() or "INV"
        ev = str(entry.get("evidence", "")).strip()
        name = slugify_lesson_name(cue or text)
        if name in existing_names:
            index["lessons"][key]["promoted_proposal_id"] = "existing"
            entry["promoted_proposal_id"] = "existing"
            continue

        desc = (cue or text)[:150].replace("\n", " ").strip()
        body = (
            f"---\n"
            f"name: {name}\n"
            f"description: {desc}\n"
            f"dimension: {dim}\n"
            f"---\n\n"
            f"# {name.replace('-', ' ').title()}\n\n"
            f"Promoted from PTRL / Harness² `{dim}` dimension (seen {count} times).\n\n"
            + (f"## Situation Trigger (Cue)\n{cue}\n\n" if cue else "")
            + f"## Guidance\n- {text}\n\n"
            f"## Evidence\n"
            + (f"{ev}\n" if ev else f"Extracted from lessons.md after run: {task[:120]}\n")
        )
        try:
            proposal = service.create(
                name=name,
                description=desc,
                body=body,
                goal=f"Automated Harness² [{dim}] promotion from PTRL lesson",
                evidence=ev or f"lesson_key={key}; occurrences={count}",
            )
            proposal_id = proposal.get("id")
            if proposal_id:
                index["lessons"][key]["promoted_proposal_id"] = proposal_id
                entry["promoted_proposal_id"] = proposal_id
                existing_names.add(name)
                created.append(proposal)
                logger.debug("PTRL: promoted lesson %s -> proposal %s", key, proposal_id)
        except Exception:
            logger.debug("PTRL: failed to promote lesson %s", key, exc_info=True)

    _save_index(ws, index)
    return created
