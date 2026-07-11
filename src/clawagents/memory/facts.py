"""Local fact store with supersession (Zep-inspired, no cloud)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Fact:
    id: str
    text: str
    created_at: float
    superseded_by: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def live(self) -> bool:
        return not self.superseded_by


def facts_path(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or Path.cwd()) / ".clawagents"
    root.mkdir(parents=True, exist_ok=True)
    return root / "facts.jsonl"


def add_fact(
    text: str,
    *,
    workspace: str | Path | None = None,
    tags: list[str] | None = None,
    supersedes: str | None = None,
) -> Fact:
    path = facts_path(workspace)
    fact = Fact(
        id=uuid.uuid4().hex[:12],
        text=text.strip(),
        created_at=time.time(),
        tags=list(tags or []),
    )
    # Mark old fact superseded
    if supersedes:
        rows = list_facts(workspace=workspace, live_only=False)
        out_lines: list[str] = []
        for r in rows:
            if r.id == supersedes:
                r.superseded_by = fact.id
            out_lines.append(json.dumps(asdict(r)))
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(fact)) + "\n")
    return fact


def list_facts(
    *,
    workspace: str | Path | None = None,
    live_only: bool = True,
) -> list[Fact]:
    path = facts_path(workspace)
    if not path.is_file():
        return []
    out: list[Fact] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            f = Fact(**raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if live_only and not f.live:
            continue
        out.append(f)
    return out


def live_facts_preamble(
    *,
    workspace: str | Path | None = None,
    max_chars: int = 2_000,
) -> str:
    facts = list_facts(workspace=workspace, live_only=True)
    if not facts:
        return ""
    lines = ["## Live Facts", ""]
    used = 0
    for f in facts[-40:]:
        row = f"- [{f.id}] {f.text}"
        if used + len(row) > max_chars:
            break
        lines.append(row)
        used += len(row)
    return "\n".join(lines) + "\n"


def promote_lesson_bullets_to_facts(
    lessons_text: str,
    *,
    workspace: str | Path | None = None,
) -> list[Fact]:
    """Extract - bullets and add as live facts (dedupe by lowercase text)."""
    import re

    existing = {f.text.lower() for f in list_facts(workspace=workspace, live_only=True)}
    created: list[Fact] = []
    for m in re.finditer(r"(?m)^\s*[-*]\s+(.+)$", lessons_text or ""):
        text = m.group(1).strip()
        if len(text) < 12 or text.lower() in existing:
            continue
        fact = add_fact(text, workspace=workspace, tags=["lesson"])
        existing.add(text.lower())
        created.append(fact)
    return created
