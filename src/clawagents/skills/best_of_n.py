"""Best-of-N parallel worktree tournament skill (Grok-inspired)."""

from pathlib import Path

_SKILL = """---
name: best-of-n
description: >
  Implement a task N ways in parallel and pick the best. Spawns multiple
  subagents in isolated worktrees, evaluates candidates, and applies the
  winner. Use when asked to "best of n", "try multiple approaches",
  "parallel implementations", "/best-of-n", or "/bon".
when-to-use: multiple independent implementations should compete
aliases: [bon, best of n, tournament]
triggers: [best-of-n, best of n, /bon, parallel implementations]
---

# /best-of-n — Parallel Implementation Tournament

Implement a task multiple different ways in parallel, evaluate all candidates,
and apply the best one.

## Usage

`/best-of-n [N] <task>`

- If the first token is a number 2–8, it sets the candidate count; the rest is the task.
- If omitted, N defaults to 3.

## Steps

1. Parse **N** (default 3, clamp 2–8) and the **task description**.

2. Spawn **N** subagents in one turn (parallel tool calls). For each candidate use `task` with:
   - `isolation`: `"worktree"`
   - `description`: `"Candidate <i>"`
   - `prompt`: the task plus
     `"You are candidate <i> of <N>. Implement fully. Summarize approach and files changed."`

3. Wait for all candidates (`get_task_output` / wait semantics if available).

4. Evaluate each on: correctness → completeness → simplicity → risk.

5. Apply the winner's changes into the main workspace (review diffs; merge carefully).

6. End with `WINNER: <number>` (1–N).

## Guardrails

- Do not claim a winner without reading each candidate summary.
- Prefer the smallest correct change when quality ties.
- If all candidates fail, report failure — do not invent a merge.
"""


def ensure_best_of_n_skill(skills_root: Path | None = None) -> Path:
    """Write bundled best-of-n SKILL.md if missing; return its path."""
    root = skills_root or Path(__file__).resolve().parent
    target = root / "best-of-n" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(_SKILL, encoding="utf-8")
    return target
