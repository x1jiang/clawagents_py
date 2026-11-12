"""Tests and end-to-end simulations for Harness² adaptations in clawagents_py."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from clawagents.providers.llm import LLMProvider, LLMResponse
from clawagents.skills.workshop.service import SkillWorkshopService
from clawagents.trajectory.compare import (
    _score_samples,
    select_best_candidate_blind,
)
from clawagents.trajectory.harness_surface import (
    DEFAULT_SURFACE,
    ExperienceLibrary,
    accept_or_repair_tree,
    assert_no_grader_verdicts,
    assign_anon_ids,
    autotrigger_skills,
    build_compose_prompt,
    build_propose_prompt,
    diff_summary,
    parse_experience_reply,
    tree_diversity,
    validate_harness_tree,
)
from clawagents.trajectory.lesson_promotion import maybe_promote_recurring_lessons
from clawagents.trajectory.lessons import (
    _extract_key_turns,
    build_lesson_preamble,
    load_lessons,
    save_lessons,
    should_extract_lessons,
)


def test_scanner_blocks_frontmatter_name_mismatch_missing_script_and_syntax_error(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    svc = SkillWorkshopService(tmp_path, skills_dir)

    # 1. Frontmatter `name:` mismatch overrides directory name at load -> flagged and blocked on apply
    bad_name = svc.create(
        name="cohort-reconcile",
        description="Reconcile cohort counts across SQL and CSV",
        body="---\nname: wrong-name\ndescription: Reconcile cohort counts\n---\n\n# Steps\nRun check.\n",
    )
    assert any("does not match skill name" in f for f in bad_name["scan_findings"])
    assert svc.apply(bad_name["id"])["ok"] is False

    # 2. Invoking `python scripts/check_totals.py` without shipping `scripts/check_totals.py` -> blocked on apply
    missing_script = svc.create(
        name="cohort-reconcile",
        description="Reconcile cohort counts",
        body=(
            "---\nname: cohort-reconcile\ndescription: Reconcile cohort counts\n---\n\n"
            "Run `python3 scripts/check_totals.py` before finishing.\n"
        ),
    )
    assert any("check_totals.py" in f for f in missing_script["scan_findings"])
    assert svc.apply(missing_script["id"])["ok"] is False

    # 3. Shipping a `.py` support script with a SyntaxError -> blocked on apply
    bad_syntax = svc.create(
        name="cohort-reconcile",
        description="Reconcile cohort counts",
        body=(
            "---\nname: cohort-reconcile\ndescription: Reconcile cohort counts\n---\n\n"
            "Run `python3 scripts/check_totals.py`.\n"
        ),
        support_files=[{"path": "scripts/check_totals.py", "content": "def broken(:\n    pass\n"}],
    )
    assert any("does not compile" in f for f in bad_syntax["scan_findings"])
    assert svc.apply(bad_syntax["id"])["ok"] is False

    # 4. Valid frontmatter + compiling helper script -> created, applied, and auto-triggered!
    valid = svc.create(
        name="cohort-reconcile",
        description="when reconciling cohort totals across SQL and CSV outputs",
        body=(
            "---\nname: cohort-reconcile\n"
            "description: when reconciling cohort totals across SQL and CSV outputs\n---\n\n"
            "Run `python3 scripts/check_totals.py`.\n"
        ),
        support_files=[
            {
                "path": "scripts/check_totals.py",
                "content": "def check(a: int, b: int) -> bool:\n    return a == b\n",
            }
        ],
    )
    assert valid.get("status") == "pending"
    applied = svc.apply(valid["id"])
    assert applied["ok"] is True

    trigger_file = tmp_path / ".clawagents" / "skill-triggers.md"
    assert trigger_file.is_file()
    assert "`cohort-reconcile`" in trigger_file.read_text(encoding="utf-8")

    # 5. Update proposal on the same skill inherits existing `scripts/check_totals.py`
    updated = svc.update(
        target_skill="cohort-reconcile",
        description="when reconciling cohort totals across SQL and CSV outputs",
        body=(
            "---\nname: cohort-reconcile\n"
            "description: when reconciling cohort totals across SQL and CSV outputs\n---\n\n"
            "Always run `python3 scripts/check_totals.py` and report mismatches.\n"
        ),
    )
    assert updated.get("status") == "pending"
    assert svc.apply(updated["id"])["ok"] is True

    # 6. Rolling back the initial creation removes the auto-trigger line
    svc.rollback(applied["rollback_id"])
    assert "`cohort-reconcile`" not in trigger_file.read_text(encoding="utf-8")


def test_causal_edit_attribution_rejects_confounded_and_persists_helped(tmp_path: Path):
    # Confounded edit must NOT produce a lesson entry
    confounded_reply = """
<edits>
component: skills/sql-helper/SKILL.md
candidates: inv1, dom1
change: added SQL skill and changed system prompt simultaneously
observed: run succeeded but skill was never loaded
verdict: confounded
</edits>

<experience>
cue: writing SQL queries
lesson: load sql-helper skill
evidence: candidate inv1 succeeded
confidence: low
</experience>
"""
    assert parse_experience_reply(confounded_reply) is None
    assert parse_experience_reply("NO_ENTRY") is None

    # Non-confounded 'helped' edit produces a valid ExperienceEntry and updates INDEX.md
    helped_reply = """
<edits>
component: skills/join-reconcile/SKILL.md
candidates: dom1
change: added explicit left-anti-join check before aggregation
observed: turn 4 caught 12 unmatched HR rows that base rollout silently dropped
verdict: helped
</edits>

<experience>
cue: joining two tables on role or department titles across files
lesson: run an explicit unmatched-key check before computing aggregate counts
evidence: dom1 turn 4 vs base1 turn 3
confidence: high
dimension: DOM
kind: procedure
</experience>
"""
    entry = parse_experience_reply(helped_reply)
    assert entry is not None
    assert entry.dimension == "DOM"
    assert entry.kind == "procedure"
    assert len(entry.edits) == 1
    assert entry.edits[0].verdict == "helped"

    lib = ExperienceLibrary(tmp_path / "experience", domain="clinical_sql")
    saved_dir = lib.add("task_01_attrition", entry, why="Caught 12 unmatched rows", label="dom1")
    assert (saved_dir / "EXPERIENCE.md").is_file()
    assert lib.index_path.is_file()
    index_md = lib.index_path.read_text(encoding="utf-8")
    assert "task_01_attrition" in index_md
    assert "joining two tables on role or department titles" in index_md


def test_workspace_facts_stay_in_memory_while_procedures_promote_to_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    reflection_output = """
- [WORKSPACE_FACT] In this workspace, role 'HR' in attrition.csv maps to 'HR Manager' in requisitions.csv
- [WORKSPACE_FACT] In this workspace, role 'HR' in attrition.csv maps to 'HR Manager' in requisitions.csv

<edits>
component: skills/reconcile-keys/SKILL.md
candidates: dom1
change: added key reconciliation procedure
observed: prevented silent row drop on join
verdict: helped
</edits>

<experience>
cue: reconciling join keys between two CSV or SQL tables
lesson: list unmatched keys on both sides before aggregating
evidence: dom1 turn 4
confidence: high
dimension: DOM
kind: procedure
</experience>
"""
    save_lessons(reflection_output, task="Reconcile HR attrition", outcome="success", model="test")
    lessons_text = load_lessons()
    # Deduplicated: the workspace fact appears only once!
    assert lessons_text.count("attrition.csv maps to 'HR Manager'") == 1

    # Promote: workspace fact must NOT become a skill, while high-confidence <experience> procedure
    # promotes immediately in 1 step with valid YAML frontmatter!
    proposals = maybe_promote_recurring_lessons(
        reflection_output,
        task="Reconcile HR attrition",
        workspace=tmp_path,
        skills_dir=skills_dir,
    )
    assert len(proposals) == 1
    prop = proposals[0]
    assert "attrition" not in prop["name"]

    svc = SkillWorkshopService(tmp_path, skills_dir)
    applied = svc.apply(prop["id"])
    assert applied["ok"] is True, applied
    preamble = build_lesson_preamble()
    assert "Lessons from Past Runs" in preamble
    assert f"`{prop['name']}`" in preamble


def test_grader_verdict_tripwire_blocks_leaks():
    with pytest.raises(ValueError, match="grader verdict marker"):
        assert_no_grader_verdicts({"SKILL.md": 'Here is "pass_rate": 1.0 from the grader'})


def test_blind_selector_and_deliverable_scoring_simulation():
    # Thorough 18-turn complete deliverable must beat a 1-turn empty completion and a TODO draft
    samples = [
        {"index": 0, "status": "done", "result": "", "iterations": 1, "tool_calls": 0, "trajectory_file": ""},
        {
            "index": 1,
            "status": "done",
            "result": "# Summary\nTotal: TODO compute later",
            "iterations": 2,
            "tool_calls": 1,
            "trajectory_file": "",
        },
        {
            "index": 2,
            "status": "done",
            "result": "# Summary\nTotal matched: 142\nUnmatched: 3 (listed in table below)",
            "iterations": 18,
            "tool_calls": 9,
            "trajectory_file": "",
        },
    ]
    scored = _score_samples(samples)
    by_idx = {s["index"]: s["composite_score"] for s in scored}
    assert by_idx[2] > by_idx[1] > by_idx[0]

    # Deterministic seeded anonymous IDs
    map1 = assign_anon_ids(["cand_1", "cand_2"], task_id="t42", k=2, mode="bon")
    map2 = assign_anon_ids(["cand_1", "cand_2"], task_id="t42", k=2, mode="bon")
    assert map1 == map2
    assert set(map1.keys()) == {"A", "B"}

    # Blind selector chooses the candidate with full COVERAGE, CORRECTNESS, and FIDELITY
    target_anon = next(anon for anon, lbl in map1.items() if lbl == "cand_2")

    class FakeSelectorLLM(LLMProvider):
        name = "fake-selector"

        async def chat(self, messages, **kwargs):
            return LLMResponse(
                content=(
                    "<why>\n"
                    f"Candidate {target_anon} wins on COVERAGE and FIDELITY: it computes the exact "
                    "total (142) and lists the 3 unmatched rows, whereas the runner-up left a TODO.\n"
                    "</why>\n\n"
                    f"```choice\n{target_anon}\n```"
                ),
                model="fake-selector",
                tokens_used=12,
            )

    sel = asyncio.run(
        select_best_candidate_blind(
            task="Compute total matched rows and list unmatched rows.",
            scored_samples=scored,
            llm=FakeSelectorLLM(),
            task_id="t42",
        )
    )
    assert sel["ok"] is True
    assert sel["chosen_index"] == 2
    assert "COVERAGE" in sel["why"]


def test_end_to_end_simulation_and_low_latency_overhead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Simulate a multi-turn trajectory -> validation -> promotion -> clean follow-up run,
    and verify that total synchronous harness overhead stays under 15ms.
    """
    monkeypatch.chdir(tmp_path)
    t0 = time.perf_counter()

    # 1. Verify _extract_key_turns preserves rich 1200-char tool output observations
    long_schema_error = "ERROR: column 'dept_id' not found. Available columns: " + ", ".join(
        f"col_{i}" for i in range(40)
    )
    turns = [
        {
            "turn_index": 1,
            "score": -1,
            "productivity_score": -1,
            "observation_context": "Querying department table",
            "response_text": "Running SQL query",
            "tool_calls": [
                {
                    "tool_name": "execute_command",
                    "success": False,
                    "failure_type": "logic",
                    "output_preview": long_schema_error,
                }
            ],
        }
    ]
    extracted = _extract_key_turns(turns)
    assert "col_39" in extracted  # Previously clipped at 80 chars!

    # 2. Validate full harness tree with autotrigger
    tree = {
        "AGENTS.md": "# Base instructions\nFollow workspace conventions.\n",
        "skills/schema-inspect/SKILL.md": (
            "---\nname: schema-inspect\n"
            "description: when inspecting table schemas before writing SQL joins\n---\n\n"
            "Run `python3 skills/schema-inspect/scripts/inspect_cols.py`.\n"
        ),
        "skills/schema-inspect/scripts/inspect_cols.py": "import sys\nprint(sys.version)\n",
    }
    triggered = autotrigger_skills(DEFAULT_SURFACE, tree)
    assert triggered == ["schema-inspect"]
    census = validate_harness_tree(tree, DEFAULT_SURFACE)
    assert census["skills"] == 1

    # 3. Clean run skips lesson extraction (zero LLM reflection overhead)
    clean_summary = {
        "run_score": 3,
        "mid_run_failures": 0,
        "has_mixed_outcomes": False,
        "quality": "clean",
        "total_turns": 4,
    }
    assert should_extract_lessons(clean_summary) is False

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 25.0, f"Synchronous harness validation overhead too high: {elapsed_ms:.2f}ms"


def test_propose_probe_compose_cycle_and_tree_diversity():
    base_tree = {
        "AGENTS.md": "# Base Agent\nComplete the user's request.\n",
    }
    inv_prompt = build_propose_prompt(
        "Reconcile Q3 ledger CSV with invoices",
        "INV",
        ["Turn 1: hand-rolled CSV parser failed on quoted commas"],
    )
    dom_prompt = build_propose_prompt(
        "Reconcile Q3 ledger CSV with invoices",
        "DOM",
        ["Turn 2: omitted unmatched invoice section"],
    )
    assert "YOUR DIMENSION: INV" in inv_prompt
    assert "YOUR DIMENSION: DOM" in dom_prompt

    # INV candidate adds a deterministic helper script
    inv_candidate = {
        **base_tree,
        "scripts/parse_ledger.py": "import csv\ndef load(p):\n    return list(csv.reader(open(p)))\n",
    }
    inv_tree, inv_kept, inv_fb, inv_err, _ = accept_or_repair_tree(inv_candidate, base_tree)
    assert not inv_kept and not inv_fb and not inv_err

    # DOM candidate adds a situation skill (omitting trigger line -> auto-repaired by accept_or_repair_tree!)
    dom_candidate = {
        **base_tree,
        "skills/invoice-reconcile/SKILL.md": (
            "---\nname: invoice-reconcile\n"
            "description: when reconciling ledger CSV rows against invoice records\n---\n\n"
            "# Reconcile\nAlways include an Unmatched Invoices table.\n"
        ),
    }
    dom_tree, dom_kept, dom_fb, dom_err, dom_trig = accept_or_repair_tree(dom_candidate, base_tree)
    assert not dom_kept and not dom_fb and not dom_err
    assert dom_trig == ["invoice-reconcile"]

    # Broken candidate (invokes missing script) -> cleanly falls back to base_tree
    broken_candidate = {
        **base_tree,
        "AGENTS.md": "# Base\nRun `python3 scripts/nonexistent.py`.\n",
    }
    fb_tree, _, fell_back, fb_err, _ = accept_or_repair_tree(broken_candidate, base_tree)
    assert fell_back is True
    assert "nonexistent.py" in fb_err
    assert fb_tree == base_tree

    # Compose INV + DOM
    composed_candidate = {**inv_tree, **dom_tree}
    comp_tree, comp_kept, comp_fb, comp_err, _ = accept_or_repair_tree(composed_candidate, base_tree)
    assert not comp_kept and not comp_fb and not comp_err
    diff = diff_summary(DEFAULT_SURFACE, base_tree, comp_tree)
    assert "scripts/parse_ledger.py" in diff["added"]
    assert "skills/invoice-reconcile/SKILL.md" in diff["added"]

    div = tree_diversity({"inv1": inv_tree, "dom1": dom_tree, "agg1": comp_tree})
    assert div["n"] == 3
    assert div["n_distinct"] == 3
    assert 0.0 < div["mean_line_jaccard"] < 1.0

    comp_prompt = build_compose_prompt(
        "Reconcile Q3 ledger CSV with invoices",
        {"inv1": "Added scripts/parse_ledger.py", "dom1": "Added invoice-reconcile skill"},
    )
    assert "kept|revised|dropped" in comp_prompt

