"""Comprehensive tests for the complete Harness² architecture in clawagents_py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from clawagents.providers.llm import LLMProvider, LLMResponse
from clawagents.trajectory.harness_context import (
    RolloutEvidence,
    materialize_evidence_workspace,
)
from clawagents.trajectory.harness_ledger import (
    BudgetError,
    BudgetExceededError,
    HarnessLedger,
    expected_rollouts,
    expected_sessions,
)
from clawagents.trajectory.harness_recursion import (
    Proposal,
    Rollout,
    propose_probe_compose,
    read_tree,
    score_rollout_confidence,
)
from clawagents.trajectory.harness_runmark import (
    MARK,
    inputs_fingerprint,
    is_runmark_valid,
    prepare_clean_run_dir,
    read_runmark,
    tree_fingerprint,
    write_runmark,
)
from clawagents.trajectory.harness_substrate import (
    ClawAgentSubstrate,
    Substrate,
    backup_and_apply_harness,
    compute_unified_diffs,
    optimize_workspace_harness,
    render_trajectory_markdown,
)
from clawagents.trajectory.harness_surface import (
    parse_choice_reply,
    parse_proposal_reply,
)

BASE_TREE = {"AGENTS.md": "# Workspace Rules\nVerify outputs before finishing.\n"}


def test_runmark_completes_reuses_and_wipes_partial_dirs(tmp_path: Path):
    run_dir = tmp_path / "base1"
    run_dir.mkdir()
    # Partial file from a killed process (no .adapt_complete) must be wiped!
    (run_dir / "partial.txt").write_text("half written", encoding="utf-8")

    in_fp = inputs_fingerprint("Reconcile ledger", None)
    assert is_runmark_valid(run_dir, expected_harness=BASE_TREE, expected_inputs_digest=in_fp) is False
    reused = prepare_clean_run_dir(
        run_dir, expected_harness=BASE_TREE, expected_inputs_digest=in_fp
    )
    assert reused is False
    assert not (run_dir / "partial.txt").exists()

    # Now write completion marker and verify reuse
    (run_dir / "deliverable.md").write_text("clean output", encoding="utf-8")
    write_runmark(run_dir, run_id="base1", harness=BASE_TREE, inputs_digest=in_fp)
    rec = read_runmark(run_dir)
    assert rec is not None
    assert rec.run_id == "base1"
    assert rec.harness_fingerprint == tree_fingerprint(BASE_TREE)

    assert prepare_clean_run_dir(
        run_dir, expected_harness=BASE_TREE, expected_inputs_digest=in_fp
    ) is True
    assert (run_dir / "deliverable.md").is_file()

    # Changing task prompt invalidates runmark and cleans run_dir
    changed_in_fp = inputs_fingerprint("Reconcile Q4 ledger instead", None)
    assert prepare_clean_run_dir(
        run_dir, expected_harness=BASE_TREE, expected_inputs_digest=changed_in_fp
    ) is False
    assert not (run_dir / MARK).exists()


def test_ledger_catches_duplicate_labels_before_counts_and_enforces_max_usd():
    ledger = HarnessLedger(mode="parallel", task_id="t1", k=1)
    # 4 rollouts and 3 sessions, BUT `INV1` recorded twice and `DOM1` missing!
    for r_lbl in ("base1", "INV1", "INV1", "agg1"):
        ledger.record_rollout(r_lbl, usd=0.05)
    for s_lbl in ("INV1", "DOM1", "agg1"):
        ledger.record_session(s_lbl, usd=0.02)

    assert ledger.R == expected_rollouts(1) == 4
    assert ledger.P == expected_sessions(1) == 3
    with pytest.raises(BudgetError, match="DUPLICATE labels"):
        ledger.assert_identity()

    # Cost cap enforcement
    capped = HarnessLedger(mode="parallel", task_id="t2", k=1, max_usd=0.10)
    capped.record_rollout("base1", usd=0.06)
    with pytest.raises(BudgetExceededError, match="budget exceeded"):
        capped.record_rollout("INV1", usd=0.06)


def test_evidence_workspace_contents_vs_names_mode(tmp_path: Path):
    rollouts = [
        RolloutEvidence(
            label="base1",
            ok=True,
            delivered=True,
            trajectory="### Turn 1\nRan SQL query.",
            deliverable_files={"report.md": "SECRET_CUSTOMER_TOTAL=99999\n"},
        )
    ]
    # Mode "names": INDEX.md lists report.md, but file contents are NOT written!
    ws_names = materialize_evidence_workspace(
        tmp_path / "names_ws",
        task="Produce report.md",
        parent_harness=BASE_TREE,
        rollouts=rollouts,
        deliverables_mode="names",
        decisions={"INV1": "Added check script"},
    )
    assert "names only" in ws_names.layout
    index_text = (ws_names.context_dir / "deliverables" / "base1" / "INDEX.md").read_text()
    assert "report.md" in index_text
    assert not (ws_names.context_dir / "deliverables" / "base1" / "report.md").exists()
    assert (ws_names.context_dir / "decisions" / "INV1.md").is_file()

    # Mode "contents": report.md is materialized for inspection
    ws_contents = materialize_evidence_workspace(
        tmp_path / "contents_ws",
        task="Produce report.md",
        parent_harness=BASE_TREE,
        rollouts=rollouts,
        deliverables_mode="contents",
    )
    assert (ws_contents.context_dir / "deliverables" / "base1" / "report.md").read_text() == (
        "SECRET_CUSTOMER_TOTAL=99999\n"
    )


@pytest.mark.asyncio
async def test_ppc_runmark_resume_and_adaptive_probe_savings(tmp_path: Path):
    run_dir = tmp_path / "ppc_run"
    calls: list[str] = []

    async def rollout(tree, label):
        calls.append(label)
        return Rollout(
            ok=True,
            delivered=True,
            deliverable="Draft summary: TODO verify unmatched rows.",
            trajectory=f"ran {label}",
            usd=0.04,
        )

    async def propose(prompt, label):
        if label == "DOM1":
            return Proposal(
                edits={
                    "skills/reconcile/SKILL.md": (
                        "---\nname: reconcile\ndescription: verify unmatched rows\n---\nSteps.\n"
                    )
                },
                notes="Add reconcile skill",
                usd=0.01,
            )
        # INV1 returns KEEP_BASE -> identical fingerprint to base1!
        return Proposal(keep_base=True, usd=0.01)

    # First run with adaptive_probe=True:
    # - base1 has "TODO" so score_rollout_confidence < 0.90 -> does NOT short-circuit base
    # - INV1 keeps base -> its probe reuses base1's rollout via tree_fingerprint!
    res1 = await propose_probe_compose(
        "Reconcile rows",
        BASE_TREE,
        rollout=rollout,
        propose=propose,
        k=1,
        adaptive_probe=True,
        run_dir=run_dir,
    )
    assert res1.harness_ledger is not None
    assert "INV1" not in calls, "INV1 kept base tree so adaptive_probe should reuse base1's probe"
    assert res1.stage("INV1").reused_from == "base1"
    assert res1.harness_ledger.rollouts_saved >= 1

    # Second run resuming from the same run_dir -> base1 & DOM1 load from .adapt_complete!
    calls.clear()
    res2 = await propose_probe_compose(
        "Reconcile rows",
        BASE_TREE,
        rollout=rollout,
        propose=propose,
        k=1,
        adaptive_probe=True,
        run_dir=run_dir,
    )
    assert calls == [], "All completed probes should resume from .adapt_complete without re-running"
    assert res2.stage("base1").cached is True


@pytest.mark.asyncio
async def test_adaptive_probe_short_circuits_when_base_is_confident():
    propose_calls: list[str] = []
    rollout_calls: list[str] = []

    async def confident_rollout(tree, label):
        rollout_calls.append(label)
        return Rollout(
            ok=True,
            delivered=True,
            deliverable=(
                "# Complete Reconciliation Report\n\n"
                "Matched records: 1,240. Unmatched records: 0. Cross-checked all totals cleanly."
            ),
            trajectory="### Turn 1\nVerified every row with zero errors.",
            usd=0.05,
        )

    async def should_not_propose(prompt, label):
        propose_calls.append(label)
        return Proposal(keep_base=True)

    r_sample = await confident_rollout(BASE_TREE, "sample")
    assert score_rollout_confidence(r_sample) >= 0.90
    rollout_calls.clear()

    res = await propose_probe_compose(
        "Produce reconciliation report",
        BASE_TREE,
        rollout=confident_rollout,
        propose=should_not_propose,
        k=2,
        mode="parallel",
        adaptive_probe=True,
        confidence_threshold=0.90,
    )
    assert rollout_calls == ["base1", "base2"]
    assert propose_calls == []
    assert res.harness_ledger is not None
    assert res.harness_ledger.rollouts_executed == 2
    assert res.harness_ledger.rollouts_saved == 6  # saved 2k candidates + k finals = 6 rollouts!
    assert res.harness_ledger.sessions_saved == 6  # saved 3k = 6 edit sessions!


@pytest.mark.asyncio
async def test_substrate_trajectory_rendering_and_optimize_workspace_harness(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "AGENTS.md").write_text("# Base\nWork carefully in {{workspace}}.\n", encoding="utf-8")
    (ws / "data.csv").write_text("id,amount\n1,100\n2,250\n", encoding="utf-8")

    # Verify JSONL trajectory rendering for both SessionWriter and TrajectoryRecorder shapes
    session_jsonl = tmp_path / "session.jsonl"
    session_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"type": "turn_started", "iteration": 1}),
                json.dumps(
                    {
                        "type": "assistant_message",
                        "content": "Inspecting CSV",
                        "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "data.csv"}}],
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_result",
                        "tool_call_id": "c1",
                        "name": "read_file",
                        "success": True,
                        "output": "id,amount\n1,100\n2,250",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    md = render_trajectory_markdown(session_jsonl)
    assert "### Turn 1" in md and "read_file" in md and "1,100" in md

    class ScriptedOptimizerLLM(LLMProvider):
        name = "scripted-opt"
        model = "gpt-5.4-mini"

        async def chat(self, messages, **kwargs):
            prompt = messages[-1].content
            if "YOUR DIMENSION: DOM" in prompt:
                payload = {
                    "keep_base": False,
                    "notes": "Add csv-reconcile skill for subtotal checks",
                    "edits": {
                        "skills/csv-reconcile/SKILL.md": (
                            "---\nname: csv-reconcile\n"
                            "description: when reconciling CSV totals\n---\n\n"
                            "Always sum `amount` rows explicitly.\n"
                        )
                    },
                    "removed": [],
                }
                return LLMResponse(content=json.dumps(payload), model=self.model, tokens_used=40)
            if "YOUR DIMENSION: INV" in prompt:
                return LLMResponse(
                    content=json.dumps(
                        {"keep_base": True, "notes": "", "edits": {}, "removed": []}
                    ),
                    model=self.model,
                    tokens_used=20,
                )
            if "integration judge" in prompt:
                payload = {
                    "keep_base": False,
                    "notes": "kept -> skills/csv-reconcile/SKILL.md -> improved CSV check",
                    "edits": {
                        "skills/csv-reconcile/SKILL.md": (
                            "---\nname: csv-reconcile\n"
                            "description: when reconciling CSV totals\n---\n\n"
                            "Always sum `amount` rows explicitly.\n"
                        )
                    },
                    "removed": [],
                }
                return LLMResponse(content=json.dumps(payload), model=self.model, tokens_used=45)
            return LLMResponse(
                content=json.dumps({"why": "COVERAGE won cleanly.", "choice": "A"}),
                model=self.model,
                tokens_used=15,
            )

    llm = ScriptedOptimizerLLM()
    substrate = ClawAgentSubstrate(ws, llm=llm)
    assert isinstance(substrate, Substrate)
    assert substrate.preflight()["ok"] is True

    staging = tmp_path / "staged_check"
    install = substrate.materialise(read_tree(ws), staging, ctx={"workspace": "/tmp/demo"})
    assert install.fingerprint
    assert "/tmp/demo" in (staging / "AGENTS.md").read_text(encoding="utf-8")

    stage_events: list[str] = []

    async def fake_rollout(tree, label):
        has_skill = "skills/csv-reconcile/SKILL.md" in tree
        return Rollout(
            ok=True,
            delivered=True,
            deliverable=(
                "Total amount: 350 (verified)" if has_skill else "Draft total: TODO check"
            ),
            trajectory=f"### Turn 1\nExecuted {label} (has_skill={has_skill})",
            usd=0.01,
        )

    outcome = await optimize_workspace_harness(
        "Compute exact CSV total.",
        ws,
        llm=llm,
        k=1,
        mode="parallel",
        adaptive_probe=True,
        apply=True,
        rollout_fn=fake_rollout,
        on_stage_event=lambda kind, _data: stage_events.append(kind),
    )
    assert outcome["ok"] is True
    assert "skills/csv-reconcile/SKILL.md" in outcome["adopted_tree"]
    assert "skills/csv-reconcile/SKILL.md" in outcome["unified_diffs"]
    assert (ws / "skills" / "csv-reconcile" / "SKILL.md").is_file()
    assert (ws / ".clawagents" / "skill-triggers.md").is_file()
    assert "selection_completed" in stage_events

    # Also verify JSON choice & proposal parsers directly
    anon, lbl, why = parse_choice_reply(
        '{"why": "FIDELITY won", "choice": "B"}', {"A": "agg1", "B": "agg2"}
    )
    assert (anon, lbl, why) == ("B", "agg2", "FIDELITY won")
    edits, removed, notes, keep = parse_proposal_reply("KEEP_BASE")
    assert keep is True and edits == {} and removed == ()
    diffs = compute_unified_diffs(BASE_TREE, outcome["adopted_tree"])
    assert diffs
    backup_info = backup_and_apply_harness(
        tmp_path / "apply_target",
        BASE_TREE,
        outcome["adopted_tree"],
        backup_dir=tmp_path / "backup_snap",
    )
    assert "skills/csv-reconcile/SKILL.md" in backup_info["written"]

    # Rollback restores base AGENTS.md and deletes newly added skill files
    from clawagents.trajectory.harness_substrate import rollback_workspace_harness

    target_ws = tmp_path / "apply_target"
    assert (target_ws / "skills" / "csv-reconcile" / "SKILL.md").is_file()
    rb = rollback_workspace_harness(target_ws, tmp_path / "backup_snap")
    assert "AGENTS.md" in rb["restored"]
    assert not (target_ws / "skills" / "csv-reconcile" / "SKILL.md").exists()
    assert not (target_ws / "skills" / "csv-reconcile").exists()
    assert (target_ws / "AGENTS.md").read_text(encoding="utf-8") == BASE_TREE["AGENTS.md"]


@pytest.mark.asyncio
async def test_optimize_workspace_harness_multi_task_suite(tmp_path: Path):
    from clawagents.providers.base import LLMResponse
    from clawagents.trajectory.harness_substrate import optimize_workspace_harness

    ws = tmp_path / "suite_ws"
    ws.mkdir()
    (ws / "AGENTS.md").write_text("# Suite base\n", encoding="utf-8")

    class SuiteFakeLLM:
        model = "stub-model"

        async def chat(self, messages, **kwargs):
            text = "\n".join(str(getattr(m, "content", "")) for m in messages)
            if '"keep_base"' in text or "Propose minimal" in text:
                return LLMResponse(content='{"keep_base": true, "notes": "no edits needed", "edits": {}, "removed": []}', model="stub-model", tokens_used=10)
            return LLMResponse(content='{"why": "clean", "choice": "A"}', model="stub-model", tokens_used=10)

    seen_prompts: list[str] = []

    async def fake_rollout(tree, label):
        seen_prompts.append(label)
        return Rollout(
            ok=True,
            delivered=True,
            deliverable=f"done {label}",
            trajectory=f"ran {label}",
            usd=0.005,
        )

    out = await optimize_workspace_harness(
        ["Task one: verify unit tests", "Task two: verify lint"],
        ws,
        llm=SuiteFakeLLM(),
        k=1,
        rollout_fn=fake_rollout,
    )
    assert out["ok"] is True
    assert out["result"].chosen in {"s0", "agg1"}
    assert len(seen_prompts) >= 1


def test_cli_adapt_and_rollback_harness(tmp_path: Path, monkeypatch, capsys):
    from clawagents import __main__ as cli_main
    from clawagents.trajectory import harness_substrate as hs

    ws = tmp_path / "cli_ws"
    ws.mkdir()
    (ws / "AGENTS.md").write_text("# Old CLI rules\n", encoding="utf-8")
    backup_dir = ws / ".clawagents" / "harness_runs" / "workspace-harness" / "backup"

    async def fake_opt(task, workspace, **kwargs):
        binfo = hs.backup_and_apply_harness(
            workspace,
            {"AGENTS.md": "# Old CLI rules\n"},
            {"AGENTS.md": "# New CLI rules\n"},
            backup_dir=backup_dir,
        )
        return {
            "ok": True,
            "result": type(
                "PPC",
                (),
                {
                    "chosen": "s1",
                    "selection": {"chosen_label": "s1"},
                    "ledger": {"rollouts": 3, "stages": 3},
                    "harness_ledger": None,
                },
            )(),
            "diff_summary": "modified AGENTS.md",
            "unified_diffs": {"AGENTS.md": "+# New CLI rules"},
            "applied": binfo,
        }

    monkeypatch.setattr(hs, "optimize_workspace_harness", fake_opt)
    monkeypatch.setattr(
        sys,
        "argv",
        ["clawagents", "--adapt", "Fix tests", "--workspace", str(ws), "--apply", "--json"],
    )
    cli_main.main()
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["chosen"] == "s1"
    assert (ws / "AGENTS.md").read_text(encoding="utf-8") == "# New CLI rules\n"

    monkeypatch.setattr(
        sys,
        "argv",
        ["clawagents", "--rollback-harness", str(backup_dir), "--workspace", str(ws), "--json"],
    )
    cli_main.main()
    rb_json = json.loads(capsys.readouterr().out)
    assert "AGENTS.md" in rb_json["restored"]
    assert (ws / "AGENTS.md").read_text(encoding="utf-8") == "# Old CLI rules\n"
