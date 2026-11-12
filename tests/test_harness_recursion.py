"""Propose–Probe–Compose loop and the Harness² refinements, with scripted fakes."""

from __future__ import annotations

import asyncio
import json

import pytest

from clawagents.trajectory.harness_recursion import (
    PPCResult,
    Proposal,
    ProposerCache,
    Rollout,
    apply_edits,
    build_reflection_prompt,
    propose_probe_compose,
    read_tree,
    record_experience,
    write_tree,
)
from clawagents.trajectory.harness_surface import (
    DEFAULT_SURFACE,
    ExperienceEntry,
    ExperienceLibrary,
    SurfaceSpec,
    accept_or_repair_tree,
    autotrigger_skills,
    check_hooks,
    lint_task_prompt,
    validate_harness_tree,
)

BASE = {"AGENTS.md": "# Rules\nBe careful.\n"}
SKILL = "---\nname: {n}\ndescription: reconcile the ledger totals\n---\nSteps.\n"


def skill_edit(name: str) -> Proposal:
    return Proposal(edits={f"skills/{name}/SKILL.md": SKILL.format(n=name)}, notes=f"<notes>add {name}</notes>")


class Harness:
    """Scripted rollout/propose/select with call logs."""

    def __init__(self, proposals=None, deliver=lambda label, tree: f"answer with {len(tree)} harness file(s)"):
        self.proposals = proposals or {}
        self.deliver = deliver
        self.rollouts: list[str] = []
        self.prompts: list[tuple[str, str]] = []
        self.selector_prompts: list[str] = []
        self.choice = "A"

    async def rollout(self, tree, label):
        self.rollouts.append(label)
        text = self.deliver(label, tree)
        if isinstance(text, Exception):
            raise text
        return Rollout(ok=True, delivered=bool(text), deliverable=text or "",
                       trajectory=f"[{label}] files={sorted(tree)}")

    async def propose(self, prompt, label):
        self.prompts.append((label, prompt))
        script = self.proposals.get(label, Proposal(keep_base=True))
        if isinstance(script, list):
            script = script.pop(0)
        if isinstance(script, Exception):
            raise script
        return script

    async def select(self, prompt):
        self.selector_prompts.append(prompt)
        return f"<why>COVERAGE decided it.</why>\n```choice\n{self.choice}\n```"

    def run(self, **kw):
        kw.setdefault("rollout", self.rollout)
        kw.setdefault("propose", self.propose)
        kw.setdefault("select", self.select)
        return propose_probe_compose("Reconcile the ledger.", BASE, **kw)


# ── budgets and structure ────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
@pytest.mark.parametrize("k", [1, 2, 3])
async def test_budget_is_4k_rollouts_and_3k_stages(mode, k):
    h = Harness()
    result = await h.run(k=k, mode=mode)
    assert result.ledger == {"rollouts": 4 * k, "stages": 3 * k}
    assert len(h.rollouts) == 4 * k and len(h.prompts) == 3 * k
    assert len(result.finals) == k


async def test_parallel_layout_and_shared_evidence():
    h = Harness()
    result = await h.run(k=2, mode="parallel")
    assert [s.label for s in result.stages] == [
        "base1", "base2", "INV1", "DOM1", "INV2", "DOM2", "agg1", "agg2"]
    assert result.finals == ["agg1", "agg2"]
    # Every candidate is proposed from the same base evidence.
    for _label, prompt in h.prompts[:4]:
        assert "[base1]" in prompt and "[base2]" in prompt
    # Each composition sees all 2k candidates.
    for _label, prompt in h.prompts[4:]:
        assert all(f"Candidate `{c}`" in prompt for c in ("INV1", "DOM1", "INV2", "DOM2"))


async def test_sequential_rounds_build_on_the_previous_checkpoint():
    h = Harness(proposals={"c1": skill_edit("ledger"), "c2": skill_edit("tax")})
    result = await h.run(k=2, mode="sequential")
    assert [s.label for s in result.stages] == ["base1", "base2", "INV1", "DOM1", "c1", "INV2", "DOM2", "c2"]
    assert result.stage("INV2").parent == "c1" and result.stage("c2").parent == "c1"
    # Round 2 starts from c1's tree, so c2 carries both skills.
    assert {"skills/ledger/SKILL.md", "skills/tax/SKILL.md"} <= set(result.stage("c2").tree)
    # Round 2's editors see c1's rollout, not the base rollouts.
    inv2_prompt = dict(h.prompts)["INV2"]
    assert "[c1]" in inv2_prompt and "[base1]" not in inv2_prompt


# ── edits, accept-or-repair, fallback ────────────────────────────────────


def test_apply_edits_is_omitted_is_inherited():
    tree = apply_edits({"a": "1", "b": "2"}, Proposal(edits={"b": "3", "c": "4"}, removed=("a",)))
    assert tree == {"b": "3", "c": "4"}
    assert Proposal().is_noop and Proposal(keep_base=True, edits={"x": "y"}).is_noop


async def test_keep_base_is_recorded_not_treated_as_failure():
    result = await Harness().run(k=1)
    inv = result.stage("INV1")
    assert inv.kept_base and not inv.fell_back and inv.tree == BASE and inv.attempts == 1


async def test_valid_edit_is_accepted_and_skill_is_autotriggered():
    h = Harness(proposals={"DOM1": skill_edit("ledger")})
    result = await h.run(k=1)
    dom = result.stage("DOM1")
    assert not dom.fell_back and not dom.kept_base
    assert dom.autotriggered == ("ledger",)
    assert "`ledger`" in dom.tree[DEFAULT_SURFACE.autotrigger_file]


async def test_rejected_edit_gets_one_retry_with_the_rejection_text():
    bad = Proposal(edits={"scripts/check.py": "def broken(:\n"})
    good = Proposal(edits={"scripts/check.py": "print('ok')\n"})
    h = Harness(proposals={"INV1": [bad, good]})
    result = await h.run(k=1)
    inv = result.stage("INV1")
    assert inv.attempts == 2 and not inv.fell_back and inv.tree["scripts/check.py"] == "print('ok')\n"
    retry_prompt = [p for label, p in h.prompts if label == "INV1"][1]
    assert "YOUR PREVIOUS ATTEMPT WAS REJECTED" in retry_prompt and "scripts/check.py" in retry_prompt


async def test_twice_rejected_edit_falls_back_to_parent():
    bad = Proposal(edits={"scripts/check.py": "def broken(:\n"})
    h = Harness(proposals={"INV1": [bad, bad]})
    result = await h.run(k=1)
    inv = result.stage("INV1")
    assert inv.fell_back and inv.attempts == 2 and inv.tree == BASE and "scripts/check.py" in inv.error
    assert inv.rollout is not None and inv.rollout.ok, "the fallback tree is still probed"


async def test_proposer_exception_is_contained():
    h = Harness(proposals={"DOM1": [RuntimeError("model down"), RuntimeError("model down")]})
    result = await h.run(k=1)
    dom = result.stage("DOM1")
    assert dom.fell_back and "model down" in dom.error and dom.tree == BASE


async def test_crashed_rollout_is_evidence_not_a_crash():
    h = Harness(deliver=lambda label, tree: RuntimeError("sandbox died") if label == "INV1" else "ok")
    result = await h.run(k=1)
    r = result.stage("INV1").rollout
    assert r is not None and not r.ok and "sandbox died" in r.error
    assert "rollout FAILED" in dict(h.prompts)["agg1"]


async def test_grader_verdicts_cannot_enter_a_harness():
    leak = Proposal(edits={"AGENTS.md": '# Rules\n{"pass_rate": 0.9}\n'})
    result = await Harness(proposals={"INV1": [leak, leak]}).run(k=1)
    assert result.stage("INV1").fell_back and "grader verdict" in result.stage("INV1").error


# ── blind selection ──────────────────────────────────────────────────────


async def test_single_final_is_selected_without_a_selector_call():
    h = Harness()
    result = await h.run(k=1)
    assert result.chosen == "agg1" and result.selection["trivial"] and h.selector_prompts == []


async def test_selection_is_blind_and_reproducible():
    h = Harness()
    first = await h.run(k=3, task_id="t-1")
    second = await Harness().run(k=3, task_id="t-1")
    assert first.selection["id_map"] == second.selection["id_map"], "seeded: reruns line up"
    prompt = h.selector_prompts[0]
    for label in ("agg1", "agg2", "agg3", "base1"):
        assert label not in prompt, "candidate labels must not leak into the selector prompt"
    assert first.chosen == first.selection["id_map"]["A"] and not first.selection["fallback"]
    assert "COVERAGE" in first.selection["why"]


async def test_undelivered_finals_are_dropped_before_anonymising():
    h = Harness(deliver=lambda label, tree: "" if label == "agg2" else "an answer")
    result = await h.run(k=3)
    assert result.selection["dropped"] == ["agg2"]
    assert "agg2" not in result.selection["id_map"].values() and len(result.selection["id_map"]) == 2


async def test_selector_failure_falls_back_to_a_usable_composition():
    h = Harness()
    h.choice = "ZZ"
    result = await h.run(k=2)
    assert result.selection["fallback"] and result.chosen in ("agg1", "agg2")
    assert "selector failed" in result.selection["why"]


async def test_no_usable_composition_falls_back_to_a_delivered_base():
    h = Harness(deliver=lambda label, tree: "base answer" if label.startswith("base") else "")
    result = await h.run(k=2)
    assert result.chosen == "base1" and result.selection["fallback"]
    assert result.selection["dropped"] == ["agg1", "agg2"]


async def test_diversity_reports_mode_collapse():
    same = await Harness().run(k=2)
    assert same.diversity["n_distinct"] == 1 and same.diversity["mean_line_jaccard"] == 1.0
    varied = await Harness(proposals={"agg1": skill_edit("ledger"), "agg2": skill_edit("tax")}).run(k=2)
    assert varied.diversity["n_distinct"] == 2 and varied.diversity["mean_line_jaccard"] < 1.0


async def test_measurement_words_in_the_task_are_flagged():
    h = Harness()
    result = await propose_probe_compose(
        "Maximise the benchmark score.", BASE, rollout=h.rollout, propose=h.propose, k=1)
    assert result.warnings and "benchmark" in result.warnings[0]
    assert lint_task_prompt("Reconcile the ledger and score the leads") == ["score"]
    assert lint_task_prompt("Reconcile the underscore_names") == []


def test_invalid_arguments():
    h = Harness()
    for kw in ({"k": 0}, {"mode": "wide"}):
        with pytest.raises(ValueError):
            asyncio.run(h.run(**kw))


# ── proposer cache ───────────────────────────────────────────────────────


async def test_cache_reuses_identical_evidence_and_keeps_an_audit_trail(tmp_path):
    cache = ProposerCache(tmp_path / "cache")
    first = Harness(proposals={"DOM1": skill_edit("ledger")})
    a = await first.run(k=1, cache=cache)
    second = Harness(proposals={"DOM1": RuntimeError("must not be called")})
    b = await second.run(k=1, cache=cache)
    assert second.prompts == [], "identical evidence: every reply comes from the cache"
    assert b.stage("DOM1").tree == a.stage("DOM1").tree
    stored = [json.loads(p.read_text()) for p in (tmp_path / "cache").glob("*.json")]
    assert {s["label"] for s in stored} == {"INV1", "DOM1", "agg1"}
    assert all(s["prompt"] for s in stored)


def test_cache_key_changes_with_label_or_evidence():
    assert ProposerCache.key("INV1", "p") != ProposerCache.key("DOM1", "p")
    assert ProposerCache.key("INV1", "p") != ProposerCache.key("INV1", "p2")
    assert len(ProposerCache.key("INV1", "p")) == 16


# ── reflection and library ───────────────────────────────────────────────

GOOD_REPLY = """<edits>
component: skills/ledger/SKILL.md
candidates: DOM1, agg1
change: added a reconciliation checklist
observed: totals matched in both rollouts that had it
verdict: helped
</edits>
<experience>
cue: a deliverable aggregates figures from several files
lesson: recompute every total from source rows before writing it
evidence: agg1 caught the mismatch base1 shipped
confidence: medium
dimension: DOM
</experience>"""


async def test_reflection_prompt_covers_rejected_candidates_and_record_is_stored(tmp_path):
    bad = Proposal(edits={"scripts/check.py": "def broken(:\n"})
    h = Harness(proposals={"INV1": [bad, bad], "DOM1": skill_edit("ledger"), "agg1": skill_edit("ledger")})
    result: PPCResult = await h.run(k=1, task_id="ledger-q3")
    prompt = build_reflection_prompt("Reconcile the ledger.", result)
    assert "`INV1`" in prompt and "REJECTED" in prompt and "`agg1` (ADOPTED)" in prompt
    assert "confounded" in prompt and "NO_ENTRY" in prompt

    lib = ExperienceLibrary(tmp_path / "experience", domain="finance")
    assert lib.context_dir() is None, "an empty library is not shown to proposers"
    entry = record_experience(result, lib, GOOD_REPLY, run_id="r1")
    assert isinstance(entry, ExperienceEntry) and entry.dimension == "DOM"
    assert lib.context_dir() == lib.dir
    assert (lib.entry_dir("ledger-q3") / "harness" / "skills" / "ledger" / "SKILL.md").is_file()
    index = lib.index_path.read_text()
    assert "1 skill(s)" in index and "recompute every total" in index


async def test_confounded_or_empty_reflections_store_nothing(tmp_path):
    result = await Harness().run(k=1, task_id="t")
    lib = ExperienceLibrary(tmp_path / "experience")
    confounded = GOOD_REPLY.replace("verdict: helped", "verdict: confounded")
    assert record_experience(result, lib, confounded) is None
    assert record_experience(result, lib, "NO_ENTRY") is None
    no_cue = GOOD_REPLY.replace("cue: a deliverable aggregates figures from several files", "cue:")
    assert no_cue != GOOD_REPLY
    assert record_experience(result, lib, no_cue) is None
    assert lib.entries() == []


async def test_library_index_reaches_the_proposers(tmp_path):
    lib = ExperienceLibrary(tmp_path / "experience")
    record_experience(await Harness().run(k=1, task_id="earlier"), lib, GOOD_REPLY)
    h = Harness()
    await h.run(k=1, library=lib, task_id="later")
    assert "recompute every total" in dict(h.prompts)["INV1"]


def test_library_shows_only_earlier_tasks(tmp_path):
    lib = ExperienceLibrary(tmp_path / "experience")
    entry = ExperienceEntry(cue="c", lesson="l", evidence="e")
    for task in ("t1", "t2", "t3", "other"):
        lib.add(task, entry)
    removed = lib.retain_before("t2", ["t1", "t2", "t3"])
    assert sorted(removed) == ["t2", "t3"]
    assert lib.entries() == ["other", "t1"], "later tasks and the task itself are hidden; unrelated entries stay"
    assert lib.retain_before("unknown", ["t1"]) == []
    assert "t3" not in lib.index_path.read_text()


# ── surface validation refinements ───────────────────────────────────────

EVENTS = DEFAULT_SURFACE.hook_events


def hooks(obj) -> str:
    return json.dumps(obj)


def test_valid_hooks_pass():
    ok = hooks({"hooks": {"PreToolUse": [{"type": "command", "command": "python scripts/guard.py"}],
                          "Stop": [{"matcher": "*", "hooks": [{"type": "command", "command": "true"}]}]}})
    assert check_hooks(".clawagents/hooks.json", ok, EVENTS) == []


@pytest.mark.parametrize("source,needle", [
    ("{not json", "not valid JSON"),
    (hooks({"PreToolUse": []}), 'top-level "hooks"'),
    (hooks({"hooks": {"preToolUse": []}}), "unknown hook event"),
    (hooks({"hooks": {"Stop": {"type": "command"}}}), "must be a list"),
    (hooks({"hooks": {"Stop": [{"type": "prompt", "command": "x"}]}}), '"type": "command"'),
    (hooks({"hooks": {"Stop": [{"type": "command", "command": "  "}]}}), "non-empty command"),
    (hooks({"hooks": {"Stop": [{"matcher": "*", "hooks": []}]}}), "empty hooks list"),
])
def test_malformed_hooks_are_reported(source, needle):
    errs = check_hooks(".clawagents/hooks.json", source, EVENTS)
    assert errs and needle in " | ".join(errs)


def test_tree_validation_rejects_a_malformed_hooks_file():
    tree = dict(BASE, **{".clawagents/hooks.json": hooks({"hooks": {"OnWhatever": []}})})
    with pytest.raises(ValueError, match="unknown hook event"):
        validate_harness_tree(tree)
    tree[".clawagents/hooks.json"] = hooks({"hooks": {"Stop": [{"type": "command", "command": "true"}]}})
    validate_harness_tree(tree)


def test_off_surface_paths_are_rejected_only_when_asked():
    tree = dict(BASE, **{"notes/todo.md": "x", "scripts/ok.py": "print(1)\n"})
    validate_harness_tree(tree)
    with pytest.raises(ValueError, match="notes/todo.md is not part of the editable harness surface"):
        validate_harness_tree(tree, SurfaceSpec(reject_off_surface=True))


def test_repair_callback_gets_the_rejection_and_one_more_attempt():
    seen: list[str] = []

    def repair(error, rejected):
        seen.append(error)
        return dict(rejected, **{"scripts/check.py": "print('fixed')\n"})

    bad = dict(BASE, **{"scripts/check.py": "def broken(:\n"})
    tree, kept_base, fell_back, error, _ = accept_or_repair_tree(bad, BASE, repair=repair)
    assert not fell_back and not kept_base and error == "" and tree["scripts/check.py"] == "print('fixed')\n"
    assert len(seen) == 1 and "scripts/check.py" in seen[0]

    still_bad = accept_or_repair_tree(bad, BASE, repair=lambda e, t: t)
    assert still_bad[2] is True and still_bad[0] == BASE, "bounded: two failed validations fall back"
    no_repair = accept_or_repair_tree(bad, BASE)
    assert no_repair[2] is True and len(no_repair[3]) <= 400


def test_autotrigger_description_is_bounded():
    long_desc = "reconcile " + "the ledger " * 200
    tree = dict(BASE, **{"skills/ledger/SKILL.md": f"---\nname: ledger\ndescription: {long_desc}\n---\n"})
    assert autotrigger_skills(DEFAULT_SURFACE, tree) == ["ledger"]
    assert len(tree[DEFAULT_SURFACE.autotrigger_file]) < 700


# ── tree IO ──────────────────────────────────────────────────────────────


def test_tree_round_trips_through_a_workspace(tmp_path):
    tree = dict(BASE, **{"skills/ledger/SKILL.md": SKILL.format(n="ledger"),
                         "skills/ledger/scripts/sum.py": "print(1)\n",
                         "scripts/check.py": "print(2)\n",
                         ".clawagents/hooks.json": "{}"})
    assert sorted(write_tree(tmp_path, tree)) == sorted(tree)
    (tmp_path / "README.md").write_text("not part of the harness")
    (tmp_path / "skills" / "ledger" / "__pycache__").mkdir()
    (tmp_path / "skills" / "ledger" / "__pycache__" / "x.pyc").write_bytes(b"\x00\xff")
    assert read_tree(tmp_path) == tree


def test_write_tree_refuses_paths_outside_the_workspace(tmp_path):
    for rel in ("../escape.md", "/etc/hosts", "skills/../../escape.md"):
        with pytest.raises(ValueError, match="escapes the workspace"):
            write_tree(tmp_path / "ws", {rel: "x"})
    assert not (tmp_path / "escape.md").exists()
