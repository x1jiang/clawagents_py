"""Grok-inspired skill strategy layered on ClawAgents progressive disclosure."""

from __future__ import annotations

from pathlib import Path

import pytest

from clawagents.config.features import reset as reset_features, set_overrides
from clawagents.skills.strategy import (
    apply_skill_substitutions,
    extract_when_to_use_from_description,
    filter_skills_for_catalog,
    format_skill_catalog_line,
    skill_paths_match,
)
from clawagents.tools.skills import SkillStore, create_skill_tools, parse_skill_file


@pytest.fixture(autouse=True)
def _features():
    reset_features()
    set_overrides(
        {
            "skill_when_to_use": True,
            "skill_path_gating": True,
            "skill_substitutions": True,
            "skill_hot_reload": True,
            "skill_auto_suggest": True,
        }
    )
    yield
    reset_features()


def test_extract_when_to_use_from_description():
    clean, when = extract_when_to_use_from_description(
        "Helps with PRs. Use when: reviewing large diffs."
    )
    assert "PR" in clean or "Helps" in clean
    assert "reviewing" in when.lower()


def test_parse_when_to_use_and_paths_frontmatter(tmp_path: Path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: py-edit\n"
        "description: Edit Python modules carefully.\n"
        "when-to-use: touching *.py files in src/\n"
        "paths:\n"
        "  - src/**/*.py\n"
        "  - tests/**/*.py\n"
        "---\n"
        "Body with $ARGUMENTS and ${SKILL_DIR}.\n",
        encoding="utf-8",
    )
    skill = parse_skill_file(skill_md.read_text(encoding="utf-8"), str(skill_md))
    assert skill.when_to_use.startswith("touching")
    assert "src/**/*.py" in skill.paths
    assert skill.description.startswith("Edit Python")


def test_user_invocable_false_maps_to_disable_model_invocation():
    skill = parse_skill_file(
        "---\nname: secret\ndescription: secret\nuser-invocable: false\n---\nHi\n",
        "/tmp/secret/SKILL.md",
    )
    assert skill.disable_model_invocation is True


def test_substitutions_arguments_and_skill_dir():
    out = apply_skill_substitutions(
        "Dir=${SKILL_DIR}; all=$ARGUMENTS; first=$0",
        skill_dir="/skills/demo",
        arguments="alpha beta",
        session_id="sess-1",
    )
    assert "/skills/demo" in out
    assert "alpha beta" in out
    assert "first=alpha" in out


def test_path_gating_hides_until_touched():
    skill = parse_skill_file(
        "---\nname: gated\ndescription: gated\npaths: [\"src/**/*.py\"]\n---\nBody\n",
        "/tmp/gated/SKILL.md",
    )
    assert filter_skills_for_catalog([skill], touched_paths=[]) == []
    assert filter_skills_for_catalog([skill], touched_paths=["src/app.py"])
    assert skill_paths_match(["*.py"], ["src/app.py"])


def test_catalog_line_includes_use_when():
    line = format_skill_catalog_line(
        "demo",
        "Does a thing",
        when_to_use="editing manifests",
        desc_cap=120,
    )
    assert "Use when:" in line
    assert "editing manifests" in line


def test_use_skill_applies_substitutions(tmp_path: Path):
    root = tmp_path / "skills" / "arg-demo"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: arg-demo\ndescription: demo\n---\n"
        "Run against ${SKILL_DIR} with $ARGUMENTS.\n",
        encoding="utf-8",
    )
    store = SkillStore()
    store.add_directory(tmp_path / "skills")
    store.reload()
    tools = {t.name: t for t in create_skill_tools(store)}

    async def _run():
        return await tools["use_skill"].execute(
            {"name": "arg-demo", "arguments": "file.py"},
            run_context=None,
        )

    import asyncio

    result = asyncio.run(_run())
    assert result.success
    assert "file.py" in result.output
    assert str(root) in result.output or "arg-demo" in result.output


def test_hot_reload_picks_up_new_skill(tmp_path: Path):
    skills = tmp_path / "skills"
    skills.mkdir()
    first = skills / "one"
    first.mkdir()
    (first / "SKILL.md").write_text(
        "---\nname: one\ndescription: first\n---\nA\n",
        encoding="utf-8",
    )
    store = SkillStore()
    store.add_directory(skills)
    store.reload()
    assert {s.name for s in store.list()} == {"one"}
    store.consume_discovery_announcement()  # seed

    second = skills / "two"
    second.mkdir()
    (second / "SKILL.md").write_text(
        "---\nname: two\ndescription: second\n---\nB\n",
        encoding="utf-8",
    )
    # Touch parent mtime so maybe_hot_reload notices
    skills.touch()
    assert store.maybe_hot_reload() is True
    assert {s.name for s in store.list()} == {"one", "two"}
    note = store.consume_discovery_announcement()
    assert "two" in note


def test_relevance_boosts_when_to_use():
    from clawagents.agent import _skill_relevance_score

    skill = parse_skill_file(
        "---\nname: patch-review\ndescription: Reviews patches.\n"
        "when-to-use: reviewing large pull request diffs\n---\nX\n",
        "/tmp/patch-review/SKILL.md",
    )
    score = _skill_relevance_score(skill, "please help reviewing large pull request diffs")
    assert score >= 70
