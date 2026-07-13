"""Skill loading/use mechanism regressions.

Covers precedence, requires-block scoping, OS aliases, frontmatter edge
cases, size caps, dir dedup, and use_skill resource disclosure — patterns
aligned with openclaw / Claude Code / deepagents skill systems.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

import clawagents.tools.skills as skills_mod
from clawagents.tools.skills import (
    SkillStore,
    create_skill_tools,
    is_skill_eligible,
    parse_skill_file,
    skill_ineligibility_reason,
)


def _write_skill(root, name, body="Do the thing.", frontmatter=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = frontmatter if frontmatter is not None else f"name: {name}\ndescription: {name} skill"
    (d / "SKILL.md").write_text(f"---\n{fm}\n---\n\n{body}\n", encoding="utf-8")
    return d


def _load(store: SkillStore) -> None:
    asyncio.run(store.load_all())


# ── Precedence ──────────────────────────────────────────────────────────────

def test_later_directory_overrides_earlier_on_name_collision(tmp_path):
    low = tmp_path / "bundled"
    high = tmp_path / "workspace"
    _write_skill(low, "caveman", body="bundled body")
    _write_skill(high, "caveman", body="workspace body")

    store = SkillStore()
    store.add_directory(low)
    store.add_directory(high)
    _load(store)

    assert store.get("caveman").content == "workspace body"


def test_agent_orders_bundled_dir_first():
    """create_claw_agent must put the bundled dir lowest-precedence (first)."""
    import inspect

    from clawagents import agent as agent_mod

    src = inspect.getsource(agent_mod)
    assert "[_bundled] + base_skill_dirs" in src


def test_add_directory_dedups_repeated_paths(tmp_path):
    _write_skill(tmp_path / "skills", "demo")
    store = SkillStore()
    store.add_directory(tmp_path / "skills")
    store.add_directory(tmp_path / "skills")
    store.add_directory(str(tmp_path / "skills"))
    assert len(store.skill_dirs) == 1


# ── requires parsing scope ─────────────────────────────────────────────────

def test_metadata_block_keys_do_not_gate_eligibility():
    """Indented keys of unrelated blocks must not be read as requirements."""
    content = """---
name: demo
description: A demo
metadata:
  env: production
  os: solaris
  bins: nonexistent-binary-xyz
---

Body.
"""
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    assert skill.requires is None
    assert is_skill_eligible(skill)


def test_requires_block_scoped_parsing():
    content = """---
name: demo
description: A demo
requires:
  bins: [definitely-not-a-real-binary-xyz]
---

Body.
"""
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    assert skill.requires is not None
    assert skill.requires.bins == ["definitely-not-a-real-binary-xyz"]
    assert "missing binary" in (skill_ineligibility_reason(skill) or "")


def test_requires_env_block_list():
    content = """---
name: demo
description: A demo
requires:
  env:
    - CLAW_TEST_DEFINITELY_UNSET_VAR
---

Body.
"""
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    assert skill.requires.env == ["CLAW_TEST_DEFINITELY_UNSET_VAR"]
    assert "missing env var" in (skill_ineligibility_reason(skill) or "")


def test_openclaw_json_metadata_requires():
    content = """---
name: demo
description: A demo
metadata: {"openclaw": {"requires": {"bins": ["definitely-not-a-real-binary-xyz"]}}}
---

Body.
"""
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    assert skill.requires is not None
    assert skill.requires.bins == ["definitely-not-a-real-binary-xyz"]


def test_dotted_requires_keys_still_work():
    content = """---
name: demo
description: A demo
requires.env: SOME_UNSET_VAR_FOR_TEST
---

Body.
"""
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    assert skill.requires.env == ["SOME_UNSET_VAR_FOR_TEST"]


# ── OS aliases ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "alias", ["macos", "darwin", "mac", "osx", "darwin, linux", "any"]
)
def test_os_alias_matches_darwin(monkeypatch, alias):
    monkeypatch.setattr(sys, "platform", "darwin")
    content = f"---\nname: demo\ndescription: d\nrequires.os: {alias}\n---\n\nBody.\n"
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    assert is_skill_eligible(skill), alias


def test_os_mismatch_reports_reason(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    content = "---\nname: demo\ndescription: d\nrequires.os: windows\n---\n\nBody.\n"
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    reason = skill_ineligibility_reason(skill)
    assert reason and "requires os" in reason


# ── Frontmatter edge cases ─────────────────────────────────────────────────

def test_frontmatter_closing_at_eof_still_parses():
    content = "---\nname: eof-skill\ndescription: ends at delimiter\n---"
    skill = parse_skill_file(content, "/tmp/eof-skill/SKILL.md")
    assert skill.name == "eof-skill"
    assert skill.description == "ends at delimiter"
    assert skill.content == ""


def test_dir_skill_defaults_name_to_directory():
    content = "No frontmatter here, just instructions.\n"
    skill = parse_skill_file(content, "/skills/pdf-tools/SKILL.md")
    assert skill.name == "pdf-tools"


def test_description_falls_back_to_first_body_line():
    content = "---\nname: bare\n---\n\n# Heading\n\nUse this to convert PDFs.\n"
    skill = parse_skill_file(content, "/tmp/bare/SKILL.md")
    assert skill.description == "Use this to convert PDFs."


def test_spec_violations_warn_but_load():
    content = "---\nname: Bad_Name_Here\ndescription: d\n---\n\nBody.\n"
    skill = parse_skill_file(content, "/tmp/other-dir/SKILL.md")
    assert skill.name == "Bad_Name_Here"  # lenient: still loads
    assert any("not spec-conformant" in w for w in skill.warnings)
    assert any("does not match its directory" in w for w in skill.warnings)


# ── Store behaviors ────────────────────────────────────────────────────────

def test_oversized_skill_file_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_mod, "MAX_SKILL_FILE_BYTES", 64)
    _write_skill(tmp_path / "skills", "huge", body="x" * 4096)
    store = SkillStore()
    store.add_directory(tmp_path / "skills")
    _load(store)
    assert store.get("huge") is None
    assert any("exceeds" in w for w in store.warnings)


def test_readme_not_loaded_as_skill(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    (root / "README.md").write_text("# About these skills\n", encoding="utf-8")
    _write_skill(root, "real-skill")
    store = SkillStore()
    store.add_directory(root)
    _load(store)
    assert [s.name for s in store.list()] == ["real-skill"]


def test_ineligible_skill_tracked_with_reason(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "needs-bin",
        frontmatter=(
            "name: needs-bin\ndescription: d\n"
            "requires:\n  bins: [definitely-not-a-real-binary-xyz]"
        ),
    )
    store = SkillStore()
    store.add_directory(tmp_path / "skills")
    _load(store)
    assert store.get("needs-bin") is None
    assert "missing binary" in store.ineligible.get("needs-bin", "")


# ── Tools ──────────────────────────────────────────────────────────────────

def test_list_skills_reports_unavailable_with_reason(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "ok-skill")
    _write_skill(
        root,
        "gated",
        frontmatter=(
            "name: gated\ndescription: d\n"
            "requires:\n  bins: [definitely-not-a-real-binary-xyz]"
        ),
    )
    store = SkillStore()
    store.add_directory(root)
    _load(store)

    list_tool = [t for t in create_skill_tools(store) if t.name == "list_skills"][0]
    result = asyncio.run(list_tool.execute({}))
    assert result.success
    assert "ok-skill" in result.output
    assert "Unavailable (requirements not met)" in result.output
    assert "missing binary" in result.output


def test_use_skill_includes_base_dir_and_resources(tmp_path):
    root = tmp_path / "skills"
    d = _write_skill(root, "with-scripts", body="Run scripts/run.py to start.")
    (d / "scripts").mkdir()
    (d / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    (d / "references").mkdir()
    (d / "references" / "guide.md").write_text("# Guide\n", encoding="utf-8")

    store = SkillStore()
    store.add_directory(root)
    _load(store)

    use_tool = [t for t in create_skill_tools(store) if t.name == "use_skill"][0]
    result = asyncio.run(use_tool.execute({"name": "with-scripts"}))
    assert result.success
    assert f"Base directory for this skill: {d}" in result.output
    assert "scripts/run.py" in result.output
    assert "references/guide.md" in result.output


def test_use_skill_flat_md_has_base_dir_but_no_sibling_resources(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    (root / "flat.md").write_text(
        "---\nname: flat\ndescription: d\n---\n\nBody.\n", encoding="utf-8"
    )
    (root / "other.md").write_text(
        "---\nname: other\ndescription: d\n---\n\nBody.\n", encoding="utf-8"
    )
    store = SkillStore()
    store.add_directory(root)
    _load(store)

    use_tool = [t for t in create_skill_tools(store) if t.name == "use_skill"][0]
    result = asyncio.run(use_tool.execute({"name": "flat"}))
    assert result.success
    assert "Base directory for this skill:" in result.output
    # Sibling skills must not be presented as bundled resources.
    assert "Bundled resources" not in result.output


def test_use_skill_reports_ineligible_reason_on_miss(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "gated",
        frontmatter=(
            "name: gated\ndescription: d\n"
            "requires:\n  env:\n    - CLAW_TEST_DEFINITELY_UNSET_VAR"
        ),
    )
    store = SkillStore()
    store.add_directory(tmp_path / "skills")
    _load(store)

    use_tool = [t for t in create_skill_tools(store) if t.name == "use_skill"][0]
    result = asyncio.run(use_tool.execute({"name": "gated"}))
    assert not result.success
    assert "unavailable" in result.error
    assert "missing env var" in result.error
