"""Marketplace installs are pinnable and verifiable.

Third-party skill content is injected into the model prompt, so an unpinned
``git clone --depth 1`` means whatever upstream happens to be at HEAD becomes
instruction text. These cover the pin (``ref``), the verification
(``expect_sha``), and the recorded provenance.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from clawagents import marketplace
from clawagents.marketplace import install_from_source, list_installed

SKILL_TEMPLATE = "---\nname: demo-skill\ndescription: {desc}\n---\nbody {tag}\n"


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


@pytest.fixture()
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A two-commit repo plus both SHAs; marketplace feature forced on."""
    monkeypatch.setenv("CLAW_FEATURE_MARKETPLACE", "1")
    from clawagents.config import features

    features.reset()
    # A local repo path is a valid git source for our purposes.
    monkeypatch.setattr(marketplace, "_is_git_url", lambda _s: True)

    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(["init", "--quiet", "."], repo)
    (repo / "SKILL.md").write_text(SKILL_TEMPLATE.format(desc="v1", tag="v1"))
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "v1"], repo)
    v1 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    (repo / "SKILL.md").write_text(SKILL_TEMPLATE.format(desc="v2", tag="v2"))
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "v2"], repo)
    v2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    yield str(repo), v1, v2
    features.reset()


def _installed_body(workspace: Path) -> str:
    return (
        workspace / ".clawagents" / "skills" / "demo-skill" / "SKILL.md"
    ).read_text()


def test_ref_pins_checkout_to_that_commit(tmp_path, upstream):
    source, v1, v2 = upstream
    ws = tmp_path / "ws"
    result = install_from_source(source, workspace=ws, ref=v1)

    assert result.ok, result.error
    assert result.commit == v1
    # The pin must win over upstream HEAD (v2).
    assert "body v1" in _installed_body(ws)


def test_expect_sha_mismatch_refuses_and_installs_nothing(tmp_path, upstream):
    source, v1, v2 = upstream
    ws = tmp_path / "ws"
    result = install_from_source(source, workspace=ws, ref=v1, expect_sha=v2)

    assert not result.ok
    assert "commit mismatch" in (result.error or "")
    skills = ws / ".clawagents" / "skills"
    assert not skills.exists() or not any(skills.iterdir())


def test_expect_sha_accepts_matching_commit_and_abbreviation(tmp_path, upstream):
    source, v1, _v2 = upstream
    full = install_from_source(source, workspace=tmp_path / "a", ref=v1, expect_sha=v1)
    assert full.ok, full.error

    short = install_from_source(
        source, workspace=tmp_path / "b", ref=v1, expect_sha=v1[:12]
    )
    assert short.ok, short.error
    assert short.commit == v1


def test_unpinned_install_still_records_resolved_commit(tmp_path, upstream):
    source, _v1, v2 = upstream
    ws = tmp_path / "ws"
    result = install_from_source(source, workspace=ws)

    assert result.ok, result.error
    assert result.commit == v2
    assert list_installed(ws)[0]["commit"] == v2


def test_malformed_expect_sha_is_rejected(tmp_path, upstream):
    source, _v1, _v2 = upstream
    result = install_from_source(source, workspace=tmp_path / "ws", expect_sha="nope")
    assert not result.ok
    assert "not a commit SHA" in (result.error or "")


def test_pinning_is_rejected_for_non_git_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_MARKETPLACE", "1")
    from clawagents.config import features

    features.reset()
    monkeypatch.setattr(marketplace, "_is_git_url", lambda _s: False)
    local = tmp_path / "local"
    local.mkdir()
    (local / "SKILL.md").write_text(SKILL_TEMPLATE.format(desc="d", tag="local"))

    result = install_from_source(str(local), workspace=tmp_path / "ws", ref="abc1234")
    assert not result.ok
    assert "only applies to git sources" in (result.error or "")
    features.reset()
