"""Tests for companion CLI version floors and probes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clawagents.companions import (
    MIN_CONTEXT_MODE,
    MIN_RTK,
    format_version,
    parse_version,
    probe_companions,
    probe_context_mode,
    probe_rtk,
    version_at_least,
)


def test_parse_version():
    assert parse_version("1.0.169") == (1, 0, 169)
    assert parse_version("rtk 0.43.0") == (0, 43, 0)
    assert parse_version("v6.19.0-rc1") == (6, 19, 0)
    assert parse_version("nope") is None
    assert parse_version(None) is None


def test_version_at_least():
    assert version_at_least("1.0.169", MIN_CONTEXT_MODE) is True
    assert version_at_least("1.0.168", MIN_CONTEXT_MODE) is False
    assert version_at_least((0, 43, 0), MIN_RTK) is True
    assert version_at_least((0, 42, 9), MIN_RTK) is False
    assert version_at_least(None, MIN_RTK) is False


def test_format_version():
    assert format_version(MIN_CONTEXT_MODE) == "1.0.169"
    assert format_version(MIN_RTK) == "0.43.0"


def test_probe_context_mode_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("clawagents.companions.shutil.which", lambda _n: None)
    status = probe_context_mode()
    assert status.found is False
    assert status.ok_vs_floor is False
    assert "npm install" in status.hint


def test_probe_context_mode_from_package_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pkg_dir = tmp_path / "node_modules" / "context-mode"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "package.json").write_text(
        json.dumps({"name": "context-mode", "version": "1.0.169"}),
        encoding="utf-8",
    )
    bin_path = pkg_dir / "cli.bundle.mjs"
    bin_path.write_text("// stub\n", encoding="utf-8")

    monkeypatch.setattr(
        "clawagents.companions.shutil.which",
        lambda n: str(bin_path) if n == "context-mode" else None,
    )
    status = probe_context_mode()
    assert status.found is True
    assert status.version == "1.0.169"
    assert status.ok_vs_floor is True


def test_probe_context_mode_below_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pkg_dir = tmp_path / "cm"
    pkg_dir.mkdir()
    (pkg_dir / "package.json").write_text(
        json.dumps({"version": "1.0.100"}),
        encoding="utf-8",
    )
    bin_path = pkg_dir / "cli.mjs"
    bin_path.write_text("//\n", encoding="utf-8")
    monkeypatch.setattr(
        "clawagents.companions.shutil.which",
        lambda n: str(bin_path) if n == "context-mode" else None,
    )
    status = probe_context_mode()
    assert status.found is True
    assert status.ok_vs_floor is False
    assert "upgrade" in status.hint


def test_probe_rtk_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLAW_RTK_BIN", raising=False)
    monkeypatch.setattr("clawagents.companions.shutil.which", lambda _n: None)
    status = probe_rtk()
    assert status.found is False
    assert status.ok_vs_floor is False


def test_probe_rtk_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\necho 'rtk 0.43.0'\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("CLAW_RTK_BIN", str(fake))
    monkeypatch.setattr("clawagents.companions.shutil.which", lambda _n: str(fake))
    status = probe_rtk()
    assert status.found is True
    assert status.version == "0.43.0"
    assert status.ok_vs_floor is True


def test_probe_companions_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLAW_RTK_BIN", raising=False)
    monkeypatch.setattr("clawagents.companions.shutil.which", lambda _n: None)
    statuses = probe_companions()
    assert {s.name for s in statuses} == {"context-mode", "rtk"}
    assert all(not s.ok_vs_floor for s in statuses)
