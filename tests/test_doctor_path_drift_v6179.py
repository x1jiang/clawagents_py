"""Doctor reports install identity and PATH interpreter drift (6.17.9)."""

from __future__ import annotations

import sys
from unittest import mock

from clawagents.__main__ import _probe_other_interpreters, cmd_doctor


def test_probe_other_interpreters_skips_same_realpath(tmp_path, monkeypatch):
    fake = tmp_path / "python3"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda _name: str(fake))
    # Same realpath as current → no warnings
    hits = _probe_other_interpreters(str(fake), "6.17.9")
    assert hits == []


def test_probe_other_interpreters_reports_different_version(tmp_path, monkeypatch):
    other = tmp_path / "other-python"
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(other) if name == "python3" else None)

    class Result:
        returncode = 0
        stdout = "6.11.2\n/tmp/clawagents/__init__.py\n"
        stderr = ""

    with mock.patch("subprocess.run", return_value=Result()):
        hits = _probe_other_interpreters(sys.executable, "6.17.9")
    assert len(hits) == 1
    assert "6.11.2" in hits[0]
    assert str(other) in hits[0]


def test_cmd_doctor_prints_install_identity(capsys):
    # May fail API-key checks in empty env; still must print version line.
    cmd_doctor()
    err = capsys.readouterr().err
    assert "Installed package" in err
    assert "Interpreter" in err
    assert "Package path" in err
