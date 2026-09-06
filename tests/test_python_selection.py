"""Configured Python must survive shell execution and symlinked venvs."""
import json
import os
import shlex
import subprocess
import venv

import pytest

from clawagents.sandbox.local import LocalBackend
from clawagents.tools.exec import _child_env


@pytest.fixture
def selected_python(tmp_path, monkeypatch):
    prefix = tmp_path / "selected venv"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(prefix)
    python = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    monkeypatch.setenv("CLAWAGENTS_PYTHON", str(python))
    monkeypatch.setenv("VIRTUAL_ENV", "/stale/venv")
    return python, prefix


@pytest.mark.parametrize("env_factory", [_child_env, lambda: LocalBackend()._sanitized_env()])
def test_selected_venv_runs_in_child_environment(selected_python, env_factory):
    python, prefix = selected_python
    env = env_factory()
    assert env["PATH"].split(os.pathsep)[0] == str(python.parent)
    assert env["VIRTUAL_ENV"] == str(prefix)
    result = subprocess.run(
        ["python", "-c", "import sys; print(sys.prefix)"],
        env=env, text=True, capture_output=True, check=True,
    )
    assert result.stdout.strip() == str(prefix)


@pytest.mark.asyncio
async def test_local_execute_uses_configured_python(selected_python):
    python, prefix = selected_python
    code = "import sys, json; print(json.dumps([sys.executable, sys.prefix]))"
    result = await LocalBackend().exec("python -c " + shlex.quote(code))
    assert result.exit_code == 0
    executable, actual_prefix = json.loads(result.stdout)
    assert actual_prefix == str(prefix)
    assert os.path.dirname(executable) == str(python.parent)


@pytest.mark.parametrize("env_factory", [_child_env, lambda: LocalBackend()._sanitized_env()])
def test_missing_selection_fails_without_fallback(tmp_path, monkeypatch, env_factory):
    monkeypatch.setenv("CLAWAGENTS_PYTHON", str(tmp_path / "missing/python"))
    with pytest.raises(ValueError, match="Configured Python interpreter.*not found"):
        env_factory()


def test_without_selection_preserves_path(monkeypatch):
    monkeypatch.delenv("CLAWAGENTS_PYTHON", raising=False)
    assert _child_env()["PATH"] == os.environ["PATH"]


def test_pinning_does_not_expose_credentials(selected_python, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("DB_PASSWORD", "test-secret")
    assert "OPENAI_API_KEY" not in _child_env()
    assert "DB_PASSWORD" not in LocalBackend()._sanitized_env()
