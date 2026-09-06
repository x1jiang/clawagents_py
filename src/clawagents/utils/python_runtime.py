"""Honor an explicitly configured interpreter in local child processes."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def pin_python_env(env: dict[str, str]) -> dict[str, str]:
    """Copy an environment and prefer its selected Python, without resolving symlinks."""
    selected = env.get("CLAWAGENTS_PYTHON", "").strip()
    if not selected:
        return dict(env)
    # shutil.which retains a venv entry point rather than dereferencing it.
    python = shutil.which(selected, path=env.get("PATH", os.defpath))
    if not python:
        raise ValueError(
            f'Configured Python interpreter "{selected}" not found or not executable. '
            "Update CLAWAGENTS_PYTHON (VS Code: clawagents.pythonPath in User or Remote settings); "
            "ClawAgents will not substitute another Python."
        )
    python = os.path.abspath(python)
    bin_dir = os.path.dirname(python)
    result = dict(env)
    result["CLAWAGENTS_PYTHON"] = python
    result["PATH"] = os.pathsep.join([
        bin_dir, *(part for part in env.get("PATH", "").split(os.pathsep) if part and part != bin_dir),
    ])
    prefix = Path(bin_dir).parent
    if (prefix / "pyvenv.cfg").is_file():
        result["VIRTUAL_ENV"] = str(prefix)
    else:
        result.pop("VIRTUAL_ENV", None)
    return result
