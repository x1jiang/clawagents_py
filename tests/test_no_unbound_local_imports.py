"""Guard against the seatbelt ``shlex`` class of bug.

Any name that is imported (or assigned) inside a function is local for the
*entire* function. Using it before the first bind raises::

    cannot access free variable 'X' where it is not associated with a value
    in enclosing scope

This test fails the suite if such a pattern appears under ``src/clawagents``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "clawagents"


def _find_use_before_bind(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]

    bugs: list[str] = []

    class FnVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._check(node)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def _check(self, node: ast.AST) -> None:
            binds: dict[str, int] = {}
            loads: dict[str, list[int]] = {}
            imported: set[str] = set()

            class BodyWalk(ast.NodeVisitor):
                def visit_FunctionDef(self, n: ast.AST) -> None:
                    if n is node:
                        self.generic_visit(n)

                visit_AsyncFunctionDef = visit_FunctionDef

                def visit_ClassDef(self, n: ast.ClassDef) -> None:
                    return

                def visit_Lambda(self, n: ast.Lambda) -> None:
                    return

                def visit_Import(self, n: ast.Import) -> None:
                    for a in n.names:
                        name = a.asname or a.name.split(".")[0]
                        binds[name] = min(binds.get(name, n.lineno), n.lineno)
                        imported.add(name)

                def visit_ImportFrom(self, n: ast.ImportFrom) -> None:
                    for a in n.names:
                        if a.name == "*":
                            continue
                        name = a.asname or a.name
                        binds[name] = min(binds.get(name, n.lineno), n.lineno)
                        imported.add(name)

                def visit_Name(self, n: ast.Name) -> None:
                    if isinstance(n.ctx, ast.Store):
                        binds[n.id] = min(binds.get(n.id, n.lineno), n.lineno)
                    elif isinstance(n.ctx, ast.Load):
                        loads.setdefault(n.id, []).append(n.lineno)

                def visit_arg(self, n: ast.arg) -> None:
                    binds[n.arg] = min(binds.get(n.arg, node.lineno), node.lineno)

            BodyWalk().visit(node)

            for name in imported:
                first_bind = binds.get(name)
                if first_bind is None:
                    continue
                early = [ln for ln in loads.get(name, []) if ln < first_bind]
                if early:
                    bugs.append(
                        f"{path}:{node.name}: '{name}' used at {early[0]} "
                        f"before bind at {first_bind}"
                    )

    FnVisitor().visit(tree)
    return bugs


def test_no_use_before_local_import_bind() -> None:
    all_bugs: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        all_bugs.extend(_find_use_before_bind(path))
    if all_bugs:
        pytest.fail(
            "Local-import UnboundLocalError risks found "
            "(same class as seatbelt shlex bug):\n" + "\n".join(all_bugs)
        )


def test_seatbelt_and_bwrap_wrap_command_share_path(tmp_path, monkeypatch) -> None:
    """wrap_command must work for both backends without unbound-name crashes."""
    from unittest.mock import patch

    from clawagents.sandbox.local import LocalBackend
    from clawagents.sandbox.profiles import OSSandboxProfile, ProfileBackend

    inner = LocalBackend(root=str(tmp_path))
    seatbelt = ProfileBackend(
        inner,
        OSSandboxProfile(
            name="workspace",
            backend="seatbelt",
            network=False,
            require_binary=False,
        ),
    )
    with patch(
        "clawagents.sandbox.profiles.shutil.which",
        return_value="/usr/bin/sandbox-exec",
    ):
        wrapped = seatbelt.wrap_command("echo hi", cwd=str(tmp_path))
    assert "sandbox-exec" in wrapped
    assert "echo hi" in wrapped

    # bwrap path also must not crash when binary missing (soft fallback)
    bwrap = ProfileBackend(
        LocalBackend(root=str(tmp_path)),
        OSSandboxProfile(
            name="workspace",
            backend="bwrap",
            network=False,
            require_binary=False,
        ),
    )
    with patch("clawagents.sandbox.profiles.shutil.which", return_value=None):
        assert bwrap.wrap_command("echo hi") == "echo hi"
    assert any("bwrap unavailable" in w for w in bwrap.profile_warnings)
