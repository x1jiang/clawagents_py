"""Guard against the seatbelt ``shlex`` class of bug.

Any name that is imported (or assigned) inside a function is local for the
*entire* function. Using it before the first bind raises::

    cannot access free variable 'X' where it is not associated with a value
    in enclosing scope

That includes loads inside nested ``def`` / ``async def`` bodies that resolve
to an enclosing local (late ``import`` in the outer function).

This test fails the suite if such a pattern appears under ``src/clawagents``.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "clawagents"


def _nested_free_loads(fn_node: ast.AST) -> dict[str, list[int]]:
    """Name loads in *fn_node* that are not local to it (free / enclosing)."""
    local: set[str] = set()
    loads: dict[str, list[int]] = {}

    if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in (
            list(fn_node.args.posonlyargs)
            + list(fn_node.args.args)
            + list(fn_node.args.kwonlyargs)
        ):
            local.add(arg.arg)
        if fn_node.args.vararg:
            local.add(fn_node.args.vararg.arg)
        if fn_node.args.kwarg:
            local.add(fn_node.args.kwarg.arg)

    class Walk(ast.NodeVisitor):
        def visit_FunctionDef(self, n: ast.AST) -> None:
            if n is fn_node:
                self.generic_visit(n)
                return
            # Deeper nesting: free loads relative to *this* nested fn may still
            # resolve to the outer function under analysis — collect them too.
            for name, lines in _nested_free_loads(n).items():
                loads.setdefault(name, []).extend(lines)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, n: ast.Lambda) -> None:
            return

        def visit_Import(self, n: ast.Import) -> None:
            for a in n.names:
                local.add(a.asname or a.name.split(".")[0])

        def visit_ImportFrom(self, n: ast.ImportFrom) -> None:
            for a in n.names:
                if a.name != "*":
                    local.add(a.asname or a.name)

        def visit_Name(self, n: ast.Name) -> None:
            if isinstance(n.ctx, ast.Store):
                local.add(n.id)
            elif isinstance(n.ctx, ast.Load) and n.id not in local:
                loads.setdefault(n.id, []).append(n.lineno)

        def visit_arg(self, n: ast.arg) -> None:
            local.add(n.arg)

    Walk().visit(fn_node)
    return {k: v for k, v in loads.items() if k not in local}


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
                        return
                    # Nested def/async def: imports bind in the nested scope,
                    # but free loads may resolve to this function's locals.
                    for name, lines in _nested_free_loads(n).items():
                        loads.setdefault(name, []).extend(lines)

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
    def scan_all() -> list[str]:
        bugs: list[str] = []
        for path in sorted(ROOT.rglob("*.py")):
            bugs.extend(_find_use_before_bind(path))
        return bugs

    # CPython 3.11 can report ``AST constructor recursion depth mismatch``
    # when this deep repository scan inherits pytest-xdist's active call stack.
    # A dedicated worker keeps the parser stack shallow without weakening the
    # check or changing which files it examines.
    with ThreadPoolExecutor(max_workers=1) as pool:
        all_bugs = pool.submit(scan_all).result()
    if all_bugs:
        pytest.fail(
            "Local-import UnboundLocalError risks found "
            "(same class as seatbelt shlex bug):\n" + "\n".join(all_bugs)
        )


def test_guard_detects_nested_def_use_before_import(tmp_path: Path) -> None:
    """Meta-test: nested async def free-var use must be caught (not only genexps)."""
    src = tmp_path / "nested_shlex_bug.py"
    src.write_text(
        """
def outer(flag):
    async def inner():
        return shlex.quote("x")
    if flag:
        return inner
    import shlex
    return inner
""",
        encoding="utf-8",
    )
    bugs = _find_use_before_bind(src)
    assert bugs, (
        "guard must flag shlex used in nested async def before outer import"
    )
    assert any("shlex" in b and "outer" in b for b in bugs)


def test_guard_ignores_import_local_to_nested_only(tmp_path: Path) -> None:
    """Import inside nested def must not taint the outer function's locals."""
    src = tmp_path / "nested_local_import_ok.py"
    src.write_text(
        """
def outer():
    def inner():
        import shlex
        return shlex.quote("x")
    return inner
""",
        encoding="utf-8",
    )
    assert _find_use_before_bind(src) == []


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
