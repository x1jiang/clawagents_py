"""CodeAct sandbox must keep side effects behind the tool/permission layer.

Raw filesystem, process, and arbitrary-code access must be blocked at the AST
gate and by the curated ``__builtins__`` — otherwise ``open(...)`` and
``__import__('os').system(...)`` would bypass Plan / read-only / auto-approve.
"""

from __future__ import annotations

from pathlib import Path

from clawagents.graph.codeact import run_code_action
from clawagents.tools.registry import ToolRegistry


def _run(code: str):
    return run_code_action(code, ToolRegistry(), run_async=lambda c: None)


def test_open_is_blocked(tmp_path: Path):
    marker = tmp_path / "escape.txt"
    r = _run(f"open({str(marker)!r}, 'w').write('x'); print('wrote')")
    assert r["error"], "open() must be rejected"
    assert not marker.exists(), "no file may be written outside the tool layer"


def test_dunder_import_is_blocked():
    r = _run("print(__import__('os').getcwd())")
    assert r["error"], "__import__ must be rejected"


def test_eval_exec_compile_blocked():
    for name in ("eval('1+1')", "exec('x=1')", "compile('1','<s>','eval')"):
        r = _run(f"print({name})")
        assert r["error"], f"{name} must be rejected"


def test_import_statement_blocked():
    r = _run("import os\nprint(os.getcwd())")
    assert r["error"], "import statements must be rejected"


def test_dunder_attribute_access_blocked():
    r = _run("print(().__class__.__bases__)")
    assert r["error"], "dunder attribute traversal must be rejected"


def test_getattr_forbidden_name_blocked():
    r = _run("print(getattr((), '__class__'))")
    assert r["error"], "getattr is a forbidden name (dunder-string escape)"


def test_legit_computation_still_runs():
    r = _run("total = sum(i * i for i in range(5)); print(total); done = True")
    assert not r["error"], r["observation"]
    assert r["observation"].strip() == "30"
    assert r["done"] is True


def test_tool_calls_go_through_before_tool():
    """tools.<name>() must pass through before_tool so permissions still apply."""
    calls: list[str] = []

    def deny(name, args):
        from clawagents import HookResult

        calls.append(name)
        return HookResult(allowed=False, reason="read-only")

    r = run_code_action(
        "print(tools.write_file(path='x.txt', content='y'))",
        ToolRegistry(),
        before_tool=deny,
        run_async=lambda c: None,
    )
    assert calls == ["write_file"], "before_tool must gate tool calls"
    assert "[blocked]" in r["observation"]
