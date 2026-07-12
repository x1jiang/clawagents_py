"""CodeAct loop — model writes Python actions that call tools via an injected API."""

from __future__ import annotations

import ast
import re
import traceback
from typing import Any, Callable, Optional

from clawagents.tools.registry import ToolRegistry, ToolResult

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_CODE_TAG_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)

CODEACT_SYSTEM_ADDENDUM = """
## CodeAct mode
You solve tasks by emitting ONE Python action per turn inside a ```python fenced block.
Available API:
  - tools.<tool_name>(**kwargs)  → calls a registered tool and returns its output string
  - print(...)                   → included in the observation
Do not invent tools. Prefer small steps. When finished, print a final answer and set:
  done = True
""".strip()


def extract_code_action(text: str) -> str | None:
    if not text:
        return None
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _CODE_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    # Bare assignment / tools. call heuristic
    stripped = text.strip()
    if "tools." in stripped or "done" in stripped:
        try:
            ast.parse(stripped)
            return stripped
        except SyntaxError:
            return None
    return None


class _ToolsProxy:
    """Sync facade over ToolRegistry for CodeAct sandboxes."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        before_tool: Any = None,
        run_context: Any = None,
        run_async: Callable[[Any], Any],
    ) -> None:
        self._registry = registry
        self._before_tool = before_tool
        self._run_context = run_context
        self._run_async = run_async
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Callable[..., str]:
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(**kwargs: Any) -> str:
            args = dict(kwargs)
            if self._before_tool is not None:
                try:
                    result = self._before_tool(name, args)
                    if isinstance(result, bool) and not result:
                        return f"[blocked] denied"
                    if result is not None and hasattr(result, "allowed"):
                        if not result.allowed:
                            return f"[blocked] {getattr(result, 'reason', 'denied')}"
                        if getattr(result, "updated_args", None) is not None:
                            args = result.updated_args
                except Exception as exc:
                    return f"[before_tool error] {exc}"
            tool_result: ToolResult = self._run_async(
                self._registry.execute_tool(name, args, run_context=self._run_context)
            )
            self.calls.append({"tool": name, "args": args, "success": tool_result.success})
            if not tool_result.success:
                return f"[error] {tool_result.error or tool_result.output}"
            out = tool_result.output
            return out if isinstance(out, str) else str(out)

        return _call


def run_code_action(
    code: str,
    registry: ToolRegistry,
    *,
    before_tool: Any = None,
    run_context: Any = None,
    run_async: Callable[[Any], Any],
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Execute a CodeAct Python snippet; return observation dict."""
    import io
    import contextlib

    tools = _ToolsProxy(
        registry,
        before_tool=before_tool,
        run_context=run_context,
        run_async=run_async,
    )
    stdout = io.StringIO()
    glb: dict[str, Any] = {"tools": tools, "done": False}
    loc: dict[str, Any] = {}
    err: str | None = None
    try:
        # Safety: reject imports and dunder access via AST gate
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("imports are not allowed in CodeAct actions")
            if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
                if node.attr.startswith("__"):
                    raise ValueError("dunder attribute access is not allowed")
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, "<codeact>", "exec"), glb, loc)  # noqa: S102
    except Exception:
        err = traceback.format_exc(limit=8)

    done = bool(loc.get("done", glb.get("done", False)))
    printed = stdout.getvalue()
    observation = printed.strip()
    if err:
        observation = (observation + "\n" if observation else "") + f"[exception]\n{err}"
    if not observation:
        observation = "(no output)"
    return {
        "observation": observation[:20_000],
        "done": done and err is None,
        "tool_calls": tools.calls,
        "error": err,
    }
