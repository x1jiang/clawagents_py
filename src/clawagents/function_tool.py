"""``@function_tool`` decorator — derive a :class:`Tool` from a plain function.

Before this module, every tool had to be a class with hand-written
``parameters`` metadata. ``@function_tool`` lets you write::

    @function_tool
    async def search_web(query: str, limit: int = 5) -> str:
        '''Search the web for ``query`` and return the top ``limit`` hits.'''
        ...

and get a ready-to-register :class:`Tool` with a JSON-schema-style
``parameters`` dict derived from the function signature + docstring.

Accepted parameter types: ``str``, ``int``, ``float``, ``bool``, ``list``,
``dict``. ``Optional[...]`` is supported (becomes ``required=False``).
An optional ``run_context`` parameter is recognised and automatically
filled in by the loop — it never shows up in the LLM-visible schema.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, get_args, get_origin

from clawagents.run_context import RunContext
from clawagents.tools.registry import Tool, ToolResult


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

_PARAM_DOC_RE = re.compile(
    r"^\s*(?::param|Args?:|Parameters?:)\s*(.*?)(?=^\s*(?::return|:raises|Returns?:|Raises?:|$))",
    re.DOTALL | re.MULTILINE,
)


def _py_type_to_json(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is Union:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _py_type_to_json(non_none[0])
        return "string"
    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin is dict:
        return "object"
    return _TYPE_MAP.get(tp, "string")


def _is_optional(tp: Any) -> bool:
    if get_origin(tp) is Union:
        return type(None) in get_args(tp)
    return False


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """Extract ``param: description`` mappings from a docstring.

    Supports ``:param name: desc`` (Sphinx), ``name: desc`` under
    ``Args:`` / ``Parameters:`` (Google / NumPy). Best-effort — we do not
    require callers to use a specific docstring flavour.
    """
    if not docstring:
        return {}

    out: dict[str, str] = {}
    for line in docstring.splitlines():
        m = re.match(r"\s*:param\s+(\w+)\s*:\s*(.*)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()

    in_args = False
    indent: int | None = None
    current: str | None = None
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            indent = None
            continue
        if in_args:
            if not stripped:
                in_args = False
                current = None
                continue
            if stripped.endswith(":") and not stripped.startswith((" ", "\t")):
                in_args = False
                current = None
                continue
            line_indent = len(line) - len(line.lstrip())
            if indent is None:
                indent = line_indent
            m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if m and line_indent == indent:
                current = m.group(1)
                out.setdefault(current, m.group(2).strip())
            elif current and line_indent > (indent or 0):
                out[current] = (out.get(current, "") + " " + stripped).strip()
    return out


def _extract_short_description(docstring: str) -> str:
    if not docstring:
        return ""
    lines = [ln.strip() for ln in docstring.strip().splitlines()]
    for line in lines:
        if line:
            return line
    return ""


@dataclass
class FunctionTool:
    """Tool implementation backed by a Python function.

    Produced by :func:`function_tool`. Conforms to the :class:`Tool` protocol
    from :mod:`clawagents.tools.registry`. The underlying function may be
    sync or async, and may optionally accept a ``run_context`` parameter.
    """
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    fn: Callable[..., Any]
    is_async: bool
    accepts_run_context: bool
    context_param_name: str | None

    async def execute(
        self,
        args: dict[str, Any],
        run_context: RunContext[Any] | None = None,
    ) -> ToolResult:
        call_kwargs = dict(args)
        if self.accepts_run_context and self.context_param_name:
            call_kwargs[self.context_param_name] = run_context
        try:
            if self.is_async:
                result = await self.fn(**call_kwargs)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.fn(**call_kwargs)
                )
            if isinstance(result, ToolResult):
                return result
            if result is None:
                return ToolResult(success=True, output="")
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


def function_tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Turn a function into a :class:`FunctionTool` with derived schema.

    Can be used bare (``@function_tool``) or with arguments
    (``@function_tool(name="search", description="…")``).
    """
    def decorator(raw_fn: Callable[..., Any]) -> FunctionTool:
        sig = inspect.signature(raw_fn)
        docstring = inspect.getdoc(raw_fn) or ""
        param_docs = _parse_param_docs(docstring)

        # Resolve ``from __future__ import annotations`` style string
        # annotations into actual types. Falls back to raw annotations if
        # resolution fails (e.g. forward refs without the right globals).
        try:
            import typing as _typing
            resolved_hints = _typing.get_type_hints(raw_fn, include_extras=True)
        except Exception:
            resolved_hints = {}

        parameters: dict[str, dict[str, Any]] = {}
        context_param_name: str | None = None

        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            resolved_ann = resolved_hints.get(pname, p.annotation)
            if pname in {"run_context", "ctx", "context"} and (
                p.annotation is inspect.Parameter.empty
                or _annotation_is_run_context(resolved_ann)
                or _annotation_is_run_context(p.annotation)
            ):
                context_param_name = pname
                continue
            if p.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            annotation = (
                resolved_ann
                if resolved_ann is not inspect.Parameter.empty
                else str
            )
            json_type = _py_type_to_json(annotation)
            required = (
                p.default is inspect.Parameter.empty
                and not _is_optional(annotation)
            )
            entry: dict[str, Any] = {
                "type": json_type,
                "description": param_docs.get(pname, ""),
                "required": required,
            }
            if p.default is not inspect.Parameter.empty:
                entry["default"] = p.default
            parameters[pname] = entry

        final_name = name or raw_fn.__name__
        final_desc = description or _extract_short_description(docstring) or raw_fn.__name__
        is_async = inspect.iscoroutinefunction(raw_fn)

        tool = FunctionTool(
            name=final_name,
            description=final_desc,
            parameters=parameters,
            fn=raw_fn,
            is_async=is_async,
            accepts_run_context=context_param_name is not None,
            context_param_name=context_param_name,
        )
        return tool

    if fn is not None and callable(fn):
        return decorator(fn)
    return decorator


def _annotation_is_run_context(annotation: Any) -> bool:
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is RunContext:
        return True
    if origin is Union:
        return any(_annotation_is_run_context(a) for a in get_args(annotation))
    if isinstance(annotation, str):
        return "RunContext" in annotation
    return False


def _tool_signature_accepts_run_context(tool: Tool) -> tuple[bool, str | None]:
    """Detect whether an arbitrary tool's ``execute`` takes ``run_context``.

    Used by :class:`ToolRegistry.execute_tool` to decide whether to pass
    the context through. Works for :class:`FunctionTool`, plain class-based
    tools, and LangChain adapters alike.
    """
    if isinstance(tool, FunctionTool):
        return tool.accepts_run_context, tool.context_param_name
    execute = getattr(tool, "execute", None)
    if execute is None:
        return False, None
    try:
        sig = inspect.signature(execute)
    except (TypeError, ValueError):
        return False, None
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if pname in {"run_context", "ctx"}:
            return True, pname
        if _annotation_is_run_context(p.annotation):
            return True, pname
    return False, None
