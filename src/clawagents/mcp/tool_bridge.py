"""Adapt an MCP tool descriptor into a clawagents :class:`Tool`.

Each MCP tool advertised by a server becomes a single :class:`MCPBridgedTool`
that conforms to the existing ``Tool`` protocol (name / description /
parameters / async ``execute``). The agent loop then sees and calls it just
like a built-in tool.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from clawagents.tools.registry import ToolResult

if TYPE_CHECKING:
    from clawagents.mcp.server import MCPServer, MCPToolDescriptor


_PRIMITIVE_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


def _normalize_input_schema(input_schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert an MCP-tool JSON-Schema ``inputSchema`` into clawagents' parameter dict.

    clawagents tools use a flat ``{name: {type, description, required}}``
    shape; MCP tools use full JSON Schema. We pull out just the top-level
    properties.
    """
    if not isinstance(input_schema, dict):
        return {}
    props = input_schema.get("properties") or {}
    if not isinstance(props, dict):
        return {}
    required = set(input_schema.get("required") or [])
    out: Dict[str, Dict[str, Any]] = {}
    for pname, raw in props.items():
        if not isinstance(raw, dict):
            continue
        ptype = raw.get("type")
        if isinstance(ptype, list):
            ptype = next((t for t in ptype if t != "null"), "string")
        if ptype not in _PRIMITIVE_TYPES:
            ptype = "string"
        description = raw.get("description", "") or ""
        # Surface enum constraints in the description — the flat parameter
        # shape has no enum field, and without the allowed values the model
        # guesses (e.g. "bash" for a language enum that wants "shell").
        enum_vals = raw.get("enum")
        if isinstance(enum_vals, list) and enum_vals:
            allowed = ", ".join(str(v) for v in enum_vals[:24])
            suffix = f" Allowed values: {allowed}."
            if suffix.strip() not in description:
                description = (description.rstrip(". ") + "." if description else "") + suffix
        out[pname] = {
            "type": ptype,
            "description": description,
            "required": pname in required,
        }
        # Preserve array item schemas — Gemini rejects ARRAY props without items.
        if ptype == "array":
            items = raw.get("items")
            if isinstance(items, dict):
                item_type = items.get("type", "string")
                if isinstance(item_type, list):
                    item_type = next((t for t in item_type if t != "null"), "string")
                if item_type not in _PRIMITIVE_TYPES:
                    item_type = "string"
                out[pname]["items"] = {"type": item_type}
            else:
                out[pname]["items"] = {"type": "string"}
    return out


def _stringify_call_result(result: Any) -> tuple[bool, str, Optional[str]]:
    """Reduce an ``mcp.types.CallToolResult`` to ``(success, output, error)``.

    The MCP SDK returns a structured result with a ``content`` list of typed
    blocks (text, image, resource, etc.) and an optional ``isError`` flag.
    We concatenate text blocks; non-text blocks are summarised by their type.
    """
    if result is None:
        return True, "", None
    is_error = bool(getattr(result, "isError", False))
    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        # Fallback: best-effort string repr of a non-text block.
        block_type = getattr(block, "type", type(block).__name__)
        parts.append(f"[{block_type} block]")
    output = "\n".join(parts)
    if is_error:
        return False, output, output or "MCP tool reported isError=True"
    return True, output, None


class MCPBridgedTool:
    """A clawagents-shaped :class:`Tool` backed by an MCP server.

    Each instance forwards ``execute()`` calls to ``server.invoke_tool()``.
    """

    def __init__(
        self,
        descriptor: "MCPToolDescriptor",
        server: "MCPServer",
        *,
        name_prefix: Optional[str] = None,
    ) -> None:
        self._descriptor = descriptor
        self._server = server
        if name_prefix:
            self.name = f"{name_prefix}.{descriptor.name}"
        else:
            self.name = descriptor.name
        self.description = descriptor.description or f"MCP tool '{descriptor.name}' from server '{server.name}'."
        self.parameters: Dict[str, Dict[str, Any]] = _normalize_input_schema(descriptor.input_schema)
        self.server_name = server.name
        self.original_tool_name = descriptor.name

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        try:
            raw = await self._server.invoke_tool(self.original_tool_name, args)
        except ImportError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except Exception as exc:
            # str(TimeoutError()) is empty — always include the type so the
            # model (and logs) see an actionable reason.
            detail = str(exc).strip() or type(exc).__name__
            return ToolResult(
                success=False,
                output="",
                error=f"MCP tool '{self.original_tool_name}' on server '{self.server_name}' failed: {detail}",
            )
        success, output, error = _stringify_call_result(raw)
        return ToolResult(success=success, output=output, error=error)


def mcp_tool_to_clawagents_tool(
    descriptor: "MCPToolDescriptor",
    server: "MCPServer",
    *,
    name_prefix: Optional[str] = None,
) -> MCPBridgedTool:
    """Public factory matching the function-style spec in the task brief."""
    return MCPBridgedTool(descriptor, server, name_prefix=name_prefix)
