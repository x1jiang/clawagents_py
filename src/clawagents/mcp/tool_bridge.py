"""Adapt an MCP tool descriptor into a clawagents :class:`Tool`.

Each MCP tool advertised by a server becomes a single :class:`MCPBridgedTool`
that conforms to the existing ``Tool`` protocol (name / description /
parameters / async ``execute``). The agent loop then sees and calls it just
like a built-in tool.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from clawagents.providers.tool_schema import normalize_json_schema_node
from clawagents.tools.registry import ToolResult

if TYPE_CHECKING:
    from clawagents.mcp.server import MCPServer, MCPToolDescriptor


_BINARY_FILE_SUFFIXES = (
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".webp",
    ".xlsx",
    ".zip",
)


def _normalize_input_schema(input_schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert an MCP-tool JSON-Schema ``inputSchema`` into clawagents' parameter dict.

    Top-level params keep clawagents' ``required: bool``. Nested
    ``items`` / ``properties`` keep full JSON-Schema shape so array-of-object
    tools (e.g. ``commands: [{label, command}]``) survive into Gemini/OpenAI.
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
        # Pass root so $ref/$defs (pydantic nested models) resolve.
        node = normalize_json_schema_node(raw, root=input_schema)
        description = str(node.get("description") or "")
        # Surface enum constraints in the description when present.
        enum_vals = node.get("enum")
        if isinstance(enum_vals, list) and enum_vals:
            allowed = ", ".join(str(v) for v in enum_vals[:24])
            suffix = f" Allowed values: {allowed}."
            if suffix.strip() not in description:
                description = (description.rstrip(". ") + "." if description else "") + suffix
            node["description"] = description
        elif description:
            node["description"] = description
        node["required"] = pname in required
        out[pname] = node
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
        return (
            False,
            output,
            "MCP tool reported isError=True; see output for details",
        )
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
        self.tool_group = "mcp"
        normalized_server = str(server.name).strip().lower().replace("_", "-")
        self.context_protection = (
            normalized_server == "context-mode"
            or descriptor.name.startswith("ctx_")
        )
        if self.context_protection and descriptor.name == "ctx_execute_file":
            self.description += (
                " Text files only: do not pass binary PDF, DOCX, images, or ZIP files. "
                "For binary files, use ctx_execute and open the path inside the program, "
                "or use the dedicated document/PDF tooling."
            )

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        if self.context_protection and self.original_tool_name == "ctx_execute_file":
            path = args.get("path")
            if isinstance(path, str) and path.lower().endswith(_BINARY_FILE_SUFFIXES):
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "ctx_execute_file accepts text files only. Use ctx_execute to open "
                        "this binary file inside the program, or use the dedicated "
                        "document/PDF tooling."
                    ),
                )
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
