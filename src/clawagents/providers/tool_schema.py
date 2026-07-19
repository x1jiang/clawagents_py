"""Shared JSON-Schema fragment helpers for tool parameters.

Preserves nested ``array → object → properties`` shapes that MCP tools
declare (e.g. ``commands: [{label, command}]``). Flattening to bare
``items: {type: object}`` causes Gemini 400s and leaves OpenAI/luna without
shape guidance.
"""

from __future__ import annotations

from typing import Any

_PRIMITIVE_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object"})


def _collapse_type(ptype: Any) -> str:
    if isinstance(ptype, list):
        ptype = next((t for t in ptype if t != "null"), "string")
    if ptype not in _PRIMITIVE_TYPES:
        return "string"
    return str(ptype)


def normalize_json_schema_node(raw: Any) -> dict[str, Any]:
    """Normalize one JSON-Schema node, preserving nested properties/items."""
    if not isinstance(raw, dict):
        return {"type": "string"}
    ptype = _collapse_type(raw.get("type", "string"))
    out: dict[str, Any] = {"type": ptype}
    desc = raw.get("description")
    if isinstance(desc, str) and desc:
        out["description"] = desc
    enum_vals = raw.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        out["enum"] = enum_vals[:64]

    if ptype == "array":
        items = raw.get("items")
        if isinstance(items, dict):
            out["items"] = normalize_json_schema_node(items)
        else:
            out["items"] = {"type": "string"}
    elif ptype == "object":
        props_in = raw.get("properties")
        if isinstance(props_in, dict) and props_in:
            props_out: dict[str, Any] = {}
            for key, val in props_in.items():
                if isinstance(key, str) and isinstance(val, dict):
                    props_out[key] = normalize_json_schema_node(val)
            if props_out:
                out["properties"] = props_out
            req = raw.get("required")
            if isinstance(req, list):
                kept = [str(r) for r in req if str(r) in props_out]
                if kept:
                    out["required"] = kept
    return out


def emit_openai_schema_node(node: dict[str, Any]) -> dict[str, Any]:
    """Emit a Chat Completions / Responses JSON-Schema property node."""
    ptype = _collapse_type(node.get("type", "string"))
    out: dict[str, Any] = {"type": ptype}
    if node.get("description"):
        out["description"] = str(node["description"])
    if isinstance(node.get("enum"), list) and node["enum"]:
        out["enum"] = list(node["enum"])
    if ptype == "array":
        items = node.get("items") if isinstance(node.get("items"), dict) else {"type": "string"}
        out["items"] = emit_openai_schema_node(items)
    elif ptype == "object":
        props = node.get("properties")
        if isinstance(props, dict) and props:
            out["properties"] = {
                k: emit_openai_schema_node(v if isinstance(v, dict) else {"type": "string"})
                for k, v in props.items()
                if isinstance(k, str)
            }
            req = node.get("required")
            if isinstance(req, list):
                kept = [str(r) for r in req if str(r) in out["properties"]]
                if kept:
                    out["required"] = kept
        else:
            out["properties"] = out.get("properties") or {}
    return out


def emit_gemini_schema_node(node: dict[str, Any]) -> dict[str, Any]:
    """Emit a Gemini FunctionDeclaration schema node (UPPERCASE types)."""
    ptype = _collapse_type(node.get("type", "string")).upper()
    out: dict[str, Any] = {"type": ptype}
    if node.get("description"):
        out["description"] = str(node["description"])
    if isinstance(node.get("enum"), list) and node["enum"]:
        out["enum"] = list(node["enum"])
    if ptype == "ARRAY":
        items = node.get("items") if isinstance(node.get("items"), dict) else {"type": "string"}
        out["items"] = emit_gemini_schema_node(items)
    elif ptype == "OBJECT":
        props = node.get("properties")
        if isinstance(props, dict) and props:
            out["properties"] = {
                k: emit_gemini_schema_node(v if isinstance(v, dict) else {"type": "string"})
                for k, v in props.items()
                if isinstance(k, str)
            }
            req = node.get("required")
            if isinstance(req, list):
                kept = [str(r) for r in req if str(r) in out["properties"]]
                if kept:
                    out["required"] = kept
        else:
            # Bare OBJECT without properties is what Gemini rejects for items.
            out["properties"] = {}
    return out


def gemini_array_items_valid(node: dict[str, Any]) -> bool:
    """True when every ARRAY in *node* has a typed items schema."""
    ptype = str(node.get("type", "")).upper()
    if ptype == "ARRAY":
        items = node.get("items")
        if not isinstance(items, dict) or "type" not in items:
            return False
        return gemini_array_items_valid(items)
    if ptype == "OBJECT":
        props = node.get("properties") or {}
        if not isinstance(props, dict):
            return True
        return all(
            gemini_array_items_valid(v) for v in props.values() if isinstance(v, dict)
        )
    return True
