"""Shared JSON-Schema fragment helpers for tool parameters.

Preserves nested ``array → object → properties`` shapes that MCP tools
declare (e.g. ``commands: [{label, command}]``). Also resolves ``$ref`` /
``$defs`` (pydantic nested models) and collapses ``anyOf``/``oneOf``
(pydantic ``Optional[X]``) so schemas don't silently become ``string``.
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


def _collect_defs(root: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(root, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        block = root.get(key)
        if isinstance(block, dict):
            for name, node in block.items():
                if isinstance(name, str) and isinstance(node, dict):
                    out[name] = node
    return out


def _resolve_ref(
    ref: str,
    defs: dict[str, Any],
    root: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve a local ``#/$defs/Name`` or ``#/definitions/Name`` ref."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    parts = [p for p in ref[2:].split("/") if p]
    if len(parts) >= 2 and parts[0] in ("$defs", "definitions"):
        node = defs.get(parts[1])
        return dict(node) if isinstance(node, dict) else None
    # Walk from root for rarer pointers (#/properties/foo/…)
    cur: Any = root
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return dict(cur) if isinstance(cur, dict) else None


def _pick_union_variant(variants: list[Any]) -> dict[str, Any] | None:
    """Pick the first non-null anyOf/oneOf branch (pydantic Optional[X])."""
    non_null: list[dict[str, Any]] = []
    for v in variants:
        if not isinstance(v, dict):
            continue
        if v.get("type") == "null":
            continue
        non_null.append(v)
    if not non_null:
        return None
    # Prefer object/array/$ref branches over bare primitives when mixed.
    for v in non_null:
        if (
            "$ref" in v
            or v.get("type") in ("object", "array")
            or "properties" in v
            or "items" in v
        ):
            return v
    return non_null[0]


def normalize_json_schema_node(
    raw: Any,
    *,
    defs: dict[str, Any] | None = None,
    root: dict[str, Any] | None = None,
    _depth: int = 0,
) -> dict[str, Any]:
    """Normalize one JSON-Schema node, preserving nested properties/items."""
    if _depth > 32:
        return {"type": "string"}
    if not isinstance(raw, dict):
        return {"type": "string"}

    defs = defs if defs is not None else _collect_defs(root)
    node = dict(raw)

    def _deref(n: dict[str, Any]) -> dict[str, Any]:
        ref = n.get("$ref")
        if not isinstance(ref, str):
            return n
        resolved = _resolve_ref(ref, defs, root)
        if resolved is None:
            return n
        merged = dict(resolved)
        for k, v in n.items():
            if k == "$ref":
                continue
            merged[k] = v
        return merged

    # Resolve $ref first (pydantic List[Model] → items: {$ref: ...}).
    node = _deref(node)

    # anyOf / oneOf — pydantic Optional[X] and unions. Re-deref after pick
    # because branches are often bare ``{$ref: ...}``.
    for union_key in ("anyOf", "oneOf"):
        variants = node.get(union_key)
        if isinstance(variants, list) and variants:
            picked = _pick_union_variant(variants)
            if picked is not None:
                merged = dict(picked)
                for k, v in node.items():
                    if k in ("anyOf", "oneOf", "$ref"):
                        continue
                    if k not in merged or k in ("description", "title", "default"):
                        merged[k] = v
                node = _deref(merged)
            break

    ptype = node.get("type")
    if ptype is None and ("properties" in node or "required" in node):
        ptype = "object"
    elif ptype is None and "items" in node:
        ptype = "array"
    ptype = _collapse_type(ptype if ptype is not None else "string")

    out: dict[str, Any] = {"type": ptype}
    desc = node.get("description")
    if isinstance(desc, str) and desc:
        out["description"] = desc
    enum_vals = node.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        out["enum"] = enum_vals[:64]

    if ptype == "array":
        items = node.get("items")
        if isinstance(items, dict):
            out["items"] = normalize_json_schema_node(
                items, defs=defs, root=root, _depth=_depth + 1
            )
        else:
            out["items"] = {"type": "string"}
    elif ptype == "object":
        props_in = node.get("properties")
        if isinstance(props_in, dict) and props_in:
            props_out: dict[str, Any] = {}
            for key, val in props_in.items():
                if isinstance(key, str) and isinstance(val, dict):
                    props_out[key] = normalize_json_schema_node(
                        val, defs=defs, root=root, _depth=_depth + 1
                    )
            if props_out:
                out["properties"] = props_out
            req = node.get("required")
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
