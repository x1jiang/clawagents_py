"""$ref/$defs + anyOf/oneOf schema normalization (pydantic / FastMCP shapes)."""

from __future__ import annotations

import json

from clawagents.mcp.tool_bridge import _normalize_input_schema
from clawagents.providers.llm import (
    NativeToolSchema,
    _to_gemini_tools,
    _to_openai_tools,
)
from clawagents.providers.tool_schema import gemini_array_items_valid


def test_ref_defs_array_of_object_preserved():
    schema = {
        "type": "object",
        "$defs": {
            "Cmd": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["label", "command"],
            }
        },
        "properties": {
            "commands": {
                "type": "array",
                "items": {"$ref": "#/$defs/Cmd"},
            }
        },
        "required": ["commands"],
    }
    params = _normalize_input_schema(schema)
    items = params["commands"]["items"]
    assert items["type"] == "object"
    assert items["properties"]["command"]["type"] == "string"
    assert set(items["required"]) == {"label", "command"}

    gem = _to_gemini_tools(
        [NativeToolSchema("batch", "d", params)]
    )[0]["function_declarations"][0]["parameters"]
    assert gemini_array_items_valid(gem)
    g_items = gem["properties"]["commands"]["items"]
    assert g_items["type"] == "OBJECT"
    assert "properties" in g_items


def test_anyof_optional_integer_not_string():
    schema = {
        "type": "object",
        "properties": {
            "limit": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": "optional limit",
            },
            "path": {"type": "string"},
        },
        "required": ["path"],
    }
    params = _normalize_input_schema(schema)
    assert params["limit"]["type"] == "integer"
    assert params["limit"]["description"] == "optional limit"
    assert params["path"]["type"] == "string"


def test_oneof_optional_object_ref():
    schema = {
        "type": "object",
        "$defs": {
            "Opts": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
            }
        },
        "properties": {
            "opts": {
                "oneOf": [{"$ref": "#/$defs/Opts"}, {"type": "null"}],
            }
        },
    }
    params = _normalize_input_schema(schema)
    assert params["opts"]["type"] == "object"
    assert params["opts"]["properties"]["n"]["type"] == "integer"


def test_openai_emitter_keeps_resolved_ref():
    schema = {
        "type": "object",
        "definitions": {
            "Row": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            }
        },
        "properties": {
            "rows": {"type": "array", "items": {"$ref": "#/definitions/Row"}},
        },
        "required": ["rows"],
    }
    params = _normalize_input_schema(schema)
    oai = _to_openai_tools([NativeToolSchema("t", "d", params)])
    rows = oai[0]["function"]["parameters"]["properties"]["rows"]
    assert rows["items"]["properties"]["id"]["type"] == "string"
    # sanity: not the pre-fix collapse
    assert json.dumps(rows["items"]) != '{"type": "string"}'
