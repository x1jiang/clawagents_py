"""Structured tool arguments must not become invalid Python-repr JSON files."""
import json

import pytest

from clawagents.sandbox.local import LocalBackend
from clawagents.tools.filesystem import WriteFileTool
from clawagents.tools.registry import ToolRegistry
from clawagents.tools.validate import validate_tool_args


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [{"name": "Ada", "enabled": True, "value": None}, [1, False, {"x": "é"}]])
async def test_structured_file_content_round_trips_as_json(tmp_path, content):
    registry = ToolRegistry()
    registry.register(WriteFileTool(LocalBackend(root=str(tmp_path))))
    result = await registry.execute_tool("write_file", {"path": "data.json", "content": content})
    assert result.success
    assert json.loads((tmp_path / "data.json").read_text()) == content


def test_string_content_is_preserved_exactly(tmp_path):
    tool = WriteFileTool(LocalBackend(root=str(tmp_path)))
    text = "{'this': 'is intentional Python source'}\n"
    result = validate_tool_args(tool, {"path": "data.py", "content": text})
    assert result.valid
    assert result.coerced["content"] == text


@pytest.mark.parametrize("content", [{"value": float("nan")}, {"value": object()}])
def test_non_json_values_return_validation_error(tmp_path, content):
    tool = WriteFileTool(LocalBackend(root=str(tmp_path)))
    result = validate_tool_args(tool, {"path": "data.json", "content": content})
    assert not result.valid
