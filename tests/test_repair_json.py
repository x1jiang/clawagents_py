"""Tests for the _repair_json function in providers/llm.py."""

from clawagents.providers.llm import _repair_json


def test_valid_json():
    assert _repair_json('{"key": "value"}') == {"key": "value"}


def test_empty_string():
    assert _repair_json("") == {}


def test_truncated_object():
    result = _repair_json('{"key": "value", "nested": {"inner": "val')
    assert isinstance(result, dict)


def test_trailing_comma():
    result = _repair_json('{"key": "value",')
    assert isinstance(result, dict)


def test_truncated_at_colon():
    result = _repair_json('{"key":')
    assert isinstance(result, dict)


def test_array_input():
    result = _repair_json('[{"tool": "read_file"')
    assert isinstance(result, (dict, list))


def test_whitespace_only():
    assert _repair_json("   ") == {}
