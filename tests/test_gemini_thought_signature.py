from types import SimpleNamespace

from clawagents.providers.llm import _serialize_gemini_parts


def _part(
    *,
    text=None,
    thought=False,
    thought_signature=None,
    function_name=None,
    function_args=None,
):
    function_call = None
    if function_name:
        function_call = SimpleNamespace(name=function_name, args=function_args or {})
    return SimpleNamespace(
        text=text,
        thought=thought,
        thought_signature=thought_signature,
        function_call=function_call,
    )


def test_propagates_thought_signature_to_parallel_function_calls():
    parts = [
        _part(function_name="get_university_detail", function_args={"id": "u1"}, thought_signature="sig-abc"),
        _part(function_name="get_program_detail", function_args={"id": "p2"}),
    ]

    serialized = _serialize_gemini_parts(parts)

    assert serialized is not None
    assert serialized[0]["function_call"]["name"] == "get_university_detail"
    assert serialized[1]["function_call"]["name"] == "get_program_detail"
    assert serialized[0]["thought_signature"] == "sig-abc"
    assert serialized[1]["thought_signature"] == "sig-abc"


def test_does_not_invent_thought_signature_when_none_present():
    parts = [
        _part(function_name="tool_a", function_args={"x": 1}),
        _part(function_name="tool_b", function_args={"y": 2}),
    ]

    serialized = _serialize_gemini_parts(parts)

    assert serialized is not None
    assert "thought_signature" not in serialized[0]
    assert "thought_signature" not in serialized[1]
