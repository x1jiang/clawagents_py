from types import SimpleNamespace

from clawagents.providers.llm import _serialize_gemini_parts


def _part(
    *,
    text=None,
    thought=False,
    thought_signature=None,
    function_name=None,
    function_args=None,
    function_id=None,
):
    function_call = None
    if function_name:
        function_call = SimpleNamespace(
            name=function_name,
            args=function_args or {},
            id=function_id,
        )
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
    # Gemini 3: signature only on the first parallel FC — do not invent on siblings.
    assert serialized[0]["thought_signature"] == "sig-abc"
    assert "thought_signature" not in serialized[1]


def test_does_not_invent_thought_signature_when_none_present():
    parts = [
        _part(function_name="tool_a", function_args={"x": 1}),
        _part(function_name="tool_b", function_args={"y": 2}),
    ]

    serialized = _serialize_gemini_parts(parts)

    assert serialized is not None
    assert "thought_signature" not in serialized[0]
    assert "thought_signature" not in serialized[1]


def test_serializes_function_call_id_and_bytes_signature():
    parts = [
        _part(
            function_name="ls",
            function_args={},
            function_id="call-99",
            thought_signature=b"\x00\x01sig",
        ),
    ]
    serialized = _serialize_gemini_parts(parts)
    assert serialized is not None
    assert serialized[0]["function_call"]["id"] == "call-99"
    assert serialized[0]["_thought_signature_b64"] is True
    assert isinstance(serialized[0]["thought_signature"], str)
