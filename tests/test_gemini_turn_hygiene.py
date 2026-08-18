"""Gemini conversation turn hygiene (function-call ordering)."""

from __future__ import annotations

from clawagents.providers.llm import _sanitize_gemini_contents


def test_coalesce_merges_parallel_tool_responses_only():
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {
            "role": "model",
            "parts": [
                {"function_call": {"name": "a", "args": {}, "id": "c1"}},
                {"function_call": {"name": "b", "args": {}, "id": "c2"}},
            ],
        },
        {"role": "user", "parts": [{"function_response": {"name": "a", "response": {"result": "1"}}}]},
        {"role": "user", "parts": [{"function_response": {"name": "b", "response": {"result": "2"}}}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user", "model", "user"]
    assert len(out[2]["parts"]) == 2
    assert all("function_response" in p for p in out[2]["parts"])
    assert out[2]["parts"][0]["function_response"]["id"] == "c1"
    assert out[2]["parts"][0]["function_response"]["call_id"] == "c1"
    assert out[2]["parts"][1]["function_response"]["id"] == "c2"
    assert out[2]["parts"][1]["function_response"]["call_id"] == "c2"


def test_fr_not_mixed_with_following_user_text():
    """Gemini rejects function_response+plain text in the same user turn."""
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"function_call": {"name": "ask_user", "args": {"question": "?"}}}]},
        {"role": "user", "parts": [{"text": "hi again"}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user", "model", "user", "model", "user"]
    assert all("function_response" in p for p in out[2]["parts"])
    assert out[4]["parts"] == [{"text": "hi again"}]


def test_sanitize_drops_leading_model():
    raw = [
        {"role": "model", "parts": [{"text": "orphan"}]},
        {"role": "user", "parts": [{"text": "hi"}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert out == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_orphan_fr_after_text_model_dropped():
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "thinking only"}]},
        {"role": "user", "parts": [{"function_response": {"name": "x", "response": {"result": "1"}}}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user", "model"]
    assert out[1]["parts"] == [{"text": "thinking only"}]


def test_ensure_pairs_inserts_synthetic_fr():
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"function_call": {"name": "ask_user", "args": {}, "id": "x1"}}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert out[-1]["role"] == "user"
    assert "function_response" in out[-1]["parts"][0]
    assert out[-1]["parts"][0]["function_response"]["id"] == "x1"
    assert out[-1]["parts"][0]["function_response"]["call_id"] == "x1"


def test_flatten_tool_history_drops_fc_fr_structure():
    from clawagents.providers.llm import GEMINI_ANSWER_NUDGE, _flatten_gemini_tool_history

    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"function_call": {"name": "ls", "args": {}}}]},
        {"role": "user", "parts": [{"function_response": {"name": "ls", "response": {"result": "ok"}}}]},
        {"role": "user", "parts": [{"text": "how many?"}]},
    ]
    flat = _flatten_gemini_tool_history(raw)
    assert all(
        not any("function_call" in p or "function_response" in p for p in t["parts"])
        for t in flat
    )
    assert flat[0]["role"] == "user"
    blob = "\n".join(t["parts"][0]["text"] for t in flat)
    assert "[used ls]" in blob
    assert "[called ls" not in blob
    assert "how many?" in blob
    assert GEMINI_ANSWER_NUDGE in blob


def test_flatten_omits_write_file_args():
    from clawagents.providers.llm import _flatten_gemini_tool_history

    script = "print('x')\n" * 200
    raw = [
        {"role": "user", "parts": [{"text": "confirm the cohort"}]},
        {
            "role": "model",
            "parts": [{
                "function_call": {
                    "name": "write_file",
                    "args": {"path": "traumatic_injury/scripts/validate_pain_scores.py", "content": script},
                }
            }],
        },
        {
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": "write_file",
                    "response": {"result": "wrote file"},
                }
            }],
        },
    ]
    flat = _flatten_gemini_tool_history(raw)
    blob = "\n".join(t["parts"][0]["text"] for t in flat)
    assert "validate_pain_scores.py" not in blob
    assert script.strip() not in blob
    assert "[used write_file]" in blob
    assert "[result write_file: wrote file]" in blob


def test_looks_like_gemini_command_dump():
    from clawagents.providers.llm import looks_like_gemini_command_dump

    assert looks_like_gemini_command_dump(
        "[called write_file({path: 'x.py', content: 'huge'})]"
    )
    assert looks_like_gemini_command_dump("[used execute]\n[result execute: ok]")
    assert not looks_like_gemini_command_dump(
        "[called execute({cmd: 'ls'})]\n\n"
        "Patients with first-encounter pain 4–7 are A, B, and C."
    )
    assert not looks_like_gemini_command_dump("Here are the matching patients.")


def test_function_response_includes_call_id():
    from clawagents.providers.llm import _gemini_function_response_body

    body = _gemini_function_response_body("ls", "ok", "call-1")
    assert body["id"] == "call-1"
    assert body["call_id"] == "call-1"
    assert body["name"] == "ls"


def test_upsert_stream_function_call_dedupes_chunks():
    from types import SimpleNamespace

    from clawagents.providers.llm import NativeToolCall, _upsert_gemini_stream_function_call

    calls: list[NativeToolCall] = []
    _upsert_gemini_stream_function_call(
        calls, SimpleNamespace(name="execute", args={"cmd": "l"}, id=None)
    )
    _upsert_gemini_stream_function_call(
        calls, SimpleNamespace(name="execute", args={"cmd": "ls"}, id="api-1")
    )
    assert len(calls) == 1
    assert calls[0].tool_call_id == "api-1"
    assert calls[0].args == {"cmd": "ls"}
