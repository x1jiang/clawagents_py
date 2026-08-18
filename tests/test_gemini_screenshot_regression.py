"""Replay the Gemini 3.7 screenshot failures: command-dump answer and empty Done."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from clawagents.graph.completion_handler import CompletionHandler
from clawagents.graph.tool_turn import ToolTurnExecutor
from clawagents.providers.llm import (
    GEMINI_ANSWER_NUDGE,
    GEMINI_SUMMARIZE_MARKER,
    GeminiProvider,
    LLMMessage,
    LLMResponse,
    NativeToolCall,
    _absorb_gemini_stream_parts,
    _flatten_gemini_tool_history,
    _is_gemini_history_400,
    looks_like_gemini_command_dump,
)
from clawagents.tools.registry import ParsedToolCall

# From the chat screenshot: the entire visible reply was this shape.
_SCREENSHOT_WRITE_FILE = (
    "[called write_file({path: 'traumatic_injury/scripts/validate_pain_scores.py', "
    "content: 'import pandas as pd\\n# validate first-encounter pain 4-7\\n'}]"
)
_SCREENSHOT_RETRIEVE = "[called retrieve_tool_result({'id': 'bg-1'})]"
_SCRIPT = "import pandas as pd\n" + ("print('pain')\n" * 80)
_VALIDATE_PATH = "traumatic_injury/scripts/validate_pain_scores.py"


class _Events:
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def emit(self, kind: str, data=None) -> None:
        self.kinds.append(kind)

    def typed(self, kind: str, data=None) -> None:
        self.kinds.append(kind)


def _handler() -> CompletionHandler:
    return CompletionHandler(
        registry=None,
        run_context=SimpleNamespace(_metadata={}),
        events=_Events(),
        recorder=None,
        llm=None,
        before_tool=None,
        action_mode="tools",
        looks_like_truncated_json=lambda _text: False,
        sanitize_assistant_text=lambda text: text,
        goal_llm_complete=lambda *_a, **_k: (lambda _s: _s),
    )


def _complete(messages, content):
    return asyncio.run(
        _handler().handle(
            state=SimpleNamespace(result=None, status="running"),
            messages=messages,
            response=LLMResponse(
                content=content, model="gemini-3.7-flash", tokens_used=8
            ),
            thinking=None,
            use_native_tools=True,
            consult_advisor=lambda *_a, **_k: None,
            should_final_check=False,
        )
    )


def test_screenshot_write_file_dump_is_not_an_answer():
    """Screenshot 2: the whole reply was [called write_file({huge script})]."""
    assert looks_like_gemini_command_dump(_SCREENSHOT_WRITE_FILE)
    decision = _complete(
        [LLMMessage(role="user", content="确认创伤队列以及首次疼痛 4-7")],
        _SCREENSHOT_WRITE_FILE,
    )
    assert decision.action == "continue"


def test_screenshot_empty_after_retrieve_is_not_done():
    """Screenshot 1: tools ran, then empty STOP → Done with no prose."""
    messages = [
        LLMMessage(
            role="user",
            content="identify patients with pain score 4-7 on first encounter",
        ),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[
                {"id": "c1", "name": "execute", "args": {}},
                {"id": "c2", "name": "retrieve_tool_result", "args": {}},
            ],
        ),
        LLMMessage(role="tool", content="use_skill ok", tool_call_id="c1"),
        LLMMessage(role="tool", content="12 patients matched", tool_call_id="c2"),
    ]
    assert looks_like_gemini_command_dump(_SCREENSHOT_RETRIEVE)
    dump_msgs = list(messages)
    dump = _complete(dump_msgs, _SCREENSHOT_RETRIEVE)
    assert dump.action == "continue"
    assert GEMINI_SUMMARIZE_MARKER in str(dump_msgs[-1].content)
    empty_msgs = list(messages)
    empty = _complete(empty_msgs, "")
    assert empty.action == "continue"
    assert GEMINI_SUMMARIZE_MARKER in str(empty_msgs[-1].content)


def test_flatten_screenshot_write_file_does_not_reprint_script():
    raw = [
        {"role": "user", "parts": [{"text": "确认创伤队列以及首次疼痛 4-7"}]},
        {
            "role": "model",
            "parts": [{
                "function_call": {
                    "name": "write_file",
                    "args": {"path": _VALIDATE_PATH, "content": _SCRIPT},
                    "id": "fc1",
                }
            }],
        },
        {
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": "write_file",
                    "response": {"result": "wrote file"},
                    "id": "fc1",
                }
            }],
        },
    ]
    flat = _flatten_gemini_tool_history(raw)
    blob = "\n".join(t["parts"][0]["text"] for t in flat)
    assert _VALIDATE_PATH not in blob
    assert "import pandas" not in blob
    assert "[used write_file]" in blob
    assert "[result write_file: wrote file]" in blob
    assert GEMINI_ANSWER_NUDGE in blob
    assert not any(
        "function_call" in p or "function_response" in p
        for t in flat
        for p in t["parts"]
    )


def test_history_400_is_detected():
    assert _is_gemini_history_400(
        RuntimeError(
            "400 INVALID_ARGUMENT: Please ensure that the number of function "
            "response parts is equal to the number of function call parts "
            "of the function call turn and that the thought_signature is valid."
        )
    )
    assert not _is_gemini_history_400(RuntimeError("400 INVALID_ARGUMENT: bad image"))


def test_provider_flatten_retry_omits_write_file_args():
    """A thought_signature 400 must retry with flattened text, not the script."""
    seen: list[list] = []

    async def fake_request(contents, _config, **_kwargs):
        seen.append(contents)
        if len(seen) == 1:
            raise RuntimeError(
                "400 INVALID_ARGUMENT: thought_signature is required when "
                "replaying a function call"
            )
        return LLMResponse(
            content="创伤队列已确认；首次疼痛 4–7 的患者共 12 人。",
            model="gemini-3.7-flash",
            tokens_used=20,
        )

    messages = [
        LLMMessage(role="user", content="确认创伤队列以及首次疼痛 4-7"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{
                "id": "fc1",
                "name": "write_file",
                "args": {"path": _VALIDATE_PATH, "content": _SCRIPT},
            }],
            gemini_parts=[{
                "function_call": {
                    "name": "write_file",
                    "args": {"path": _VALIDATE_PATH, "content": _SCRIPT},
                    "id": "fc1",
                },
                "thought_signature": "sig",
            }],
        ),
        LLMMessage(role="tool", content="wrote file", tool_call_id="fc1"),
    ]

    mock_types = MagicMock()
    mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())

    with patch("clawagents.providers.llm.types", mock_types):
        provider = GeminiProvider.__new__(GeminiProvider)
        provider.client = MagicMock()
        provider.model = "gemini-3.7-flash"
        provider._max_tokens = 8192
        provider._temperature = 0
        provider.retry_policy = None
        provider._request_once = fake_request  # type: ignore[method-assign]
        result = asyncio.run(provider.chat(messages))

    assert len(seen) == 2
    first_blob = str(seen[0])
    assert "write_file" in first_blob
    assert _VALIDATE_PATH in first_blob
    second_blob = str(seen[1])
    assert "function_call" not in second_blob
    assert _VALIDATE_PATH not in second_blob
    assert _SCRIPT[:40] not in second_blob
    assert "[used write_file]" in second_blob
    assert GEMINI_ANSWER_NUDGE in second_blob
    assert "12" in result.content
    assert not looks_like_gemini_command_dump(result.content)


def test_tool_turn_hides_command_dump_from_chat():
    events = _Events()
    executor = ToolTurnExecutor.__new__(ToolTurnExecutor)
    executor._use_native_tools = True
    executor._events = events
    executor._session_writer = None

    async def _skip(**_kwargs):
        return None

    executor._execute_single = _skip  # type: ignore[method-assign]
    asyncio.run(
        executor.execute(
            state=SimpleNamespace(),
            messages=[],
            response=LLMResponse(
                content=_SCREENSHOT_WRITE_FILE,
                model="gemini-3.7-flash",
                tokens_used=4,
            ),
            thinking=None,
            tool_calls=[ParsedToolCall(tool_name="write_file", args={})],
            native_tool_calls=[
                NativeToolCall(tool_name="write_file", args={}, tool_call_id="fc1")
            ],
            round_index=1,
        )
    )
    assert "assistant_message" not in events.kinds

    events.kinds.clear()
    asyncio.run(
        executor.execute(
            state=SimpleNamespace(),
            messages=[],
            response=LLMResponse(
                content="Writing a validator next.",
                model="gemini-3.7-flash",
                tokens_used=4,
            ),
            thinking=None,
            tool_calls=[ParsedToolCall(tool_name="write_file", args={})],
            native_tool_calls=[
                NativeToolCall(tool_name="write_file", args={}, tool_call_id="fc1")
            ],
            round_index=1,
        )
    )
    assert "assistant_message" in events.kinds


def test_stream_parts_do_not_duplicate_function_calls():
    first = SimpleNamespace(
        function_call=SimpleNamespace(name="execute", args={"cmd": "l"}, id=None),
        text=None,
        thought=False,
    )
    second = SimpleNamespace(
        function_call=SimpleNamespace(name="execute", args={"cmd": "ls"}, id="api-9"),
        text=None,
        thought=False,
    )
    parts: list = []
    _absorb_gemini_stream_parts(parts, [first])
    _absorb_gemini_stream_parts(parts, [second])
    assert len(parts) == 1
    assert parts[0].function_call.args == {"cmd": "ls"}
    assert parts[0].function_call.id == "api-9"
