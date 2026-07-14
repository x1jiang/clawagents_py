"""Provider wire-format handling for user-message image blocks.

Images attach to the first user message in the internal OpenAI-style
``image_url`` shape. The Anthropic and Gemini providers already convert
that shape on the wire; these tests pin the remaining surfaces: OpenAI
Responses input items, Bedrock Converse content blocks, and compaction's
text view of multimodal content (which must never ``str()`` a base64
payload into a prompt or reuse key).
"""

from __future__ import annotations

import base64

_PNG_B64 = base64.b64encode(b"not-a-real-png-but-fine-for-wire-tests").decode("ascii")
_DATA_URL = f"data:image/png;base64,{_PNG_B64}"

USER_BLOCKS = [
    {"type": "text", "text": "look at this"},
    {"type": "image_url", "image_url": {"url": _DATA_URL}},
]


# ── OpenAI Responses wire: image_url must become input_image ───────────────


def test_responses_input_converts_user_image_blocks():
    from clawagents.providers.llm import _messages_to_responses_input

    instructions, items = _messages_to_responses_input(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": USER_BLOCKS},
        ]
    )
    assert instructions == "sys"
    (user_item,) = items
    parts = user_item["content"]
    assert isinstance(parts, list)
    assert [p["type"] for p in parts] == ["input_text", "input_image"]
    assert parts[0]["text"] == "look at this"
    # Responses API wants image_url as a plain string, not a nested object.
    assert parts[1]["image_url"] == _DATA_URL


def test_responses_input_string_content_unchanged():
    from clawagents.providers.llm import _messages_to_responses_input

    _, items = _messages_to_responses_input([{"role": "user", "content": "hi"}])
    assert items == [{"role": "user", "content": "hi"}]


def test_responses_input_skips_message_when_no_part_converts():
    from clawagents.providers.llm import _messages_to_responses_input

    _, items = _messages_to_responses_input(
        [
            {"role": "user", "content": [{"type": "mystery"}]},
            {"role": "user", "content": "still here"},
        ]
    )
    assert items == [{"role": "user", "content": "still here"}]


def test_responses_input_assistant_list_becomes_output_text():
    from clawagents.providers.llm import _messages_to_responses_input

    _, items = _messages_to_responses_input(
        [{"role": "assistant", "content": [{"type": "text", "text": "prev answer"}]}]
    )
    (item,) = items
    assert item["content"] == [{"type": "output_text", "text": "prev answer"}]


# ── Bedrock Converse: native image blocks, never str(list) ────────────────


def test_converse_blocks_string_passthrough():
    from clawagents.providers.llm import _converse_content_blocks

    assert _converse_content_blocks("hi") == [{"text": "hi"}]
    assert _converse_content_blocks(None) == [{"text": ""}]


def test_converse_blocks_convert_text_and_image():
    from clawagents.providers.llm import _converse_content_blocks

    blocks = _converse_content_blocks(USER_BLOCKS)
    assert blocks[0] == {"text": "look at this"}
    img = blocks[1]["image"]
    assert img["format"] == "png"
    assert img["source"]["bytes"] == base64.b64decode(_PNG_B64)


def test_converse_blocks_jpg_alias_normalized():
    from clawagents.providers.llm import _converse_content_blocks

    url = f"data:image/jpg;base64,{_PNG_B64}"
    blocks = _converse_content_blocks([{"type": "image_url", "image_url": {"url": url}}])
    assert blocks[0]["image"]["format"] == "jpeg"


def test_converse_blocks_drop_unsupported_and_invalid():
    from clawagents.providers.llm import _converse_content_blocks

    blocks = _converse_content_blocks(
        [
            {"type": "image_url", "image_url": {"url": f"data:image/svg+xml;base64,{_PNG_B64}"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%not-b64%%%"}},
            {"type": "text", "text": "kept"},
        ]
    )
    assert blocks == [{"text": "kept"}]
    # The base64 payload must not leak into any text block.
    assert all(_PNG_B64 not in b.get("text", "") for b in blocks)


# ── Compaction text view: bounded placeholders, stable distinct keys ───────


def test_content_key_text_string_passthrough():
    from clawagents.graph.agent_loop import _content_key_text

    assert _content_key_text("plain") == "plain"


def test_content_key_text_replaces_images_with_bounded_placeholder():
    from clawagents.graph.agent_loop import _content_key_text

    key = _content_key_text(USER_BLOCKS)
    assert "look at this" in key
    assert _PNG_B64 not in key
    assert len(key) < 200


def test_content_key_text_distinct_images_get_distinct_keys():
    from clawagents.graph.agent_loop import _content_key_text

    a = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    b = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]
    assert _content_key_text(a) != _content_key_text(b)
    assert _content_key_text(a) == _content_key_text(list(a))  # deterministic
