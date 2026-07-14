"""Multimodal user-message input: invoke(images=...) → first user message
carries image blocks, and each provider formats them correctly."""

from __future__ import annotations

import base64

import pytest

from clawagents.agent import create_claw_agent
from clawagents.media.images import (
    build_user_image_block,
    image_url_to_anthropic_block,
)
from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse

# A real 1×1 transparent PNG (67 bytes) — small enough to skip sanitization.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)
_PNG_B64 = base64.b64encode(_PNG_1x1).decode("ascii")


class _RecordingLLM(LLMProvider):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[list[LLMMessage]] = []

    @property
    def seen(self) -> list[LLMMessage]:
        # First call = the agent turn under test. Later calls can be
        # trajectory-judge traffic when another test leaves that feature
        # enabled process-wide; overwrite-style recording made this flaky.
        return self.calls[0] if self.calls else []

    async def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        return LLMResponse(content="done", model="recording", tokens_used=0)


# ── build_user_image_block ─────────────────────────────────────────────────

def test_build_block_from_base64():
    block = build_user_image_block(_PNG_B64, "image/png")
    assert block["type"] == "image_url"
    url = block["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_build_block_from_data_url_recovers_mime():
    data_url = f"data:image/png;base64,{_PNG_B64}"
    block = build_user_image_block(data_url)  # no explicit media_type
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_block_from_raw_bytes():
    block = build_user_image_block(_PNG_1x1, "image/png")
    assert block["type"] == "image_url"


def test_build_block_invalid_base64_returns_text_drop():
    block = build_user_image_block("!!!not base64!!!", "image/png")
    assert block["type"] == "text"


# ── image_url_to_anthropic_block ───────────────────────────────────────────

def test_image_url_to_anthropic_base64():
    part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}}
    out = image_url_to_anthropic_block(part)
    assert out == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64},
    }


def test_image_url_to_anthropic_http_url():
    part = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    out = image_url_to_anthropic_block(part)
    assert out == {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}


def test_image_url_to_anthropic_ignores_non_image():
    assert image_url_to_anthropic_block({"type": "text", "text": "hi"}) is None


# ── invoke(images=...) places image blocks in the first user message ───────

@pytest.mark.asyncio
async def test_invoke_attaches_image_to_first_user_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm = _RecordingLLM()
    agent = create_claw_agent(llm, memory=[], skills=[])
    await agent.invoke(
        "What is in this screenshot?",
        images=[{"data": _PNG_B64, "media_type": "image/png"}],
        max_iterations=1,
    )
    user_msgs = [m for m in llm.seen if m.role == "user"]
    assert user_msgs, "no user message reached the provider"
    content = user_msgs[0].content
    assert isinstance(content, list), "image attach should make content a block list"
    assert any(p.get("type") == "text" for p in content)
    imgs = [p for p in content if p.get("type") == "image_url"]
    assert len(imgs) == 1
    assert imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_invoke_without_images_keeps_string_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm = _RecordingLLM()
    agent = create_claw_agent(llm, memory=[], skills=[])
    await agent.invoke("plain text task", max_iterations=1)
    user_msgs = [m for m in llm.seen if m.role == "user"]
    assert isinstance(user_msgs[0].content, str)


# ── Anthropic provider converts image_url → image on the wire ──────────────

def test_anthropic_message_content_converts_image_url():
    # Exercises the real provider helper used inside AnthropicProvider.chat.
    from clawagents.providers.llm import _anthropic_message_content

    blocks = _anthropic_message_content(
        [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}},
        ]
    )
    image_blocks = [b for b in blocks if b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["type"] == "base64"
    assert image_blocks[0]["source"]["media_type"] == "image/png"
    # No stray image_url survived into the Anthropic payload.
    assert not any(b.get("type") == "image_url" for b in blocks)
    # Text block preserved.
    assert any(b.get("type") == "text" for b in blocks)


def test_anthropic_message_content_passes_through_strings_and_tool_results():
    from clawagents.providers.llm import _anthropic_message_content

    assert _anthropic_message_content("plain") == "plain"
    tool_blocks = [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]
    assert _anthropic_message_content(tool_blocks) == tool_blocks
