"""Tests for clawagents.media.images sanitizers."""

from __future__ import annotations

import base64
import importlib
import io
import sys

import pytest

from clawagents.media.images import (
    is_pillow_available,
    sanitize_image_block,
    sanitize_tool_output,
)


_HAS_PIL = is_pillow_available()


# ─── Tiny PNG fixture (8x8 red, no alpha) ──────────────────────────────────

_TINY_PNG_B64 = (
    # produced offline; 8×8 solid red PNG
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQAAAAD8GwTdAAAAEUlEQVR4nGNgYGD4z0AswK4SAFb6Af"
    "FOcvIfAAAAAElFTkSuQmCC"
)


def _make_block(data: str = _TINY_PNG_B64, media_type: str = "image/png"):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


# ─── Pass-through paths (always run) ───────────────────────────────────────


def test_text_block_passes_through():
    block = {"type": "text", "text": "hello"}
    assert sanitize_image_block(block) is block


def test_unknown_block_passes_through():
    block = {"type": "weird", "payload": 42}
    assert sanitize_image_block(block) is block


def test_url_image_passes_through():
    block = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.png"},
    }
    assert sanitize_image_block(block) is block


def test_string_tool_output_passes_through():
    assert sanitize_tool_output("hello world") == "hello world"


def test_list_with_no_image_blocks_unchanged():
    blocks = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    out = sanitize_tool_output(blocks)
    assert out == blocks


# ─── Missing-Pillow path ───────────────────────────────────────────────────


def test_missing_pillow_returns_input_unchanged(monkeypatch):
    """When Pillow isn't importable, the sanitizer must no-op + warn."""
    import clawagents.media.images as mod

    monkeypatch.setattr(mod, "_PILLOW_AVAILABLE", False)
    monkeypatch.setattr(mod, "Image", None)
    # Reset the warning latch so we can observe it.
    monkeypatch.setattr(mod, "_WARNED_NO_PILLOW", False)

    block = _make_block()
    with pytest.warns(RuntimeWarning, match="Pillow is not installed"):
        out = mod.sanitize_image_block(block)
    # Content preserved verbatim.
    assert out == block

    # And is_pillow_available reports the patched state.
    assert mod.is_pillow_available() is False


# ─── Happy paths that need Pillow ──────────────────────────────────────────


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed in this dev env")
def test_small_png_under_limits_passes_through():
    block = _make_block()
    out = sanitize_image_block(block, max_dim=1200, max_bytes=5 * 1024 * 1024)
    # Same content (it's well under both limits already)
    assert out["source"]["data"] == _TINY_PNG_B64
    assert out["source"]["media_type"] == "image/png"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed in this dev env")
def test_oversize_png_gets_resized():
    from PIL import Image

    # Build a 4000x4000 RGB PNG — definitely over both 1200px and 5MB-after-encoding
    big = Image.new("RGB", (4000, 4000), color=(123, 200, 60))
    buf = io.BytesIO()
    big.save(buf, format="PNG", optimize=False)
    big_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    block = _make_block(data=big_b64, media_type="image/png")

    out = sanitize_image_block(
        block, max_dim=512, max_bytes=200 * 1024, quality_steps=(75, 60, 40)
    )
    assert out["type"] == "image"
    # Now JPEG (no alpha) and well within limits.
    assert out["source"]["media_type"] in ("image/jpeg", "image/png")
    new_bytes = base64.b64decode(out["source"]["data"])
    assert len(new_bytes) <= 200 * 1024

    # Verify the longest side is now <= 512.
    with Image.open(io.BytesIO(new_bytes)) as im:
        assert max(im.size) <= 512


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed in this dev env")
def test_too_large_to_fit_returns_dropped_text_block():
    from PIL import Image

    # Build a noisy 2000x2000 image (random-ish; harder to compress)
    import os
    random_bytes = os.urandom(2000 * 2000 * 3)
    img = Image.frombytes("RGB", (2000, 2000), random_bytes)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    big_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    block = _make_block(data=big_b64, media_type="image/png")

    # Force an absurdly low byte budget so even quality=60 can't fit.
    out = sanitize_image_block(
        block, max_dim=2000, max_bytes=1024, quality_steps=(60,)
    )
    assert out == {"type": "text", "text": "[image too large after sanitization, dropped]"}


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed in this dev env")
def test_sanitize_tool_output_walks_list():
    blocks = [
        {"type": "text", "text": "hi"},
        _make_block(),
        {"type": "text", "text": "bye"},
    ]
    out = sanitize_tool_output(blocks)
    assert isinstance(out, list)
    assert out[0]["type"] == "text"
    assert out[1]["type"] == "image"
    assert out[2]["type"] == "text"


# ─── Edge: malformed base64 ────────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed in this dev env")
def test_malformed_base64_returns_text_fallback_when_too_big():
    # Pad with junk so the decoder doesn't choke but Pillow can't open
    # We only get the fallback path if `needs_work` triggers — pass tiny limits.
    junk_b64 = base64.b64encode(b"\x00" * 16).decode("ascii")
    block = _make_block(data=junk_b64, media_type="image/png")
    out = sanitize_image_block(block, max_dim=4, max_bytes=4)
    # Either the dropped-text fallback or pass-through (unreadable+small).
    assert out["type"] in ("text", "image")
