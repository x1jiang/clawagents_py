"""Image sanitization for tool-result content blocks.

Anthropic's Messages API rejects images > 5MB and tends to fail on images
much larger than ~2000px on a side. Tools that surface remote images, screen
captures, or large file attachments can blow past those limits and silently
break the conversation. This module clamps base64 image blocks down to safe
limits via Pillow.

Pillow is *optional*: if it is not installed, the sanitizers return the input
unchanged after emitting a one-time warning. URL-source images and non-image
blocks always pass through untouched.

Public API:

    sanitize_image_block(block, *, max_dim=1200, max_bytes=5*1024*1024,
                         quality_steps=(90, 75, 60)) -> dict
    sanitize_tool_output(output, *, max_dim=1200, max_bytes=5*1024*1024)
        -> list[dict] | str
    is_pillow_available() -> bool
"""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Any, Dict, List, Tuple, Union

# ─── Pillow availability ──────────────────────────────────────────────────

try:  # pragma: no cover - import guard
    from PIL import Image as _PILImage

    Image: Any = _PILImage
    _PILLOW_AVAILABLE = True
    # Decompression-bomb hardening: pin the pixel ceiling explicitly so a
    # crafted image can't exhaust memory even if another import in the
    # process disabled Pillow's default guard (a common footgun:
    # ``Image.MAX_IMAGE_PIXELS = None``).
    if getattr(_PILImage, "MAX_IMAGE_PIXELS", None) is None:
        _PILImage.MAX_IMAGE_PIXELS = 178_956_970  # Pillow's stock default
except Exception:  # pragma: no cover - exercised by mocked tests
    Image = None
    _PILLOW_AVAILABLE = False

_WARNED_NO_PILLOW = False


def is_pillow_available() -> bool:
    """Whether Pillow imported successfully at module load time."""
    return _PILLOW_AVAILABLE


def _warn_missing_pillow_once() -> None:
    global _WARNED_NO_PILLOW
    if _WARNED_NO_PILLOW:
        return
    _WARNED_NO_PILLOW = True
    warnings.warn(
        "Pillow is not installed — image sanitization is a no-op. "
        "Install with `pip install 'clawagents[media]'` to enable resize/recompress.",
        RuntimeWarning,
        stacklevel=2,
    )


# ─── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_MAX_DIM = 1200
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_QUALITY_STEPS: Tuple[int, ...] = (90, 75, 60)
_DROPPED_TEXT = "[image too large after sanitization, dropped]"


# ─── Helpers ───────────────────────────────────────────────────────────────


def _is_image_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") != "image":
        return False
    src = block.get("source")
    return isinstance(src, dict)


def _decode_b64(data: str) -> bytes:
    # Be lenient: strip data URL prefix if a caller passed one in.
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data, validate=False)


def _resize_and_compress(
    raw: bytes,
    *,
    max_dim: int,
    max_bytes: int,
    quality_steps: Tuple[int, ...],
    media_type: str,
) -> Tuple[bytes, str] | None:
    """Try to fit ``raw`` into ``max_bytes`` after clamping the longest side
    to ``max_dim``. Returns ``(bytes, media_type)`` or ``None`` if no quality
    setting succeeded.
    """
    assert _PILLOW_AVAILABLE and Image is not None  # noqa: S101 - guarded by callers

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return None

    # Decide output format: keep PNG for images with alpha, otherwise JPEG.
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    out_format = "PNG" if (has_alpha and media_type == "image/png") else "JPEG"
    out_media = "image/png" if out_format == "PNG" else "image/jpeg"

    # Calculate target size, preserving aspect ratio.
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    if out_format == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")

    # PNG path doesn't honor quality steps — try once with optimize.
    if out_format == "PNG":
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, out_media
        # Fall through and try JPEG as a last resort, even though we'd lose alpha.
        rgb = img.convert("RGB")
        for q in quality_steps:
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=q, optimize=True)
            jpg = buf.getvalue()
            if len(jpg) <= max_bytes:
                return jpg, "image/jpeg"
        return None

    # JPEG path — walk down quality steps.
    for q in quality_steps:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, out_media
    return None


# ─── Public API ────────────────────────────────────────────────────────────


def sanitize_image_block(
    block: Dict[str, Any],
    *,
    max_dim: int = DEFAULT_MAX_DIM,
    max_bytes: int = DEFAULT_MAX_BYTES,
    quality_steps: Tuple[int, ...] = DEFAULT_QUALITY_STEPS,
) -> Dict[str, Any]:
    """Sanitize a single Anthropic-style image content block.

    - base64 source: decode → if bytes or any side exceeds limits, resize the
      longest side down to ``max_dim`` and recompress (JPEG, or PNG when the
      input is a PNG with alpha) walking through ``quality_steps``. If even
      the smallest quality step doesn't fit ``max_bytes``, the block is
      replaced with a text block explaining the drop.
    - URL source: passes through unchanged (we don't refetch).
    - Non-image blocks: pass through unchanged.

    The function never raises on malformed input — it returns a text fallback
    block so the conversation can still progress.
    """
    if not _is_image_block(block):
        return block

    source = block.get("source") or {}
    src_type = source.get("type")
    if src_type != "base64":
        # URL or unknown — leave it alone.
        return block

    if not _PILLOW_AVAILABLE:
        _warn_missing_pillow_once()
        return block

    data = source.get("data")
    media_type = source.get("media_type") or "image/jpeg"
    if not isinstance(data, str) or not data:
        return block

    try:
        raw = _decode_b64(data)
    except (binascii.Error, ValueError):
        return {
            "type": "text",
            "text": "[image source data was not valid base64, dropped]",
        }

    # Cheap path: if it's already small in bytes, also peek dimensions before bailing.
    needs_work = len(raw) > max_bytes
    if not needs_work:
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                w, h = probe.size
                if max(w, h) > max_dim:
                    needs_work = True
        except Exception:
            # Unreadable but small — leave block as-is rather than dropping.
            return block

    if not needs_work:
        return block

    result = _resize_and_compress(
        raw,
        max_dim=max_dim,
        max_bytes=max_bytes,
        quality_steps=quality_steps,
        media_type=media_type,
    )
    if result is None:
        return {"type": "text", "text": _DROPPED_TEXT}

    new_bytes, new_media = result
    new_b64 = base64.b64encode(new_bytes).decode("ascii")
    new_block = dict(block)
    new_block["source"] = {
        "type": "base64",
        "media_type": new_media,
        "data": new_b64,
    }
    return new_block


def build_user_image_block(
    data: Union[str, bytes],
    media_type: str = "image/png",
    *,
    max_dim: int = DEFAULT_MAX_DIM,
    max_bytes: int = DEFAULT_MAX_BYTES,
    quality_steps: Tuple[int, ...] = DEFAULT_QUALITY_STEPS,
) -> Dict[str, Any]:
    """Build a sanitized OpenAI-style ``image_url`` block for a user message.

    ``data`` may be a base64 string (with or without a ``data:`` prefix) or
    raw bytes. The image is clamped to ``max_dim`` / ``max_bytes`` (reusing the
    tool-result sanitizer) and returned as::

        {"type": "image_url",
         "image_url": {"url": "data:<mime>;base64,<b64>"}}

    This is the canonical internal shape: the OpenAI provider passes it through
    natively, the Gemini provider converts it to ``inline_data``, and the
    Anthropic provider converts it to an ``image``/``source`` block. If the
    bytes can't be decoded or shrunk to fit, a ``text`` block explaining the
    drop is returned instead so the turn still progresses.
    """
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    else:
        raw_str = data.strip()
        if raw_str.startswith("data:") and "," in raw_str:
            header, raw_str = raw_str.split(",", 1)
            # data:<mime>;base64 — recover the declared mime if the caller
            # didn't pass one explicitly.
            if header.startswith("data:"):
                declared = header[5:].split(";", 1)[0].strip()
                if declared and media_type == "image/png":
                    media_type = declared
        # Drop internal whitespace/newlines (MIME-wrapped base64) so the value
        # is canonical before validation and transport.
        b64 = "".join(raw_str.split())

    # Validate the base64 up front so the contract is deterministic regardless
    # of whether Pillow is installed (the sanitizer's Pillow-missing guard
    # would otherwise return the un-decoded block unchanged).
    try:
        base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return {
            "type": "text",
            "text": "[image source data was not valid base64, dropped]",
        }

    # Reuse the battle-tested tool-result sanitizer by wrapping as an
    # Anthropic-style block, then convert the (possibly shrunk) result.
    sanitized = sanitize_image_block(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        },
        max_dim=max_dim,
        max_bytes=max_bytes,
        quality_steps=quality_steps,
    )
    if sanitized.get("type") != "image":
        # Sanitizer replaced it with a text drop/error block — pass through.
        return sanitized

    src = sanitized.get("source") or {}
    out_media = src.get("media_type") or media_type
    out_data = src.get("data") or b64
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{out_media};base64,{out_data}"},
    }


def image_url_to_anthropic_block(part: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert an OpenAI-style ``image_url`` block to an Anthropic ``image``
    block. Returns ``None`` if the part isn't a data-URL image_url.

    Anthropic's Messages API has no ``image_url`` type; it requires
    ``{"type":"image","source":{"type":"base64","media_type":…,"data":…}}``
    (or a ``url`` source). URL-only ``image_url`` values are mapped to a URL
    source; malformed values yield ``None`` so the caller can drop them.
    """
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    url = ((part.get("image_url") or {}).get("url")) or ""
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:") and ";base64," in url:
        header, b64 = url[5:].split(";base64,", 1)
        media_type = header.split(";", 1)[0].strip() or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def sanitize_tool_output(
    output: Union[List[Dict[str, Any]], str],
    *,
    max_dim: int = DEFAULT_MAX_DIM,
    max_bytes: int = DEFAULT_MAX_BYTES,
    quality_steps: Tuple[int, ...] = DEFAULT_QUALITY_STEPS,
) -> Union[List[Dict[str, Any]], str]:
    """Sanitize a tool result (transcript string or list of content blocks).

    Strings pass through unchanged (no images possible). Lists are walked
    block-by-block; image blocks go through :func:`sanitize_image_block`,
    everything else is preserved verbatim.
    """
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return output

    return [
        sanitize_image_block(
            b,
            max_dim=max_dim,
            max_bytes=max_bytes,
            quality_steps=quality_steps,
        )
        if _is_image_block(b)
        else b
        for b in output
    ]


__all__ = [
    "is_pillow_available",
    "sanitize_image_block",
    "sanitize_tool_output",
    "build_user_image_block",
    "image_url_to_anthropic_block",
    "DEFAULT_MAX_DIM",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_QUALITY_STEPS",
]
