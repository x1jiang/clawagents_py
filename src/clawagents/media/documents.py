"""File-attachment content blocks (PDF / DOCX) for user messages.

PDFs become the canonical OpenAI-style ``file`` part::

    {"type": "file",
     "file": {"filename": "report.pdf",
              "file_data": "data:application/pdf;base64,<b64>"}}

The OpenAI Chat Completions provider passes it through natively; the other
wires convert it (Anthropic ``document``, Responses ``input_file``, Bedrock
Converse ``document``, Gemini ``inline_data``).

DOCX has no native support on any provider, so it is text-extracted here —
stdlib ``zipfile`` + ``xml.etree`` only, no new dependency — and returned as
a plain ``text`` block, which every provider and the compaction/session
paths already handle.

Anything that can't be converted degrades to a short ``text`` note (never
raises), matching ``media/images.py`` semantics.

Public API:

    build_user_file_block(data, media_type="application/pdf", *, name=None,
                          max_bytes=..., max_text_chars=...) -> dict
    file_part_to_anthropic_block(part) -> dict | None
"""

from __future__ import annotations

import base64
import binascii
import io
import zipfile
from typing import Any, Dict, Optional, Union
from xml.etree import ElementTree

PDF_MEDIA_TYPE = "application/pdf"
DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

# Per-file cap. Anthropic rejects requests over ~32MB total, and base64
# inflates by ~1.33×, so 10MB decoded per file keeps a few attachments plus
# history comfortably under every provider's request ceiling.
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
# Extracted DOCX text cap (~25k tokens) with an explicit truncation marker.
DEFAULT_MAX_DOCX_CHARS = 100_000
# Zip-bomb guard: refuse to inflate a document.xml larger than this.
_MAX_DOCX_XML_BYTES = 50 * 1024 * 1024

_DOCX_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _note(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _extract_docx_text(raw: bytes, *, max_chars: int) -> Optional[str]:
    """Pull paragraph text out of a .docx (zip of XML). Returns None on any
    parse failure — callers degrade to a text note instead of raising.

    ``xml.etree`` refuses custom entity expansion, and the uncompressed-size
    guard bounds decompression, so a hostile file can fail but not blow up
    the process.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            info = zf.getinfo("word/document.xml")
            if info.file_size > _MAX_DOCX_XML_BYTES:
                return None
            xml_bytes = zf.read("word/document.xml")
        root = ElementTree.fromstring(xml_bytes)
    except Exception:  # noqa: BLE001 - corrupt zip/XML → note fallback
        return None

    paras: list[str] = []
    total = 0
    for p in root.iter(f"{_DOCX_W_NS}p"):
        text = "".join(t.text or "" for t in p.iter(f"{_DOCX_W_NS}t"))
        paras.append(text)
        total += len(text) + 1
        if total > max_chars:
            break
    text = "\n".join(paras).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text


def build_user_file_block(
    data: Union[str, bytes],
    media_type: str = PDF_MEDIA_TYPE,
    *,
    name: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_text_chars: int = DEFAULT_MAX_DOCX_CHARS,
) -> Dict[str, Any]:
    """Build a content block for one user file attachment.

    ``data`` may be a base64 string (with or without a ``data:`` prefix) or
    raw bytes. PDFs return the canonical ``file`` part; DOCX returns an
    extracted-text block; everything unconvertible returns a short text note
    so the turn still progresses.
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
                if declared and media_type == PDF_MEDIA_TYPE:
                    media_type = declared
        b64 = "".join(raw_str.split())

    label = name or ("attachment.docx" if media_type == DOCX_MEDIA_TYPE else "attachment.pdf")

    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return _note(f"[file {label} data was not valid base64, dropped]")

    if len(raw) > max_bytes:
        return _note(
            f"[file {label} too large "
            f"({len(raw) // (1024 * 1024)}MB > {max_bytes // (1024 * 1024)}MB), dropped]"
        )

    if media_type == PDF_MEDIA_TYPE:
        return {
            "type": "file",
            "file": {
                "filename": label,
                "file_data": f"data:{PDF_MEDIA_TYPE};base64,{b64}",
            },
        }

    if media_type == DOCX_MEDIA_TYPE:
        text = _extract_docx_text(raw, max_chars=max_text_chars)
        if text is None:
            return _note(f"[document {label} could not be parsed (.docx expected), dropped]")
        return _note(f"[Attached document: {label}]\n\n{text}")

    return _note(f"[file {label} ({media_type}) is not a supported attachment type, dropped]")


def file_part_to_anthropic_block(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a canonical ``file`` part to an Anthropic ``document`` block.

    Anthropic's Messages API has no ``file`` type; PDFs go as
    ``{"type":"document","source":{"type":"base64",…}}`` (or a ``url``
    source). Returns ``None`` for anything else so callers drop the part
    instead of sending an invalid block.
    """
    if not isinstance(part, dict) or part.get("type") != "file":
        return None
    f = part.get("file") or {}
    fd = f.get("file_data") or ""
    if not isinstance(fd, str) or not fd:
        return None
    title = f.get("filename") or None
    if fd.startswith("data:") and ";base64," in fd:
        header, b64 = fd[5:].split(";base64,", 1)
        media_type = header.split(";", 1)[0].strip() or PDF_MEDIA_TYPE
        if media_type != PDF_MEDIA_TYPE:
            return None
        block: Dict[str, Any] = {
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }
        if title:
            block["title"] = title
        return block
    if fd.startswith(("http://", "https://")):
        block = {"type": "document", "source": {"type": "url", "url": fd}}
        if title:
            block["title"] = title
        return block
    return None


__all__ = [
    "PDF_MEDIA_TYPE",
    "DOCX_MEDIA_TYPE",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_DOCX_CHARS",
    "build_user_file_block",
    "file_part_to_anthropic_block",
]
