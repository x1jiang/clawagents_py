"""File attachments (PDF / DOCX) on the first user message.

PDFs travel as the canonical OpenAI-style ``file`` part
(``{"type":"file","file":{"filename":…,"file_data":"data:application/pdf;base64,…"}}``)
and are converted per provider wire (Anthropic ``document``, Responses
``input_file``, Converse ``document``, Gemini ``inline_data``). DOCX has no
native provider support anywhere, so it is text-extracted (stdlib zip+XML)
into a plain text block at build time.
"""

from __future__ import annotations

import base64
import io
import zipfile

import pytest

from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse

_PDF_BYTES = b"%PDF-1.4 fake little pdf for wire tests"
_PDF_B64 = base64.b64encode(_PDF_BYTES).decode("ascii")
_PDF_DATA_URL = f"data:application/pdf;base64,{_PDF_B64}"

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

FILE_PART = {
    "type": "file",
    "file": {"filename": "report.pdf", "file_data": _PDF_DATA_URL},
}


def _make_docx(paragraphs: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body></w:document>",
        )
        zf.writestr("[Content_Types].xml", "<Types/>")
    return buf.getvalue()


class _RecordingLLM(LLMProvider):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[list[LLMMessage]] = []

    @property
    def seen(self) -> list[LLMMessage]:
        return self.calls[0] if self.calls else []

    async def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        return LLMResponse(content="done", model="recording", tokens_used=0)


# ── build_user_file_block: PDF → canonical file part ───────────────────────


def test_build_file_block_pdf_canonical():
    from clawagents.media.documents import build_user_file_block

    block = build_user_file_block(_PDF_B64, "application/pdf", name="report.pdf")
    assert block["type"] == "file"
    assert block["file"]["filename"] == "report.pdf"
    assert block["file"]["file_data"] == _PDF_DATA_URL


def test_build_file_block_accepts_data_url_and_bytes():
    from clawagents.media.documents import build_user_file_block

    from_url = build_user_file_block(_PDF_DATA_URL, name="a.pdf")
    from_bytes = build_user_file_block(_PDF_BYTES, "application/pdf", name="a.pdf")
    assert from_url["type"] == "file"
    assert from_bytes["type"] == "file"
    assert from_url["file"]["file_data"] == from_bytes["file"]["file_data"]


def test_build_file_block_pdf_too_large_drops_to_text():
    from clawagents.media.documents import build_user_file_block

    block = build_user_file_block(
        _PDF_B64, "application/pdf", name="big.pdf", max_bytes=4
    )
    assert block["type"] == "text"
    assert "big.pdf" in block["text"]


def test_build_file_block_invalid_base64_drops_to_text():
    from clawagents.media.documents import build_user_file_block

    block = build_user_file_block("%%%not-b64%%%", "application/pdf", name="x.pdf")
    assert block["type"] == "text"


def test_build_file_block_unknown_type_drops_to_text():
    from clawagents.media.documents import build_user_file_block

    block = build_user_file_block(_PDF_B64, "application/zip", name="x.zip")
    assert block["type"] == "text"
    assert "x.zip" in block["text"]


# ── build_user_file_block: DOCX → extracted text block ─────────────────────


def test_build_file_block_docx_extracts_text():
    from clawagents.media.documents import build_user_file_block

    raw = _make_docx(["hello world", "second paragraph"])
    block = build_user_file_block(
        base64.b64encode(raw).decode("ascii"), DOCX_MIME, name="notes.docx"
    )
    assert block["type"] == "text"
    assert "notes.docx" in block["text"]
    assert "hello world" in block["text"]
    assert "second paragraph" in block["text"]


def test_build_file_block_docx_truncates_long_text():
    from clawagents.media.documents import build_user_file_block

    raw = _make_docx(["A" * 500, "B" * 500])
    block = build_user_file_block(
        base64.b64encode(raw).decode("ascii"),
        DOCX_MIME,
        name="long.docx",
        max_text_chars=100,
    )
    assert block["type"] == "text"
    assert "truncated" in block["text"]
    assert len(block["text"]) < 400


def test_build_file_block_docx_corrupt_falls_back_to_note():
    from clawagents.media.documents import build_user_file_block

    block = build_user_file_block(
        base64.b64encode(b"this is not a zip").decode("ascii"),
        DOCX_MIME,
        name="broken.docx",
    )
    assert block["type"] == "text"
    assert "broken.docx" in block["text"]


# ── invoke(files=…) reaches the provider on the first user message ─────────


@pytest.mark.asyncio
async def test_invoke_attaches_file_to_first_user_message(tmp_path, monkeypatch):
    from clawagents.agent import create_claw_agent

    monkeypatch.chdir(tmp_path)
    llm = _RecordingLLM()
    agent = create_claw_agent(llm, memory=[], skills=[])
    await agent.invoke(
        "Summarize this report.",
        files=[{"data": _PDF_B64, "media_type": "application/pdf", "name": "r.pdf"}],
        max_iterations=1,
    )
    user_msgs = [m for m in llm.seen if m.role == "user"]
    assert user_msgs, "no user message reached the provider"
    content = user_msgs[0].content
    assert isinstance(content, list), "file attach should make content a block list"
    assert any(p.get("type") == "text" for p in content)
    file_parts = [p for p in content if p.get("type") == "file"]
    assert len(file_parts) == 1
    assert file_parts[0]["file"]["filename"] == "r.pdf"


# ── provider wire conversions ──────────────────────────────────────────────


def test_anthropic_content_converts_file_part_to_document():
    from clawagents.providers.llm import _anthropic_message_content

    blocks = _anthropic_message_content([{"type": "text", "text": "read"}, dict(FILE_PART)])
    docs = [b for b in blocks if b.get("type") == "document"]
    assert len(docs) == 1
    assert docs[0]["source"] == {
        "type": "base64",
        "media_type": "application/pdf",
        "data": _PDF_B64,
    }
    assert docs[0].get("title") == "report.pdf"
    assert not any(b.get("type") == "file" for b in blocks)


def test_responses_parts_convert_file_part_to_input_file():
    from clawagents.providers.llm import _content_to_responses_parts

    parts = _content_to_responses_parts([dict(FILE_PART)], "user")
    assert parts == [
        {"type": "input_file", "filename": "report.pdf", "file_data": _PDF_DATA_URL}
    ]


def test_converse_blocks_convert_file_part_to_document():
    from clawagents.providers.llm import _converse_content_blocks

    blocks = _converse_content_blocks([dict(FILE_PART)])
    (doc,) = blocks
    assert doc["document"]["format"] == "pdf"
    assert doc["document"]["source"]["bytes"] == _PDF_BYTES
    # Converse names allow alphanumerics/space/hyphen/parens/brackets only.
    assert doc["document"]["name"]


def test_converse_document_name_sanitized():
    from clawagents.providers.llm import _converse_content_blocks

    part = {
        "type": "file",
        "file": {"filename": "Q3/財務 report!!.pdf", "file_data": _PDF_DATA_URL},
    }
    (doc,) = _converse_content_blocks([part])
    name = doc["document"]["name"]
    assert all(c.isalnum() or c in " -()[]" for c in name)
    assert "  " not in name
    assert name


def test_gemini_part_from_file_block():
    from clawagents.providers.llm import _gemini_part_from_block

    part = _gemini_part_from_block(dict(FILE_PART))
    assert part == {
        "inline_data": {"mime_type": "application/pdf", "data": _PDF_BYTES}
    }
    assert _gemini_part_from_block({"type": "text", "text": "hi"}) == {"text": "hi"}
    img = _gemini_part_from_block(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PDF_B64}"}}
    )
    assert img is not None and "inline_data" in img


# ── budget accounting and compaction safety ────────────────────────────────


def test_tokenizer_counts_file_parts():
    from clawagents.tokenizer import count_tokens_content

    big_b64 = base64.b64encode(b"x" * 64_000).decode("ascii")
    big_part = {
        "type": "file",
        "file": {
            "filename": "big.pdf",
            "file_data": f"data:application/pdf;base64,{big_b64}",
        },
    }
    with_file = count_tokens_content([{"type": "text", "text": "hi"}, big_part])
    without = count_tokens_content([{"type": "text", "text": "hi"}])
    # A ~64KB PDF must register on the token budget (≫ the text alone),
    # or compaction preflight under-counts and requests overflow.
    assert with_file - without > 1000


def test_content_key_text_file_placeholder_is_bounded():
    from clawagents.graph.agent_loop import _content_key_text

    key = _content_key_text([{"type": "text", "text": "read this"}, dict(FILE_PART)])
    assert "read this" in key
    assert _PDF_B64 not in key
    assert "file attachment" in key  # placeholder marker, not silent omission
    assert len(key) < 200
