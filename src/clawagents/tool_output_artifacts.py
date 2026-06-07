"""Offload large tool outputs to artifact files (OpenHarness 0.1.9 pattern)."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Optional

DEFAULT_INLINE_CHARS = 12_000
DEFAULT_PREVIEW_CHARS = 2_000


def _safe_name(tool_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tool_name)[:64] or "tool"


def tool_artifact_dir(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or Path.cwd()) / ".clawagents" / "tool-artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def offload_tool_output_if_needed(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
    workspace: str | Path | None = None,
    inline_limit: int = DEFAULT_INLINE_CHARS,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> tuple[str, Optional[Path]]:
    if len(output) <= inline_limit:
        return output, None
    artifact_path = (
        tool_artifact_dir(workspace)
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{_safe_name(tool_name)}-{uuid.uuid4().hex[:12]}.txt"
    )
    artifact_path.write_text(output, encoding="utf-8", errors="replace")
    preview = output[:preview_chars]
    omitted = max(0, len(output) - len(preview))
    inline = (
        "[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Tool use id: {tool_use_id}\n"
        f"Original size: {len(output)} chars\n"
        f"Full output saved to: {artifact_path}\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        inline += f" ({omitted} chars omitted)"
    if preview:
        inline += f"\n\nPreview:\n{preview}"
    return inline, artifact_path
