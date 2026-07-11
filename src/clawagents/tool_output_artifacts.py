"""Offload large tool outputs to artifact files (OpenHarness 0.1.9 pattern).

Also stores reversible full text keyed by tool_use_id so the agent can call
``retrieve_tool_result`` after content crushing.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from clawagents.memory.content_crush import CrushResult, crush_tool_output

DEFAULT_INLINE_CHARS = 12_000
DEFAULT_PREVIEW_CHARS = 2_000


def _safe_name(tool_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tool_name)[:64] or "tool"


def _safe_id(tool_use_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", tool_use_id or "")[:80]
    return cleaned or uuid.uuid4().hex[:12]


def tool_artifact_dir(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or Path.cwd()) / ".clawagents" / "tool-artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _meta_path(directory: Path, artifact_id: str) -> Path:
    return directory / f"{artifact_id}.meta.json"


def _body_path(directory: Path, artifact_id: str) -> Path:
    return directory / f"{artifact_id}.txt"


def store_tool_artifact(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
    kind: str = "prose",
    workspace: str | Path | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[str, Path]:
    """Persist full tool output; return (artifact_id, body_path)."""
    directory = tool_artifact_dir(workspace)
    artifact_id = _safe_id(tool_use_id)
    # Avoid clobbering if the same id is reused with different content.
    body = _body_path(directory, artifact_id)
    if body.exists():
        artifact_id = f"{artifact_id}-{uuid.uuid4().hex[:8]}"
        body = _body_path(directory, artifact_id)
    body.write_text(output, encoding="utf-8", errors="replace")
    meta = {
        "id": artifact_id,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "kind": kind,
        "chars": len(output),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path": str(body),
    }
    if extra_meta:
        meta.update(extra_meta)
    _meta_path(directory, artifact_id).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return artifact_id, body


def load_tool_artifact(
    artifact_id: str,
    *,
    workspace: str | Path | None = None,
    max_chars: int | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Load full (or capped) artifact text. Returns (ok, text_or_error, meta)."""
    directory = tool_artifact_dir(workspace)
    aid = _safe_id(artifact_id)
    meta_file = _meta_path(directory, aid)
    body = _body_path(directory, aid)
    meta: dict[str, Any] | None = None
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
    if not body.exists():
        # Legacy timestamped filenames — scan metas for matching id / tool_use_id
        for candidate in directory.glob("*.meta.json"):
            try:
                m = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if m.get("id") == artifact_id or m.get("tool_use_id") == artifact_id:
                path = Path(m.get("path") or "")
                if path.is_file():
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if max_chars is not None and len(text) > max_chars:
                        text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
                    return True, text, m
        return False, f"No tool artifact found for id={artifact_id!r}", None
    text = body.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    return True, text, meta


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
    artifact_id, artifact_path = store_tool_artifact(
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        output=output,
        kind="raw",
        workspace=workspace,
    )
    preview = output[:preview_chars]
    omitted = max(0, len(output) - len(preview))
    inline = (
        "[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Artifact id: {artifact_id}\n"
        f"Tool use id: {tool_use_id}\n"
        f"Original size: {len(output)} chars\n"
        f"Full output saved to: {artifact_path}\n"
        f"Retrieve with: retrieve_tool_result(id=\"{artifact_id}\")\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        inline += f" ({omitted} chars omitted)"
    if preview:
        inline += f"\n\nPreview:\n{preview}"
    return inline, artifact_path


def search_tool_artifacts(
    query: str,
    *,
    workspace: str | Path | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Lightweight local search over stored tool-artifact bodies + meta.

    Prefer this over re-running expensive tools when looking for a prior dump.
    Uses simple case-insensitive substring match (no cloud index).
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    directory = tool_artifact_dir(workspace)
    hits: list[dict[str, Any]] = []
    for meta_file in sorted(directory.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        body_path = Path(meta.get("path") or "")
        if not body_path.is_file():
            body_path = _body_path(directory, str(meta.get("id") or meta_file.stem.replace(".meta", "")))
        snippet = ""
        hay = f"{meta.get('tool_name', '')} {meta.get('id', '')} {meta.get('kind', '')}".lower()
        matched = q in hay
        if body_path.is_file():
            try:
                text = body_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if q in text.lower():
                matched = True
                idx = text.lower().find(q)
                start = max(0, idx - 80)
                end = min(len(text), idx + len(q) + 120)
                snippet = text[start:end].replace("\n", " ")
            elif not matched:
                continue
        if not matched:
            continue
        hits.append({
            "id": meta.get("id"),
            "tool_name": meta.get("tool_name"),
            "kind": meta.get("kind"),
            "chars": meta.get("chars"),
            "snippet": snippet[:240],
            "created_at": meta.get("created_at"),
        })
        if len(hits) >= max(1, limit):
            break
    return hits


def prepare_tool_output_for_context(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
    workspace: str | Path | None = None,
    crush_threshold: int = 2_000,
    inline_limit: int = DEFAULT_INLINE_CHARS,
) -> tuple[str, Optional[str]]:
    """Crush oversized outputs and store full text when crushed or huge.

    Returns ``(prompt_text, artifact_id_or_None)``.
    """
    if not isinstance(output, str):
        output = str(output)

    crush: CrushResult = crush_tool_output(
        output, tool_name=tool_name, threshold=crush_threshold
    )

    # Always store when we crushed or when still over inline limit.
    need_store = crush.did_crush or len(output) > inline_limit
    artifact_id: str | None = None
    if need_store:
        artifact_id, _path = store_tool_artifact(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            output=output,
            kind=crush.kind,
            workspace=workspace,
            extra_meta={
                "crushed_chars": crush.crushed_chars,
                "did_crush": crush.did_crush,
            },
        )

    if not crush.did_crush and len(output) <= inline_limit:
        return output, artifact_id

    if crush.did_crush and artifact_id:
        header = (
            f"[Crushed tool output kind={crush.kind} id={artifact_id}]\n"
            f"Original: {crush.original_chars} chars → {crush.crushed_chars} chars. "
            f"Call retrieve_tool_result(id=\"{artifact_id}\") for the full output.\n\n"
        )
        return header + crush.text, artifact_id

    # Over inline limit but crush did not shrink — stub with preview, reuse store.
    preview_chars = DEFAULT_PREVIEW_CHARS
    preview = output[:preview_chars]
    omitted = max(0, len(output) - len(preview))
    aid = artifact_id or tool_use_id
    inline = (
        "[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Artifact id: {aid}\n"
        f"Original size: {len(output)} chars\n"
        f"Retrieve with: retrieve_tool_result(id=\"{aid}\")\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        inline += f" ({omitted} chars omitted)"
    if preview:
        inline += f"\n\nPreview:\n{preview}"
    return inline, artifact_id
