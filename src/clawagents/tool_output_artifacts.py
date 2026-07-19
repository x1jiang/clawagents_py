"""Offload large tool outputs to artifact files (OpenHarness 0.1.9 pattern).

Also stores reversible full text keyed by tool_use_id so the agent can call
``retrieve_tool_result`` after content crushing.

Security: body paths are always derived from sanitized artifact IDs under
``.clawagents/tool-artifacts/``. Metadata ``path`` fields are never trusted
for reads outside that directory.
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


def _path_under_dir(directory: Path, path: Path) -> Path | None:
    """Return resolved ``path`` only if it is a file inside ``directory``."""
    try:
        root = directory.resolve()
        resolved = path.expanduser().resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    return resolved


def _body_for_meta(directory: Path, meta: dict[str, Any], meta_file: Path) -> Path | None:
    """Resolve a readable body path for meta — ID-derived first, never escape dir."""
    aid = _safe_id(str(meta.get("id") or meta_file.stem.replace(".meta", "")))
    derived = _body_path(directory, aid)
    if derived.is_file():
        return derived
    # Legacy absolute/relative path in meta — only if contained in the artifact dir.
    raw = str(meta.get("path") or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = directory / candidate
    return _path_under_dir(directory, candidate)


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
        # Relative name only — loaders never follow absolute paths outside dir.
        "path": body.name,
    }
    if extra_meta:
        # Drop hostile path overrides from callers.
        extra = {k: v for k, v in extra_meta.items() if k != "path"}
        meta.update(extra)
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

    def _read(path: Path, m: dict[str, Any] | None) -> tuple[bool, str, dict[str, Any] | None]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return True, text, m

    if body.is_file():
        return _read(body, meta)

    # Legacy / alternate ids — scan metas; body path must stay under directory.
    for candidate in directory.glob("*.meta.json"):
        try:
            m = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("id") == artifact_id or m.get("tool_use_id") == artifact_id:
            path = _body_for_meta(directory, m, candidate)
            if path is not None:
                return _read(path, m)
    return False, f"No tool artifact found for id={artifact_id!r}", None


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
        f"Full output saved to: {artifact_path.name}\n"
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
        body_path = _body_for_meta(directory, meta, meta_file)
        snippet = ""
        hay = f"{meta.get('tool_name', '')} {meta.get('id', '')} {meta.get('kind', '')}".lower()
        matched = q in hay
        if body_path is not None:
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


# Aggressive in-loop crush (feature ``aggressive_tool_crush``) — tighter than
# the default Headroom-inspired thresholds so large tool dumps never linger
# in the prompt waiting for a hook or model-chosen ctx_* call.
_AGGRESSIVE_CRUSH_THRESHOLD = 1_200
_AGGRESSIVE_TARGET_CHARS = 2_000
_AGGRESSIVE_INLINE_LIMIT = 6_000
# Exact-match edits (apply_patch / hashline) need verbatim code/log views.
# Crushing 2.5K→2.0K is all risk; keep a higher floor for those kinds.
_CODEISH_CRUSH_FLOOR = 4_000
_CODEISH_KINDS = frozenset({"code", "log", "diff"})
# Skill pages / catalogs / archived restores are control-plane: the model must
# never operate on a crushed fraction of its instructions (auto-drain pages
# also flow through this path since v6.20.15).
_CONTROL_PLANE_NO_CRUSH = frozenset({
    "use_skill",
    "list_skills",
    "retrieve_tool_result",
})


def prepare_tool_output_for_context(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
    workspace: str | Path | None = None,
    crush_threshold: int | None = None,
    inline_limit: int | None = None,
    target_chars: int | None = None,
    success: bool | None = None,
) -> tuple[str, Optional[str]]:
    """Crush oversized outputs and store full text when crushed or huge.

    Returns ``(prompt_text, artifact_id_or_None)``.

    When ``CLAW_FEATURE_AGGRESSIVE_TOOL_CRUSH=1`` (default), uses tighter
    thresholds unless the caller overrides ``crush_threshold`` /
    ``inline_limit`` / ``target_chars``. Code/log/diff outputs use a higher
    floor (~4K) so edit tools are not fed compressed views.

    Control-plane tools (``use_skill``, ``list_skills``, ``retrieve_tool_result``)
    are never crushed — skill instructions must stay verbatim.

    Failed tool results (``success=False``) are never aggressively crushed —
    denial paths (e.g. credentials.db EPERM) must stay verbatim for diagnosis.
    """
    if not isinstance(output, str):
        output = str(output)

    if tool_name in _CONTROL_PLANE_NO_CRUSH:
        return output, None

    # Failures: keep full text in context (still archive if enormous).
    if success is False:
        hard_cap = 48_000
        if len(output) <= hard_cap:
            return output, None
        artifact_id, _path = store_tool_artifact(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            output=output,
            kind="prose",
            workspace=workspace,
            extra_meta={"did_crush": False, "failed_tool_verbatim": True},
        )
        preview = output[:12_000]
        header = (
            f"[Failed tool output archived id={artifact_id}]\n"
            f"Original: {len(output)} chars (not crushed). "
            f"Call retrieve_tool_result(id=\"{artifact_id}\") for the remainder.\n\n"
        )
        return header + preview, artifact_id

    thresh = crush_threshold
    inline = inline_limit
    target = target_chars
    try:
        from clawagents.config.features import is_enabled
        from clawagents.memory.content_crush import detect_content_kind

        kind = detect_content_kind(output, tool_name=tool_name)
        if is_enabled("aggressive_tool_crush"):
            if thresh is None:
                thresh = _AGGRESSIVE_CRUSH_THRESHOLD
            if inline is None:
                inline = _AGGRESSIVE_INLINE_LIMIT
            if target is None:
                target = _AGGRESSIVE_TARGET_CHARS
        if kind in _CODEISH_KINDS and thresh is not None:
            thresh = max(thresh, _CODEISH_CRUSH_FLOOR)
        if kind in _CODEISH_KINDS and target is not None:
            target = max(target, _CODEISH_CRUSH_FLOOR)
    except Exception:
        pass
    if thresh is None:
        thresh = 2_000
    if inline is None:
        inline = DEFAULT_INLINE_CHARS
    if target is None:
        target = 3_500

    crush: CrushResult = crush_tool_output(
        output,
        tool_name=tool_name,
        threshold=thresh,
        target_chars=target,
    )

    # Always store when we crushed or when still over inline limit.
    need_store = crush.did_crush or len(output) > inline
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

    if not crush.did_crush and len(output) <= inline:
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
    stub = (
        "[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Artifact id: {aid}\n"
        f"Original size: {len(output)} chars\n"
        f"Retrieve with: retrieve_tool_result(id=\"{aid}\")\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        stub += f" ({omitted} chars omitted)"
    if preview:
        stub += f"\n\nPreview:\n{preview}"
    return stub, artifact_id
