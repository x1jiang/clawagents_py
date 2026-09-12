"""Offload large tool outputs to artifact files (OpenHarness 0.1.9 pattern).

Also stores reversible full text keyed by tool_use_id so the agent can call
``retrieve_tool_result`` after content crushing.

Security: body paths are always derived from sanitized artifact IDs under
``.clawagents/tool-artifacts/``. Metadata ``path`` fields are never trusted
for reads outside that directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from clawagents.memory.content_crush import (
    EVIDENCE_RECEIPT_PREFIX,
    CrushResult,
    build_failed_diagnostic_receipt,
    crush_tool_output,
    is_diagnostic_command,
)

DEFAULT_INLINE_CHARS = 12_000
DEFAULT_PREVIEW_CHARS = 2_000
DEFAULT_RECALL_CHARS = 16_000
DEFAULT_RECALL_LINES = 400


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
        return _path_under_dir(directory, derived)
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


def store_exec_artifact_from_spills(
    *,
    tool_use_id: str,
    stdout: str,
    stderr: str,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    workspace: str | Path | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[str, Path, int]:
    """Adopt complete command streams into a retrievable artifact.

    Spilled streams are copied incrementally, never loaded wholesale. Source
    spill files are removed after adoption. In-memory strings are used only
    for streams small enough not to have spilled.
    """
    directory = tool_artifact_dir(workspace)
    artifact_id = _safe_id(tool_use_id)
    body = _body_path(directory, artifact_id)
    if body.exists():
        artifact_id = f"{artifact_id}-{uuid.uuid4().hex[:8]}"
        body = _body_path(directory, artifact_id)

    chars = 0

    def _copy(source: str | Path | None, fallback: str, target: Any) -> int:
        written = 0
        if source is None:
            target.write(fallback)
            return len(fallback)
        with Path(source).open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                written += len(chunk)
        return written

    try:
        with body.open("w", encoding="utf-8", errors="replace") as target:
            chars += _copy(stdout_path, stdout, target)
            if stderr_path is not None or stderr:
                separator = "\n" if chars else ""
                marker = f"{separator}[stderr] "
                target.write(marker)
                chars += len(marker)
                chars += _copy(stderr_path, stderr, target)
    finally:
        for source in (stdout_path, stderr_path):
            if source is not None:
                try:
                    os.unlink(source)
                except FileNotFoundError:
                    pass

    meta = {
        "id": artifact_id,
        "tool_name": "execute",
        "tool_use_id": tool_use_id,
        "kind": "log",
        "chars": chars,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path": body.name,
        "complete_command_output": True,
    }
    if extra_meta:
        meta.update({k: v for k, v in extra_meta.items() if k != "path"})
    _meta_path(directory, artifact_id).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return artifact_id, body, chars


def _find_tool_artifact(
    artifact_id: str,
    *,
    workspace: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    directory = tool_artifact_dir(workspace)
    aid = _safe_id(artifact_id)
    meta_file = _meta_path(directory, aid)
    body = _body_path(directory, aid)
    meta: dict[str, Any] | None = None
    if meta_file.exists():
        try:
            value = json.loads(meta_file.read_text(encoding="utf-8"))
            meta = value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            meta = None

    if body.is_file():
        return _path_under_dir(directory, body), meta

    # Legacy / alternate ids — scan metas; body path must stay under directory.
    for candidate in directory.glob("*.meta.json"):
        try:
            m = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(m, dict) and (m.get("id") == artifact_id or m.get("tool_use_id") == artifact_id):
            path = _body_for_meta(directory, m, candidate)
            if path is not None:
                return path, m
    return None, None


def load_tool_artifact(
    artifact_id: str,
    *,
    workspace: str | Path | None = None,
    max_chars: int | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Backward-compatible full/capped loader; use the page helper for recall."""
    path, meta = _find_tool_artifact(artifact_id, workspace=workspace)
    if path is None:
        return False, f"No tool artifact found for id={artifact_id!r}", None
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            text = handle.read() if max_chars is None else handle.read(max(0, max_chars) + 1)
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return True, text, meta
    except OSError as exc:
        return False, f"Cannot read tool artifact: {exc}", meta


def load_tool_artifact_page(
    artifact_id: str,
    *,
    workspace: str | Path | None = None,
    offset: int | None = None,
    line_start: int | None = None,
    line_count: int = DEFAULT_RECALL_LINES,
    max_chars: int = DEFAULT_RECALL_CHARS,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Read bounded characters, preserving newlines and Unicode exactly.

    Offsets count Python Unicode characters, not UTF-8 bytes. Lines are 1-based
    and delimited by LF. A page may end inside an oversized line: next_offset
    always resumes at the first unread character. Prefix scans use bounded
    chunks rather than readline(), which could allocate a huge single line.
    """
    if offset is not None and line_start is not None:
        return False, "Use offset or line_start, not both", None
    if (offset is not None and offset < 0) or (line_start is not None and line_start < 1) or line_count < 1 or max_chars < 1:
        return False, "offset must be nonnegative; line_start, line_count and max_chars must be positive", None
    max_chars = min(max_chars, 500_000)
    line_count = min(line_count, 10_000)
    path, meta = _find_tool_artifact(artifact_id, workspace=workspace)
    if path is None:
        return False, f"No tool artifact found for id={artifact_id!r}", None
    position = 0
    skipped_lines = 0
    pending = ""
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            if line_start is not None:
                while skipped_lines < line_start - 1:
                    chunk = handle.read(8192)
                    if not chunk:
                        break
                    cut = 0
                    while skipped_lines < line_start - 1:
                        index = chunk.find("\n", cut)
                        if index < 0:
                            cut = len(chunk)
                            break
                        cut = index + 1
                        skipped_lines += 1
                    position += cut
                    pending = chunk[cut:]
            else:
                remaining = offset or 0
                while remaining:
                    chunk = handle.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    position += len(chunk)
                    skipped_lines += chunk.count("\n")
            sample = pending[:max_chars + 1]
            if len(sample) < max_chars + 1:
                sample += handle.read(max_chars + 1 - len(sample))
            end = min(len(sample), max_chars)
            cut = 0
            for _ in range(line_count):
                index = sample.find("\n", cut, end)
                if index < 0:
                    break
                cut = index + 1
            else:
                end = cut
            page = sample[:end]
            info = dict(meta or {})
            info.update(offset=position, next_offset=position + len(page), eof=(end == len(sample) and len(pending) <= len(sample)), line_start=skipped_lines + 1)
            return True, page, info
    except OSError as exc:
        return False, f"Cannot read tool artifact: {exc}", meta


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
# html included: large markup dumps still need a floor; prefer code sniff first
# for numbered sources that embed templates.
_CODEISH_KINDS = frozenset({"code", "log", "diff", "html"})
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
    command: str | None = None,
    efficiency: dict[str, Any] | None = None,
) -> tuple[str, Optional[str]]:
    """Prepare an optional compact view, preserving original output on error.

    Publish efficiency changes only after transformation succeeds. Nested
    receipt fallback counters are copied because reducers mutate them in place.
    """
    try:
        pending = dict(efficiency) if efficiency is not None else None
        if pending is not None and isinstance(pending.get("reducer_fallbacks"), dict):
            pending["reducer_fallbacks"] = dict(pending["reducer_fallbacks"])
        prepared = _prepare_tool_output_for_context(
            tool_name=tool_name, tool_use_id=tool_use_id, output=output,
            workspace=workspace, crush_threshold=crush_threshold,
            inline_limit=inline_limit, target_chars=target_chars,
            success=success, command=command, efficiency=pending,
        )
        if efficiency is not None and pending is not None:
            efficiency.update(pending)
        return prepared
    except Exception:
        logging.getLogger(__name__).debug(
            "Tool output transformation failed; preserving original output",
            exc_info=True,
        )
        return output, None


def _prepare_tool_output_for_context(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
    workspace: str | Path | None = None,
    crush_threshold: int | None = None,
    inline_limit: int | None = None,
    target_chars: int | None = None,
    success: bool | None = None,
    command: str | None = None,
    efficiency: dict[str, Any] | None = None,
) -> tuple[str, Optional[str]]:
    """Crush oversized outputs and store full text when crushed or huge.

    Returns ``(prompt_text, artifact_id_or_None)``.

    When ``CLAW_FEATURE_AGGRESSIVE_TOOL_CRUSH=1`` (default), uses tighter
    thresholds unless the caller overrides ``crush_threshold`` /
    ``inline_limit`` / ``target_chars``. Code/log/diff outputs use a higher
    floor (~4K) so edit tools are not fed compressed views.

    Control-plane tools (``use_skill``, ``list_skills``, ``retrieve_tool_result``)
    are never crushed — skill instructions must stay verbatim.

    Failed diagnostic commands may become verified, artifact-backed receipts.
    Other failures keep the existing verbatim/archive behavior.
    """
    if not isinstance(output, str):
        output = str(output)

    if tool_name in _CONTROL_PLANE_NO_CRUSH or EVIDENCE_RECEIPT_PREFIX in output:
        return output, None

    # Edit confirmations and harness status markers are never reduced. Only
    # the follow-up command's suffix is eligible for independent compression.
    fusion = re.search(r"(?m)^\[then_run:(succeeded|failed|skipped)\][^\n]*(?:\n|$)", output)
    if fusion and tool_name in {"edit_file", "apply_patch", "write_file", "hashline_edit"}:
        if fusion.group(1) == "skipped":
            return output, None
        prefix, suffix = output[:fusion.end()], output[fusion.end():]
        payload_prefix = ""
        payload_trailer = ""
        try:
            json_start = suffix.find("{")
            if json_start < 0:
                raise ValueError("No execute payload")
            payload, json_end = json.JSONDecoder().raw_decode(suffix[json_start:])
            payload_prefix = suffix[:json_start]
            payload_trailer = suffix[json_start + json_end:]
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("command_executed"):
            # Archive this decoded, exact-text view; JSON escape sequences do
            # not share the source line/character coordinates of stream text.
            parts = [payload_prefix, f"Command exited with code {payload.get('exit_code', '?')}"]
            for stream in ("stdout", "stderr"):
                if isinstance(payload.get(stream), str) and payload[stream]:
                    parts.append(f"{stream}:\n{payload[stream]}")
            for field in ("interpretation", "warning"):
                if isinstance(payload.get(field), str) and payload[field]:
                    parts.append(f"{field}: {payload[field]}")
            if payload_trailer:
                parts.append(payload_trailer)
            suffix = "\n".join(parts)
        reduced, aid = prepare_tool_output_for_context(
            tool_name="execute", tool_use_id=tool_use_id + "-then-run",
            output=suffix, workspace=workspace, crush_threshold=crush_threshold,
            inline_limit=inline_limit, target_chars=target_chars,
            success=fusion.group(1) == "succeeded", command=command, efficiency=efficiency,
        )
        # If no reduction occurred, preserve the original complete payload.
        return (prefix + reduced, aid) if aid else (output, None)

    # Prefer complete spilled command streams to an inline preview. Receipts
    # must point at the exact source whose hash and line numbers they report.
    source = output
    source_id: str | None = None
    diagnostic = success is False and tool_name in {"execute", "execute_command", "bash", "run_command"} and is_diagnostic_command(command)
    archive = re.search(r"\[Complete command output archived id=([\w.-]+); (\d+) chars\.", output) if diagnostic else None
    if archive:
        if int(archive.group(2)) > 600_000:
            diagnostic = False
        else:
            ok, full_source, _meta = load_tool_artifact(archive.group(1), workspace=workspace, max_chars=600_001)
            if (
                ok and len(full_source) <= 600_000 and _meta
                and _meta.get("complete_command_output") is True
                and _meta.get("command") == (command or "")[:1000]
                and _meta.get("chars") == len(full_source) == int(archive.group(2))
            ):
                source, source_id = full_source, archive.group(1)
            else:
                diagnostic = False

    # Only command-gated, bounded diagnostics are eligible for receipts.
    if diagnostic and 4096 <= len(source) <= 600_000:
        candidate_id = source_id or _safe_id(tool_use_id)
        receipt, reason = build_failed_diagnostic_receipt(
            source, artifact_id=candidate_id, command=command or "",
            target_chars=target_chars if target_chars is not None else 3500,
        )
        if receipt is not None and (len(receipt) >= len(output) or len(receipt.encode("utf-8")) >= len(output.encode("utf-8"))):
            receipt, reason = None, "receipt-not-smaller"
        if receipt is not None:
            try:
                artifact_id = source_id
                if artifact_id is None:
                    artifact_id, _path = store_tool_artifact(
                        tool_name=tool_name, tool_use_id=tool_use_id, output=source,
                        kind="log", workspace=workspace,
                        extra_meta={"did_crush": True, "evidence_receipt": EVIDENCE_RECEIPT_PREFIX},
                    )
                if artifact_id != candidate_id:
                    receipt, reason = build_failed_diagnostic_receipt(
                        source, artifact_id=artifact_id, command=command or "",
                        target_chars=target_chars if target_chars is not None else 3500,
                    )
            except OSError:
                receipt, reason = None, "artifact-write-failed"
            if receipt is not None:
                if efficiency is not None:
                    efficiency["reducer_bytes_saved"] = efficiency.get("reducer_bytes_saved", 0) + max(0, len(output.encode("utf-8")) - len(receipt.encode("utf-8")))
                return receipt, artifact_id
        if efficiency is not None:
            fallbacks = efficiency.setdefault("reducer_fallbacks", {})
            key = reason or "receipt-rejected"
            fallbacks[key] = fallbacks.get(key, 0) + 1

    # Non-diagnostic failures and rejected receipts retain the existing view.
    if success is False:
        hard_cap = 48_000
        if len(output) <= hard_cap:
            return output, None
        try:
            artifact_id, _path = store_tool_artifact(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                output=output,
                kind="prose",
                workspace=workspace,
                extra_meta={"did_crush": False, "failed_tool_verbatim": True},
            )
        except OSError:
            # A receipt/preview without recoverable evidence is unsafe.
            return output, None
        preview = output[:12_000]
        header = (
            f"[Failed tool output archived id={artifact_id}]\n"
            f"Original: {len(output)} chars (not crushed). "
            f"Call retrieve_tool_result(id=\"{artifact_id}\") for pages (16000 chars / 400 lines); continue with next_offset.\n\n"
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
    artifact_id = None
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
            f"Call retrieve_tool_result(id=\"{artifact_id}\") for pages (16000 chars / 400 lines); continue with next_offset.\n\n"
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
