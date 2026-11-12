"""Trial identity and crash-safe rollout completion markers (Harness² ``runmark``).

Completion is recorded explicitly by the code that observed a rollout return cleanly;
the mere presence of output files is never enough (a killed worker may leave partial
files behind). Each ``.adapt_complete`` marker records:

* ``harness_fingerprint`` — deterministic digest of the harness tree that produced
  the rollout (``"base"`` when running the unmodified base harness).
* ``inputs_fingerprint`` — optional deterministic digest of the model-visible task
  prompt and any task source files, so a cached rollout from an older version of the
  task is never reused.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MARK = ".adapt_complete"
RUNMARK_VERSION = 1


def _ignored_path(path: Path) -> bool:
    """Runtime byproducts that must never alter harness or task identity."""
    return (
        "__pycache__" in path.parts
        or ".pytest_cache" in path.parts
        or ".ruff_cache" in path.parts
        or path.suffix == ".pyc"
        or path.name in (".DS_Store", MARK)
    )


def tree_fingerprint(harness: Mapping[str, str] | Path | str | None) -> str:
    """Stable 16-hex SHA-256 digest of a harness tree or staged directory.

    Passing ``None`` represents the unmodified inherited base harness and digests
    to ``"base"``.
    """
    if harness is None:
        return "base"
    if isinstance(harness, Mapping):
        h = hashlib.sha256()
        for rel in sorted(harness):
            if _ignored_path(Path(rel)):
                continue
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(str(harness[rel]).encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    root = Path(harness)
    if not root.is_dir():
        return "absent"
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or _ignored_path(p.relative_to(root)):
            continue
        rel_posix = p.relative_to(root).as_posix()
        h.update(rel_posix.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()[:16]


def inputs_fingerprint(
    task_prompt: str,
    files_dir: Path | str | None = None,
) -> str:
    """Stable 16-hex SHA-256 digest of the model-visible task prompt and source files."""
    h = hashlib.sha256()
    h.update(b"prompt\0")
    h.update(str(task_prompt).encode("utf-8"))
    h.update(b"\0")
    if files_dir is not None:
        root = Path(files_dir)
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if not p.is_file() or _ignored_path(p.relative_to(root)):
                    continue
                h.update(p.relative_to(root).as_posix().encode("utf-8"))
                h.update(b"\0")
                try:
                    h.update(p.read_bytes())
                except OSError:
                    h.update(b"<unreadable>")
                h.update(b"\0")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class RunmarkRecord:
    """Parsed ``.adapt_complete`` marker payload."""

    version: int
    run_id: str
    harness_fingerprint: str
    completed_at: float
    inputs_fingerprint: str | None = None
    extra: Mapping[str, Any] | None = None


def write_runmark(
    run_dir: Path | str,
    *,
    run_id: str,
    harness: Mapping[str, str] | Path | str | None = None,
    inputs_digest: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically record that ``run_id`` finished cleanly in ``run_dir``."""
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": RUNMARK_VERSION,
        "run_id": run_id,
        "harness_fingerprint": tree_fingerprint(harness),
        "completed_at": time.time(),
    }
    if inputs_digest is not None:
        payload["inputs_fingerprint"] = inputs_digest
    if extra:
        payload["extra"] = dict(extra)

    marker = target_dir / MARK
    tmp = target_dir / f"{MARK}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(marker)
    return marker


def read_runmark(run_dir: Path | str) -> RunmarkRecord | None:
    """Read ``.adapt_complete`` from ``run_dir``, or return ``None`` if absent/corrupt."""
    marker = Path(run_dir) / MARK
    if not marker.is_file():
        return None
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    run_id = raw.get("run_id")
    h_fp = raw.get("harness_fingerprint")
    if not isinstance(run_id, str) or not isinstance(h_fp, str):
        return None
    in_fp = raw.get("inputs_fingerprint")
    extra = raw.get("extra")
    return RunmarkRecord(
        version=int(raw.get("version") or RUNMARK_VERSION),
        run_id=run_id,
        harness_fingerprint=h_fp,
        completed_at=float(raw.get("completed_at") or 0.0),
        inputs_fingerprint=str(in_fp) if isinstance(in_fp, str) else None,
        extra=dict(extra) if isinstance(extra, dict) else None,
    )


def is_runmark_valid(
    run_dir: Path | str,
    *,
    expected_harness: Mapping[str, str] | Path | str | None = None,
    expected_inputs_digest: str | None = None,
) -> bool:
    """Return True iff ``run_dir`` finished cleanly with matching harness & task digests."""
    rec = read_runmark(run_dir)
    if rec is None:
        return False
    want_harness = tree_fingerprint(expected_harness)
    if rec.harness_fingerprint != want_harness:
        return False
    if expected_inputs_digest is not None and rec.inputs_fingerprint != expected_inputs_digest:
        return False
    return True


def prepare_clean_run_dir(
    run_dir: Path | str,
    *,
    expected_harness: Mapping[str, str] | Path | str | None = None,
    expected_inputs_digest: str | None = None,
) -> bool:
    """Return True if ``run_dir`` has a valid completed rollout ready to reuse.

    When ``run_dir`` exists without a valid marker (e.g., partial output from a killed
    worker or stale run from an edited harness/task), wipes and recreates ``run_dir``
    so the fresh rollout starts from a clean slate, then returns False.
    """
    target = Path(run_dir)
    if is_runmark_valid(
        target,
        expected_harness=expected_harness,
        expected_inputs_digest=expected_inputs_digest,
    ):
        return True
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return False


__all__ = [
    "MARK",
    "RUNMARK_VERSION",
    "RunmarkRecord",
    "inputs_fingerprint",
    "is_runmark_valid",
    "prepare_clean_run_dir",
    "read_runmark",
    "tree_fingerprint",
    "write_runmark",
]
