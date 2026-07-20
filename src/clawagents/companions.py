"""Companion CLI version floors and probes (context-mode, rtk, graphify).

Keep floors in lockstep with clawagents_vscode/src/companionDeps.ts.
Bump these when shipping a clawagents release that requires newer companions.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

VersionTuple = Tuple[int, int, int]

# npm context-mode — https://www.npmjs.com/package/context-mode
MIN_CONTEXT_MODE: VersionTuple = (1, 0, 169)
# Homebrew / cargo rtk (Rust Token Killer) — https://www.rtk-ai.app/
MIN_RTK: VersionTuple = (0, 43, 0)
# PyPI graphifyy (import graphify) — https://github.com/Graphify-Labs/graphify
MIN_GRAPHIFY: VersionTuple = (0, 9, 20)

CONTEXT_MODE_BINARY = "context-mode"
RTK_BINARY = "rtk"
GRAPHIFY_PACKAGE = "graphifyy"
GRAPHIFY_BINARY = "graphify"

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True)
class CompanionStatus:
    name: str
    found: bool
    version: Optional[str]
    min_version: str
    ok_vs_floor: bool
    hint: str
    path: Optional[str] = None

    def summary(self) -> str:
        if not self.found:
            return f"{self.name}: missing — {self.hint}"
        ver = self.version or "?"
        status = "ok" if self.ok_vs_floor else "below floor"
        where = f" @ {self.path}" if self.path else ""
        return f"{self.name}: {ver} ({status}, need >={self.min_version}){where}"


def parse_version(text: str | None) -> Optional[VersionTuple]:
    """Extract the first x.y.z from a version string."""
    if not text:
        return None
    m = _VERSION_RE.search(text.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def version_at_least(version: str | VersionTuple | None, minimum: VersionTuple) -> bool:
    if isinstance(version, tuple):
        parts = version
    else:
        parts = parse_version(version)
    if parts is None:
        return False
    return parts >= minimum


def format_version(v: VersionTuple) -> str:
    return f"{v[0]}.{v[1]}.{v[2]}"


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _walk_package_json(start: Path, *, max_up: int = 8) -> Optional[Path]:
    cur = start if start.is_dir() else start.parent
    for _ in range(max_up):
        candidate = cur / "package.json"
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def context_mode_version(binary: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Return (version, path) for context-mode, or (None, None) if missing."""
    path = binary or _which(CONTEXT_MODE_BINARY)
    if not path:
        return None, None
    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(path)
    pkg = _walk_package_json(resolved)
    if pkg is None:
        # Some installs point at a shim; try npm list as a last resort.
        ver = _npm_global_version("context-mode")
        return ver, path
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, path
    ver = data.get("version")
    if isinstance(ver, str) and ver.strip():
        return ver.strip(), path
    return None, path


def _npm_global_version(package: str) -> Optional[str]:
    npm = _which("npm")
    if not npm:
        return None
    try:
        proc = subprocess.run(
            [npm, "list", "-g", package, "--depth=0", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    deps = data.get("dependencies") or {}
    entry = deps.get(package) or {}
    ver = entry.get("version")
    return ver.strip() if isinstance(ver, str) else None


def rtk_version(binary: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Return (version, path) for rtk, or (None, None) if missing."""
    path = binary or os.environ.get("CLAW_RTK_BIN") or _which(RTK_BINARY)
    if not path:
        return None, None
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, path
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_version(text)
    if parsed is None:
        return None, path
    return format_version(parsed), path


def probe_context_mode(
    *,
    min_version: VersionTuple = MIN_CONTEXT_MODE,
) -> CompanionStatus:
    ver, path = context_mode_version()
    found = path is not None
    ok = found and version_at_least(ver, min_version)
    if not found:
        hint = "npm install -g context-mode@latest  (Node ≥ 22.5)"
    elif not ok:
        hint = (
            f"upgrade: npm install -g context-mode@latest  "
            f"(have {ver or '?'}, need >={format_version(min_version)})"
        )
    else:
        hint = "ok"
    return CompanionStatus(
        name="context-mode",
        found=found,
        version=ver,
        min_version=format_version(min_version),
        ok_vs_floor=bool(ok),
        hint=hint,
        path=path,
    )


def probe_rtk(*, min_version: VersionTuple = MIN_RTK) -> CompanionStatus:
    ver, path = rtk_version()
    found = path is not None
    ok = found and version_at_least(ver, min_version)
    if not found:
        hint = "brew install rtk  (or see https://www.rtk-ai.app/)"
    elif not ok:
        hint = (
            f"upgrade: brew upgrade rtk  "
            f"(have {ver or '?'}, need >={format_version(min_version)})"
        )
    else:
        hint = "ok"
    return CompanionStatus(
        name="rtk",
        found=found,
        version=ver,
        min_version=format_version(min_version),
        ok_vs_floor=bool(ok),
        hint=hint,
        path=path,
    )


def graphify_version(
    python: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return (version, path) for graphifyy in *python* (default: this interpreter).

    Path is the graphify console script when present, else the python used to
    import the package (``python -m graphify``).
    """
    py = python or sys.executable
    # Prefer importlib against the current interpreter when probing self.
    if py == sys.executable or Path(py).resolve() == Path(sys.executable).resolve():
        try:
            ver = importlib.metadata.version(GRAPHIFY_PACKAGE)
        except importlib.metadata.PackageNotFoundError:
            ver = None
        if ver:
            path = _which(GRAPHIFY_BINARY) or py
            return ver.strip(), path
        # Module may be importable under a different dist name in some installs.
        try:
            import graphify  # noqa: F401

            ver2 = getattr(graphify, "__version__", None)
            if isinstance(ver2, str) and ver2.strip():
                return ver2.strip(), _which(GRAPHIFY_BINARY) or py
        except ImportError:
            pass
        return None, None

    try:
        proc = subprocess.run(
            [
                py,
                "-c",
                (
                    "import importlib.metadata as m\n"
                    f"print(m.version({GRAPHIFY_PACKAGE!r}))\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if proc.returncode != 0:
        return None, None
    text = (proc.stdout or "").strip().splitlines()
    ver = text[0].strip() if text else None
    if not ver:
        return None, None
    return ver, py


def probe_graphify(
    *,
    min_version: VersionTuple = MIN_GRAPHIFY,
    python: Optional[str] = None,
) -> CompanionStatus:
    ver, path = graphify_version(python)
    found = ver is not None
    ok = found and version_at_least(ver, min_version)
    if not found:
        hint = "pip install 'graphifyy[mcp]'  (https://github.com/Graphify-Labs/graphify)"
    elif not ok:
        hint = (
            f"upgrade: pip install -U 'graphifyy[mcp]'  "
            f"(have {ver or '?'}, need >={format_version(min_version)})"
        )
    else:
        hint = "ok"
    return CompanionStatus(
        name="graphify",
        found=found,
        version=ver,
        min_version=format_version(min_version),
        ok_vs_floor=bool(ok),
        hint=hint,
        path=path,
    )


def probe_companions(
    *,
    names: Optional[Sequence[str]] = None,
) -> list[CompanionStatus]:
    """Probe companion CLIs. Default: context-mode + rtk + graphify."""
    wanted = set(names) if names else {"context-mode", "rtk", "graphify"}
    out: list[CompanionStatus] = []
    if "context-mode" in wanted:
        out.append(probe_context_mode())
    if "rtk" in wanted:
        out.append(probe_rtk())
    if "graphify" in wanted:
        out.append(probe_graphify())
    return out


def companions_ok(statuses: Optional[Iterable[CompanionStatus]] = None) -> bool:
    items = list(statuses) if statuses is not None else probe_companions()
    return all(s.ok_vs_floor for s in items)


__all__ = [
    "GRAPHIFY_PACKAGE",
    "MIN_CONTEXT_MODE",
    "MIN_GRAPHIFY",
    "MIN_RTK",
    "CompanionStatus",
    "companions_ok",
    "context_mode_version",
    "format_version",
    "graphify_version",
    "parse_version",
    "probe_companions",
    "probe_context_mode",
    "probe_graphify",
    "probe_rtk",
    "rtk_version",
    "version_at_least",
]
