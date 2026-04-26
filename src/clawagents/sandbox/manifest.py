from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SandboxManifestEntry:
    name: str
    type: str
    target: str
    source: str | None = None
    repo: str | None = None
    ref: str | None = None
    read_only: bool = False


@dataclass(frozen=True)
class SandboxManifest:
    entries: list[SandboxManifestEntry]
    env: dict[str, str]
    workdir: str | None = None


def _default_target(name: str, entry: Mapping[str, Any]) -> str:
    if entry.get("target"):
        return str(entry["target"])
    if entry.get("type") == "git":
        repo = str(entry.get("repo", ""))
        tail = repo.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        return tail or name
    return name


def _normalize_entry(name: str, raw: Mapping[str, Any]) -> SandboxManifestEntry:
    entry_type = str(raw.get("type", ""))
    if not name.strip():
        raise ValueError("Sandbox manifest entry name is required")
    if entry_type == "path":
        source = str(raw.get("source", ""))
        if not source.strip():
            raise ValueError(f"Sandbox manifest path entry '{name}' requires a source")
        return SandboxManifestEntry(
            name=name,
            type="path",
            source=source,
            target=_default_target(name, raw),
            read_only=bool(raw.get("read_only", raw.get("readOnly", False))),
        )
    if entry_type == "git":
        repo = str(raw.get("repo", ""))
        if not repo.strip():
            raise ValueError(f"Sandbox manifest git entry '{name}' requires a repo")
        return SandboxManifestEntry(
            name=name,
            type="git",
            repo=repo,
            ref=str(raw["ref"]) if raw.get("ref") is not None else None,
            target=_default_target(name, raw),
        )
    raise ValueError(f"Sandbox manifest entry '{name}' has unsupported type: {entry_type!r}")


def normalize_sandbox_manifest(raw: Mapping[str, Any] | None = None) -> SandboxManifest:
    data = raw or {}
    raw_entries = data.get("entries") or {}
    if isinstance(raw_entries, Mapping):
        entries = [_normalize_entry(str(name), entry) for name, entry in raw_entries.items()]
    elif isinstance(raw_entries, list):
        entries = [_normalize_entry(str(idx), entry) for idx, entry in enumerate(raw_entries)]
    else:
        raise ValueError("Sandbox manifest entries must be an object or list")

    return SandboxManifest(
        entries=entries,
        env={str(k): str(v) for k, v in dict(data.get("env") or {}).items()},
        workdir=str(data["workdir"]) if data.get("workdir") is not None else None,
    )
