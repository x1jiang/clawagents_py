"""Disk plugin compatibility loader for Claude/OpenHarness-style plugins."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PluginSkill:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class PluginCommand:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class LoadedCompatPlugin:
    name: str
    description: str
    path: Path
    skills: list[PluginSkill] = field(default_factory=list)
    commands: list[PluginCommand] = field(default_factory=list)
    hooks: dict[str, Any] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)


def load_plugin(path: str | Path) -> LoadedCompatPlugin | None:
    root = Path(path)
    manifest_path = _find_manifest(root)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None

    name = str(manifest.get("name") or root.name)
    description = str(manifest.get("description") or "")
    skills = _load_skills(root / str(manifest.get("skills_dir") or "skills"))
    commands = _load_commands(root / str(manifest.get("commands_dir") or "commands"))
    hooks = _load_json_object(root / str(manifest.get("hooks_file") or "hooks.json"))
    mcp_raw = _load_json_object(root / str(manifest.get("mcp_file") or ".mcp.json"))
    mcp_servers = mcp_raw.get("servers", mcp_raw) if isinstance(mcp_raw, dict) else {}
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    return LoadedCompatPlugin(name, description, root, skills, commands, hooks, mcp_servers)


def discover_plugins(root: str | Path) -> list[LoadedCompatPlugin]:
    base = Path(root)
    if not base.exists():
        return []
    plugins: list[LoadedCompatPlugin] = []
    for child in sorted(base.iterdir()):
        if child.is_dir():
            plugin = load_plugin(child)
            if plugin is not None:
                plugins.append(plugin)
    return plugins


def _find_manifest(root: Path) -> Path | None:
    for candidate in (root / "plugin.json", root / ".claude-plugin" / "plugin.json"):
        if candidate.exists():
            return candidate
    return None


def _frontmatter_value(text: str, key: str) -> str:
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    if end < 0:
        return ""
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def _load_skills(root: Path) -> list[PluginSkill]:
    if not root.exists():
        return []
    out: list[PluginSkill] = []
    for file in sorted(root.rglob("*.md")):
        text = file.read_text(encoding="utf-8", errors="replace")
        name = _frontmatter_value(text, "name") or (file.parent.name if file.name == "SKILL.md" else file.stem)
        desc = _frontmatter_value(text, "description")
        out.append(PluginSkill(name, desc, file))
    return out


def _load_commands(root: Path) -> list[PluginCommand]:
    if not root.exists():
        return []
    out: list[PluginCommand] = []
    for file in sorted(root.rglob("*.md")):
        first = ""
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                first = line.strip().lstrip("#").strip()
                break
        out.append(PluginCommand(file.stem, first, file))
    return out


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

