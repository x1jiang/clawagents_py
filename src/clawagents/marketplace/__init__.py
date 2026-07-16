"""Skill/plugin marketplace install channel."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

Kind = Literal["skill", "plugin"]


@dataclass(frozen=True)
class InstallResult:
    ok: bool
    kind: Kind
    name: str
    path: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "error": self.error,
        }


def marketplace_root(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or Path.cwd()).resolve() / ".clawagents" / "marketplace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def skills_install_dir(workspace: str | Path | None = None) -> Path:
    d = Path(workspace or Path.cwd()).resolve() / ".clawagents" / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plugins_install_dir(workspace: str | Path | None = None) -> Path:
    d = Path(workspace or Path.cwd()).resolve() / ".clawagents" / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:64] or "pkg"


def _is_git_url(source: str) -> bool:
    if source.startswith("git@"):
        return True
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https", "ssh", "git"):
        return source.endswith(".git") or "github.com" in source or "gitlab" in source
    return False


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _detect_kind(path: Path, kind: Kind | None) -> Kind:
    if kind:
        return kind
    if (path / "plugin.json").exists() or (path / ".claude-plugin" / "plugin.json").exists():
        return "plugin"
    if (path / "SKILL.md").exists() or list(path.glob("**/SKILL.md")):
        return "skill"
    # single markdown skill file
    if path.is_file() and path.suffix == ".md":
        return "skill"
    return "skill"


def _skill_name(path: Path) -> str:
    skill_md = path / "SKILL.md" if path.is_dir() else path
    if skill_md.is_file():
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end > 0:
                for line in text[3:end].splitlines():
                    if line.startswith("name:"):
                        return line.split(":", 1)[1].strip().strip("\"'")
        return skill_md.stem
    return path.name


def install_from_source(
    source: str,
    *,
    kind: Kind | None = None,
    workspace: str | Path | None = None,
    name: str | None = None,
) -> InstallResult:
    """Install a skill or plugin from a local path or git URL."""
    from clawagents.config.features import is_enabled

    if not is_enabled("marketplace"):
        return InstallResult(False, kind or "skill", "", "", "marketplace feature disabled")

    ws = Path(workspace or Path.cwd()).resolve()
    src_path: Path | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None

    try:
        if _is_git_url(source):
            tmp = tempfile.TemporaryDirectory(prefix="claw-mkt-")
            dest = Path(tmp.name) / "repo"
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", source, str(dest)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode != 0:
                return InstallResult(
                    False,
                    kind or "skill",
                    "",
                    "",
                    proc.stderr.strip() or "git clone failed",
                )
            src_path = dest
        else:
            src_path = Path(source).expanduser().resolve()
            if not src_path.exists():
                return InstallResult(False, kind or "skill", "", "", f"not found: {source}")

        resolved_kind = _detect_kind(src_path, kind)
        pkg_name = name or (
            _skill_name(src_path) if resolved_kind == "skill" else src_path.name
        )
        slug = _slug(pkg_name)

        if resolved_kind == "plugin":
            from clawagents.plugin_compat import load_plugin

            target = plugins_install_dir(ws) / slug
            if src_path.is_file():
                return InstallResult(False, "plugin", pkg_name, "", "plugin source must be a directory")
            _copy_tree(src_path, target)
            loaded = load_plugin(target)
            if loaded is None:
                return InstallResult(
                    False, "plugin", pkg_name, str(target), "invalid plugin.json"
                )
            _record_install(ws, resolved_kind, loaded.name, str(target), source)
            return InstallResult(True, "plugin", loaded.name, str(target))

        # skill
        target_dir = skills_install_dir(ws) / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        if src_path.is_file():
            dest_file = target_dir / "SKILL.md"
            shutil.copy2(src_path, dest_file)
        else:
            skill_md = src_path / "SKILL.md"
            if skill_md.exists():
                _copy_tree(src_path, target_dir)
            else:
                # copy first SKILL.md found
                found = next(src_path.rglob("SKILL.md"), None)
                if found is None:
                    return InstallResult(
                        False, "skill", pkg_name, "", "no SKILL.md in source"
                    )
                _copy_tree(found.parent, target_dir)
        _record_install(ws, "skill", pkg_name, str(target_dir), source)
        return InstallResult(True, "skill", pkg_name, str(target_dir))
    finally:
        if tmp is not None:
            tmp.cleanup()


def _record_install(
    workspace: Path,
    kind: Kind,
    name: str,
    path: str,
    source: str,
) -> None:
    root = marketplace_root(workspace)
    index = root / "installed.json"
    data: dict[str, Any] = {"packages": []}
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {"packages": []}
    packages = [
        p for p in data.get("packages", [])
        if not (p.get("kind") == kind and p.get("name") == name)
    ]
    packages.append({"kind": kind, "name": name, "path": path, "source": source})
    data["packages"] = packages
    index.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def list_installed(workspace: str | Path | None = None) -> list[dict[str, Any]]:
    index = marketplace_root(workspace) / "installed.json"
    if not index.is_file():
        return []
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
        return list(data.get("packages") or [])
    except (OSError, json.JSONDecodeError):
        return []


__all__ = [
    "InstallResult",
    "install_from_source",
    "list_installed",
    "marketplace_root",
    "skills_install_dir",
    "plugins_install_dir",
]
