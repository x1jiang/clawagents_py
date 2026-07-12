"""Custom agent modes (Kilo-inspired) — persona + tool gates from JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from clawagents.graph.agent_loop import BeforeToolHook, HookResult
from clawagents.permissions.mode import PermissionMode


@dataclass
class AgentMode:
    id: str
    name: str = ""
    instruction: str = ""
    tool_allowlist: list[str] = field(default_factory=list)
    tool_denylist: list[str] = field(default_factory=list)
    permission_mode: str = "DEFAULT"
    auto_approve: bool = False

    def display_name(self) -> str:
        return self.name or self.id


_BUILTIN_MODES: dict[str, AgentMode] = {
    "ask": AgentMode(
        id="ask",
        name="Ask",
        instruction=(
            "Prefer explaining and answering questions. Avoid editing files or "
            "running destructive commands unless the user explicitly asks."
        ),
        tool_denylist=["write_file", "edit_file", "apply_patch", "execute", "git_commit"],
        permission_mode="PLAN",
    ),
    "architect": AgentMode(
        id="architect",
        name="Architect",
        instruction=(
            "Focus on design, plans, and trade-offs. Prefer read-only exploration "
            "and write_plan. Do not implement large changes until asked."
        ),
        tool_allowlist=[
            "read_file", "list_dir", "grep", "glob", "repo_map", "write_plan",
            "web_search", "ask_user",
        ],
        permission_mode="PLAN",
    ),
    "code": AgentMode(
        id="code",
        name="Code",
        instruction=(
            "Implement changes carefully with tests when possible. Use apply_patch "
            "or edit_file for surgical edits; run verify commands after edits."
        ),
        permission_mode="DEFAULT",
    ),
    "ci": AgentMode(
        id="ci",
        name="CI",
        instruction=(
            "Headless CI agent: complete the task with minimal chatter. Prefer "
            "deterministic tools; do not ask the user questions."
        ),
        permission_mode="BYPASS",
        auto_approve=True,
    ),
}


def _parse_mode(raw: dict[str, Any], fallback_id: str = "") -> AgentMode | None:
    mid = str(raw.get("id") or fallback_id or "").strip()
    if not mid:
        return None
    allow = raw.get("tool_allowlist") or raw.get("allowlist") or []
    deny = raw.get("tool_denylist") or raw.get("denylist") or []
    if not isinstance(allow, list):
        allow = []
    if not isinstance(deny, list):
        deny = []
    pm = str(raw.get("permission_mode") or "DEFAULT").strip().upper()
    if pm not in {m.name for m in PermissionMode}:
        pm = "DEFAULT"
    return AgentMode(
        id=mid,
        name=str(raw.get("name") or mid),
        instruction=str(raw.get("instruction") or ""),
        tool_allowlist=[str(x) for x in allow],
        tool_denylist=[str(x) for x in deny],
        permission_mode=pm,
        auto_approve=bool(raw.get("auto_approve") or False),
    )


def _load_modes_file(path: Path) -> dict[str, AgentMode]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, AgentMode] = {}
    if isinstance(data, dict) and isinstance(data.get("modes"), list):
        items = data["modes"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [
            {**v, "id": k} if isinstance(v, dict) else v
            for k, v in data.items()
            if k != "modes"
        ]
    else:
        return {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mode = _parse_mode(item)
        if mode:
            out[mode.id] = mode
    return out


def load_modes(workspace: str | Path | None = None) -> dict[str, AgentMode]:
    """Merge builtins < ~/.clawagents/modes.json < workspace .clawagents/modes.json."""
    modes = dict(_BUILTIN_MODES)
    home = Path.home() / ".clawagents" / "modes.json"
    modes.update(_load_modes_file(home))
    ws = Path(workspace or os.getcwd())
    modes.update(_load_modes_file(ws / ".clawagents" / "modes.json"))
    return modes


def get_mode(mode_id: str, *, workspace: str | Path | None = None) -> AgentMode | None:
    if not mode_id:
        return None
    return load_modes(workspace).get(mode_id)


def resolve_permission_mode(mode: AgentMode) -> PermissionMode:
    try:
        return PermissionMode[mode.permission_mode]
    except KeyError:
        return PermissionMode.DEFAULT


def make_mode_before_tool(mode: AgentMode) -> BeforeToolHook:
    allow = {n.strip() for n in mode.tool_allowlist if n and str(n).strip()}
    deny = {n.strip() for n in mode.tool_denylist if n and str(n).strip()}

    def _hook(tool_name: str, args: dict[str, Any]) -> HookResult:
        if deny and tool_name in deny:
            return HookResult(
                allowed=False,
                reason=f"mode '{mode.id}' denylist blocks tool '{tool_name}'",
            )
        if allow and tool_name not in allow:
            return HookResult(
                allowed=False,
                reason=f"mode '{mode.id}' allowlist excludes tool '{tool_name}'",
            )
        return HookResult(allowed=True)

    return _hook


def compose_before_tool(
    *hooks: Optional[BeforeToolHook],
) -> Optional[BeforeToolHook]:
    active = [h for h in hooks if h is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _composed(tool_name: str, args: dict[str, Any]) -> HookResult:
        cur_args = args
        for h in active:
            result = h(tool_name, cur_args)
            if isinstance(result, bool):
                if not result:
                    return HookResult(allowed=False, reason="blocked by before_tool")
                continue
            if isinstance(result, HookResult):
                if not result.allowed:
                    return result
                if result.updated_args is not None:
                    cur_args = result.updated_args
        return HookResult(allowed=True, updated_args=cur_args)

    return _composed
