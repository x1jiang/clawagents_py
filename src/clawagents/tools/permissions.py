"""Granular tool permission rules — Grok-style deny-wins evaluation.

Usage:
    from clawagents.tools.permissions import PermissionEngine, PermissionRule, load_permission_engine

    engine = load_permission_engine(workspace)
    # before_tool / registry gate
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, List


@dataclass
class PermissionRule:
    """A single permission rule for a tool.

    Attributes:
        tool: Glob pattern for tool name (e.g., "execute*", "write_file", "Bash(*)")
        path_pattern: Glob for file paths in tool args
        arg_pattern: Glob matched against JSON-serialized args
        decision: "allow" | "deny" | "ask"
        message: Optional message when this rule triggers
        priority: Higher priority sorts first within the same decision class
    """

    tool: str = "*"
    path_pattern: str = "*"
    arg_pattern: str = "*"
    decision: str = "allow"
    message: str = ""
    priority: int = 0


_DECISION_RANK = {"deny": 3, "ask": 2, "allow": 1}

# Filesystem writers that share one security class with write_file/edit_file.
# Mirrors the file-writing members of
# ``clawagents.permissions.mode.WRITE_CLASS_TOOLS`` (kept as a local literal to
# avoid an import cycle; the test suite asserts the two stay in sync). Execute,
# git and subagent tools are deliberately excluded — they are gated elsewhere.
_FS_WRITE_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "apply_patch",
    "hashline_edit",
    "create_file",
    "replace_in_file",
    "insert_in_file",
    "insert_lines",
    "patch_file",
    "delete_file",
})


class PermissionEngine:
    """Declarative permission engine.

    Matching rules are collected; **deny wins** over ask over allow (Grok
    parity). Among the winning decision class, highest priority message is kept.
    """

    def __init__(self, default_decision: str = "allow"):
        self._rules: List[PermissionRule] = []
        self._default = default_decision
        # Optional host callback for "ask" decisions:
        # ``(tool_name, args, message) -> bool`` (True = approved).
        # When unset, ask is fail-closed (denied) — same as gate() historically.
        self.ask_handler: Any = None

    def add_rule(self, rule: PermissionRule) -> "PermissionEngine":
        self._rules.append(rule)
        return self

    def add_rules(self, rules: List[PermissionRule]) -> "PermissionEngine":
        self._rules.extend(rules)
        return self

    def _extract_path(self, args: dict[str, Any]) -> str:
        return str(
            args.get("path")
            or args.get("file_path")
            or args.get("target_path")
            or args.get("filePath")
            or ""
        )

    def _tool_aliases(self, tool_name: str) -> list[str]:
        """Accept Grok-style Bash(git *) patterns as execute aliases."""
        names = [tool_name]
        if tool_name in {"execute", "bash", "shell"}:
            names.extend(["execute", "Bash", "bash", "shell"])
        # Every filesystem writer is one security class: a rule written against
        # the common ``write_file``/``edit_file`` names (including the default
        # secret-path deny/ask rules) must also gate the newer writers
        # (hashline_edit, create_file, delete_file, …). Without this, those
        # tools fall through to the ``allow`` default and can write ``.env`` /
        # ``**/credentials*`` that write_file/edit_file are denied from touching.
        if tool_name in _FS_WRITE_TOOLS:
            names.extend(["Edit", "Write", "edit_file", "write_file"])
        return names

    def _rule_matches(self, rule: PermissionRule, tool_name: str, args: dict[str, Any]) -> bool:
        aliases = self._tool_aliases(tool_name)
        tool_ok = any(fnmatch(alias, rule.tool) or fnmatch(tool_name, rule.tool) for alias in aliases)
        # Also allow rule.tool like "Bash(git *)" → treat outer as execute + arg pattern
        if not tool_ok and rule.tool.startswith("Bash(") and rule.tool.endswith(")"):
            inner = rule.tool[5:-1].strip()
            if tool_name in {"execute", "bash", "shell"}:
                cmd = str(args.get("command") or "")
                tool_ok = fnmatch(cmd, inner) or fnmatch(cmd, inner.replace(" ", "*"))
        if not tool_ok:
            return False

        file_path = self._extract_path(args)
        if rule.path_pattern != "*" and file_path:
            if not (
                fnmatch(file_path, rule.path_pattern)
                or fnmatch(os.path.basename(file_path), rule.path_pattern)
            ):
                return False
        elif rule.path_pattern != "*" and not file_path:
            return False

        if rule.arg_pattern != "*":
            args_str = json.dumps(args, default=str)
            if not fnmatch(args_str, rule.arg_pattern):
                return False
        return True

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        matches: list[PermissionRule] = [
            rule for rule in self._rules if self._rule_matches(rule, tool_name, args)
        ]
        if not matches:
            return self._default, ""

        # Deny wins, then ask, then allow; within class, highest priority.
        matches.sort(
            key=lambda r: (_DECISION_RANK.get(r.decision, 0), r.priority),
            reverse=True,
        )
        winner = matches[0]
        return winner.decision, winner.message

    def check(self, tool_name: str, args: dict[str, Any]) -> bool:
        """before_tool-compatible: True for allow, or ask approved by ask_handler."""
        ok, _ = self.gate(tool_name, args)
        return ok

    def gate(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Registry gate: (allowed, error_message).

        ``ask`` invokes ``ask_handler`` when set (host approval UI); otherwise
        fail-closed as deny so scripts without a host stay safe.
        """
        decision, message = self.evaluate(tool_name, args)
        if decision == "allow":
            return True, ""
        if decision == "ask":
            handler = self.ask_handler
            if callable(handler):
                try:
                    if bool(handler(tool_name, args if isinstance(args, dict) else {}, message or "")):
                        return True, ""
                except Exception:
                    pass
            return False, message or f"Permission ask required for {tool_name}"
        return False, message or f"Denied by permission rule: {tool_name}"

    @classmethod
    def from_config(cls, rules_data: list[dict[str, Any]]) -> "PermissionEngine":
        engine = cls()
        for data in rules_data:
            engine.add_rule(PermissionRule(**{k: v for k, v in data.items() if k in PermissionRule.__dataclass_fields__}))
        return engine


def _build_default_secure_rules() -> list[PermissionRule]:
    """Secure defaults — path patterns from ``clawagents.security.secret_paths``."""
    from clawagents.security.secret_paths import default_secure_path_rules

    rules: list[PermissionRule] = [
        PermissionRule(
            tool="execute",
            arg_pattern="*rm -rf /**",
            decision="deny",
            priority=100,
            message="Refused destructive rm",
        ),
        PermissionRule(
            tool="execute",
            arg_pattern="*sudo *",
            decision="ask",
            priority=50,
            message="sudo requires approval",
        ),
    ]
    for path_pattern, decision, message in default_secure_path_rules():
        priority = 80 if decision == "deny" else 40
        # Named against write_file; PermissionEngine aliases cover all FS writers.
        rules.append(
            PermissionRule(
                tool="write_file",
                path_pattern=path_pattern,
                decision=decision,
                priority=priority,
                message=message,
            )
        )
    return rules


_DEFAULT_SECURE_RULES = _build_default_secure_rules()


def load_permission_engine(
    workspace: str | Path | None = None,
    *,
    extra_rules: list[PermissionRule] | None = None,
    include_defaults: bool = True,
) -> PermissionEngine:
    """Load rules from ``.clawagents/permissions.json`` plus secure defaults."""
    from clawagents.config.features import is_enabled

    engine = PermissionEngine(default_decision="allow")
    if not is_enabled("permission_rules"):
        return engine

    if include_defaults:
        engine.add_rules(list(_DEFAULT_SECURE_RULES))

    ws = Path(workspace or os.getcwd())
    for candidate in (
        ws / ".clawagents" / "permissions.json",
        ws / "permissions.json",
    ):
        if not candidate.is_file():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            rules = raw if isinstance(raw, list) else raw.get("rules") or []
            if isinstance(rules, list):
                engine.add_rules(
                    [
                        PermissionRule(**{k: v for k, v in row.items() if k in PermissionRule.__dataclass_fields__})
                        for row in rules
                        if isinstance(row, dict)
                    ]
                )
        except Exception:
            continue

    if extra_rules:
        engine.add_rules(extra_rules)
    return engine


__all__ = [
    "PermissionRule",
    "PermissionEngine",
    "load_permission_engine",
]
