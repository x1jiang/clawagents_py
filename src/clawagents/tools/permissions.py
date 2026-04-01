"""Granular tool permission rules (learned from Claude Code: useCanUseTool).

Provides a declarative permission system that wraps the existing before_tool hook
with Allow/Deny/Ask rules based on tool name, argument patterns, and file paths.

Usage:
    from clawagents.tools.permissions import PermissionEngine, PermissionRule

    engine = PermissionEngine()
    engine.add_rule(PermissionRule(
        tool="execute*",          # glob match
        decision="deny",
        message="Shell execution is disabled",
    ))
    engine.add_rule(PermissionRule(
        tool="write_file",
        path_pattern="*.py",     # only allow writing Python files
        decision="allow",
    ))

    # Use as before_tool hook:
    agent = create_claw_agent(before_tool=engine.check)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, List, Optional


@dataclass
class PermissionRule:
    """A single permission rule for a tool.

    Attributes:
        tool: Glob pattern for tool name (e.g., "execute*", "write_file")
        path_pattern: Glob pattern for file paths in tool args (e.g., "*.py", "/tmp/*")
        arg_pattern: Glob pattern matched against JSON-serialized args
        decision: "allow" | "deny" | "ask"
        message: Optional message to display when this rule triggers
        priority: Higher priority rules are evaluated first (default: 0)
    """
    tool: str = "*"
    path_pattern: str = "*"
    arg_pattern: str = "*"
    decision: str = "allow"  # "allow" | "deny" | "ask"
    message: str = ""
    priority: int = 0


class PermissionEngine:
    """Declarative permission engine for tool execution.

    Evaluates rules in priority order (highest first). The first matching
    rule determines the decision. If no rule matches, the default is "allow".
    """

    def __init__(self, default_decision: str = "allow"):
        self._rules: List[PermissionRule] = []
        self._default = default_decision
        self._sorted = False

    def add_rule(self, rule: PermissionRule) -> "PermissionEngine":
        """Add a permission rule. Returns self for chaining."""
        self._rules.append(rule)
        self._sorted = False
        return self

    def add_rules(self, rules: List[PermissionRule]) -> "PermissionEngine":
        """Add multiple permission rules."""
        self._rules.extend(rules)
        self._sorted = False
        return self

    def _ensure_sorted(self) -> None:
        if not self._sorted:
            self._rules.sort(key=lambda r: r.priority, reverse=True)
            self._sorted = True

    def _extract_path(self, args: dict[str, Any]) -> str:
        """Extract file path from tool args for path-based matching."""
        return str(args.get("path") or args.get("file_path") or args.get("target_path") or "")

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        """Evaluate permissions for a tool call.

        Returns:
            (decision, message) tuple where decision is "allow", "deny", or "ask"
        """
        self._ensure_sorted()

        file_path = self._extract_path(args)

        for rule in self._rules:
            # Match tool name
            if not fnmatch(tool_name, rule.tool):
                continue

            # Match file path pattern
            if rule.path_pattern != "*" and file_path:
                if not fnmatch(file_path, rule.path_pattern):
                    continue

            # Match arg pattern (against JSON-serialized args)
            if rule.arg_pattern != "*":
                import json
                args_str = json.dumps(args)
                if not fnmatch(args_str, rule.arg_pattern):
                    continue

            return rule.decision, rule.message

        return self._default, ""

    def check(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if a tool call is allowed.

        Compatible with the before_tool hook signature.
        Returns True if allowed, False if denied.
        """
        decision, message = self.evaluate(tool_name, args)
        return decision == "allow"

    @classmethod
    def from_config(cls, rules_data: list[dict[str, Any]]) -> "PermissionEngine":
        """Create a PermissionEngine from a list of rule dicts (e.g., from YAML/JSON config).

        Example config:
            [
                {"tool": "execute*", "decision": "deny"},
                {"tool": "write_file", "path_pattern": "/etc/*", "decision": "deny"},
            ]
        """
        engine = cls()
        for data in rules_data:
            engine.add_rule(PermissionRule(**data))
        return engine
