"""Granular tool permission rules — Grok-style deny-wins evaluation.

Usage:
    from clawagents.tools.permissions import PermissionEngine, PermissionRule, load_permission_engine

    engine = load_permission_engine(workspace)
    # before_tool / registry gate
"""

from __future__ import annotations

import json
import os
import posixpath
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, List


RuleMatcher = Callable[[str, dict[str, Any]], bool]


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
    # Internal semantic matchers let built-in security rules reason about
    # structured arguments. They are deliberately not loadable from JSON.
    matcher: RuleMatcher | None = field(default=None, repr=False, compare=False)


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
        if rule.matcher is not None and not rule.matcher(tool_name, args):
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
            engine.add_rule(_rule_from_config(data))
        return engine


def _rule_from_config(data: dict[str, Any]) -> PermissionRule:
    fields = PermissionRule.__dataclass_fields__
    return PermissionRule(**{k: v for k, v in data.items() if k in fields and k != "matcher"})


_DYNAMIC_RM_TARGET_CHARS = frozenset("$`*?[]{}\n\r\x00")
_ROOT_LIKE_RM_TARGETS = frozenset({"", "/", ".", "..", "*", "./*", "~", "~/"})


def _is_literal_tmp_descendant(target: str) -> bool:
    """Only accept one statically named descendant of ``/tmp``.

    Variable expansion, globbing, traversal, and cleanup of ``/tmp`` itself
    remain denied. This is intentionally narrower than general shell safety:
    it exists solely for the common create-clean-scratch-directory workflow.
    """
    raw = target.rstrip("/")
    if not raw.startswith("/tmp/") or raw == "/tmp":
        return False
    if any(char in raw for char in _DYNAMIC_RM_TARGET_CHARS):
        return False
    if any(part in {"", ".", ".."} for part in raw.split("/")[2:]):
        return False
    return posixpath.normpath(raw) == raw


def _recursive_force_rm(tokens: list[str]) -> tuple[bool, list[str], bool]:
    """Return (is_recursive_force, targets, option_shape_is_proven)."""
    recursive = False
    force = False
    targets: list[str] = []
    options_proven = True
    options_done = False

    for token in tokens[1:]:
        if not options_done and token == "--":
            options_done = True
            continue
        if not options_done and token.startswith("--"):
            if token == "--recursive":
                recursive = True
            elif token == "--force":
                force = True
            else:
                options_proven = False
            continue
        if not options_done and token.startswith("-") and token != "-":
            flags = token[1:]
            recursive = recursive or "r" in flags or "R" in flags
            force = force or "f" in flags
            if any(flag not in "fRrv" for flag in flags):
                options_proven = False
            continue
        targets.append(token)

    return recursive and force, targets, options_proven


def _default_destructive_rm_matcher(_tool_name: str, args: dict[str, Any]) -> bool:
    """Match recursive-force deletes that the default policy must refuse.

    A direct, fully literal cleanup of exactly one ``/tmp/<name>`` directory is
    the sole absolute-path exception. Relative build-directory cleanup remains
    subject to the execute tool's path-aware WARN/BLOCK validator.
    """
    from clawagents.tools.bash_validator import (
        _WRAPPER_PROGRAMS,
        _collect_clauses,
        _peel_wrapper,
        _split_first_token,
    )

    command = str(args.get("command") or "")

    def must_deny(clause: str, depth: int = 0) -> bool:
        program, tokens = _split_first_token(clause)
        if program in _WRAPPER_PROGRAMS and depth < 6:
            inner = _peel_wrapper(program, tokens)
            if inner:
                # Wrapper evaluation makes the executed argv less obvious. Keep
                # recursive-force deletion behind explicit approval even when
                # its textual target happens to be under /tmp.
                for nested in _collect_clauses(inner) or [inner]:
                    nested_program, nested_tokens = _split_first_token(nested)
                    if nested_program == "rm" and _recursive_force_rm(nested_tokens)[0]:
                        return True
                    if must_deny(nested, depth + 1):
                        return True
            return False
        if program != "rm":
            return False

        is_recursive_force, targets, options_proven = _recursive_force_rm(tokens)
        if not is_recursive_force:
            return False
        if options_proven and len(targets) == 1 and _is_literal_tmp_descendant(targets[0]):
            return False

        for target in targets:
            if target.startswith("/"):
                return True
            if target in _ROOT_LIKE_RM_TARGETS or target.startswith("~"):
                return True
            if "$" in target or "`" in target:
                return True
        return False

    return any(must_deny(clause) for clause in (_collect_clauses(command) or [command]))


def _build_default_secure_rules() -> list[PermissionRule]:
    """Secure defaults — path patterns from ``clawagents.security.secret_paths``."""
    from clawagents.security.secret_paths import default_secure_path_rules

    rules: list[PermissionRule] = [
        PermissionRule(
            tool="execute",
            decision="deny",
            priority=100,
            message="Refused destructive rm",
            matcher=_default_destructive_rm_matcher,
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
                        _rule_from_config(row)
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
