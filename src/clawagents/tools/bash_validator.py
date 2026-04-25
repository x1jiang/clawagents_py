"""Bash semantic validator.

Inspired by ``claw-code-main/rust/crates/runtime/src/bash_validation.rs``.
We classify the *first* program name in a shell command and combine that
with shape heuristics on the argument list to reach a category and an
ALLOW/WARN/BLOCK decision.

The validator is conservative on the ALLOW side and explicit on the BLOCK
side: a small set of clearly destructive shapes is blocked, a wider set of
state-changing shapes is warned, and everything else is allowed. Unknown
programs default to ALLOW so we don't surprise users running their own
binaries.

Public API: :func:`validate_bash` returning :class:`BashDecision`.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import List


class CommandCategory(str, Enum):
    READ_ONLY = "READ_ONLY"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"
    NETWORK = "NETWORK"
    PROCESS = "PROCESS"
    PACKAGE = "PACKAGE"
    SYSTEM_ADMIN = "SYSTEM_ADMIN"
    UNKNOWN = "UNKNOWN"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class BashDecision:
    category: CommandCategory
    decision: Decision
    reason: str
    matched_pattern: str


# ─── Static command tables ───────────────────────────────────────────────

_READ_ONLY_PROGRAMS: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "which", "whereis", "pwd",
    "echo", "printf", "true", "false", "type", "command",
    "grep", "egrep", "fgrep", "rg", "ag",
    "sort", "uniq", "tr", "cut", "awk", "sed",  # default mode read-only; -i flagged below
    "diff", "cmp", "stat", "file", "du", "df",
    "env", "date", "uptime", "id", "whoami", "hostname",
    "ps", "top", "htop",
    "tree", "basename", "dirname", "realpath", "readlink",
    "find",  # only when missing -delete / -exec rm
})

_PACKAGE_PROGRAMS: frozenset[str] = frozenset({
    "apt", "apt-get", "yum", "dnf", "pacman", "brew",
    "pip", "pip3", "pipx", "uv",
    "npm", "yarn", "pnpm", "bun",
    "cargo", "gem", "go", "rustup",
    "poetry", "conda", "mamba",
})

_PROCESS_PROGRAMS: frozenset[str] = frozenset({
    "kill", "pkill", "killall", "xkill",
})

_SYSTEM_ADMIN_PROGRAMS: frozenset[str] = frozenset({
    "sudo", "su", "doas",
    "systemctl", "service", "launchctl",
    "mount", "umount",
    "useradd", "userdel", "usermod", "groupadd", "groupdel",
    "chmod", "chown", "chgrp",
    "iptables", "ufw", "pfctl",
    "reboot", "shutdown", "halt", "poweroff",
})

_NETWORK_PROGRAMS: frozenset[str] = frozenset({
    "curl", "wget", "ssh", "scp", "rsync", "ftp", "sftp",
    "nc", "netcat", "telnet", "nslookup", "dig", "host",
})

_WRITE_PROGRAMS: frozenset[str] = frozenset({
    "cp", "mv", "mkdir", "rmdir", "touch", "ln", "install", "tee",
    "truncate", "mkfifo", "mknod",
})

_DESTRUCTIVE_PROGRAMS: frozenset[str] = frozenset({
    "rm", "shred", "dd", "mkfs",
})


_FORK_BOMB_RE = re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")
_REDIRECT_TO_BLOCK_DEV_RE = re.compile(r">\s*/dev/(?:sd[a-z]+|nvme\d+|hd[a-z]+|disk\d+)")
_GIT_READ_SUBCMD: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "blame", "branch", "remote",
    "config", "describe", "ls-files", "ls-tree", "rev-parse",
    "stash", "tag",
})


def _split_first_token(command: str) -> tuple[str, List[str]]:
    """Return (program_name, full_arg_tokens) for the first command in
    a (possibly compound) shell line. We split on the first ``;``, ``&&``,
    ``||``, or ``|`` so we look at the head command only.
    """
    s = command.strip()
    # Strip env-var assignments at the front: ``FOO=bar baz``
    while s and re.match(r"[A-Za-z_][A-Za-z0-9_]*=", s):
        idx = s.find(" ")
        if idx < 0:
            break
        s = s[idx + 1 :].lstrip()
    # Truncate at first compound separator.
    head = re.split(r"\s*(?:\|\||&&|;|\|)\s*", s, maxsplit=1)[0]
    try:
        tokens = shlex.split(head, comments=False, posix=True)
    except ValueError:
        # Unbalanced quotes, etc. — fall back to whitespace split.
        tokens = head.split()
    if not tokens:
        return "", []
    return tokens[0], tokens


def _classify_rm(tokens: List[str]) -> BashDecision | None:
    """Sub-classify ``rm`` based on flag shape."""
    flags = [t for t in tokens[1:] if t.startswith("-")]
    paths = [t for t in tokens[1:] if not t.startswith("-")]

    has_recursive = any("r" in f.lstrip("-") or "R" in f.lstrip("-") for f in flags)
    has_force = any("f" in f.lstrip("-") for f in flags)

    bad_targets = {"/", "/*", ".", "./*", "..", "*", "~", "~/"}
    if any(p in bad_targets for p in paths) and (has_recursive or has_force):
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            f"rm with recursive/force on root-like target ({paths})",
            "rm -rf <root>",
        )
    if has_recursive and has_force:
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.WARN,
            "rm -rf is destructive; review the path carefully",
            "rm -rf",
        )
    return BashDecision(
        CommandCategory.DESTRUCTIVE,
        Decision.WARN,
        "rm removes files; review the path",
        "rm",
    )


def _classify_dd(tokens: List[str]) -> BashDecision | None:
    joined = " ".join(tokens)
    if re.search(r"\bof=/dev/(?:sd|nvme|hd|disk)", joined):
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            "dd writing to a block device wipes the disk",
            "dd of=/dev/sd*",
        )
    return BashDecision(
        CommandCategory.DESTRUCTIVE,
        Decision.WARN,
        "dd performs raw disk writes; review the of= target",
        "dd",
    )


def _classify_find(tokens: List[str]) -> BashDecision:
    args = tokens[1:]
    if "-delete" in args:
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            "find -delete recursively removes matched paths",
            "find -delete",
        )
    if "-exec" in args:
        # Best-effort: look for rm as the command after -exec.
        try:
            i = args.index("-exec")
            if i + 1 < len(args) and args[i + 1].endswith("rm"):
                return BashDecision(
                    CommandCategory.DESTRUCTIVE,
                    Decision.BLOCK,
                    "find -exec rm recursively removes matched paths",
                    "find -exec rm",
                )
        except ValueError:
            pass
    return BashDecision(
        CommandCategory.READ_ONLY,
        Decision.ALLOW,
        "find without -delete/-exec rm is read-only",
        "find",
    )


def _classify_chmod_chown(tokens: List[str]) -> BashDecision:
    program = tokens[0]
    args = tokens[1:]
    has_recursive = any(t in ("-R", "--recursive") for t in args)
    targets = [t for t in args if not t.startswith("-")]
    if program == "chmod" and "777" in args and has_recursive and any(t in ("/", "/*") for t in targets):
        return BashDecision(
            CommandCategory.SYSTEM_ADMIN,
            Decision.WARN,
            "chmod -R 777 / opens the entire filesystem; reviewing",
            "chmod -R 777 /",
        )
    if program == "chown" and has_recursive and any("root" in t for t in targets):
        return BashDecision(
            CommandCategory.SYSTEM_ADMIN,
            Decision.WARN,
            "chown -R root touches ownership at scale; reviewing",
            "chown -R root",
        )
    return BashDecision(
        CommandCategory.SYSTEM_ADMIN,
        Decision.WARN,
        f"{program} modifies permissions/ownership",
        program,
    )


def _classify_package(tokens: List[str]) -> BashDecision:
    program = tokens[0]
    args = tokens[1:]
    sub = args[0] if args else ""
    mutating = {
        "install", "uninstall", "remove", "rm", "add", "i",
        "upgrade", "update", "publish", "unpublish",
    }
    if sub in mutating:
        return BashDecision(
            CommandCategory.PACKAGE,
            Decision.WARN,
            f"{program} {sub} mutates installed packages",
            f"{program} {sub}",
        )
    return BashDecision(
        CommandCategory.PACKAGE,
        Decision.ALLOW,
        f"{program} {sub or '<noop>'} appears non-mutating",
        program,
    )


def _classify_git(tokens: List[str]) -> BashDecision:
    args = tokens[1:]
    sub = args[0] if args else ""
    if sub in _GIT_READ_SUBCMD:
        return BashDecision(
            CommandCategory.READ_ONLY,
            Decision.ALLOW,
            f"git {sub} is read-only",
            f"git {sub}",
        )
    if sub in {"reset", "clean", "rebase", "checkout", "restore", "switch", "rm", "mv"}:
        return BashDecision(
            CommandCategory.WRITE,
            Decision.WARN,
            f"git {sub} can rewrite/discard local changes",
            f"git {sub}",
        )
    if sub in {"push", "pull", "fetch", "clone", "submodule"}:
        return BashDecision(
            CommandCategory.NETWORK,
            Decision.WARN if sub == "push" else Decision.ALLOW,
            f"git {sub} interacts with remotes",
            f"git {sub}",
        )
    if sub in {"commit", "add", "stash", "merge", "tag"}:
        return BashDecision(
            CommandCategory.WRITE,
            Decision.ALLOW,
            f"git {sub} mutates the local repo",
            f"git {sub}",
        )
    return BashDecision(
        CommandCategory.UNKNOWN,
        Decision.ALLOW,
        f"git {sub or '<noop>'} not specifically classified",
        "git",
    )


def _classify_sed(tokens: List[str]) -> BashDecision:
    args = tokens[1:]
    in_place = any(a == "-i" or a.startswith("-i") and not a.startswith("--include") for a in args)
    if in_place:
        return BashDecision(
            CommandCategory.WRITE,
            Decision.WARN,
            "sed -i edits files in place",
            "sed -i",
        )
    return BashDecision(
        CommandCategory.READ_ONLY,
        Decision.ALLOW,
        "sed without -i is read-only",
        "sed",
    )


def validate_bash(command: str) -> BashDecision:
    """Classify a bash command and decide ALLOW / WARN / BLOCK.

    The decision is *advisory*: callers (the exec tool) decide what to do
    with WARN — typically prepending a notice to the output and proceeding.
    BLOCK should always cause a refusal.
    """
    raw = (command or "").strip()
    if not raw:
        return BashDecision(
            CommandCategory.UNKNOWN,
            Decision.ALLOW,
            "empty command",
            "",
        )

    # Whole-command shape checks first (these dominate any program name).
    if _FORK_BOMB_RE.search(raw):
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            "fork bomb detected",
            ":(){ :|:& };:",
        )
    if _REDIRECT_TO_BLOCK_DEV_RE.search(raw):
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            "redirect into a block device wipes the disk",
            "> /dev/sd*",
        )

    program, tokens = _split_first_token(raw)
    if not program:
        return BashDecision(
            CommandCategory.UNKNOWN,
            Decision.ALLOW,
            "no program name found",
            "",
        )

    # Per-program dispatch.
    if program == "rm":
        return _classify_rm(tokens) or BashDecision(
            CommandCategory.DESTRUCTIVE, Decision.WARN, "rm removes files", "rm",
        )
    if program == "dd":
        return _classify_dd(tokens) or BashDecision(
            CommandCategory.DESTRUCTIVE, Decision.WARN, "dd raw write", "dd",
        )
    if program == "find":
        return _classify_find(tokens)
    if program == "shred":
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            "shred overwrites and removes files irreversibly",
            "shred",
        )
    if program.startswith("mkfs"):
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            "mkfs formats a filesystem",
            "mkfs.*",
        )
    if program == "truncate":
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            "truncate can zero-out files",
            "truncate",
        )

    if program in {"chmod", "chown", "chgrp"}:
        return _classify_chmod_chown(tokens)

    if program == "git":
        return _classify_git(tokens)

    if program == "sed":
        return _classify_sed(tokens)

    if program in _PACKAGE_PROGRAMS:
        return _classify_package(tokens)

    if program in _PROCESS_PROGRAMS:
        return BashDecision(
            CommandCategory.PROCESS,
            Decision.WARN,
            f"{program} terminates processes",
            program,
        )
    if program in _SYSTEM_ADMIN_PROGRAMS:
        return BashDecision(
            CommandCategory.SYSTEM_ADMIN,
            Decision.WARN,
            f"{program} performs system administration",
            program,
        )
    if program in _NETWORK_PROGRAMS:
        return BashDecision(
            CommandCategory.NETWORK,
            Decision.ALLOW,
            f"{program} talks to the network",
            program,
        )
    if program in _WRITE_PROGRAMS:
        return BashDecision(
            CommandCategory.WRITE,
            Decision.ALLOW,
            f"{program} modifies the filesystem",
            program,
        )
    if program in _READ_ONLY_PROGRAMS:
        return BashDecision(
            CommandCategory.READ_ONLY,
            Decision.ALLOW,
            f"{program} is read-only",
            program,
        )

    return BashDecision(
        CommandCategory.UNKNOWN,
        Decision.ALLOW,
        f"{program} not specifically classified; default ALLOW",
        program,
    )
