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
# Block-device redirect detector. Tolerates whitespace/quotes/FD-prefix
# (``1>``), so ``echo x >'/dev/sda'`` and ``1>/dev/sda`` are both caught.
_REDIRECT_TO_BLOCK_DEV_RE = re.compile(
    r"(?:^|[^>])>+\s*['\"]?\s*/dev/(?:sd[a-z]+|nvme\d+|hd[a-z]+|disk\d+)"
)
# ``tee`` writing into a block device — same risk class as ``> /dev/sd*``.
_TEE_BLOCK_DEV_RE = re.compile(r"\btee\b\s+(?:-\S+\s+)*['\"]?/dev/(?:sd[a-z]+|nvme\d+|hd[a-z]+|disk\d+)")
# ``tee`` writing into root-owned config that gives shell/perm escalation.
_TEE_SENSITIVE_RE = re.compile(
    r"\btee\b\s+(?:-\S+\s+)*['\"]?(?:/etc/(?:passwd|shadow|sudoers|hosts|ssh/|pam\.d/)"
    r"|/root/|/var/spool/cron/)",
    re.I,
)
_GIT_READ_SUBCMD: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "blame", "branch", "remote",
    "config", "describe", "ls-files", "ls-tree", "rev-parse",
    "stash", "tag",
})


def _split_first_token(command: str) -> tuple[str, List[str]]:
    """Return (program_name, full_arg_tokens) for a single shell clause.

    Caller is expected to pass clauses produced by :func:`_collect_clauses`
    (no compound separators inside). Still defensive: if the caller passes
    a compound, only the head is examined — the per-clause walk catches
    the rest.
    """
    s = command.strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    while s and re.match(r"[A-Za-z_][A-Za-z0-9_]*=", s):
        idx = s.find(" ")
        if idx < 0:
            break
        s = s[idx + 1 :].lstrip()
    head = re.split(r"\s*(?:\|\||&&|;|\|)\s*", s, maxsplit=1)[0]
    try:
        tokens = shlex.split(head, comments=False, posix=True)
    except ValueError:
        tokens = head.split()
    if not tokens:
        return "", []
    return tokens[0], tokens


_ROOT_LIKE_LITERALS: frozenset[str] = frozenset({
    "", "/", "/*", ".", "./*", "..", "*", "~", "~/", "$HOME", "${HOME}",
})

# System directories — ``rm -rf <any of these>`` (or anything beneath
# them) is treated as destructive enough to BLOCK rather than WARN.
_SYSTEM_ROOTS: tuple[str, ...] = (
    "/etc", "/var", "/usr", "/lib", "/lib64", "/sbin", "/bin", "/boot",
    "/opt", "/srv", "/sys", "/proc", "/dev", "/private", "/Users",
    "/home", "/root", "/Library", "/Applications", "/System",
)


def _is_root_like_path(raw_path: str) -> bool:
    """Return True if ``raw_path`` is the filesystem root, a HOME ref, or a
    system directory (or a path beneath one). Robust to surrounding quotes
    and ``--`` separators.
    """
    p = raw_path.strip().strip("'\"").rstrip("/") or "/"
    if p in _ROOT_LIKE_LITERALS:
        return True
    if p.startswith("~") or p.startswith("$HOME") or p.startswith("${HOME}"):
        return True
    for d in _SYSTEM_ROOTS:
        if p == d or p.startswith(d + "/"):
            return True
    return False


def _classify_rm(tokens: List[str]) -> BashDecision | None:
    """Sub-classify ``rm`` based on flag shape."""
    args = [t for t in tokens[1:] if t != "--"]
    flags = [t for t in args if t.startswith("-")]
    paths = [t for t in args if not t.startswith("-")]

    long_recursive = any(f in ("--recursive", "-R", "-r") for f in flags)
    long_force = any(f == "--force" for f in flags)
    short_combined = [f.lstrip("-") for f in flags if f.startswith("-") and not f.startswith("--")]
    has_recursive = long_recursive or any(("r" in f) or ("R" in f) for f in short_combined)
    has_force = long_force or any("f" in f for f in short_combined)

    if any(_is_root_like_path(p) for p in paths) and (has_recursive or has_force):
        return BashDecision(
            CommandCategory.DESTRUCTIVE,
            Decision.BLOCK,
            f"rm with recursive/force on root-like or system target ({paths})",
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
    EXEC_FLAGS = {"-exec", "-execdir", "-ok", "-okdir"}
    SHELL_PROGRAMS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
    i = 0
    while i < len(args):
        if args[i] in EXEC_FLAGS:
            j = i + 1
            while j < len(args) and args[j] not in (";", "+", r"\;"):
                base = args[j].rsplit("/", 1)[-1]
                if base == "rm" or base == "shred":
                    return BashDecision(
                        CommandCategory.DESTRUCTIVE,
                        Decision.BLOCK,
                        f"find {args[i]} {base} recursively removes matched paths",
                        f"find {args[i]} {base}",
                    )
                if base in SHELL_PROGRAMS:
                    return BashDecision(
                        CommandCategory.DESTRUCTIVE,
                        Decision.BLOCK,
                        f"find {args[i]} {base} -c <cmd> obscures the executed command",
                        f"find {args[i]} {base}",
                    )
                j += 1
            i = j
        else:
            i += 1
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
            Decision.BLOCK,
            "chmod -R 777 / opens the entire filesystem",
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
    # Detect both short ``-i`` / ``-i.bak`` and GNU long form ``--in-place``.
    # ``--include`` (a different flag) must NOT match.
    in_place = any(
        a == "-i"
        or (a.startswith("-i") and not a.startswith("--"))
        or a == "--in-place"
        or a.startswith("--in-place=")
        for a in args
    )
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


_CLAUSE_SEP_RE = re.compile(r"\s*(?:\|\||&&|\||;|&|\n)\s*")
_SUBST_RE = re.compile(r"\$\(([^()]+)\)|`([^`]+)`")
# ``bash -c 'rm -rf /'``: extract the quoted arg after ``-c``.
_SHELL_C_RE = re.compile(
    r"\b(?:bash|sh|zsh|dash|ksh|fish)\s+(?:-\S+\s+)*-c\s+"
    r"(?:'([^']*)'|\"([^\"]*)\"|(\S+))"
)
# Redirect into an unquoted variable: we can't statically know the target.
_REDIRECT_TO_VAR_RE = re.compile(r">+\s*\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _strip_subshell(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def _collect_clauses(command: str) -> List[str]:
    """Return every shell clause that ``command`` will execute.

    Splits on ``;`` ``&&`` ``||`` ``|`` ``&`` and newlines, strips outer
    parentheses, recurses into ``$(...)`` / backtick command
    substitutions, and unwraps ``bash -c '<cmd>'`` style invocations.
    Best-effort: we don't fully parse the shell grammar, but this covers
    the common bypass cases.
    """
    out: List[str] = []
    work = [_strip_subshell(command)]
    seen: set[str] = set()
    while work:
        s = work.pop()
        if s in seen:
            continue
        seen.add(s)
        for sub in _SUBST_RE.findall(s):
            inner = sub[0] or sub[1]
            inner = _strip_subshell(inner)
            if inner:
                work.append(inner)
        for m in _SHELL_C_RE.finditer(s):
            inner = m.group(1) or m.group(2) or m.group(3) or ""
            inner = _strip_subshell(inner)
            if inner:
                work.append(inner)
        for part in _CLAUSE_SEP_RE.split(s):
            part = _strip_subshell(part)
            if not part:
                continue
            out.append(part)
    return out


def _severity(d: BashDecision) -> int:
    return {Decision.ALLOW: 0, Decision.WARN: 1, Decision.BLOCK: 2}[d.decision]


def _validate_single_clause(raw: str) -> BashDecision:
    """Validate one shell clause (no compound separators inside).

    Whole-clause shape checks run first, then per-program dispatch.
    """
    if _REDIRECT_TO_BLOCK_DEV_RE.search(raw):
        return BashDecision(
            CommandCategory.DESTRUCTIVE, Decision.BLOCK,
            "redirect into a block device wipes the disk", "> /dev/sd*",
        )
    if _TEE_BLOCK_DEV_RE.search(raw):
        return BashDecision(
            CommandCategory.DESTRUCTIVE, Decision.BLOCK,
            "tee into a block device wipes the disk", "tee /dev/sd*",
        )
    if _TEE_SENSITIVE_RE.search(raw):
        return BashDecision(
            CommandCategory.SYSTEM_ADMIN, Decision.BLOCK,
            "tee into a privileged config path can subvert system trust",
            "tee /etc/...",
        )
    if _REDIRECT_TO_VAR_RE.search(raw):
        # We can't statically resolve the variable's value — flag for
        # human review rather than blocking outright.
        return BashDecision(
            CommandCategory.WRITE, Decision.WARN,
            "redirecting to an unquoted variable target — verify it",
            ">$VAR",
        )

    program, tokens = _split_first_token(raw)
    if not program:
        return BashDecision(
            CommandCategory.UNKNOWN, Decision.ALLOW,
            "no program name found", "",
        )

    return _dispatch_program(program, tokens)


def validate_bash(command: str) -> BashDecision:
    """Classify a bash command and decide ALLOW / WARN / BLOCK.

    The decision is *advisory*: callers (the exec tool) decide what to do
    with WARN — typically prepending a notice to the output and proceeding.
    BLOCK should always cause a refusal.

    Compound commands (``;`` ``&&`` ``||`` ``|`` ``&``), subshells
    (``(...)``), and command substitutions (``$(...)`` / backticks) are
    each validated; the strictest decision wins.
    """
    raw = (command or "").strip()
    if not raw:
        return BashDecision(
            CommandCategory.UNKNOWN, Decision.ALLOW, "empty command", "",
        )

    # Refuse null bytes / unprintable control chars: C-level path APIs
    # truncate at the null, so ``rm -rf /\\x00`` would inspect a non-root
    # path here and operate on ``/`` at execution time.
    if any(c == "\x00" or (ord(c) < 0x20 and c not in "\t\n\r") for c in raw):
        return BashDecision(
            CommandCategory.DESTRUCTIVE, Decision.BLOCK,
            "command contains a null byte or unprintable control character",
            "<NUL>",
        )

    # Whole-command shape checks that don't survive clause splitting.
    if _FORK_BOMB_RE.search(raw):
        return BashDecision(
            CommandCategory.DESTRUCTIVE, Decision.BLOCK,
            "fork bomb detected", ":(){ :|:& };:",
        )

    clauses = _collect_clauses(raw) or [raw]
    worst: BashDecision | None = None
    for clause in clauses:
        d = _validate_single_clause(clause)
        if d.decision == Decision.BLOCK:
            return d
        if worst is None or _severity(d) > _severity(worst):
            worst = d
    assert worst is not None
    return worst


def _dispatch_program(program: str, tokens: List[str]) -> BashDecision:
    """Per-program dispatch — the body of the original validate_bash."""

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
