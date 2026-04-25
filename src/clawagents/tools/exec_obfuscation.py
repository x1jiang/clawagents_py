"""Command obfuscation detector.

Ports the spirit of ``openclaw-main/src/infra/exec-obfuscation-detect.ts``.
Catches commands that try to bypass an allowlist by encoding/decoding into
a shell exec — e.g. ``base64 -d | sh``, ``curl … | sh``, process
substitution from network sources, hex/octal escape strings into ``eval``.

A small allowlist suppresses the well-known safe ``curl … | sh`` installers
(Homebrew, rustup, nvm, pnpm, bun, get.docker, install.python-poetry).

Public API: :func:`detect_obfuscation` returning :class:`ObfuscationFinding`
or ``None`` on a clean command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ObfuscationFinding:
    matched_patterns: List[str]
    reasons: List[str]

    @property
    def detected(self) -> bool:
        return bool(self.matched_patterns)


_SHELLS = r"(?:sh|bash|zsh|dash|ksh|fish)"

# (id, description, regex) — port of OBFUSCATION_PATTERNS.
_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    (
        "base64-pipe-exec",
        "Base64 decode piped to shell execution",
        re.compile(rf"base64\s+(?:-d|--decode)\b.*\|\s*{_SHELLS}\b", re.I),
    ),
    (
        "hex-pipe-exec",
        "Hex decode (xxd) piped to shell execution",
        re.compile(rf"xxd\s+-r\b.*\|\s*{_SHELLS}\b", re.I),
    ),
    (
        "printf-pipe-exec",
        "printf with escape sequences piped to shell execution",
        re.compile(rf"printf\s+.*\\x[0-9a-fA-F]{{2}}.*\|\s*{_SHELLS}\b", re.I),
    ),
    (
        "eval-decode",
        "eval with encoded/decoded input",
        re.compile(r"eval\s+.*(?:base64|xxd|printf|decode)", re.I),
    ),
    (
        "command-substitution-decode-exec",
        "Shell -c with command substitution decode/obfuscation",
        re.compile(
            rf"{_SHELLS}\s+-c\s+[\"'][^\"']*\$\([^)]*"
            r"(?:base64\s+(?:-d|--decode)|xxd\s+-r|printf\s+.*\\x[0-9a-fA-F]{2})"
            r"[^)]*\)[^\"']*[\"']",
            re.I,
        ),
    ),
    (
        "process-substitution-remote-exec",
        "Shell process substitution from remote content",
        re.compile(rf"{_SHELLS}\s+<\(\s*(?:curl|wget)\b", re.I),
    ),
    (
        "source-process-substitution-remote",
        "source/. with process substitution from remote content",
        re.compile(r"(?:^|[;&\s])(?:source|\.)\s+<\(\s*(?:curl|wget)\b", re.I),
    ),
    (
        "shell-heredoc-exec",
        "Shell heredoc execution",
        re.compile(rf"{_SHELLS}\s+<<-?\s*['\"]?[a-zA-Z_][\w-]*['\"]?", re.I),
    ),
    (
        "octal-escape",
        "Bash octal escape sequences (potential command obfuscation)",
        re.compile(r"\$'(?:[^']*\\[0-7]{3}){2,}"),
    ),
    (
        "hex-escape",
        "Bash hex escape sequences (potential command obfuscation)",
        re.compile(r"\$'(?:[^']*\\x[0-9a-fA-F]{2}){2,}"),
    ),
    (
        "python-exec-encoded",
        "Python/Perl/Ruby with base64 or encoded execution",
        re.compile(
            r"(?:python[23]?|perl|ruby)\s+-[ec]\s+.*"
            r"(?:base64|b64decode|decode|exec|system|eval)",
            re.I,
        ),
    ),
    (
        "curl-pipe-shell",
        "Remote content (curl/wget) piped to shell execution",
        re.compile(rf"(?:curl|wget)\s+.*\|\s*{_SHELLS}\b", re.I),
    ),
    (
        "var-expansion-obfuscation",
        "Variable assignment chain with expansion (potential obfuscation)",
        re.compile(r"(?:[a-zA-Z_]\w{0,2}=\S+\s*;\s*){2,}.*\$(?:[a-zA-Z_]|\{[a-zA-Z_])"),
    ),
]

# Allowlist suppressions for well-known installers — port of
# ``FALSE_POSITIVE_SUPPRESSIONS``. Each entry suppresses a specific
# pattern id when its regex matches and there is at most one URL.
_SUPPRESSIONS: List[Tuple[List[str], re.Pattern]] = [
    (
        ["curl-pipe-shell"],
        re.compile(
            r"curl\s+.*https?://(?:raw\.githubusercontent\.com/Homebrew|brew\.sh)\b",
            re.I,
        ),
    ),
    (
        ["curl-pipe-shell"],
        re.compile(
            r"curl\s+.*https?://(?:raw\.githubusercontent\.com/nvm-sh/nvm"
            r"|sh\.rustup\.rs|get\.docker\.com|install\.python-poetry\.org)\b",
            re.I,
        ),
    ),
    (
        ["curl-pipe-shell"],
        re.compile(
            r"curl\s+.*https?://(?:get\.pnpm\.io|bun\.sh/install)\b",
            re.I,
        ),
    ),
]

_URL_RE = re.compile(r"https?://\S+")


def detect_obfuscation(command: str) -> Optional[ObfuscationFinding]:
    """Return a finding if the command looks obfuscated, else ``None``.

    Each finding lists the matched pattern ids and human descriptions.
    Suppressions only apply when the command contains at most one URL —
    chained "curl ... | curl ... | sh" is never auto-suppressed.
    """
    if not command or not command.strip():
        return None

    url_count = len(_URL_RE.findall(command))
    matched_ids: List[str] = []
    reasons: List[str] = []

    for pat_id, desc, regex in _PATTERNS:
        if not regex.search(command):
            continue

        suppressed = False
        if url_count <= 1:
            for ids, supp_re in _SUPPRESSIONS:
                if pat_id in ids and supp_re.search(command):
                    suppressed = True
                    break
        if suppressed:
            continue

        matched_ids.append(pat_id)
        reasons.append(desc)

    if not matched_ids:
        return None
    return ObfuscationFinding(matched_patterns=matched_ids, reasons=reasons)
