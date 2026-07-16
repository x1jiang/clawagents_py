"""Grok-inspired skill invocation helpers (kept separate from the tool surface).

Integrates with ClawAgents' stronger progressive-disclosure model:
  - when_to_use listing / ranking boosts
  - $ARGUMENTS / ${SKILL_DIR} substitutions
  - path-gated skill activation
  - catalog auto-suggest for high-confidence matches
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

# Description prefixes Grok extracts into when-to-use
_WHEN_PREFIXES = (
    "use when",
    "invoke when",
    "apply when",
    "trigger when",
    "useful when",
    "best when",
)

_ARG_TOKEN_RE = re.compile(
    r"\$ARGUMENTS(?:\[(\d+)\])?|\$(\d+)\b|\$\{ARGUMENTS(?:\[(\d+)\])?\}"
)
_SKILL_DIR_RE = re.compile(
    r"\$\{(?:SKILL_DIR|CLAUDE_SKILL_DIR|CLAW_SKILL_DIR)\}|\$SKILL_DIR\b"
)
_SESSION_ID_RE = re.compile(
    r"\$\{(?:SESSION_ID|CLAUDE_SESSION_ID|CLAW_SESSION_ID)\}|\$SESSION_ID\b"
)


def extract_when_to_use_from_description(description: str) -> tuple[str, str]:
    """Split embedded 'Use when …' from description → (clean_desc, when_to_use)."""
    desc = (description or "").strip()
    if not desc:
        return "", ""
    m = re.search(
        r"(?i)(?:^|[.;]\s+)((?:use|invoke|apply|trigger|useful|best)\s+when)\s*:?\s*(.+)$",
        desc,
        flags=re.DOTALL,
    )
    if m:
        when = m.group(2).strip().rstrip(".")
        cleaned = desc[: m.start()].strip(" ;.\n")
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned, when
    lower = desc.casefold()
    for prefix in _WHEN_PREFIXES:
        if lower.startswith(prefix):
            rest = desc[len(prefix) :].lstrip(" :")
            parts = re.split(r"(?<=[.!?])\s+", rest, maxsplit=1)
            when = parts[0].strip().rstrip(".")
            cleaned = parts[1].strip() if len(parts) > 1 else ""
            return cleaned, when
    return desc, ""


def apply_skill_substitutions(
    body: str,
    *,
    skill_dir: str,
    arguments: str | Sequence[str] | None = None,
    session_id: str | None = None,
) -> str:
    """Expand $ARGUMENTS / $N / ${SKILL_DIR} / ${SESSION_ID} (Grok parity)."""
    text = body or ""
    if isinstance(arguments, (list, tuple)):
        arg_list = [str(a) for a in arguments]
        arg_blob = " ".join(arg_list)
    elif arguments is None:
        arg_list = []
        arg_blob = ""
    else:
        arg_blob = str(arguments)
        arg_list = arg_blob.split()

    def _arg_repl(match: re.Match[str]) -> str:
        idx_raw = match.group(1) or match.group(2) or match.group(3)
        if idx_raw is None:
            return arg_blob
        try:
            idx = int(idx_raw)
        except ValueError:
            return arg_blob
        if 0 <= idx < len(arg_list):
            return arg_list[idx]
        return ""

    had_arg_token = bool(_ARG_TOKEN_RE.search(text))
    text = _ARG_TOKEN_RE.sub(_arg_repl, text)
    text = _SKILL_DIR_RE.sub(str(skill_dir), text)
    if session_id:
        text = _SESSION_ID_RE.sub(session_id, text)
    else:
        text = _SESSION_ID_RE.sub("", text)

    # If body never referenced arguments, append them (Grok fallback)
    if arg_blob and not had_arg_token:
        text = text.rstrip() + f"\n\n**ARGUMENTS:** {arg_blob}"
    return text


def skill_paths_match(paths: Sequence[str], touched: Iterable[str]) -> bool:
    """Return True if any touched path matches a gitignore-style glob in ``paths``."""
    patterns = [p for p in paths if p]
    if not patterns:
        return True  # no gate
    touched_list = [t.replace("\\", "/") for t in touched if t]
    if not touched_list:
        return False

    def _match(path: str, pat: str) -> bool:
        base = os.path.basename(path)
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(base, pat):
            return True
        # Expand ** to a recursive wildcard Python fnmatch understands poorly.
        if "**" in pat:
            # src/**/*.py → match any path under src/ ending with .py
            # **/foo → match foo anywhere
            regex = (
                "^"
                + re.escape(pat)
                .replace(r"\*\*/", "(.*/)?")
                .replace(r"\*\*", ".*")
                .replace(r"\*", "[^/]*")
                .replace(r"\?", ".")
                + "$"
            )
            if re.match(regex, path):
                return True
        if "*" not in pat and "?" not in pat:
            if path == pat or path.startswith(pat.rstrip("/") + "/"):
                return True
        return False

    for pattern in patterns:
        pat = pattern.replace("\\", "/")
        for path in touched_list:
            if _match(path, pat):
                return True
    return False


def filter_skills_for_catalog(
    skills: Sequence[Any],
    *,
    touched_paths: Sequence[str] | None = None,
) -> list[Any]:
    """Drop path-gated skills until a matching file has been touched."""
    from clawagents.config.features import is_enabled

    if not is_enabled("skill_path_gating"):
        return list(skills)
    touched = list(touched_paths or ())
    out: list[Any] = []
    for skill in skills:
        paths = getattr(skill, "paths", None) or []
        if not paths:
            out.append(skill)
            continue
        if skill_paths_match(paths, touched):
            out.append(skill)
    return out


def format_skill_catalog_line(
    name: str,
    description: str,
    *,
    when_to_use: str = "",
    desc_cap: int = 160,
) -> str:
    """Grok-style catalog line with optional ``Use when:`` suffix."""
    desc = (description or "").strip()
    when = (when_to_use or "").strip()
    if when and desc.casefold().endswith(when.casefold()):
        # already embedded
        when = ""
    if desc_cap > 0 and len(desc) > desc_cap:
        desc = desc[: desc_cap - 1].rstrip() + "…"
    if when:
        when_cap = max(40, min(160, desc_cap))
        if len(when) > when_cap:
            when = when[: when_cap - 1].rstrip() + "…"
        if desc:
            return f"- **{name}**: {desc} — Use when: {when}"
        return f"- **{name}** — Use when: {when}"
    if desc:
        return f"- **{name}**: {desc}"
    return f"- **{name}**"


def auto_suggest_lines(
    scored: Sequence[tuple[Any, float]],
    *,
    threshold: float = 70.0,
    limit: int = 3,
) -> list[str]:
    """High-confidence nudge lines (does not auto-load bodies)."""
    from clawagents.config.features import is_enabled

    if not is_enabled("skill_auto_suggest"):
        return []
    hits = [(s, score) for s, score in scored if score >= threshold]
    hits.sort(key=lambda p: (-p[1], str(getattr(p[0], "name", "")).lower()))
    lines: list[str] = []
    for skill, score in hits[:limit]:
        name = str(getattr(skill, "name", "") or "")
        if not name:
            continue
        lines.append(
            f'Strongly consider calling `use_skill(name="{name}")` now '
            f"(match score {score:.0f})."
        )
    return lines


def note_touched_path(run_context: Any, path: str | None) -> None:
    """Record a workspace-relative/absolute path for path-gated skills."""
    if run_context is None or not path:
        return
    meta = getattr(run_context, "_metadata", None)
    if not isinstance(meta, dict):
        return
    bucket = meta.setdefault("touched_paths", [])
    if not isinstance(bucket, list):
        return
    normalized = str(path).replace("\\", "/").strip()
    if normalized and normalized not in bucket:
        bucket.append(normalized)
        # Cap growth
        if len(bucket) > 200:
            del bucket[:-200]


def collect_touched_paths(run_context: Any) -> list[str]:
    if run_context is None:
        return []
    meta = getattr(run_context, "_metadata", None)
    if not isinstance(meta, dict):
        return []
    raw = meta.get("touched_paths") or []
    return [str(x) for x in raw if x]


__all__ = [
    "extract_when_to_use_from_description",
    "apply_skill_substitutions",
    "skill_paths_match",
    "filter_skills_for_catalog",
    "format_skill_catalog_line",
    "auto_suggest_lines",
    "note_touched_path",
    "collect_touched_paths",
]
