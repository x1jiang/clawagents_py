import copy
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from clawagents.tools.registry import Tool, ToolResult

# Agent Skills spec limits (agentskills.io; mirrored from deepagents/Claude Code).
# Violations warn — they never reject a skill (lenient like Claude Code).
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024
# Oversized SKILL.md files are skipped outright (openclaw caps at 256K,
# deepagents at 10M; 1M is a safe middle ground for an instruction file).
MAX_SKILL_FILE_BYTES = 1024 * 1024

# Max bundled-resource entries shown by use_skill.
_MAX_RESOURCE_ENTRIES = 20

@dataclass
class SkillRequires:
    os: Optional[str] = None
    bins: Optional[List[str]] = None
    env: Optional[List[str]] = None

@dataclass
class Skill:
    name: str
    description: str
    content: str
    path: str
    # None = no boundary declared; [] = explicitly allow no data-plane tools.
    allowed_tools: Optional[List[str]] = None
    aliases: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    anti_triggers: List[str] = field(default_factory=list)
    requires: Optional[SkillRequires] = None
    forbidden_actions: List[str] = field(default_factory=list)
    workspace_layout: str = ""
    success_criteria: str = ""
    workflow_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Claude Code / openclaw `disable-model-invocation`: keep the skill out of
    # the model-facing catalog and refuse use_skill (user-invocation only).
    disable_model_invocation: bool = False

    @property
    def base_dir(self) -> str:
        """Directory containing the skill file (resources live beside it)."""
        return str(Path(self.path).parent)

    @property
    def is_dir_skill(self) -> bool:
        """True for `<dir>/SKILL.md` skills, which own their directory."""
        return Path(self.path).name.lower() == "skill.md"


@dataclass(frozen=True)
class CatalogSnapshot:
    generation: int
    content_hash: str
    scan_duration_ms: float
    parsed_files: int
    reused_files: int
    collisions: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedSkillCacheEntry:
    metadata: tuple[int, int, int]
    content_hash: str
    skill: Skill


_PARSED_SKILL_CACHE_MAX = 256
_PARSED_SKILL_CACHE: "OrderedDict[str, _ParsedSkillCacheEntry]" = OrderedDict()
_PARSED_SKILL_CACHE_LOCK = threading.RLock()
_CATALOG_GENERATION = 0


def _next_catalog_generation() -> int:
    global _CATALOG_GENERATION
    with _PARSED_SKILL_CACHE_LOCK:
        _CATALOG_GENERATION += 1
        return _CATALOG_GENERATION


def _read_parsed_skill(skill_file: Path) -> tuple[Skill, bool]:
    """Parse changed skills and reuse immutable snapshots for unchanged files."""
    canonical = str(skill_file.resolve())
    stat = skill_file.stat()
    metadata = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _PARSED_SKILL_CACHE_LOCK:
        cached = _PARSED_SKILL_CACHE.get(canonical)
        if cached is not None and cached.metadata == metadata:
            _PARSED_SKILL_CACHE.move_to_end(canonical)
            return copy.deepcopy(cached.skill), True

    raw = skill_file.read_bytes()
    after = skill_file.stat()
    after_metadata = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if metadata != after_metadata:
        raw = skill_file.read_bytes()
        after = skill_file.stat()
        after_metadata = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    content_hash = hashlib.sha256(raw).hexdigest()

    with _PARSED_SKILL_CACHE_LOCK:
        cached = _PARSED_SKILL_CACHE.get(canonical)
        if cached is not None and cached.content_hash == content_hash:
            refreshed = _ParsedSkillCacheEntry(after_metadata, content_hash, cached.skill)
            _PARSED_SKILL_CACHE[canonical] = refreshed
            _PARSED_SKILL_CACHE.move_to_end(canonical)
            return copy.deepcopy(cached.skill), True

    skill = parse_skill_file(raw.decode("utf-8"), canonical)
    entry = _ParsedSkillCacheEntry(after_metadata, content_hash, copy.deepcopy(skill))
    with _PARSED_SKILL_CACHE_LOCK:
        _PARSED_SKILL_CACHE[canonical] = entry
        _PARSED_SKILL_CACHE.move_to_end(canonical)
        while len(_PARSED_SKILL_CACHE) > _PARSED_SKILL_CACHE_MAX:
            _PARSED_SKILL_CACHE.popitem(last=False)
    return skill, False


def _default_skill_name(file_path: str) -> str:
    """`<dir>/SKILL.md` is named after the directory (Claude Code rule);
    flat `foo.md` skills fall back to the file stem."""
    p = Path(file_path)
    if p.name.lower() == "skill.md" and p.parent.name:
        return p.parent.name
    return p.stem


def _validate_skill_name(name: str, file_path: str) -> List[str]:
    """Agent Skills spec checks — warn, never reject."""
    warnings: List[str] = []
    if len(name) > MAX_SKILL_NAME_LENGTH:
        warnings.append(
            f"skill name exceeds {MAX_SKILL_NAME_LENGTH} chars: {name[:40]}…"
        )
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        warnings.append(
            f'skill name "{name}" is not spec-conformant '
            "(lowercase letters/digits/hyphens; no leading/trailing/double hyphen)"
        )
    p = Path(file_path)
    if p.name.lower() == "skill.md" and p.parent.name and name != p.parent.name:
        warnings.append(
            f'skill name "{name}" does not match its directory "{p.parent.name}"'
        )
    return warnings


def _fallback_description(body: str) -> str:
    """First meaningful markdown line (Claude Code fallback for missing
    description) so bare .md skills still surface something in the catalog."""
    for line in (body or "").splitlines():
        text = line.strip()
        if not text or text.startswith(("#", "```", "<!--", "---", "|")):
            continue
        text = re.sub(r"[*_`>]+", "", text).strip()
        if text:
            return text[:200]
    return ""


def _parse_inline_list(raw: str) -> List[str]:
    cleaned = re.sub(r'[\[\]"\']', "", raw)
    return [x.strip() for x in re.split(r"[\s,]+", cleaned) if x.strip()]


def _parse_requires_block(yaml_content: str) -> Tuple[Optional[str], Optional[List[str]], Optional[List[str]]]:
    """Parse eligibility requirements without false-matching other blocks.

    Accepts, in priority order:
      1. dotted keys at top level: ``requires.os: darwin``
      2. a scoped ``requires:`` block (indented ``os:``/``bins:``/``env:``,
         inline or block lists)
      3. openclaw-style single-line JSON metadata:
         ``metadata: {"openclaw": {"os": [...], "requires": {"bins": [...]}}}``

    Earlier versions matched ``^\\s+os:`` anywhere in the frontmatter, so an
    indented key inside an unrelated block (e.g. ``metadata:``) silently gated
    the skill. Requirements are now only read from the shapes above.
    """
    os_val: Optional[str] = None
    bins: Optional[List[str]] = None
    env: Optional[List[str]] = None

    # 1. Dotted keys.
    m = re.search(r"^requires\.os:\s*(.+)$", yaml_content, re.MULTILINE)
    if m:
        os_val = m.group(1).strip()
    m = re.search(r"^requires\.bins:\s*(.+)$", yaml_content, re.MULTILINE)
    if m:
        bins = _parse_inline_list(m.group(1))
    m = re.search(r"^requires\.env:\s*(.+)$", yaml_content, re.MULTILINE)
    if m:
        env = _parse_inline_list(m.group(1))

    # 2. Scoped `requires:` block — only lines indented under it.
    block_match = re.search(
        r"^requires:\s*\n((?:[ \t]+[^\n]*\n?)+)", yaml_content, re.MULTILINE
    )
    if block_match:
        block = block_match.group(1)

        def _block_value(key: str) -> Optional[List[str]]:
            inline = re.search(rf"^[ \t]+{key}:[ \t]*(\S.*)$", block, re.MULTILINE)
            if inline and inline.group(1).strip():
                return _parse_inline_list(inline.group(1))
            # Block list: `env:` followed by deeper-indented `- ITEM` lines.
            lst = re.search(
                rf"^([ \t]+){key}:\s*\n((?:\1[ \t]+-[^\n]*\n?)+)", block, re.MULTILINE
            )
            if lst:
                return [
                    i.strip()
                    for i in re.findall(r"-\s*([^\n]+)", lst.group(2))
                    if i.strip()
                ]
            return None

        os_items = _block_value("os")
        if os_val is None and os_items:
            os_val = " ".join(os_items)
        if bins is None:
            bins = _block_value("bins")
        if env is None:
            env = _block_value("env")

    # 3. openclaw single-line JSON metadata (best-effort, strict JSON only).
    if os_val is None and bins is None and env is None:
        meta_match = re.search(r"^metadata:\s*(\{.+\})\s*$", yaml_content, re.MULTILINE)
        if meta_match:
            try:
                meta = json.loads(meta_match.group(1))
                oc = meta.get("openclaw") if isinstance(meta, dict) else None
                if isinstance(oc, dict):
                    oc_os = oc.get("os")
                    if isinstance(oc_os, list) and oc_os:
                        os_val = " ".join(str(x) for x in oc_os)
                    oc_req = oc.get("requires")
                    if isinstance(oc_req, dict):
                        if isinstance(oc_req.get("bins"), list):
                            bins = [str(b) for b in oc_req["bins"]]
                        if isinstance(oc_req.get("env"), list):
                            env = [str(e) for e in oc_req["env"]]
            except (ValueError, TypeError):
                pass

    return os_val, bins, env


def parse_skill_file(content: str, file_path: str) -> Skill:
    name = _default_skill_name(file_path)
    description = ""
    body = content
    allowed_tools: Optional[List[str]] = None
    aliases: List[str] = []
    triggers: List[str] = []
    anti_triggers: List[str] = []
    requires: Optional[SkillRequires] = None
    forbidden_actions: List[str] = []
    workspace_layout: str = ""
    success_criteria: str = ""
    workflow_steps: List[str] = []
    warnings: List[str] = []
    disable_model_invocation = False

    # Closing `---` may sit at EOF (no trailing newline / empty body).
    frontmatter_match = re.match(
        r"^---\s*\n([\s\S]*?)\n---\s*(?:\n([\s\S]*))?$", content
    )
    if frontmatter_match:
        yaml_content = frontmatter_match.group(1) or ""
        body = frontmatter_match.group(2) or ""

        name_match = re.search(r"^name:\s*(.+)$", yaml_content, re.MULTILINE)
        if name_match:
            explicit = name_match.group(1).strip().strip("\"'")
            if explicit:
                name = explicit

        description = _parse_frontmatter_description(yaml_content)

        # Parse allowed-tools: space/comma-delimited string (optionally YAML-ish
        # brackets/quotes, consistent with requires list parsing).
        tools_match = re.search(r"^allowed-tools:\s*(.*)$", yaml_content, re.MULTILINE)
        if tools_match:
            allowed_tools = _parse_inline_list(tools_match.group(1))

        # Only the literal true counts (Claude Code boolean parsing rule).
        dmi_match = re.search(
            r"^disable-model-invocation:\s*[\"']?true[\"']?\s*$",
            yaml_content,
            re.MULTILINE,
        )
        disable_model_invocation = bool(dmi_match)

        req_os, req_bins, req_env = _parse_requires_block(yaml_content)
        if req_os or req_bins or req_env:
            requires = SkillRequires(os=req_os, bins=req_bins, env=req_env)

        def _parse_block_list(key: str, yaml_src: str) -> Optional[List[str]]:
            """Parse a YAML key that may have an inline value or a block list of '- item' entries."""
            # First try: key followed immediately by block list items on next lines
            block_pattern = re.compile(
                r"^" + re.escape(key) + r":\s*\n((?:[ \t]+-[^\n]*\n?)+)",
                re.MULTILINE,
            )
            bm = block_pattern.search(yaml_src)
            if bm:
                block_raw = bm.group(1)
                items = re.findall(r"^[ \t]+-\s+(.+)$", block_raw, re.MULTILINE)
                return [item.strip() for item in items if item.strip()]

            # Second try: inline value on same line
            inline_pattern = re.compile(
                r"^" + re.escape(key) + r":\s+(.+)$",
                re.MULTILINE,
            )
            im = inline_pattern.search(yaml_src)
            if im:
                return _parse_inline_list(im.group(1).strip())

            return None

        def _parse_phrase_list(key: str) -> List[str]:
            block = re.search(
                r"^" + re.escape(key) + r":\s*\n((?:[ \t]+-[^\n]*\n?)+)",
                yaml_content,
                re.MULTILINE,
            )
            if block:
                return [
                    item.strip().strip("\"'")
                    for item in re.findall(r"^[ \t]+-\s+(.+)$", block.group(1), re.MULTILINE)
                    if item.strip()
                ]
            inline = re.search(
                r"^" + re.escape(key) + r":\s+(.+)$",
                yaml_content,
                re.MULTILINE,
            )
            if not inline:
                return []
            raw = inline.group(1).strip().strip("[]")
            return [item.strip().strip("\"'") for item in raw.split(",") if item.strip()]

        aliases = _parse_phrase_list("aliases")
        triggers = _parse_phrase_list("triggers")
        anti_triggers = (
            _parse_phrase_list("anti-triggers")
            or _parse_phrase_list("anti_triggers")
            or []
        )

        # Parse forbidden-actions: inline or block list
        fa_items = _parse_block_list("forbidden-actions", yaml_content)
        if fa_items is not None:
            forbidden_actions = fa_items

        # Parse workspace-layout: single-line string or literal block scalar
        layout_match = re.search(r'^workspace-layout:\s*\|?\s*"?([^"|\n][^"]*)"?$', yaml_content, re.MULTILINE)
        if layout_match:
            workspace_layout = layout_match.group(1).strip()
        else:
            # Literal block scalar (|) — grab indented content
            layout_block = re.search(r"^workspace-layout:\s*\|\s*\n((?:[ \t]+[^\n]*\n?)+)", yaml_content, re.MULTILINE)
            if layout_block:
                workspace_layout = layout_block.group(1)

        # Parse success-criteria: single-line string
        criteria_match = re.search(r'^success-criteria:\s*"?([^"\n]+)"?$', yaml_content, re.MULTILINE)
        if criteria_match:
            success_criteria = criteria_match.group(1).strip()

        # Parse workflow-steps: inline or block list
        ws_items = _parse_block_list("workflow-steps", yaml_content)
        if ws_items is not None:
            workflow_steps = ws_items

    if not description:
        description = _fallback_description(body)
    if len(description) > MAX_SKILL_DESCRIPTION_LENGTH:
        warnings.append(
            f"description exceeds {MAX_SKILL_DESCRIPTION_LENGTH} chars; truncated"
        )
        description = description[: MAX_SKILL_DESCRIPTION_LENGTH - 1].rstrip() + "…"
    warnings.extend(_validate_skill_name(name, file_path))

    return Skill(
        name=name,
        description=description,
        content=body.strip(),
        path=file_path,
        allowed_tools=allowed_tools,
        aliases=aliases,
        triggers=triggers,
        anti_triggers=anti_triggers,
        requires=requires,
        forbidden_actions=forbidden_actions,
        workspace_layout=workspace_layout,
        success_criteria=success_criteria,
        workflow_steps=workflow_steps,
        warnings=warnings,
        disable_model_invocation=disable_model_invocation,
    )


def _parse_frontmatter_description(yaml_content: str) -> str:
    """Parse skill description (plain, quoted, or YAML block scalar)."""
    block = re.search(
        r"^description:\s*[|>]-?\s*\n((?:[ \t]+.*\n?)+)",
        yaml_content,
        re.MULTILINE,
    )
    if block:
        lines = [ln.strip() for ln in block.group(1).splitlines() if ln.strip()]
        return " ".join(lines).strip()

    quoted = re.search(r'^description:\s*"(.*)"\s*$', yaml_content, re.MULTILINE)
    if quoted:
        return quoted.group(1).strip()

    single = re.search(r"^description:\s*'(.*)'\s*$", yaml_content, re.MULTILINE)
    if single:
        return single.group(1).strip()

    plain = re.search(r"^description:\s*(.+)$", yaml_content, re.MULTILINE)
    if plain:
        return plain.group(1).strip().strip("\"'")
    return ""


_OS_ALIASES = {
    "darwin": "darwin", "macos": "darwin", "mac": "darwin", "osx": "darwin",
    "win32": "win32", "windows": "win32", "win": "win32",
    "linux": "linux",
}


def _normalize_os_values(raw: str) -> List[str]:
    """Map user-facing OS names (macos, windows, …) to sys.platform values."""
    out: List[str] = []
    for part in re.split(r"[\s,]+", (raw or "").strip().lower()):
        if not part:
            continue
        if part in ("any", "all", "*"):
            return []  # matches everything
        out.append(_OS_ALIASES.get(part, part))
    return out


def skill_ineligibility_reason(skill: Skill) -> Optional[str]:
    """Why a skill cannot run here, or None if eligible."""
    if not skill.requires:
        return None
    req = skill.requires
    if req.os:
        wanted = _normalize_os_values(req.os)
        if wanted and sys.platform not in wanted:
            return f"requires os {req.os} (current: {sys.platform})"
    for b in req.bins or []:
        if shutil.which(b) is None:
            return f"missing binary: {b}"
    for var in req.env or []:
        if not os.environ.get(var):
            return f"missing env var: {var}"
    return None


def is_skill_eligible(skill: Skill) -> bool:
    return skill_ineligibility_reason(skill) is None


# ── Load-time content inspection (supply-chain hardening) ───────────────────
# Auto-discovered skills are injected into the model prompt without a human
# in the loop, so an attacker who lands a SKILL.md in a scanned directory can
# smuggle instructions the operator never sees. Two documented vectors:
#   * invisible-Unicode smuggling (Unicode Tags block, bidi overrides,
#     zero-width chars) — the "Rules File Backdoor" / Trojan Source pattern;
#   * remote-exec one-liners in the instruction body (ClawHub / VirusTotal).
# This is defense-in-depth, NEVER a trust decision — scanners are evadable.
# On a high-signal hit the skill is *quarantined* (kept for diagnostics, kept
# out of the model-facing catalog, refused by use_skill) rather than deleted.

# Unicode Tags block: invisible chars models still read — the prime smuggling
# vector, ~zero legitimate use in prose.
_TAG_CHARS_RE = re.compile(r"[\U000E0000-\U000E007F]")
# Bidirectional overrides / isolates (Trojan Source): reorder rendered text.
_BIDI_OVERRIDE_RE = re.compile(r"[‪-‮⁦-⁩]")
# Zero-width / soft-hyphen / BOM: stripped as hygiene + warned, but common
# enough (emoji ZWJ, stray BOM) that they don't quarantine on their own.
_ZERO_WIDTH_RE = re.compile(r"[​-‍⁠﻿­]")
# C0/C1 control chars except tab/newline/carriage-return.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# High-signal remote-execution signatures. Tuned for precision: these are
# unambiguous in an auto-loaded instruction file. Lower-signal mentions
# (bare ``rm -rf`` / ``subprocess.`` / ``eval(``) are intentionally NOT here
# — they appear in legitimate skill instructions and belong to the stricter
# authoring-time gate (skills/workshop/scanner.py), not the load gate.
_DANGEROUS_LOAD_PATTERNS: List[tuple] = [
    (
        re.compile(
            r"\b(?:curl|wget|fetch)\b[^\n|]{0,400}\|\s*(?:sudo\s+)?(?:ba|z|k|a)?sh\b",
            re.IGNORECASE,
        ),
        "pipes a network download straight into a shell",
    ),
    (
        re.compile(r"\b(?:iex|invoke-expression)\s*[\(\"']", re.IGNORECASE),
        "PowerShell Invoke-Expression of dynamic content",
    ),
    (
        re.compile(r"new-object\s+net\.webclient", re.IGNORECASE),
        "PowerShell remote download via Net.WebClient",
    ),
    (
        re.compile(
            r"base64\s+(?:-d|--decode|-D)\b[^\n]{0,200}\|\s*(?:ba)?sh", re.IGNORECASE
        ),
        "base64-decodes content and pipes it to a shell",
    ),
    (
        re.compile(
            r"\|\s*base64\s+(?:-d|--decode|-D)\b[^\n]{0,80}\|\s*(?:python|node|perl|ruby)",
            re.IGNORECASE,
        ),
        "base64-decodes content and pipes it to an interpreter",
    ),
]


def _strip_invisible(text: str) -> tuple[str, int]:
    """Remove zero-width / soft-hyphen / BOM / control chars. Returns
    (cleaned, removed_count). Tab/newline/CR are preserved."""
    if not text:
        return text, 0
    removed = len(_ZERO_WIDTH_RE.findall(text)) + len(_CONTROL_RE.findall(text))
    if removed:
        text = _CONTROL_RE.sub("", _ZERO_WIDTH_RE.sub("", text))
    return text, removed


def scan_skill_content(name: str, description: str, body: str) -> list[str]:
    """High-signal findings that quarantine a skill at load time.

    Covers the always-injected metadata (name+description) and the body.
    Returns human-readable reasons; empty means the skill passed.
    """
    findings: list[str] = []
    for label, text in (("name", name), ("description", description), ("body", body)):
        if _TAG_CHARS_RE.search(text or ""):
            findings.append(f"invisible Unicode Tag characters in {label}")
        if _BIDI_OVERRIDE_RE.search(text or ""):
            findings.append(f"bidirectional-override characters in {label}")
    for pattern, why in _DANGEROUS_LOAD_PATTERNS:
        if pattern.search(body or ""):
            findings.append(why)
    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _skill_scan_enabled() -> bool:
    """Load-time quarantine is on unless CLAW_SKILL_SCAN=off (invisible-char
    stripping still runs regardless — it is pure hygiene)."""
    return (os.environ.get("CLAW_SKILL_SCAN") or "").strip().lower() not in (
        "off",
        "0",
        "false",
        "no",
    )


class SkillStore:
    """Loads skills from directories.

    Precedence: directories are loaded in the order added and a later
    directory overrides an earlier one on name collision (openclaw semantics)
    — callers must add lowest-precedence roots (e.g. bundled) first.
    """

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.skill_dirs: List[str] = []
        # name → reason for skills whose runtime requirements failed.
        self.ineligible: Dict[str, str] = {}
        # name → reason for skills that failed the load-time content scan
        # (invisible-Unicode / remote-exec). Kept out of the model catalog.
        self.quarantined: Dict[str, str] = {}
        # Human-readable loader diagnostics (spec violations, skipped files).
        self.warnings: List[str] = []
        self._seen_dirs: set[str] = set()
        self._parsed_files = 0
        self._reused_files = 0
        self._collisions: list[str] = []
        self.catalog_snapshot = CatalogSnapshot(
            generation=0,
            content_hash=hashlib.sha256(b"[]").hexdigest(),
            scan_duration_ms=0.0,
            parsed_files=0,
            reused_files=0,
            collisions=(),
        )

    @property
    def diagnostics(self) -> CatalogSnapshot:
        return self.catalog_snapshot

    def add_directory(self, d: str | Path):
        path = Path(d)
        if not path.exists():
            return
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in self._seen_dirs:
            return
        self._seen_dirs.add(key)
        self.skill_dirs.append(str(path))

    def _load_skill_file(self, skill_file: Path):
        try:
            if skill_file.stat().st_size > MAX_SKILL_FILE_BYTES:
                self.warnings.append(
                    f"{skill_file}: skipped (exceeds {MAX_SKILL_FILE_BYTES // 1024}KB limit)"
                )
                return
            skill, reused = _read_parsed_skill(skill_file)
        except (OSError, UnicodeDecodeError):
            return
        if reused:
            self._reused_files += 1
        else:
            self._parsed_files += 1
        if not skill.name.strip():
            self.warnings.append(f"{skill_file}: skipped (empty skill name)")
            return

        # ── Trust boundary: inspect content before it can reach the prompt ──
        # 1. Scan the *raw* text (before stripping) so smuggled instructions
        #    can't be hidden from the scanner by the same chars we'd remove.
        findings = scan_skill_content(skill.name, skill.description, skill.content)
        # 2. Strip invisible/zero-width/control chars from everything that gets
        #    injected — pure hygiene, always on.
        skill.name, n_name = _strip_invisible(skill.name)
        skill.description, n_desc = _strip_invisible(skill.description)
        skill.content, n_body = _strip_invisible(skill.content)
        if not skill.name.strip():
            self.warnings.append(f"{skill_file}: skipped (skill name empty after sanitize)")
            return
        if n_name or n_desc or n_body:
            self.warnings.append(
                f"{skill_file}: stripped {n_name + n_desc + n_body} invisible/control "
                f"char(s) from skill text"
            )

        # A later directory is authoritative for a normalized skill identity,
        # even when the replacement is unavailable or quarantined.  Remove all
        # lower-precedence states before evaluating the replacement so an old
        # runnable body cannot remain active beside a newer blocked copy.
        identity = _norm_skill_key(skill.name)
        for mapping in (self.skills, self.ineligible, self.quarantined):
            for previous_name in list(mapping):
                if _norm_skill_key(previous_name) == identity:
                    self._collisions.append(
                        f'{identity}: "{previous_name}" shadowed by "{skill.name}"'
                    )
                    mapping.pop(previous_name, None)

        for w in skill.warnings:
            self.warnings.append(f"{skill_file}: {w}")

        if findings and _skill_scan_enabled():
            reason = "; ".join(findings)
            self.quarantined[skill.name] = reason
            self.warnings.append(
                f"{skill_file}: QUARANTINED (content scan) — {reason}. "
                f"Set CLAW_SKILL_SCAN=off to load anyway after review."
            )
            # A quarantined load must not leave the name usable.
            self.skills.pop(skill.name, None)
            return
        if findings:
            # Scan disabled by env: load but surface the findings loudly.
            self.warnings.append(
                f"{skill_file}: content-scan findings ignored (CLAW_SKILL_SCAN=off) — "
                + "; ".join(findings)
            )

        reason = skill_ineligibility_reason(skill)
        if reason is not None:
            self.ineligible[skill.name] = reason
            return
        self.skills[skill.name] = skill
        # A clean load supersedes stale ineligible/quarantine records.
        self.ineligible.pop(skill.name, None)
        self.quarantined.pop(skill.name, None)

    async def load_all(self):
        started = time.perf_counter()
        self.skills.clear()
        self.ineligible.clear()
        self.quarantined.clear()
        self.warnings.clear()
        self._parsed_files = 0
        self._reused_files = 0
        self._collisions.clear()
        for d in self.skill_dirs:
            p = Path(d)
            if not p.exists() or not p.is_dir():
                continue

            # Directory itself is a skill (…/caveman/SKILL.md)
            self_skill = p / "SKILL.md"
            if self_skill.exists():
                self._load_skill_file(self_skill)

            try:
                entries = sorted(p.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir():
                        skill_file = entry / "SKILL.md"
                        if skill_file.exists():
                            self._load_skill_file(skill_file)
                    elif entry.suffix == ".md" and entry.name.lower() not in (
                        "skill.md",  # already loaded by the dir-skill branch
                        "readme.md", "agents.md", "claude.md",  # docs, not skills
                    ):
                        self._load_skill_file(entry)
                except OSError:
                    continue

        snapshot_rows = sorted(
            (
                "loaded",
                _norm_skill_key(skill.name),
                skill.name,
                skill.description,
                skill.path,
                tuple(skill.allowed_tools or ()),
                tuple(skill.aliases),
                tuple(skill.triggers),
                tuple(skill.anti_triggers),
                hashlib.sha256(skill.content.encode("utf-8")).hexdigest(),
            )
            for skill in self.skills.values()
        )
        snapshot_rows.extend(
            ("ineligible", _norm_skill_key(name), reason)
            for name, reason in sorted(self.ineligible.items())
        )
        snapshot_rows.extend(
            ("quarantined", _norm_skill_key(name), reason)
            for name, reason in sorted(self.quarantined.items())
        )
        content_hash = hashlib.sha256(
            json.dumps(snapshot_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.catalog_snapshot = CatalogSnapshot(
            generation=_next_catalog_generation(),
            content_hash=content_hash,
            scan_duration_ms=(time.perf_counter() - started) * 1000,
            parsed_files=self._parsed_files,
            reused_files=self._reused_files,
            collisions=tuple(dict.fromkeys(self._collisions)),
        )

    def list(self) -> List[Skill]:
        """Model-invocable skills (feeds the catalog and skill tools)."""
        return [s for s in self.skills.values() if not s.disable_model_invocation]

    def list_all(self) -> List[Skill]:
        """Every loaded skill, including user-invocation-only ones."""
        return list(self.skills.values())

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)


def _list_skill_resources(skill: Skill) -> List[str]:
    """Relative paths of files bundled with a dir-based skill (scripts/,
    references/, assets/, …) so the agent can read or run them."""
    if not skill.is_dir_skill:
        return []
    base = Path(skill.base_dir)
    out: List[str] = []
    try:
        for path in sorted(base.rglob("*")):
            if len(out) >= _MAX_RESOURCE_ENTRIES:
                out.append("…")
                break
            if not path.is_file() or path.name == "SKILL.md":
                continue
            rel = path.relative_to(base)
            if any(part.startswith(".") for part in rel.parts):
                continue
            out.append(str(rel))
    except OSError:
        return out
    return out


def _default_search_score(skill: Skill, query: str) -> float:
    """Fallback scorer for callers outside ClawAgent's richer ranker."""
    terms = {
        token[:-1] if len(token) > 3 and token.endswith("s") else token
        for token in re.findall(r"[a-z0-9]{3,}", query.lower())
    }
    fields = " ".join(
        [skill.name, skill.description, *skill.aliases, *skill.triggers]
    ).lower()
    field_terms = {
        token[:-1] if len(token) > 3 and token.endswith("s") else token
        for token in re.findall(r"[a-z0-9]{3,}", fields)
    }
    if any(trigger.lower() in query.lower() for trigger in skill.anti_triggers):
        return -100.0
    return float(len(terms & field_terms))


def create_skill_tools(
    store: SkillStore,
    relevance_scorer: Callable[[Any, str], float] | None = None,
    available_tool_names: Callable[[], set[str]] | None = None,
) -> List[Tool]:
    score_skill = relevance_scorer or _default_search_score

    class ListSkillsTool:
        name = "list_skills"
        description = (
            "Search and page through available skill names and short descriptions. "
            "Prefer the skills catalog already in the system prompt; call this "
            "only when that catalog was truncated or you need a skill not shown."
        )
        parameters: Dict[str, Dict[str, Any]] = {
            "query": {
                "type": "string",
                "description": "Optional case-insensitive name/description search",
                "required": False,
            },
            "offset": {
                "type": "integer",
                "description": "Zero-based result offset (default 0)",
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Page size, 1-25 (default 20)",
                "required": False,
            },
        }

        async def execute(self, args: Dict[str, Any]) -> ToolResult:
            skills = sorted(store.list(), key=lambda item: item.name.lower())
            if not skills and not store.ineligible and not store.quarantined:
                return ToolResult(success=True, output="No skills available.")

            query = str(args.get("query", "") or "").strip().lower()
            relevance_scores: dict[int, float] = {}
            try:
                offset = max(0, int(args.get("offset", 0) or 0))
                limit = max(1, min(25, int(args.get("limit", 20) or 20)))
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    output="",
                    error="offset and limit must be integers",
                )
            if query:
                scored = [(score_skill(skill, query), skill) for skill in skills]
                scored.sort(key=lambda pair: (-pair[0], pair[1].name.lower()))
                skills = [skill for score, skill in scored if score > 0]
                relevance_scores = {id(skill): score for score, skill in scored if score > 0}

            total = len(skills)
            page = skills[offset : offset + limit]
            lines = []
            for s in page:
                description = " ".join((s.description or "(no description)").split())
                if len(description) > 240:
                    description = description[:239].rstrip() + "…"
                line = f"- **{s.name}**: {description}"
                if query:
                    score = relevance_scores.get(id(s), 0.0)
                    confidence = "strong" if score >= 50 else "relevant" if score >= 12 else "possible"
                    line += f"\n  → Match: {confidence} ({score:.1f})"
                if s.allowed_tools:
                    shown_tools = s.allowed_tools[:12]
                    tools_text = ", ".join(shown_tools)
                    if len(s.allowed_tools) > len(shown_tools):
                        tools_text += ", …"
                    line += f"\n  → Allowed tools: {tools_text}"
                lines.append(line)
            end = min(total, offset + len(page))
            output = (
                f"Available skills matching query ({offset}-{end} of {total}):\n"
                + ("\n".join(lines) if lines else "No matching skills.")
            )
            if end < total:
                output += f"\n\nMore results: call list_skills with offset={end}."
            if not query and offset == 0 and store.ineligible:
                unavailable_items = sorted(store.ineligible.items())
                unavailable = "\n".join(
                    f"- **{name}**: {reason}"
                    for name, reason in unavailable_items[:10]
                )
                output += f"\n\nUnavailable (requirements not met):\n{unavailable}"
                if len(unavailable_items) > 10:
                    output += f"\n- …and {len(unavailable_items) - 10} more unavailable skills"
            if not query and offset == 0 and store.quarantined:
                quarantined_items = sorted(store.quarantined.items())
                blocked = "\n".join(
                    f"- **{name}**: {reason}"
                    for name, reason in quarantined_items[:10]
                )
                output += (
                    "\n\nQuarantined (failed security content scan — not loaded):\n"
                    + blocked
                )
                if len(quarantined_items) > 10:
                    output += f"\n- …and {len(quarantined_items) - 10} more quarantined skills"
            return ToolResult(success=True, output=output)

    class UseSkillTool:
        name = "use_skill"
        description = (
            "Read instructions for one skill by name in contiguous pages. Call this early when "
            "a listed skill matches the user's task (project setup, cohort/SQL "
            "workflows, document formats, etc.) — do not reinvent that workflow. "
            "Follow next_offset until complete. Names are matched case-insensitively; "
            "hyphens/underscores are equivalent."
        )
        parameters = {
            "name": {"type": "string", "description": "Name of the skill to load", "required": True},
            "offset": {
                "type": "integer",
                "description": "Character offset in rendered instructions (default 0)",
                "required": False,
            },
            "max_chars": {
                "type": "integer",
                "description": "Page size, 1000-10000 characters (default 10000)",
                "required": False,
            },
            "expected_hash": {
                "type": "string",
                "description": "Required content hash for continuation pages",
                "required": False,
            },
        }

        async def execute(self, args: Dict[str, Any], run_context: Any = None) -> ToolResult:
            name = str(args.get("name", "")).strip()
            skill = resolve_skill(store, name)

            if skill is not None and skill.disable_model_invocation:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f'Skill "{skill.name}" sets disable-model-invocation and can '
                        "only be invoked by the user, not by the model."
                    ),
                )

            if not skill:
                alias_matches = resolve_skill_aliases(store, name)
                if len(alias_matches) > 1:
                    return ToolResult(
                        success=False,
                        output="",
                        error=(
                            f'Ambiguous skill alias "{name}". Use a canonical name: '
                            + ", ".join(sorted(item.name for item in alias_matches))
                        ),
                    )
                available = sorted(s.name for s in store.list())
                suggestions = suggest_skills(store, name, limit=5)
                hint = ""
                if suggestions:
                    hint = " Did you mean: " + ", ".join(suggestions) + "?"
                note = ""
                for qname, reason in store.quarantined.items():
                    if _norm_skill_key(qname) == _norm_skill_key(name):
                        note = (
                            f' Skill "{qname}" was QUARANTINED by the content '
                            f"scanner and cannot be loaded: {reason}."
                        )
                        break
                if not note:
                    for iname, reason in store.ineligible.items():
                        if _norm_skill_key(iname) == _norm_skill_key(name):
                            note = f' Skill "{iname}" exists but is unavailable: {reason}.'
                            break
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f'Skill "{name}" not found.{note}{hint} '
                        f"Available: {', '.join(available) if available else 'none'}"
                    ),
                )

            effective_allowed_tools = (
                list(skill.allowed_tools) if skill.allowed_tools is not None else None
            )
            if effective_allowed_tools is not None:
                control_keys = {_norm_skill_key("use_skill"), _norm_skill_key("list_skills")}
                if available_tool_names is not None:
                    available_map = {
                        _norm_skill_key(tool_name): tool_name
                        for tool_name in available_tool_names()
                    }
                    canonical: list[str] = []
                    unknown: list[str] = []
                    for declared in effective_allowed_tools:
                        key = _norm_skill_key(declared)
                        if key in control_keys:
                            continue
                        resolved = available_map.get(key)
                        if resolved is None:
                            unknown.append(declared)
                        elif resolved not in canonical:
                            canonical.append(resolved)
                    if unknown:
                        return ToolResult(
                            success=False,
                            output="",
                            error=(
                                f"Skill '{skill.name}' declares unknown allowed-tools: "
                                + ", ".join(unknown)
                            ),
                        )
                    effective_allowed_tools = canonical
                else:
                    effective_allowed_tools = [
                        tool_name
                        for tool_name in effective_allowed_tools
                        if _norm_skill_key(tool_name) not in control_keys
                    ]

            parts = [f"# Skill: {skill.name}"]
            if effective_allowed_tools is not None:
                parts.append(
                    "Active allowed-tools boundary: "
                    + (", ".join(effective_allowed_tools) or "no data-plane tools")
                )
            # Resources referenced by the skill body (scripts/…, references/…)
            # resolve relative to this directory — without it the agent cannot
            # locate them (Claude Code prepends the same line).
            parts.append(f"Base directory for this skill: {skill.base_dir}")
            resources = _list_skill_resources(skill)
            if resources:
                parts.append(
                    "Bundled resources (relative to base directory): "
                    + ", ".join(resources)
                )

            if skill.forbidden_actions:
                parts.append("\n## Forbidden Actions")
                for action in skill.forbidden_actions:
                    parts.append(f"- {action}")

            if skill.workspace_layout:
                parts.append("\n## Workspace Layout")
                parts.append(skill.workspace_layout)

            if skill.success_criteria:
                parts.append("\n## Success Criteria")
                parts.append(skill.success_criteria)

            if skill.workflow_steps:
                parts.append("\n## Workflow Steps")
                for i, step in enumerate(skill.workflow_steps, 1):
                    parts.append(f"{i}. {step}")

            parts.append(f"\n{skill.content}")
            rendered = "\n".join(parts)
            content_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            try:
                offset = max(0, int(args.get("offset", 0) or 0))
                max_chars = max(1000, min(10_000, int(args.get("max_chars", 10_000) or 10_000)))
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    output="",
                    error="offset and max_chars must be integers",
                )
            if offset > len(rendered):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"offset {offset} exceeds instruction length {len(rendered)}",
                )

            expected_hash = str(args.get("expected_hash", "") or "")
            pending_name = getattr(run_context, "pending_skill_name", None)
            if pending_name:
                if (
                    _norm_skill_key(pending_name) != _norm_skill_key(skill.name)
                    or offset != getattr(run_context, "pending_skill_next_offset", None)
                    or expected_hash != getattr(run_context, "pending_skill_content_hash", None)
                    or content_hash != getattr(run_context, "pending_skill_content_hash", None)
                ):
                    return ToolResult(
                        success=False,
                        output="",
                        error=(
                            "Skill continuation is not contiguous or its content hash "
                            "changed; restart at offset 0."
                        ),
                    )
            elif offset != 0:
                return ToolResult(
                    success=False,
                    output="",
                    error="Skill loading must start at offset 0.",
                )

            end = min(len(rendered), offset + max_chars)
            if end < len(rendered):
                paragraph = rendered.rfind("\n\n", offset + max_chars // 2, end)
                if paragraph > offset:
                    end = paragraph + 2
            chunk = rendered[offset:end]
            page_header = (
                f"[Skill {skill.name} sha256={content_hash}: "
                f"characters {offset}-{end} of {len(rendered)}]\n"
            )
            if end < len(rendered):
                continuation = (
                    f"\n\n[More instructions remain. Call use_skill with "
                    f'name="{skill.name}", offset={end}, and '
                    f'expected_hash="{content_hash}".]'
                )
                next_offset: int | None = end
            else:
                continuation = "\n\n[End of skill instructions.]"
                next_offset = None
            if run_context is not None and hasattr(run_context, "record_skill_page"):
                run_context.record_skill_page(
                    skill.name,
                    effective_allowed_tools,
                    content_hash,
                    next_offset=next_offset,
                    total_chars=len(rendered),
                )
            return ToolResult(success=True, output=page_header + chunk + continuation)

    return [ListSkillsTool(), UseSkillTool()]


def _norm_skill_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", (name or "").strip().lower())


def resolve_skill(store: SkillStore, name: str) -> Optional[Skill]:
    """Resolve a skill by name or an explicit alias."""
    raw = (name or "").strip()
    if not raw:
        return None
    hit = store.get(raw)
    if hit:
        return hit
    skills = store.list()
    lower_map = {s.name.lower(): s for s in skills}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]
    norm_map = {_norm_skill_key(s.name): s for s in skills}
    normalized = _norm_skill_key(raw)
    if normalized in norm_map:
        return norm_map[normalized]
    alias_matches = resolve_skill_aliases(store, raw)
    return alias_matches[0] if len(alias_matches) == 1 else None


def resolve_skill_aliases(store: SkillStore, name: str) -> List[Skill]:
    normalized = _norm_skill_key(name)
    return [
        skill
        for skill in store.list()
        if any(_norm_skill_key(alias) == normalized for alias in skill.aliases)
    ]


def suggest_skills(store: SkillStore, name: str, limit: int = 5) -> List[str]:
    """Close name matches for use_skill typos / near-misses."""
    import difflib

    raw = (name or "").strip()
    if not raw:
        return []
    skills = store.list()
    names = [s.name for s in skills]
    if not names:
        return []
    direct = difflib.get_close_matches(raw, names, n=limit, cutoff=0.45)
    if direct:
        return direct
    alias_to_names: dict[str, list[str]] = {}
    for skill in skills:
        for alias in skill.aliases:
            alias_to_names.setdefault(alias, []).append(skill.name)
    alias_hits = difflib.get_close_matches(raw, list(alias_to_names), n=limit, cutoff=0.45)
    if alias_hits:
        return list(
            dict.fromkeys(
                name
                for alias in alias_hits
                for name in sorted(alias_to_names[alias])
            )
        )[:limit]
    norm_names = {_norm_skill_key(n): n for n in names}
    soft = difflib.get_close_matches(_norm_skill_key(raw), list(norm_names), n=limit, cutoff=0.45)
    return [norm_names[k] for k in soft]
