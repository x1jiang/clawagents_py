import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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
    allowed_tools: List[str] = field(default_factory=list)
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
    allowed_tools: List[str] = []
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

        # Parse allowed-tools: space/comma-delimited string
        tools_match = re.search(r"^allowed-tools:\s*(.+)$", yaml_content, re.MULTILINE)
        if tools_match:
            allowed_tools = [t.strip(",") for t in tools_match.group(1).split() if t.strip(",")]

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
            content = skill_file.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            return
        skill = parse_skill_file(content, str(skill_file))
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


def create_skill_tools(store: SkillStore) -> List[Tool]:

    class ListSkillsTool:
        name = "list_skills"
        description = (
            "List all available skill names and short descriptions. "
            "Prefer the skills catalog already in the system prompt; call this "
            "only when that catalog was truncated or you need a skill not shown."
        )
        parameters: Dict[str, Dict[str, Any]] = {}

        async def execute(self, args: Dict[str, Any]) -> ToolResult:
            skills = store.list()
            if not skills and not store.ineligible and not store.quarantined:
                return ToolResult(success=True, output="No skills available.")

            lines = []
            for s in skills:
                line = f"- **{s.name}**: {s.description or '(no description)'}"
                if s.allowed_tools:
                    line += f"\n  → Allowed tools: {', '.join(s.allowed_tools)}"
                lines.append(line)
            output = f"Available skills ({len(skills)}):\n" + "\n".join(lines)
            if store.ineligible:
                unavailable = "\n".join(
                    f"- **{name}**: {reason}"
                    for name, reason in sorted(store.ineligible.items())
                )
                output += f"\n\nUnavailable (requirements not met):\n{unavailable}"
            if store.quarantined:
                blocked = "\n".join(
                    f"- **{name}**: {reason}"
                    for name, reason in sorted(store.quarantined.items())
                )
                output += (
                    "\n\nQuarantined (failed security content scan — not loaded):\n"
                    + blocked
                )
            return ToolResult(success=True, output=output)

    class UseSkillTool:
        name = "use_skill"
        description = (
            "Load full instructions for one skill by name. Call this early when "
            "a listed skill matches the user's task (project setup, cohort/SQL "
            "workflows, document formats, etc.) — do not reinvent that workflow. "
            "Names are matched case-insensitively; hyphens/underscores are equivalent."
        )
        parameters = {
            "name": {"type": "string", "description": "Name of the skill to load", "required": True}
        }

        async def execute(self, args: Dict[str, Any]) -> ToolResult:
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

            parts = [f"# Skill: {skill.name}"]
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
            return ToolResult(success=True, output="\n".join(parts))

    return [ListSkillsTool(), UseSkillTool()]


def _norm_skill_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", (name or "").strip().lower())


def resolve_skill(store: SkillStore, name: str) -> Optional[Skill]:
    """Resolve a skill by exact, case-insensitive, or hyphen/underscore-normalized name."""
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
    return norm_map.get(_norm_skill_key(raw))


def suggest_skills(store: SkillStore, name: str, limit: int = 5) -> List[str]:
    """Close name matches for use_skill typos / near-misses."""
    import difflib

    raw = (name or "").strip()
    if not raw:
        return []
    names = [s.name for s in store.list()]
    if not names:
        return []
    direct = difflib.get_close_matches(raw, names, n=limit, cutoff=0.45)
    if direct:
        return direct
    norm_names = {_norm_skill_key(n): n for n in names}
    soft = difflib.get_close_matches(_norm_skill_key(raw), list(norm_names), n=limit, cutoff=0.45)
    return [norm_names[k] for k in soft]
