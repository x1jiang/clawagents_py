import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from clawagents.tools.registry import Tool, ToolResult

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

def parse_skill_file(content: str, file_path: str) -> Skill:
    default_name = Path(file_path).stem
    name = default_name
    description = ""
    body = content
    allowed_tools: List[str] = []
    requires: Optional[SkillRequires] = None

    frontmatter_match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$", content)
    if frontmatter_match:
        yaml_content = frontmatter_match.group(1) or ""
        body = frontmatter_match.group(2) or ""

        name_match = re.search(r"^name:\s*(.+)$", yaml_content, re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()

        desc_match = re.search(r'^description:\s*"?([^"]+)"?$', yaml_content, re.MULTILINE)
        if desc_match:
            description = desc_match.group(1).strip()

        # Parse allowed-tools: space/comma-delimited string
        tools_match = re.search(r"^allowed-tools:\s*(.+)$", yaml_content, re.MULTILINE)
        if tools_match:
            allowed_tools = [t.strip(",") for t in tools_match.group(1).split() if t.strip(",")]

        # Parse requires block for eligibility gating
        os_match = re.search(r"^requires\.os:\s*(.+)$", yaml_content, re.MULTILINE) \
            or re.search(r"^\s+os:\s*(.+)$", yaml_content, re.MULTILINE)
        bins_match = re.search(r"^requires\.bins:\s*(.+)$", yaml_content, re.MULTILINE) \
            or re.search(r"^\s+bins:\s*(.+)$", yaml_content, re.MULTILINE)
        env_match = re.search(r"^requires\.env:\s*(.+)$", yaml_content, re.MULTILINE) \
            or re.search(r"^\s+env:\s*(.+)$", yaml_content, re.MULTILINE)

        if os_match or bins_match or env_match:
            def _parse_list(raw: str) -> List[str]:
                cleaned = re.sub(r'[\[\]"\']', "", raw)
                return [x.strip() for x in re.split(r"[\s,]+", cleaned) if x.strip()]

            requires = SkillRequires(
                os=os_match.group(1).strip() if os_match else None,
                bins=_parse_list(bins_match.group(1)) if bins_match else None,
                env=_parse_list(env_match.group(1)) if env_match else None,
            )

    return Skill(name=name, description=description, content=body.strip(), path=file_path,
                 allowed_tools=allowed_tools, requires=requires)


def is_skill_eligible(skill: Skill) -> bool:
    if not skill.requires:
        return True
    req = skill.requires
    if req.os and sys.platform != req.os:
        return False
    if req.bins:
        for b in req.bins:
            if shutil.which(b) is None:
                return False
    if req.env:
        for var in req.env:
            if not os.environ.get(var):
                return False
    return True


class SkillStore:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.skill_dirs: List[str] = []

    def add_directory(self, d: str | Path):
        path = Path(d)
        if path.exists():
            self.skill_dirs.append(str(path))

    async def load_all(self):
        for d in self.skill_dirs:
            p = Path(d)
            if not p.exists() or not p.is_dir():
                continue
                
            try:
                entries = list(p.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir():
                        skill_file = entry / "SKILL.md"
                        if skill_file.exists():
                            content = skill_file.read_text("utf-8")
                            skill = parse_skill_file(content, str(skill_file))
                            if is_skill_eligible(skill):
                                self.skills[skill.name] = skill
                    elif entry.suffix == ".md":
                        content = entry.read_text("utf-8")
                        skill = parse_skill_file(content, str(entry))
                        if is_skill_eligible(skill):
                            self.skills[skill.name] = skill
                except (OSError, UnicodeDecodeError):
                    continue

    def list(self) -> List[Skill]:
        return list(self.skills.values())

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)


def create_skill_tools(store: SkillStore) -> List[Tool]:
    
    class ListSkillsTool:
        name = "list_skills"
        description = "List all available skills the agent can use."
        parameters = {}

        async def execute(self, args: Dict[str, Any]) -> ToolResult:
            skills = store.list()
            if not skills:
                return ToolResult(success=True, output="No skills available.")
            
            lines = []
            for s in skills:
                line = f"- **{s.name}**: {s.description or '(no description)'}"
                if s.allowed_tools:
                    line += f"\n  → Allowed tools: {', '.join(s.allowed_tools)}"
                lines.append(line)
            return ToolResult(success=True, output=f"Available skills ({len(skills)}):\n" + "\n".join(lines))

    class UseSkillTool:
        name = "use_skill"
        description = "Load and read a specific skill to learn its instructions. Use list_skills first to see what's available."
        parameters = {
            "name": {"type": "string", "description": "Name of the skill to load", "required": True}
        }

        async def execute(self, args: Dict[str, Any]) -> ToolResult:
            name = str(args.get("name", ""))
            skill = store.get(name)
            
            if not skill:
                available = ", ".join([s.name for s in store.list()])
                return ToolResult(success=False, output="", error=f"Skill \"{name}\" not found. Available: {available or 'none'}")
            
            return ToolResult(success=True, output=f"# Skill: {skill.name}\n\n{skill.content}")

    return [ListSkillsTool(), UseSkillTool()]
