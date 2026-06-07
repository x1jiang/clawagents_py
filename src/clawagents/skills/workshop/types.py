from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ProposalStatus = Literal["pending", "applied", "rejected", "quarantined", "stale"]
ProposalAction = Literal["create", "update", "revise", "list", "inspect", "apply", "reject", "quarantine"]
SupportFolder = Literal["assets", "examples", "references", "scripts", "templates"]

SUPPORT_FOLDERS: tuple[SupportFolder, ...] = (
    "assets",
    "examples",
    "references",
    "scripts",
    "templates",
)


@dataclass
class SupportFile:
    path: str
    content: str


@dataclass
class SkillProposalRecord:
    id: str
    name: str
    description: str
    status: ProposalStatus
    action: Literal["create", "update"]
    target_skill: Optional[str] = None
    target_hash: Optional[str] = None
    goal: str = ""
    evidence: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    scan_findings: list[str] = field(default_factory=list)
    support_files: list[SupportFile] = field(default_factory=list)
