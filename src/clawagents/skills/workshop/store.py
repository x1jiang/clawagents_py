from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from clawagents.skills.workshop.scanner import scan_proposal_content, support_path_findings
from clawagents.skills.workshop.types import SkillProposalRecord, SupportFile


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


class ProposalValidationError(ValueError):
    def __init__(self, findings: list[str]) -> None:
        super().__init__("; ".join(findings))
        self.findings = findings


class SkillWorkshopStore:
    """File-backed proposal store under `.clawagents/skill-workshop/`."""

    def __init__(self, workspace: str | Path, skills_dir: str | Path | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.skills_dir = Path(skills_dir or self.workspace / "skills").resolve()
        self.root = self.workspace / ".clawagents" / "skill-workshop"
        self.proposals_dir = self.root / "proposals"
        self.rollback_dir = self.root / "rollback"
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_dir.mkdir(parents=True, exist_ok=True)

    def _proposal_dir(self, proposal_id: str) -> Path:
        return self.proposals_dir / proposal_id

    def _meta_path(self, proposal_id: str) -> Path:
        return self._proposal_dir(proposal_id) / "meta.json"

    def _body_path(self, proposal_id: str) -> Path:
        return self._proposal_dir(proposal_id) / "PROPOSAL.md"

    def list_proposals(self) -> list[SkillProposalRecord]:
        out: list[SkillProposalRecord] = []
        if not self.proposals_dir.is_dir():
            return out
        for entry in sorted(self.proposals_dir.iterdir()):
            if entry.is_dir() and (entry / "meta.json").is_file():
                rec = self.get(entry.name)
                if rec:
                    out.append(rec)
        return out

    def get(self, proposal_id: str) -> Optional[SkillProposalRecord]:
        meta_path = self._meta_path(proposal_id)
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pairs, _ = self._support_snapshot(proposal_id)
        support = [SupportFile(path=path, content=content) for path, content in pairs]
        return SkillProposalRecord(
            id=meta["id"],
            name=meta["name"],
            description=meta.get("description", ""),
            status=meta.get("status", "pending"),
            action=meta.get("action", "create"),
            target_skill=meta.get("target_skill"),
            target_hash=meta.get("target_hash"),
            goal=meta.get("goal", ""),
            evidence=meta.get("evidence", ""),
            created_at=float(meta.get("created_at", 0)),
            updated_at=float(meta.get("updated_at", 0)),
            scan_findings=list(meta.get("scan_findings", [])),
            support_files=support,
        )

    def _support_snapshot(self, proposal_id: str) -> tuple[list[tuple[str, str]], list[str]]:
        support_root = self._proposal_dir(proposal_id) / "support"
        support: list[tuple[str, str]] = []
        findings: list[str] = []
        if not support_root.is_dir():
            return support, findings
        for path in sorted(support_root.rglob("*")):
            if not path.is_file() and not path.is_symlink():
                continue
            rel = path.relative_to(support_root).as_posix()
            path_findings = support_path_findings(rel, support_root)
            findings.extend(path_findings)
            if path_findings or not path.is_file():
                continue
            support.append((rel, path.read_text(encoding="utf-8")))
        return support, findings

    def create_proposal(
        self,
        *,
        name: str,
        description: str,
        body: str,
        action: str = "create",
        target_skill: str | None = None,
        goal: str = "",
        evidence: str = "",
        support_files: list[tuple[str, str]] | None = None,
        scan_findings: list[str] | None = None,
    ) -> SkillProposalRecord:
        proposal_id = uuid.uuid4().hex[:12]
        now = time.time()
        pairs = list(support_files or [])
        support_root = self._proposal_dir(proposal_id) / "support"
        path_findings = [
            finding
            for rel, _ in pairs
            for finding in support_path_findings(rel, support_root)
        ]
        if path_findings:
            raise ProposalValidationError(list(dict.fromkeys(path_findings)))
        target_hash = None
        if action == "update" and target_skill:
            skill_path = self.skills_dir / target_skill / "SKILL.md"
            if skill_path.is_file():
                target_hash = _sha256_text(_read_text(skill_path))
        meta: dict[str, Any] = {
            "id": proposal_id,
            "name": name,
            "description": description,
            "status": "pending",
            "action": action,
            "target_skill": target_skill,
            "target_hash": target_hash,
            "goal": goal,
            "evidence": evidence,
            "created_at": now,
            "updated_at": now,
            "scan_findings": scan_findings or [],
        }
        pdir = self._proposal_dir(proposal_id)
        pdir.mkdir(parents=True, exist_ok=True)
        self._body_path(proposal_id).write_text(body, encoding="utf-8")
        if pairs:
            for rel, content in pairs:
                parts = PurePosixPath(rel.replace("\\", "/")).parts
                dest = support_root.joinpath(*parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
        self._meta_path(proposal_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rec = self.get(proposal_id)
        assert rec is not None
        return rec

    def update_status(self, proposal_id: str, status: str) -> Optional[SkillProposalRecord]:
        meta_path = self._meta_path(proposal_id)
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["status"] = status
        meta["updated_at"] = time.time()
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return self.get(proposal_id)

    def proposal_body(self, proposal_id: str) -> str:
        return _read_text(self._body_path(proposal_id))

    def skill_path(self, name: str) -> Path:
        return self.skills_dir / name / "SKILL.md"

    def skill_hash(self, name: str) -> Optional[str]:
        path = self.skill_path(name)
        if not path.is_file():
            return None
        return _sha256_text(_read_text(path))

    def save_rollback(self, skill_name: str, snapshot: dict[str, Any]) -> str:
        rollback_id = f"{skill_name}-{int(time.time())}"
        path = self.rollback_dir / f"{rollback_id}.json"
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return rollback_id

    def load_rollback(self, rollback_id: str) -> Optional[dict[str, Any]]:
        path = self.rollback_dir / f"{rollback_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def snapshot_skill(self, name: str) -> dict[str, Any]:
        skill_root = self.skills_dir / name
        files: dict[str, str] = {}
        if skill_root.is_dir():
            for path in skill_root.rglob("*"):
                if path.is_file():
                    rel = str(path.relative_to(skill_root))
                    files[rel] = path.read_text(encoding="utf-8")
        return {"name": name, "files": files}

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        name = snapshot["name"]
        skill_root = self.skills_dir / name
        if skill_root.is_dir():
            for path in list(skill_root.rglob("*")):
                if path.is_file():
                    path.unlink()
            for path in sorted(skill_root.rglob("*"), reverse=True):
                if path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        for rel, content in snapshot.get("files", {}).items():
            dest = skill_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    def apply_proposal(self, proposal_id: str) -> tuple[bool, str, Optional[str]]:
        rec = self.get(proposal_id)
        if not rec:
            return False, "proposal not found", None
        if rec.status != "pending":
            return False, f"proposal status is {rec.status}", None
        if rec.action == "update" and rec.target_skill and rec.target_hash:
            current = self.skill_hash(rec.target_skill)
            if current and current != rec.target_hash:
                self.update_status(proposal_id, "stale")
                return False, "target skill changed since proposal; marked stale", None
        body = self.proposal_body(proposal_id)
        pairs, path_findings = self._support_snapshot(proposal_id)
        findings = list(path_findings)
        findings.extend(
            scan_proposal_content(rec.name, rec.description, body, pairs)
        )
        if findings:
            return False, f"scan blocked apply: {'; '.join(dict.fromkeys(findings))}", None
        skill_name = rec.target_skill if rec.action == "update" else rec.name
        rollback_id = self.save_rollback(skill_name, self.snapshot_skill(skill_name))
        skill_root = self.skills_dir / skill_name
        skill_root.mkdir(parents=True, exist_ok=True)
        self.skill_path(skill_name).write_text(body, encoding="utf-8")
        for rel, content in pairs:
            parts = PurePosixPath(rel.replace("\\", "/")).parts
            dest = skill_root.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        self.update_status(proposal_id, "applied")
        return True, f"applied skill {skill_name}", rollback_id
