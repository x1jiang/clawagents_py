from __future__ import annotations

from pathlib import Path
from typing import Any

from clawagents.skills.workshop.scanner import scan_proposal_content
from clawagents.skills.workshop.store import ProposalValidationError, SkillWorkshopStore
from clawagents.skills.workshop.types import SkillProposalRecord


class SkillWorkshopService:
    def __init__(self, workspace: str | Path, skills_dir: str | Path | None = None) -> None:
        self.store = SkillWorkshopStore(workspace, skills_dir)

    def create(
        self,
        *,
        name: str,
        description: str,
        body: str,
        goal: str = "",
        evidence: str = "",
        support_files: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        pairs = [(f["path"], f["content"]) for f in (support_files or [])]
        findings = scan_proposal_content(name, description, body, pairs)
        try:
            rec = self.store.create_proposal(
                name=name,
                description=description,
                body=body,
                action="create",
                goal=goal,
                evidence=evidence,
                support_files=pairs,
                scan_findings=findings,
            )
        except ProposalValidationError as exc:
            return self._blocked(exc.findings)
        return self._serialize(rec, findings)

    def update(
        self,
        *,
        target_skill: str,
        description: str,
        body: str,
        goal: str = "",
        evidence: str = "",
        support_files: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        pairs = [(f["path"], f["content"]) for f in (support_files or [])]
        findings = scan_proposal_content(target_skill, description, body, pairs)
        try:
            rec = self.store.create_proposal(
                name=target_skill,
                description=description,
                body=body,
                action="update",
                target_skill=target_skill,
                goal=goal,
                evidence=evidence,
                support_files=pairs,
                scan_findings=findings,
            )
        except ProposalValidationError as exc:
            return self._blocked(exc.findings)
        return self._serialize(rec, findings)

    def revise(self, proposal_id: str, *, body: str, description: str | None = None) -> dict[str, Any]:
        rec = self.store.get(proposal_id)
        if not rec or rec.status != "pending":
            return {"ok": False, "error": "proposal not pending"}
        pairs = [(s.path, s.content) for s in rec.support_files]
        desc = description if description is not None else rec.description
        findings = scan_proposal_content(rec.name, desc, body, pairs)
        self.store._body_path(proposal_id).write_text(body, encoding="utf-8")
        meta_path = self.store._meta_path(proposal_id)
        import json
        import time

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["description"] = desc
        meta["scan_findings"] = findings
        meta["updated_at"] = time.time()
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        updated = self.store.get(proposal_id)
        assert updated
        return self._serialize(updated, findings)

    def list(self) -> list[dict[str, Any]]:
        return [self._serialize(r, r.scan_findings) for r in self.store.list_proposals()]

    def inspect(self, proposal_id: str) -> dict[str, Any]:
        rec = self.store.get(proposal_id)
        if not rec:
            return {"ok": False, "error": "not found"}
        return {
            **self._serialize(rec, rec.scan_findings),
            "body": self.store.proposal_body(proposal_id),
        }

    def apply(self, proposal_id: str) -> dict[str, Any]:
        rec = self.store.get(proposal_id)
        if not rec:
            return {"ok": False, "error": "not found"}
        if rec.scan_findings:
            # Every finding the scanner emits is a real reason to refuse writing
            # the proposal to a live SKILL.md — most importantly the
            # "suspicious pattern …" findings (rm -rf, ``curl … | sh``, ``eval(``,
            # ``__import__`` …) and the oversize/too-many/bad-path ones. The old
            # substring gate ("exceeds"/"invalid"/"must be") let the security and
            # resource findings through, making the malicious-pattern check
            # cosmetic. Block on any finding.
            if rec.scan_findings:
                return {"ok": False, "error": "scan blocked apply", "findings": rec.scan_findings}
        ok, msg, rollback_id = self.store.apply_proposal(proposal_id)
        return {"ok": ok, "message": msg, "rollback_id": rollback_id}

    def reject(self, proposal_id: str, reason: str = "") -> dict[str, Any]:
        rec = self.store.update_status(proposal_id, "rejected")
        if not rec:
            return {"ok": False, "error": "not found"}
        return {"ok": True, "status": "rejected", "reason": reason}

    def quarantine(self, proposal_id: str, reason: str = "") -> dict[str, Any]:
        rec = self.store.update_status(proposal_id, "quarantined")
        if not rec:
            return {"ok": False, "error": "not found"}
        return {"ok": True, "status": "quarantined", "reason": reason}

    def rollback(self, rollback_id: str) -> dict[str, Any]:
        snap = self.store.load_rollback(rollback_id)
        if not snap:
            return {"ok": False, "error": "rollback not found"}
        self.store.restore_snapshot(snap)
        return {"ok": True, "restored": snap.get("name")}

    def _serialize(self, rec: SkillProposalRecord, findings: list[str]) -> dict[str, Any]:
        return {
            "id": rec.id,
            "name": rec.name,
            "description": rec.description,
            "status": rec.status,
            "action": rec.action,
            "target_skill": rec.target_skill,
            "target_hash": rec.target_hash,
            "goal": rec.goal,
            "evidence": rec.evidence,
            "scan_findings": findings,
            "support_file_count": len(rec.support_files),
        }

    @staticmethod
    def _blocked(findings: list[str]) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "scan blocked proposal",
            "findings": list(dict.fromkeys(findings)),
        }
