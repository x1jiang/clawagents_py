from __future__ import annotations

from pathlib import Path
from typing import Any

from clawagents.skills.workshop.impact import IMPACT_PREVIEW_ENTRIES, IMPACT_RELATIVE_PATH
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
            **self._impact_payload(),
        }

    def apply(self, proposal_id: str) -> dict[str, Any]:
        rec = self.store.get(proposal_id)
        if not rec:
            return {"ok": False, "error": "not found"}
        before_md = self.store.read_skill_md(self._skill_name(rec))
        proposed_md = self.store.proposal_body(proposal_id)
        if rec.scan_findings:
            # Every finding the scanner emits is a real reason to refuse writing
            # the proposal to a live SKILL.md — most importantly the
            # "suspicious pattern …" findings (rm -rf, ``curl … | sh``, ``eval(``,
            # ``__import__`` …) and the oversize/too-many/bad-path ones. The old
            # substring gate ("exceeds"/"invalid"/"must be") let the security and
            # resource findings through, making the malicious-pattern check
            # cosmetic. Block on any finding.
            self.store.append_impact(
                outcome="apply_blocked",
                skill_name=self._skill_name(rec),
                action=rec.action,
                proposal_id=rec.id,
                reason="scan blocked apply",
                scan_findings=rec.scan_findings,
                old_skill_md=before_md,
                new_skill_md=proposed_md,
            )
            return {"ok": False, "error": "scan blocked apply", "findings": rec.scan_findings}
        ok, msg, rollback_id = self.store.apply_proposal(proposal_id)
        refreshed = self.store.get(proposal_id) or rec
        if ok:
            outcome = "applied"
        elif refreshed.status == "stale" or "stale" in msg:
            outcome = "stale"
        else:
            outcome = "apply_blocked"
        self.store.append_impact(
            outcome=outcome,
            skill_name=self._skill_name(rec),
            action=rec.action,
            proposal_id=rec.id,
            rollback_id=rollback_id or "",
            reason=msg,
            scan_findings=refreshed.scan_findings,
            old_skill_md=before_md,
            new_skill_md=proposed_md,
        )
        return {"ok": ok, "message": msg, "rollback_id": rollback_id}

    def reject(self, proposal_id: str, reason: str = "") -> dict[str, Any]:
        rec = self.store.get(proposal_id)
        if not rec:
            return {"ok": False, "error": "not found"}
        before_md = self.store.read_skill_md(self._skill_name(rec))
        updated = self.store.update_status(proposal_id, "rejected", reason=reason)
        if not updated:
            return {"ok": False, "error": "not found"}
        self.store.append_impact(
            outcome="rejected",
            skill_name=self._skill_name(updated),
            action=updated.action,
            proposal_id=updated.id,
            reason=reason,
            scan_findings=updated.scan_findings,
            old_skill_md=before_md,
            new_skill_md=self.store.proposal_body(proposal_id),
        )
        return {"ok": True, "status": "rejected", "reason": reason}

    def quarantine(self, proposal_id: str, reason: str = "") -> dict[str, Any]:
        rec = self.store.get(proposal_id)
        if not rec:
            return {"ok": False, "error": "not found"}
        before_md = self.store.read_skill_md(self._skill_name(rec))
        updated = self.store.update_status(proposal_id, "quarantined", reason=reason)
        if not updated:
            return {"ok": False, "error": "not found"}
        self.store.append_impact(
            outcome="quarantined",
            skill_name=self._skill_name(updated),
            action=updated.action,
            proposal_id=updated.id,
            reason=reason,
            scan_findings=updated.scan_findings,
            old_skill_md=before_md,
            new_skill_md=self.store.proposal_body(proposal_id),
        )
        return {"ok": True, "status": "quarantined", "reason": reason}

    def rollback(self, rollback_id: str) -> dict[str, Any]:
        snap = self.store.load_rollback(rollback_id)
        if not snap:
            return {"ok": False, "error": "rollback not found"}
        skill_name = str(snap.get("name") or "")
        before_md = self.store.read_skill_md(skill_name)
        self.store.restore_snapshot(snap)
        restored_md = str(snap.get("files", {}).get("SKILL.md") or "")
        self.store.append_impact(
            outcome="rolled_back",
            skill_name=skill_name,
            rollback_id=rollback_id,
            reason="restored skill snapshot",
            old_skill_md=before_md,
            new_skill_md=restored_md,
        )
        return {"ok": True, "restored": skill_name}

    def impact(self, limit: int | None = IMPACT_PREVIEW_ENTRIES) -> dict[str, Any]:
        return {
            "ok": True,
            **self._impact_payload(limit),
            "content": self.store.read_impact(limit),
        }

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
            "reason": rec.reason,
        }

    def _impact_payload(self, limit: int | None = IMPACT_PREVIEW_ENTRIES) -> dict[str, Any]:
        return {
            "skill_impact_path": str(self.store.impact_path),
            "skill_impact_relative_path": IMPACT_RELATIVE_PATH,
            "skill_impact_preview": self.store.read_impact(limit),
        }

    @staticmethod
    def _skill_name(rec: SkillProposalRecord) -> str:
        if rec.action == "update" and rec.target_skill:
            return rec.target_skill
        return rec.name

    @staticmethod
    def _blocked(findings: list[str]) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "scan blocked proposal",
            "findings": list(dict.fromkeys(findings)),
        }
