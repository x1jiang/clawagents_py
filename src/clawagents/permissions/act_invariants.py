"""Deterministic Plan→Act verification contracts.

An approved plan may name exact commands under a ``Verification gates``
heading.  The registry records those commands only when they succeed after the
latest mutation, and refuses high-impact external actions until every gate is
fresh.  This is deliberately enforced below the prompt layer so compaction or
model drift cannot silently discard it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


_STATE_VERSION = 1
_STATE_PATH = Path(".clawagents") / "act-invariants.json"
_VERIFICATION_HEADING = re.compile(
    r"\b(?:verification|validation|evidence)\s+(?:gates?|checks?|commands?)\b",
    re.IGNORECASE,
)
_RECONCILIATION_HEADING = re.compile(
    r"\b(?:(?:post[- ]?(?:action|publish|deploy))\s+"
    r"(?:reconciliation|verification|checks?|commands?)|"
    r"(?:reconciliation|completion)\s+(?:gates?|checks?|commands?))\b",
    re.IGNORECASE,
)
_INVARIANT_HEADING = re.compile(
    r"\b(?:invariants?|success criteria|acceptance criteria|preconditions?|"
    r"safety requirements?|requirements?)\b",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?:\[[ xX]\]\s*)?(.*\S)\s*$")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_HIGH_IMPACT_PATTERNS = (
    re.compile(r"(?:^|[;&|]\s*)git\s+push\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)docker\s+(?:push|compose\s+up)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)kubectl\s+(?:apply|create|delete|replace|rollout)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)helm\s+(?:install|upgrade|uninstall)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:npm|pnpm|yarn|twine|cargo)\s+publish\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:systemctl|launchctl)\s+(?:start|enable|load|bootstrap)\b", re.IGNORECASE),
    re.compile(r"\bPUBLISH_ENABLED\s*=\s*(?:1|true|yes|on)\b", re.IGNORECASE),
    re.compile(
        r"(?:^|[;&|]\s*)(?:\S*/)?(?:python\d*\s+)?\S*(?:publish|deploy|release)[\w.-]*\.py\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsmbclient\b.*(?:\bput\b|\bmput\b|\bmkdir\b|\bdel(?:ete)?\b)", re.IGNORECASE | re.DOTALL),
)
_SAFE_EXTERNAL_MARKERS = re.compile(
    r"(?:--dry-run\b|--check\b|--no-publish\b|PUBLISH_ENABLED\s*=\s*(?:0|false|no|off)\b)",
    re.IGNORECASE,
)
_EXPLICIT_EXTERNAL_MARKERS = re.compile(
    r"(?:--confirm\b|PUBLISH_ENABLED\s*=\s*(?:1|true|yes|on)\b)",
    re.IGNORECASE,
)
_GENERIC_VERIFICATION_PATTERNS = (
    re.compile(r"(?:^|[;&|]\s*)(?:\S*/)?pytest\b", re.IGNORECASE),
    re.compile(r"\bpython\d*\s+-m\s+(?:pytest|unittest|compileall|py_compile)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:cargo|go)\s+test\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:mypy|pyright|ruff\s+check|eslint|shellcheck)\b", re.IGNORECASE),
    re.compile(r"\btsc\b[^\n;&|]*--noEmit\b", re.IGNORECASE),
    re.compile(r"(?:^|[/_.-])(?:test|tests|validate|validation|verify|diagnose|smoke)(?:[/_.-]|$)", re.IGNORECASE),
)
_MUTATING_COMMAND_PATTERNS = (
    re.compile(r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:rm|mv|cp|mkdir|rmdir|touch|chmod|chown|install)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)git\s+(?:commit|merge|rebase|cherry-pick|tag|add|rm|mv)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)docker\s+(?:build|pull|load|import)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:npm|pnpm|yarn|pip|pip3|uv)\s+(?:install|add|remove|uninstall)\b", re.IGNORECASE),
    re.compile(r"(?:^|[^<])(?:>>?|2>>?)\s*[^&]"),
)


def _workspace(run_context: Any | None, workspace: str | Path | None = None) -> Path:
    if workspace is not None:
        return Path(workspace).expanduser().resolve()
    meta = getattr(run_context, "_metadata", None)
    if isinstance(meta, dict) and isinstance(meta.get("workspace"), str):
        return Path(meta["workspace"]).expanduser().resolve()
    return Path.cwd().resolve()


def _state_path(run_context: Any | None, workspace: str | Path | None = None) -> Path:
    return _workspace(run_context, workspace) / _STATE_PATH


def _normalize_command(command: str) -> str:
    return " ".join(str(command or "").strip().split())


def _extract_plan_contract(
    plan_text: str,
) -> tuple[list[str], list[str], list[str]]:
    invariants: list[str] = []
    commands: list[str] = []
    reconciliation_commands: list[str] = []
    section = ""
    fenced = False

    for raw in plan_text.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading and not fenced:
            title = heading.group(1)
            if _RECONCILIATION_HEADING.search(title):
                section = "reconciliation"
            elif _VERIFICATION_HEADING.search(title):
                section = "verification"
            elif _INVARIANT_HEADING.search(title):
                section = "invariants"
            else:
                section = ""
            continue

        if section in {"verification", "reconciliation"}:
            candidates = _INLINE_CODE.findall(line)
            if fenced and line.strip():
                candidates.append(line.strip())
            for candidate in candidates:
                normalized = _normalize_command(candidate)
                target = (
                    reconciliation_commands
                    if section == "reconciliation"
                    else commands
                )
                if normalized and normalized not in target:
                    target.append(normalized)
        elif section == "invariants":
            bullet = _BULLET.match(line)
            if bullet:
                text = bullet.group(1).strip()
                if text and text not in invariants:
                    invariants.append(text)

    if not invariants:
        for raw in plan_text.splitlines():
            bullet = _BULLET.match(raw)
            if not bullet:
                continue
            text = bullet.group(1).strip()
            if re.search(r"\b(?:must|never|only after|before)\b", text, re.IGNORECASE):
                invariants.append(text)
    return invariants[:24], commands[:24], reconciliation_commands[:24]


def _validate_contract(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("version") != _STATE_VERSION:
        raise ValueError("unsupported or missing contract version")
    if not isinstance(data.get("plan_sha256"), str):
        raise ValueError("missing plan hash")
    if not isinstance(data.get("verification_commands"), list):
        raise ValueError("invalid verification command list")
    if not isinstance(data.get("satisfied"), dict):
        raise ValueError("invalid verification evidence")
    if not isinstance(data.get("approved"), bool):
        raise ValueError("missing approval state")
    data.setdefault("reconciliation_commands", [])
    data.setdefault("reconciliation_satisfied", {})
    data.setdefault("reconciliation_pending", False)
    if not isinstance(data.get("reconciliation_commands"), list):
        raise ValueError("invalid reconciliation command list")
    if not isinstance(data.get("reconciliation_satisfied"), dict):
        raise ValueError("invalid reconciliation evidence")
    if not isinstance(data.get("reconciliation_pending"), bool):
        raise ValueError("invalid reconciliation state")
    return data


def _read_contract(
    run_context: Any | None, workspace: str | Path | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    path = _state_path(run_context, workspace)
    if not path.is_file():
        return None, None
    try:
        return _validate_contract(json.loads(path.read_text(encoding="utf-8"))), None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_contract(
    run_context: Any | None = None, workspace: str | Path | None = None
) -> dict[str, Any] | None:
    """Load the approved contract, returning ``None`` when absent or invalid."""
    contract, _ = _read_contract(run_context, workspace)
    return contract


def _write_contract(
    contract: dict[str, Any], run_context: Any | None, workspace: str | Path | None = None
) -> None:
    path = _state_path(run_context, workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def approve_plan_contract(
    plan_text: str, run_context: Any, workspace: str | Path | None = None
) -> dict[str, Any] | None:
    """Create a fresh deterministic contract for an approved non-empty plan."""
    body = str(plan_text or "").strip()
    if not body:
        clear_contract(run_context, workspace)
        return None
    invariants, commands, reconciliation_commands = _extract_plan_contract(body)
    # Keep ordinary lightweight plans backward compatible. The deterministic
    # gate activates only when the plan actually declares an invariant or an
    # exact verification command.
    if not invariants and not commands and not reconciliation_commands:
        clear_contract(run_context, workspace)
        return None
    contract: dict[str, Any] = {
        "version": _STATE_VERSION,
        "approved": True,
        "plan_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "invariants": invariants,
        "verification_commands": commands,
        "reconciliation_commands": reconciliation_commands,
        "generation": 0,
        "satisfied": {},
        "reconciliation_satisfied": {},
        "reconciliation_pending": False,
        "fallback_verified_generation": -1,
        "authorization_consumed": False,
        "last_mutation_tool": None,
        "last_external_command": None,
        "last_external_success": None,
    }
    _write_contract(contract, run_context, workspace)
    return contract


def mark_plan_pending(
    run_context: Any,
    plan_text: str = "",
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Persist a fail-closed marker while a replacement plan awaits approval."""
    body = str(plan_text or "").strip()
    invariants, commands, reconciliation_commands = _extract_plan_contract(body)
    contract: dict[str, Any] = {
        "version": _STATE_VERSION,
        "approved": False,
        "plan_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "invariants": invariants,
        "verification_commands": commands,
        "reconciliation_commands": reconciliation_commands,
        "generation": 0,
        "satisfied": {},
        "reconciliation_satisfied": {},
        "reconciliation_pending": False,
        "fallback_verified_generation": -1,
        "authorization_consumed": False,
        "last_mutation_tool": None,
        "last_external_command": None,
        "last_external_success": None,
    }
    _write_contract(contract, run_context, workspace)
    return contract


def clear_contract(
    run_context: Any | None = None, workspace: str | Path | None = None
) -> None:
    """Remove approval state when a plan is replaced or planning restarts."""
    try:
        _state_path(run_context, workspace).unlink(missing_ok=True)
    except OSError:
        pass


def is_high_impact_command(command: str) -> bool:
    raw = str(command or "").replace("\r\n", "\n").replace("\r", "\n")
    segments = re.split(r"(?:&&|\|\||[;|\n])", raw)
    for segment in segments:
        normalized = _normalize_command(segment)
        if not normalized:
            continue
        if (
            _SAFE_EXTERNAL_MARKERS.search(normalized)
            and not _EXPLICIT_EXTERNAL_MARKERS.search(normalized)
        ):
            continue
        if any(pattern.search(normalized) for pattern in _HIGH_IMPACT_PATTERNS):
            return True
    return False


def _is_generic_verification(command: str) -> bool:
    normalized = _normalize_command(command)
    return bool(normalized) and any(
        pattern.search(normalized) for pattern in _GENERIC_VERIFICATION_PATTERNS
    )


def _is_mutation(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name in {
        "write_file", "edit_file", "apply_patch", "hashline_edit", "create_file",
        "replace_in_file", "insert_in_file", "insert_lines", "patch_file",
    }:
        return True
    if tool_name != "execute":
        return False
    command = _normalize_command(str(args.get("command") or ""))
    return any(pattern.search(command) for pattern in _MUTATING_COMMAND_PATTERNS)


def _unsafe_reconciliation_commands(contract: dict[str, Any]) -> list[str]:
    """Return reconciliation entries that can themselves change external state."""
    return [
        command
        for command in contract.get("reconciliation_commands") or []
        if is_high_impact_command(command)
        or _is_mutation("execute", {"command": command})
    ]


def _missing_commands(contract: dict[str, Any]) -> list[str]:
    generation = int(contract.get("generation") or 0)
    satisfied = contract.get("satisfied") or {}
    return [
        command
        for command in contract.get("verification_commands") or []
        if satisfied.get(command) != generation
    ]


def _missing_reconciliation_commands(contract: dict[str, Any]) -> list[str]:
    satisfied = contract.get("reconciliation_satisfied") or {}
    return [
        command
        for command in contract.get("reconciliation_commands") or []
        if not satisfied.get(command)
    ]


def gate_tool_call(tool_name: str, args: dict[str, Any], run_context: Any) -> str | None:
    """Return a fail-closed reason for an unsafe call, otherwise ``None``."""
    contract, load_error = _read_contract(run_context)
    command = str(args.get("command") or "")
    high_impact = tool_name == "execute" and is_high_impact_command(command)
    if load_error:
        if high_impact:
            return (
                "Refused by plan invariant gate: the persisted approval contract is "
                f"unreadable ({load_error}). Re-enter Plan mode and approve a fresh plan."
            )
        return None
    if contract is None:
        if high_impact:
            return (
                "Refused by production action gate: high-impact external actions "
                "require an approved plan. Enter Plan mode and include exact "
                "pre-action commands under 'Verification gates' plus exact "
                "post-action commands under 'Post-action reconciliation'."
            )
        return None
    if not contract.get("approved"):
        if high_impact:
            return (
                "Refused by plan invariant gate: a replacement plan is pending "
                "approval. Return to Plan mode, finish the plan, and approve "
                "exit_plan_mode before any publish/deploy action."
            )
        return None

    if contract.get("reconciliation_pending"):
        reconciliation = contract.get("reconciliation_commands") or []
        normalized = _normalize_command(command)
        if tool_name == "execute" and normalized in reconciliation:
            if high_impact or _is_mutation(tool_name, args):
                return (
                    "Refused by production action gate: the approved reconciliation "
                    "entry can mutate state or repeat the external action. Re-enter "
                    "Plan mode and replace it with a read-only remote-state check."
                )
            return None
        missing_reconciliation = _missing_reconciliation_commands(contract)
        checklist = "\n".join(f"- {item}" for item in missing_reconciliation)
        if high_impact or _is_mutation(tool_name, args):
            detail = (
                f" Run the remaining reconciliation command(s):\n{checklist}"
                if checklist
                else " Re-enter Plan mode and define post-action reconciliation commands."
            )
            return (
                "Refused by production action gate: an earlier external action "
                "may have partially succeeded and reconciliation is still pending."
                f"{detail}"
            )
        return None
    if not high_impact:
        normalized = _normalize_command(command)
        planned = contract.get("verification_commands") or []
        if _is_mutation(tool_name, args) and normalized not in planned:
            generation = int(contract.get("generation") or 0)
            contract["generation"] = generation + 1
            contract["satisfied"] = {}
            contract["reconciliation_satisfied"] = {}
            contract["fallback_verified_generation"] = -1
            contract["authorization_consumed"] = False
            contract["last_mutation_tool"] = tool_name
            _write_contract(contract, run_context)
        return None
    if contract.get("authorization_consumed"):
        return (
            "Refused by plan invariant gate: the previous verification authorization "
            "was consumed by an attempted high-impact action. Re-run every verification "
            "gate before another publish/deploy action; a failed exit does not prove "
            "that the external side effect was rolled back."
        )

    if not contract.get("reconciliation_commands"):
        return (
            "Refused by production action gate: the approved plan has no exact "
            "post-action reconciliation commands. Re-enter Plan mode and add "
            "backticked commands under a 'Post-action reconciliation' heading "
            "that verify completion markers/counts and remote state."
        )

    unsafe_reconciliation = _unsafe_reconciliation_commands(contract)
    if unsafe_reconciliation:
        checklist = "\n".join(f"- {item}" for item in unsafe_reconciliation)
        return (
            "Refused by production action gate: post-action reconciliation must "
            "be read-only and cannot repeat or mutate the external action. Re-enter "
            f"Plan mode and replace these command(s):\n{checklist}"
        )

    missing = _missing_commands(contract)
    if not contract.get("verification_commands"):
        generation = int(contract.get("generation") or 0)
        if contract.get("fallback_verified_generation") != generation:
            missing = ["a successful test, validation, or dry-run command"]
    if not missing:
        return None

    last_mutation = contract.get("last_mutation_tool")
    freshness = (
        f" after the latest mutation ({last_mutation})" if last_mutation else ""
    )
    checklist = "\n".join(f"- {item}" for item in missing)
    return (
        "Refused by plan invariant gate: fresh verification evidence is missing"
        f"{freshness}. Run the remaining approved gate(s) successfully, as separate "
        f"commands, then retry:\n{checklist}"
    )


def observe_tool_attempt(
    tool_name: str,
    args: dict[str, Any],
    *,
    run_context: Any,
) -> None:
    """Persist external uncertainty before a high-impact tool actually starts."""
    command = _normalize_command(str(args.get("command") or ""))
    if tool_name != "execute" or not is_high_impact_command(command):
        return
    contract, load_error = _read_contract(run_context)
    if contract is None or load_error or not contract.get("approved"):
        return
    if (
        contract.get("reconciliation_pending")
        and command in (contract.get("reconciliation_commands") or [])
    ):
        return
    contract["authorization_consumed"] = True
    contract["reconciliation_pending"] = True
    contract["reconciliation_satisfied"] = {}
    contract["last_external_command"] = command
    contract["last_external_success"] = None
    _write_contract(contract, run_context)


def observe_tool_result(
    tool_name: str,
    args: dict[str, Any],
    *,
    success: bool,
    run_context: Any,
) -> str:
    """Update evidence after a tool completes and return a concise model note."""
    contract, load_error = _read_contract(run_context)
    if contract is None or load_error:
        return ""
    if not contract.get("approved"):
        return ""

    command = _normalize_command(str(args.get("command") or ""))
    generation = int(contract.get("generation") or 0)

    reconciliation = contract.get("reconciliation_commands") or []
    if (
        tool_name == "execute"
        and command in reconciliation
        and contract.get("reconciliation_pending")
    ):
        if not success:
            return "[production action gate: reconciliation failed; still pending]"
        contract.setdefault("reconciliation_satisfied", {})[command] = True
        remaining = _missing_reconciliation_commands(contract)
        if not remaining:
            contract["reconciliation_pending"] = False
        _write_contract(contract, run_context)
        if remaining:
            return (
                "[production action gate: reconciliation check satisfied; "
                f"{len(remaining)} remaining]"
            )
        return (
            "[production action gate: reconciliation complete; rerun all "
            "pre-action verification gates before another external action]"
        )

    if tool_name == "execute" and is_high_impact_command(command):
        contract["last_external_success"] = bool(success)
        _write_contract(contract, run_context)
        outcome = "reported success" if success else "failed or was interrupted"
        return (
            f"[production action gate: external action {outcome}; "
            "authorization consumed and reconciliation required]"
        )

    if not success:
        return ""

    planned = contract.get("verification_commands") or []
    if (
        tool_name == "execute"
        and command in planned
        and not contract.get("reconciliation_pending")
    ):
        contract.setdefault("satisfied", {})[command] = generation
        contract["authorization_consumed"] = False
        _write_contract(contract, run_context)
        remaining = len(_missing_commands(contract))
        return (
            "[plan invariant gate: verification gate satisfied; "
            f"{remaining} remaining]"
        )

    if tool_name == "execute" and not planned and _is_generic_verification(command):
        contract["fallback_verified_generation"] = generation
        contract["authorization_consumed"] = False
        _write_contract(contract, run_context)
        return "[plan invariant gate: fresh verification evidence recorded]"

    return ""


def completion_block_reason(
    run_context: Any | None = None,
    workspace: str | Path | None = None,
) -> str | None:
    """Block a final response while external state remains uncertain."""
    contract, load_error = _read_contract(run_context, workspace)
    if load_error or contract is None or not contract.get("approved"):
        return None
    if not contract.get("reconciliation_pending"):
        return None
    missing = _missing_reconciliation_commands(contract)
    checklist = "\n".join(f"- {item}" for item in missing)
    return (
        "[Production action gate] Do not finish yet: an external action may "
        "have partially succeeded. Run the approved post-action reconciliation "
        f"command(s) before reporting completion:\n{checklist}"
    )


def contract_preamble(
    run_context: Any | None = None, workspace: str | Path | None = None
) -> str:
    """Compact prompt reminder for the active deterministic contract."""
    contract, load_error = _read_contract(run_context, workspace)
    if load_error:
        return "## Plan invariant gate\n\nContract unreadable; high-impact actions fail closed.\n"
    if contract is None:
        return ""
    if not contract.get("approved"):
        return (
            "## Plan invariant gate\n\n"
            "A replacement plan is pending approval. High-impact actions fail "
            "closed until Plan mode exits successfully.\n"
        )
    if contract.get("reconciliation_pending"):
        remaining = len(_missing_reconciliation_commands(contract))
        return (
            "## Production action gate\n\n"
            f"Post-action reconciliation pending ({remaining} command(s) remaining). "
            "Do not mutate, retry the external action, or report completion until "
            "every approved reconciliation command succeeds.\n"
        )
    commands = contract.get("verification_commands") or []
    missing = _missing_commands(contract)
    if commands:
        status = f"{len(missing)} verification gate(s) remaining"
    else:
        generation = int(contract.get("generation") or 0)
        ready = contract.get("fallback_verified_generation") == generation
        status = "fresh verification recorded" if ready else "1 verification gate remaining"
    if contract.get("authorization_consumed"):
        status = "authorization consumed; verification must be repeated"
    return (
        "## Plan invariant gate\n\n"
        f"{status}. High-impact publish/deploy commands fail closed until the "
        "approved verification contract is fresh. Any later edit invalidates it. "
        "Every attempted external action consumes authorization and requires "
        "post-action reconciliation before completion or retry.\n"
    )
