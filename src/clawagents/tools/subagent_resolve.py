"""Pure subagent resolution (Grok Build layered model).

Layers (highest → lowest precedence per field):
  spawn override → SubAgentSpec → persona/mode → parent defaults

Separates:
  - type / agent name
  - persona (behavioral overlay)
  - capability (tool allow/deny)
  - isolation (none | worktree)
  - model pin
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

from clawagents.tools.subagent import SubAgentSpec

IsolationMode = Literal["none", "worktree"]
CapabilityMode = Literal["read-only", "read-write", "execute", "all"]

_READ_ONLY_DENY = frozenset({
    "write_file", "edit_file", "apply_patch", "insert_lines", "execute",
    "git_commit", "git_push", "worktree_create", "worktree_remove",
})
_READ_WRITE_DENY = frozenset({
    "execute", "git_commit", "git_push",
})


@dataclass(frozen=True)
class ResolvedSubAgent:
    type: str
    persona_instructions: str | None = None
    capability: CapabilityMode = "all"
    isolation: IsolationMode = "none"
    model: str | None = None
    system_prompt: str | None = None
    max_iterations: int = 5
    use_native_tools: bool = True
    credential_proxy: bool = False
    tool_allowlist: frozenset[str] | None = None
    tool_denylist: frozenset[str] = field(default_factory=frozenset)
    spec: SubAgentSpec | None = None

    def denied_tools(self) -> frozenset[str]:
        deny = set(self.tool_denylist)
        if self.capability == "read-only":
            deny |= set(_READ_ONLY_DENY)
        elif self.capability == "read-write":
            deny |= set(_READ_WRITE_DENY)
        elif self.capability == "execute":
            pass  # execute allowed; still honour explicit denylist
        return frozenset(deny)


def _as_isolation(value: Any, default: IsolationMode = "none") -> IsolationMode:
    if value in ("none", "worktree"):
        return value  # type: ignore[return-value]
    return default


def _as_capability(value: Any, default: CapabilityMode = "all") -> CapabilityMode:
    if value in ("read-only", "read-write", "execute", "all"):
        return value  # type: ignore[return-value]
    return default


def resolve_subagent(
    agent: str | None,
    *,
    specs: Sequence[SubAgentSpec] | None = None,
    args: Mapping[str, Any] | None = None,
    personas: Mapping[str, str] | None = None,
    parent_isolation: IsolationMode = "none",
    parent_capability: CapabilityMode = "all",
    parent_model: str | None = None,
) -> ResolvedSubAgent:
    """Resolve a spawn request into a fully layered :class:`ResolvedSubAgent`."""
    args = dict(args or {})
    specs = list(specs or [])
    personas = dict(personas or {})

    agent_name = str(agent or args.get("agent") or "general-purpose")
    spec = next((s for s in specs if s.name == agent_name), None)

    # Spec defaults
    isolation = getattr(spec, "isolation", None) if spec else None
    capability = getattr(spec, "capability", None) if spec else None
    model = getattr(spec, "model", None) if spec else None
    persona_key = getattr(spec, "persona", None) if spec else None
    system_prompt = spec.system_prompt if spec else None
    max_iterations = spec.max_iterations if spec else 5
    use_native = spec.use_native_tools if spec else True
    cred_proxy = bool(spec.credential_proxy) if spec else False
    allow = getattr(spec, "tool_allowlist", None) if spec else None
    deny = getattr(spec, "tool_denylist", None) if spec else None

    # Parent defaults fill unset fields (isolation does NOT inherit by default
    # — Grok parity: children default to none unless explicitly set).
    isolation = isolation or "none"
    capability = capability or parent_capability or "all"
    model = model or parent_model

    # Spawn overrides win
    if "isolation" in args and args["isolation"] is not None:
        isolation = args["isolation"]
    if "capability" in args and args["capability"] is not None:
        capability = args["capability"]
    if "model" in args and args["model"]:
        model = str(args["model"])
    if "persona" in args and args["persona"]:
        persona_key = str(args["persona"])
    if "max_iterations" in args and args["max_iterations"] is not None:
        try:
            max_iterations = max(1, int(args["max_iterations"]))
        except (TypeError, ValueError):
            pass
    if "system_prompt" in args and args["system_prompt"]:
        system_prompt = str(args["system_prompt"])

    isolation = _as_isolation(isolation, "none")
    # Parent isolation never forces children into worktree unless spawn asks.
    if isolation == "none" and parent_isolation == "worktree" and args.get("isolation") is None:
        isolation = "none"
    capability = _as_capability(capability, "all")

    persona_instructions = None
    if persona_key:
        persona_instructions = personas.get(str(persona_key))
        if persona_instructions is None and isinstance(persona_key, str):
            # Allow inline persona text via args
            inline = args.get("persona_instructions")
            if isinstance(inline, str) and inline.strip():
                persona_instructions = inline

    if system_prompt and persona_instructions:
        system_prompt = (
            system_prompt.rstrip()
            + "\n\n<system-reminder>\n"
            + persona_instructions.strip()
            + "\n</system-reminder>"
        )
    elif persona_instructions and not system_prompt:
        system_prompt = (
            "<system-reminder>\n"
            + persona_instructions.strip()
            + "\n</system-reminder>"
        )

    allow_fs = frozenset(allow) if allow else None
    deny_fs = frozenset(deny or ())

    return ResolvedSubAgent(
        type=agent_name,
        persona_instructions=persona_instructions,
        capability=capability,
        isolation=isolation,
        model=str(model) if model else None,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        use_native_tools=use_native,
        credential_proxy=cred_proxy,
        tool_allowlist=allow_fs,
        tool_denylist=deny_fs,
        spec=spec,
    )


__all__ = [
    "IsolationMode",
    "CapabilityMode",
    "ResolvedSubAgent",
    "resolve_subagent",
]
