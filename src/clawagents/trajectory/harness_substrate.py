"""Runtime substrate and turnkey workspace optimizer for Harness² PPC.

Distills ``harness2/substrate/base.py`` and ``harness2/core/proposer.py`` into
ClawAgents' native runtime:

* :class:`Substrate` — five-method protocol decoupling the PPC search loop
  from how an agent workspace is staged, executed, and read back.
* :class:`ClawAgentSubstrate` — isolated staging directory materialisation,
  ``{{token}}`` template interpolation, ``skip_memory=True`` probe execution,
  and structured Markdown trajectory rendering from ``SessionReader`` or
  ``TrajectoryRecorder`` JSONL logs.
* :func:`optimize_workspace_harness` — end-to-end Propose–Probe–Compose over a
  workspace's ``.clawagents/`` / ``skills/`` / ``scripts/`` surface with
  grammar-constrained JSON proposals & blind selection, ``.adapt_complete``
  runmarks, ``HarnessLedger`` cost guardrails, and unified diff output.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from clawagents.trajectory.harness_context import DeliverablesMode
from clawagents.trajectory.harness_recursion import (
    Mode,
    PPCResult,
    Proposal,
    ProposerCache,
    Rollout,
    RolloutFn,
    build_reflection_prompt,
    propose_probe_compose,
    read_tree,
    record_experience,
    write_tree,
)
from clawagents.trajectory.harness_runmark import tree_fingerprint
from clawagents.trajectory.harness_surface import (
    DEFAULT_SURFACE,
    ExperienceLibrary,
    SurfaceSpec,
    Tree,
    autotrigger_skills,
    diff_summary,
    parse_proposal_reply,
    proposal_json_schema,
    selector_json_schema,
    validate_harness_tree,
)

_SKIP_WORKSPACE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)

_TOKEN_INTERP_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def estimate_usage_usd(
    model: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> float:
    """Best-effort USD estimate for budget guardrails when no custom estimator is given."""
    m = (model or "").lower()
    # Rates per 1M tokens: (uncached_in, cached_in, out)
    if "opus" in m:
        in_rate, cache_rate, out_rate = 15.0, 1.50, 75.0
    elif "haiku" in m or "mini" in m or "nano" in m or "flash-lite" in m:
        in_rate, cache_rate, out_rate = 0.40, 0.08, 1.60
    elif "flash" in m:
        in_rate, cache_rate, out_rate = 0.50, 0.10, 3.00
    elif "grok" in m:
        in_rate, cache_rate, out_rate = 2.00, 0.50, 10.00
    else:
        # Balanced frontier default (Sonnet / GPT-5.x / Gemini Pro tier)
        in_rate, cache_rate, out_rate = 2.50, 0.25, 12.00
    usd = (
        max(0, int(input_tokens)) * in_rate
        + max(0, int(cached_input_tokens)) * cache_rate
        + max(0, int(output_tokens)) * out_rate
    ) / 1_000_000.0
    return round(usd, 6)


def interpolate_harness_tree(tree: Mapping[str, str], ctx: Mapping[str, str] | None) -> Tree:
    """Substitute ``{{key}}`` tokens present in ``ctx`` across text harness files."""
    if not ctx:
        return dict(tree)

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(ctx[key]) if key in ctx else match.group(0)

    return {rel: _TOKEN_INTERP_RE.sub(_sub, content) for rel, content in tree.items()}


def _trunc_middle(text: str, limit: int) -> str:
    if len(text) <= limit or limit <= 0:
        return text
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n...[+{len(text) - limit:,} chars]...\n{text[-tail:]}"


def render_trajectory_markdown(
    source: Path | str | Sequence[Mapping[str, Any]],
    *,
    obs_trunc: int = 1_200,
    max_chars: int = 24_000,
    args_trunc: int = 600,
) -> str:
    """Render a ``SessionWriter`` or ``TrajectoryRecorder`` log into proposer Markdown.

    Preserves both the head and tail of long tool observations so stack traces
    and command exit codes at the bottom of outputs remain visible to INV/DOM
    editors.
    """
    records: list[Mapping[str, Any]] = []
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            return ""
        for raw_line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    else:
        records = [dict(r) for r in source if isinstance(r, Mapping)]

    if not records:
        return ""

    blocks: list[str] = []
    # Case 1: TrajectoryRecorder NDJSON (each line has `turn_index` or `type == "summary"`)
    if any("turn_index" in r for r in records):
        for r in records:
            if r.get("type") == "summary" or "turn_index" not in r:
                continue
            idx = r.get("turn_index", 0)
            resp_text = str(r.get("response_text") or "").strip()
            lines = [f"### Turn {idx}"]
            if resp_text:
                lines.append(_trunc_middle(resp_text, obs_trunc))
            for tc in r.get("tool_calls") or []:
                if not isinstance(tc, Mapping):
                    continue
                tname = str(tc.get("tool_name") or "tool")
                ok = bool(tc.get("success", True))
                args_str = json.dumps(tc.get("args") or {}, sort_keys=True, default=str)
                status = "OK" if ok else f"FAIL ({tc.get('failure_type') or 'error'})"
                lines.append(f"- `{tname}` [{status}] args={_trunc_middle(args_str, args_trunc)}")
                preview = str(tc.get("output_preview") or tc.get("error") or "").strip()
                if preview:
                    lines.append(f"  observation: {_trunc_middle(preview, obs_trunc)}")
            blocks.append("\n".join(lines))
    else:
        # Case 2: SessionWriter JSONL (`turn_started`, `assistant_message`, `tool_result`)
        turn_no = 0
        for ev in records:
            etype = str(ev.get("type") or "")
            if etype == "turn_started":
                turn_no = int(ev.get("iteration") or (turn_no + 1))
                blocks.append(f"### Turn {turn_no}")
            elif etype == "assistant_message":
                content = str(ev.get("content") or "").strip()
                if content:
                    blocks.append(_trunc_middle(content, obs_trunc))
                for tc in ev.get("tool_calls") or []:
                    if isinstance(tc, Mapping):
                        args_str = json.dumps(tc.get("args") or {}, sort_keys=True, default=str)
                        blocks.append(
                            f"- call `{tc.get('name', 'tool')}` "
                            f"args={_trunc_middle(args_str, args_trunc)}"
                        )
            elif etype == "tool_result":
                tname = str(ev.get("name") or "tool")
                ok = bool(ev.get("success", True))
                out_raw = ev.get("output", "")
                out_str = out_raw if isinstance(out_raw, str) else json.dumps(out_raw, default=str)
                err_str = str(ev.get("error") or "").strip()
                combined = (f"{err_str}\n{out_str}".strip() if err_str else out_str.strip())
                blocks.append(
                    f"- result `{tname}` [{'OK' if ok else 'FAIL'}]: "
                    f"{_trunc_middle(combined, obs_trunc)}"
                )

    rendered = "\n\n".join(blocks).strip()
    return _trunc_middle(rendered, max_chars)


def compute_unified_diffs(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, str]:
    """Compute per-file unified diffs between two harness trees."""
    diffs: dict[str, str] = {}
    for rel in sorted(set(before) | set(after)):
        old = before.get(rel)
        new = after.get(rel)
        if old == new:
            continue
        old_lines = (old or "").splitlines(keepends=True)
        new_lines = (new or "").splitlines(keepends=True)
        diff_text = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel}" if old is not None else "/dev/null",
                tofile=f"b/{rel}" if new is not None else "/dev/null",
            )
        )
        if diff_text:
            diffs[rel] = diff_text
    return diffs


@dataclass(frozen=True)
class Install:
    """Materialized harness tree inside an isolated rollout directory."""

    staging_dir: Path
    fingerprint: str
    written_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunRequest:
    """Execution request passed to :meth:`Substrate.execute`."""

    task: str
    workspace: Path
    label: str = "run"
    max_iterations: int = 25
    timeout_s: float = 180.0


@runtime_checkable
class Substrate(Protocol):
    """Harness² 5-method runtime substrate contract."""

    def surface(self) -> SurfaceSpec: ...

    def materialise(
        self,
        tree: Mapping[str, str],
        staging_dir: Path | str,
        ctx: Mapping[str, str] | None = None,
    ) -> Install: ...

    async def execute(self, request: RunRequest, install: Install) -> Rollout: ...

    def read_trajectory(
        self,
        source: Path | str | Sequence[Mapping[str, Any]],
        *,
        obs_trunc: int = 1_200,
        max_chars: int = 24_000,
        args_trunc: int = 600,
    ) -> str: ...

    def preflight(self) -> dict[str, Any]: ...


def _copy_workspace_skeleton(src: Path, dst: Path) -> dict[str, str]:
    """Copy non-transient workspace files to ``dst`` and return baseline text digests."""
    baseline: dict[str, str] = {}
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return baseline
    for root_str, dirnames, filenames in os.walk(src):
        root_p = Path(root_str)
        rel_root = root_p.relative_to(src)
        # Prune heavy or runtime-transient directories in-place
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _SKIP_WORKSPACE_DIRS
            and not (rel_root == Path(".clawagents") and d in ("sessions", "trajectories", "harness_runs"))
        ]
        for fname in filenames:
            if fname in (".DS_Store", ".adapt_complete") or fname.endswith(".pyc"):
                continue
            src_file = root_p / fname
            if not src_file.is_file():
                continue
            try:
                if src_file.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            rel_posix = src_file.relative_to(src).as_posix()
            dst_file = dst / rel_posix
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            try:
                baseline[rel_posix] = dst_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                baseline[rel_posix] = f"<binary:{dst_file.stat().st_size}>"
    return baseline


def _collect_changed_deliverables(
    staging_dir: Path,
    baseline: Mapping[str, str],
    harness_paths: set[str],
) -> dict[str, str]:
    """Collect text files created or modified by the agent outside the harness surface."""
    changed: dict[str, str] = {}
    for p in sorted(staging_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(staging_dir).as_posix()
        if rel in harness_paths or rel.startswith(".clawagents/"):
            continue
        if any(part in _SKIP_WORKSPACE_DIRS for part in p.relative_to(staging_dir).parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if baseline.get(rel) != text:
            changed[rel] = text
    return changed


class ClawAgentSubstrate:
    """Native ClawAgents implementation of the Harness² :class:`Substrate` protocol."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        llm: Any = None,
        surface_spec: SurfaceSpec = DEFAULT_SURFACE,
        max_iterations: int = 25,
        timeout_s: float = 180.0,
        cost_estimator: Callable[[Any], float] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.llm = llm
        self._surface = surface_spec
        self.max_iterations = max_iterations
        self.timeout_s = timeout_s
        self.cost_estimator = cost_estimator
        self._baselines: dict[Path, dict[str, str]] = {}

    def surface(self) -> SurfaceSpec:
        return self._surface

    def preflight(self) -> dict[str, Any]:
        errors: list[str] = []
        if not self.workspace.exists():
            errors.append(f"Workspace does not exist: {self.workspace}")
        elif not os.access(self.workspace, os.W_OK):
            errors.append(f"Workspace is not writable: {self.workspace}")

        model_name = getattr(self.llm, "model", None) or getattr(self.llm, "name", None) or ""
        if self.llm is None:
            has_env_key = any(
                bool(os.environ.get(k))
                for k in (
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "XAI_API_KEY",
                )
            )
            if not has_env_key:
                try:
                    from clawagents.config.config import load_config

                    cfg = load_config()
                    has_env_key = any(
                        bool(getattr(cfg, attr, None))
                        for attr in (
                            "openai_api_key",
                            "anthropic_api_key",
                            "gemini_api_key",
                            "xai_api_key",
                        )
                    )
                except Exception:
                    has_env_key = False
            if not has_env_key:
                errors.append("No LLM provider supplied and no provider API key configured")

        return {
            "ok": not errors,
            "workspace": str(self.workspace),
            "model": str(model_name),
            "errors": errors,
        }

    def materialise(
        self,
        tree: Mapping[str, str],
        staging_dir: Path | str,
        ctx: Mapping[str, str] | None = None,
    ) -> Install:
        target = Path(staging_dir).resolve()
        if target.exists():
            shutil.rmtree(target)
        baseline = _copy_workspace_skeleton(self.workspace, target)

        # Clear inherited surface files in staging so removed files in `tree` stay removed.
        existing_surface = read_tree(target, self._surface)
        for rel in existing_surface:
            p = target / rel
            if p.is_file():
                p.unlink()

        interpolated = interpolate_harness_tree(tree, ctx)
        autotrigger_skills(self._surface, interpolated)
        validate_harness_tree(interpolated, self._surface)
        written = write_tree(target, interpolated)
        self._baselines[target] = baseline
        return Install(
            staging_dir=target,
            fingerprint=tree_fingerprint(interpolated),
            written_files=tuple(written),
        )

    def read_trajectory(
        self,
        source: Path | str | Sequence[Mapping[str, Any]],
        *,
        obs_trunc: int = 1_200,
        max_chars: int = 24_000,
        args_trunc: int = 600,
    ) -> str:
        return render_trajectory_markdown(
            source,
            obs_trunc=obs_trunc,
            max_chars=max_chars,
            args_trunc=args_trunc,
        )

    def _resolve_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        from clawagents.agent import _resolve_model
        from clawagents.config.config import load_config

        load_config()
        return _resolve_model(None, streaming=False, api_key=None, context_window=None)

    async def execute(self, request: RunRequest, install: Install) -> Rollout:
        from clawagents.agent import create_claw_agent
        from clawagents.run_context import RunContext

        llm = self._resolve_llm()
        prev_cwd = Path.cwd()
        staging = install.staging_dir
        baseline = self._baselines.get(staging, {})
        try:
            os.chdir(staging)
            agent = create_claw_agent(
                model=llm,
                streaming=False,
                trajectory=True,
                learn=False,
            )
            rc: RunContext[Any] = RunContext(skip_memory=True)
            rc._metadata["workspace"] = str(staging)
            state = await agent.invoke(
                request.task,
                max_iterations=request.max_iterations or self.max_iterations,
                timeout_s=request.timeout_s or self.timeout_s,
                run_context=rc,
            )
        finally:
            try:
                os.chdir(prev_cwd)
            except OSError:
                pass

        traj_md = ""
        if getattr(state, "trajectory_file", ""):
            traj_md = self.read_trajectory(state.trajectory_file)
        elif getattr(state, "session_file", ""):
            traj_md = self.read_trajectory(state.session_file)

        changed_files = _collect_changed_deliverables(
            staging, baseline, set(install.written_files)
        )
        final_text = str(getattr(state, "result", "") or "").strip()
        deliverable_parts: list[str] = []
        if final_text:
            deliverable_parts.append(final_text)
        for rel_path, content in sorted(changed_files.items()):
            deliverable_parts.append(f"## File `{rel_path}`\n\n{content}")
        combined_deliverable = "\n\n".join(deliverable_parts).strip()

        usage = getattr(state, "usage", None)
        if self.cost_estimator is not None and usage is not None:
            usd = float(self.cost_estimator(usage))
        elif usage is not None:
            usd = estimate_usage_usd(
                getattr(llm, "model", None),
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
                cached_input_tokens=getattr(usage, "cached_input_tokens", 0),
            )
        else:
            usd = 0.0

        ok = getattr(state, "status", "error") == "done"
        return Rollout(
            ok=ok,
            delivered=bool(combined_deliverable),
            deliverable=combined_deliverable,
            trajectory=traj_md or final_text,
            error="" if ok else str(getattr(state, "result", "") or "run did not finish cleanly"),
            usd=usd,
            deliverable_files=changed_files,
        )


def _proposer_system_prompt(surface: SurfaceSpec) -> str:
    return (
        "You are a Harness² workspace architect. Propose minimal, evidence-backed edits "
        "to the workspace harness surface.\n"
        f"- Editable always-on files (hard cap {surface.always_on_max:,} chars total): "
        f"{', '.join(surface.always_on)}\n"
        f"- On-demand skills: `{surface.skill_dir}/<skill-name>/SKILL.md` (with `name:` and "
        "`description:` YAML frontmatter) and optional `scripts/<helper>.py`\n"
        "- Deterministic helper scripts: `scripts/<name>.py`\n"
        "- Hooks file: `.clawagents/hooks.json`\n\n"
        "Respond with a single JSON object matching the schema:\n"
        '{"keep_base": false, "notes": "<trajectory moment motivating each edit>", '
        '"edits": {"<relative_path>": "<full_file_content>"}, "removed": []}\n'
        'If no edit is justified by observed trajectory failures, set "keep_base": true '
        'with empty "edits" and "removed".'
    )


async def _llm_propose_once(
    llm: Any,
    prompt: str,
    surface: SurfaceSpec,
) -> Proposal:
    from clawagents.providers.llm import LLMMessage
    from clawagents.structured_output import invoke_structured_control

    messages = [
        LLMMessage(role="system", content=_proposer_system_prompt(surface)),
        LLMMessage(role="user", content=prompt),
    ]
    parsed, resp = await invoke_structured_control(
        llm,
        messages,
        schema=proposal_json_schema(),
        parse_fn=parse_proposal_reply,
    )
    edits, removed, notes, keep_base = parsed
    prompt_t = int(getattr(resp, "prompt_tokens", 0) or 0)
    total_t = int(getattr(resp, "tokens_used", 0) or 0)
    cache_t = int(getattr(resp, "cache_read_tokens", 0) or 0)
    out_t = int(getattr(resp, "completion_tokens", max(total_t - prompt_t, 0)) or 0)
    usd = estimate_usage_usd(
        getattr(resp, "model", None) or getattr(llm, "model", None),
        input_tokens=max(0, prompt_t - cache_t),
        output_tokens=out_t,
        cached_input_tokens=cache_t,
    )
    return Proposal(
        edits=edits,
        removed=removed,
        notes=notes,
        keep_base=keep_base,
        usd=usd,
    )


async def _llm_select_once(
    llm: Any,
    selector_prompt: str,
    allowed_choices: Sequence[str],
) -> str:
    from clawagents.providers.llm import LLMMessage
    from clawagents.structured_output import invoke_structured_control

    messages = [LLMMessage(role="user", content=selector_prompt)]
    raw_text, _resp = await invoke_structured_control(
        llm,
        messages,
        schema=selector_json_schema(allowed_choices),
        parse_fn=lambda txt: txt,
    )
    return raw_text


def backup_and_apply_harness(
    workspace: Path | str,
    before_tree: Mapping[str, str],
    adopted_tree: Mapping[str, str],
    *,
    backup_dir: Path | str | None = None,
    surface: SurfaceSpec = DEFAULT_SURFACE,
) -> dict[str, Any]:
    """Validate ``adopted_tree``, snapshot overwritten files, and apply to ``workspace``."""
    ws = Path(workspace).resolve()
    candidate = dict(adopted_tree)
    triggered = autotrigger_skills(surface, candidate)
    validate_harness_tree(candidate, surface)

    if backup_dir is not None:
        bdir = Path(backup_dir).resolve()
        bdir.mkdir(parents=True, exist_ok=True)
        for rel, old_content in before_tree.items():
            dest = bdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(old_content, encoding="utf-8")

    # Remove surface files deleted by the adopted tree
    removed_files = sorted(set(before_tree) - set(candidate))
    for rel in removed_files:
        target = (ws / rel).resolve()
        if ws in target.parents and target.is_file():
            target.unlink()

    written = write_tree(ws, candidate)
    return {
        "written": written,
        "removed": removed_files,
        "autotriggered": triggered,
        "backup_dir": str(backup_dir) if backup_dir is not None else None,
    }


async def optimize_workspace_harness(
    task: str,
    workspace: Path | str,
    *,
    llm: Any = None,
    proposer_llm: Any = None,
    selector_llm: Any = None,
    k: int = 1,
    mode: Mode = "parallel",
    task_id: str = "workspace-harness",
    surface: SurfaceSpec = DEFAULT_SURFACE,
    adaptive_probe: bool = True,
    confidence_threshold: float = 0.90,
    max_usd: float | None = None,
    deliverables_mode: DeliverablesMode = "contents",
    run_dir: Path | str | None = None,
    library: ExperienceLibrary | None = None,
    cache: ProposerCache | None = None,
    apply: bool = False,
    reflect: bool = False,
    max_iterations: int = 25,
    timeout_s: float = 180.0,
    rollout_fn: RolloutFn | None = None,
    on_stage_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run Propose–Probe–Compose on ``workspace`` and return result + unified diffs."""
    ws = Path(workspace).resolve()
    substrate = ClawAgentSubstrate(
        ws,
        llm=llm,
        surface_spec=surface,
        max_iterations=max_iterations,
        timeout_s=timeout_s,
    )
    if rollout_fn is None:
        pf = substrate.preflight()
        if not pf["ok"]:
            raise RuntimeError(f"Harness preflight failed: {'; '.join(pf['errors'])}")

    resolved_worker_llm = llm or substrate._resolve_llm()
    edit_llm = proposer_llm or resolved_worker_llm
    pick_llm = selector_llm or edit_llm

    base_tree = read_tree(ws, surface)
    owns_temp = run_dir is None
    effective_run_dir = (
        Path(tempfile.mkdtemp(prefix="clawagents-ppc-"))
        if run_dir is None
        else Path(run_dir).resolve()
    )
    effective_cache = cache or ProposerCache(effective_run_dir / "proposer_cache")

    try:
        async def _substrate_rollout(tree: Tree, label: str) -> Rollout:
            if rollout_fn is not None:
                return await rollout_fn(tree, label)
            staging = effective_run_dir / "staging" / label
            install = substrate.materialise(tree, staging, ctx={"workspace": str(staging)})
            req = RunRequest(
                task=task,
                workspace=staging,
                label=label,
                max_iterations=max_iterations,
                timeout_s=timeout_s,
            )
            return await substrate.execute(req, install)

        async def _propose(prompt: str, _label: str) -> Proposal:
            return await _llm_propose_once(edit_llm, prompt, surface)

        # Extract candidate IDs from selector prompt header (`one of: A, B`) for enum schema
        id_line_re = re.compile(r"<one of:\s*([A-Z,\s]+)>")

        async def _select(selector_prompt: str) -> str:
            m = id_line_re.search(selector_prompt)
            choices = [c.strip() for c in m.group(1).split(",")] if m else []
            return await _llm_select_once(pick_llm, selector_prompt, choices)

        ppc_result: PPCResult = await propose_probe_compose(
            task,
            base_tree,
            rollout=_substrate_rollout,
            propose=_propose,
            select=_select,
            k=k,
            mode=mode,
            task_id=task_id,
            surface=surface,
            library=library,
            cache=effective_cache,
            adaptive_probe=adaptive_probe,
            confidence_threshold=confidence_threshold,
            max_usd=max_usd,
            run_dir=effective_run_dir,
            deliverables_mode=deliverables_mode,
            on_stage_event=on_stage_event,
        )

        adopted_tree = ppc_result.adopted_tree
        summary = diff_summary(surface, base_tree, adopted_tree)
        unified = compute_unified_diffs(base_tree, adopted_tree)

        experience_entry = None
        if reflect and library is not None:
            from clawagents.providers.llm import LLMMessage

            refl_prompt = build_reflection_prompt(task, ppc_result, surface)
            refl_resp = await edit_llm.chat([LLMMessage(role="user", content=refl_prompt)])
            experience_entry = record_experience(
                ppc_result,
                library,
                str(getattr(refl_resp, "content", "") or ""),
            )

        applied_info = None
        if apply and unified:
            backup_dir = ws / ".clawagents" / "harness_runs" / task_id / "backup"
            applied_info = backup_and_apply_harness(
                ws,
                base_tree,
                adopted_tree,
                backup_dir=backup_dir,
                surface=surface,
            )

        return {
            "ok": True,
            "result": ppc_result,
            "base_tree": base_tree,
            "adopted_tree": adopted_tree,
            "diff_summary": summary,
            "unified_diffs": unified,
            "applied": applied_info,
            "experience": experience_entry.to_dict() if experience_entry else None,
        }
    finally:
        if owns_temp:
            shutil.rmtree(effective_run_dir, ignore_errors=True)


__all__ = [
    "ClawAgentSubstrate",
    "Install",
    "RunRequest",
    "Substrate",
    "backup_and_apply_harness",
    "compute_unified_diffs",
    "estimate_usage_usd",
    "interpolate_harness_tree",
    "optimize_workspace_harness",
    "render_trajectory_markdown",
]
