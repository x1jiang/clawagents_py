"""A/B experiment: clawagents with ATLAS on vs off.

Measures iterations, tool calls, tokens, wall time, transcript size, and
whether ATLAS injected checkpoints / final gates. Writes JSON + markdown.

Usage (from clawagents_py/):
  set -a && source ../.env && set +a
  PYTHONPATH=src python examples/atlas_basic/ab_experiment.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "ab_results"
CONFIG = Path(__file__).resolve().parent / "atlas.json"

# Cheap/fast model for experiment cost control
MODEL = os.environ.get("ATLAS_AB_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


@dataclass
class RunMetrics:
    task_id: str
    atlas: bool
    ok: bool
    status: str
    result_preview: str
    iterations: int
    tool_calls: int
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    transcript_chars: int
    atlas_checkpoint_msgs: int
    atlas_final_gate_msgs: int
    atlas_protocol_injected: bool
    error: str = ""
    notes: list[str] = field(default_factory=list)


TASKS = [
    {
        "id": "easy_list",
        "task": (
            "In the current directory, list Python files under examples/ "
            "(names only). Reply with a bullet list. Do not edit anything."
        ),
        "expect": "should be cheap; ATLAS overhead = final gate only",
    },
    {
        "id": "fail_then_fix",
        "task": (
            "Run this shell command exactly once first: "
            "`python -c 'import definitely_missing_module_xyz'` "
            "(it will fail). Then diagnose the ImportError briefly and "
            "confirm the correct fix is to install or avoid that module. "
            "Do not retry the same failing import."
        ),
        "expect": "ATLAS should fire on tool failure; may reduce blind retries",
    },
    {
        "id": "verify_before_done",
        "task": (
            "Create a temp file examples/atlas_basic/_ab_tmp.txt containing "
            "exactly the text HELLO_ATLAS. Then read it back and only then "
            "say DONE. If the file content is wrong, fix it before DONE."
        ),
        "expect": "final gate may catch premature DONE without verify",
    },
]


def _count_atlas_markers(messages: list) -> tuple[int, int, bool]:
    protocol = False
    checkpoints = 0
    finals = 0
    for m in messages or []:
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        low = content.lower()
        if "atlas runtime" in low or "atlas runtime interaction" in low:
            protocol = True
        if "atlas reflection" in low or "observe:" in low and "correlate:" in low:
            if "final submission gate" in low:
                finals += 1
            elif "tool failure" in low or "checkpoint" in low or "atlas checkpoint" in low:
                checkpoints += 1
            elif "final atlas status" in low:
                finals += 1
        if "atlas blocked completion" in low or "final submission gate" in low:
            finals += 1
        if content.startswith("ATLAS ") or "[atlas]" in low:
            if "final" in low:
                finals += 1
            else:
                checkpoints += 1
    return checkpoints, finals, protocol


def _usage_parts(usage) -> tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    if hasattr(usage, "to_dict"):
        d = usage.to_dict()
        return (
            int(d.get("prompt_tokens") or d.get("input_tokens") or 0),
            int(d.get("completion_tokens") or d.get("output_tokens") or 0),
            int(d.get("total_tokens") or 0),
        )
    prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
    completion = int(
        getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
    )
    total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
    return prompt, completion, total


async def run_one(task_id: str, task: str, atlas: bool) -> RunMetrics:
    from clawagents import create_claw_agent

    events: list[str] = []

    def on_event(kind, data=None):
        msg = ""
        if isinstance(data, dict):
            msg = str(data.get("message") or data.get("content") or "")[:200]
        events.append(f"{kind}:{msg}")

    t0 = time.monotonic()
    try:
        agent = create_claw_agent(
            model=MODEL,
            atlas=atlas,
            atlas_config=str(CONFIG) if atlas else None,
            trajectory=True,
            rethink=False,
            learn=False,
            streaming=False,
            max_iterations=12,
        )
        state = await agent.invoke(task, on_event=on_event, timeout_s=180)
        elapsed = time.monotonic() - t0
        messages = getattr(state, "messages", None) or []
        transcript = "\n".join(
            f"{getattr(m, 'role', '?')}: {getattr(m, 'content', '')}" for m in messages
        )
        ck, fg, proto = _count_atlas_markers(messages)
        # Also count from events
        for e in events:
            if "ATLAS: checkpoint" in e:
                ck += 1
            if "ATLAS: final submission gate" in e:
                fg += 1
            if "ATLAS: injected runtime protocol" in e:
                proto = True
        prompt, completion, total = _usage_parts(getattr(state, "usage", None))
        result = str(getattr(state, "result", "") or "")
        return RunMetrics(
            task_id=task_id,
            atlas=atlas,
            ok=getattr(state, "status", "") != "error",
            status=str(getattr(state, "status", "")),
            result_preview=result[:240].replace("\n", " "),
            iterations=int(getattr(state, "iterations", 0) or 0),
            tool_calls=int(getattr(state, "tool_calls", 0) or 0),
            elapsed_s=round(elapsed, 2),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total or (prompt + completion),
            transcript_chars=len(transcript),
            atlas_checkpoint_msgs=ck,
            atlas_final_gate_msgs=fg,
            atlas_protocol_injected=proto,
            notes=[n for n in events if n.startswith("warn:") or n.startswith("context:ATLAS")][
                :12
            ],
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        return RunMetrics(
            task_id=task_id,
            atlas=atlas,
            ok=False,
            status="exception",
            result_preview="",
            iterations=0,
            tool_calls=0,
            elapsed_s=round(elapsed, 2),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            transcript_chars=0,
            atlas_checkpoint_msgs=0,
            atlas_final_gate_msgs=0,
            atlas_protocol_injected=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _md_table(rows: list[RunMetrics]) -> str:
    headers = [
        "task",
        "atlas",
        "ok",
        "iters",
        "tools",
        "sec",
        "tokens",
        "transcript",
        "ckpts",
        "gates",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.task_id,
                    "on" if r.atlas else "off",
                    "✓" if r.ok else "✗",
                    str(r.iterations),
                    str(r.tool_calls),
                    f"{r.elapsed_s:.1f}",
                    str(r.total_tokens),
                    str(r.transcript_chars),
                    str(r.atlas_checkpoint_msgs),
                    str(r.atlas_final_gate_msgs),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Need OPENAI_API_KEY or GEMINI_API_KEY in env")

    # Prefer OpenAI path for atlas_model learning calls if present
    cfg = json.loads(CONFIG.read_text())
    if os.environ.get("OPENAI_API_KEY"):
        cfg["atlas_model"] = os.environ.get("ATLAS_AB_ATLAS_MODEL") or "gpt-4o-mini"
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")

    results: list[RunMetrics] = []
    for spec in TASKS:
        for atlas in (False, True):
            print(f"\n=== {spec['id']} atlas={atlas} ===", flush=True)
            m = await run_one(spec["id"], spec["task"], atlas)
            results.append(m)
            print(
                f"  ok={m.ok} iters={m.iterations} tools={m.tool_calls} "
                f"sec={m.elapsed_s} tokens={m.total_tokens} "
                f"chars={m.transcript_chars} ckpts={m.atlas_checkpoint_msgs} "
                f"gates={m.atlas_final_gate_msgs} err={m.error[:120]}",
                flush=True,
            )

    # Pairwise deltas
    deltas = []
    for spec in TASKS:
        off = next(r for r in results if r.task_id == spec["id"] and not r.atlas)
        on = next(r for r in results if r.task_id == spec["id"] and r.atlas)
        deltas.append(
            {
                "task_id": spec["id"],
                "expect": spec["expect"],
                "token_delta": on.total_tokens - off.total_tokens,
                "time_delta_s": round(on.elapsed_s - off.elapsed_s, 2),
                "transcript_delta": on.transcript_chars - off.transcript_chars,
                "iter_delta": on.iterations - off.iterations,
                "off_ok": off.ok,
                "on_ok": on.ok,
                "checkpoints": on.atlas_checkpoint_msgs,
                "gates": on.atlas_final_gate_msgs,
            }
        )

    payload = {
        "model": MODEL,
        "results": [asdict(r) for r in results],
        "deltas": deltas,
    }
    out_json = OUT_DIR / "ab_results.json"
    out_md = OUT_DIR / "ab_results.md"
    out_json.write_text(json.dumps(payload, indent=2))

    md = [
        f"# ATLAS A/B experiment (`{MODEL}`)",
        "",
        _md_table(results),
        "",
        "## Pairwise deltas (on − off)",
        "",
        "| task | token Δ | time Δ | transcript Δ | iter Δ | ckpts | gates | ok off→on |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in deltas:
        md.append(
            f"| {d['task_id']} | {d['token_delta']} | {d['time_delta_s']} | "
            f"{d['transcript_delta']} | {d['iter_delta']} | {d['checkpoints']} | "
            f"{d['gates']} | {d['off_ok']}→{d['on_ok']} |"
        )
    md.append("")
    md.append("## Task expectations")
    for spec in TASKS:
        md.append(f"- **{spec['id']}**: {spec['expect']}")
    out_md.write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    asyncio.run(main())
