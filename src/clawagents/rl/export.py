"""JSONL / ChatML / TRL / Atropos export helpers.

These functions don't import any training framework — they only reshape
trajectories into the dict layouts the frameworks expect. The
:mod:`clawagents.rl.adapters` module provides actual :class:`Trainer`
integration (lazy-imported).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from clawagents.rl.trajectory import Trajectory


def _open_for_write(path: str | os.PathLike[str]) -> Any:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("w", encoding="utf-8")


def export_jsonl(
    trajectories: Iterable[Trajectory],
    path: str | os.PathLike[str],
) -> int:
    """Write trajectories as JSONL (one trajectory per line). Returns the count."""
    n = 0
    with _open_for_write(path) as fh:
        for traj in trajectories:
            fh.write(traj.to_json())
            fh.write("\n")
            n += 1
    return n


def load_jsonl(path: str | os.PathLike[str]) -> list[Trajectory]:
    """Read trajectories previously written by :func:`export_jsonl`."""
    out: list[Trajectory] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(Trajectory.from_json(line))
    return out


def to_chatml(traj: Trajectory) -> list[dict[str, Any]]:
    """Convert a trajectory to a ChatML-compatible message list.

    The output is the form expected by ``transformers``'
    ``apply_chat_template`` and most TRL trainers.
    """
    msgs: list[dict[str, Any]] = []
    for step in traj.steps:
        if step.role == "assistant" and step.tool_calls:
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": step.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in step.tool_calls
                ],
            }
            msgs.append(entry)
        elif step.role == "tool":
            msgs.append(
                {
                    "role": "tool",
                    "content": step.content,
                    "tool_call_id": step.tool_call_id or "",
                    **({"name": step.name} if step.name else {}),
                }
            )
        else:
            msgs.append({"role": step.role, "content": step.content})
    return msgs


def to_trl_sft(traj: Trajectory) -> dict[str, Any]:
    """Shape one row for TRL's :class:`SFTTrainer`.

    TRL accepts either a ``messages`` list or a ``prompt`` /
    ``completion`` pair. We emit both so the user can pick whichever
    matches their pipeline.
    """
    messages = to_chatml(traj)
    final = traj.final_assistant
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    return {
        "messages": messages,
        "prompt": [m for m in prompt_msgs],
        "completion": [
            {"role": "assistant", "content": final.content if final else ""}
        ],
        "metadata": {
            "run_id": traj.run_id,
            "task": traj.task,
            "model": traj.model,
            "reward": traj.reward,
        },
    }


def to_trl_dpo(
    chosen: Trajectory,
    rejected: Trajectory,
) -> dict[str, Any]:
    """Shape a preference pair for TRL's :class:`DPOTrainer`.

    Both trajectories should share the same prompt prefix (system +
    user). The function uses ``chosen``'s prefix as the reference.
    """
    chosen_final = chosen.final_assistant
    rejected_final = rejected.final_assistant
    prompt_msgs = [
        {"role": s.role, "content": s.content}
        for s in chosen.steps
        if s.role in ("system", "user")
    ]
    return {
        "prompt": prompt_msgs,
        "chosen": [
            {
                "role": "assistant",
                "content": chosen_final.content if chosen_final else "",
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": rejected_final.content if rejected_final else "",
            }
        ],
        "metadata": {
            "chosen_run_id": chosen.run_id,
            "rejected_run_id": rejected.run_id,
            "chosen_reward": chosen.reward,
            "rejected_reward": rejected.reward,
        },
    }


def to_atropos_rollout(traj: Trajectory) -> dict[str, Any]:
    """Shape a rollout for the Atropos / Nous environment harness.

    Atropos rollouts are dictionaries with::

        { "messages": [...], "score": float, "metadata": {...} }

    so we just thread our trajectory through.
    """
    return {
        "messages": to_chatml(traj),
        "score": traj.reward if traj.reward is not None else 0.0,
        "rewards": dict(traj.rewards) if traj.rewards else {},
        "metadata": {
            "run_id": traj.run_id,
            "task": traj.task,
            "model": traj.model,
            **traj.metadata,
        },
    }


def export_trl_sft_jsonl(
    trajs: Sequence[Trajectory],
    path: str | os.PathLike[str],
) -> int:
    """Write a TRL-SFT-shaped JSONL file."""
    n = 0
    with _open_for_write(path) as fh:
        for t in trajs:
            fh.write(json.dumps(to_trl_sft(t), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def export_atropos_rollouts_jsonl(
    trajs: Sequence[Trajectory],
    path: str | os.PathLike[str],
) -> int:
    """Write an Atropos rollouts JSONL file."""
    n = 0
    with _open_for_write(path) as fh:
        for t in trajs:
            fh.write(json.dumps(to_atropos_rollout(t), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n
