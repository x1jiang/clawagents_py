"""Reinforcement-learning fine-tuning hooks for ClawAgents.

This package layers on top of :mod:`clawagents.trajectory` to expose
agent runs as training data for RL/SFT/DPO pipelines (TRL, Atropos,
SLIME). It is intentionally hermetic: nothing here imports a training
framework at module load time — the relevant adapters import lazily and
raise :class:`MissingRLDependencyError` if the user hasn't installed the
optional package.

Surface area::

    from clawagents.rl import (
        # Data model
        Trajectory, TrajectoryStep, TrajectoryRole,
        # Recorder for capturing live runs
        RLRecorder, RecorderConfig,
        # Scorers
        RewardScorer, ContainsScorer, ExactMatchScorer,
        RegexScorer, LengthPenaltyScorer, CompositeScorer,
        # Export
        export_jsonl, load_jsonl,
        to_trl_sft, to_trl_dpo, to_atropos_rollout, to_chatml,
        # Adapters (optional)
        TrlAdapter, TRL_AVAILABLE,
        AtroposAdapter, ATROPOS_AVAILABLE,
    )

Typical workflow::

    rec = RLRecorder()
    agent = create_claw_agent(name="claw")
    agent.on_event = rec.observe          # capture turns
    answer = agent.run("solve x^2 = 16")

    traj = rec.finalise(prompt="solve x^2 = 16", final=answer)
    traj.reward = ContainsScorer("x = 4")(traj)

    export_jsonl([traj], "runs.jsonl")

The exporter writes ChatML-compatible JSONL by default; the
``to_trl_*`` and ``to_atropos_*`` helpers reshape that into framework-
specific layouts.
"""

from clawagents.rl.errors import (
    MissingRLDependencyError,
    RLError,
)
from clawagents.rl.trajectory import (
    Trajectory,
    TrajectoryRole,
    TrajectoryStep,
    ToolCall,
    to_next_state_transitions,
)
from clawagents.rl.recorder import (
    RLRecorder,
    RecorderConfig,
)
from clawagents.rl.scorers import (
    RewardScorer,
    ContainsScorer,
    ExactMatchScorer,
    RegexScorer,
    LengthPenaltyScorer,
    CompositeScorer,
)
from clawagents.rl.export import (
    export_jsonl,
    load_jsonl,
    to_chatml,
    to_trl_sft,
    to_trl_dpo,
    to_atropos_rollout,
)
from clawagents.rl.adapters import (
    TrlAdapter,
    AtroposAdapter,
    TRL_AVAILABLE,
    ATROPOS_AVAILABLE,
)

__all__ = [
    # Errors
    "MissingRLDependencyError",
    "RLError",
    # Data model
    "Trajectory",
    "TrajectoryRole",
    "TrajectoryStep",
    "ToolCall",
    "to_next_state_transitions",
    # Recorder
    "RLRecorder",
    "RecorderConfig",
    # Scorers
    "RewardScorer",
    "ContainsScorer",
    "ExactMatchScorer",
    "RegexScorer",
    "LengthPenaltyScorer",
    "CompositeScorer",
    # Export
    "export_jsonl",
    "load_jsonl",
    "to_chatml",
    "to_trl_sft",
    "to_trl_dpo",
    "to_atropos_rollout",
    # Adapters
    "TrlAdapter",
    "AtroposAdapter",
    "TRL_AVAILABLE",
    "ATROPOS_AVAILABLE",
]
