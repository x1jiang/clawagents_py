from clawagents.trajectory.recorder import (
    TrajectoryRecorder, TurnRecord, RunSummary, ToolCallRecord,
    classify_failure,
    prune_trajectories,
)
from clawagents.trajectory.lessons import (
    extract_lessons,
    save_lessons,
    load_lessons,
    build_lesson_preamble,
    build_rethink_with_lessons,
    should_extract_lessons,
    export_lessons,
    import_lessons,
)
from clawagents.trajectory.verifier import (
    compute_deterministic_score,
    detect_task_type,
    verify_task_outcome,
    compute_adaptive_rethink_threshold,
)
from clawagents.trajectory.compare import (
    compare_samples,
    select_best_candidate_blind,
)
from clawagents.trajectory.harness_surface import (
    DEFAULT_SURFACE,
    EditAttribution,
    ExperienceEntry,
    ExperienceLibrary,
    SurfaceSpec,
    accept_or_repair_tree,
    assign_anon_ids,
    autotrigger_skills,
    check_hooks,
    lint_task_prompt,
    validate_harness_tree,
)
from clawagents.trajectory.harness_recursion import (
    PPCResult,
    Proposal,
    ProposerCache,
    Rollout,
    propose_probe_compose,
    record_experience,
)
from clawagents.trajectory.judge import (
    judge_run,
)
