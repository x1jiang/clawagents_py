# SkillOpt probe (clawagents)

Source: [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) · local mock + clawagents A/B on 2026-07-14.

## What SkillOpt is

Offline **skill-document optimizer** for a frozen model: rollout → reflect → bounded edits → **held-out validation gate** → export `best_skill.md`. Companion **SkillOpt-Sleep** does nightly harvest → mine → replay → consolidate → stage for human adopt.

Not a skill-retrieval/ranker. Zero extra inference cost at deploy once the skill file exists.

## SkillOpt-Sleep (their harness, mock)

| persona | baseline → after | lift | gate blocks harmful |
|---|---|---|---|
| researcher | 0.33 → 1.00 | +0.67 | yes |
| programmer | 0.32 → 1.00 | +0.68 | yes |

`python -m skillopt_sleep.experiments.run_experiment --backend mock --assert-improves` → **PASS**.

Empty-project dry-run: `0 sessions → 0 tasks` (noop). Harvest only understands Claude Code / Codex transcript layouts — not clawagents `.clawagents` sessions.

OpenClaw plugin under `plugins/openclaw/` is **reference-only / not runnable** (absolute paths, API drift).

## Clawagents micro A/B (`gpt-5-nano`)

Always-injected operating procedure (SkillOpt deploy shape), 3 arXiv-id tasks:

| condition | mean | scores | sec |
|---|---|---|---|
| none | 0.667 | [0.5, 0.5, 1.0] | 10.0 |
| harmful_ungated | 0.000 | [0, 0, 0] | 36.0 |
| skillopt_gated | 1.000 | [1, 1, 1] | 8.4 |

- Ungated “title only” procedure collapses accuracy and burns tools (web_fetch/search).
- Gated-style format rules lift to perfect and stay cheap.
- Matches SkillOpt-Sleep’s published stress case: without a holdout gate, self-edits can destroy competence.

## Relevance to clawagents

| Area | Overlap | Gap |
|---|---|---|
| PTRL lessons + `skill_workshop` promotion | Same *idea* (learn from runs → durable skill text) | Promotion is recurrence-count (`≥3`), **not** scored held-out replay |
| Progressive `use_skill` catalog | Consumes `SKILL.md` artifacts SkillOpt could write | SkillOpt assumes always-on skill at train/deploy; our catalog is opt-in via `use_skill` |
| ATLAS | Different layer (runtime failure taxonomy) | Don’t conflate |
| Sleep / nightly | Closest product fit | Need transcript adapter + task mining for clawagents sessions |

## Recommendation

**Useful as offline opt-in**, not a default runtime toggle.

1. Borrow the **held-out gate** before promoting PTRL lessons → skills (biggest safety win).
2. Optional: thin `skillopt-sleep` wrapper once we can harvest clawagents transcripts into their session digest format.
3. Skip shipping paper `skillopt-train` benchmarks into the product path (heavy, separate research engine).
4. Do not treat SkillOpt as a replacement for skill ranking — different problem.
