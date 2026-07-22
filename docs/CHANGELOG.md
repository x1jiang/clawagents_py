# ClawAgents Changelog

## Unreleased

- Refactor: split the agent loop into smaller modules; extracted agent_loop utilities into dedicated modules to reduce cognitive load and improve testability.
- Feature: context observatory / context-monitor integration merged; enables LLM context inspection and monitoring during runs.
- Docs: changelog now includes a dedicated Unreleased section to track upcoming changes before formal version release.


### v6.20.42 — Lower-churn patch, sandbox, timeout, and PTY recovery (July 2026)

- **Patches:** route ambiguous/stale apply_patch failures toward refreshed
  single-hunk or hashline edits.
- **Sandbox:** refuse unauthorized `unsandboxed=true` before execution; explain
  temporary private gcloud config under OS sandbox profiles.
- **Execute:** auto-background timed-out local commands through default
  OS-sandbox profiles without dropping the wrap.
- **PTY:** retain completed screens and exit diagnostics instead of returning
  `unknown session_id` after reap.

### v6.20.41 — Resilient skill paging and actionable audit findings (July 2026)

- Treat repeated same-name `use_skill` calls as continuation pages.
- Explain when a tool is outside a skill boundary and stays unavailable.
- Classify nonzero `npm audit` reports as security findings without weakening
  failed status.
- Keep deploy safeguards framework-generic; reconciliation must stay read-only.

### v6.20.40 — Fail-closed external-action reconciliation (July 2026)

- Require approved pre-action verification and post-action reconciliation for
  external publish/deploy actions.
- Consume authorization before execution so failures/timeouts cannot hide
  partial remote state.
- Block retries, mutations, and final completion until reconciliation succeeds.

### v6.20.39 — Context Mode binary guard + hashline recovery (July 2026)

- **ctx_execute_file:** reject binary PDF/DOCX/image/ZIP inputs before UTF-8
  decode; description steers agents to `ctx_execute` or document tooling.
- **MCP errors:** keep detailed failure text in `output` without duplicating it
  into the short `error` field.
- **hashline_edit:** OpenAI-strict `edits.items` schema; lenient JSON-string
  edit items; incomplete `LINE:HASH` anchors return both fresh endpoints.
- **Test harness:** AST unbound-local scan runs in a shallow worker thread to
  avoid CPython 3.11 recursion-depth mismatches under pytest-xdist.

### v6.20.38 — Prompt-cache affinity + streaming telemetry (July 2026)

- **OpenAI prompt cache:** stable hashed `prompt_cache_key` + session affinity
  headers so multi-turn runs reuse the same opaque cache identity.
- **Incremental token ledger:** provider-reported input tokens checkpoint the
  transcript; later rounds estimate only newly appended messages.
- **Telemetry:** TTFT, input/cache tokens, and observed peak RSS on usage events;
  Gemini cache-read accounting included.
- **Bounded exec output:** command streams keep head+tail while discarding the
  middle; complete spills can be adopted as retrievable artifacts.
- **Luna/MCP:** context-protection MCP tools (e.g. `ctx_*`) stay visible under
  the reduced active tool profile; dynamic MCP tools appear in `mcp` groups.

### v6.20.37 — Plan invariant enforcement (July 2026)

- **Plan invariants survive Act:** approved plans can declare exact backticked
  commands under `Verification gates`; the shared registry persists their state
  and blocks publish/deploy-style commands until every gate succeeds.
- **Fresh evidence only:** any later source mutation invalidates all prior checks,
  successful high-impact actions consume their authorization, pending plans stay
  fail-closed across turns, and corrupt contract state cannot silently bypass the
  gate. Invariant-only plans require a recognized test/validation/dry-run command.


### v6.20.36 — Bounded scratch cleanup permissions (July 2026)

- **Permission precision:** allow direct recursive cleanup of exactly one literal
  `/tmp/<name>` directory instead of rejecting every absolute `rm -rf` command.
- **Safety retained:** root/system paths, `/tmp` itself, globs, variables,
  traversal, multiple targets, and wrapper-launched recursive deletes stay denied;
  explicit user permission rules still override the built-in exception.
- **False-failure cleanup:** constrain encoded-interpreter detection to the actual
  `python -c` / `perl -e` / `ruby -e` payload, normalize empty `grep`/`rg` exit 1
  as “no matches,” and keep empty-command failures visible.
- **Recovery guidance:** identify intentional quarantine outcomes and missing-file
  probes so agents inspect manifests/parents instead of retrying unchanged; detect
  missing subprocess executables and separate cleanup cascades from the root cause.
- **Patch diagnostics:** multi-hunk SEARCH failures identify the failing hunk and
  staged predecessors while confirming atomic rollback; JSON output is parsed
  before write to reject copied escape sequences or other structural corruption.
  Near-match percentages no longer round mismatches to 100%, and diagnostics
  expose the first differing column plus Markdown list/table structure mistakes.
- **Opaque shell failures:** distinguish advisory validator warnings from the
  actual exit, explain `&&` short-circuiting, and identify redirected log paths
  when a nonzero command returns no captured stdout/stderr.
- **Loop cascades:** recognize missing-input producer failures followed by
  empty-JSON consumer errors, preserve partial successes, and explain that shell
  `for` loops require explicit per-iteration guards.
- **Python environments:** classify `ModuleNotFoundError` as a selected-interpreter
  dependency issue, prefer the project virtual environment and same-interpreter
  `-m pip show` checks, and discourage accidental global installs.


### v6.20.35 — Failure discipline + shell-secret redaction (July 2026)

- **Execute diagnosis:** classify external authentication rejection, unavailable
  packages, unsafe secret interpolation, and client syntax failures so agents stop
  switching runtimes/tools when the evidence requires user or configuration action.
- **Secret hygiene:** redact mixed-case high-entropy values accidentally interpreted
  as shell commands before execute stdout/stderr reaches model, UI, or persistence.
- **Loop safety:** the existing three-consecutive-failure stop-and-classify guard is
  active by default; optional ``rethink`` still controls advisor/learning behavior.
- **Shell-session heredocs:** separate cwd/env bookkeeping with a newline so quoted
  heredoc terminators remain valid and the user's original exit code is preserved.


### v6.20.34 — UI tool results + Codex apply_patch (July 2026)

- **UI tool results:** stream up to 8KB of uncrushed tool text to hosts while
  keeping model/console ``preview_chars`` small.
- **Failed execute:** observation shows exit code + stderr/stdout before the
  long command; short ``error`` field is exit-code-only.
- **apply_patch:** accept single-file Codex ``*** Begin Patch`` / ``*** Update
  File`` envelopes (path must match; multi-file / Add / Delete / Move rejected).

### v6.20.33 — Scratch /tmp write parity + secret CRLF hygiene (July 2026)

- **`write_file` /tmp parity:** writable OS sandbox profiles allow OS temp /
  `/tmp` / `/private/tmp` for in-process file tools (matches seatbelt execute).
- **Secret CRLF scrub:** strip lone `\r` from password/key env values after
  dotenv load; prefer `python3` in auto-verify and Graphify skill docs.

### v6.20.32 — Graphify skill: code-only bootstrap (July 2026)

- Document `extract --code-only` / `update` as the reliable offline path; plain
  `extract` can exit 0 without writing `graph.json` when the semantic pass fails.

### v6.20.31 — Native Graphify companion (July 2026)

- **`MIN_GRAPHIFY` / `probe_graphify()`** — version floor for PyPI `graphifyy` (local knowledge graph).
- Bundled skill `skills/graphify/SKILL.md` for CLI/MCP workflows.
- VS Code wires Graphify as a Context Mode–style MCP companion (see clawagents-vscode).

### v6.20.30 — Contained-read reuse + mode tool profiles (July 2026)

- **Overlapping reads:** reuse a prior read stub only when the new line range is
  fully contained in a prior range (or the prior was unbounded). Partial overlaps
  (e.g. 0–100 then 50–150) run a real read so required lines are not dropped.
- **`token_estimator_info()`:** diagnostics can report tiktoken vs chars÷4.
- **Mode tool profiles:** read-only / coding / goal active sets via
  `apply_mode_active_profile` (Luna / openai-gpt56).

### v6.20.29 — Luna active tools + loop guard + economic fixes (July 2026)

- Active tool surface for GPT-5.6: core profile + `activate_tool_group` (web/git/pty/…)
- Deterministic identical/overlapping `read_file` reuse stubs; harness soft/hard loop thresholds 2/3
- Soft-trim/micro-compact before 272K long-context cliff; harness prompt applied in create_claw_agent

### v6.20.28 — Luna economic context + efficiency harness (July 2026)

- GPT-5.6 family: `long_context_threshold` 272K; soft-trim / micro-compact fire before the long-context pricing cliff.
- New `openai-gpt56` harness (efficiency rules + earlier tool clearing); harness prompt suffix applied in `create_claw_agent`.

### v6.20.27 — UsageEvent cache fields (July 2026)

- `UsageEvent` now carries `cached_input_tokens` / `cache_creation_tokens` (promoted from stream data) so hosts can show cache hit rate and cache-aware cost.

### v6.20.26 — Plan mode Grok Build parity (July 2026)

- UI `chat_mode=read_only|plan` sets engine `PermissionMode.PLAN` by default.
- PLAN allows `write_plan` / enter+exit plan tools / skills / ask_user; edits only to `.clawagents/plan.md` or `.grok/plan.md`; other write-class tools stay blocked until `exit_plan_mode` is approved.

### v6.20.25 — YAML skill frontmatter + nested shlex guard (July 2026)

- Skill ``SKILL.md`` frontmatter parsed with ``yaml.safe_load`` (flow-style / quoted ``allowed-tools``, CRLF, flush-left dashes).
- Unbound-local import guard collects free loads from nested ``def``/``async def`` (meta-tested).

### v6.20.24 — Bedrock id + skill drain gate (July 2026)

- ``is_bedrock_model_id`` requires a foundation-model vendor prefix after geo (``us.``/``eu.``/…).
- Skill allow-list preflight refuses disallowed tools *before* auto-drain so pending pages are not discarded.

### v6.20.23 — Round-2 audit: skills, apply_patch, MCP, compaction (July 2026)

- Circuit breaker defaults on (``provider_circuit_breaker=1``) now that nested Responses retries are gone.
- Compaction message reuse keys on tool-call ids / names so empty assistant/tool bodies do not swap ``tool_calls_meta``.
- Active skill ``allowed-tools`` filters the LLM tool list (not only at call time); Claude Code aliases (``Read``/``Bash``/``git_status``/…).
- Skill frontmatter: flush-left YAML dashes + CRLF ``allowed-tools`` lists parse correctly.
- ``apply_patch``: tolerate trailing space on fence markers; softer whitespace match; nearest-match hint on SEARCH miss.
- MCP: pass ``client_session_timeout_seconds`` to the SDK; connect timeout; reconnect when session is ERRORED.

### v6.20.22 — Responses retry, $ref schemas, model-prefix classifiers (July 2026)

- Non-streaming Responses no longer nests ``_with_retry`` around ``_stream_with_retry_responses`` (was up to 16 HTTP calls).
- MCP/tool schemas resolve ``$ref``/``$defs`` and ``anyOf``/``oneOf`` (pydantic Optional / nested models).
- Strip ``openai.`` / ``azure.`` / ``mantle.`` prefixes in Responses/reasoning classifiers.
- Responses stream tool accumulation keys by ``call_id`` (not only ``output_index``).
- Clearer circuit-breaker-open errors when the feature is enabled.

### v6.20.21 — Nested tool schemas + model identity (July 2026)

- MCP bridge and OpenAI / Responses / Gemini / Anthropic / Bedrock emitters preserve
  nested ``items.properties`` / ``required`` (fixes Gemini 400
  ``properties[…].items: missing field`` for tools like ``ctx_batch_execute``).
- Shared helper: ``clawagents.providers.tool_schema``.
- System prompt includes a static ``## Model identity`` line keyed to the
  configured ``provider/model`` (stops Claude/Google identity flip-flops).

### v6.20.20 — Skill continue=true + crush/code sniff + retrieve under skill gate (July 2026)

- ``use_skill(continue=true)`` resumes from server-side offset/hash (no 64-char sha256 echo); continuation prompts and errors mention hash mismatches honestly.
- Numbered source (``nl -ba`` / line dumps) classified as ``code`` before HTML sniff so embedded templates keep the 4K floor.
- ``retrieve_tool_result`` is skill-gate control-plane (matches crush recovery headers).

### v6.20.19 — Control-plane crush exemption + post-edit syntax gate (July 2026)

- ``use_skill`` / ``list_skills`` / ``retrieve_tool_result`` outputs are never crushed (skill instructions stay verbatim; auto-drain pages too).
- After successful write tools on ``.js`` / ``.mjs`` / ``.cjs`` / ``.py`` / ``.sh``, append a ``[syntax_gate]`` result from ``node --check`` / ``py_compile`` / ``bash -n``.

### v6.20.18 — Chat mode ↔ OS sandbox contract (July 2026)

- ``sandbox_profile_for_chat_mode``: ``full_access``+gate → ``off``, ``read_only`` → ``read-only`` (wired in ``create_claw_agent(chat_mode=…)``).
- Workspace env banner shows ``sandbox: <profile>``.
- ``seatbelt.sb`` stamped with ``; generated by clawagents X.Y.Z``.
- Failed tool outputs are not aggressively crushed (credentials.db paths stay verbatim).
- ``execute(unsandboxed=true)`` + auto-retry once on sandbox EPERM when Full access is on.
- Clearer ``/dev/null`` vs home-config EPERM hints.

### v6.20.17 — Seatbelt /dev/null + clearer sandbox hints (July 2026)

- Seatbelt writable profiles always allow ``/dev/null`` (CLIs redirect there constantly).
- ``execute`` sandbox hints call out home-config / gcloud credential denials and how to disable the OS sandbox.

### v6.20.16 — snapshot_diff file-cap note (July 2026)

- ``snapshot_diff``: when more than 40 files are present, header says ``showing 40 of N`` (no silent cap).

### v6.20.15 — Skill-loading auto-continuation (July 2026)

- Multi-page ``use_skill``: when a skill page is pending, the harness auto-finishes remaining pages (deterministic name/offset/hash) then runs the tool the model asked for — no refusal loop / wasted turns.
- Hash-mismatch / non-contiguous continuation clears pending state so ``restart at offset 0`` is executable (fixes permanent deadlock when the skill file changes mid-load).
- ``use_skill(abort=true)`` escape hatch clears a pending load without reading to EOF.

### v6.20.14 — apply_patch corruption guard + snapshot_diff + crush floor (July 2026)

- ``apply_patch``: line-based SEARCH/REPLACE parser (empty REPLACE = deletion); refuse writes that introduce fence markers; return unified diff of applied change; stricter unified-diff hunks.
- ``snapshot_diff`` tool: git-free review vs ``.clawagents/snapshots/``.
- Workspace env preamble: ``is_git_repo``, ``scratch_dir``; prefer ``snapshot_diff`` when no git.
- Aggressive crush: code/log/diff and ``read_file``/hashline kinds use ≥4K floor (stop crushing 2.5K reads).
- Seatbelt: allow ``/tmp`` + ``/private/tmp``; exec hints on Operation not permitted.

### v6.20.13 — Clearer git / hashline / execute DX (July 2026)

- ``git_status`` / ``git_diff``: soft success with a clear notice when cwd is not a git repo (skip further git); commit/undo still hard-fail.
- ``execute``: exit 128 “not a git repository” gets an interpretation that says not to chain ``&& git`` after syntax checks.
- ``hashline_edit``: malformed anchors include sample valid ``LINE:HASH…`` anchors from the file; tool description insists on full anchors.
- ``apply_patch`` SEARCH-miss message points at re-read / hashline_edit.

### v6.20.12 — Omit temperature for Mantle GPT-5.6 / Responses (July 2026)

- OpenAI / Mantle Responses: do not send ``temperature`` for GPT-5.5/5.6 (incl. ``openai.gpt-5.6-luna``) and o-series — API returns 400 ``temperature is deprecated for this model``.
- Mantle: bare ``gpt-5.6-*`` routes to ``/openai/v1/responses`` and is normalized to ``openai.gpt-5.6-*`` (bare ids 404).

### v6.20.11 — Omit temperature for Claude Opus 4.7+ (July 2026)

- Anthropic Messages / Mantle Claude: do not send ``temperature`` (or sampling params) for Opus 4.7+ — API returns 400 ``temperature is deprecated for this model``.

### v6.20.10 — Provider/model resolution hardening (July 2026)

- Canonical `providers.model_classify` (geo prefixes, LiteLLM `provider/` strip, key-field routing).
- `PROVIDER=` env hint no longer silently falls through when the matching key is missing.
- PromptHook default resolver imports `agent._resolve_model` (was fail-open on ImportError).
- Profile `provider` drives routing; `apac.` Mantle strip fixed; FallbackProvider quarantine message clarified.

### v6.20.9 — Capabilities contract + workspace-scoped artifacts (July 2026)

- `clawagents.capabilities` / capability flags for hosts (`gemini_array_items`, `workspace_scoped_agent`, `raw_tool_output`, `artifact_workspace_arg`).
- Agent loop passes `workspace=` into `prepare_tool_output_for_context` so artifacts land under the agent workspace, not bare cwd.
- Model context profiles moved to `graph/model_profiles.py`; verbose history lives in this changelog (README slimmed).

### v6.20.8 — Artifact security + raw tool archival (July 2026)

- Tool artifact load/search never follows metadata paths outside `.clawagents/tool-artifacts/`.
- `ToolResult.raw_output` keeps the full dump while `output` may be truncated for UI/cache; `retrieve_tool_result` can recover full text.

### v6.20.7 — Central secret paths (July 2026)

- `clawagents.security.secret_paths` is the single source for sandbox globs, permission defaults, and hunk-watcher ignores.
- Basename matching uses `*credentials*` / `*secrets*` (no `secretary.txt` false positive).

### v6.20.6 — Dead handoffs + DX (July 2026)

- Pin LLM model across nested runs; `run_context` session/event wiring; `instruction`↔`system_prompt` / `workspace=` aliases.
- Anthropic conversation `cache_control`; tokenize cache; `temporary_overrides` for `features=`.

### v6.20.5 — Mantle multi-path routing (July 2026)

- **Mantle (OneHUB):** `anthropic.*` → `/anthropic/v1/messages` (Anthropic SDK); frontier `openai.gpt-5.3/4/5/6*` → `/openai/v1/responses`; chat-ok models stay on `/v1/chat/completions`. Fixes Claude Haiku 400 on chat completions.
- Optional `EngineConfig.anthropic_base_url` for custom Anthropic-compatible hosts.

### v6.20.4 — Execute cancel + shell-session sync (July 2026)

- Cancelled foreground `execute` now SIGKILLs the process group (CancelledError was skipping cleanup and leaving orphans)
- Explicit `is_background` no longer injects unused PWD/ENV trailers; auto-background-on-timeout syncs shell-session cwd/env when the job finishes

### v6.20.3 — Execute path hardening (July 2026)

- Background jobs: ProfileBackend wrap + LocalBackend-aligned env scrub; `start_new_session` + killpg cancel
- bwrap: touch missing secret overlay targets (`.env`); surface sandbox soft-fallback warnings
- Permanent AST test against use-before-local-import (seatbelt `shlex` class)

### v6.20.2 — Seatbelt execute fix (July 2026)

- Fix `ProfileBackend.exec` unbound `shlex` on seatbelt path (macOS OS sandbox)

### v6.20.1 — Grok harness hardening (July 2026)

- Shell session: trailer-only `__CLAW_PWD__` / `__CLAW_ENV__` consume; sticky key/value caps; python3|python dump fallback
- edit_file: `create_if_missing` only when path absent; reject stringy `"false"` bools
- execute: negative `block_until_ms`/`timeout` ignored; streaming retain cap
- hashline_grep / pty_start input bounds

### v6.20.0 — Grok harness ports (July 2026)

- Deeper `edit_file` (external-mod miss text, `create_if_missing`, soft RBE guidance)
- `execute` `block_until_ms` + streaming `tool_progress` + sticky session env (feature-gated)
- `hashline_grep` + registry path/parallel-safe + prompt nudge
- Execute↔PTY description routing; `pty_start` uses shell_session cwd
- VS Code companion pin: **≥6.20.0**

### v6.19.0 — Companion lockstep (July 2026)

- `clawagents.companions`: version floors + probes for Context Mode and RTK
- `clawagents --doctor` Companions section with upgrade hints
- VS Code companion: **1.0.55** (auto-ensure on sidecar start)

### v6.18.0 — Grok-inspired edit/execute harness (July 2026)

- `hashline_read` / `hashline_edit` (feature `hashline_tools`)
- `execute` shell session cwd + auto-background on timeout + optional `is_background`
- RTK wrap + aggressive in-loop tool-output crush
- Richer `edit_file` miss diagnostics

### v6.17.8 — Residual P1/P2 closures (July 2026)

- Secret filter on agent-write/snapshot/hunk baselines; `.env` path normalization fix
- Webhook DNS pin; breaker on all stream providers; doom-loop force-response channel

### v6.17.7 — P2/P3 correctness (July 2026)

- Circuit breaker endpoint isolation + streaming + non-burning BreakerOpen waits
- Session FTS5 populate; smart_store FTS replace; flush cycle guard; dream lock finally
- Doom-loop response channel; HistoryThenSteps graduated; PTY reaper; interject export

### v6.17.6 — P1 security hardening (July 2026)

- Taxonomy/`hooks.json` gated behind `external_hooks` + `hook_taxonomy` (both default off)
- macOS seatbelt quoting escape fixed; in-process secret path deny; webhook SSRF fail-closed
- PTY plan-mode + env scrub; hunk watcher ignores secrets; dream writes under `.clawagents/`
- Rewind prompt_index monotonic across host RunContext resets

### v6.17.5 — Tool error regressions (July 2026)

- Claude skill `allowed-tools` YAML lists + aliases
- grep glob-as-path; apply_patch Begin Patch guidance

### v6.17.4 — Act vs Goal isolation (July 2026)

- Goal tools / reminder / verifier gated on `goal_mode`
- Act/Plan pauses active `.clawagents/goal` state

### v6.17.3 — Tier-2 wiring completion (July 2026)

- Taxonomy: StopFailure, PostToolUseFailure, PermissionDenied, Notification, SubagentStart/Stop
- Typed hunk attribution; conversation rewind markers; bwrap secret-deny overlays
- Companion — VS Code **1.0.47**

### v6.17.2 — Fix LLM complete/chat contract (July 2026)

- `LLMProvider.complete()` aliases `chat()`; goal/flush/dream use `chat()` (no `stream=`)
- Companion — VS Code **1.0.47**

### v6.17.1 — Circuit-breaker + interject synthetic turns (July 2026)

- Half-open probe lease reclaim so cancelled probes cannot strand the breaker
- Interject: each redirect is a standalone synthetic user turn; stranded undrained redirects become queued prompts
- Feature flag `provider_circuit_breaker` (default off); companion VS Code **1.0.46** (voice dictation + stranded queue)

### v6.17.0 — Grok Build Tier 1+2 parity (July 2026)

- Smart memory (boost/decay/dream/flush/hybrid FTS), PTY sessions, structured output, doom-loop resample
- Compaction HistoryThenSteps + greppable segments; hunk watcher + session rewind
- Hook taxonomy (14 events) + HTTPS webhook SSRF blocklist; sandbox fail-closed / secret deny / add-only profiles
- Companion — VS Code **1.0.45**

### v6.16.0 — Removed ATLAS + permission/hunk hardening (July 2026)

- Removed ATLAS integration (`clawagents.atlas` gone; `atlas=` / `CLAW_ATLAS` ignored)
- Goal autopilot is the only long-horizon path; `goal_mode` injects the GOAL nudge
- Subagents inherit permission engine; hunk accept paths confined; permission `ask` → host handler
- Companion — VS Code **1.0.44**

### v6.15.0 — Goal autopilot + OS sandbox (July 2026)

- Goal autopilot (planner → majority verifier → strategist)
- Workspace OS sandbox default; seatbelt/bwrap auto-upgrade; deny-then-allow writes
- Deny-wins permission rules on by default; best-of-n skill; prefire compaction; mid-turn interject
- Companion — VS Code **1.0.43**


### v6.14.2 — Skill strategy (Grok-aligned) (July 2026)

- `when-to-use` / path-gated skills, `$ARGUMENTS` / `${SKILL_DIR}` substitutions
- Hot reload + discovery announcements; high-confidence `use_skill` suggest (no auto-load bodies)
- Compaction carryover tracks `invoked_skills` from fully-read skills

### v6.14.1 — Full-replace compaction (July 2026)

- Grok-style full-replace assemble after summarize; tool-pair snap; AGENTS/plan reinject

### v6.14.0 — Grok Build parity pack (July 2026)

- Plan approval gate, subagent resolution layers, `task(isolation=worktree)`
- Attributed hunk review, marketplace install, autopilot loop, OS sandbox profiles, incremental scope graph

### v6.12.0 — Native Amazon Bedrock (July 2026)

- `BedrockProvider` (Claude / `AsyncAnthropicBedrock`) and `BedrockConverseProvider` (Nova, Llama, …)
- Optional extra `clawagents[bedrock]` (`anthropic` + `boto3`)
- Gateway path unchanged when `openai_base_url` / `base_url` is set

### v6.11.2 — Tavily `web_search` (July 2026)

- Built-in `web_search` tool via Tavily (`TAVILY_API_KEY`); fixed HTTPS host only.
- Registered alongside `web_fetch` on `create_claw_agent`.

### v6.11.1 — CodeAct sandbox + checkpoint ref hardening (July 2026)

- Curated CodeAct builtins; block open/import/eval escapes that bypass permissions.
- Validate checkpoint SHAs before git reset/diff.
- Fix evals CLI `judge_run` argument order.

### v6.11.0 — Steal-list pack (July 2026)

- Shadow-git restore modes + turn binding; path-faithful file snapshots.
- Always-on `.clawagents/rules/` + CLAUDE.md every LLM round.
- Custom modes JSON + CLI `--mode` / `--auto`; CodeAct `action_mode=code`.
- `clawagents evals` suite runner; `approval_handler` / require_approval tools.

### v6.10.8 — Compaction correctness (July 2026)

- Do not declare compaction success while still over the context budget; escalate to summarization.
- Reuse original message objects through compression (session tracker + tool linkage).
- Include judge-call tokens in run `Usage`.

### v6.10.7 — Peer-inspired harness pack (July 2026)

- Repo map, commit-boundary context ledger, shadow-git checkpoints.
- `apply_patch`, core memory blocks, git/worktree tools, plan.md handoff.
- Harness clear-tool knobs; compaction thrash guard; local fact supersession.

### v6.10.6 — Context headroom pack (July 2026)

- Prompt-cache stable prefixes; lessons after `__CACHE_BOUNDARY__`; alphabetical tool schemas.
- Tiered `read_file` (L0/L1/L2); HTML/diff/test crushers; multimodal sanitize on ingest.
- Wire `compress_messages_safe` + compact hooks; output trim; micro-compact artifact ids.
- Failure lessons → `AGENTS.md`; local artifact search via `retrieve_tool_result(query=…)`.

### v6.10.5 — Gemini history 400 recovery (July 2026)

- Retry once with flattened tool history on FR/FC / thought_signature 400s.
- Keep thought_signature only on the first parallel function_call part.
- Close native tool pairs when an external hook blocks a call.

### v6.10.4 — Gemini 3 id + thought_signature (July 2026)

- Echo `function_call.id` on matching `function_response`.
- Preserve / base64-round-trip `thought_signature` in `gemini_parts`.
- Rebuild Gemini contents with strict FC→FR pairing.

### v6.10.3 — Gemini FR/FC turn purity (July 2026)

- Do not mix `function_response` with plain user text in one turn.
- Drop orphan FRs; rebuild model turns when `gemini_parts` omit `function_call`.

### v6.10.2 — Gemini turn hygiene (July 2026)

- Coalesce consecutive Gemini `user`/`model` turns (parallel tool results were each a separate `user` turn and broke alternation).
- Synthesize missing `function_response` parts when a function_call was skipped or never answered.
- Close native tool pairs on `before_tool` / approval rejection so session replay stays valid.

### v6.10.1 — Provider, MCP, and session patch (July 2026)

Python-focused reliability patch for hosts that embed the agent (VS Code sidecar,
threaded servers) and for GPT-5.5/5.6 + Gemini tool calling.

- **Tool schemas** — OpenAI/Gemini converters always attach `items` for arrays.
- **GPT-5.5/5.6** — prefer Responses API; Chat Completions fallback still forces
  `reasoning_effort=none` when tools are present; sanitize orphan `role=tool`
  messages before the request.
- **Sessions** — drop leading orphan tools on limited preload; persist the user
  task message; re-sanitize after compaction; insert replayed history in order.
- **MCP** — reconnect when the owning event loop changes; regression coverage
  with a real stdio echo server.
- **API** — `skills_exclude` on `create_claw_agent`; richer typed stream events;
  OpenAI `cached_tokens`; google-genai status/auth classification fixes.
- **Signals** — tolerate missing signal handlers off the main thread.

### v6.10.0 — Reliability and parity release (July 2026)

Cross-cutting hardening from the code-review backlog: session persistence,
parallel hook enforcement, provider correctness, context recovery, and
agent-loop telemetry — mirrored in the TypeScript port where applicable.

- **Session persistence** — identity-based tracking replaces index cursors that
  broke on compaction or mid-list inserts.
- **Parallel tools** — external policy hooks and session writes on every branch.
- **History offload** — redacted before writing to `.clawagents/history/`.
- **Providers** — consistent token accounting; stream partials keep tool calls;
  `repair_json` closes truncated strings; error taxonomy word-boundary matching.
- **Tokenizer** — multimodal counts use real text, not BPE-compressed filler.
- **Compaction** — safe split index keeps tool_call/tool_result pairs intact.
- **Agent loop** — per-round iteration counter; idempotent prompt injection;
  micro-compact gated on usage; overflow shrinks effective window; advisor/handoff
  transcript fixes.
- **Process** — command-queue barrier exclusivity; heartbeat work-task cancel;
  bounded gateway WS sessions; ACP call-id matching; PIL pixel ceiling.

Release verification: **Python 916 passed, 8 skipped** (`scripts/run_tests.sh`).

### v6.9.2 — Security and provider hardening (July 2026)

Patch release closing security review findings and provider parity gaps.

- **Bash validator** — wrapper peeling for `env`/`sudo`/`timeout`/`eval`/…,
  alias bypass handling, root-path normalization.
- **Gateway CORS** — localhost-only default origins; no credentials with `*`.
- **Plan-mode escape** — agent-as-tool forwards parent `run_context`.
- **Providers** — cache-boundary stripping, Anthropic `temperature=0`, Gemini
  image URL safety, OpenAI empty-choices guard.
- **Steer hook** — nudges append `LLMMessage` not dicts.
- **Skill workshop** — block apply on any scan finding.
- **Sandbox** — `is_secret_name()` env redaction beyond static denylist.
- **Coordinator** — normalize malformed task shapes from LLM output.
- **Registry** — write-snapshot path confined to workspace root.

Release verification: **Python 97 security-regression tests passed**.

### v6.9.1 — CI/test hardening (June 2026)

Patch release ensuring the full pytest suite passes in keyless CI environments
(no `.env`, empty `OPENAI_API_KEY`). Factory and integration tests now mock
provider construction or pass proper `LLMProvider` stubs instead of relying on
local API keys leaking in from parent directories.

### v6.9.0 — History recall, CLI output formats, governed skill promotion (June 2026)

Minor release focused on machine-readable CLI output, cross-session memory
recall, and closing the loop from PTRL lessons to governed skill proposals.

- **`search_history` tool** — searches the cross-session archive (SQLite
  `sessions.db` + optional JSONL logs) and returns raw message snippets with
  session id, role, and highlighted excerpt. Supports `session_id` filter and
  `format=json`.
- **`--output-format`** — `clawagents --task` accepts `text`, `json`, or
  `stream-json` for scripting and CI integration.
- **PTRL lesson promotion** — lesson bullets seen ≥3 times create pending
  `skill_workshop` proposals tracked in `.clawagents/lesson-index.json`.
- **`skill_workshop` tool** — governed skill authoring workflow aligned with
  the TypeScript port (create, update, apply, reject, quarantine, rollback).
- **Search consolidation** — shared `search_sqlite_messages`, `snippet_from_content`,
  and canonical lesson utilities in `trajectory/lessons`.

Release verification: **Python 18 consolidation/feature tests passed**;
**TypeScript 545 passed, 4 skipped**, `tsc --noEmit`.

### v6.8.1 — Prompt architecture and release packaging polish (May 2026)

Patch release focused on keeping the Python and TypeScript packages aligned for
installed users after the OpenHarness-inspired operational surface work.

- **Prompt assembly module** — `clawagents.prompts` now owns system prompt
  construction, lesson preambles, `__CACHE_BOUNDARY__` placement, and dynamic
  memory/skill prompt injection.
- **Hook compatibility** — prompt injection remains compatible with the legacy
  dict-shaped messages used by older `before_llm` integrations.
- **OpenHarness comparison** — the feature matrix now includes
  [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) with conservative
  full/partial markers.

Release verification: **Python 851 passed, 3 skipped** plus bytecode
compilation; TypeScript sibling: **526 passed, 4 skipped**, `tsc --noEmit`, and
build.

### v6.8.0 — OpenHarness-inspired operational surfaces (May 2026)

Minor release focused on making ClawAgents easier to inspect, configure,
recover, and integrate without changing the core agent loop contract.

- **Static readiness previews** — `clawagents --dry-run --profile <name> --task
  "<prompt>"` reports resolved provider settings, auth readiness, inspectable
  tools, likely matching tools, and next actions without calling a model or
  executing tools.
- **Named provider profiles** — built-in `openai`, `gemini`, `anthropic`, and
  `ollama` profiles plus project/user profile files give stable provider
  aliases. Explicit factory parameters still take precedence.
- **Structured permission decisions** — permission evaluation now returns a
  reusable decision object with allow/confirmation/reason fields and feeds the
  registry hard-block path for plan-mode and sensitive-path decisions.
- **Background task tools** — the registry can expose task create/status/output
  /stop/list tools backed by the existing background job manager, so long-running
  work can be tracked instead of blocking an agent turn.
- **Plugin compatibility loader** — metadata-only loading for `plugin.json` and
  `.claude-plugin/plugin.json` reads plugin manifests, markdown skills/commands,
  hooks, and MCP server declarations without executing arbitrary plugin code.
- **MCP auth/reconnect helper** — MCP manager configs can be updated with new
  environment/header auth material and reconnected deliberately.

Release verification: **Python 844 passed, 3 skipped** plus bytecode
compilation and dry-run smoke; TypeScript sibling: **520 passed, 4 skipped**,
`tsc --noEmit`, build, and matching dry-run smoke.

### v6.7.1 — Tool discovery and compact-agent recovery (April 2026)

Patch release focused on generalizable low-latency tool use for compact
models. `tool_discover` is registered by default so agents can inspect the
available tool universe before committing to a call, and lookup now searches
tool names, descriptions, and keyword metadata. That makes discovery robust
when a model remembers the action it needs but not the exact tool name.

Native-tool failures now keep useful output in the observation stream instead
of reducing everything to a generic error. The built-in `execute` tool returns
structured JSON for nonzero exits (`command`, `exit_code`, `stdout`,
`stderr`, `output`, `timed_out`), and repeated identical `execute` failures
include a recovery hint that nudges the agent to inspect the captured output
or change command strategy.

Planning/todo guidance was also tightened so quick read-only or single-step
tasks do not pay unnecessary planning overhead, while multi-step repair tasks
still get explicit progress tracking. Focused release verification covers the
infra-improvement regression tests and bytecode compilation for Python, plus
TypeScript typecheck and matching infra-improvement tests.

### v6.7.0 — Security hardening across validator, web_fetch, redact, sandbox (April 2026)

Minor release. Adversarial probing of the v6.6.4 surfaces uncovered a
cluster of bypasses; this release closes them. Test totals after this
release: **Python 835 passed, 3 skipped**; **TypeScript 511 passed,
4 skipped** plus parity checks; `tsc --noEmit` clean. **49 new
regression tests** ride alongside the fixes (44 Python, 5 TypeScript).

**Bash validator hardening** — `validate_bash` now walks every shell
clause, including the contents of `(...)`, `$(...)`, backticks, and
`bash -c '<cmd>'`/`sh -c '<cmd>'` wrappers; the strictest verdict
across all clauses wins. The previous head-only inspection meant
`ls && rm -rf /var/log`, `(rm -rf /)`, `echo $(rm -rf /)`, and
`bash -c 'rm -rf /'` all silently passed. Additional shapes now
`BLOCK`: `rm -rf "$HOME"` / `rm -rf $HOME/x` and any `rm` of a system
directory (`/etc`, `/var`, `/usr`, `/home`, …); `tee /dev/sda` and
`tee /etc/passwd` / `tee -a /etc/sudoers`; quoted block-device
redirects (`>'/dev/sda'`); FD-prefixed redirects (`1>/dev/sda`);
`find -exec sh -c '…'` and `find -execdir`; `chmod -R 777 /`;
`sed --in-place` (long form, previously unrecognised). Null bytes
and unprintable control characters in any command are also `BLOCK`
(closes the C-string truncation evasion).

**Web fetch SSRF — DNS-rebinding TOCTOU eliminated** — `web_fetch` now
resolves the host once per hop and connects to the validated IP
directly, sending the original hostname via the `Host` header and SNI.
A controlled DNS server can no longer return a public address to the
validator and a private one (loopback, `169.254.169.254` / cloud
metadata) to the actual fetch. Body reads are bounded at 4 MiB and
truncated streamingly so a hostile server can't OOM the agent. Each
redirect hop gets its own timeout. `Location` headers that downgrade
HTTPS → HTTP across a redirect are refused.

**Obfuscation detector — host-suffix bypass closed** — the curl-pipe-
shell installer allowlist used `\b`-anchored regexes, but `.` is a
non-word character so `brew\.sh\b` matched `brew.sh.evil.com`.
Allowlist is now keyed on parsed hostname (with required path prefix
for `raw.githubusercontent.com`), not regex.

**`edit_file` empty-target corruption** — `target=""` plus
`replace_all=true` previously inserted the replacement between every
character of the file, silently corrupting it. Now refused.

**Redaction coverage** — `redact()` now scrubs PEM private-key blocks
(any `-----BEGIN […] PRIVATE KEY-----` / `END` block), `Authorization:
Bearer <token>` / `Authorization: Basic …` headers, AWS *secret* access
keys (the previous regex covered only the access-key ID), URL
basic-auth credentials (`https://user:pass@host`), and shorter
generic-secret values. The Docker sandbox env-name policy now reuses
`is_secret_name()` from `redact.py` plus a small extras regex covering
vendor-prefixed shapes (`GITHUB_PAT`, `STRIPE_SK_LIVE`,
`DATABASE_URL`, `DSN`); the previous end-anchored regex missed
`AWS_SECRET_ACCESS_KEY`, `GITHUB_PAT`, `DATABASE_PASSWORD_PROD`, etc.
and forwarded them into containers via `-e`.

**Subprocess timeouts no longer orphan children** — the local sandbox
now starts each shell in a new session and `SIGKILL`s the whole
process group on timeout, so long-running grandchildren of `sh -c`
don't outlive the parent.

**Concurrency** — `RunContext.iteration_budget` lazy-init is serialised
under an `asyncio.Lock`; sub-agents sharing a context can no longer
clobber each other's budget. Callsite is `await
run_context.ensure_iteration_budget(size)`.

**Other quality fixes** — `RetryPolicy.shouldRetry` now correctly
allows `maxRetries=N` to perform `N` retries (was off-by-one); `jitter`
is clamped to `[0, 1]` to prevent zero-delay retry storms; the MCP
manager tracks connected servers so a partial-failure `start()`
doesn't double-register tools on retry, and shutdown errors are
aggregated into a thrown `Error` instead of a span no caller observes;
`compressMessagesSafe` no longer produces two consecutive same-role
messages when the head is empty (Anthropic rejects that). The
overbroad `"curl http"` / `"wget http"` legacy substring (which also
matched `https://` because `https` starts with `http`) is removed —
the bash validator's NETWORK classification now applies cleanly.

### v6.6.4 — Keyword discovery and infrastructure parity (April 2026)

Patch release for the v6.6 line. Test totals after this release:
**Python 786 passed, 3 skipped**; **TypeScript 509 passed, 4 skipped**
plus **49 parity checks**; `tsc --noEmit` clean.

- **Keyword-backed compact discovery** — tools can now declare explicit
  keyword aliases, `tool_discover` searches names, descriptions, and those
  aliases, and `tool_describe`/registry inspection expose the metadata so
  compact tool universes stay useful even when the model uses a near-synonym.
- **Bounded tool profiles** — catalog helpers can publish smaller tool views
  for focused agents while preserving the full registry for callers that need
  it.
- **Infrastructure parity** — Docker sandbox support, resumable `RunResult`
  metadata, SQLite result caching for safe cacheable tools, explorer helpers,
  gym-style eval aliases, and next-state trajectory export helpers now ship in
  both the Python and TypeScript packages.
- **Cache safety defaults** — read/search-style filesystem outputs remain
  uncached by default to avoid persisting sensitive repository contents, while
  explicitly cacheable pure tools can reuse results across runs.

### v6.6.3 — Efficiency and release hardening (April 2026)

Patch release for the v6.6 line. Test totals after this release:
**Python 778 passed, 3 skipped**; **TypeScript 497 passed, 4 skipped**
plus **49 parity checks**; `tsc --noEmit` clean. Real `.env` smoke tests
passed for Gemini and OpenAI, including read-only `read_file` tool use and
`task` subagent delegation in both ports.

- **Non-blocking local filesystem backend** — async `LocalBackend` file,
  directory, and stat operations now offload synchronous pathlib work with
  `asyncio.to_thread()`, so parallel-safe tool calls can yield the event loop
  instead of serializing on local disk I/O.
- **Append-only run summaries** — trajectory finalization now appends one
  JSONL row to `runs.jsonl` instead of reading and atomically rewriting the
  full historical log for every run.
- **Bounded session preload** — agent session hydration now passes a default
  preload limit of 200 prior messages to session backends, with
  `session_preload_limit=None` available when callers explicitly want the
  full persisted history.
- **Cross-package efficiency parity** — the TypeScript sibling now caps large
  in-process diffs and single-file grep matches, and its session preload uses
  the same bounded default.

### v6.6.1 — Approval, proxy, ACP, and release hardening (April 2026)

Patch/security release for the v6.6 line. Test totals after this release:
**Python 769 passed, 3 skipped**; **TypeScript 489 passed, 4 skipped**;
mypy clean, `tsc --noEmit` clean.

- **Parallel tool approvals** — batched/native tool execution now checks
  `RunContext` approval state before dispatch, so sticky denials and pending
  approvals cannot be bypassed by a multi-tool response.
- **Credential proxy SDK mode** — the sandbox credential proxy now forwards
  provider SDK path requests such as `/v1/models`, restricts upstream origins,
  and refuses redirects that would leak injected credentials across origins or
  protocol downgrades.
- **Lazy tool schema parity** — factory-published schemas now match the
  implementation arguments for `edit_file`, `grep`, and `tree`
  (`target` / `replacement`, `glob_filter`, `max_depth`).
- **ACP default runner parity** — `AcpServer.serve(create_claw_agent(...))`
  now accepts real ClawAgents instances via `invoke()` and normalizes
  `AgentState.result` into protocol messages.
- **Hermetic runner override** — `CLAW_TEST_WORKERS` is preserved before the
  runner scrubs credentials and other `CLAW_*` variables.

### v6.6.0 — Hermes-parity feature release: browser tools, scheduler, ACP, RL hooks (April 2026)

Feature release. Four big Hermes-side capabilities now ship on both
Python and TypeScript ports, each behind an optional dependency so the
core install stays slim. Test totals after this release: **Python 762
passed**, **TypeScript 478 passed**, mypy clean, `tsc --noEmit` clean.

- **🌐 Browser tools** (`clawagents.browser`) — Playwright-driven browser
  control for agents that need to read or interact with the live web.
  `BrowserSession` exposes a stable async API (`navigate`, `snapshot`,
  `click`, `type_text`, `fill_form`, `scroll`, `wait_for_selector`,
  `screenshot`, `close`) over a pluggable provider (`LocalProvider` for
  Playwright; `BrowserbaseProviderStub` / `BrowserUseProviderStub` ready
  to be filled in for cloud back-ends). `create_browser_tools()` adapts
  the session into ClawAgents tools with per-action accessibility-tree
  snapshots so the model sees the page through the same axtree Hermes
  uses. Playwright is an optional peer (`pip install clawagents[browser]`);
  importing the module without it works fine — only `session.start()`
  raises `MissingPlaywrightError`. `MAX_NODES = 800`-cap on snapshots,
  navigation allow-/deny-lists, and a `renderSnapshot()` helper for
  prompt-friendly trees.
- **⏰ Cron / scheduled jobs** (`clawagents.cron`) — minimal but
  production-shaped scheduler for agent-driven cron, one-shots, and
  intervals. `parse_schedule()` handles `every 30s`, `at 2026-04-23T18:00`,
  and 5-field cron expressions; cron support uses the optional
  `croniter` package and degrades cleanly when missing. `Scheduler`
  provides `create_job` / `get_job` / `pause_job` / `resume_job` /
  `trigger_job` / `remove_job` plus a `run_due` driver that emits
  `JobNotifier` events (`job_started`, `job_finished`, `job_failed`,
  `job_skipped`). Job store is plain JSON on disk; runners can be any
  callable, so users can wire it to `agent.invoke(...)` or shell.
  Mirrors Hermes' "agents as a workflow engine" pattern.
- **🔌 ACP adapter** (`clawagents.acp`) — bridges any ClawAgents agent
  to **Zed's Agent Client Protocol** over stdio so editors / IDEs that
  speak ACP can drive a ClawAgents agent the same way they drive
  Claude Code or Codex. `AcpServer.serve()` registers an
  `AgentSessionFactory`, accepts ACP `initialize` / `newSession` /
  `prompt` / `cancel` requests, and translates ClawAgents stream events
  into ACP `session/update` messages (`agent_message_chunk`,
  `agent_thought_chunk`, `tool_call.start` / `.complete`, `permission`).
  Per-session `AgentSession` wraps prompt history, permission
  callbacks, and `StopReason` propagation. The optional
  `agent-client-protocol` package is loaded lazily — importing
  `clawagents.acp` works without it; only `serve()` raises
  `MissingAcpDependencyError`. Round-trip tested against Hermes'
  reference message shape.
- **🎯 RL fine-tuning hooks** (`clawagents.rl`) — capture live agent
  runs as training-ready trajectories and export them to **TRL**,
  **Atropos**, **SLIME**, or generic JSONL. `RLRecorder` plugs into
  `agent.on_event` and assembles a `Trajectory` (system / user /
  assistant + `tool_calls` / tool messages) in correct ChatML order,
  with config knobs for `max_tool_result_chars`, `redact_tool_args`,
  and `capture_system_prompt`. Pluggable `RewardScorer`s (`Contains`,
  `ExactMatch`, `Regex`, `LengthPenalty`, `Composite`) attach a scalar
  reward + per-component breakdown. Export helpers: `export_jsonl`,
  `to_chatml`, `to_trl_sft`, `to_trl_dpo`, `to_atropos_rollout`. Lazy
  `TrlAdapter` and `AtroposAdapter` only import `trl` / `atropos` when
  the user actually drives a trainer or rollout collector — install
  hints surface as `MissingRLDependencyError`.

**Backwards compatibility:** All four features are additive and
opt-in. Importing the new submodules has no side effects; nothing in
the core `create_claw_agent()` / `agent.invoke()` path changed. The
optional peers (`playwright`, `croniter`, `agent-client-protocol`,
`trl`, `atropos`) are only required at the moment you actually
`session.start()` / parse a cron expression / `serve()` over ACP /
build a TRL dataset.

### v6.5.0 — Hermes-inspired hardening: depth, isolation, heartbeats, path-scoped parallelism (April 2026)

Architecture/correctness release. Ten patterns ported from the Hermes agent are
now live on **both** Python and TypeScript ports — every change comes with
regression tests on both. Test totals after this release: **Python 662 passed**,
**TypeScript 370 passed**, mypy clean, `tsc --noEmit` clean.

**Tier 1 — runtime safety & isolation:**

- **🪜 Subagent depth limits** (`graph/coordinator`, `tools/subagent`, `graph/forked_agent`) — `RunContext` now tracks `subagent_depth`. The `task` tool refuses to delegate when the parent is already at `depth >= 2`, returning a structured error instead of silently spawning a third tier. Forks inherit the depth counter; the cap mirrors Hermes' "no recursive delegation" rule and prevents exponential subagent fan-out.
- **🧠 Memory-isolated forks/subagents** (`graph/forked_agent`, `memory/loader`) — both `forked_agent` and the built-in `task` tool now accept `skip_memory=True` (default for forks). When set, memory loaders are bypassed so a sandboxed fork cannot see the parent's `AGENTS.md`/skills/notes — closing a previously-silent context-leak path. Forks also get their own `IterationBudget` so a runaway research fork cannot starve the parent's remaining turns.
- **💓 Activity heartbeats** (`session/heartbeat`, `gateway/server`, `graph/agent_loop`) — long-running tool calls now emit periodic `tool_heartbeat` events (`tool_name`, `call_id`, `elapsed_s`) every ~20s through `run_with_heartbeat`. Gateway clients can use these to keep WebSocket channels alive and surface progress, eliminating false timeouts on slow shell/web/sandbox calls. Best-effort: emitter exceptions are swallowed so they never mask the real result.
- **⏱️ Per-agent IterationBudget** (`iteration_budget`, `graph/agent_loop`, `graph/forked_agent`) — replaces the implicit `max_turns` counter with an explicit `IterationBudget` object that lives on `RunContext`. Subagents and forks each get their own budget sized from `delegation.max_iterations` (default `DEFAULT_DELEGATION_MAX_ITERATIONS`), so one chatty fork can't drain the parent's turn pool. Surfaces the same `consume()`/`refund()`/`exhausted` shape Hermes uses, making it easy to tee budgets across recursive delegation.
- **🌿 Path-scoped parallel tool execution** (`tools/registry`) — `execute_tools_parallel` no longer fans out blindly. Tools are tagged `parallel_safe` (read-only by default for `read_file` / `list_dir` / `glob` / `search_files` / `grep` / `web_fetch`) with optional `path_scoped_arg` ("path", "url", …); the registry partitions calls into ordered batches so reads run concurrently while any writer or path-scope collision serialises behind them. Capped at `MAX_PARALLEL_TOOL_WORKERS = 8` to keep file-handle pressure bounded. Mirrors Hermes' parallel-read / serial-write contract.

**Tier 2 — extensibility & cache-discipline:**

- **🔌 Plugin hook expansion** (`plugins`) — new top-level `Plugin` + `PluginManager` (`from clawagents import Plugin, PluginManager`). Plugins compose three hook families with priority-based ordering: `pre_tool` (first-deny veto / args-rewrite, alias `before_tool`), `transform_tool_result` (sequential post-execution rewrite, alias `after_tool`), and `before_llm` (prompt-massage). Replaces the previous "single hook wins" model with a deterministic chain that's easy to unit-test.
- **📁 `display_clawagents_home()`** (`paths`) — runtime helper that resolves the package install root and rewrites it to a placeholder (`<clawagents-home>`) for tool descriptions, error messages, and traces. Makes prompt cache hits stable across user homes / dev / CI by stripping absolute paths from anything that ends up in the LLM context window.
- **🧊 Prompt-cache-aware `CommandDef`** (`commands`) — slash-command definitions now carry an explicit `cache_impact` (`"none" | "soft_break" | "hard_break"`) and parse a `--now` flag (`/skills install foo --now`) so users can opt into immediate state mutation; default is `cache_impact="none"`, `--now` upgrades to `"hard_break"` and forces a fresh prompt build. Mirrors Hermes' "deferred by default to preserve prompt cache" contract.
- **📜 Prompt-cache policy** (`AGENTS.md`) — new top-level rule documents the cache invariants (stable system prompt prefix, no per-turn timestamps in cached blocks, deferred slash-command state mutations, `display_clawagents_home()` for paths) so contributors keep the cache hit rate above the 80%+ Hermes target.

**Tier 3 — testing infrastructure:**

- **🧪 Hermetic test runner + pinned xdist** (`scripts/run_tests.sh`, `pyproject.toml`) — canonical CI-mirrored runner that pins `pytest-xdist` to 4 workers (override via `CLAW_TEST_WORKERS`), forces `TZ=UTC` / `LANG=C.UTF-8` / `PYTHONHASHSEED=0`, and scrubs credentials plus non-runner `CLAW_*` env vars before pytest sees them. Gives every contributor the exact environment CI runs in, eliminating local-vs-CI flakes. Mirrored by `clawagents/scripts/run_tests.sh` for the TypeScript port (`node:test --test-concurrency=4` via `tsx`).

**Backwards compatibility:** All 10 features are additive. Existing
`create_claw_agent()` / `agent.invoke()` call sites keep working; the new
machinery activates automatically (depth tracking, heartbeats, parallel-safe
tagging) or via opt-in (`Plugin`, `--now`, `skip_memory`, `IterationBudget`).

### v6.4.1 — Public-API export polish (no behavior change)

Patch release. Surfaces `PromptHook` and `PromptHookVerdict` at the top-level
`clawagents` package (Python) and `clawagents` module (TypeScript) so users
can `from clawagents import PromptHook` instead of reaching into
`clawagents.hooks.prompt_hook`. No code-path changes; both ports remain at
516/226 passing.

### v6.4.0 — Tracing, MCP, Handoffs, Plan Mode (April 2026)

Big feature release. Nine new subsystems shipped on **both** Python and TypeScript ports — every change comes with regression tests on both. Test totals: **Python 516 passed**, **TypeScript 226 passed**, mypy clean, `tsc --noEmit` clean.

**Tier 1 — production interop & safety:**

- **🔭 Tracing infrastructure** (`clawagents.tracing`) — hierarchical Span model with 8 kinds (`agent` / `turn` / `generation` / `tool` / `handoff` / `guardrail` / `subagent` / `custom`), pluggable `TracingProcessor` + `TracingExporter` ABCs, batched `BatchTraceProcessor` with background flush, ready-made `JsonlSpanExporter` / `ConsoleSpanExporter` / `NoopSpanExporter`, and `agent_span` / `turn_span` / `generation_span` / `tool_span` / `handoff_span` context managers. Spans propagate via Python `contextvars` (TS: `AsyncLocalStorage`). Replaces flat trajectory JSONL — drop in OTLP/Langfuse/Logfire by writing one exporter.
- **🔌 MCP (Model Context Protocol) integration** (`clawagents.mcp`) — full client supporting **stdio**, **SSE**, and **Streamable-HTTP** transports. `MCPServerStdio` / `MCPServerSse` / `MCPServerStreamableHttp` follow openai-agents-python's shape; `MCPServerManager` lifecycles a list of servers; `MCPBridgedTool` adapts MCP tools into `ToolRegistry` so they coexist with native tools, hooks, and approval flows. SDK is an optional dep (`pip install clawagents[mcp]` / `npm install @modelcontextprotocol/sdk`). 11 lifecycle phases tracked per server with tracing spans.
- **🔁 Handoffs + `Agent.as_tool()`** — fills the previously-stub `on_handoff` lifecycle hook. `Handoff` dataclass + `handoff()` builder lets one agent transfer control to another (with optional `input_filter` for history trimming). `agent.as_tool(tool_name=…, tool_description=…)` is the complementary primitive: expose any agent as a callable tool to a parent agent. Built-in `remove_all_tools` filter strips tool calls/results before handoff. New `HandoffOccurredEvent` typed stream event.
- **🛡️ Exec safety v2** (`clawagents.permissions`, `clawagents.tools.{plan_mode,bash_validator,exec_obfuscation}`) — three security upgrades shipped together: (1) `PermissionMode` enum (`DEFAULT|PLAN|ACCEPT_EDITS|BYPASS`) on `RunContext` plus `enter_plan_mode` / `exit_plan_mode` built-in tools — write-class tools refuse in `PLAN`. (2) Bash semantic validator classifies every command (`READ_ONLY|WRITE|DESTRUCTIVE|NETWORK|PROCESS|PACKAGE|SYSTEM_ADMIN|UNKNOWN`) with a 47-row corpus and decision (`ALLOW|WARN|BLOCK`). (3) Command obfuscation detector catches base64/hex/printf decode-then-exec, `<(curl …)`, `curl … | sh`, `eval` decoders, and 9 other patterns — with an allowlist for known-safe installers (rustup, brew, nvm, …).
- **🪝 Hook event taxonomy expansion + `PromptHook`** — extended `RunHooks` with 8 additive events: `on_pre_compact`, `on_post_compact`, `on_subagent_start`, `on_subagent_end`, `on_user_prompt_submit`, `on_session_start`, `on_session_end`, `on_tool_failure`. New `PromptHook(prompt, model)` evaluates a guardrail using a small/cheap model with strict-JSON `{"ok":bool, "reason":str}` verdict — write a natural-language guardrail in `settings.json` instead of Python code. Fails open on timeout/error so a noisy hook can't deadlock the agent.

**Tier 2 — ergonomics & correctness:**

- **❓ AskUserQuestion structured tool** (`clawagents.tools.ask_user_question`) — structured HITL primitive: 1-3 multi-choice questions per call, 2-4 options each, implicit `"Other (please specify)"` always appended. Renders cleanly to Telegram inline buttons / WhatsApp quick-replies. Delegates rendering via `on_ask` callback.
- **⚙️ Settings hierarchy** (`clawagents.settings`) — `user → project → local → flag → policy` precedence, deep-merged. Policy layer (`/etc/clawagents/policy-settings.json`) ALWAYS wins, so even runtime flags can't override an MDM-style enforced rule. Repo root walks up looking for `.git`/`pyproject.toml`/`package.json`. `get_setting("hooks.before_tool")` for dotted-path access.
- **🖼️ Image sanitization** (`clawagents.media.images`) — clamps tool-result base64 image blocks to ≤1200px / ≤5MB before transcript ingest, walking quality steps `(90, 75, 60)` until under limit. Closes a silent-failure path on Anthropic's 5MB limit. Pillow is **optional** (`pip install clawagents[media]`).

**Tier 3 — testing infrastructure:**

- **🎭 Mock-provider parity harness** (`clawagents.testing.mock_provider`) — deterministic fake LLM service (`MockLLMService`) bound to `127.0.0.1:0`. Real provider clients point at it via `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` env vars. Routes via `X-Parity-Scenario:` header or `PARITY_SCENARIO: <name>` system message. Five built-in scenarios. Pure stdlib, zero new deps.

**v6.5 backlog (deferred):** Anthropic prompt-cache tracking + cache-break detection, auth-profile rotation with cooldowns, multi-provider routing prefix + LiteLLM extension, file checkpoint snapshots, cache-TTL provider eligibility map, `tool_use_behavior` / `StopAtTools`, granular lifecycle payload widening, skills hot-reload watcher, `finalize` cleanup hook, `edit_scope` allowlist in skills, multi-tier numeric verifier reward, replayable per-task archives.

### v6.3.0.post1 — Docs Re-publish (no code changes)

PEP 440 post-release. Identical code to `6.3.0`; re-published so the PyPI page
shows the corrected README (version badge, feature-matrix header, latest-release
callout). `pip install clawagents` resolves to this artifact.

### v6.3.0 — Sandbox & Security Hardening, Strict Type Checking

Security/correctness release. Eleven bugs fixed across both the Python and TypeScript ports, plus a full mypy cleanup. All tests green: **334 passed**, **mypy clean** (0 errors, exit 0).

**Security fixes:**
- **Sandbox escape via symlink (TS)** — `LocalBackend.safePath` was lexical-only (`path.resolve`), so an agent that ran `ln -s /etc evil` could read `/etc/*` through the symlink. Now uses `realpathSync` for both cwd and resolved paths so symlinks are followed before the containment check. Python was already safe via `Path.resolve()`.
- **SSRF gap (TS)** — `web_fetch`'s IPv6 link-local check only matched `fe8X`, missing `fe9X`/`feaX`/`febX`. Now matches the full `fe80::/10` range (`/^fe[89ab]/i`). Python uses `ipaddress.is_link_local`, no change needed.
- **`> /dev/null` blocked legitimate use (both)** — `BLOCKED_PATTERNS` had `"> /dev/null"` (typo for `"> /dev/sd"`), which blocked the common shell idiom `cmd > /dev/null`. Removed.
- **`rm /` regex parity (TS)** — `DANGEROUS_RE` was missing the `*` quantifier on the flag group, so `rm /` (no flags) slipped past while Python's regex blocked it. Aligned.
- **`wget http` / `curl http` parity (TS)** — added to TS `BLOCKED_PATTERNS` to match Python. Agents should use the `web_fetch` tool (with SSRF guards) for HTTP, not raw shell utilities.

**Correctness fixes:**
- **Multimodal system message crashed context shedding (Py)** — `_preflight_context_check` called `.replace()` and string-slicing on system messages without checking if `content` was a `list[dict]` (multimodal). Now guards each tier with `isinstance(content, str)` and emits a `warn` event if the system message is multimodal.
- **Arbitrary role from `pre_llm` hook (Py)** — external hooks could pass any string as `role`, blowing up Pydantic validation in `LLMMessage`. Now coerces unknown roles to `"user"` and emits a `warn`.
- **Parallel native tool-call indexing (Py)** — when `before_tool` rejected a call OR returned `updated_args`, `native_tool_call_objects` was indexed by approved-list index (off-by-one) and the identity check `tc is approved_calls[i]` failed (because `updated_args` constructs a new `ParsedToolCall`). Tool-call IDs sent back to the LLM were wrong, causing native function-calling failures. Now tracks `(orig_idx, call)` pairs through the approval loop.
- **Subagent env-mutation race (Py)** — concurrent subagent runs with `credential_proxy` enabled raced on `os.environ`. The second run captured the first's overrides as its "original" env, then stamped them back into place after the first run had already stopped its proxy. Wrapped the env-mutate / run / env-restore window in an `asyncio.Lock`. No-proxy path is unaffected.
- **`classify_error` rejected `BaseException` (Py)** — `asyncio.CancelledError` and similar inherit from `BaseException`, not `Exception`. Widened `classify_error`, `_extract_status`, and `ErrorDescriptor.original` to accept `BaseException`.
- **Gemini provider `None` parts iteration (Py)** — streaming chunks could surface `None` for `candidate.content.parts` after a `hasattr` check that says only the attribute exists. Switched to `getattr(getattr(_cand, "content", None), "parts", None)` and explicit truthiness check.

**Type checking:**
- Full mypy cleanup: 46 errors → 0. Real bugs fixed (None-iter, `AsyncOpenAI`/`AsyncAzureOpenAI` mismatch, missing telegram updater check, kwargs widening). False positives addressed by renaming reused variables, adding explicit `dict[str, Any]` annotations on union-typed locals, and `parameters: Dict[str, Dict[str, Any]]` annotations on tool implementations to satisfy the `Tool` protocol.
- Added `[tool.mypy]` block to `pyproject.toml` with `warn_unused_ignores = true` and `ignore_missing_imports = true`. Run `python -m mypy` — clean run shows `Success: no issues found in 72 source files`. Mypy now exits non-zero on errors so CI can gate on it.

**Regression coverage added:**
- `tests/test_exec_safety.py` — denylist behavior (legitimate idioms allowed, destructive patterns blocked)
- `tests/test_agent_loop_bugs.py` — multimodal shedding paths + role coercion
- `tests/test_parallel_native_indexing.py` — both rejection-skip and updated-args indexing paths
- `tests/test_subagent_env_race.py` — concurrent credential-proxy runs don't corrupt env

### v6.2.1 — Release Hardening, Redirect-Safe `web_fetch`, and Parity Smokes

Patch release focused on making the v6.2 line safer to install, test, and operate.

- **Redirect-aware SSRF protection** — `web_fetch` disables automatic redirects and manually revalidates every hop before network I/O. Public-to-private redirects to loopback, RFC1918, link-local, reserved, multicast, or cloud metadata IPs are refused by default.
- **Hermetic SSRF regression tests** — added `tests/test_web_fetch_ssrf.py` covering public-to-private redirects, redirect loops, direct private IP refusal, and legitimate public-to-public redirects.
- **Local-source pytest resolution** — `pyproject.toml` now sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so local test runs cannot accidentally import an older installed wheel from `site-packages`.
- **Cross-package parity smoke** — added `scripts/smoke_gemma4.py`, mirroring the TypeScript smoke script and printing provider, base URL, and stored model for Ollama/Gemma4, `gpt-5.4`, `gemini-3.1-pro`, and `claude-opus-4-6`.
- **Release verification** — `python -m pytest` reports **319 passed, 2 skipped**; the SSRF-specific suite reports **5 passed**.

### v6.2.0 — OpenAI-Agents Parity, Ollama/Gemma4 First-Class Routing, 63 Model Profiles

A substantial additive release. Everything is backward compatible — existing `create_claw_agent()` calls, env vars, and tool registrations work unchanged.

**1. Ten OpenAI-Agents-SDK parity surfaces** (all additive, all new modules)

| Surface | Module | What it adds |
|:---|:---|:---|
| **Run Context** | `clawagents.run_context` | `RunContext` carries per-run state, approvals, and arbitrary user data through hooks and tools. |
| **Usage Tracking** | `clawagents.usage` | `Usage` + `RequestUsage` aggregate token/latency stats across turns, providers, and sub-agents. |
| **Lifecycle Hooks** | `clawagents.lifecycle` | `RunHooks` / `AgentHooks` with typed `LLMStart/LLMEnd/ToolStart/ToolEnd/AgentStart/AgentEnd/RunStart/RunEnd/Handoff` payloads. `composite_hooks` chains multiple observers without interference. |
| **Guardrails** | `clawagents.guardrails` | `input_guardrail` / `output_guardrail` decorators, `GuardrailTripwireTriggered`, behavior modes (raise / log / filter). |
| **Stream Events** | `clawagents.stream_events` | First-class `TurnStartedEvent`, `AssistantDeltaEvent`, `ToolCallPlannedEvent`, `ApprovalRequiredEvent`, `UsageEvent`, `GuardrailTrippedEvent`, `FinalOutputEvent`, `ErrorStreamEvent`. Consumable via `on_stream_event` callback. |
| **Retry Policy** | `clawagents.retry` | `RetryPolicy` dataclass + `DEFAULT_RETRY_POLICY`. Exponential backoff with jitter, per-error-class overrides. |
| **Function Tools** | `clawagents.function_tool` | `@function_tool` decorator auto-derives JSON Schema from Python type hints. Zero boilerplate. |
| **Session Backends** | `clawagents.session` | Unified `Session` protocol with `InMemorySession`, `JsonlFileSession`, `SQLiteSession`. Drop-in persistence. |
| **Structured Outputs** | `output_type=` arg on `create_claw_agent` / `agent.invoke` | Return typed objects via Pydantic model, dataclass, `dict`, `list`, or `str`. Coerced after run completes; failures emit a `warn` stream event. |
| **Tool Approval** | `approval_handler=` arg + `ApprovalRequiredEvent` | HITL gate — async callable receives `{tool, args}` and returns `True` / `False` / a redirect dict. Integrates with `ApprovalRequiredEvent` for streaming UIs. |

**2. Ollama & Gemma 4 first-class routing**

`create_provider()` now auto-routes 24 Ollama-family prefixes to `http://localhost:11434/v1` with no config needed. Use either the bare tag (`gemma4:e4b`) or the explicit routing form (`ollama/gemma4:e4b`).

| Family | Examples | Routed to |
|:---|:---|:---|
| **Gemma 4** | `gemma4`, `gemma4:e2b`, `gemma4:e4b`, `gemma4:26b`, `gemma4:31b` | Ollama @ :11434/v1 |
| **Gemma 3 / 3n / 2** | `gemma3`, `gemma3n:e4b`, `gemma2`, `gemma` | Ollama @ :11434/v1 |
| **Llama / Qwen / Mistral / Phi / Deepseek / Codellama** | `llama3`, `qwen2`, `mistral`, `mixtral`, `phi4`, `deepseek-r1`, `codellama`, … | Ollama @ :11434/v1 |
| **Explicit routing** | `ollama/<any-tag>` | Ollama @ :11434/v1 (prefix stripped) |

Override with `OPENAI_BASE_URL` if you run Ollama on a different host/port. API key is auto-set to the placeholder `"ollama"`.

**3. 63 model profiles + model-aware context budget**

The `_MODEL_PROFILES` table now covers frontier (GPT-5.4 → 400K, Gemini 3.1 → 1M, Claude 4.6 Opus), Ollama (Gemma4 e2b/e4b → 128K, 26b/31b → 256K), and a long tail of OSS variants. `_resolve_context_budget()` walks insertion order for deterministic prefix matching (most-specific first).

**4. Cross-package parity** — the TypeScript sibling `clawagents` (see [x1jiang/clawagents](https://github.com/x1jiang/clawagents)) has the identical 24-entry Ollama prefix list, 63-entry model profile table with the same (window, ratio) values, and the same `create_provider` routing logic. Parity can be exercised manually with the matching smoke scripts in each repo (`clawagents_py/scripts/smoke_gemma4.py` and `clawagents/scripts/smoke-gemma4.ts`); both print the same provider, base URL and stored model for `gemma4:*`, `ollama/...`, `gpt-5.4`, `gemini-3.1-pro` and `claude-opus-4-6`. The GitHub Actions workflow added in v6.2.1 runs `pytest`, `python -m build`, and `twine check` on every push.

**5. Quality / debug pass**

- Async agent loop hardening — new turn-started events, tighter cancellation semantics, cleaner state hand-off to sub-agents.
- Added `tests/test_openai_agents_surfaces.py` — full coverage for RunContext, Usage, Hooks, Guardrails, StreamEvents, Retry, FunctionTool, Session backends.
- Test suite: **314 passed, 2 skipped**.

**New public exports** (from `clawagents`):
`RunContext`, `ApprovalRecord`, `Usage`, `RequestUsage`, `RunHooks`, `AgentHooks`, `composite_hooks`, `InputGuardrail`, `OutputGuardrail`, `input_guardrail`, `output_guardrail`, `GuardrailBehavior`, `GuardrailResult`, `GuardrailTripwireTriggered`, `StreamEvent` (+ 10 concrete event types), `stream_event_from_kind`, `RetryPolicy`, `DEFAULT_RETRY_POLICY`, `function_tool`, `InMemorySession`, `JsonlFileSession`, `SQLiteSession`.

### v6.1.1 — Credential Isolation & Lazy Tool Provisioning

| Feature | Description |
|:---|:---|
| **Credential Isolation** | `execute` tool strips sensitive env vars (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) from subprocess environment. Claude-generated code can no longer read API keys via `env` or `os.environ`. |
| **Lazy Tool Provisioning** | Sandbox-backed tools (filesystem, exec, advanced-fs, web) defer module import to first `execute()` call. Schema is available immediately for the LLM. Reduces startup overhead. |

### v6.1.0 — Advisor Model: Smart Model Guides Cheap Model

Pair a stronger "advisor" model with a cheaper "executor" model. The executor runs every turn; the advisor is consulted 2-3 times per task for strategic guidance. Cross-provider supported — any model can advise any other model.

| Feature | Description |
|:---|:---|
| **Advisor Model** | New `advisor_model` config field. Set it and the agent gets smarter. Don't set it, nothing changes. Fully backward compatible. |
| **Three Trigger Points** | (1) After initial orientation, before planning. (2) When stuck (consecutive failures). (3) Before declaring done. |
| **Cross-Provider** | Mix providers freely: `gpt-5.4-nano` executor + `claude-opus-4-6` advisor, or any combination. |
| **CLI Flag** | `--advisor MODEL` flag for one-line usage. |
| **Env Config** | `ADVISOR_MODEL`, `ADVISOR_API_KEY`, `ADVISOR_MAX_CALLS` env vars. |

```python
agent = create_claw_agent(
    "gpt-5.4-nano",
    advisor_model="gpt-5.4",
)
```

### v6.0.0 — Production Hardening: 17 Improvements

**High Priority**

| Feature | Description |
|:---|:---|
| **Native Tool Call Patching (H1)** | `_patch_dangling_tool_calls` now handles native function calling (`tool_calls_meta`), not just text-mode JSON. Injects synthetic cancelled responses for orphaned tool_call IDs. Prevents 400 API errors in HITL scenarios. |
| **Three-Tier Provider Fallback (H2)** | New `FallbackProvider` wraps any LLM with `primary → named fallback → global fallback` chain. Quarantines providers after consecutive failures, periodic health-check restores. Config via `fallback_models` param or `CLAWAGENTS_FALLBACK_MODELS` env var. |
| **Credential Proxy (H3)** | New `CredentialProxy` — local HTTP proxy that injects API keys into outbound requests so sandboxed sub-agents never see raw credentials. Opt-in via `CLAW_FEATURE_CREDENTIAL_PROXY=1`. |
| **Rich Hook Result Model (H4)** | `BeforeToolHook` now accepts `HookResult` return (backward-compatible with bool). Hooks can block with reason, redirect args, inject messages. New `HookResult` dataclass exported from public API. |
| **Fraction-Based Summarization (H5)** | Soft-trim threshold now derives from per-model `budget_ratio` instead of hardcoded 0.60. GPT=0.60, Gemini=0.675, Claude=0.6375. Auto-adapts to any model's context window. |
| **Lazy Static Tool Registry (H7)** | New `LazyTool` class + `ToolRegistry.register_lazy()`. Tools are imported only on first `execute()` call. Fast startup with large tool sets. |

**Medium Priority**

| Feature | Description |
|:---|:---|
| **Subagent State Isolation (M1)** | `EXCLUDED_STATE_KEYS` prevents parent state (messages, todos, trajectory, lessons, session) from leaking into child sub-agents. |
| **SKILL.md Constraint Documents (M4)** | Skills now support `forbidden-actions`, `workspace-layout`, `success-criteria`, `workflow-steps` in YAML frontmatter. Structured constraints for sandboxed code execution. |
| **Pre-Compact Transcript Archival (M5)** | Before context compaction, full transcript is archived to `.clawagents/transcripts/`. Opt-in via `CLAW_FEATURE_TRANSCRIPT_ARCHIVAL=1`. |
| **Atomic File Writes (M7)** | Trajectory recorder and session persistence now use temp-then-rename pattern via `atomic_write_text()`. Prevents corruption on crash. |
| **Barrier-Based Scheduling (M8)** | Command queue now supports barrier entries. Destructive ops wait for active tasks to complete before executing. |
| **Session Heartbeat (M9)** | New `SessionHeartbeat` class auto-releases stale sessions after timeout. Resource management for multi-user deployments. |
| **Cross-Provider Test Suite (M10)** | 14 conformance tests (7 per backend) ensuring `LocalBackend` and `InMemoryBackend` both satisfy the `SandboxBackend` protocol. |

**New files:** `providers/fallback.py`, `sandbox/credential_proxy.py`, `utils/atomic_write.py`, `session/heartbeat.py`, `tests/test_cross_provider.py`

**New feature flags:** `transcript_archival` (off), `credential_proxy` (off)

**New exports:** `HookResult`, `FallbackProvider`, `CredentialProxy`, `SessionHeartbeat`, `LazyTool`, `atomic_write_text`, `atomic_write_bytes`

### v5.28.0 — Error Taxonomy, Prompt Caching, Session Persistence & External Hooks

Four production-grade features ported from the [claw-code-main](https://github.com/anthropics/claw-code) Rust reference implementation:

| Feature | Description |
|:---|:---|
| **Prompt Cache Boundary** | Inserts `__CACHE_BOUNDARY__` marker in system prompt. Anthropic provider splits into static (cached via `cache_control: ephemeral`) + dynamic blocks. Reduces input token costs on multi-turn sessions. ON by default. |
| **Error Taxonomy & Recovery** | Classifies all LLM/tool errors into 7 discrete classes (`context_window`, `provider_auth`, `provider_rate_limit`, `provider_retry_exhausted`, `provider_internal`, `provider_transport`, `runtime_io`). Each class has `retryable`, `recovery_hint`, and optional `failover_model`. Structured error events emitted via `onEvent`. ON by default. |
| **Session Persistence** | Saves agent sessions as append-only JSONL to `.clawagents/sessions/`. Events: `system_prompt`, `turn_started`, `assistant_message`, `tool_result`, `usage`, `turn_completed`. New CLI: `--sessions` (list) and `--resume [ID\|latest]` (continue). Opt-in. |
| **External Hook System** | Shell commands that run before/after tool execution and LLM calls. Config via `.clawagents/hooks.json` or `CLAW_HOOK_*` env vars. Hooks receive JSON on stdin, return JSON on stdout. `pre_tool_use` can block or modify args. 10s timeout, fail-open. Opt-in. |

Also:
- **Anthropic cache token extraction** — `cache_creation_tokens` and `cache_read_tokens` now populated from both streaming and non-streaming Anthropic responses.
- **`AgentState.session_file`** — New field tracks the session JSONL path when persistence is enabled.
- **New public exports** — `ErrorClass`, `ErrorDescriptor`, `classify_error`, `get_recovery_recipe`, `SessionWriter`, `SessionReader`, `list_sessions`, `HooksConfig`, `ExternalHookRunner`, `load_hooks_config`.

### v5.27.3 — Gemini Signature Regression Coverage
- **Gemini signature regression test** — Added targeted tests for `_serialize_gemini_parts` to ensure `thought_signature` is preserved on the first parallel `function_call` part (not copied onto siblings).
- **Parallel integration test reliability** — Fixed integration test fixture validation mismatch so large-output parallel execution is validated correctly.

### v5.27.2 — Gemini 3 Thought Signature Fix
- **Gemini 3 Propagation** — Propagated `thought_signature` to all parallel `function_call` parts in the response, preventing `400 INVALID_ARGUMENT` during multi-tool execution.

### v5.27.1 — Timeout Bugfix
- **Fixed NameError** — Added `timeout_s` parameter to `ClawAgent.invoke` to prevent an exception when a global timeout is not provided.

### v5.27.0 — Claude Code Architectural Patterns

Ported 10 production-grade architectural patterns from Anthropic's Claude Code directly into ClawAgents. These features are controllable via environment variables or constructor injection:

| Feature | Description |
|:---|:---|
| **Micro-Compact Memory** | Aggressively clears giant tool results to save context. |
| **File History Snapshots** | Safely backs up files to `.clawagents/snapshots/` before writing. |
| **Prompt Cache Tracking** | Real-time stats on Anthropic/OpenAI prompt cache hits. |
| **Typed Memory Taxonomy** | Auto-parses `project`, `user`, and `feedback` memories via frontmatter. |
| **Write-Ahead Logging (WAL)** | Crash-resilient interaction logging. |
| **Granular Permission Rules** | Define glob-based `Allow`/`Deny` execution policies. |
| **Background Memory Extraction** | Periodically scans conversations and extracts metadata. |
| **Orchestration** | Access to `run_forked_agent` and `run_coordinator` (swarm routing). |

### v5.26.0 — Bundled OpenViking Skill, Updated ByteRover Skill

| Feature | Description |
|:---|:---|
| **OpenViking skill** | Bundled `skills/openviking/SKILL.md` teaches the agent to use the `ov` CLI for tiered context retrieval (L0/L1/L2). Auto-enabled when `ov` is on PATH |
| **ByteRover skill updated** | Refreshed to match `byterover-cli` v1.8.0 — added `--headless`, `--folder`, removed obsolete commands |
| **Generic bundled skill loader** | Skill loader now scans the entire bundled `skills/` directory instead of hardcoding individual skills |

### v5.25.0 — Gemini Streaming Fix

| Feature | Description |
|:---|:---|
| **Fix Gemini SDK warning** | Eliminated "non-text parts in the response" warning by iterating `candidates[].content.parts[]` instead of accessing the `.text` property on streaming chunks containing function calls |
| **Consistent text extraction** | Streaming path now uses the same parts-based extraction as the non-streaming `_request_once`, filtering out thought parts |

### v5.24.0 — Zero-Config Channel Auto-Detection

| Feature | Description |
|:---|:---|
| **Auto-detect channels from env vars** | `clawagents --serve` now reads `TELEGRAM_BOT_TOKEN`, `WHATSAPP_AUTH_DIR`, `SIGNAL_ACCOUNT` from `.env` and auto-starts the ChannelRouter — zero code required |
| **`--doctor` channel status** | `clawagents --doctor` reports which messaging channels are configured |
| **`.env.example` updated** | All channel env vars documented with inline comments |
| **`--init` scaffold** | `clawagents --init` generates `.env` with channel variables pre-commented |

### v5.23.0 — WebSocket Gateway, Multi-Channel Messaging (Telegram, WhatsApp, Signal)

Full multi-platform messaging support inspired by OpenClaw's channel architecture:

| Feature | Description |
|:---|:---|
| **WebSocket gateway** | FastAPI native WebSocket endpoint at `/ws` alongside existing HTTP. Methods: `chat.send` (streaming events), `chat.history`, `chat.inject`, `ping`. Auth via `?token=` query param |
| **Channel adapter interface** | `ChannelAdapter` protocol + `ChannelMessage` dataclass — standard contract for any messaging platform |
| **Telegram adapter** | Uses [python-telegram-bot](https://python-telegram-bot.org/). Config: `{"bot_token": "..."}` |
| **WhatsApp adapter** | Baileys subprocess (Node.js) or WhatsApp Business API. Config: `{"mode": "baileys", "auth_dir": ".whatsapp-auth"}` |
| **Signal adapter** | Uses [signal-cli](https://github.com/AsamK/signal-cli) subprocess with JSON-RPC. Config: `{"account": "+1234567890"}` |
| **Channel router** | `ChannelRouter` dispatches inbound messages to agents, routes replies back. Per-session serialization via `KeyedAsyncQueue`, optional debouncer, hooks |

```python
from clawagents import create_claw_agent, ChannelRouter
from clawagents.channels.telegram import TelegramAdapter
from clawagents.channels.whatsapp import WhatsAppAdapter

router = ChannelRouter(lambda: create_claw_agent("gpt-5-mini"))
router.register(TelegramAdapter())
router.register(WhatsAppAdapter())
await router.start_all({
    "telegram": {"bot_token": "123456:ABC..."},
    "whatsapp": {"mode": "baileys", "auth_dir": ".whatsapp-auth"},
})
```

### v5.22.0 — Tool Result Caching, Parameter Validation & ComposeTool

3 features inspired by ToolUniverse's tool management patterns:

| Feature | Description |
|:---|:---|
| **Tool result caching** | LRU in-memory cache (`ResultCacheManager`) avoids redundant tool calls. Tools opt in with `cacheable = True`. Per-tool TTL overrides via `result_cache.set_tool_ttl()`. Built-in cacheable tools: `read_file`, `grep`, `web_fetch`. Default: 256 entries, 60s TTL |
| **Parameter validation + coercion** | `validate_tool_args()` checks required params and type-matches before execution. Lenient coercion handles common LLM quirks: `"42"` → `42`, `"true"` → `True`, JSON strings → objects/arrays. Enabled by default on `ToolRegistry` |
| **ComposeTool** | `create_compose_tool()` chains multiple tools in a deterministic pipeline without an LLM in the loop. Lighter than sub-agents for predictable workflows. Steps receive previous results and a `call_tool` helper. Failures short-circuit with clear error messages |

### v5.21.0 — Context Engine, Loop Detection & Compaction Overhaul

8 improvements inspired by the latest OpenClaw architecture:

| Feature | Description |
|:---|:---|
| **Chunked compaction with retry** | Compaction now splits old messages into ~30K-token chunks, summarizes each separately with up to 3 retries (exponential backoff), and explicitly preserves file paths, function names, error messages, and commands verbatim |
| **Better loop detection** | Result hashing detects "different args, same result" stalls; ping-pong detection catches A→B→A→B oscillation; global circuit breaker hard-stops at 30 no-progress calls |
| **Context pruning (soft-trim)** | New `_soft_trim_messages` runs at 60% context usage (before the 75% compaction trigger). Trims old tool results >1000 chars, removes duplicates, and stubs stale image data |
| **Skill eligibility gating** | Skills can declare `requires:` in YAML frontmatter (`os`, `bins`, `env`). Ineligible skills are filtered at load time |
| **Skill prompt budget** | Max 20 skills / 4000 chars injected into the system prompt. Full list accessible via `list_skills` |
| **Control token sanitization** | Strips leaked model control tokens (`<\|assistant\|>`, `<\|endoftext\|>`, full-width variants) from final output |
| **Head+tail truncation** | Eviction fallback and content preview now use head+tail (preserving error messages at the end). Also fixes a bug where few-line, huge-character content bypassed preview truncation |
| **Pluggable context engine** | New `ContextEngine` ABC with `after_turn`, `compact`, `bootstrap`, `cleanup` lifecycle hooks. `DefaultContextEngine` is a no-op pass-through. Registry: `register_context_engine()` / `resolve_context_engine()` |

### v5.20.4 — Gemini MALFORMED_FUNCTION_CALL Retry

| Feature | Description |
|:---|:---|
| **Gemini malformed FC retry** | When Gemini returns `finish_reason=MALFORMED_FUNCTION_CALL` with 0 parts (common with complex parallel tool calls), the provider now automatically retries with `tool_config.mode=ANY` instead of stopping the agent |
| **Streaming + non-streaming** | Fix applied to both streaming (`_stream_with_retry`) and non-streaming (`_request_once`) code paths |
| **Recursion guard** | `_malformed_retry` flag prevents infinite retry loops if mode=ANY also fails |

### v5.20.3 — GPT-5 Temperature Corrections

| Feature | Description |
|:---|:---|
| **GPT-5-nano temperature** | Live API tests confirmed `gpt-5-nano` requires `temperature=1` (not 0). Fixed in `_FIXED_TEMPERATURE_MODELS` |

### v5.20.0 — Temperature & Compaction Fixes

| Feature | Description |
|:---|:---|
| **Temperature fix** | GPT-5 models no longer forced to `temperature=1.0`. Only o-series models (o1, o3, o4-mini) retain the fixed override. This restores deterministic behavior when `TEMPERATURE=0` is set |
| **Compaction overhaul** | Context compaction no longer causes the agent to "forget" what it was doing. Five improvements: (1) `RECENT_MESSAGES_TO_KEEP` increased from 6 → 20, (2) tool call/result pairs are never split, (3) summary prompt now includes original task + structured preservation instructions, (4) compacted summary inserted as `role="user"` with `[System — Compacted History]` prefix instead of `role="assistant"`, (5) text log for summarization includes structured `[TOOL CALLS]` and `[TOOL RESULT]` markers |
| **Debug cleanup** | All development instrumentation removed from production code |

### v5.19.0 — Anthropic Provider, Security, Architecture Overhaul

| Feature | Description |
|:---|:---|
| **Anthropic/Claude provider** | First-class support for Claude models via `ANTHROPIC_API_KEY`. Install with `pip install clawagents[anthropic]` |
| **Optional Gemini** | `google-genai` is now an optional dependency. Install with `pip install clawagents[gemini]` or `pip install clawagents[all]` |
| **`py.typed` + `__version__`** | PEP 561 type stub marker and `clawagents.__version__` export for downstream tools |
| **Lazy config loading** | No more module-level side effects — `.env` discovery happens on first `load_config()` call |
| **Lazy `Path.cwd()`** | All module-level `Path.cwd()` calls replaced with lazy functions — safe for import from any directory |
| **Gateway authentication** | `GATEWAY_API_KEY` env var enables Bearer token auth on POST endpoints |
| **CORS support** | Gateway now supports `GATEWAY_CORS_ORIGINS` for cross-origin requests |
| **Improved blocked patterns** | Expanded dangerous command detection with regex matching |
| **API key masking** | `clawagents --doctor` now masks keys (shows `********...last4`) |
| **Azure detection** | New `OPENAI_API_TYPE=azure` env var for explicit Azure OpenAI configuration |
| **Global timeout** | `--timeout N` CLI flag and `CLAW_TIMEOUT` env var for agent run time limits |
| **`--verbose` / `--quiet`** | CLI flags for controlling output verbosity |
| **`--prune-trajectories N`** | Delete trajectory files older than N days |
| **Lesson export/import** | `export_lessons()` / `import_lessons()` for sharing lessons between projects |
| **Trajectory pruning** | `prune_trajectories(max_age_days)` utility function |
| **`pydantic-settings`** | Now properly listed as a dependency (was missing) |
| **pyproject.toml metadata** | Added license, authors, classifiers, URLs, optional dependency groups |
| **New tests** | Tests for `_repair_json`, trajectory recorder, config module |

### v5.18.0 — Doctor, Trajectory Inspector & Config Improvements

| Feature | Description |
|:---|:---|
| **`clawagents --doctor`** | New diagnostic command checks `.env` discovery, API keys, active model, LLM settings, PTRL flags, local endpoint reachability, trajectory history, and `AGENTS.md` presence |
| **`clawagents --trajectory [N]`** | Inspect the last N run summaries: score, quality, failures, judge verdict, duration — human-readable trajectory output |
| **Startup banner** | Every `--task` and `--serve` now prints `provider=X model=Y env=Z ptrl=...` for instant visibility into active config |
| **`CLAWAGENTS_ENV_FILE`** | New env var to explicitly point to a `.env` file path. Priority: `CLAWAGENTS_ENV_FILE` > `cwd/.env` > `cwd/../.env`. Useful for CI, Docker, multi-project |
| **Publish hygiene** | GitHub releases no longer include `.clawagents/`, `.pytest_cache/`, logs, or other runtime artifacts |
| **Config/docs consistency tests** | 6 pytest tests verify every `EngineConfig` field appears in `.env.example` and `README.md` |
| **`--port` in TypeScript** | Gateway server port now configurable via `--port N` in TypeScript CLI |

### v5.17.0 — Quick Start Scaffold & Examples

| Feature | Description |
|:---|:---|
| **`clawagents --init`** | New CLI command scaffolds a starter project in the current directory: generates `.env` (with all providers commented out), `run_agent.py` (ready-to-run starter script with 5 provider options), and `AGENTS.md` (memory template) |
| **`clawagents --help`** | Shows usage with examples, quick start instructions |
| **`clawagents --task`** | Run a single task from the command line |
| **`clawagents --serve`** | Start the HTTP gateway server from CLI |
| **Examples directory** | 8 ready-to-run example scripts: OpenAI, Gemini, Azure, Ollama, vLLM, Bedrock, custom tools, and multi-sample comparison |
| **README overhaul** | New "30-Second Quick Start" section, examples table, clearer onboarding flow |

### v5.16.0 — LLM-as-Judge & Thinking Token Preservation

| Feature | Description |
|:---|:---|
| **G. LLM-as-Judge verification** | After each run (when `learn=True`), a separate, focused LLM call evaluates whether the task was actually accomplished. Returns a 0-3 score with justification — more reliable than heuristic scoring. Results stored as `judge_score` and `judge_justification` on `RunSummary` |
| **H. Thinking token preservation** | Models like Qwen3 and DeepSeek that emit `<think>...</think>` blocks are now fully supported. Thinking content is extracted before tool-call parsing, preserved on messages and trajectory records, and stripped from visible output. Available via `strip_thinking_tokens()` utility |

### v5.15.0 — Deterministic Verification & GRPO-Inspired Comparison

| Feature | Description |
|:---|:---|
| **A. Deterministic rewards** | Tool execution results (exit codes, test pass/fail counts) are now used as objective ground truth for scoring, replacing pure LLM self-assessment. Each turn and run summary includes `deterministic_score` and `verified_score` fields |
| **B. Multi-sample comparison** | New `agent.compare(task, n_samples=3)` method runs the same task N times and picks the best result using objective scoring — inspired by SkyRL's Group Relative Policy Optimization (GRPO) |
| **C. Task-type-aware verification** | Auto-detects task type (coding/file/search/refactor/general) and applies type-specific verifiers. Coding tasks use test results; file tasks check write success; refactoring checks edits + tests |
| **D. Progressive context caching** | System prompt token count is computed once and cached, avoiding redundant re-counting on every turn. Logged at startup for budget visibility |
| **E. RFT-ready transitions** | Each trajectory now exports `{run_id}_rft.json` with (observation, action, reward, done) tuples per step — structured for future Rejection Fine-Tuning pipelines |
| **F. Adaptive rethink threshold** | Rethink trigger threshold now adjusts dynamically: complex tasks (coding/refactor) get more patience (threshold=5), simple tasks (search/file) trigger sooner (threshold=3), and late in runs threshold drops to minimum (2) |

### v5.14.0 — SkyRL-Inspired PTRL Improvements

| Feature | Description |
|:---|:---|
| 🚦 **Quality gate for lesson extraction** | Lessons only extracted from runs with mixed outcomes (both successes and failures). Zero-variance runs (all-success or all-failure with no contrast) are skipped — inspired by SkyRL's GRPO dynamic sampling |
| ⏰ **Lesson staleness decay** | Each lesson block is now timestamped + model-tagged (`@timestamp [model]`). `load_lessons(max_age_s=N)` filters out stale lessons. Prevents prompt pollution from outdated advice |
| 🔤 **Format vs. logic failure classification** | Every failed tool call is classified as `"format"` (bad JSON, wrong params) or `"logic"` (valid call, wrong approach). Rethink messages now include format-specific or strategy-specific guidance |
| 📊 **Per-step reward attribution** | Each `TurnRecord` now includes `observation_context` (what the agent saw before deciding), `productivity_score` (-1.0 to 1.0), and `failure_type` per tool call. `RunSummary` adds `format_failures`, `logic_failures`, `has_mixed_outcomes`, and `finish_reason` |
| 🧠 **Enhanced self-analysis prompt** | Post-run LLM analysis now receives failure type breakdown and productivity scores for targeted lesson extraction |

### v5.13.0 — Prompt-Time Reinforcement Learning (PTRL)

| Feature | Description |
|:---|:---|
| 🧠 **PTRL: Post-run self-analysis** | After each run, the LLM reviews its own trajectory and extracts 2-5 actionable lessons, saved to `.clawagents/lessons.md` |
| 📖 **PTRL: Pre-run lesson injection** | On subsequent runs, stored lessons are injected into the system prompt so the agent avoids past mistakes |
| 🔄 **PTRL: Enhanced mid-run rethink** | When consecutive failures trigger a rethink, relevant past lessons are included in the rethink message |
| 🎛️ **`learn` flag / `CLAW_LEARN` env** | Opt-in via `learn=True` or `CLAW_LEARN=1`. Automatically enables trajectory logging |
| 📐 **Default `context_window` → 1,000,000** | Increased from 128,000 to support modern large-context models |
| 🔧 **macOS sandbox symlink fix** | `LocalBackend` now resolves symlinks at init (fixes `/var` → `/private/var` on macOS) |
| ✅ **All 150 tests passing** | Fixed 48 pre-existing test failures (sandbox path traversal, LLMMessage subscript, mock assertions) |

### v5.12.1 — Streamlit / Jupyter Compatibility

| Feature | Description |
|:---|:---|
| 🔧 **Signal handler fix** | `add_signal_handler` now catches `RuntimeError` in addition to `NotImplementedError`/`OSError`, fixing crashes in Streamlit, Jupyter, and other non-main-thread environments |

### v5.12.0 — Gemini 3 Thought Signature Support

| Feature | Description |
|:---|:---|
| 🧠 **`thought_signature` preservation** | Gemini 3 thinking models (e.g. `gemini-3-flash-preview`) require `thought` and `thought_signature` fields to be echoed back during multi-turn function calling. ClawAgents now captures the full response parts and replays them verbatim, preventing 400 errors. |
| 🔄 **`gemini_parts` field** | New optional field on `LLMMessage` and `LLMResponse` carries raw Gemini response parts through the conversation history. Used automatically — no user action required. |

### v5.11.0 — Configurable Limits

| Feature | Description |
|:---|:---|
| 🔢 **`max_iterations`** | Now settable at construction or via `MAX_ITERATIONS` env (default 200, was hardcoded in caller) |
| 📏 **`preview_chars`** | Tool-output preview length configurable via `CLAW_PREVIEW_CHARS` env (default 120) |
| 📄 **`response_chars`** | Response text length in trajectory records via `CLAW_RESPONSE_CHARS` env (default 500) |
| ⚙️ **Priority** | Explicit param > env var > default for all three |

### v5.10.0 — Discrete Reward Bands & Weighted Scoring

| Feature | Description |
|:---|:---|
| 🎯 **Discrete reward bands** | Run scores mapped to -1 … +3 bands (inspired by CUDA-Agent PPO reward shaping) |
| ⚖️ **Weighted execution scoring** | `execute`, `shell`, `run_code` weighted 2× higher than generic tools |
| 🏷️ **Run quality grading** | Each run classified as `clean`, `noisy`, or `failed` for trajectory filtering |
| 🛡️ **Gameable tool exclusion** | `think`, `todolist`, `use_skill`, etc. excluded from scoring to prevent reward hacking |

### v5.9.0 — Trajectory Logging & Rethink

| Feature | Description |
|:---|:---|
| 📊 **Trajectory logging** | Structured recording of every turn, tool call, and outcome to `runs.jsonl` |
| 🔄 **Consecutive-failure rethink** | After 3 consecutive meaningful failures, injects a system "rethink" prompt |
| 🎛️ **Opt-in flags** | `trajectory=True` / `CLAW_TRAJECTORY=1` and `rethink=True` / `CLAW_RETHINK=1` |

### v5.8.0 — JSON Resilience

| Feature | Description |
|:---|:---|
| 🔧 **JSON repair** | `_repair_json()` utility fixes truncated JSON from hitting `max_completion_tokens` |
| 🔁 **Truncated JSON retry** | Detects incomplete JSON tool calls and prompts the LLM to resend |

### v5.7.0 — Model-Specific Temperature

| Feature | Description |
|:---|:---|
| 🌡️ **Fixed-temperature models** | Reasoning models (o-series, gpt-5, gpt-5-mini, gpt-5-turbo) auto-override to `temperature=1.0`. Non-reasoning models (gpt-5-nano, gpt-5-micro, gpt-4o) respect configured temperature |
| 🌡️ **Configurable temperature** | `TEMPERATURE` env var + `temperature` parameter on `create_claw_agent` |

### v5.6.0 — LLM Parameter Fixes

| Feature | Description |
|:---|:---|
| 🔑 **`max_completion_tokens`** | OpenAI calls now use `max_completion_tokens` (replacing deprecated `max_tokens`) |
| 🔑 **`max_output_tokens`** | Gemini calls now pass `max_output_tokens` correctly |
| ⚙️ **Config priority** | Explicit param > `.env` > default — no more shadowing of env values |

### v5.5.0 — Foundation

| Feature | Description |
|:---|:---|
| 🔌 **Pluggable Sandbox** | `SandboxBackend` protocol with `LocalBackend` + `InMemoryBackend` |
| 🌐 **Gateway Server** | FastAPI server with SSE streaming and 4-lane queue |
| 🗂️ **Advanced FS Tools** | `tree`, `diff`, `insert_lines` |
| 🧠 **Think Tool** | Structured reasoning without side effects |
| 🌍 **Web Fetch** | URL fetching with HTML cleanup |
| 💬 **Ask User** | Interactive stdin-based input |
| 📜 **History Offloading** | Full audit trail preserved after compaction |
| 🔒 **Tool Access Control** | `block_tools()` / `allow_only_tools()` at runtime |
| 💉 **Context Injection** | `inject_context()` hook for every LLM call |
| ✂️ **Output Truncation** | `truncate_output()` to cap tool output size |

---

### Earlier headlines (from README)

#### v6.13.1

- **ATLAS fail-closed** — reflection-harvest and final-gate exceptions abort the run instead of silently releasing an answer.
- **Pinned ATLAS revision** — install docs and hints pin `atlas-skill` to commit `3a917f3e0b993e3bfd77f652b013193aed167964`.
- **Companion** — VS Code **1.0.36**.

#### v6.13.0

- **ATLAS harness** — optional `atlas=True` / `CLAW_ATLAS=1` supervision layer: runtime protocol, tool-failure / subagent checkpoints, blocking final gate, redacted `record_trace` + taxonomy learning via `atlas_runtime`.
- **Install ATLAS runtime** — `pip install 'atlas-skill @ git+https://github.com/multi-agent-systems-failure-taxonomy/ATLAS.git@3a917f3e0b993e3bfd77f652b013193aed167964'` (marker extra `clawagents[atlas]` is a no-op on PyPI).
- **Companion** — VS Code **1.0.35** Settings checkbox for ATLAS.

#### v6.12.13

- **Skill retrieval** — high-recall intent coverage (aliases / triggers / anti-triggers / morphology) instead of fixed token-saving cutoffs; short follow-ups inherit prior intent.
- **Paged `use_skill`** — contiguous, content-hash-bound pages; no data-plane tools until every page is read.
- **Composed `allowed-tools`** — intersecting boundaries only (cannot widen authority by loading another skill).
- **Workshop hardening** — path/content validation on writes.
- **Companion** — VS Code **1.0.32**; TypeScript **6.12.13**.

#### v6.12.12

- **`invoke(images=…)`** — attach vision images to the first user message (OpenAI `image_url` canonical; Anthropic / Responses / Bedrock / Gemini conversions).
- **`invoke(files=…)`** — attach PDF (native `file` / `document` blocks) or DOCX (text-extracted) to the first user message.
- **Companion** — VS Code **1.0.31** (image + PDF/DOCX attach UI); TypeScript **6.12.12**.

#### v6.12.11

- **Docs** — README aligned with the current skill loader, ACP schema fixes, and VS Code **1.0.30** companion.

#### v6.12.10

- **`disable-model-invocation`** — skills with `disable-model-invocation: true` stay out of the model catalog and refuse `use_skill` (user-invocation only).
- **ACP tool_call schema** — required `title` / `kind`; tool content wrapped for spec-strict clients.

#### v6.12.9

- **Skill loader** — bundled dirs load first (user/workspace override); safer `requires` parsing; ineligibility reasons; resource disclosure in `use_skill`; size caps and frontmatter fallbacks.

#### v6.12.8

- **Per-turn skill ranking** — catalog re-ordered against the latest user message with a Recommended section.
- **Stronger `use_skill`** — fuzzy / case-insensitive names; better YAML descriptions; injection upsert (no duplicate catalogs).

#### v6.12.7

- **Dynamic skill-catalog budget** — ~1.5% of context (floor 4k / ceiling 16k chars); description-first truncation; `CLAW_SKILL_LISTING_*` overrides.

#### v6.12.6

- **`list_skills` registered** alongside `use_skill` for overflow catalog discovery.

#### v6.12.5 / v6.12.4

- **`CLAWAGENTS_SKIP_DOTENV`** — long-lived hosts (VS Code sidecar) skip mid-process `.env` reloads.
- **Host API keys** — workspace `.env` no longer clobbers SecretStorage / spawn-injected keys (`CLAWAGENTS_DOTENV_OVERRIDE=0`).

#### v6.12.3

- **`wire_api`** — `auto` | `responses` | `chat_completions` for OpenAI-compatible proxies (Codex Responses-only gateways that 404 `/chat/completions`).
- **`ssl_verify`** — disable TLS verify for private-CA corporate endpoints.
- **Auto Responses** — GPT-5.5/5.6/Codex prefer `/v1/responses` even on custom `base_url` (no longer limited to api.openai.com).
- **SSE proxies** — non-stream Responses requests collect via streaming (gateways that ignore `stream=false`).

#### v6.12.2

- **Responses API** — `OpenAIProvider` auto-selects `/v1/responses` vs Chat Completions from model + endpoint (GPT-5.5/5.6/Codex on official OpenAI; Ollama/BAG/Azure stay on Chat Completions; sticky fallback if Responses is missing).
- **Reasoning + tools** — Responses path keeps `reasoning.effort` with function tools (no more forced `none` on GPT-5.5/5.6 when Responses is available).

#### v6.12.1

- **`reasoning_effort`** on `create_claw_agent` / OpenAI provider (`none`|`low`|`medium`|`high`|`xhigh`|`max`; UI aliases Light→low, Extra High→xhigh)
- Chat Completions + tools on GPT-5.5/5.6 force `none` when falling back from Responses

#### v6.12.0

- **Native AWS Bedrock** — Claude via `AsyncAnthropicBedrock` (IAM / HIPAA path); Nova and other models via Converse API. `pip install 'clawagents[bedrock]'`.
- **Model routing** — Bedrock IDs (`us.anthropic.…`, `amazon.nova-…`, `bedrock/…`) use native providers when `base_url` is unset; OpenAI-compatible BAG/LiteLLM still works with `base_url`.
- **Profiles** — `profile="bedrock"` (native) and `profile="bedrock-gateway"` (proxy).
- **Config** — `AWS_REGION` / `AWS_PROFILE` / access keys + `BEDROCK_MODEL` / `PROVIDER=bedrock`.

#### v6.11.2

- **`web_search`** — Tavily-backed search tool (`TAVILY_API_KEY`). Ranked URLs + snippets; use `web_fetch` for full pages.

#### v6.11.1

- **CodeAct sandbox** — curated `__builtins__` + AST forbid-list so `open`/`__import__`/`eval` cannot bypass tool permissions.
- **Checkpoint refs** — reject malformed SHAs before `git reset` / `diff`.
- **Evals judge** — align `judge_run` call signature with the trajectory judge API.

#### v6.11.0

- **Shadow-git restore modes** — `files` / `conversation` / `both` with turn binding + `checkpoint_diff`.
- **Always-on rules** — `CLAUDE.md` + `.clawagents/rules/**` re-injected every LLM round.
- **Custom modes** — `.clawagents/modes.json` + builtins; CLI `--mode` / `--auto`.
- **CodeAct** — `create_claw_agent(action_mode="code")` Python-as-action loop.
- **Evals** — `python -m clawagents evals <suite.json>` + library `approval_handler`.

#### v6.10.8

- **Compaction budget** — `compress_messages_safe` re-measures after the safe tier and escalates to summarization when still over context (no more false “under budget”).
- **Message identity** — reuse original message objects when role+content survive compression so session tracking and `tool_calls_meta` / `tool_call_id` stay intact.
- **Judge usage** — LLM-as-Judge token spend flows into the run’s `Usage`.

#### v6.10.7

- **Repo map** — ranked symbol map tool (+ optional prompt inject).
- **Context ledger** — commit-boundary restorable memory with `rehydrate_ledger`.
- **Shadow-git checkpoints** — Cline-style undo without touching project git.
- **`apply_patch`** — SEARCH/REPLACE and unified-diff surgical edits.
- **Core memory / memory bank / live facts** — editable blocks + superseding facts.
- **Git + worktree tools** — status/diff/commit/undo; isolated worktrees for parallel agents.
- **Plan handoff** — `write_plan` → `.clawagents/plan.md`; harness clear-tool knobs; compaction thrash guard.

#### v6.10.6

- **Prompt-cache align** — normalize static system prefix; lessons sit *after* `__CACHE_BOUNDARY__`; tools listed alphabetically.
- **Tiered `read_file`** — `tier=L0` outline/symbols, `L1` paginated body, `L2` large/full read.
- **Compaction** — wire `compress_messages_safe` + `on_pre_compact` / `on_post_compact`; output-side trim; content budgets for tool results.
- **Crushers** — HTML / diff / pytest-junit crushers; multimodal `sanitize_tool_output` on ingest.
- **Recoverability** — micro-compact stubs keep artifact ids; `retrieve_tool_result(query=…)` searches local artifacts.
- **Failure learn** — append durable lesson bullets into workspace `AGENTS.md`.

#### v6.10.5

- **Gemini 400 recovery** — on FR/FC / thought_signature `INVALID_ARGUMENT`, retry once with tool turns flattened to plain text.
- **Signature fidelity** — do not copy `thought_signature` onto sibling parallel FCs.
- **External hook skip** — close assistant+tool pair instead of a bare `[Tool Skipped]` user turn.

#### v6.10.4

- **Gemini FC/FR ids** — use API `function_call.id` (or a stable generated id stamped into both FC and FR).
- **thought_signature** — base64 round-trip for session JSON; always prefer preserved `gemini_parts` when replaying tool turns.
- **Sanitize rewrite** — strict FC→FR pairing, orphan FR drop, spacer model only when plain user text follows FR.

#### v6.10.3

- **Gemini FR purity** — keep `function_response` turns FR-only (do not coalesce with following user text); insert a spacer model turn when needed.
- **Orphan FR drop** — remove `function_response` that does not follow a model `function_call`.
- **gemini_parts safety** — if stored parts lack `function_call` but `tool_calls_meta` exists, rebuild the model turn from meta.

#### v6.10.2

- **Gemini conversation hygiene** — merge consecutive `user`/`model` turns (parallel tool results); insert synthetic `function_response` when a call was skipped; drop leading orphan model turns. Fixes `INVALID_ARGUMENT` about function-call turn ordering.
- **Skipped-tool transcript** — `before_tool` / RunContext rejects now close the native assistant+tool pair instead of appending a bare `[Tool Skipped]` user message.

#### v6.10.1

- **Gemini / OpenAI tool schemas** — array parameters always declare `items.type` (fixes Gemini `400 INVALID_ARGUMENT` on tools).
- **GPT-5.5 / GPT-5.6 + tools** — prefer Responses API; Chat Completions fallback still sets `reasoning_effort=none`.
- **Orphan tool messages** — session preload and OpenAI formatting drop tool results without a matching `tool_calls` id (fixes provider 400 after limited history).
- **MCP loop affinity** — stdio/SSE sessions reconnect when invoked from a different event loop than registration (VS Code / threaded hosts).
- **`skills_exclude`** — `create_claw_agent(..., skills_exclude=[...])` drops named skills after load.
- **Streaming telemetry** — `assistant_delta` / intermediate `assistant_message` on the typed event stream; OpenAI prompt-cache `cached_tokens` surfaced.
- **Error taxonomy** — safer HTTP status coercion for google-genai string status enums; clearer API-key invalid classification.
- **Worker-thread signals** — agent loop tolerates uvloop/asyncio refusing signal handlers off the main thread.

#### v6.10.0

- **Session persistence** — identity-based message tracking survives compaction and dangling-tool-call patching without losing or duplicating persisted turns.
- **Parallel tool policy hooks** — external pre/post hooks and session writes apply in parallel batches; policy gates cannot be bypassed by batching a forbidden call with a safe one.
- **History offload redaction** — compacted history files run through the same secret redaction as every other persistence surface.
- **Provider reliability** — normalized `prompt_tokens` / `tokens_used` across OpenAI, Gemini, and Anthropic; mid-stream retry preserves accumulated tool calls; truncated JSON string recovery; Anthropic parallel `tool_result` coalescing and array `items` in tool schemas.
- **Context & loop fixes** — accurate multimodal token counts; safe compaction split boundaries; deduped prompt injection; micro-compact gated on context usage; overflow recovery shrinks effective window; advisor duplicate-message and handoff transcript fixes; one iteration increment per loop round.
- **Infrastructure** — command-queue barrier exclusivity and strong task refs; heartbeat cancels in-flight work on agent cancel; bounded gateway WebSocket session store; ACP tool-id matching for out-of-order completions; explicit PIL decompression-bomb guard.

#### v6.9.2

- **Bash validator hardening** — peels launcher wrappers (`env`, `sudo`, `timeout`, `eval`, …) so inner destructive commands can't bypass BLOCK rules; normalizes root-like paths; handles alias bypass (`\rm`).
- **Gateway CORS safe defaults** — loopback-only origins by default instead of `*`; disables credentials when wildcard is explicitly configured.
- **Plan-mode escape fix** — agent-as-tool subagents now inherit the parent's permission mode.
- **Provider parity** — strips `__CACHE_BOUNDARY__` from OpenAI/Gemini prompts; Anthropic honours `temperature=0`; Gemini skips malformed image URLs; OpenAI handles empty `choices` arrays.
- **Steer hook fix** — nudge messages use `LLMMessage` objects instead of raw dicts.
- **Skill workshop** — blocks apply on any scanner finding (including malicious-pattern detections).
- **Sandbox env redaction** — broad secret-name matcher catches `*_TOKEN`, `*_API_KEY`, `*PASSWORD*`, etc.

#### v6.9.0

```bash
clawagents --task "summarize prior debugging runs" --output-format json
clawagents --task "find where we discussed pytest timeouts"  # uses search_history tool
```

- **`search_history` tool** — cross-session archive search over `.clawagents/sessions.db` plus optional JSONL event logs; returns raw prior user/assistant/tool snippets (not summaries). Current-session search stays on the session backend.
- **`--output-format`** — `text` (default), `json`, or `stream-json` on `clawagents --task` for automation-friendly stdout.
- **PTRL → skill promotion** — recurring lesson bullets (≥3 occurrences) auto-create **pending** `skill_workshop` proposals in `.clawagents/lesson-index.json`.
- **`skill_workshop` tool** — governed skill authoring (`create` / `update` / `apply` / `reject` / `rollback`) without writing live `SKILL.md` directly.
- **Consolidated search stack** — shared SQLite LIKE helpers (`session/search`) and snippet formatting (`session/snippet`); lesson bullet utilities live in `trajectory/lessons`.

#### v6.8.1

```bash
clawagents --dry-run --profile ollama --task "inspect this repo"
```

- **Shared prompt assembly** centralizes system prompt construction, lesson preambles, cache-boundary placement, and dynamic memory/skill injection in `clawagents.prompts`.
- **Legacy hook compatibility** keeps dict-shaped `before_llm` messages working while exposing reusable prompt helpers for downstream integrations.
- **OpenHarness comparison** adds [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) as a peer in the feature matrix with conservative full/partial markers.
- **Dry-run previews** report provider resolution, auth readiness, inspectable tools, likely matching tools, and next actions without calling an LLM or executing tools.
- **Provider profiles** give stable aliases for common backends while still letting explicit `create_claw_agent()` parameters override profile values.
- **Background task tools** expose long-running command management (`task_create`, `task_status`, `task_output`, `task_stop`, `task_list`) through the normal tool registry.
- **Plugin compatibility loading** reads `plugin.json` / `.claude-plugin/plugin.json` metadata, skills, commands, hooks, and MCP server declarations without executing plugin code.
- **MCP auth refresh** lets agents update MCP server auth material and reconnect configured servers deliberately.

## Trajectory Logging & RL-Inspired Scoring

ClawAgents includes an optional **trajectory system** inspired by reinforcement learning techniques from [CUDA-Agent](https://github.com/NexaAI/CUDA-Agent) and [OpenClaw-RL](https://github.com/anthropics/openclaw-rl). Enable it with `trajectory=True` or `CLAW_TRAJECTORY=1`.

### What gets logged

Every agent run records:
- **Turn-level data**: tool calls, arguments, success/failure, output previews
- **Weighted turn scores**: execution tools (shell, code runners) weighted 2× higher than generic tools
- **Run summary**: total turns, tool calls, successes/failures, elapsed time

### Discrete reward bands

Each run receives a score from **-1 to +3**:

| Score | Meaning |
|:---:|:---|
| **+3** | All tools succeeded, task completed cleanly |
| **+2** | Minor hiccups but overall success |
| **+1** | Partial success with some failures |
| **0** | Inconclusive — mixed results |
| **-1** | Majority of tool calls failed |

### Quality grading

Runs are classified for downstream filtering:

| Quality | Criteria |
|:---|:---|
| `clean` | Score ≥ 2 and ≤ 2 mid-run failures |
| `noisy` | Score ≥ 0 but too many mid-run failures |
| `failed` | Score < 0 |

### Anti-gaming protections

Tools like `think`, `todolist`, `use_skill`, `list_skills`, and `update_todo` are excluded from scoring — they can't inflate success rates.

### Consecutive-failure rethink

With `rethink=True` or `CLAW_RETHINK=1`, the agent monitors tool outcomes in real-time. After **3 consecutive meaningful failures**, it injects a system message:

> *"You have had 3 consecutive tool failures. Stop and rethink your approach before continuing."*

This simple mechanism prevents the agent from spiraling into repeated failed attempts.

### Output

Run summaries are appended to `.clawagents/trajectories/runs.jsonl`:

```json
{
  "run_id": "a1b2c3d4",
  "model": "gpt-5-mini",
  "total_turns": 8,
  "tool_calls": 12,
  "successes": 10,
  "failures": 2,
  "run_score": 2,
  "quality": "clean",
  "elapsed_ms": 45230,
  "turns": [...]
}
```

---

## Roadmap

- [ ] Docker sandbox backend (protocol ready)
- [ ] Semantic browser automation (accessibility tree)
- [ ] Prompt caching (Anthropic-style)
- [ ] Persistent memory learning from trajectory data (advanced — RFT-style rule extraction)
- [x] Post-run self-analysis + lesson extraction ✅ (v5.13 — PTRL)
- [x] Pre-run lesson injection ✅ (v5.13 — PTRL)
- [x] Enhanced mid-run rethink with past lessons ✅ (v5.13 — PTRL)
- [x] Trajectory logging + discrete reward bands ✅ (v5.9–5.10)
- [x] Consecutive-failure rethink injection ✅ (v5.9)
- [x] Weighted execution scoring + quality grading ✅ (v5.10)
- [x] JSON repair + truncated JSON retry ✅ (v5.8)
- [x] Model-specific temperature override ✅ (v5.7)
- [x] Configurable temperature / max_completion_tokens ✅ (v5.6)
- [x] Pluggable sandbox backend ✅ (v5.5)
- [x] Lane-based queue serialization ✅ (v5.5)
- [x] Skill progressive disclosure ✅ (v5.5)
- [x] Gateway HTTP server ✅ (v5.5)

---

## License

MIT

---

<p align="center">
  <strong>Built with 🦞 by the ClawAgents team</strong>
</p>
