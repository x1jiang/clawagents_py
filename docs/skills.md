# Skills & Memory

Back to [Documentation Index](README.md) · [README](../README.md)

## Skills System

Skills are **reusable instruction sets** that teach the agent domain-specific knowledge without blindly injecting every body. Retrieval is quality-first: a compact name index preserves discoverability, structured aliases/triggers/anti-triggers and morphology-aware ranking retain every plausible intent up to a generous ambiguity cap, and short follow-ups inherit the prior substantive request. `list_skills` uses the same ranked search when the model needs a wider view.

`use_skill` delivers large instructions in contiguous, content-hash-bound pages. No data-plane tool can run until every page is read; skipped, stale, or reordered continuations fail. Multiple completed skills compose, and their declared `allowed-tools` boundaries intersect so loading another skill cannot widen authority. An explicit empty boundary allows no data-plane tools.

### Skill Directory Structure

```
your-project/
├── skills/                  # Auto-discovered (or .skills/, skill/, .skill/, Skills/)
│   ├── code_review/
│   │   └── SKILL.md         # ← Skill defined as a folder + SKILL.md
│   ├── sql_expert.md         # ← Skill defined as a single .md file
│   └── deploy_checklist.md
├── AGENTS.md                 # Project memory (auto-injected)
└── src/
    └── ...
```

### Writing a Skill

Every skill is a Markdown file with optional YAML frontmatter:

**Example 1 — `skills/code_review/SKILL.md`**

```markdown
---
name: code_review
description: "Perform thorough code reviews following team standards"
allowed-tools: read_file grep glob think
---

# Code Review Skill

When reviewing code, follow these steps:

## 1. Structure Check
- Verify the file follows our module pattern (one class per file)
- Check imports are grouped: stdlib → third-party → local
- Ensure `__init__.py` exports are up to date

## 2. Logic Review
- Look for unhandled edge cases (empty inputs, None values)
- Verify error messages are actionable
- Check that async functions are properly awaited

## 3. Security
- No hardcoded secrets or API keys
- SQL queries use parameterized statements
- User input is sanitized before use

## 4. Output Format
Provide your review as:
- ✅ **Approved** — no issues found
- ⚠️ **Changes requested** — list specific issues with file:line references
- 🚫 **Blocked** — critical issues that must be fixed
```

**Example 2 — `skills/sql_expert.md`** (single-file skill)

```markdown
---
name: sql_expert
description: "Write optimized SQL queries for PostgreSQL"
allowed-tools: execute read_file think
---

# SQL Expert

You are a PostgreSQL expert. When writing queries:

## Rules
1. Always use explicit `JOIN` syntax (never implicit joins in WHERE)
2. Use CTEs (`WITH` clauses) for complex multi-step queries
3. Add `EXPLAIN ANALYZE` when the user asks about performance
4. Use parameterized queries — never interpolate user values
5. Default to `LIMIT 100` unless the user specifies otherwise

## Patterns

### Pagination
Use keyset pagination for large tables:
```sql
SELECT * FROM events
WHERE id > :last_seen_id
ORDER BY id
LIMIT 50;
```

### Aggregation
Always include the raw count alongside percentages:
```sql
SELECT
    status,
    COUNT(*) AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM orders
GROUP BY status
ORDER BY n DESC;
```
```

**Example 3 — `skills/deploy_checklist.md`**

```markdown
---
name: deploy_checklist
description: "Step-by-step production deployment checklist"
---

# Deployment Checklist

Before deploying to production, complete every step:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] No lint errors: `ruff check src/`
- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Docker image builds: `docker build -t app:latest .`
- [ ] Smoke test on staging environment
- [ ] Database migrations reviewed and tested
- [ ] Rollback plan documented
```

### How Skills Work at Runtime

```python
# Skills are auto-discovered from ./skills/ directory
agent = create_claw_agent("gemini-3-flash")

# Or specify custom skill directories
agent = create_claw_agent("gpt-5", skills=["./my-skills", "./shared-skills"])
```

When skills are available, the agent gets two additional tools:

```python
# 1. List available skills
{"tool": "list_skills", "args": {}}
# → Available skills (3):
#   - **code_review**: Perform thorough code reviews following team standards
#     → Allowed tools: read_file, grep, glob, think
#   - **sql_expert**: Write optimized SQL queries for PostgreSQL
#     → Allowed tools: execute, read_file, think
#   - **deploy_checklist**: Step-by-step production deployment checklist

# 2. Load a specific skill's instructions
{"tool": "use_skill", "args": {"name": "sql_expert"}}
# → Returns the full skill content, injected into the agent's context
```

The agent **decides on its own** when to use a skill. If you ask it to "write a query to find all overdue orders," and a `sql_expert` skill exists, it will load the skill first, then write the query following those rules.

---

### Bundled Skills

| Skill | Purpose | Prerequisite | Auto-enabled? |
|:---|:---|:---|:---:|
| **[OpenViking](https://github.com/volcengine/OpenViking)** | Tiered context retrieval (L0/L1/L2) over repos and docs | `pip install openviking` + running `openviking-server` | Only when `ov` CLI is on PATH |

**OpenViking** is a structured context database. Use `ov add-resource` to ingest repos or docs, then `ov find` for semantic search. Results use a virtual filesystem (`viking://`) with **L0** (abstract), **L1** (overview), **L2** (full content).

**OpenViking prerequisites:**
1. Install: `pip install openviking --upgrade`
2. Configure: create `~/.openviking/ov.conf` with embedding model and VLM settings (see [OpenViking docs](https://github.com/volcengine/OpenViking))
3. Start server: `openviking-server`
4. Put `ov` on your PATH (often `~/Library/Python/3.11/bin` on macOS)

> **Note:** ByteRover was removed — its default LLM provider is a cloud internet service.

Override with explicit paths:
```python
agent = create_claw_agent(
    "gpt-5",
    memory="./docs/AGENTS.md",
    skills=["./my-skills", "./shared-skills"]
)
```

---

## Memory & Context Management

### Project Memory
Loads `AGENTS.md` (and `CLAWAGENTS.md`) from the working directory and injects their content into every LLM call. Use for project-level context and conventions.

### Auto-Compaction
When the conversation exceeds **75% of `CONTEXT_WINDOW`**:
1. Full history **offloaded** to `.clawagents/history/compacted_<ts>_<N>msgs.json`
2. Older messages **summarized** into a single placeholder message tagged `[System — Compacted History]`
3. Last 20 messages kept intact

This provides **unlimited conversation length** with full audit trail preservation.

---
