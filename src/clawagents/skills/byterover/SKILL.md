---
name: byterover
description: "Knowledge management for AI agents. Use the `brv` CLI to store and retrieve project patterns, decisions, and architectural rules in .brv/context-tree. Use before work (brv query) and after implementing (brv curate). Install: npm install -g byterover-cli."
---

# ByteRover Knowledge Management

Use the `brv` CLI to manage your project's long-term memory.

**Install:** Optional. With npm, `byterover-cli` is an optional dependency of ClawAgents (so `brv` is available from the project's `node_modules/.bin`). With Python, the agent runs `brv` via `npx byterover-cli`, so Node/npx is sufficient. You can also install globally: `npm install -g byterover-cli`.

Knowledge is stored in `.brv/context-tree/` as human-readable Markdown. No authentication needed for query/curate; login only for cloud sync.

## Workflow

1. **Before thinking:** Run `brv query` to understand existing patterns.
2. **After implementing:** Run `brv curate` to save new patterns/decisions.

## Commands

### Query knowledge

Use when the user wants recall, your context lacks needed info, or before an action to check rules/preferences. Do not use when the info is already in context.

```bash
brv query "How is authentication implemented?"
brv query "What are the API rate limits?"
```

Headless mode with JSON output (for automation):

```bash
brv query "How does auth work?" --headless --format json
```

### Curate context

Use when the user wants you to remember something, or there are meaningful decisions to persist. Do not use for transient or general knowledge.

```bash
brv curate "Auth uses JWT with 24h expiry. Tokens stored in httpOnly cookies via authMiddleware.ts"
```

With source files (max 5):

```bash
brv curate "Authentication middleware details" -f src/middleware/auth.ts
brv curate "JWT implementation" --files src/auth/jwt.ts --files docs/auth.md
```

Curate an entire folder:

```bash
brv curate --folder src/auth/
brv curate "Analyze authentication module" -d src/auth/
```

### Project setup

Initialize ByteRover for the current project:

```bash
brv init
brv init --force
```

### Authentication (for cloud sync)

```bash
brv login --api-key YOUR_KEY
```

### Cloud sync (optional)

```bash
brv pull
brv pull --branch feature-branch
brv push
```

### Status

```bash
brv status
```

## Error handling

- "No ByteRover instance is running" → run `brv` in a separate terminal first
- "Not logged in" → only needed for push/pull; run `brv login --api-key <key>`
- "Maximum 5 files allowed" → use ≤5 `-f` flags
- "File does not exist" → verify path from project root
- "Sandbox environment detected" → run `brv` outside the sandbox/IDE terminal

## Best practices

- Query before starting work; curate after completing work.
- Use precise queries and concise, specific curate text.
- Attach files with `-f` instead of pasting content.
- Use `-d` to curate an entire folder when multiple related files changed.
- Mark outdated knowledge as OUTDATED when replacing it.
- Use `--headless --format json` for automation pipelines.

Source: [ByteRover on ClawHub](https://clawhub.ai/byteroverinc/byterover)
