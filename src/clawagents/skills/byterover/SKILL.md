---
name: byterover
description: "Knowledge management for AI agents. Use the `brv` CLI to store and retrieve project patterns, decisions, and architectural rules in .brv/context-tree. Use before work (brv query) and after implementing (brv curate). Install: npm install -g byterover-cli."
---

# ByteRover Knowledge Management

Use the `brv` CLI to manage your project's long-term memory.

**Install:** Optional. When using ClawAgents (Python), the agent runs `brv` via `npx byterover-cli`, so Node/npx is sufficient. You can also install globally: `npm install -g byterover-cli`.

Knowledge is stored in `.brv/context-tree/` as human-readable Markdown. No authentication needed for query/curate; login only for cloud sync.

## Workflow

1. **Before thinking:** Run `brv query` to understand existing patterns.
2. **After implementing:** Run `brv curate` to save new patterns/decisions.

## Commands

### Query knowledge

Use when the user wants recall, your context lacks needed info, or before an action to check rules/preferences. Do not use when the info is already in context.

```bash
brv query "How is authentication implemented?"
```

### Curate context

Use when the user wants you to remember something, or there are meaningful decisions to persist. Do not use for transient or general knowledge.

```bash
brv curate "Auth uses JWT with 24h expiry. Tokens stored in httpOnly cookies via authMiddleware.ts"
```

With source files (max 5):

```bash
brv curate "Authentication middleware details" -f src/middleware/auth.ts
```

View curate history:

```bash
brv curate view
brv curate view cur-1739700001000
brv curate view detail
```

### Provider setup

Default provider (no API key):

```bash
brv providers connect byterover
```

Use your own LLM:

```bash
brv providers list
brv providers connect openai --api-key sk-xxx --model gpt-4.1
```

### Cloud sync (optional)

```bash
brv login --api-key YOUR_KEY
brv space list
brv space switch --team TEAM --name SPACE
brv pull
brv push
```

### Status

```bash
brv status
```

## Error handling

- "No provider connected" → `brv providers connect byterover`
- "Not authenticated" → only needed for push/pull; run `brv login --help` if required
- "Maximum 5 files allowed" → use ≤5 `-f` flags
- "File does not exist" → verify path from project root

## Best practices

- Query before starting work; curate after completing work.
- Use precise queries and concise, specific curate text.
- Attach files with `-f` instead of pasting content.
- Mark outdated knowledge as OUTDATED when replacing it.

Source: [ByteRover on ClawHub](https://clawhub.ai/byteroverinc/byterover)
