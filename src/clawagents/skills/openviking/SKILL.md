---
name: openviking
description: "Context database for AI agents. Use the `ov` CLI to store, organize, and retrieve project context (resources, memories, skills) in a hierarchical filesystem paradigm with tiered L0/L1/L2 loading. Query before work to load relevant context; add resources to build the knowledge base. Install: pip install openviking"
requires.bins: ov
---

# OpenViking — Context Database

Use the `ov` CLI to manage structured context for the agent. OpenViking organizes memories, resources, and skills into a virtual filesystem under the `viking://` protocol with three tiers: **L0** (abstract, ~100 tokens), **L1** (overview, ~2k tokens), **L2** (full content).

**Install:** `pip install openviking --upgrade`. The OpenViking server must be running (`openviking-server`) with a valid config at `~/.openviking/ov.conf`.

## When to use

- **Before answering questions** about project architecture, docs, decisions, or past work — always run `ov find` first.
- **When the user adds a new resource** (repo, URL, document) — run `ov add-resource`.
- **When exploring what's available** — run `ov ls` or `ov tree`.
- **Do not use** when the information is already in your context window.

## Commands

### Check server status

```bash
ov status
```

### Add a resource

Add a GitHub repo, URL, or local path to the knowledge base. OpenViking automatically processes it into L0/L1/L2 tiers.

```bash
ov add-resource https://github.com/user/repo
ov add-resource https://docs.example.com/guide
ov add-resource /path/to/local/docs
```

Use `--wait` to block until processing completes:

```bash
ov add-resource https://github.com/user/repo --wait
```

Filter files during import:

```bash
ov add-resource ./my-project --include "*.py,*.md" --ignore-dirs "node_modules,dist"
```

### Browse the filesystem

List contents of a directory:

```bash
ov ls
ov ls viking://resources/
ov ls viking://user/memories/
ov ls viking://agent/skills/
```

Show a directory tree (default depth 3, limit with `-L`):

```bash
ov tree viking://resources/my_project -L 2
ov tree viking://agent/ -L 3
```

### Search for context

Semantic search across all context — the primary retrieval command:

```bash
ov find "how does authentication work"
ov find "database migration strategy" -n 5
```

Scope search to a specific URI:

```bash
ov find "auth flow" --uri viking://resources/my_project
```

Context-aware search (uses session history for better results):

```bash
ov search "next steps" --session-id <session-id>
```

Text search (grep-style):

```bash
ov grep "auth" --uri viking://resources/my_project/docs
ov grep "TODO" --ignore-case
```

### Read content at different tiers

Read L0 abstract (cheapest, ~100 tokens — use for quick relevance check):

```bash
ov abstract viking://resources/my_project
ov abstract viking://resources/my_project/docs/api
```

Read L1 overview (~2k tokens — use for planning and decision-making):

```bash
ov overview viking://resources/my_project
ov overview viking://resources/my_project/docs
```

Read L2 full content (use only when deep detail is needed):

```bash
ov read viking://resources/my_project/docs/api/auth.md
```

### Memory management

Add a memory directly:

```bash
ov add-memory "Auth uses JWT with 24h expiry, stored in httpOnly cookies"
```

Browse stored memories:

```bash
ov ls viking://user/memories/
ov ls viking://agent/memories/
```

## Retrieval strategy

Follow this pattern for optimal token efficiency:

1. **Start with `ov find`** to semantically locate relevant directories.
2. **Read L0 abstracts** (`ov abstract <uri>`) of top results to confirm relevance.
3. **Read L1 overviews** (`ov overview <uri>`) of the most relevant hits for planning.
4. **Read L2 full content** (`ov read <uri>`) only for the specific files you need.

This tiered approach minimizes token consumption while maximizing context quality.

## Filesystem structure

```
viking://
├── resources/        # Project docs, repos, web pages
├── user/             # User preferences and memories
│   └── memories/
└── agent/            # Agent skills, instructions, task memories
    ├── skills/
    ├── memories/
    └── instructions/
```

## Error handling

- "Connection refused" / "Network error" → start the server: `openviking-server`
- "Config not found" → create `~/.openviking/ov.conf` (see OpenViking docs)
- "Resource not found" → check URI with `ov ls` at the parent directory
- "Processing in progress" → resource was added recently; wait or use `--wait` flag

## Best practices

- Always `ov find` before answering architecture or design questions.
- Use `ov tree -L 2` to understand structure before diving into specific files.
- Prefer L0/L1 reads over L2 — only load full content when necessary.
- After adding a resource, allow time for processing before querying.
- Use `ov grep` for exact text matches; use `ov find` for semantic queries.
- Use `-o json` on any command for machine-readable output.

Source: [OpenViking on GitHub](https://github.com/volcengine/OpenViking)
