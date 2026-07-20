---
name: graphify
description: "Local knowledge graph for code and docs (Graphify). Extract a folder into graphify-out/graph.json, then query with CLI or MCP before bulk reads. Install: pip install 'graphifyy[mcp]'"
requires.bins: graphify
---

# Graphify — local knowledge graph

Turn a folder of code, schemas, docs, papers, or notes into a queryable NetworkX graph. Prefer graph queries over dumping whole trees into context.

**Install:** `pip install 'graphifyy[mcp]'`  
**Upstream:** https://github.com/Graphify-Labs/graphify

## When to use

- Architecture / dependency / “how does X connect to Y?” questions
- Exploring an unfamiliar codebase or personal notes corpus
- Before large `read_file` / `grep` sweeps when a graph already exists
- After major edits: incremental `graphify update`

Do **not** paste `graph.json` or `GRAPH_REPORT.md` into the system prompt — query via CLI/MCP.

## Bootstrap (workspace)

Prefer **code-only** extract. A plain `extract` without `--code-only` runs a
semantic/docs pass that needs an LLM backend; without keys it can exit 0 and
still leave **no** `graph.json`.

```bash
# Prefer ClawAgents layout (VS Code uses this path). Use python3 (not bare python).
export GRAPHIFY_OUT=.clawagents/graphify
python3 -m graphify extract . --code-only
# Augment after edits (AST, no LLM):
python3 -m graphify update .
```

Full extract (docs/images, needs LLM):

```bash
export GRAPHIFY_OUT=.clawagents/graphify
python3 -m graphify extract .
```

Upstream default output is `graphify-out/` if `GRAPHIFY_OUT` is unset. If that
already exists, VS Code can **Use existing graphify-out** to copy it into
`.clawagents/graphify/`.

## Query

```bash
graphify query "authentication flow" --budget 2000
graphify path "AuthService" "UserStore"
graphify explain "ScopeGraph"
```

Or use MCP tools when the Graphify server is connected: `query_graph`, `get_node`, `get_neighbors`, `shortest_path`, `god_nodes`, `graph_stats`.

```bash
python3 -m graphify.serve .clawagents/graphify/graph.json
```

## Incremental update

```bash
export GRAPHIFY_OUT=.clawagents/graphify
python3 -m graphify update .
```

## Personal / global KB

```bash
graphify global add /path/to/notes --as notes
# serve ~/.graphify/global-graph.json via MCP, or set ClawAgents graphify_graph_path
```

## ClawAgents VS Code

1. Settings → enable **Graphify**
2. Command Palette → **ClawAgents: Graphify — Extract/Update Workspace**
3. Optional: set **graph path** for a personal KB or `~/.graphify/global-graph.json`
4. Ensure package in the sidecar venv: **ClawAgents: Ensure Companions** (installs `graphifyy[mcp]` into the managed interpreter)

## Outputs

| Path | Role |
|------|------|
| `.clawagents/graphify/graph.json` | Canonical graph (ClawAgents default) |
| `graphify-out/graph.json` | Upstream default |
| `GRAPH_REPORT.md` | Human/agent summary (read on demand) |
| `wiki/` | Optional markdown wiki (`--wiki`) |
