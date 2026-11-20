# ClawAgents Python — Documentation

## Core Guides

| Guide | Description |
| --- | --- |
| [README](../README.md) | Overview, installation, 30-second quick start, and ecosystem links |
| [Configuration & Providers](configuration.md) | `.env` resolution, PTRL, Advisor model, GRPO, Azure, Bedrock/Mantle, Local models, MCP, Browser, Cron, ACP, RL hooks, and full environment variables reference |
| [Python API Reference](api-reference.md) | `create_claw_agent(...)` parameters, hooks & access control, instance methods, and auto-discovery |
| [CLI & Gateway Server](cli-and-gateway.md) | CLI commands, HTTP/WebSocket gateway, bind/auth, messaging channel allowlists, lane concurrency, and trust boundaries |
| [Architecture, Tools & Benchmarks](architecture.md) | Core components, built-in & custom tools, sandbox backends, feature matrix, and comparative benchmarks |
| [Skills & Memory](skills.md) | `SKILL.md` format, writing custom skills, bundled skills, `AGENTS.md` memory, and auto-compaction |

## Model & Engineering Notes

| Document | Description |
| --- | --- |
| [Model Catalog & Pricing Audit](model-catalog-audit.md) | Verified provider model IDs, context windows, thinking/Fast-mode contracts, and pricing sources |
| [Meta / Muse-Glimmer Integration](meta-glimmer.md) | Self-hosted Muse-Glimmer-30B setup, reasoning controls, context limits, and benchmarks |
| [Gemma Q4 Coordination](gemma-coordination.md) | Local `gemma-agentic` Q4 setup, worker routing, and verified completion |
| [Agent Efficiency Notes](efficiency.md) | Fused follow-ups, paged artifact recall, run-scoped todos, and diagnostic reduction |
| [Efficiency Benchmarks](../benchmarks/efficiency/README.md) | Paired efficiency benchmark CLI and manifests |

## Security, Releases & Contributing

| Document | Description |
| --- | --- |
| [Security Policy](../SECURITY.md) | Trust boundaries, sandboxing, credential isolation, and vulnerability reporting |
| [Changelog](../CHANGELOG.md) | Recent releases (`v6.20.80`–present) |
| [Changelog Archive](changelog-archive.md) | Historical releases (`v6.20.77` and earlier) |
| [Agent Instructions](../AGENTS.md) | Repo conventions and architecture notes for coding agents |

## Related Repositories

- **VS Code / Cursor Extension:** [github.com/x1jiang/clawagents-vscode](https://github.com/x1jiang/clawagents-vscode) ([Docs](https://github.com/x1jiang/clawagents-vscode/tree/main/docs))
- **Desktop App (macOS):** [github.com/x1jiang/clawagents-desktop](https://github.com/x1jiang/clawagents-desktop)
- **TypeScript SDK:** [github.com/x1jiang/clawagents](https://github.com/x1jiang/clawagents)
