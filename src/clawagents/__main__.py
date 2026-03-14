"""ClawAgents CLI — run tasks, start the gateway, or scaffold a new project.

Usage:
    clawagents --init                    # Scaffold a starter project in current dir
    clawagents --doctor                  # Check configuration health
    clawagents --task "Fix the bug"      # Run a single task
    clawagents --trajectory              # Inspect last run's trajectory
    clawagents --serve --port 3000       # Start the gateway server
    python -m clawagents --init          # Same as above via python -m
"""

import sys
import argparse
import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent


# ─── Init / Scaffold ──────────────────────────────────────────────────────

_ENV_TEMPLATE = dedent("""\
    # ClawAgents Configuration
    # Uncomment ONE provider section below.

    # ── OpenAI ──────────────────────────────────────────────────────────
    # PROVIDER=openai
    # OPENAI_API_KEY=sk-...
    # OPENAI_MODEL=gpt-5-mini

    # ── Google Gemini ───────────────────────────────────────────────────
    # PROVIDER=gemini
    # GEMINI_API_KEY=AIza...
    # GEMINI_MODEL=gemini-3-flash-preview

    # ── Azure OpenAI ────────────────────────────────────────────────────
    # PROVIDER=openai
    # OPENAI_API_KEY=your-azure-key
    # OPENAI_MODEL=gpt-4o
    # OPENAI_BASE_URL=https://YOUR_RESOURCE.openai.azure.com/
    # OPENAI_API_VERSION=2024-12-01-preview

    # ── Local Model (Ollama / vLLM / LM Studio) ────────────────────────
    # PROVIDER=openai
    # OPENAI_MODEL=llama3.1
    # OPENAI_BASE_URL=http://localhost:11434/v1

    # ── Shared Settings ─────────────────────────────────────────────────
    STREAMING=1
    CONTEXT_WINDOW=1000000
    MAX_TOKENS=8192
    TEMPERATURE=0

    # ── Optional: PTRL (Prompt-Time Reinforcement Learning) ─────────────
    # CLAW_TRAJECTORY=1
    # CLAW_RETHINK=1
    # CLAW_LEARN=1

    # ── Optional: Messaging Channels (auto-detected by --serve) ───────
    # TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    # WHATSAPP_AUTH_DIR=.whatsapp-auth
    # SIGNAL_ACCOUNT=+1234567890
    # CHANNEL_DEBOUNCE_MS=500
""")

_EXAMPLE_SCRIPT = dedent('''\
    """ClawAgents Quick Start — edit and run this file."""
    import asyncio
    from clawagents import create_claw_agent


    async def main():
        # ── Option 1: Auto-detect from .env (simplest) ──
        agent = create_claw_agent()

        # ── Option 2: Explicit model ──
        # agent = create_claw_agent("gpt-5-mini")

        # ── Option 3: With learning enabled ──
        # agent = create_claw_agent("gpt-5-mini", learn=True, rethink=True)

        # ── Option 4: Local model (Ollama) ──
        # agent = create_claw_agent("llama3.1", base_url="http://localhost:11434/v1")

        # ── Option 5: Azure OpenAI ──
        # agent = create_claw_agent(
        #     "gpt-4o",
        #     api_key="your-azure-key",
        #     base_url="https://YOUR_RESOURCE.openai.azure.com/",
        #     api_version="2024-12-01-preview",
        # )

        # Run a task
        result = await agent.invoke("List all Python files in the current directory")
        print("\\n" + "=" * 60)
        print("RESULT:")
        print(result.result)
        print(f"\\nTool calls: {result.tool_calls}, Iterations: {result.iterations}")


    if __name__ == "__main__":
        asyncio.run(main())
''')

_AGENTS_MD_TEMPLATE = dedent("""\
    # Agent Memory

    This file is automatically injected into the agent's system prompt.
    Add project-specific context, coding standards, or instructions here.

    ## Project
    - Language: Python
    - Framework: (your framework here)

    ## Coding Standards
    - Use type hints
    - Write docstrings for public functions
    - Keep functions under 50 lines
""")


def cmd_init():
    """Scaffold a starter ClawAgents project in the current directory."""
    cwd = Path.cwd()
    created = []

    env_path = cwd / ".env"
    if not env_path.exists():
        env_path.write_text(_ENV_TEMPLATE, encoding="utf-8")
        created.append(".env")
    else:
        sys.stderr.write("  .env already exists — skipping\n")

    script_path = cwd / "run_agent.py"
    if not script_path.exists():
        script_path.write_text(_EXAMPLE_SCRIPT, encoding="utf-8")
        created.append("run_agent.py")
    else:
        sys.stderr.write("  run_agent.py already exists — skipping\n")

    agents_path = cwd / "AGENTS.md"
    if not agents_path.exists():
        agents_path.write_text(_AGENTS_MD_TEMPLATE, encoding="utf-8")
        created.append("AGENTS.md")
    else:
        sys.stderr.write("  AGENTS.md already exists — skipping\n")

    if created:
        sys.stderr.write(f"\n✓ Created: {', '.join(created)}\n")
    sys.stderr.write(dedent("""\

        Next steps:
          1. Edit .env — uncomment your provider and add your API key
          2. Run:  python run_agent.py
          3. Or:   clawagents --task "your task here"

        Docs: https://github.com/x1jiang/clawagents_py
    """))


# ─── Doctor ───────────────────────────────────────────────────────────────

def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    msg = f"  {mark} {label}"
    if detail:
        msg += f" — {detail}"
    sys.stderr.write(msg + "\n")
    return ok


def cmd_doctor():
    """Check configuration health and report issues."""
    sys.stderr.write("\nClawAgents Doctor\n" + "=" * 40 + "\n\n")
    issues = 0

    # 1. Load config first (triggers .env discovery)
    import clawagents.config.config as _cfg
    config = _cfg.load_config()

    # 2. .env file discovery (must check AFTER load_config sets env_file)
    if _cfg.env_file:
        _check(".env file", True, str(_cfg.env_file))
    else:
        _check(".env file", False, "not found in cwd or parent (run `clawagents --init`)")
        issues += 1

    # 3. API keys
    has_openai = bool(config.openai_api_key)
    has_gemini = bool(config.gemini_api_key)
    has_base_url = bool(config.openai_base_url)

    if has_openai:
        _check("OpenAI API key", True, f"{'*' * 8}...{config.openai_api_key[-4:]}" if len(config.openai_api_key) > 12 else "set")
    elif has_base_url:
        _check("OpenAI API key", True, "not needed (base_url set for local model)")
    else:
        _check("OpenAI API key", False, "OPENAI_API_KEY not set")
        issues += 1

    if has_gemini:
        _check("Gemini API key", True, f"{'*' * 8}...{config.gemini_api_key[-4:]}" if len(config.gemini_api_key) > 12 else "set")
    else:
        _check("Gemini API key", False, "GEMINI_API_KEY not set")
        if not has_openai and not has_base_url:
            issues += 1

    # 4. Active provider/model
    if has_openai or has_gemini or has_base_url:
        model = _cfg.get_default_model(config)
        provider = os.getenv("PROVIDER", "auto-detect")
        _check("Active model", True, f"provider={provider}  model={model}")
    else:
        _check("Active model", False, "no API key or base_url configured")
        issues += 1

    # 5. Custom endpoint
    if has_base_url:
        _check("Custom endpoint", True, config.openai_base_url)
        if config.openai_api_version:
            _check("Azure API version", True, config.openai_api_version)

    # 6. LLM tuning
    sys.stderr.write("\n  LLM Settings:\n")
    sys.stderr.write(f"    max_tokens={config.max_tokens}  temperature={config.temperature}  "
                     f"context_window={config.context_window}  streaming={config.streaming}\n")

    # 7. PTRL flags
    traj = os.getenv("CLAW_TRAJECTORY", "0") in ("1", "true", "yes")
    rethink = os.getenv("CLAW_RETHINK", "0") in ("1", "true", "yes")
    learn = os.getenv("CLAW_LEARN", "0") in ("1", "true", "yes")
    max_iter = os.getenv("MAX_ITERATIONS", "200")

    sys.stderr.write(f"\n  PTRL: trajectory={'on' if traj else 'off'}  "
                     f"rethink={'on' if rethink else 'off'}  "
                     f"learn={'on' if learn else 'off'}  "
                     f"max_iterations={max_iter}\n")

    # 8. Local endpoint reachability
    if has_base_url and ("localhost" in config.openai_base_url or "127.0.0.1" in config.openai_base_url):
        try:
            import urllib.request
            url = config.openai_base_url.rstrip("/") + "/models"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Authorization", f"Bearer {config.openai_api_key or 'not-needed'}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                _check("Local endpoint reachable", resp.status == 200, url)
        except Exception as e:
            _check("Local endpoint reachable", False, f"{config.openai_base_url} — {e}")
            issues += 1

    # 9. Trajectory directory
    traj_dir = Path.cwd() / ".clawagents" / "trajectories"
    runs_file = traj_dir / "runs.jsonl"
    if runs_file.exists():
        line_count = sum(1 for _ in runs_file.open())
        _check("Trajectory history", True, f"{line_count} runs in {runs_file}")
    elif traj:
        _check("Trajectory history", True, "enabled but no runs yet")
    else:
        _check("Trajectory history", True, "disabled (set CLAW_TRAJECTORY=1 to enable)")

    # 10. AGENTS.md / memory
    agents_md = Path.cwd() / "AGENTS.md"
    if agents_md.exists():
        size = agents_md.stat().st_size
        _check("AGENTS.md", True, f"{size} bytes")
    else:
        _check("AGENTS.md", True, "not found (optional — create one with project context)")

    # 11. Messaging channels
    from clawagents.channels.auto import detect_channels, describe_channels
    channels = detect_channels()
    if channels:
        ch_list = ", ".join(describe_channels(channels))
        _check("Messaging channels", True, ch_list)
    else:
        _check("Messaging channels", True,
               "none configured (set TELEGRAM_BOT_TOKEN, WHATSAPP_AUTH_DIR, or SIGNAL_ACCOUNT)")

    # Summary
    sys.stderr.write("\n" + "=" * 40 + "\n")
    if issues == 0:
        sys.stderr.write("✓ All checks passed. Ready to run.\n\n")
    else:
        sys.stderr.write(f"✗ {issues} issue(s) found. Fix the items above.\n\n")


# ─── Task Runner ──────────────────────────────────────────────────────────

def _build_banner() -> str:
    """Build a one-line startup banner showing active config."""
    import clawagents.config.config as _cfg
    config = _cfg.load_config()
    model = _cfg.get_default_model(config)
    provider = os.getenv("PROVIDER", "auto")
    env_src = str(_cfg.env_file) if _cfg.env_file else "none"

    flags = []
    if os.getenv("CLAW_LEARN", "0") in ("1", "true", "yes"):
        flags.append("learn")
    if os.getenv("CLAW_RETHINK", "0") in ("1", "true", "yes"):
        flags.append("rethink")
    if os.getenv("CLAW_TRAJECTORY", "0") in ("1", "true", "yes"):
        flags.append("trajectory")
    flag_str = "+".join(flags) if flags else "none"

    return f"ClawAgents | provider={provider} model={model} env={env_src} ptrl={flag_str}"


async def cmd_task(task: str, timeout_s: int = 0):
    """Run a single task and print the result."""
    from clawagents.agent import create_claw_agent

    banner = _build_banner()
    agent = create_claw_agent()
    tool_count = len(agent.tools.list())
    sys.stderr.write(f"{banner} | {tool_count} tools\n")

    result = await agent.invoke(task, timeout_s=timeout_s)
    if result.result:
        print(result.result)


# ─── Trajectory Inspector ────────────────────────────────────────────────

def cmd_trajectory(n: int = 1):
    """Show the last N run summaries from trajectory logs."""
    runs_file = Path.cwd() / ".clawagents" / "trajectories" / "runs.jsonl"
    if not runs_file.exists():
        sys.stderr.write("No trajectory data found.\n")
        sys.stderr.write("Enable with: CLAW_TRAJECTORY=1 in .env or trajectory=True in create_claw_agent()\n")
        return

    lines = runs_file.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        sys.stderr.write("Trajectory file is empty — no runs recorded yet.\n")
        return

    last_n = lines[-n:]
    for i, line in enumerate(last_n):
        try:
            run = json.loads(line)
        except json.JSONDecodeError:
            continue

        run_id = run.get("run_id", "?")[:12]
        model = run.get("model", "?")
        task = run.get("task", "?")
        turns = run.get("total_turns", 0)
        calls = run.get("total_tool_calls", 0)
        score = run.get("run_score", "?")
        quality = run.get("quality", "?")
        duration = run.get("duration_s", 0)
        success_rate = run.get("tool_success_rate", 0)
        judge = run.get("judge_score")
        judge_text = run.get("judge_justification", "")
        task_type = run.get("task_type", "")
        fmt_fail = run.get("format_failures", 0)
        logic_fail = run.get("logic_failures", 0)
        verified = run.get("verified_score")

        if len(last_n) > 1:
            sys.stderr.write(f"\n── Run {i+1}/{len(last_n)} ──\n")
        else:
            sys.stderr.write("\n── Latest Run ──\n")

        sys.stderr.write(f"  Run ID:    {run_id}\n")
        sys.stderr.write(f"  Model:     {model}\n")
        sys.stderr.write(f"  Task:      {task[:80]}{'...' if len(task) > 80 else ''}\n")
        if task_type:
            sys.stderr.write(f"  Type:      {task_type}\n")
        sys.stderr.write(f"  Duration:  {duration:.1f}s\n")
        sys.stderr.write(f"  Turns:     {turns}  Tool calls: {calls}  Success rate: {success_rate:.0%}\n")
        sys.stderr.write(f"  Score:     {score}/3  Quality: {quality}\n")

        if fmt_fail or logic_fail:
            sys.stderr.write(f"  Failures:  format={fmt_fail}  logic={logic_fail}\n")
        if verified is not None:
            sys.stderr.write(f"  Verified:  {verified:.2f} ({run.get('verified_method', '')})\n")
        if judge is not None:
            sys.stderr.write(f"  Judge:     {judge}/3")
            if judge_text:
                sys.stderr.write(f" — {judge_text[:100]}")
            sys.stderr.write("\n")

    sys.stderr.write("\n")


# ─── Gateway Server ──────────────────────────────────────────────────────

def cmd_serve(port: int):
    """Start the HTTP + WS gateway server, plus any auto-detected channels."""
    from clawagents.gateway.server import create_app, start_gateway
    from clawagents.channels.auto import detect_channels, describe_channels, start_channel_router
    import uvicorn

    banner = _build_banner()
    channels = detect_channels()

    if channels:
        ch_desc = ", ".join(describe_channels(channels))
        sys.stderr.write(f"{banner} | gateway on port {port} | channels: {ch_desc}\n")

        app, llm, active_model = create_app()

        async def _run():
            router = await start_channel_router(llm)
            config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
            server = uvicorn.Server(config)
            try:
                await server.serve()
            finally:
                if router:
                    await router.stop_all()

        gateway_api_key = os.getenv("GATEWAY_API_KEY", "")
        auth_status = "enabled" if gateway_api_key else "disabled (set GATEWAY_API_KEY to enable)"
        sys.stderr.write(f"   Auth: {auth_status}\n")
        sys.stderr.write(f"   Endpoints: POST /chat | POST /chat/stream | WS /ws | GET /queue | GET /health\n\n")
        asyncio.run(_run())
    else:
        sys.stderr.write(f"{banner} | gateway on port {port}\n")
        start_gateway(port)


# ─── Trajectory Pruning ───────────────────────────────────────────────

def cmd_prune_trajectories(days: int):
    """Delete trajectory files older than N days."""
    import time
    traj_dir = Path.cwd() / ".clawagents" / "trajectories"
    if not traj_dir.exists():
        print("No trajectories directory found.")
        return
    cutoff = time.time() - days * 86400
    removed = 0
    for f in traj_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    print(f"Pruned {removed} trajectory file(s) older than {days} days.")


# ─── Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ClawAgents — lean, full-stack agentic AI framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              clawagents --init                    Scaffold a new project (.env, run_agent.py, AGENTS.md)
              clawagents --doctor                  Check configuration health
              clawagents --task "List all files"   Run a task directly from the command line
              clawagents --trajectory              Show last run's trajectory summary
              clawagents --trajectory 5            Show last 5 runs
              clawagents --serve --port 3000       Start the HTTP gateway server

            Messaging channels (auto-detected from .env):
              TELEGRAM_BOT_TOKEN=...          → starts Telegram bot
              WHATSAPP_AUTH_DIR=.wa-auth      → starts WhatsApp (Baileys QR pairing)
              SIGNAL_ACCOUNT=+1234567890      → starts Signal (via signal-cli)

            Quick start:
              pip install clawagents
              clawagents --init
              # Edit .env with your API key
              python run_agent.py
        """),
    )
    parser.add_argument("--init", action="store_true", help="Scaffold a starter project in the current directory")
    parser.add_argument("--doctor", action="store_true", help="Check configuration health")
    parser.add_argument("--task", type=str, help="Run a single task from CLI")
    parser.add_argument("--trajectory", nargs="?", const=1, type=int, metavar="N", help="Show last N run summaries (default: 1)")
    parser.add_argument("--serve", action="store_true", help="Start the HTTP gateway server")
    parser.add_argument("--port", type=int, default=3000, help="Port for the gateway server (default: 3000)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (show all tool results)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (only show final result)")
    parser.add_argument("--timeout", type=int, default=0, help="Global timeout in seconds (0 = no timeout)")
    parser.add_argument("--prune-trajectories", type=int, metavar="DAYS", help="Delete trajectory files older than N days")
    args = parser.parse_args()

    if args.prune_trajectories is not None:
        cmd_prune_trajectories(args.prune_trajectories)
        return

    if args.init:
        cmd_init()
    elif args.doctor:
        cmd_doctor()
    elif args.trajectory is not None:
        cmd_trajectory(args.trajectory)
    elif args.task:
        try:
            asyncio.run(cmd_task(args.task, timeout_s=args.timeout))
        except KeyboardInterrupt:
            sys.stderr.write("\nInterrupted.\n")
            sys.exit(1)
    elif args.serve:
        cmd_serve(args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
