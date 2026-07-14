"""Minimal ATLAS × ClawAgents smoke example.

Prereqs:
  pip install -e '.[atlas]'
  export OPENAI_API_KEY=...   # task model
  # atlas_model in atlas.json also needs a provider key for learning calls

Usage (from clawagents_py/):
  CLAW_ATLAS=1 CLAW_ATLAS_CONFIG=examples/atlas_basic/atlas.json \\
    python examples/atlas_basic/run.py
"""

from __future__ import annotations

import asyncio
import os


async def main() -> None:
    from clawagents import create_claw_agent

    config = os.environ.get(
        "CLAW_ATLAS_CONFIG",
        os.path.join(os.path.dirname(__file__), "atlas.json"),
    )
    agent = create_claw_agent(
        atlas=True,
        atlas_config=config,
        rethink=True,
    )
    state = await agent.invoke(
        "List the files in the current directory, then briefly summarize what you see."
    )
    print(state.result)
    if state.trajectory_file:
        print(f"trajectory: {state.trajectory_file}")


if __name__ == "__main__":
    asyncio.run(main())
