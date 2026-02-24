import sys
import argparse
import asyncio
from pathlib import Path

from clawagents.config.config import load_config, get_default_model
from clawagents.providers.llm import create_provider
from clawagents.tools.filesystem import filesystem_tools
from clawagents.tools.exec import exec_tools
from clawagents.tools.skills import SkillStore, create_skill_tools
from clawagents.gateway.server import start_gateway

from clawagents.agent import create_claw_agent

async def load_all_tools():
    tools = []
    tools.extend(filesystem_tools)
    tools.extend(exec_tools)

    skill_store = SkillStore()
    cwd = Path.cwd()
    skill_store.add_directory(cwd / "skills")
    skill_store.add_directory(cwd.parent / "openclaw-main" / "skills")
    
    await skill_store.load_all()
    skills = skill_store.list()
    if skills:
        sys.stderr.write(f"{len(skills)} skills loaded\n")
    tools.extend(create_skill_tools(skill_store))
    return tools

async def async_main():
    parser = argparse.ArgumentParser(description="ClawAgents Python Engine")
    parser.add_argument("--task", type=str, help="Run a single task from CLI")
    parser.add_argument("--port", type=int, default=3000, help="Port to run the gateway on")
    args = parser.parse_args()

    config = load_config()
    active_model = get_default_model(config)
    llm = create_provider(active_model, config)
    tools = await load_all_tools()
    
    agent = create_claw_agent(model=llm, tools=tools, streaming=config.streaming)
    tool_count = len(agent.tools.list())

    sys.stderr.write(f"ClawAgents | {active_model} | {tool_count} tools\n")

    if args.task:
        await agent.invoke(args.task)
        return

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, start_gateway, args.port)

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Fatal: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
