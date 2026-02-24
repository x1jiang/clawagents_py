import os
import time
import asyncio
from pathlib import Path

from clawagents.config.config import load_config, get_default_model
from clawagents.providers.llm import create_provider
from clawagents.tools.registry import ToolRegistry
from clawagents.tools.filesystem import filesystem_tools
from clawagents.tools.exec import exec_tools
from clawagents.tools.skills import SkillStore, create_skill_tools
from clawagents.agent import create_claw_agent

TASK = """Do the following multi-step task:
1. List the files in the current directory
2. Read pyproject.toml and extract the project name and all dependency names
3. Write a file called python-project-report.md that contains a markdown summary with: project name, and list of dependencies
Tell me when done and show the contents of the report."""

async def initialize_tools() -> list:
    tools = []
    tools.extend(filesystem_tools)
    tools.extend(exec_tools)
    
    skill_store = SkillStore()
    cwd = Path.cwd()
    skill_store.add_directory(cwd / "skills")
    skill_store.add_directory(cwd.parent / "openclaw-main" / "skills")
    await skill_store.load_all()
    
    tools.extend(create_skill_tools(skill_store))
        
    return tools


async def run_benchmark():
    config = load_config()
    active_model = get_default_model(config)
    llm = create_provider(active_model, config)
    streaming = config.streaming

    print(f"\n🦞 ClawAgents Benchmark (Python Port)")
    print(f"   Provider: {llm.name} | Model: {active_model} | Streaming: {streaming}")
    print(f"   Task: {TASK}\n")

    start_init = time.time()
    tools = await initialize_tools()
    
    agent = create_claw_agent(model=llm, tools=tools, streaming=streaming)
    
    init_time = time.time() - start_init
    print(f"   Init time: {init_time:.2f}s")

    # Run
    start_task = time.time()
    result = await agent.invoke(TASK)
    task_time = time.time() - start_task

    total_time = init_time + task_time
    print(f"\n━━━ ClawAgents_Py Result ━━━")
    print(result.result[:500])
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\n⏱  Init: {init_time:.2f}s | Task: {task_time:.2f}s | Total: {total_time:.2f}s")
    print(f"   Iterations: {result.iterations} | Tool calls: {result.tool_calls}")

    return {
        "engine": "clawagents_py",
        "init_time": init_time,
        "task_time": task_time,
        "total_time": total_time,
        "iterations": result.iterations,
        "tool_calls": result.tool_calls,
    }

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark())
    except Exception as e:
        print(f"\n❌ ClawAgents benchmark failed: {e}")
        import traceback
        traceback.print_exc()
