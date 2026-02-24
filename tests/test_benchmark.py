"""
Pytest-based performance benchmarks for clawagents_py.

Run from clawagents_py/: pytest tests/test_benchmark.py -v -s
Requires API keys (OPENAI_API_KEY or GEMINI_API_KEY) and network.
Marked as benchmark; skip with: pytest -m "not benchmark"
"""

import os
import time
import pytest
from pathlib import Path

from clawagents.config.config import load_config, get_default_model
from clawagents.providers.llm import create_provider
from clawagents.tools.registry import ToolRegistry
from clawagents.tools.filesystem import filesystem_tools
from clawagents.tools.exec import exec_tools
from clawagents.agent import create_claw_agent


BENCHMARK_TASKS = [
    "Read pyproject.toml and tell me the version number.",
    "Use the execute tool to run `echo benchmark` and return the output.",
]


@pytest.fixture
def agent():
    """Create agent with streaming enabled (default)."""
    config = load_config()
    llm = create_provider(get_default_model(config), config)
    tools = filesystem_tools + exec_tools
    return create_claw_agent(model=llm, tools=tools, streaming=True)


@pytest.mark.benchmark
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"),
    reason="API key required for benchmark",
)
@pytest.mark.asyncio
async def test_benchmark_simple_task(agent):
    """Benchmark a simple read task in streaming mode."""
    task = "Read pyproject.toml and tell me the version number."
    start = time.time()
    result = await agent.invoke(task, max_iterations=2)
    elapsed = time.time() - start
    assert result.status in ("done", "error")
    assert elapsed > 0
    # Log metrics for CI/reports
    print(f"\n  [benchmark] task_time={elapsed:.2f}s iterations={result.iterations} tool_calls={result.tool_calls}")


@pytest.mark.benchmark
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"),
    reason="API key required for benchmark",
)
@pytest.mark.asyncio
async def test_benchmark_streaming_is_default():
    """Verify streaming is the default mode."""
    config = load_config()
    assert config.streaming is True
