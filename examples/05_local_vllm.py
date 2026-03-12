"""Example: Local model with vLLM

Prerequisites:
    1. Install vLLM: pip install vllm
    2. Start server: vllm serve Qwen/Qwen3-8B --port 8000
    3. Server runs OpenAI-compatible API at http://localhost:8000
"""
import asyncio
from clawagents import create_claw_agent


async def main():
    agent = create_claw_agent(
        "Qwen/Qwen3-8B",                            # model name in vLLM
        base_url="http://localhost:8000/v1",
        # Qwen3 models emit <think> tags — automatically handled by Feature H
    )
    result = await agent.invoke("Write a Python function to calculate fibonacci numbers")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
