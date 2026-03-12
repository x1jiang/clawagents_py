"""Example: Local model with Ollama

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.1
    3. Ollama runs on http://localhost:11434 by default
"""
import asyncio
from clawagents import create_claw_agent


async def main():
    agent = create_claw_agent(
        "llama3.1",                                  # model name in Ollama
        base_url="http://localhost:11434/v1",        # Ollama's OpenAI-compatible endpoint
        # No api_key needed for local models
    )
    result = await agent.invoke("List all files in the current directory")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
