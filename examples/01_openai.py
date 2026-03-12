"""Example: OpenAI (GPT-5, GPT-4o, etc.)"""
import asyncio
from clawagents import create_claw_agent


async def main():
    agent = create_claw_agent(
        "gpt-5-mini",
        api_key="sk-...",        # or set OPENAI_API_KEY in .env
        learn=True,              # enable PTRL (optional)
        rethink=True,            # enable rethink on failures (optional)
    )
    result = await agent.invoke("List all Python files and summarize the project structure")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
