"""Example: Google Gemini"""
import asyncio
from clawagents import create_claw_agent


async def main():
    agent = create_claw_agent(
        "gemini-3-flash",
        api_key="AIza...",       # or set GEMINI_API_KEY in .env
    )
    result = await agent.invoke("Read README.md and suggest improvements")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
