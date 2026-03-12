"""Example: Azure OpenAI"""
import asyncio
from clawagents import create_claw_agent


async def main():
    agent = create_claw_agent(
        "gpt-4o",                                             # Azure deployment name
        api_key="your-azure-key",                             # or OPENAI_API_KEY in .env
        base_url="https://YOUR_RESOURCE.openai.azure.com/",   # or OPENAI_BASE_URL in .env
        api_version="2024-12-01-preview",                     # or OPENAI_API_VERSION in .env
    )
    result = await agent.invoke("Analyze the codebase for security issues")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
