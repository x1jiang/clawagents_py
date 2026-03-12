"""Example: AWS Bedrock via OpenAI-compatible gateway

Prerequisites:
    1. Set up Bedrock Access Gateway: https://github.com/aws-samples/bedrock-access-gateway
       Or use LiteLLM proxy: pip install litellm && litellm --model bedrock/anthropic.claude-3-sonnet
    2. The gateway handles AWS credentials (IAM role, access key, etc.)
    3. Gateway exposes OpenAI-compatible API at http://localhost:8080
"""
import asyncio
from clawagents import create_claw_agent


async def main():
    agent = create_claw_agent(
        "anthropic.claude-3-sonnet-20240229-v1:0",  # Bedrock model ID
        base_url="http://localhost:8080/v1",         # gateway endpoint
        api_key="bedrock",                           # gateway handles real auth
    )
    result = await agent.invoke("Review the codebase and suggest improvements")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
