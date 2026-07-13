"""Example: Amazon Bedrock — native IAM (HIPAA) and OpenAI-compatible gateway.

Native path (recommended for HIPAA / BAA workloads):
    pip install 'clawagents[bedrock]'
    # AWS credentials via env, ~/.aws/credentials, or instance/task role
    export AWS_REGION=us-east-1
    # Enable Claude model access in the Bedrock console for your account/region

Gateway path (Bedrock Access Gateway / LiteLLM):
    Gateway handles AWS auth and exposes an OpenAI-compatible /v1 API.
"""
import asyncio
import os

from clawagents import create_claw_agent


async def native_bedrock():
    """Claude on Bedrock via AsyncAnthropicBedrock (no Anthropic API key)."""
    agent = create_claw_agent(
        # Cross-region inference profile ID (or foundation model ID)
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # Optional: create_claw_agent(profile="bedrock")
    )
    result = await agent.invoke("Say hello in one short sentence.")
    print(result.result)


async def nova_via_converse():
    """Amazon Nova (and other non-Claude Bedrock models) via Converse API."""
    agent = create_claw_agent("amazon.nova-pro-v1:0")
    result = await agent.invoke("Say hello in one short sentence.")
    print(result.result)


async def bedrock_gateway():
    """OpenAI-compatible proxy (BAG / LiteLLM)."""
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1")
    agent = create_claw_agent(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        base_url=base,
        api_key=os.environ.get("OPENAI_API_KEY", "bedrock"),
    )
    result = await agent.invoke("Say hello in one short sentence.")
    print(result.result)


async def main():
    mode = (os.environ.get("BEDROCK_EXAMPLE") or "native").lower()
    if mode in ("gateway", "bag", "litellm"):
        await bedrock_gateway()
    elif mode in ("nova", "converse"):
        await nova_via_converse()
    else:
        await native_bedrock()


if __name__ == "__main__":
    asyncio.run(main())
