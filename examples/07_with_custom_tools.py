"""Example: Adding custom tools to the agent"""
import asyncio
from clawagents import create_claw_agent


class DatabaseQueryTool:
    """Custom tool example — the agent can call this to query a database."""
    name = "query_db"
    description = "Execute a SQL query against the project database. Args: {query: string}"
    parameters = {"query": {"type": "string", "description": "SQL query to execute"}}

    async def execute(self, args: dict) -> dict:
        query = args.get("query", "")
        # Replace with your actual database logic
        return {"success": True, "output": f"Executed: {query}\nRows: 42"}


async def main():
    agent = create_claw_agent(
        "gpt-5-mini",
        tools=[DatabaseQueryTool()],     # your custom tools alongside built-ins
        instruction="You have access to a project database. Use query_db to answer data questions.",
    )
    result = await agent.invoke("How many users signed up last month?")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
