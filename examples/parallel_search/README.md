# Web research with Parallel Search MCP

Use ClawAgents' existing `MCPServerStreamableHttp` client to search for sources
and read a known page with [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).
The anonymous endpoint provides free web search in Fast mode and page fetching,
with no account or Parallel API key required. Free access has lower rate limits
and is intended for exploration and light use.

This recipe calls the MCP tools directly, so it needs no LLM configuration. It
does not register tools on an agent or change the built-in Tavily-backed
`web_search` and existing `web_fetch` tools.

## Install and run

Use Python 3.10 or newer in a fresh directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'clawagents[mcp]'
```

On Windows, activate with `.venv\Scripts\activate` instead. Save the following as
`parallel_research.py`. It discovers the available tools, searches for Python
TaskGroup documentation, then fetches excerpts from the Python asyncio reference.

```python
import asyncio
from uuid import uuid4

from clawagents import MCPServerStreamableHttp


def print_result(result):
    for block in result.content:
        if block.type == "text":
            print(block.text)
    if result.isError:
        raise RuntimeError("MCP tool failed; see the response above.")


async def main():
    session_id = str(uuid4())
    queries = ["Python asyncio TaskGroup cancellation documentation"]
    async with MCPServerStreamableHttp(
        name="parallel-search",
        params={
            "url": "https://search.parallel.ai/mcp",
            "headers": {"User-Agent": "clawagents-parallel-mcp-example"},
            "timeout": 60.0,
            "sse_read_timeout": 60.0,
        },
        client_session_timeout_seconds=60.0,
    ) as server:
        tools = await server.list_tools()
        print("Available tools:", ", ".join(tool.name for tool in tools))
        print_result(await server.invoke_tool("web_search", {
            "objective": "Find official Python guidance on TaskGroup cancellation.",
            "search_queries": queries,
            "session_id": session_id,
        }))
        print_result(await server.invoke_tool("web_fetch", {
            "urls": ["https://docs.python.org/3/library/asyncio-task.html"],
            "objective": "Explain TaskGroup cancellation and exception handling.",
            "search_queries": queries,
            "session_id": session_id,
        }))


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python parallel_research.py
```

The output includes source URLs and relevant excerpts. To research another topic,
change the objective and queries, then set `urls` to the pages you want to read.
Keep the same `session_id` across related search and fetch calls; create a new one
for each new research session. The script prints tool errors and stops rather
than treating them as research results. If you reach the free-tier limit, wait
before retrying; do not change the session identifier to bypass it.

The context manager closes the client when the script finishes or raises an
error. See the [MCP server abstractions](../../src/clawagents/mcp/server.py) for
other client options.
