"""Interactive Tools — ask_user

Allows the agent to ask the user a question and wait for a response.
Reads from stdin in CLI mode.
"""

import asyncio
import sys
from typing import Any, Dict, List

from clawagents.tools.registry import Tool, ToolResult


class AskUserTool:
    name = "ask_user"
    description = (
        "Ask the user a question and wait for their response. "
        "Use when you need clarification, confirmation, or input to proceed. "
        "Only use this when the task is genuinely ambiguous \u2014 don't over-ask."
    )
    parameters = {
        "question": {"type": "string", "description": "The question to ask the user", "required": True},
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        question = str(args.get("question", ""))
        if not question:
            return ToolResult(success=False, output="", error="No question provided")

        loop = asyncio.get_running_loop()
        try:
            def _ask():
                sys.stderr.write(f"\n\U0001f99e Agent asks: {question}\n> ")
                sys.stderr.flush()
                return input()

            answer = await loop.run_in_executor(None, _ask)
            return ToolResult(success=True, output=f"User response: {answer}")
        except EOFError:
            return ToolResult(success=False, output="", error="No user input available (non-interactive mode)")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"ask_user failed: {str(e)}")


interactive_tools: List[Tool] = [AskUserTool()]
