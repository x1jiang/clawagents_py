"""Interactive Tools — ask_user

Allows the agent to ask the user a question and wait for a response.
Defaults to stdin (CLI). Hosts can inject ``ask_fn`` for GUI / webview HITL.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable, Dict, List, Optional

from clawagents.tools.registry import Tool, ToolResult

# Sync callback: question -> answer text, or None if the user skipped / timed out.
AskFn = Callable[[str], Optional[str]]


class AskUserTool:
    name = "ask_user"
    description = (
        "Ask the user a question and wait for their response. "
        "Use when you need clarification, confirmation, or input to proceed. "
        "Only use this when the task is genuinely ambiguous — don't over-ask."
    )
    parameters = {
        "question": {"type": "string", "description": "The question to ask the user", "required": True},
    }

    def __init__(self, ask_fn: AskFn | None = None):
        self._ask_fn = ask_fn

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        question = str(args.get("question", ""))
        if not question:
            return ToolResult(success=False, output="", error="No question provided")

        loop = asyncio.get_running_loop()
        try:
            if self._ask_fn is not None:
                answer = await loop.run_in_executor(None, self._ask_fn, question)
                if answer is None:
                    return ToolResult(
                        success=False,
                        output="",
                        error="User skipped the question (or timed out).",
                    )
                return ToolResult(success=True, output=f"User response: {answer}")

            def _ask() -> str:
                sys.stderr.write(f"\n\U0001f99e Agent asks: {question}\n> ")
                sys.stderr.flush()
                return input()

            answer = await loop.run_in_executor(None, _ask)
            return ToolResult(success=True, output=f"User response: {answer}")
        except EOFError:
            return ToolResult(
                success=False,
                output="",
                error="No user input available (non-interactive mode)",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"ask_user failed: {str(e)}")


interactive_tools: List[Tool] = [AskUserTool()]
