"""Application-verified terminal action for backend coordination jobs."""
import inspect
from typing import Any, Callable

from clawagents.tools.registry import ToolResult


class FinishCoordinationTool:
    name = 'finish_coordination'
    keywords = ['coordination', 'finish', 'acceptance']
    description = ('Finish the job after all required work is complete. Call this tool alone. '
                   'The application independently checks acceptance; a failed check keeps the job open. '
                   'Do not repeat already successful work; call this once the requested artifacts are ready.')
    parameters: dict[str, dict[str, Any]] = {'summary': {'type':'string','required':True,'description':'Concise final result and evidence.'}}
    cacheable = False

    def __init__(self, check: Callable[[], Any]):
        if not callable(check):
            raise TypeError('completion_check must be callable')
        self._check = check

    async def execute(self, args):
        summary = str(args.get('summary') or '').strip()
        if not summary:
            return ToolResult(False,'',error='Provide a final summary')
        accepted = self._check()
        if inspect.isawaitable(accepted):
            accepted = await accepted
        if accepted is not True:
            return ToolResult(False,'',error='Acceptance checks have not passed. Complete or repair the outstanding work before finishing.')
        return ToolResult(True,summary,return_direct=True)
