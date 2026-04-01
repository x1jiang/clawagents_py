"""Coordinator/Swarm Orchestration Mode (learned from Claude Code: coordinatorMode.ts).

Implements a two-tier execution model:
  - Coordinator: Plans and delegates, sees all results, synthesizes final answer.
               Has NO direct tool access (no filesystem, no execute).
  - Workers: Execute specific tasks with full tool access but limited context.

The coordinator communicates with workers via structured task notifications
and receives results back for synthesis.

Usage:
    from clawagents.graph.coordinator import run_coordinator

    result = await run_coordinator(
        task="Refactor the auth module to use JWT tokens",
        llm=llm,
        tools=registry,
        max_workers=3,
    )

Controlled by: CLAW_FEATURE_COORDINATOR=1
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Coordinator System Prompt ─────────────────────────────────────────────

COORDINATOR_SYSTEM_PROMPT = """\
You are a Coordinator Agent. You plan and delegate tasks to Worker agents.

## Your Role
- Analyze the user's request and break it down into sub-tasks
- Delegate each sub-task to a Worker agent
- Synthesize Worker results into a final answer
- You do NOT have direct tool access (no filesystem, no execute)

## Communication Protocol
To delegate a task to a Worker, respond with:
```json
{"action": "delegate", "tasks": [
  {"id": "task_1", "prompt": "Detailed sub-task description", "tools": ["read_file", "grep"]},
  {"id": "task_2", "prompt": "Another sub-task", "tools": ["execute", "write_file"]}
]}
```

To provide the final synthesized answer:
```json
{"action": "complete", "result": "Your final answer here"}
```

## Rules
1. Break complex tasks into 2-5 independent sub-tasks
2. Each sub-task should be self-contained with clear success criteria
3. Specify which tools each Worker needs
4. After receiving all Worker results, synthesize and provide the final answer
5. If a Worker fails, you may retry with a modified prompt or work around it

## Worker Results
Worker results will be provided in this format:
[Worker Result: task_id]
Status: success/error
Result: <worker output>
"""


@dataclass
class WorkerTask:
    """A task delegated to a worker agent."""
    id: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, done, error
    result: str = ""
    duration_s: float = 0.0


@dataclass
class CoordinatorState:
    """State of the coordinator orchestration."""
    task: str
    workers: list[WorkerTask] = field(default_factory=list)
    max_workers: int = 3
    rounds: int = 0
    max_rounds: int = 10
    status: str = "running"
    final_result: str = ""


async def _run_worker(
    worker_task: WorkerTask,
    llm: Any,
    tools: Any,
    context_window: int,
) -> WorkerTask:
    """Execute a single worker task using the forked agent pattern."""
    from clawagents.graph.forked_agent import run_forked_agent

    t0 = time.monotonic()
    try:
        state = await run_forked_agent(
            fork_prompt=worker_task.prompt,
            llm=llm,
            tools=tools,
            allowed_tools=worker_task.tools if worker_task.tools else None,
            max_turns=8,
            context_window=context_window,
        )
        worker_task.status = "done" if state.status == "done" else "error"
        worker_task.result = state.result
    except Exception as exc:
        worker_task.status = "error"
        worker_task.result = f"Worker error: {exc}"
    finally:
        worker_task.duration_s = time.monotonic() - t0

    return worker_task


def _parse_coordinator_response(content: str) -> dict[str, Any]:
    """Parse the coordinator's JSON response."""
    content = content.strip()

    # Try direct parse
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from code fences
    import re
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    return {"action": "complete", "result": content}


async def run_coordinator(
    task: str,
    llm: Any,
    tools: Any = None,
    max_workers: int = 3,
    max_rounds: int = 10,
    context_window: int = 200_000,
    on_event: Any = None,
) -> CoordinatorState:
    """Run the coordinator/swarm orchestration loop.

    Args:
        task: The user's task to accomplish
        llm: LLM provider instance
        tools: Tool registry (passed to workers, not available to coordinator)
        max_workers: Maximum concurrent workers
        max_rounds: Maximum coordinator-worker round trips
        context_window: Context window for worker agents
        on_event: Event callback

    Returns:
        CoordinatorState with the final result
    """
    from clawagents.config.features import is_enabled
    if not is_enabled("coordinator"):
        raise RuntimeError("Coordinator mode is not enabled. Set CLAW_FEATURE_COORDINATOR=1")

    from clawagents.providers.llm import LLMMessage

    emit = on_event or (lambda *a, **kw: None)
    state = CoordinatorState(task=task, max_workers=max_workers, max_rounds=max_rounds)

    # Build coordinator conversation
    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=COORDINATOR_SYSTEM_PROMPT),
        LLMMessage(role="user", content=task),
    ]

    for round_idx in range(max_rounds):
        state.rounds = round_idx + 1

        # Get coordinator's plan
        try:
            response = await llm.chat(messages)
        except Exception as exc:
            state.status = "error"
            state.final_result = f"Coordinator LLM error: {exc}"
            break

        messages.append(LLMMessage(role="assistant", content=response.content))

        # Parse coordinator response
        parsed = _parse_coordinator_response(response.content)
        action = parsed.get("action", "complete")

        if action == "complete":
            state.status = "done"
            state.final_result = parsed.get("result", response.content)
            emit("agent_done", {
                "message": f"Coordinator completed in {state.rounds} rounds with {len(state.workers)} workers"
            })
            break

        elif action == "delegate":
            tasks = parsed.get("tasks", [])
            if not tasks:
                messages.append(LLMMessage(
                    role="user",
                    content="[System] No tasks were specified. Please provide tasks to delegate or complete the task.",
                ))
                continue

            # Create worker tasks
            worker_tasks = []
            for t in tasks[:max_workers]:
                wt = WorkerTask(
                    id=t.get("id", f"task_{len(state.workers) + 1}"),
                    prompt=t.get("prompt", ""),
                    tools=t.get("tools", []),
                    status="running",
                )
                state.workers.append(wt)
                worker_tasks.append(wt)

            emit("context", {
                "message": f"Coordinator delegating {len(worker_tasks)} tasks: {[t.id for t in worker_tasks]}"
            })

            # Execute workers concurrently
            await asyncio.gather(*[
                _run_worker(wt, llm, tools, context_window)
                for wt in worker_tasks
            ])

            # Feed results back to coordinator
            results_text = []
            for wt in worker_tasks:
                results_text.append(
                    f"[Worker Result: {wt.id}]\n"
                    f"Status: {wt.status}\n"
                    f"Duration: {wt.duration_s:.1f}s\n"
                    f"Result: {wt.result[:2000]}"
                )
                emit("tool_result", {
                    "name": f"worker:{wt.id}",
                    "success": wt.status == "done",
                    "preview": wt.result[:120],
                })

            messages.append(LLMMessage(
                role="user",
                content="## Worker Results\n\n" + "\n\n".join(results_text),
            ))

        else:
            # Unknown action — treat as final answer
            state.status = "done"
            state.final_result = response.content
            break

    if state.status == "running":
        state.status = "error"
        state.final_result = f"Coordinator exceeded {max_rounds} rounds without completing."

    return state
