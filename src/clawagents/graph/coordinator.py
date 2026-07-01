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
from typing import Any, Optional, Protocol

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


class WorkerBackend(Protocol):
    """Backend capable of executing a coordinator worker task."""

    async def run(
        self,
        worker_task: WorkerTask,
        llm: Any,
        tools: Any,
        context_window: int,
    ) -> WorkerTask: ...


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


class ForkedAgentWorkerBackend:
    """Default worker backend that uses the in-process forked-agent runner."""

    async def run(
        self,
        worker_task: WorkerTask,
        llm: Any,
        tools: Any,
        context_window: int,
    ) -> WorkerTask:
        return await _run_worker(worker_task, llm, tools, context_window)


class SubprocessWorkerBackend:
    """Headless worker backend using a small JSON-over-stdin protocol.

    The child process receives:
    ``{"id", "prompt", "tools", "context_window"}``
    and may return JSON with ``status`` and ``result`` fields. Plain stdout is
    treated as a successful result for lightweight scripts.
    """

    def __init__(
        self,
        command: list[str],
        *,
        timeout_s: float = 120.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("SubprocessWorkerBackend requires a command")
        self.command = list(command)
        self.timeout_s = timeout_s
        self.cwd = cwd
        self.env = env

    async def run(
        self,
        worker_task: WorkerTask,
        llm: Any,
        tools: Any,
        context_window: int,
    ) -> WorkerTask:
        t0 = time.monotonic()
        payload = {
            "id": worker_task.id,
            "prompt": worker_task.prompt,
            "tools": worker_task.tools,
            "context_window": context_window,
        }
        worker_task.status = "running"
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
            )
            input_bytes = json.dumps(payload).encode("utf-8")
            stdout, stderr = await asyncio.wait_for(proc.communicate(input_bytes), self.timeout_s)
            out_text = stdout.decode("utf-8", errors="replace").strip()
            err_text = stderr.decode("utf-8", errors="replace").strip()
            data: dict[str, Any]
            try:
                data = json.loads(out_text) if out_text else {}
            except json.JSONDecodeError:
                data = {"status": "done" if proc.returncode == 0 else "error", "result": out_text}
            status = str(data.get("status") or ("done" if proc.returncode == 0 else "error"))
            worker_task.status = "done" if status == "done" else "error"
            result = data.get("result")
            worker_task.result = str(result if result is not None else out_text or err_text)
            if proc.returncode not in (0, None) and worker_task.status == "done":
                worker_task.status = "error"
        except asyncio.TimeoutError:
            worker_task.status = "error"
            worker_task.result = f"Worker subprocess timed out after {self.timeout_s:.1f}s"
            try:
                if proc is not None:
                    proc.kill()
            except Exception:
                pass
        except Exception as exc:
            worker_task.status = "error"
            worker_task.result = f"Worker subprocess error: {exc}"
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
    worker_backend: WorkerBackend | None = None,
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
    backend = worker_backend or ForkedAgentWorkerBackend()
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
            # ``tasks`` is LLM-controlled: it may arrive as a single object, a
            # list of plain strings, or a list of task objects. Normalize before
            # indexing so a malformed shape doesn't crash the orchestration with
            # ``AttributeError``/``TypeError`` instead of feeding the error back.
            if isinstance(tasks, dict):
                tasks = [tasks]
            norm_tasks: list[dict] = []
            if isinstance(tasks, list):
                for t in tasks[:max_workers]:
                    if isinstance(t, str):
                        norm_tasks.append({"prompt": t})
                    elif isinstance(t, dict):
                        norm_tasks.append(t)
                    # Non-str/non-dict entries are ignored.
            if not norm_tasks:
                messages.append(LLMMessage(
                    role="user",
                    content="[System] No valid tasks were specified. Provide a list of task objects (each with a 'prompt') to delegate, or complete the task.",
                ))
                continue

            # Create worker tasks
            worker_tasks = []
            for t in norm_tasks:
                wt = WorkerTask(
                    id=str(t.get("id") or f"task_{len(state.workers) + 1}"),
                    prompt=str(t.get("prompt", "")),
                    tools=t.get("tools", []) if isinstance(t.get("tools"), list) else [],
                    status="running",
                )
                state.workers.append(wt)
                worker_tasks.append(wt)

            emit("context", {
                "message": f"Coordinator delegating {len(worker_tasks)} tasks: {[t.id for t in worker_tasks]}"
            })

            # Execute workers concurrently
            await asyncio.gather(*[
                backend.run(wt, llm, tools, context_window)
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
