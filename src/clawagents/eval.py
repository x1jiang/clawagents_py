from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol


Message = dict[str, str]


class TextEnvironment(Protocol):
    async def init(self) -> Mapping[str, Any]:
        ...

    async def step(self, action: str) -> Mapping[str, Any]:
        ...

    def get_metrics(self) -> Mapping[str, Any]:
        ...


Responder = Callable[[list[Message]], str | Awaitable[str]]
AgentEnvironment = TextEnvironment
AgentResponder = Responder


@dataclass(frozen=True)
class TextEvaluationStep:
    action: str
    observations: list[Message]
    reward: float
    done: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TextEvaluationResult:
    steps: list[TextEvaluationStep]
    total_reward: float
    metrics: dict[str, Any]
    metadata: dict[str, Any]


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def run_text_environment(
    responder: Responder,
    env: Any,
    *,
    max_turns: int = 20,
) -> TextEvaluationResult:
    initial = await _maybe_await(env.init())
    messages = list(initial.get("observations") or [])
    steps: list[TextEvaluationStep] = []
    total_reward = 0.0

    try:
        for _turn in range(max(1, max_turns)):
            action = await _maybe_await(responder(messages))
            step = await _maybe_await(env.step(str(action)))
            reward = float(step.get("reward", 0.0))
            total_reward += reward
            observations = list(step.get("observations") or [])
            steps.append(TextEvaluationStep(
                action=str(step.get("postprocessed_action") or action),
                observations=observations,
                reward=reward,
                done=bool(step.get("done", False)),
                metadata=dict(step.get("metadata") or {}),
            ))
            messages = [
                *messages,
                {"role": "assistant", "content": str(action)},
                *observations,
            ]
            if bool(step.get("done", False)):
                break
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            await _maybe_await(close())

    get_metrics = getattr(env, "get_metrics", None)
    metrics = await _maybe_await(get_metrics()) if get_metrics is not None else {}
    return TextEvaluationResult(
        steps=steps,
        total_reward=total_reward,
        metrics=dict(metrics or {}),
        metadata=dict(initial.get("metadata") or {}),
    )


async def run_agent_environment(
    responder: AgentResponder,
    env: Any,
    *,
    max_turns: int = 20,
) -> TextEvaluationResult:
    return await run_text_environment(responder, env, max_turns=max_turns)
