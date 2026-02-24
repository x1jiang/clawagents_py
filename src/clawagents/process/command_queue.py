import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
from dataclasses import dataclass, field

from clawagents.logging.diagnostic import diagnostic_logger as diag, log_lane_dequeue, log_lane_enqueue
from clawagents.process.lanes import CommandLane

class CommandLaneClearedError(Exception):
    def __init__(self, lane: Optional[str] = None):
        super().__init__(f"Command lane \"{lane}\" cleared" if lane else "Command lane cleared")


@dataclass
class QueueEntry:
    task: Callable[[], Coroutine[Any, Any, Any]]
    future: asyncio.Future
    enqueued_at: float
    warn_after_ms: int
    on_wait: Optional[Callable[[float, int], None]] = None


@dataclass
class LaneState:
    lane: str
    queue: List[QueueEntry] = field(default_factory=list)
    active_task_ids: Set[int] = field(default_factory=set)
    max_concurrent: int = 1
    draining: bool = False
    generation: int = 0


lanes: Dict[str, LaneState] = {}
next_task_id = 1


def get_lane_state(lane: str) -> LaneState:
    if lane not in lanes:
        lanes[lane] = LaneState(lane=lane)
    return lanes[lane]


def complete_task(state: LaneState, task_id: int, task_generation: int) -> bool:
    if task_generation != state.generation:
        return False
    state.active_task_ids.discard(task_id)
    return True


def drain_lane(lane: str):
    state = get_lane_state(lane)
    if state.draining:
        return
    state.draining = True

    def pump():
        global next_task_id
        while len(state.active_task_ids) < state.max_concurrent and state.queue:
            entry = state.queue.pop(0)
            waited_ms = (time.time() - entry.enqueued_at) * 1000

            if waited_ms >= entry.warn_after_ms:
                if entry.on_wait:
                    entry.on_wait(waited_ms, len(state.queue))
                diag.warn(f"lane wait exceeded: lane={lane} waitedMs={int(waited_ms)} queueAhead={len(state.queue)}")
            
            log_lane_dequeue(lane, waited_ms, len(state.queue))
            
            task_id = next_task_id
            next_task_id += 1
            task_generation = state.generation
            state.active_task_ids.add(task_id)

            async def _run_task(entry=entry, task_id=task_id, task_generation=task_generation):
                start_time = time.time()
                try:
                    result = await entry.task()
                    completed_current = complete_task(state, task_id, task_generation)
                    if completed_current:
                        diag.debug(f"lane task done: lane={lane} durationMs={int((time.time() - start_time)*1000)} active={len(state.active_task_ids)} queued={len(state.queue)}")
                        pump()
                    if not entry.future.done():
                        entry.future.set_result(result)
                except Exception as err:
                    completed_current = complete_task(state, task_id, task_generation)
                    is_probe_lane = lane.startswith("auth-probe:") or lane.startswith("session:probe-")
                    if not is_probe_lane:
                        diag.error(f"lane task error: lane={lane} durationMs={int((time.time() - start_time)*1000)} error=\"{str(err)}\"")
                    if completed_current:
                        pump()
                    if not entry.future.done():
                        entry.future.set_exception(err)

            # Fire and forget the background execution
            asyncio.create_task(_run_task())

        state.draining = False

    pump()


def set_command_lane_concurrency(lane: str, max_concurrent: int):
    cleaned = lane.strip() or CommandLane.Main.value
    state = get_lane_state(cleaned)
    state.max_concurrent = max(1, int(max_concurrent))
    drain_lane(cleaned)


async def enqueue_command_in_lane(
    lane: str,
    task: Callable[[], Coroutine[Any, Any, Any]],
    warn_after_ms: int = 2000,
    on_wait: Optional[Callable[[float, int], None]] = None
) -> Any:
    cleaned = lane.strip() or CommandLane.Main.value
    state = get_lane_state(cleaned)
    
    # We must use the current event loop
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    state.queue.append(QueueEntry(
        task=task,
        future=future,
        enqueued_at=time.time(),
        warn_after_ms=warn_after_ms,
        on_wait=on_wait
    ))
    
    log_lane_enqueue(cleaned, len(state.queue) + len(state.active_task_ids))
    drain_lane(cleaned)
    
    return await future


async def enqueue_command(
    task: Callable[[], Coroutine[Any, Any, Any]],
    warn_after_ms: int = 2000,
    on_wait: Optional[Callable[[float, int], None]] = None
) -> Any:
    return await enqueue_command_in_lane(CommandLane.Main.value, task, warn_after_ms, on_wait)


def get_queue_size(lane: str = CommandLane.Main.value) -> int:
    resolved = lane.strip() or CommandLane.Main.value
    state = lanes.get(resolved)
    if not state:
        return 0
    return len(state.queue) + len(state.active_task_ids)


def get_total_queue_size() -> int:
    return sum(len(s.queue) + len(s.active_task_ids) for s in lanes.values())


def clear_command_lane(lane: str = CommandLane.Main.value) -> int:
    cleaned = lane.strip() or CommandLane.Main.value
    state = lanes.get(cleaned)
    if not state:
        return 0
    
    removed = len(state.queue)
    pending = state.queue[:]
    state.queue.clear()
    
    for entry in pending:
        if not entry.future.done():
            entry.future.set_exception(CommandLaneClearedError(cleaned))
            
    return removed


def reset_all_lanes():
    lanes_to_drain: List[str] = []
    for state in lanes.values():
        state.generation += 1
        state.active_task_ids.clear()
        state.draining = False
        if state.queue:
            lanes_to_drain.append(state.lane)
            
    for lane in lanes_to_drain:
        drain_lane(lane)


def get_active_task_count() -> int:
    return sum(len(s.active_task_ids) for s in lanes.values())


async def wait_for_active_tasks(timeout_ms: int) -> Dict[str, bool]:
    poll_interval_s = 0.05
    deadline = time.time() + (timeout_ms / 1000.0)
    
    active_at_start: Set[int] = set()
    for state in lanes.values():
        active_at_start.update(state.active_task_ids)

    while True:
        if not active_at_start:
            return {"drained": True}
            
        has_pending = False
        for state in lanes.values():
            if any(t in active_at_start for t in state.active_task_ids):
                has_pending = True
                break
                
        if not has_pending:
            return {"drained": True}
            
        if time.time() >= deadline:
            return {"drained": False}
            
        await asyncio.sleep(poll_interval_s)
