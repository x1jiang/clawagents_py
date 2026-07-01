"""Tests for clawagents.steer."""

from __future__ import annotations

import asyncio

import pytest

from clawagents.run_context import RunContext
from clawagents.steer import (
    NextTurnQueue,
    SteerHook,
    SteerMessage,
    SteerQueue,
    drain_next_turn,
    drain_steer,
    peek_next_turn,
    peek_steer,
    queue_message,
    steer,
)


def test_steer_queue_push_drain_roundtrip():
    q = SteerQueue()
    assert len(q) == 0
    assert not q
    q.push("first")
    q.push(SteerMessage(text="second", role="system"))
    assert len(q) == 2
    out = q.drain()
    assert [m.text for m in out] == ["first", "second"]
    assert out[1].role == "system"
    assert len(q) == 0


def test_steer_queue_extend_iterable():
    q = SteerQueue()
    q.extend(["a", "b", SteerMessage(text="c", role="developer")])
    assert [m.text for m in q.drain()] == ["a", "b", "c"]


def test_steer_queue_peek_does_not_consume():
    q = SteerQueue()
    q.push("hi")
    snap = q.peek()
    assert [m.text for m in snap] == ["hi"]
    assert len(q) == 1
    snap.clear()  # mutating the returned list must not affect the real queue
    assert len(q) == 1
    assert [m.text for m in q.peek()] == ["hi"]


def test_run_context_attach_steer_lazy():
    rc = RunContext()
    steer(rc, "please switch to Python", role="user")
    pending = peek_steer(rc)
    assert len(pending) == 1
    assert pending[0].text == "please switch to Python"
    assert pending[0].role == "user"
    drained = drain_steer(rc)
    assert len(drained) == 1
    assert peek_steer(rc) == []


def test_next_turn_queue_independent_from_steer():
    rc = RunContext()
    steer(rc, "mid-run nudge")
    queue_message(rc, "after-run task")
    assert [m.text for m in peek_steer(rc)] == ["mid-run nudge"]
    assert [m.text for m in peek_next_turn(rc)] == ["after-run task"]
    drain_steer(rc)
    assert peek_steer(rc) == []
    assert [m.text for m in peek_next_turn(rc)] == ["after-run task"]
    drain_next_turn(rc)
    assert peek_next_turn(rc) == []


def test_steer_hook_appends_to_messages_in_place():
    rc = RunContext()
    steer(rc, "be terse")
    steer(rc, "use bullets")
    hook = SteerHook()
    messages: list[dict[str, str]] = [{"role": "user", "content": "hello"}]

    asyncio.run(hook.on_llm_start(rc, model="gpt-5", messages=messages))

    # Injected nudges must be real LLMMessage objects (the loop + every provider
    # read ``.role``/``.content`` attributes, not dict keys).
    assert messages[0] == {"role": "user", "content": "hello"}
    assert (messages[1].role, messages[1].content) == ("user", "[steer] be terse")
    assert (messages[2].role, messages[2].content) == ("user", "[steer] use bullets")
    assert peek_steer(rc) == []  # drained


def test_steer_hook_is_noop_when_queue_empty():
    rc = RunContext()
    hook = SteerHook()
    messages: list[dict[str, str]] = [{"role": "user", "content": "hello"}]
    asyncio.run(hook.on_llm_start(rc, model="gpt-5", messages=messages))
    assert messages == [{"role": "user", "content": "hello"}]


def test_steer_hook_custom_prefix_and_no_prefix():
    rc = RunContext()
    steer(rc, "compress now")
    hook = SteerHook(prefix=None)
    msgs: list = []
    asyncio.run(hook.on_llm_start(rc, model="x", messages=msgs))
    assert (msgs[0].role, msgs[0].content) == ("user", "compress now")

    rc2 = RunContext()
    steer(rc2, "switch language", role="system")
    hook2 = SteerHook(prefix="[op]")
    msgs2: list = []
    asyncio.run(hook2.on_llm_start(rc2, model="x", messages=msgs2))
    assert (msgs2[0].role, msgs2[0].content) == ("system", "[op] switch language")


def test_thread_safe_concurrent_pushes(monkeypatch):
    """Two threads push concurrently; drain returns every message exactly once."""
    import threading

    q = SteerQueue()
    n = 200

    def worker(prefix: str) -> None:
        for i in range(n):
            q.push(f"{prefix}-{i}")

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    drained = q.drain()
    assert len(drained) == 2 * n
    texts = {m.text for m in drained}
    assert len(texts) == 2 * n


def test_drain_returns_empty_list_when_no_messages():
    rc = RunContext()
    assert drain_steer(rc) == []
    assert drain_next_turn(rc) == []


def test_steer_queue_class_isolation():
    """SteerQueue and NextTurnQueue should not collide via metadata key."""
    rc = RunContext()
    queue_message(rc, "later")
    # Even though both are stored on the same dict, the steer queue must
    # be created as a fresh empty SteerQueue, not reuse the next-turn queue.
    steer_pending = peek_steer(rc)
    nxt_pending = peek_next_turn(rc)
    assert steer_pending == []
    assert [m.text for m in nxt_pending] == ["later"]
