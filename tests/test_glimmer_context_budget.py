"""Reserve generated tokens inside finite server context limits."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawagents.graph import context_management as cm
from clawagents.graph.turn_driver import TurnDriver
from clawagents.providers.llm import LLMMessage, NativeToolSchema


def _driver(cap=16384, window=32768):
    driver = TurnDriver.__new__(TurnDriver)
    driver._llm = SimpleNamespace(_max_tokens=cap, model='Muse-Glimmer-30B')
    driver._context_window = window
    driver._resolved_model_name = 'Muse-Glimmer-30B'
    driver._token_multiplier = 1.0
    driver._token_ledger = None
    driver._cached_system_tokens = 0
    driver._run_context = None
    driver._loop_tracker = None
    driver._external_hooks = None
    driver._events = SimpleNamespace(emit=lambda *_: None)
    driver._native_schemas = None
    driver._before_llm = None
    driver._overflow_retries = 0
    driver._session_journal = SimpleNamespace(note=lambda *_a, **_kw: None)
    driver._micro_compact = lambda messages, count: (messages, count)
    driver._apply_external_pre_llm = AsyncMock()
    driver._apply_before_llm = lambda messages: messages
    return driver


def test_32k_context_reserves_16k_output_and_schema():
    llm = SimpleNamespace(_max_tokens=16384)
    budget = cm._resolve_input_budget(32768, llm, 'Muse-Glimmer-30B', 1000)
    assert 0 < budget <= 32768 - 16384 - 1000


def test_factory_invoke_rejects_oversized_protected_input_before_provider(tmp_path, monkeypatch):
    from clawagents import create_claw_agent

    monkeypatch.chdir(tmp_path)
    agent = create_claw_agent(
        profile='meta', base_url='http://127.0.0.1:1/v1',
        context_window=32768, max_tokens=16384, reasoning_effort='low',
        workspace=tmp_path, mode='ci', tool_discovery=False,
        trajectory=False, rethink=False, learn=False,
        features={name: False for name in (
            'background_memory', 'core_memory', 'memory_bank', 'memory_dream',
            'smart_memory', 'context_ledger', 'fact_store', 'repo_map_inject',
            'session_persistence',
        )},
    )
    forbidden = AsyncMock(side_effect=AssertionError('Oversized input reached provider'))
    monkeypatch.setattr(agent.llm, 'chat', forbidden)
    prompt = 'Preserve this complete task text. ' + 'boundary ' * 20000
    # Character length is not a token count: establish the test boundary using
    # the same estimator as the product, including environments with tiktoken.
    assert cm._estimate_messages_tokens([LLMMessage(role='user', content=prompt)]) > 16384

    async def invoke():
        try:
            return await agent.invoke(prompt, timeout_s=30)
        finally:
            await agent.llm.client.close()

    result = asyncio.run(invoke())
    forbidden.assert_not_awaited()
    assert result.status == 'error'
    assert 'Task incomplete' in result.result and 'output reserve' in result.result
    assert any(message.role == 'user' and message.content == prompt for message in result.messages)


@pytest.mark.parametrize('cap', [None, 0, -1, True, '16384', float('nan')])
def test_missing_or_invalid_cap_preserves_existing_headroom(cap):
    assert cm._resolve_input_budget(196608, SimpleNamespace(_max_tokens=cap), 'Muse-Glimmer-30B') == int(196608 * .8)


def test_large_default_and_fallback_dynamic_growth():
    primary = SimpleNamespace(_max_tokens=16384)
    llm = SimpleNamespace(primary=primary)
    assert cm._resolve_input_budget(196608, llm, 'Muse-Glimmer-30B') == int(196608 * .8)
    before = cm._resolve_input_budget(32768, llm, 'Muse-Glimmer-30B')
    primary._max_tokens = 24576
    assert cm._resolve_input_budget(32768, llm, 'Muse-Glimmer-30B') < before
    primary._max_tokens = 16384
    assert cm._resolve_input_budget(32768, llm, 'Muse-Glimmer-30B') == before


def test_driver_compacts_before_sending_over_reserved_limit():
    driver = _driver()
    messages = [LLMMessage(role='system', content='system'),
                LLMMessage(role='user', content='current task'),
                LLMMessage(role='assistant', content='history')]
    driver._budget_tokens = lambda msgs, multiplier=None: 20000 if len(msgs) > 2 else 1000
    driver._compact = AsyncMock(return_value=messages[:2])
    result = asyncio.run(driver._prepare_messages(messages))
    driver._compact.assert_awaited_once()
    assert result[-1].content == 'current task'


def test_unshrinkable_user_input_is_reported_not_truncated():
    driver = _driver()
    user = LLMMessage(role='user', content='user data' * 20000)
    driver._compact = AsyncMock(side_effect=lambda messages: messages)
    with pytest.raises(cm._InputBudgetExceeded, match='output'):
        asyncio.run(driver._prepare_messages([LLMMessage(role='system', content='s'), user]))
    assert user.content == 'user data' * 20000


def test_compaction_uses_output_reserve_even_when_no_summarization_needed(monkeypatch):
    seen = []
    monkeypatch.setattr('clawagents.memory.compact_tool_results.compact_tool_results',
                        lambda messages, **kw: (seen.append(kw['max_input_tokens']) or messages, False))
    messages = [LLMMessage(role='system', content='s'), LLMMessage(role='user', content='task')]
    asyncio.run(cm._compact_if_needed(messages, 32768,
                SimpleNamespace(_max_tokens=16384), lambda *_: None,
                model_name='Muse-Glimmer-30B', native_schema_tokens=1000))
    assert 0 < seen[0] <= 32768 - 16384 - 1000


def test_driver_accounts_for_native_schema_reserve():
    driver = _driver()
    before = driver._input_budget()
    driver._native_schemas = [NativeToolSchema('large', 'x' * 8000, {})]
    assert driver._input_budget() < before


def test_overflow_recovery_keeps_physical_window_for_output_reserve():
    driver = _driver()
    driver._compact = AsyncMock(side_effect=lambda messages: messages)
    driver._budget_tokens = lambda *_: 15000
    messages = [LLMMessage(role='user', content='task')]
    result = asyncio.run(driver._recover_from_error(
        messages, SimpleNamespace(), 0, RuntimeError('maximum context length exceeded'),
    ))
    assert result.action == 'retry'
    assert driver._context_window == 32768
    assert driver._token_multiplier > 1
    driver._compact.assert_awaited_once()


def test_preflight_reserves_output_without_changing_user_text():
    user = LLMMessage(role='user', content='task data' * 12000)
    warnings = []
    out, _, _ = cm._preflight_context_check(
        [LLMMessage(role='system', content='system'), user], 32768, '', None, None,
        lambda kind, data: warnings.append((kind, data)),
        llm=SimpleNamespace(_max_tokens=16384),
    )
    assert out[-1] is user and user.content == 'task data' * 12000
    assert any(kind == 'warn' for kind, _ in warnings)


def test_unshrinkable_request_stops_before_provider_dispatch(monkeypatch):
    from clawagents.graph import turn_driver as td

    driver = _driver()
    driver._caller = SimpleNamespace(call=AsyncMock())
    driver._compact = AsyncMock(side_effect=lambda messages: messages)
    driver._budget_tokens = lambda *_: 25000
    monkeypatch.setattr(td, '_wal_write', lambda *_: None)
    state = SimpleNamespace(status='running', result='')
    messages = [LLMMessage(role='user', content='protected user data')]
    result = asyncio.run(driver.call(messages, state=state, round_index=0,
                                     cancel_event=asyncio.Event()))
    assert result.action == 'stop' and state.status == 'error'
    assert 'output' in state.result and 'incomplete' in state.result
    driver._caller.call.assert_not_awaited()


def test_compaction_thrash_cannot_bypass_final_output_reserve(monkeypatch):
    driver = _driver()
    driver._budget_tokens = lambda messages, multiplier=None: 20000 if len(messages) > 1 else 1000
    driver._fire_hook = None
    driver._taxonomy_dispatcher = None
    driver._compaction_savings = [0.01, 0.01, 0.01]
    monkeypatch.setattr('clawagents.memory.compaction.is_compression_thrashing', lambda *_: True)
    messages = [LLMMessage(role='system', content='system'),
                LLMMessage(role='user', content='protected task')]
    with pytest.raises(cm._InputBudgetExceeded):
        asyncio.run(driver._prepare_messages(messages))
    assert messages[-1].content == 'protected task'


def test_summary_calls_respect_reserve_without_mutating_provider_cap():
    llm = SimpleNamespace(_max_tokens=16384, chat=AsyncMock(return_value='summary'))
    protected = cm._OutputReservedCompactionLLM(llm, 32768, 'Muse-Glimmer-30B', 1.0)
    with pytest.raises(cm._InputBudgetExceeded):
        asyncio.run(protected.chat([LLMMessage(role='user', content='history data ' * 20000)]))
    llm.chat.assert_not_awaited()
    assert asyncio.run(protected.chat([LLMMessage(role='user', content='short history')])) == 'summary'
    llm.chat.assert_awaited_once()
    assert llm._max_tokens == 16384
