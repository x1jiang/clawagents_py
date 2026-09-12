"""Fused edits retain mutation, permission and ordering guarantees."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from clawagents.run_context import RunContext
from clawagents.sandbox.local import LocalBackend
from clawagents.tools.filesystem import WriteFileTool, EditFileTool
from clawagents.tools.apply_patch import ApplyPatchTool
from clawagents.tools.hashline import HashlineEditTool
from clawagents.tools.registry import ToolRegistry, ToolResult


@pytest.fixture
def registry(tmp_path):
    reg = ToolRegistry()
    backend = LocalBackend(str(tmp_path))
    for cls in (WriteFileTool, EditFileTool, ApplyPatchTool, HashlineEditTool):
        reg.register(cls(backend))
    return reg


@pytest.mark.parametrize('name', ['write_file', 'edit_file', 'apply_patch', 'hashline_edit'])
def test_schema_advertises_fusion(registry, name):
    schema = registry.get(name).parameters['then_run']
    assert schema['type'] == 'object'
    assert schema['properties']['command']['type'] == 'string'
    assert schema['required'] is False


@pytest.mark.asyncio
async def test_success_and_timeout_units(registry, tmp_path):
    async def followup(args, unchanged):
        assert await unchanged()
        assert args == {'command': 'pytest tests/test_one.py', 'timeout': 45000}
        assert (tmp_path / 'a.txt').read_text() == 'new'
        return 'succeeded', ToolResult(True, '1 passed')
    result = await registry.execute_tool('write_file', {
        'path': 'a.txt', 'content': 'new',
        'then_run': {'command': 'pytest tests/test_one.py', 'timeout': 45000},
    }, followup=followup)
    assert result.success
    assert 'Wrote' in result.raw_output
    assert '[then_run:succeeded]\n1 passed' in result.raw_output


@pytest.mark.asyncio
async def test_mutation_and_command_failures_are_distinct(registry, tmp_path):
    followup = AsyncMock(return_value=('failed', ToolResult(False, 'FAILED test_x', 'exit 1')))
    bad = await registry.execute_tool('edit_file', {
        'path': 'absent.txt', 'target': 'old', 'replacement': 'new',
        'then_run': {'command': 'pytest'},
    }, followup=followup)
    assert not bad.success
    assert '[then_run:skipped]' in bad.raw_output
    followup.assert_not_called()
    failed = await registry.execute_tool('write_file', {
        'path': 'a.txt', 'content': 'new', 'then_run': {'command': 'pytest'},
    }, followup=followup)
    assert not failed.success
    assert (tmp_path / 'a.txt').read_text() == 'new'
    assert 'Wrote' in failed.raw_output and '[then_run:failed]' in failed.raw_output
    assert 'FAILED test_x' in failed.raw_output


@pytest.mark.asyncio
async def test_invalid_fusion_refused_before_mutation(registry, tmp_path):
    for value in ({'command': ''}, {'command': 'x', 'timeout': -1}, 'pytest', {'command': 'x', 'unsandboxed': True}):
        result = await registry.execute_tool('write_file', {'path': 'a.txt', 'content': 'new', 'then_run': value})
        assert not result.success
        assert not (tmp_path / 'a.txt').exists()


@pytest.mark.asyncio
async def test_no_approval_path_skips_command_keeps_edit(registry, tmp_path):
    result = await registry.execute_tool('write_file', {
        'path': 'a.txt', 'content': 'new', 'then_run': {'command': 'touch unsafe'},
    })
    assert '[then_run:skipped]' in result.raw_output
    assert (tmp_path / 'a.txt').read_text() == 'new'
    assert not (tmp_path / 'unsafe').exists()


@pytest.mark.asyncio
async def test_syntax_failure_skips_command(registry, tmp_path):
    followup = AsyncMock()
    result = await registry.execute_tool('write_file', {
        'path': str(tmp_path / 'bad.py'), 'content': 'def broken(:\n',
        'then_run': {'command': 'pytest'},
    }, followup=followup)
    assert '[syntax_gate]' in result.raw_output and 'FAILED' in result.raw_output
    assert '[then_run:skipped]' in result.raw_output
    followup.assert_not_called()


@pytest.mark.asyncio
async def test_hash_guard_after_approval_wait(registry, tmp_path):
    async def followup(args, unchanged):
        (tmp_path / 'a.txt').write_text('external edit')
        assert not await unchanged()
        return 'skipped', ToolResult(False, 'File changed during approval')
    result = await registry.execute_tool('write_file', {
        'path': 'a.txt', 'content': 'new', 'then_run': {'command': 'pytest'},
    }, followup=followup)
    assert '[then_run:skipped]' in result.raw_output
    assert (tmp_path / 'a.txt').read_text() == 'external edit'


@pytest.mark.asyncio
async def test_same_canonical_file_locked_through_command(registry, tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    async def followup(args, unchanged):
        entered.set()
        await release.wait()
        assert (tmp_path / 'a.txt').read_text() == 'first'
        return 'succeeded', ToolResult(True, 'ok')
    first = asyncio.create_task(registry.execute_tool('write_file', {
        'path': 'a.txt', 'content': 'first', 'then_run': {'command': 'pytest'},
    }, followup=followup))
    await asyncio.wait_for(entered.wait(), 2)
    (tmp_path / 'link.txt').symlink_to(tmp_path / 'a.txt')
    second = asyncio.create_task(registry.execute_tool('write_file', {'path': './link.txt', 'content': 'second'}))
    await asyncio.sleep(.03)
    assert not second.done()
    release.set()
    await asyncio.gather(first, second)
    assert (tmp_path / 'a.txt').read_text() == 'second'


def test_explicit_followup_skips_auto_verify(monkeypatch):
    from clawagents.graph.tool_observation import _post_tool_side_effects
    monkeypatch.setenv('CLAW_FEATURE_AUTO_VERIFY', '1')
    verify = Mock(return_value='duplicate check')
    monkeypatch.setattr('clawagents.tools.auto_verify.maybe_verify_after_edit', verify)
    _post_tool_side_effects('write_file', {'then_run': {'command': 'pytest'}}, True, 'Wrote', emit=lambda *a: None)
    verify.assert_not_called()


@pytest.mark.asyncio
async def test_followup_policy_and_approval(registry):
    from clawagents.graph.tool_turn import ToolTurnExecutor
    from clawagents.graph.tool_batch import ToolPolicyGate
    from clawagents.graph.agent_loop import HookResult
    executor = ToolTurnExecutor.__new__(ToolTurnExecutor)
    executor._run_context = RunContext()
    executor._registry = registry
    executor._events = SimpleNamespace(emit=Mock(), typed=Mock())
    executor._loop_tracker = SimpleNamespace(is_circuit_broken=lambda: False, is_hard_looping=lambda *a: False,
                                             note_mutation=Mock(), record=Mock(), record_result=Mock(return_value=None))
    executor._require_approval_set = {'execute'}
    executor._approval_handler = AsyncMock(return_value=False)
    executor._policy_gate = ToolPolicyGate(external_hooks=None, taxonomy_dispatcher=None, before_tool=None,
                                           hook_result_type=HookResult, events=executor._events)
    executor._call_runner = SimpleNamespace(execute=AsyncMock(return_value=ToolResult(True, 'ok')))
    executor._result_processor = SimpleNamespace(apply_middleware=AsyncMock(side_effect=lambda c,r:r))
    state = SimpleNamespace(tool_calls=0)
    guard = AsyncMock(return_value=True)
    status, _ = await executor._execute_followup({'command': 'pytest'}, guard, call_id='edit1:then_run', messages=[], state=state)
    assert status == 'skipped'
    executor._approval_handler.assert_awaited_once()
    executor._call_runner.execute.assert_not_called()
    executor._approval_handler = AsyncMock(return_value=True)
    status, _ = await executor._execute_followup({'command': 'pytest'}, guard, call_id='edit2:then_run', messages=[], state=state)
    assert status == 'succeeded'
    assert state.tool_calls == 1
    guard.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('batch,allow', [(False, True), (True, True), (False, False), (True, False)])
async def test_full_graph_fusion_preserves_hooks_and_transcript(registry, tmp_path, batch, allow):
    from clawagents.graph.agent_loop import run_agent_graph
    from clawagents.providers.llm import LLMResponse, NativeToolCall
    events, hooks = [], []

    class Command:
        name, description, parameters = 'execute', 'Fake local diagnostic', {'command': {'type': 'string'}}
        async def execute(self, args):
            assert (tmp_path / 'a.txt').read_text() == 'new'
            hooks.append(('executed', args['command']))
            return ToolResult(True, '1 passed')

    class Model:
        model = 'test-model'
        calls = 0
        async def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                calls = [NativeToolCall('write_file', {'path': 'a.txt', 'content': 'new', 'then_run': {'command': 'pytest'}}, 'write1')]
                if batch:
                    calls.append(NativeToolCall('write_file', {'path': 'b.txt', 'content': 'second'}, 'write2'))
                return LLMResponse('', self.model, 10, tool_calls=calls)
            tool_messages = [m for m in messages if m.role == 'tool']
            assert len(tool_messages) == (2 if batch else 1)
            assert all(m.tool_call_id != 'write1:then_run' for m in tool_messages)
            assert ('[then_run:succeeded]' if allow else '[then_run:skipped]') in str(tool_messages[0].content)
            return LLMResponse('Finished', self.model, 5)

    def before(name, args):
        hooks.append(('before', name))
        return name != 'execute' or allow

    def after(name, args, result):
        hooks.append(('after', name))
        return result

    registry.register(Command())
    result = await run_agent_graph('Write and check', Model(), registry, streaming=False,
        max_iterations=3, before_tool=before, after_tool=after, session_end_tail=False,
        on_event=lambda kind,data:events.append((kind,data)),
        features={'shadow_checkpoints': False, 'context_ledger': False}, run_context=RunContext())
    assert ('before', 'execute') in hooks
    assert (('executed', 'pytest') in hooks) == allow
    assert (('after', 'execute') in hooks) == allow
    assert result.tool_calls == (2 if batch else 1) + int(allow)
    assert result.efficiency['round_trips_avoided'] == int(allow)
    assert [v for k,v in events if k == 'efficiency'][-1]['efficiency']['round_trips_avoided'] == int(allow)
    assert (tmp_path / 'a.txt').read_text() == 'new'


@pytest.mark.asyncio
async def test_cancel_releases_file_lock(registry):
    entered = asyncio.Event()
    async def followup(args, unchanged):
        entered.set()
        await asyncio.Event().wait()
    pending = asyncio.create_task(registry.execute_tool('write_file', {
        'path': 'a.txt', 'content': 'first', 'then_run': {'command': 'pytest'},
    }, followup=followup))
    await asyncio.wait_for(entered.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    result = await asyncio.wait_for(registry.execute_tool('write_file', {'path': 'a.txt', 'content': 'second'}), 2)
    assert result.success


@pytest.mark.asyncio
async def test_retargeted_alias_cannot_write_through_another_files_lock(registry, tmp_path):
    entered = {name: asyncio.Event() for name in ('a', 'b')}
    release = {name: asyncio.Event() for name in ('a', 'b')}
    async def followup(args, unchanged):
        name = args['command']
        entered[name].set()
        await release[name].wait()
        return 'succeeded', ToolResult(True, name)
    tasks = [asyncio.create_task(registry.execute_tool('write_file', {
        'path': name + '.txt', 'content': name, 'then_run': {'command': name},
    }, followup=followup)) for name in ('a', 'b')]
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 2)
    alias = tmp_path / 'alias.txt'
    alias.symlink_to(tmp_path / 'a.txt')
    queued = asyncio.create_task(registry.execute_tool('write_file', {'path': 'alias.txt', 'content': 'overwrite'}))
    try:
        await asyncio.sleep(.02)
        assert not queued.done()
        alias.unlink()
        alias.symlink_to(tmp_path / 'b.txt')
        release['a'].set()
        result = await asyncio.wait_for(queued, 2)
        assert not result.success
        assert 'target changed' in result.error
        assert not tasks[1].done()
        assert (tmp_path / 'b.txt').read_text() == 'b'
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_batch_verifies_each_new_edit_not_cached_previous_output(registry, tmp_path, monkeypatch):
    from clawagents.graph.agent_loop import run_agent_graph
    from clawagents.providers.llm import LLMResponse, NativeToolCall
    executions, postprocessed = [], []
    class Command:
        name, description, parameters = 'execute', 'Check contents', {'command': {'type': 'string'}}
        async def execute(self, args):
            text = (tmp_path / 'a.txt').read_text()
            executions.append(text)
            return ToolResult(True, f'verified {text}')
    class Model:
        model = 'test-model'
        count = 0
        async def chat(self, messages, **kwargs):
            self.count += 1
            if self.count == 1:
                return LLMResponse('', self.model, 1, tool_calls=[
                    NativeToolCall('write_file', {'path': 'a.txt', 'content': text, 'then_run': {'command': 'check a.txt'}}, f'edit{i}')
                    for i, text in enumerate(['first', 'second'])])
            outputs = [str(m.content) for m in messages if m.role == 'tool']
            assert 'verified first' in outputs[0]
            assert 'verified second' in outputs[1]
            return LLMResponse('Done', self.model, 1)
    def side_effects(name, args, success, output, **kwargs):
        postprocessed.append(name)
        return output
    monkeypatch.setattr('clawagents.graph.tool_observation._post_tool_side_effects', side_effects)
    registry.register(Command())
    state = await run_agent_graph('Edit twice and verify both', Model(), registry, streaming=False,
        max_iterations=3, session_end_tail=False, features={'shadow_checkpoints': False, 'context_ledger': False})
    assert executions == ['first', 'second']
    assert postprocessed.count('execute') == 2
    assert state.efficiency['round_trips_avoided'] == 2
