"""Projection experiments retain original history and exact archived evidence."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawagents.run_context import RunContext
from clawagents.providers.llm import LLMMessage, LLMResponse
from clawagents.memory.observation_projection import (
    archive_observation, project_observations, commit_projection, restore_observations,
)
from clawagents.tool_output_artifacts import load_tool_artifact


def register(tmp_path, sends=2, **kwargs):
    context = RunContext(observation_full_sends=sends)
    context._metadata['workspace'] = str(tmp_path)
    body = 'αβ😀 exact evidence\n' * 1000
    output = archive_observation(context, tool_name='read_file', call_id='read', output=body, success=True, **kwargs)
    return context, body, output


@pytest.mark.parametrize('sends', [1, 2])
def test_only_committed_requests_age_observations(tmp_path, sends):
    context, body, output = register(tmp_path, sends)
    message = LLMMessage(role='tool', content=output, tool_call_id='read')
    history = [message]
    for _ in range(sends):
        for _ in range(3):
            preview = project_observations(history, context)
            assert preview.messages[0].content == output
        commit_projection(preview, context)
    projected = project_observations(history, context)
    assert len(projected.messages[0].content) < len(body)
    assert 'retrieve_tool_result' in projected.messages[0].content
    assert history[0] is message and message.content == output
    assert restore_observations(projected.messages, projected)[0].content == message.content
    commit_projection(projected, context)
    assert context.efficiency['tokens_avoided_by_handles'] > 0
    entry = next(iter(context._observations.values()))
    assert load_tool_artifact(entry['artifact_id'], workspace=tmp_path)[1] == body


def test_default_failures_receipts_and_control_plane_do_not_pack(tmp_path):
    context, body, _ = register(tmp_path, 0)
    assert not context._observations
    context.observation_full_sends = 2
    for tool, success, value in [('execute', False, body), ('retrieve_tool_result', True, body), ('read_file', True, 'clawagents_evidence_receipt_v1\n' + body), ('read_file', True, 'small')]:
        assert archive_observation(context, tool_name=tool, call_id='x', output=value, success=success) is None


def test_archive_failure_keeps_original_and_no_state(tmp_path, monkeypatch):
    context = RunContext(observation_full_sends=2)
    monkeypatch.setattr('clawagents.memory.observation_projection.store_tool_artifact', lambda **kwargs: (_ for _ in ()).throw(OSError('disk full')))
    assert archive_observation(context, tool_name='execute', call_id='x', output='x'*12000, success=True) is None
    assert not context._observations


def test_branch_context_has_independent_send_counts(tmp_path):
    first, body, output = register(tmp_path)
    second, _, _ = register(tmp_path)
    history = [LLMMessage(role='tool', content=output, tool_call_id='read')]
    commit_projection(project_observations(history, first), first)
    assert next(iter(second._observations.values()))['sends'] == 0
    assert project_observations(history, second).messages == history


def test_missing_archive_and_changed_observation_fail_open(tmp_path):
    context, body, output = register(tmp_path, 1)
    message = LLMMessage(role='tool', content=output, tool_call_id='read')
    commit_projection(project_observations([message], context), context)
    changed = LLMMessage(role='tool', content=output + '\nimportant new warning', tool_call_id='read')
    projected = project_observations([changed], context)
    assert 'important new warning' in projected.messages[0].content
    entry = next(iter(context._observations.values()))
    from pathlib import Path
    Path(entry['path']).unlink()
    assert project_observations([message], context).messages[0] is message


def test_prepare_preview_preserves_hook_changes(tmp_path):
    context, _, output = register(tmp_path, 1)
    original = LLMMessage(role='tool', content=output, tool_call_id='read')
    commit_projection(project_observations([original], context), context)
    preview = project_observations([original], context)
    preview.messages[0].content = 'redacted by hook'
    assert restore_observations(preview.messages, preview)[0].content == 'redacted by hook'
    assert original.content == output


def test_copied_hook_messages_restore_full_history(tmp_path):
    from copy import copy
    context, body, output = register(tmp_path, 1)
    original = LLMMessage(role='tool', content=output, tool_call_id='read')
    commit_projection(project_observations([original], context), context)
    preview = project_observations([original], context)
    hooked = copy(preview.messages[0])
    hooked.content = 'hook note\n' + hooked.content
    hooked.thinking = 'new metadata'
    restored = restore_observations([hooked], preview)
    assert body in restored[0].content
    assert restored[0].content.startswith('hook note\n')
    assert restored[0].thinking == 'new metadata'
    preview.messages[0].thinking = 'in-place metadata update'
    assert restore_observations(preview.messages, preview)[0].thinking == 'in-place metadata update'


def test_missing_artifact_invalidates_projected_token_checkpoint(tmp_path):
    from pathlib import Path
    from clawagents.graph.turn_driver import IncrementalTokenLedger
    context, _, output = register(tmp_path, 1)
    history = [LLMMessage(role='tool', content=output, tool_call_id='read')]
    commit_projection(project_observations(history, context), context)
    projected = project_observations(history, context)
    ledger = IncrementalTokenLedger(lambda messages: sum(len(m.content) for m in messages))
    ledger.record_provider_usage(projected.messages, 100)
    entry = next(iter(context._observations.values()))
    Path(entry['path']).unlink()
    full = project_observations(history, context)
    assert ledger.estimate(full.messages) == len(output)


def test_real_caller_projects_but_does_not_age_failed_request(tmp_path):
    import asyncio
    from clawagents.graph.turn_llm import TurnLLMCaller
    context, body, output = register(tmp_path, 1)
    history = [LLMMessage(role='tool', content=output, tool_call_id='read')]
    llm = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError('provider unavailable')))
    caller = TurnLLMCaller(llm=llm, events=SimpleNamespace(typed=lambda *args: None, emit=lambda *args: None), hooks=SimpleNamespace(hooks=[]), registry=None, session_writer=None, external_hooks=None, accumulate_usage=lambda *args, **kwargs: None)
    kwargs = dict(resolved_model_name='fixture', use_native_tools=False, tools_supplied=False, initial_schemas=None, handoffs=[], streaming=False, cancel_event=None, run_context=context, output_type=None)
    with pytest.raises(RuntimeError):
        asyncio.run(caller.call(history, **kwargs))
    assert next(iter(context._observations.values()))['sends'] == 0
    llm.chat.side_effect = None
    llm.chat.return_value = LLMResponse(content='ok', tokens_used=1, model='fixture')
    asyncio.run(caller.call(history, **kwargs))
    assert body in llm.chat.call_args.args[0][0].content
    call_result = asyncio.run(caller.call(history, **kwargs))
    assert body not in llm.chat.call_args.args[0][0].content
    assert call_result.request_messages == llm.chat.call_args.args[0]
    assert history[0].content == output


def test_real_tool_processor_delays_initial_crush_and_fails_open(tmp_path, monkeypatch):
    from clawagents.graph.tool_batch import ToolResultProcessor
    from clawagents.tools.registry import ParsedToolCall, ToolResult
    context = RunContext(observation_full_sends=2)
    context._metadata['workspace'] = str(tmp_path)
    events = SimpleNamespace(emit=lambda *args: None, typed=lambda *args: None)
    processor = ToolResultProcessor(external_hooks=None, taxonomy_dispatcher=None,
        after_tool=None, events=events, session_writer=None, run_context=context, preview_chars=100)
    body = 'def precise_function():\n    return "exact evidence"\n' * 300
    call = ParsedToolCall(tool_name='read_file', args={'path': 'source.py'})
    prepared = processor.prepare(call, ToolResult(True, body), call_id='first')
    assert body in prepared.output and context.efficiency['tokens_avoided_by_handles'] == 0
    monkeypatch.setattr('clawagents.memory.observation_projection.store_tool_artifact', lambda **kwargs: (_ for _ in ()).throw(OSError('disk full')))
    prepared = processor.prepare(call, ToolResult(True, body), call_id='second')
    assert body in prepared.output


def test_text_batch_preserves_surrounding_evidence_and_metadata(tmp_path):
    context, body, output = register(tmp_path, 1)
    text = '[Tool Results]\nread_file({"path":"a.py"}) => ' + output + '\nexecute({}) => verification failed'
    original = LLMMessage(role='user', content=text, thinking='keep')
    commit_projection(project_observations([original], context), context)
    projected = project_observations([original], context)
    assert body not in projected.messages[0].content
    assert 'verification failed' in projected.messages[0].content
    assert projected.messages[0].thinking == 'keep'
    assert original.content == text
    commit_projection(projected, context)
    counters = dict(context.efficiency)
    commit_projection(projected, context)
    assert context.efficiency == counters


def test_compaction_releases_state_without_deleting_artifacts(tmp_path):
    context, body, output = register(tmp_path)
    entry = next(iter(context._observations.values()))
    commit_projection(project_observations([LLMMessage(role='user', content='new task')], context), context)
    assert not context._observations
    assert load_tool_artifact(entry['artifact_id'], workspace=tmp_path)[1] == body


@pytest.mark.parametrize('invalid', [-1, 3, True, '2', 1.5])
def test_invalid_allowance_rejected(invalid):
    with pytest.raises(ValueError, match='observation_full_sends'):
        RunContext(observation_full_sends=invalid)
