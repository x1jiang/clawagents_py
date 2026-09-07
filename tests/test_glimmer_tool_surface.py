"""Glimmer starts lean while retaining mode boundaries and optional capability."""

import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from clawagents.agent import create_claw_agent
from clawagents import harness_profiles
from clawagents.providers.llm import LLMProvider
from clawagents.tools.registry import ToolRegistry
from clawagents.tools.tool_groups import (
    ActivateToolGroupTool,
    CODING_TOOL_NAMES,
    GOAL_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    apply_mode_active_profile,
)


LEAN = {'read_file', 'edit_file', 'write_file', 'execute', 'ls', 'glob', 'grep',
        'activate_tool_group', 'retrieve_tool_result', 'ask_user'}
FORCED = {'tool_discover', 'tool_describe', 'tool_profile', 'finish_coordination'}


def _registry():
    registry = ToolRegistry()
    for name in GOAL_TOOL_NAMES | {'web_fetch', 'context_reader'}:
        registry.register(SimpleNamespace(name=name, description=name, parameters={},
                                          context_protection=name == 'context_reader'))
    return registry


def test_initial_surface_is_intersection_plus_required_controls():
    registry = _registry()
    active = set(apply_mode_active_profile(registry, initial_tools=LEAN | {'web_fetch'}))
    assert active == LEAN | FORCED | {'context_reader'}
    assert registry.get('hashline_edit') is not None
    assert not registry.is_tool_active('hashline_edit')
    assert not registry.is_tool_active('web_fetch')


@pytest.mark.parametrize('mode,goal,wanted', [
    ('read_only', False, READ_ONLY_TOOL_NAMES),
    ('plan', False, READ_ONLY_TOOL_NAMES),
    ('ask', False, READ_ONLY_TOOL_NAMES),
    ('goal', False, GOAL_TOOL_NAMES),
    ('auto', True, GOAL_TOOL_NAMES),
])
def test_read_only_and_goal_modes_preserve_existing_surface(mode, goal, wanted):
    registry = _registry()
    active = set(apply_mode_active_profile(
        registry, chat_mode=mode, goal_mode=goal, initial_tools=LEAN,
    ))
    assert active == wanted | {'context_reader'}
    if wanted is READ_ONLY_TOOL_NAMES:
        assert not active & {'execute', 'write_file', 'edit_file', 'task'}
    else:
        assert {'task', 'write_todos', 'update_todo'} <= active


@pytest.mark.parametrize('invalid', [None, 'read_file', {'read_file': True}, [], [None], ['']])
def test_invalid_initial_tool_metadata_keeps_default_profile(invalid):
    registry = _registry()
    assert set(apply_mode_active_profile(registry, initial_tools=invalid)) == (
        CODING_TOOL_NAMES | {'context_reader'}
    )


def test_optional_group_restores_hidden_editors_planning_and_skills():
    registry = _registry()
    apply_mode_active_profile(registry, initial_tools=LEAN)
    activation = ActivateToolGroupTool(registry)
    listed = asyncio.run(activation.execute({'group': 'list'}))
    groups = {g['group']: set(g['tools']) for g in json.loads(listed.output)['groups']}
    hidden = {'hashline_read', 'hashline_grep', 'hashline_edit', 'apply_patch',
              'think', 'write_plan', 'write_todos', 'update_todo', 'list_skills', 'use_skill'}
    assert hidden <= groups['coding_full']
    result = asyncio.run(activation.execute({'group': 'coding_full'}))
    assert result.success
    assert hidden <= set(registry.active_tool_names())
    assert hidden <= set(result.added_tool_names)
    assert not registry.is_tool_active('web_fetch')


class _FakeLLM(LLMProvider):
    name = 'fake'

    def __init__(self, model):
        self.model = model

    async def chat(self, messages, **kwargs):
        raise AssertionError('This test must not call a model')


@pytest.mark.parametrize('model,lean', [
    ('Muse-Glimmer-30B', True), ('served-custom-30b', True), ('gpt-5.6-luna', False),
])
def test_agent_applies_meta_initial_tools_and_preserves_luna(tmp_path, monkeypatch, model, lean):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness_profiles, '_profile_paths', lambda: [])
    monkeypatch.setitem(harness_profiles._MODEL_ALIASES, 'served-custom-30b', 'meta-glimmer')
    original = harness_profiles.resolve_harness_profile

    def resolve(name):
        profile = original(name)
        if profile is not None and profile.name == 'meta-glimmer':
            return replace(profile, metadata={**profile.metadata, 'initial_tools': sorted(LEAN)})
        return profile

    monkeypatch.setattr(harness_profiles, 'resolve_harness_profile', resolve)
    agent = create_claw_agent(model=_FakeLLM(model), workspace=tmp_path,
                              skills=[], memory=[], tool_discovery=False)
    names = {tool.name for tool in agent.tools.list()}
    assert agent.tools.get('think') is not None
    assert ('think' not in names) is lean
    assert {'read_file', 'edit_file', 'execute', 'activate_tool_group'} <= names


def test_profile_failure_logs_warning_without_breaking_agent(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)

    def broken_profile(_name):
        raise ValueError('profile unavailable')

    monkeypatch.setattr(harness_profiles, 'resolve_harness_profile', broken_profile)
    agent = create_claw_agent(model=_FakeLLM('fake-local'), workspace=tmp_path,
                              skills=[], memory=[], tool_discovery=False)
    assert agent.tools.get('read_file') is not None
    assert 'harness profile prompt skipped' in caplog.text
    assert 'harness tool-surface profile skipped' in caplog.text


@pytest.mark.parametrize('mode', ['plan', 'read_only'])
def test_full_coding_activation_cannot_bypass_runtime_plan_permissions(tmp_path, monkeypatch, mode):
    from clawagents.permissions.mode import PermissionMode
    from clawagents.providers.llm import LLMResponse, NativeToolCall

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness_profiles, '_profile_paths', lambda: [])
    schemas = []

    class AdversarialLLM(_FakeLLM):
        async def chat(self, messages, **kwargs):
            schemas.append({tool.name for tool in kwargs.get('tools') or []})
            calls = [
                NativeToolCall('activate_tool_group', {'group': 'coding_full'}, 'activate'),
                NativeToolCall('write_file', {'path': 'forbidden-write.txt', 'content': 'bad'}, 'write'),
                NativeToolCall('execute', {'command': 'printf bad > forbidden-shell.txt'}, 'shell'),
            ]
            index = len(schemas) - 1
            return LLMResponse(
                content='' if index < len(calls) else 'Read-only review complete.',
                model=self.model, tokens_used=10,
                tool_calls=[calls[index]] if index < len(calls) else None,
            )

    events = []
    agent = create_claw_agent(
        model=AdversarialLLM('Muse-Glimmer-30B'), workspace=tmp_path,
        chat_mode=mode, skills=[], memory=[], tool_discovery=False,
        streaming=False, features={'session_persistence': False},
    )
    state = asyncio.run(agent.invoke(
        'Review the workspace without making changes.', max_iterations=6,
        on_event=lambda kind, data: events.append((kind, data)),
    ))
    assert len(schemas) == 4
    assert not {'execute', 'write_file'} & schemas[0]
    assert {'execute', 'write_file'} <= schemas[1]
    results = {data['name']: data for kind, data in events if kind == 'tool_result'}
    assert results['activate_tool_group']['success'] is True
    assert results['write_file']['success'] is False
    assert results['execute']['success'] is False
    assert 'plan mode' in str(results['write_file']['output']).lower()
    assert 'plan mode' in str(results['execute']['output']).lower()
    assert state.run_context.permission_mode == PermissionMode.PLAN
    assert not (tmp_path / 'forbidden-write.txt').exists()
    assert not (tmp_path / 'forbidden-shell.txt').exists()
