"""Gemma serving and heterogeneous delegation regressions."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from clawagents.tools.subagent import SubAgentSpec, TaskTool


@pytest.mark.asyncio
@pytest.mark.parametrize('message', ['[cancelled]', 'Reached maximum of 5 tool rounds.'])
async def test_incomplete_child_is_not_success(message):
    async def run(**kwargs):
        return SimpleNamespace(status='done', result=message, tool_calls=1, iterations=5)
    with patch('clawagents.graph.agent_loop.run_agent_graph', run):
        result = await TaskTool(None, None).execute({'description': 'work'})
    assert not result.success
    assert 'incomplete' in result.error.lower()


@pytest.mark.asyncio
async def test_explicit_worker_keeps_its_provider():
    parent = SimpleNamespace(model='gemma4-agentic-v2')
    worker = SimpleNamespace(model='remote-worker', endpoint='different-host')
    seen = {}
    async def run(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(status='done', result='verified', tool_calls=1, iterations=1)
    spec = SubAgentSpec('coder', 'Implementation', llm=worker)
    with patch('clawagents.graph.agent_loop.run_agent_graph', run):
        result = await TaskTool(parent, None, [spec]).execute({'description':'work','agent':'coder'})
    assert result.success
    assert seen['llm'] is worker
    assert parent.model == 'gemma4-agentic-v2'
    assert seen['run_context'].skip_memory


@pytest.mark.asyncio
async def test_unknown_worker_is_rejected_with_configured_catalog():
    tool = TaskTool(None, None, [SubAgentSpec('coder', 'Implementation')])
    result = await tool.execute({'description':'work','agent':'made-up'})
    assert not result.success
    assert 'coder' in result.error


def test_gemma_chat_options_are_model_specific():
    from clawagents.providers.gemma import apply_chat_options
    kwargs = {'max_completion_tokens':4096,'temperature':0,'tools':[{}]}
    apply_chat_options(kwargs, 'gemma4-agentic-v2')
    assert kwargs['temperature'] == 0
    assert kwargs['max_tokens'] == 4096
    assert 'max_completion_tokens' not in kwargs
    assert kwargs['extra_body']['repeat_penalty'] == 1.1
    assert kwargs['parallel_tool_calls'] is False
    ordinary = {'max_completion_tokens':4096}
    apply_chat_options(ordinary, 'gpt-5.6-luna')
    assert ordinary == {'max_completion_tokens':4096}


@pytest.mark.asyncio
async def test_gemma_profile_small_context_and_delegation(monkeypatch, tmp_path):
    from clawagents import create_claw_agent
    monkeypatch.setenv('GEMMA_AGENTIC_BASE_URL','http://127.0.0.1:18080/v1')
    monkeypatch.setenv('OPENAI_API_KEY','must-not-leak')
    with patch('clawagents.provider_profiles._profile_paths', return_value=[]):
        agent = create_claw_agent(profile='gemma-agentic', workspace=tmp_path, skills=[], memory=[])
    try:
        assert agent.llm.client.api_key == 'not-needed'
        assert agent.llm._max_tokens == 4096
        assert not agent.llm._should_use_responses(True)
        names={t.name for t in agent.tools.list()}
        assert 'task' in names
        assert len(names) <= 14
    finally:
        await agent.llm.client.close()


@pytest.mark.asyncio
async def test_worker_deadline_returns_incomplete():
    import asyncio
    async def run(**kwargs):
        await asyncio.sleep(5)
    tool = TaskTool(None, None, [SubAgentSpec('slow', 'Slow worker', timeout_seconds=0.01)])
    with patch('clawagents.graph.agent_loop.run_agent_graph', run):
        result = await tool.execute({'description':'work','agent':'slow'})
    assert not result.success
    assert 'deadline' in result.error


@pytest.mark.parametrize('timeout',[0,-1,float('inf'),float('nan')])
def test_worker_timeout_validation(timeout):
    with pytest.raises(ValueError):
        TaskTool(None,None,[SubAgentSpec('worker','Worker',timeout_seconds=timeout)])


@pytest.mark.asyncio
async def test_worker_model_override_cannot_reuse_wrong_endpoint():
    spec=SubAgentSpec('coder','Coder',llm=SimpleNamespace(model='worker'))
    result=await TaskTool(None,None,[spec]).execute({'description':'work','agent':'coder','model':'other'})
    assert not result.success
    assert 'conflicts' in result.error


@pytest.mark.asyncio
async def test_explicit_worker_iteration_cap_cannot_be_inflated():
    seen = {}
    async def run(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(status='done',result='verified',tool_calls=1,iterations=1)
    spec=SubAgentSpec('worker','Worker',max_iterations=2,llm=SimpleNamespace(model='fixture'))
    with patch('clawagents.graph.agent_loop.run_agent_graph',run):
        await TaskTool(None,None,[spec]).execute({'agent':'worker','description':'work','max_iterations':100})
    assert seen['max_iterations']==2


@pytest.mark.asyncio
async def test_fixture_worker_positive_and_failure_controls():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from benchmark_gemma_coordination import FixtureWorker
    from clawagents.providers.llm import LLMMessage
    worker=FixtureWorker('sales',fail_once=True)
    messages=[LLMMessage(role='user',content='Sum 17,23,11')]
    with pytest.raises(RuntimeError):
        await worker.chat(messages)
    result=await worker.chat(messages)
    assert '51' in result.content
    missing=await worker.chat([LLMMessage(role='user',content='Sum these')])
    assert 'Missing input' in missing.content


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming',[False,True])
async def test_actual_chat_request_contains_gemma_options(streaming,tmp_path):
    from clawagents import create_claw_agent
    from clawagents.providers.llm import LLMMessage,NativeToolSchema
    from openai.types.chat import ChatCompletion,ChatCompletionChunk
    agent=create_claw_agent(profile='gemma-agentic',streaming=streaming,workspace=tmp_path,skills=[],memory=[])
    captured={}
    async def create(**kwargs):
        captured.update(kwargs)
        if streaming:
            async def chunks():
                yield ChatCompletionChunk(id='x',object='chat.completion.chunk',created=0,model='gemma4-agentic-v2',choices=[{'index':0,'delta':{'content':'ok'},'finish_reason':'stop'}])
            return chunks()
        return ChatCompletion(id='x',object='chat.completion',created=0,model='gemma4-agentic-v2',choices=[{'index':0,'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}])
    try:
        with patch.object(agent.llm.client.chat.completions,'create',create):
            await agent.llm.chat([LLMMessage(role='user',content='test')],on_chunk=(lambda chunk: None) if streaming else None,tools=[NativeToolSchema('lookup','Lookup',{})])
        assert captured['extra_body']['repeat_penalty']==1.1
        assert captured['extra_body']['chat_template_kwargs']['enable_thinking']
        assert captured['max_tokens']==4096
        assert captured['parallel_tool_calls'] is False
    finally:
        await agent.llm.client.close()


@pytest.mark.asyncio
async def test_loop_safety_stop_is_failure_not_done():
    from unittest.mock import AsyncMock, Mock
    from clawagents.graph.round_dispatcher import RoundDispatcher
    parsed=SimpleNamespace(should_resample=False,tool_calls=[object()],native_tool_calls=[],response=SimpleNamespace(content=''),thinking='')
    driver=SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(messages=[],action='proceed',response=None)))
    interpreter=SimpleNamespace(parse=Mock(return_value=parsed))
    handoff=SimpleNamespace(dispatch=AsyncMock(return_value=SimpleNamespace(handled=False)))
    safety=SimpleNamespace(check=Mock(return_value=SimpleNamespace(action='stop',message='Tool loop detected (write_file). Stopping.')))
    dispatcher=RoundDispatcher(driver=driver,response_interpreter=interpreter,completion_handler=None,handoff_router=handoff,safety=safety,tool_executor=None,run_context=None,use_native_tools=True,consult_advisor=None,should_final_check=lambda state:False)
    state=SimpleNamespace(status='running',result='')
    result=await dispatcher.dispatch(state,[],round_index=0,cancel_event=None)
    assert result.action=='stop'
    assert state.status=='error'


@pytest.mark.asyncio
async def test_acceptance_tool_fails_closed_and_is_not_cached():
    from clawagents.tools.coordination import FinishCoordinationTool
    from clawagents.tools.registry import ToolRegistry
    accepted=False
    registry=ToolRegistry()
    registry.register(FinishCoordinationTool(lambda:accepted))
    result=await registry.execute_tool('finish_coordination',{'summary':'ready'})
    assert not result.success and not result.return_direct
    accepted=True
    result=await registry.execute_tool('finish_coordination',{'summary':'ready'})
    assert result.success and result.return_direct
    accepted=False
    result=await registry.execute_tool('finish_coordination',{'summary':'ready'})
    assert not result.success


@pytest.mark.asyncio
async def test_verified_terminal_action_ends_without_another_model_call(tmp_path):
    from clawagents import create_claw_agent
    from clawagents.providers.llm import LLMResponse,NativeToolCall,LLMProvider
    class Worker(LLMProvider):
        model='gemma4-agentic-v2'
        calls=0
        async def chat(self,*args,**kwargs):
            self.calls+=1
            assert self.calls==1, 'Terminal action must not trigger another model call'
            return LLMResponse('',self.model,0,tool_calls=[NativeToolCall('finish_coordination',{'summary':'Verified'},'finish-1')])
    llm=Worker()
    agent=create_claw_agent(model=llm,profile='gemma-agentic',completion_check=lambda:True,workspace=tmp_path,skills=[],memory=[],tool_discovery=False)
    result=await agent.invoke('Complete the verified job',max_iterations=3)
    assert result.status=='done' and result.result=='Verified'
    assert llm.calls==1


@pytest.mark.asyncio
async def test_child_cannot_access_parent_terminal_action():
    from clawagents.tools.coordination import FinishCoordinationTool
    from clawagents.tools.registry import ToolRegistry
    registry=ToolRegistry()
    registry.register(FinishCoordinationTool(lambda:True))
    seen={}
    async def run(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(status='done',result='worker done',tool_calls=0,iterations=1)
    with patch('clawagents.graph.agent_loop.run_agent_graph',run):
        await TaskTool(None,registry).execute({'description':'work'})
    assert seen['tools'].get('finish_coordination') is None
    assert registry.get('finish_coordination') is not None


def test_failed_tool_outputs_are_not_reused_as_success():
    from clawagents.graph.loop_tracker import _ToolCallTracker
    tracker=_ToolCallTracker()
    args={'agent':'worker','description':'work'}
    tracker.record_result('task',args,'Temporary error',success=False)
    assert tracker.reuse_tool_output('task',args) is None
    tracker.record_result('task',args,'Completed',success=True)
    assert tracker.reuse_tool_output('task',args) is not None


def test_terminal_acceptance_is_always_rechecked():
    from clawagents.graph.loop_tracker import _ToolCallTracker
    tracker=_ToolCallTracker()
    tracker.record_result('finish_coordination',{'summary':'ready'},'Accepted')
    assert tracker.reuse_tool_output('finish_coordination',{'summary':'ready'}) is None


@pytest.mark.asyncio
async def test_real_loop_retries_failed_worker_and_finishes(tmp_path):
    from clawagents import create_claw_agent
    from clawagents.providers.llm import LLMProvider,LLMResponse,NativeToolCall
    class Worker(LLMProvider):
        model='fixture-worker'
        calls=0
        async def chat(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:
                raise RuntimeError('Temporary worker outage')
            return LLMResponse('Verified output',self.model,0)
    worker=Worker()
    class Coordinator(LLMProvider):
        model='gemma4-agentic-v2'
        calls=0
        async def chat(self,*args,**kwargs):
            self.calls+=1
            if self.calls<=2:
                call=NativeToolCall('task',{'agent':'worker','description':'work'},str(self.calls))
            else:
                call=NativeToolCall('finish_coordination',{'summary':'Recovered and verified'},'finish')
            return LLMResponse('',self.model,0,tool_calls=[call])
    agent=create_claw_agent(model=Coordinator(),profile='gemma-agentic',workspace=tmp_path,skills=[],memory=[],tool_discovery=False,
                           subagents=[SubAgentSpec('worker','Worker',llm=worker)],completion_check=lambda:worker.calls==2)
    result=await agent.invoke('Do the work',max_iterations=4)
    assert worker.calls==2
    assert result.status=='done'
    assert result.result=='Recovered and verified'
