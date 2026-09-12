import pytest
from clawagents.graph.model_profiles import resolve_model_profile, resolve_long_context_threshold

@pytest.mark.parametrize("model,window,output", [
 ("grok-4.6",500000,None), ("gemini-3.8-flash",1048576,65536),
 ("gpt-5.5",1050000,128000), ("gpt-5.4",1050000,128000),
 ("gpt-5.4-mini",400000,None), ("gpt-5.4-nano",400000,None),
 ("claude-fable-5-1",1000000,128000),("claude-opus-5",1000000,128000),
 ("claude-sonnet-5",1000000,128000),
])
def test_current_model_limits(model,window,output):
 p=resolve_model_profile(model)
 assert p["max_input_tokens"]==window
 if output:
  assert p["max_output_tokens"]==output

@pytest.mark.parametrize("model",["gpt-5.5","gpt-5.4","openai.gpt-5.4-2026-03-05"])
def test_current_openai_cliffs(model):
 assert resolve_long_context_threshold(model)==272000

@pytest.mark.parametrize("model,window,output",[('minimax.minimax-m2.5', 196000, 8000), ('mistral.devstral-2-123b', 256000, 32000), ('qwen.qwen3-coder-next', 256000, 16000), ('nvidia.nemotron-super-3-120b', 256000, 32000), ('mistral.mistral-large-3-675b-instruct', 256000, 32000), ('us.amazon.nova-2-lite-v1:0', 1000000, 65000), ('us.meta.llama4-maverick-17b-instruct-v1:0', 1000000, 8000)])
def test_new_aws_profiles(model,window,output):
 from clawagents.providers.model_classify import is_bedrock_model_id
 assert is_bedrock_model_id(model)
 profile=resolve_model_profile(model)
 assert profile["max_input_tokens"]==window
 assert profile["max_output_tokens"]==output
