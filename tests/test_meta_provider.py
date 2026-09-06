"""Meta profile routing and context regression coverage (no network)."""
import json

from clawagents.agent import create_claw_agent
from clawagents.config.config import EngineConfig, get_default_model
from clawagents.graph.model_profiles import resolve_model_profile
from clawagents.harness_profiles import resolve_harness_profile
from clawagents.provider_profiles import load_provider_profiles, resolve_provider_profile


def test_meta_defaults_isolate_cloud_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "cloud-key-must-not-be-forwarded")
    agent = create_claw_agent(profile="meta", workspace=tmp_path, skills=[], memory=[])
    assert agent.llm.model == "Muse-Glimmer-30B"
    assert str(agent.llm.client.base_url) == "http://129.106.31.72:7790/v1/"
    assert agent.llm.client.api_key == "not-needed"
    assert not agent.llm._should_use_responses(has_tools=True)
    assert "Tool efficiency:" in agent.system_prompt
    assert agent.tools.is_tool_active("read_file")
    assert agent.tools.is_tool_active("activate_tool_group")
    assert not agent.tools.is_tool_active("web_fetch")


def test_meta_env_and_explicit_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("glimmer_30B_backend", "http://localhost:9000/v1")
    monkeypatch.setenv("glimmer_30B_model", "custom-glimmer")
    monkeypatch.setenv("META_API_KEY", "meta-only")
    p = resolve_provider_profile("meta")
    assert (p.model, p.base_url, p.api_key) == ("custom-glimmer", "http://localhost:9000/v1", "meta-only")
    p = resolve_provider_profile("meta", model="override", base_url="http://localhost:9001/v1", api_key="explicit")
    assert (p.model, p.base_url, p.api_key) == ("override", "http://localhost:9001/v1", "explicit")


def test_meta_profile_file_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("glimmer_30B_model", "env-model")
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"meta": {"model": "file-model", "base_url": "http://localhost/v1"}}))
    assert load_provider_profiles([path])["meta"].model == "file-model"


def test_meta_provider_env_selects_profile(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROVIDER", "meta")
    assert get_default_model(EngineConfig()) == "Muse-Glimmer-30B"
    agent = create_claw_agent(workspace=tmp_path, skills=[], memory=[])
    assert agent.llm.model == "Muse-Glimmer-30B"
    assert agent.llm.client.api_key == "not-needed"
    assert str(agent.llm.client.base_url) == "http://129.106.31.72:7790/v1/"


def test_meta_context_and_harness():
    assert resolve_model_profile("Muse-Glimmer-30B")["max_input_tokens"] == 196_608
    p = resolve_harness_profile("Muse-Glimmer-30B")
    assert p.name == "meta-glimmer"
    assert p.clear_tool_keep == 2
    assert resolve_harness_profile("gpt-5.6-luna").name == "openai-gpt56"


def test_bare_glimmer_uses_meta_and_explicit_args_win(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    agent = create_claw_agent(
        "Muse-Glimmer-30B", base_url="http://localhost:9900/v1", api_key="dedicated",
        workspace=tmp_path, skills=[], memory=[],
    )
    assert agent.llm.client.api_key == "dedicated"
    assert str(agent.llm.client.base_url) == "http://localhost:9900/v1/"


def test_meta_empty_key_never_falls_back_to_cloud(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "cloud-secret")
    agent = create_claw_agent(profile="meta", api_key="", workspace=tmp_path, skills=[], memory=[])
    assert agent.llm.client.api_key == "not-needed"
