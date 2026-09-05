"""Configurable built-in base system prompt.

Precedence (highest first):
    1. ``base_prompt=`` parameter (inline text or path to a file)
    2. ``CLAW_BASE_PROMPT_FILE`` env var (path)
    3. ``CLAW_BASE_PROMPT`` env var (inline text)
    4. ``<workspace>/.clawagents/base-prompt.md``
    5. ``~/.clawagents/base-prompt.md``
    6. built-in ``DEFAULT_BASE_SYSTEM_PROMPT``

``instruction=`` / ``system_prompt=`` keep *replacing* the base prompt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clawagents.prompts.base import (
    DEFAULT_BASE_SYSTEM_PROMPT,
    resolve_base_system_prompt,
)
from clawagents.providers.llm import LLMProvider

_ENV_KEYS = (
    "CLAW_BASE_PROMPT", "CLAW_BASE_PROMPT_FILE",
    "CLAW_BASE_PROMPT_APPEND", "CLAW_BASE_PROMPT_APPEND_FILE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Isolate the user-level ``~/.clawagents/base-prompt.md`` fallback.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    yield


class _StubLLM(LLMProvider):
    name = "stub"
    model = "stub-model"

    def __init__(self, model: str = "stub-model"):
        self.model = model

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None, **kwargs):
        raise NotImplementedError()


# ─── resolver ───────────────────────────────────────────────────────────


def test_default_is_builtin_prompt(tmp_path: Path):
    text = resolve_base_system_prompt(workspace=tmp_path)
    assert text == DEFAULT_BASE_SYSTEM_PROMPT
    assert text.startswith("You are a ClawAgent")


def test_explicit_inline_override_wins(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_BASE_PROMPT", "from env")
    assert resolve_base_system_prompt("inline text", workspace=tmp_path) == "inline text"


def test_explicit_path_override_reads_file(tmp_path: Path):
    f = tmp_path / "my-prompt.md"
    f.write_text("from file\n", encoding="utf-8")
    assert resolve_base_system_prompt(f, workspace=tmp_path) == "from file"
    assert resolve_base_system_prompt(str(f), workspace=tmp_path) == "from file"


def test_explicit_empty_string_disables_base(tmp_path: Path):
    assert resolve_base_system_prompt("", workspace=tmp_path) == ""


def test_env_inline_overrides_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_BASE_PROMPT", "env prompt")
    assert resolve_base_system_prompt(workspace=tmp_path) == "env prompt"


def test_env_file_beats_env_inline(tmp_path: Path, monkeypatch):
    f = tmp_path / "p.md"
    f.write_text("env file prompt", encoding="utf-8")
    monkeypatch.setenv("CLAW_BASE_PROMPT", "env inline")
    monkeypatch.setenv("CLAW_BASE_PROMPT_FILE", str(f))
    assert resolve_base_system_prompt(workspace=tmp_path) == "env file prompt"


def test_workspace_file_used_when_no_env(tmp_path: Path):
    (tmp_path / ".clawagents").mkdir()
    (tmp_path / ".clawagents" / "base-prompt.md").write_text("workspace prompt", encoding="utf-8")
    assert resolve_base_system_prompt(workspace=tmp_path) == "workspace prompt"


def test_env_inline_beats_workspace_file(tmp_path: Path, monkeypatch):
    (tmp_path / ".clawagents").mkdir()
    (tmp_path / ".clawagents" / "base-prompt.md").write_text("workspace prompt", encoding="utf-8")
    monkeypatch.setenv("CLAW_BASE_PROMPT", "env inline")
    assert resolve_base_system_prompt(workspace=tmp_path) == "env inline"


def test_user_home_file_used_when_no_workspace_file(tmp_path: Path):
    home = tmp_path / "home" / ".clawagents"
    home.mkdir(parents=True)
    (home / "base-prompt.md").write_text("home prompt", encoding="utf-8")
    assert resolve_base_system_prompt(workspace=tmp_path) == "home prompt"


def test_append_param_is_added_after_base(tmp_path: Path):
    text = resolve_base_system_prompt("base", append="extra rule", workspace=tmp_path)
    assert text == "base\n\nextra rule"


def test_append_env_is_added_after_base(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_BASE_PROMPT_APPEND", "env extra")
    text = resolve_base_system_prompt("base", workspace=tmp_path)
    assert text == "base\n\nenv extra"


def test_missing_env_file_falls_through_to_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_BASE_PROMPT_FILE", str(tmp_path / "nope.md"))
    assert resolve_base_system_prompt(workspace=tmp_path) == DEFAULT_BASE_SYSTEM_PROMPT


# ─── create_claw_agent composition ──────────────────────────────────────


def test_agent_loop_still_exports_base_system_prompt_alias():
    from clawagents.graph.agent_loop import BASE_SYSTEM_PROMPT

    assert BASE_SYSTEM_PROMPT == DEFAULT_BASE_SYSTEM_PROMPT


def test_create_agent_without_instruction_uses_resolved_base(tmp_path: Path, monkeypatch):
    from clawagents.agent import create_claw_agent

    monkeypatch.setenv("CLAW_BASE_PROMPT", "env base prompt")
    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path)
    assert agent.system_prompt == "env base prompt"


def test_create_agent_base_prompt_param(tmp_path: Path):
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, base_prompt="param base")
    assert agent.system_prompt == "param base"


def test_create_agent_base_prompt_append_param(tmp_path: Path):
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent(
        model=_StubLLM(), workspace=tmp_path, base_prompt="param base",
        base_prompt_append="more rules",
    )
    assert agent.system_prompt == "param base\n\nmore rules"


def test_instruction_replaces_base_prompt(tmp_path: Path):
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent(
        model=_StubLLM(), workspace=tmp_path,
        base_prompt="param base", instruction="You are a reviewer.",
    )
    assert agent.system_prompt == "You are a reviewer."
    assert "param base" not in agent.system_prompt


def test_harness_suffix_keeps_base_prompt_when_no_instruction(tmp_path: Path):
    """Regression: default gpt-5.6 runs used to ship only the Luna suffix."""
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent(model=_StubLLM("gpt-5.6-terra"), workspace=tmp_path)
    assert agent.system_prompt.startswith("You are a ClawAgent")
    assert "Efficiency rules (follow strictly)" in agent.system_prompt


def test_mode_instruction_keeps_base_prompt_when_no_instruction(tmp_path: Path):
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, mode="ask")
    assert "Prefer explaining and answering questions" in agent.system_prompt
    assert "You are a ClawAgent" in agent.system_prompt


def test_mode_instruction_still_prepends_to_user_instruction(tmp_path: Path):
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent(
        model=_StubLLM(), workspace=tmp_path, mode="ask", instruction="You are a reviewer.",
    )
    assert agent.system_prompt.startswith("Prefer explaining")
    assert agent.system_prompt.endswith("You are a reviewer.")
    assert "You are a ClawAgent" not in agent.system_prompt


def test_direct_agent_loop_fallback_honours_env(tmp_path: Path, monkeypatch):
    """``ClawAgent`` built directly (no create_claw_agent) still picks up env."""
    from clawagents.graph import run_bootstrapper

    monkeypatch.setenv("CLAW_BASE_PROMPT", "env base prompt")
    monkeypatch.chdir(tmp_path)
    assert run_bootstrapper._default_base_prompt() == "env base prompt"


# ─── append mechanisms ──────────────────────────────────────────────────
#
# Precedence (highest first):
#     1. ``base_prompt_append=`` parameter (inline text or file path)
#     2. ``CLAW_BASE_PROMPT_APPEND_FILE`` env var (path)
#     3. ``CLAW_BASE_PROMPT_APPEND`` env var (inline text)
#     4. ``<workspace>/.clawagents/base-prompt-append.md``
#     5. ``~/.clawagents/base-prompt-append.md``


def test_append_param_path_reads_file(tmp_path: Path):
    f = tmp_path / "extra.md"
    f.write_text("file extra\n", encoding="utf-8")
    assert resolve_base_system_prompt("base", append=str(f), workspace=tmp_path) == "base\n\nfile extra"


def test_append_env_file_beats_env_inline(tmp_path: Path, monkeypatch):
    f = tmp_path / "extra.md"
    f.write_text("env file extra", encoding="utf-8")
    monkeypatch.setenv("CLAW_BASE_PROMPT_APPEND", "env inline extra")
    monkeypatch.setenv("CLAW_BASE_PROMPT_APPEND_FILE", str(f))
    assert resolve_base_system_prompt("base", workspace=tmp_path) == "base\n\nenv file extra"


def test_append_workspace_file_used_when_no_env(tmp_path: Path):
    (tmp_path / ".clawagents").mkdir()
    (tmp_path / ".clawagents" / "base-prompt-append.md").write_text("workspace extra", encoding="utf-8")
    assert resolve_base_system_prompt("base", workspace=tmp_path) == "base\n\nworkspace extra"


def test_append_env_inline_beats_workspace_file(tmp_path: Path, monkeypatch):
    (tmp_path / ".clawagents").mkdir()
    (tmp_path / ".clawagents" / "base-prompt-append.md").write_text("workspace extra", encoding="utf-8")
    monkeypatch.setenv("CLAW_BASE_PROMPT_APPEND", "env inline extra")
    assert resolve_base_system_prompt("base", workspace=tmp_path) == "base\n\nenv inline extra"


def test_append_home_file_used_when_no_workspace_file(tmp_path: Path):
    home = tmp_path / "home" / ".clawagents"
    home.mkdir(parents=True)
    (home / "base-prompt-append.md").write_text("home extra", encoding="utf-8")
    assert resolve_base_system_prompt("base", workspace=tmp_path) == "base\n\nhome extra"


def test_base_file_and_append_file_compose(tmp_path: Path):
    d = tmp_path / ".clawagents"
    d.mkdir()
    (d / "base-prompt.md").write_text("workspace base", encoding="utf-8")
    (d / "base-prompt-append.md").write_text("workspace extra", encoding="utf-8")
    assert resolve_base_system_prompt(workspace=tmp_path) == "workspace base\n\nworkspace extra"


def test_append_applies_on_top_of_instruction(tmp_path: Path):
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent(
        model=_StubLLM(), workspace=tmp_path,
        instruction="You are a reviewer.", base_prompt_append="Answer in French.",
    )
    assert agent.system_prompt == "You are a reviewer.\n\nAnswer in French."


def test_append_workspace_file_applies_on_top_of_instruction(tmp_path: Path):
    from clawagents.agent import create_claw_agent

    (tmp_path / ".clawagents").mkdir()
    (tmp_path / ".clawagents" / "base-prompt-append.md").write_text("workspace extra", encoding="utf-8")
    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, instruction="You are a reviewer.")
    assert agent.system_prompt == "You are a reviewer.\n\nworkspace extra"
    assert "You are a ClawAgent" not in agent.system_prompt


def test_cli_flags_reach_cmd_task(monkeypatch):
    """``--base-prompt`` / ``--base-prompt-append`` are plumbed into cmd_task."""
    import clawagents.__main__ as m

    seen: dict = {}

    async def fake_cmd_task(task, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(m, "cmd_task", fake_cmd_task)
    monkeypatch.setattr(
        "sys.argv",
        ["clawagents", "--task", "hi", "--base-prompt", "B", "--base-prompt-append", "A"],
    )
    m.main()
    assert seen["base_prompt"] == "B"
    assert seen["base_prompt_append"] == "A"
