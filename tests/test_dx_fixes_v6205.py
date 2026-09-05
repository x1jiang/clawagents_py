"""DX / usability fixes (v6.20.5).

- block_tools/allow_only_tools/truncate_output must COMPOSE with an existing
  hook, never replace it — replacing wiped out the permission/plan-mode gate
  installed by create_claw_agent (a "safety" call that made the agent unsafe).
- `clawagents evals` passed a nonexistent invoke(trajectory=) kwarg → TypeError
  on every case; trajectory belongs at construction.
- set_overrides warns on unknown flag names (silent typos were no-ops).
"""

from __future__ import annotations

import inspect


from clawagents.agent import ClawAgent, create_claw_agent
from clawagents.graph.agent_loop import HookResult
from clawagents.tools.registry import ToolResult


def _agent_with_gate(gate):
    obj = ClawAgent.__new__(ClawAgent)
    obj.before_tool = gate
    obj.after_tool = None
    return obj


def _denied(r):
    return r is False or getattr(r, "allowed", None) is False


def _allowed(r):
    return r is True or getattr(r, "allowed", None) is True


def test_block_tools_preserves_existing_gate():
    def perm_gate(name, args):
        if name == "write_file" and str(args.get("path", "")).endswith(".env"):
            return HookResult(allowed=False, reason="secret denied")
        return True

    a = _agent_with_gate(perm_gate)
    a.block_tools("execute")
    assert _denied(a.before_tool("execute", {}))  # newly blocked
    assert _denied(a.before_tool("write_file", {"path": "app/.env"}))  # gate intact
    assert _allowed(a.before_tool("write_file", {"path": "src/x.py"}))


def test_allow_only_tools_preserves_existing_gate():
    def perm_gate(name, args):
        if name == "read_file" and str(args.get("path", "")).endswith(".env"):
            return HookResult(allowed=False, reason="secret denied")
        return True

    a = _agent_with_gate(perm_gate)
    a.allow_only_tools("read_file", "grep")
    assert _denied(a.before_tool("execute", {}))  # not in allow-set
    assert _denied(a.before_tool("read_file", {"path": ".env"}))  # gate still denies
    assert _allowed(a.before_tool("read_file", {"path": "README.md"}))


def test_block_tools_without_prior_gate_still_blocks():
    a = _agent_with_gate(None)
    a.block_tools("execute")
    assert _denied(a.before_tool("execute", {}))
    assert _allowed(a.before_tool("read_file", {"path": "x"}))


def test_truncate_output_chains_existing_after_tool():
    calls = []

    def existing(name, args, result):
        calls.append(name)
        return result

    a = _agent_with_gate(None)
    a.after_tool = existing
    a.truncate_output(10)
    out = a.after_tool("t", {}, ToolResult(success=True, output="x" * 50, error=None))
    assert calls == ["t"], "existing after_tool must still run"
    assert "truncated" in out.output and len(out.output) < 50


def test_evals_cli_uses_construction_trajectory_not_invoke_kwarg():
    # invoke() must NOT accept trajectory; create_claw_agent must.
    assert "trajectory" not in inspect.signature(ClawAgent.invoke).parameters
    assert "trajectory" in inspect.signature(create_claw_agent).parameters
    # And the eval harness must call it the fixed way.
    import clawagents.evals_cli as ec

    src = inspect.getsource(ec._run_case)
    assert "invoke(task, trajectory=True)" not in src
    assert "create_claw_agent(trajectory=True" in src


def test_set_overrides_warns_on_unknown_flag(caplog):
    from clawagents.config import features as feat

    feat.reset()
    with caplog.at_level("WARNING"):
        feat.set_overrides({"micro_compat": True})  # typo for micro_compact
    assert any("unknown feature flag" in r.message for r in caplog.records)
    feat.reset()


def test_set_overrides_known_flag_no_warning(caplog):
    from clawagents.config import features as feat

    feat.reset()
    with caplog.at_level("WARNING"):
        feat.set_overrides({"micro_compact": True})
    assert not any("unknown feature flag" in r.message for r in caplog.records)
    assert feat.is_enabled("micro_compact") is True
    feat.reset()
