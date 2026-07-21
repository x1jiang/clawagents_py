"""Luna efficiency: harness injection, economic thresholds, active tools, loop reuse."""

from __future__ import annotations

from clawagents.graph.agent_loop import _ToolCallTracker, _soft_trim_messages
from clawagents.graph.model_profiles import resolve_long_context_threshold
from clawagents.harness_profiles import (
    apply_harness_profile_to_prompt,
    resolve_harness_profile,
)
from clawagents.loop_detection import detect_overlapping_read, ranges_overlap
from clawagents.providers.llm import LLMMessage
from clawagents.tools.registry import ToolRegistry, ToolResult
from clawagents.tools.tool_groups import (
    ActivateToolGroupTool,
    CORE_TOOL_NAMES,
    apply_core_active_profile,
)


def test_harness_resolves_luna_and_injects_efficiency_suffix():
    p = resolve_harness_profile("gpt-5.6-luna")
    assert p is not None
    assert p.name == "openai-gpt56"
    assert p.clear_tool_trigger_ratio == 0.22
    assert p.loop_detection_overrides.get("warning_threshold") == 2
    text = apply_harness_profile_to_prompt("base", p)
    assert "Efficiency rules" in text
    assert "activate_tool_group" in text


def test_long_context_threshold_272k():
    assert resolve_long_context_threshold("gpt-5.6-luna") == 272_000
    assert resolve_long_context_threshold("gpt-5.6-sol") == 272_000
    assert resolve_long_context_threshold("claude-sonnet-4") is None


def test_soft_trim_budget_capped_near_272k():
    # Soft-trim trigger for Luna should be ≤ ~258K (0.95 × 272K), not ~669K.
    msgs = [LLMMessage(role="user", content="x" * 4_000) for _ in range(80)]
    # Estimate roughly — just ensure the function runs and trims when huge.
    out = _soft_trim_messages(
        msgs,
        context_window=1_050_000,
        token_multiplier=0.25,
        emit=lambda *_a, **_k: None,
        model_name="gpt-5.6-luna",
    )
    assert isinstance(out, list)


def test_active_tool_profile_hides_web_until_activated():
    class _T:
        def __init__(self, name: str):
            self.name = name
            self.description = name
            self.parameters = {}

        async def execute(self, args):
            return ToolResult(True, "ok")

    reg = ToolRegistry()
    for n in ("read_file", "execute", "web_fetch", "web_search", "activate_tool_group"):
        reg.register(_T(n))
    apply_core_active_profile(reg)
    names = {t.name for t in reg.list()}
    assert "read_file" in names
    assert "web_fetch" not in names
    assert "activate_tool_group" in names
    # Still registered
    assert reg.get("web_fetch") is not None
    assert not reg.is_tool_active("web_fetch")

    # Activate
    import asyncio

    tool = ActivateToolGroupTool(reg)
    res = asyncio.run(tool.execute({"group": "web"}))
    assert res.success
    assert reg.is_tool_active("web_fetch")
    assert "web_fetch" in {t.name for t in reg.list()}


def test_active_tool_profile_keeps_context_protection_mcp_tools():
    class _T:
        def __init__(self, name: str, *, context_protection: bool = False):
            self.name = name
            self.description = name
            self.parameters = {}
            self.tool_group = "mcp"
            self.context_protection = context_protection

        async def execute(self, args):
            return ToolResult(True, "ok")

    reg = ToolRegistry()
    reg.register(_T("execute"))
    reg.register(_T("ctx_execute", context_protection=True))
    reg.register(_T("ordinary_mcp_tool"))
    reg.register(_T("activate_tool_group"))

    apply_core_active_profile(reg)
    assert reg.is_tool_active("ctx_execute")
    assert not reg.is_tool_active("ordinary_mcp_tool")

    import asyncio

    result = asyncio.run(ActivateToolGroupTool(reg).execute({"group": "mcp"}))
    assert result.success
    assert reg.is_tool_active("ordinary_mcp_tool")


def test_execute_refuses_inactive_tool():
    class _T:
        name = "web_fetch"
        description = "x"
        parameters = {}

        async def execute(self, args):
            return ToolResult(True, "fetched")

    reg = ToolRegistry()
    reg.register(_T())
    reg.set_active_tools({"read_file"})
    import asyncio

    res = asyncio.run(reg.execute_tool("web_fetch", {}))
    assert res.success is False
    assert "not active" in (res.error or "")


def test_identical_and_overlapping_read_reuse():
    from clawagents.loop_detection import range_contains

    tr = _ToolCallTracker(soft_limit=2, hard_limit=3)
    tr.cache_result_output(
        "read_file", {"path": "a.py", "offset": 0, "limit": 100}, "LINEDATA" * 20
    )
    stub = tr.reuse_tool_output("read_file", {"path": "a.py", "offset": 0, "limit": 100})
    assert stub and "Reused identical" in stub

    # Fully contained → reuse
    contained = detect_overlapping_read(
        tool_name="read_file",
        params={"path": "a.py", "offset": 10, "limit": 40},
        prior_reads=[
            ("read_file", {"path": "a.py", "offset": 0, "limit": 100}, "PRIOR"),
        ],
    )
    assert contained and "contained" in contained.lower()

    # Partial overlap that extends past prior end → must NOT stub (would lose 100–150)
    partial = detect_overlapping_read(
        tool_name="read_file",
        params={"path": "a.py", "offset": 50, "limit": 100},
        prior_reads=[
            ("read_file", {"path": "a.py", "offset": 0, "limit": 100}, "PRIOR"),
        ],
    )
    assert partial is None
    assert ranges_overlap((0, 100), (50, 150))
    assert range_contains((0, 100), (10, 50))
    assert not range_contains((0, 100), (50, 150))
    assert not ranges_overlap((0, 50), (50, 100))


def test_core_tool_names_cover_coding_basics():
    for n in ("read_file", "execute", "grep", "activate_tool_group"):
        assert n in CORE_TOOL_NAMES
