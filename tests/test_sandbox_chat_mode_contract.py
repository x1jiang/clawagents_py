"""Chat mode ↔ OS sandbox contract + seatbelt diagnostics."""

from __future__ import annotations

import asyncio
import json

from clawagents.sandbox.profiles import (
    _seatbelt_profile_text,
    sandbox_profile_for_chat_mode,
)


def test_full_access_maps_to_off_when_gated():
    assert (
        sandbox_profile_for_chat_mode("full_access", allow_full_access=True) == "off"
    )
    assert (
        sandbox_profile_for_chat_mode("full_access", allow_full_access=False) is None
    )


def test_read_only_maps_to_readonly_profile():
    assert sandbox_profile_for_chat_mode("read_only") == "read-only"
    assert sandbox_profile_for_chat_mode("plan") == "read-only"


def test_explicit_profile_wins():
    assert (
        sandbox_profile_for_chat_mode(
            "full_access",
            allow_full_access=True,
            explicit="workspace",
        )
        == "workspace"
    )


def test_seatbelt_writable_allows_dev_null():
    text = _seatbelt_profile_text(cwd="/ws", network=True, read_only=False)
    assert '(allow file-write-data (literal "/dev/null"))' in text


def test_failed_tool_output_not_crushed():
    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

    blob = (
        "Unable to create private file ... ~/.config/gcloud/credentials.db\n"
        "/dev/null: Operation not permitted\n"
    ) * 80
    assert len(blob) > 2500
    out, aid = prepare_tool_output_for_context(
        tool_name="execute",
        tool_use_id="t1",
        output=blob,
        success=False,
    )
    assert "credentials.db" in out
    assert "[Crushed tool output" not in out
    assert aid is None or "Failed tool" in out or "credentials.db" in out


def test_gcloud_sandbox_failure_offers_private_scratch_config():
    from clawagents.tools.exec import _format_nonzero_command_output

    payload = json.loads(
        _format_nonzero_command_output(
            "gcloud auth list",
            1,
            "",
            "Unable to create private file ~/.config/gcloud/credentials.db: "
            "Operation not permitted",
            "",
        )
    )
    interpretation = payload["interpretation"]
    assert "CLOUDSDK_CONFIG" in interpretation
    assert "$TMPDIR" in interpretation
    assert "never commit" in interpretation
    assert "Do not retry the unchanged command" in interpretation


def test_unsandboxed_request_without_full_access_does_not_execute(tmp_path):
    from clawagents.run_context import RunContext
    from clawagents.tools.exec import ExecTool

    class ProfileStub:
        kind = "profile:test:local"
        cwd = str(tmp_path)

        def __init__(self):
            self.called = False

        def wrap_command(self, command, cwd=None):
            return command

        async def exec(self, *args, **kwargs):
            self.called = True
            raise AssertionError("command must not execute")

    backend = ProfileStub()
    result = asyncio.run(
        ExecTool(backend).execute(
            {"command": "gcloud auth list", "unsandboxed": True},
            run_context=RunContext(),
        )
    )

    assert result.success is False
    assert "unsandboxed_not_authorized" in (result.error or "")
    assert "command was not run" in (result.error or "")
    assert backend.called is False


def test_desktop_seatbelt_source_has_dev_null_allow():
    """Parity guard: desktop fork must not lag py on /dev/null allow."""
    from pathlib import Path

    workspace = Path(__file__).resolve().parents[2]  # openclawVSdeepagents/
    desktop = (
        workspace
        / "clawagents_desktop"
        / "backend"
        / "src"
        / "clawagents"
        / "sandbox"
        / "profiles.py"
    )
    import pytest
    if not desktop.is_file():
        pytest.skip(f"clawagents_desktop repository not present locally: {desktop}")
    text = desktop.read_text(encoding="utf-8")
    assert 'allow file-write-data (literal "/dev/null")' in text
    assert "sandbox_profile_for_chat_mode" in text
