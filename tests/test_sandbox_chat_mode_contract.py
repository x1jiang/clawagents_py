"""Chat mode ↔ OS sandbox contract + seatbelt diagnostics."""

from __future__ import annotations

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
    assert desktop.is_file(), f"missing desktop profiles: {desktop}"
    text = desktop.read_text(encoding="utf-8")
    assert 'allow file-write-data (literal "/dev/null")' in text
    assert "sandbox_profile_for_chat_mode" in text
