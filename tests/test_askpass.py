"""Tests for the interactive AskPass bridge (`clawagents.tools.askpass` and `ExecTool`)."""

from __future__ import annotations

import os
import sys

import pytest

from clawagents.sandbox.local import LocalBackend
from clawagents.tools.askpass import (
    AskpassSession,
    is_secret_prompt,
    looks_like_secret_question,
    rewrite_command_for_askpass,
)
from clawagents.tools.exec import ExecTool
from clawagents.tools.interactive import AskUserTool


def test_is_secret_prompt_distinguishes_passwords_and_host_keys() -> None:
    assert is_secret_prompt("xjiang2@129.106.31.72's password:") is True
    assert is_secret_prompt("Enter passphrase for key '/Users/x/.ssh/id_ed25519':") is True
    assert is_secret_prompt("[sudo] password for xjiang2:") is True
    assert (
        is_secret_prompt(
            "The authenticity of host '129.106.31.72 (129.106.31.72)' can't be established.\n"
            "ED25519 key fingerprint is SHA256:abc.\n"
            "Are you sure you want to continue connecting (yes/no/[fingerprint])?"
        )
        is False
    )


def test_looks_like_secret_question() -> None:
    assert looks_like_secret_question("Please enter your SSH password for .72") is True
    assert looks_like_secret_question("What is your sudo password?") is True
    assert looks_like_secret_question("Which directory should I use?") is False


def test_rewrite_command_for_askpass_normalizes_ssh_batchmode_and_sudo() -> None:
    cmd = 'ssh -o BatchMode=yes -o ConnectTimeout=5 xjiang2@129.106.31.72 "hostname"'
    rewritten = rewrite_command_for_askpass(cmd, has_askpass=True)
    assert "BatchMode=no" in rewritten
    assert "BatchMode=yes" not in rewritten

    # Without askpass, BatchMode=yes is preserved.
    assert rewrite_command_for_askpass(cmd, has_askpass=False) == cmd

    # Bare sudo is rewritten to sudo -A when askpass is enabled.
    assert (
        rewrite_command_for_askpass("sudo apt-get update", has_askpass=True)
        == "sudo -A apt-get update"
    )
    assert (
        rewrite_command_for_askpass("sudo -n true", has_askpass=True)
        == "sudo -n true"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell askpass test")
@pytest.mark.asyncio
async def test_exec_tool_routes_ssh_askpass_prompt_without_leaking_secret(tmp_path) -> None:
    captured_prompts: list[tuple[str, bool, str | None]] = []

    def fake_ask_prompt(
        prompt: str,
        *,
        secret: bool = True,
        command: str | None = None,
    ) -> str | None:
        captured_prompts.append((prompt, secret, command))
        return "s3cr3t-ssh-pass!"

    class DummyRunContext:
        def __init__(self) -> None:
            self._metadata = {"ask_prompt_fn": fake_ask_prompt}

    sb = LocalBackend(str(tmp_path))
    tool = ExecTool(sb)
    ctx = DummyRunContext()

    # Simulate OpenSSH invoking $SSH_ASKPASS for a password and verifying it
    # internally without printing the password to stdout/stderr.
    cmd = (
        'pass=$("$SSH_ASKPASS" "xjiang2@129.106.31.72\'s password:") && '
        'if [ "$pass" = "s3cr3t-ssh-pass!" ]; then echo "AUTH_OK"; else echo "AUTH_FAIL"; exit 1; fi'
    )
    res = await tool.execute({"command": cmd}, run_context=ctx)
    assert res.success is True
    assert "AUTH_OK" in str(res.output)
    assert "s3cr3t-ssh-pass!" not in str(res.output)
    assert len(captured_prompts) == 1
    prompt_text, is_sec, orig_cmd = captured_prompts[0]
    assert "xjiang2@129.106.31.72's password:" in prompt_text
    assert is_sec is True
    assert orig_cmd == cmd


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell askpass test")
def test_askpass_session_cancellation_returns_nonzero() -> None:
    import subprocess

    def cancel_prompt(prompt: str, **kwargs) -> str | None:
        return None

    with AskpassSession(cancel_prompt, command="ssh host") as session:
        env = os.environ.copy()
        env.update(session.env_overlay())
        proc = subprocess.run(
            [session.script_path, "Password:"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == 1
        assert proc.stdout == ""


@pytest.mark.asyncio
async def test_ask_user_tool_passes_secret_flag() -> None:
    seen: list[tuple[str, bool]] = []

    def host_ask(question: str, *, secret: bool = False) -> str | None:
        seen.append((question, secret))
        return "done"

    tool = AskUserTool(ask_fn=host_ask)
    res = await tool.execute({"question": "Enter your SSH password:"})
    assert res.success is True
    assert seen == [("Enter your SSH password:", True)]
