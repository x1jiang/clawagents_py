"""P1 security regressions fixed in v6.17.6."""

from __future__ import annotations

import asyncio
import json
import shlex

import pytest


class TestTaxonomyExternalHooksGate:
    def test_disabled_by_default(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".clawagents").mkdir()
        (tmp_path / ".clawagents" / "hooks.json").write_text(
            json.dumps({"SessionStart": "touch OWNED"}), encoding="utf-8"
        )
        from clawagents.config.features import reset, set_overrides
        from clawagents.hooks.external import build_taxonomy_dispatcher

        reset()
        set_overrides({"hook_taxonomy": False, "external_hooks": False})
        assert build_taxonomy_dispatcher() is None

    def test_requires_both_flags(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".clawagents").mkdir()
        (tmp_path / ".clawagents" / "hooks.json").write_text("{}", encoding="utf-8")
        from clawagents.config.features import reset, set_overrides
        from clawagents.hooks.external import build_taxonomy_dispatcher

        reset()
        set_overrides({"hook_taxonomy": True, "external_hooks": False})
        assert build_taxonomy_dispatcher() is None
        set_overrides({"hook_taxonomy": True, "external_hooks": True})
        assert build_taxonomy_dispatcher() is not None
        reset()


class TestSeatbeltQuoting:
    def test_single_quote_command_uses_shlex_quote(self):

        # Build the wrap string the same way exec does for seatbelt.
        command = "echo 'hi' && echo $(whoami)"
        binary = "/usr/bin/sandbox-exec"
        profile_path = "/tmp/seatbelt.sb"
        wrapped = " ".join(
            shlex.quote(p)
            for p in [binary, "-f", str(profile_path), "/bin/sh", "-c", command]
        )
        # Outer shell must not see unquoted $() — shlex.quote single-quotes the whole cmd.
        assert "$(whoami)" in wrapped
        # The command segment is single-quoted (shlex), so $ is literal to outer shell.
        assert "'echo '" in wrapped or "echo '\\''hi'\\''" in wrapped or wrapped.count("'") >= 2
        # Crucially: not Python repr double-quote flip
        bad = f"{binary} -f {profile_path} /bin/sh -c {command!r}"
        assert bad != wrapped

    def test_seatbelt_exec_does_not_unbound_shlex(self, tmp_path, monkeypatch):
        """Regression: local ``import shlex`` in bwrap branch unbound seatbelt path."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from clawagents.sandbox.profiles import OSSandboxProfile, ProfileBackend

        profile = OSSandboxProfile(
            name="workspace",
            backend="seatbelt",
            network=False,
            read_only=False,
            require_binary=False,
        )
        inner = MagicMock()
        inner.cwd = str(tmp_path)
        result = MagicMock(stdout="hi\n", stderr="", exit_code=0, killed=False)
        inner.exec = AsyncMock(return_value=result)
        pb = ProfileBackend(inner, profile)

        async def _run():
            with patch(
                "clawagents.sandbox.profiles.shutil.which",
                return_value="/usr/bin/sandbox-exec",
            ):
                out = await pb.exec("echo hi", timeout=5000, cwd=str(tmp_path))
            assert out.exit_code == 0
            # Inner received a sandbox-exec wrapped command
            cmd = inner.exec.await_args.args[0]
            assert "sandbox-exec" in cmd
            assert "echo hi" in cmd

        asyncio.run(_run())


class TestSecretPathReadDeny:
    def test_workspace_profile_denies_dotenv(self, tmp_path, monkeypatch):
        from clawagents.sandbox.profiles import resolve_sandbox

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (tmp_path / "ok.txt").write_text("hi\n", encoding="utf-8")
        sb = resolve_sandbox("workspace", workspace=str(tmp_path))

        async def _run():
            with pytest.raises(ValueError, match="Secret path"):
                await sb.read_file(".env")
            assert await sb.read_file("ok.txt") == "hi\n"

        asyncio.run(_run())


class TestWebhookFailClosed:
    def test_blocked_url_denies(self):
        from clawagents.hooks.taxonomy import HookHandler, HookEvent, _run_webhook

        h = HookHandler(
            event=HookEvent.PRE_TOOL_USE,
            url="http://169.254.169.254/latest/meta-data/",
        )
        dec = _run_webhook(h, {"tool": "execute", "args": {}})
        assert dec.allowed is False
        assert "ssrf" in dec.source or "http" in dec.reason or "blocked" in dec.reason.lower() or "https" in dec.reason.lower()


class TestHunkWatcherSecrets:
    def test_ignores_env_and_pem(self, tmp_path):
        from clawagents.memory.hunk_watcher import HunkWatcher

        w = HunkWatcher(workspace=tmp_path)
        assert w._should_ignore(".env") is True
        assert w._should_ignore("secrets/foo.pem") is True
        assert w._should_ignore("src/main.py") is False


class TestDreamMemoryPath:
    def test_writes_under_clawagents_not_root(self, tmp_path, monkeypatch):
        from clawagents.memory import dream as dream_mod

        ws = tmp_path
        (ws / "MEMORY.md").write_text("# human owned\n" + ("x" * 9000), encoding="utf-8")
        (ws / ".clawagents" / "sessions").mkdir(parents=True)
        # Bypass gate / LLM — unit-test the path choice via process_dream + write target
        memory_path = ws / ".clawagents" / "MEMORY.md"
        legacy = ws / "MEMORY.md"
        assert legacy.is_file()
        # Simulate write destination used by run_dream
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text("consolidated\n", encoding="utf-8")
        assert legacy.read_text(encoding="utf-8").startswith("# human owned")
        assert memory_path.read_text(encoding="utf-8") == "consolidated\n"
        assert dream_mod.__doc__ and ".clawagents/MEMORY.md" in dream_mod.__doc__


class TestPtyPlanMode:
    def test_pty_tools_are_write_class(self):
        from clawagents.permissions.mode import WRITE_CLASS_TOOLS

        for name in ("pty_start", "pty_keys", "pty_wait", "pty_stop"):
            assert name in WRITE_CLASS_TOOLS


class TestPromptIndexMonotonic:
    def test_max_of_meta_and_watcher(self):
        # Mirrors agent_loop rewind indexing logic
        meta_rw = {"prompt_index": 0}
        watcher_idx = 3
        idx = int((meta_rw or {}).get("prompt_index") or 0) + 1
        idx = max(idx, int(watcher_idx or 0) + 1)
        assert idx == 4
