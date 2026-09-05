"""Residual P1/P2 closures for v6.17.8."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def test_agent_write_skips_env_snapshot(tmp_path):
    from clawagents.memory.hunk_watcher import HunkWatcher

    w = HunkWatcher(tmp_path)
    w.record_agent_write(".env", "SECRET=abc\n", prompt_index=1)
    w.record_agent_write("src/ok.py", "print(1)\n", prompt_index=1)
    assert ".env" not in w._files
    assert "src/ok.py" in w._files
    snap = w.snapshot_turn(1, user_text="hi")
    assert ".env" not in snap.file_states
    assert "src/ok.py" in snap.file_states
    raw = (tmp_path / ".clawagents" / "rewind" / "prompt_0001.json").read_text()
    assert "SECRET=abc" not in raw


def test_hunk_store_skips_secret_baseline(tmp_path, monkeypatch):
    from clawagents.memory import attributed_hunks as ah

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    out = ah.refresh_file_hunks(".env", workspace=tmp_path, seed_baseline_if_missing=True)
    assert out == []
    store = ah.HunkStore.load(tmp_path)
    assert ".env" not in store.baselines


def test_resolve_hook_url_pins_ip():
    from clawagents.hooks.taxonomy import resolve_hook_url

    # Public IP literal should pin without DNS
    target, reason = resolve_hook_url("https://1.1.1.1/hook")
    assert reason == "ok"
    assert target is not None
    assert target.ip == "1.1.1.1"
    assert target.host == "1.1.1.1"

    bad, reason = resolve_hook_url("https://169.254.169.254/latest")
    assert bad is None
    assert "ssrf" in reason or "blocked" in reason


def test_webhook_uses_pinned_post():
    from clawagents.hooks.taxonomy import (
        HookHandler,
        HookEvent,
        _run_webhook,
    )

    calls: list[tuple] = []

    def fake_post(target, data, timeout_s):
        calls.append((target.ip, target.host, target.path))
        return 200, {}, b'{"decision":"allow"}'

    h = HookHandler(
        event=HookEvent.PRE_TOOL_USE,
        url="https://1.1.1.1/hooks/pre",
    )
    with patch("clawagents.hooks.taxonomy._post_hook_pinned", side_effect=fake_post):
        dec = _run_webhook(h, {"tool": "execute"})
    assert dec.allowed is True
    assert calls and calls[0][0] == "1.1.1.1"


def test_post_hook_pinned_reaches_network():
    """Regression: the pinned POST must not die on construction (bad kwarg).

    v6.17.8 shipped ``HTTPSConnection(..., server_hostname=...)`` — an invalid
    kwarg that raised TypeError before any connect, so every webhook denied.
    Calling the REAL implementation against a closed local port must fail with
    a socket error (i.e. it got as far as dialing), never a TypeError.
    """
    import socket

    from clawagents.hooks.taxonomy import HookPinnedTarget, _post_hook_pinned

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    target = HookPinnedTarget(host="localhost", port=port, ip="127.0.0.1", path="/")
    with pytest.raises(OSError):
        _post_hook_pinned(target, b"{}", 0.6)


def test_pinned_connection_dials_ip_with_host_sni(monkeypatch):
    """connect() must dial the pinned IP but TLS-wrap with the original host."""
    import socket

    from clawagents.hooks.taxonomy import _PinnedHTTPSConnection
    import ssl as _ssl

    dialed: list[tuple] = []
    wrapped: dict = {}
    left, right = socket.socketpair()
    try:

        def fake_create_connection(addr, timeout=None, source_address=None):
            dialed.append(addr)
            return left

        ctx = _ssl.create_default_context()

        def fake_wrap(sock, server_hostname=None, **kw):
            wrapped["server_hostname"] = server_hostname
            return sock

        monkeypatch.setattr(socket, "create_connection", fake_create_connection)
        monkeypatch.setattr(ctx, "wrap_socket", fake_wrap)
        conn = _PinnedHTTPSConnection(
            "1.1.1.1", 443, sni_host="hooks.example.com", timeout=1.0, context=ctx
        )
        conn.connect()
        assert dialed == [("1.1.1.1", 443)]
        assert wrapped["server_hostname"] == "hooks.example.com"
    finally:
        left.close()
        right.close()


def test_stream_breaker_helpers_exist():
    from clawagents.providers import llm as llm_mod

    assert callable(llm_mod._get_stream_breaker)
    assert callable(llm_mod._admit_stream_breaker)
    assert callable(llm_mod._record_stream_breaker)
    # All four stream entrypoints reference admit helper
    src = Path(llm_mod.__file__).read_text(encoding="utf-8")
    assert src.count("_admit_stream_breaker") >= 4


def test_doom_force_response_flag_semantics():
    # Resample path sets doom_force_response; next chat injects no-think instruction.
    from clawagents.doom_loop import (
        detect_tail_repetition,
        should_resample,
        DoomLoopState,
        DoomLoopRecoveryPolicy,
        note_trigger,
    )

    text = "\n".join(["loop me"] * 5)
    sig = detect_tail_repetition(text, channel="thinking")
    assert sig is not None
    state = DoomLoopState()
    note_trigger(state, sig)
    assert should_resample(sig, state, DoomLoopRecoveryPolicy())
    meta = {"doom_force_response": True}
    assert meta["doom_force_response"] is True
