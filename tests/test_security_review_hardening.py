"""Regression tests for the security review hardening pass.

Covers: gateway bind/origin/auth, channel sender allowlists, the CodeAct
permission-gate bypass and snippet timeout, the seatbelt profile location,
and bash-validator bypass shapes.
"""

from __future__ import annotations

import inspect
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from clawagents.channels.auto import (
    load_sender_allowlists,
    make_sender_gate,
    start_channel_router,
)
from clawagents.channels.router import ChannelRouter
from clawagents.channels.types import ChannelMessage
from clawagents.gateway.server import create_app, origin_allowed, print_bind_warning
from clawagents.graph.codeact import run_code_action
from clawagents.tools.bash_validator import Decision, validate_bash
from clawagents.tools.registry import ToolRegistry


# ── Gateway: bind host ───────────────────────────────────────────────────


def test_serve_with_channels_does_not_hardcode_all_interfaces():
    from clawagents import __main__ as cli

    src = inspect.getsource(cli.cmd_serve)
    assert '"0.0.0.0"' not in src
    assert "resolve_bind_host" in src


def test_bind_warning_only_for_network_reachable_without_auth(capsys):
    assert print_bind_warning("127.0.0.1", False) is False
    assert print_bind_warning("0.0.0.0", True) is False
    assert print_bind_warning("0.0.0.0", False) is True
    assert "non-loopback" in capsys.readouterr().out


# ── Gateway: cross-origin requests ───────────────────────────────────────

_DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


@pytest.mark.parametrize(
    "origin,host,expected",
    [
        (None, "localhost:3000", True),  # curl / SDK: no Origin header
        ("http://localhost:3000", "localhost:3000", True),  # listed
        ("http://127.0.0.1:8123", "127.0.0.1:8123", True),  # same-origin loopback
        ("http://[::1]:8123", "[::1]:8123", True),
        ("https://evil.example", "localhost:3000", False),
        ("http://localhost:9999", "localhost:3000", False),  # other local app
        # DNS rebinding: attacker hostname resolves to 127.0.0.1, so Origin
        # and Host agree — must still be refused.
        ("http://evil.example:3000", "evil.example:3000", False),
        ("null", "localhost:3000", False),  # sandboxed iframe / file://
    ],
)
def test_origin_allowed(origin, host, expected):
    assert origin_allowed(origin, host, _DEFAULT_ORIGINS) is expected


def test_origin_allowed_wildcard_opt_in():
    assert origin_allowed("https://anything.example", "h:1", ["*"]) is True


@pytest.fixture
def gateway_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("GATEWAY_CORS_ORIGINS", raising=False)
    with patch("clawagents.gateway.server.create_claw_agent") as mock_create:
        agent = AsyncMock()
        result = AsyncMock()
        result.status = "done"
        result.result = "ok"
        result.iterations = 1
        agent.invoke.return_value = result
        mock_create.return_value = agent
        app, _, _ = create_app()
        yield TestClient(app), agent


def test_cross_origin_simple_post_is_refused(gateway_client):
    """A no-preflight ``text/plain`` POST from another site must not run a task."""
    client, agent = gateway_client
    resp = client.post(
        "/chat",
        content='{"task": "rm everything"}',
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
    )
    assert resp.status_code == 403
    assert not agent.invoke.called


def test_post_without_origin_still_works(gateway_client):
    client, agent = gateway_client
    resp = client.post("/chat", json={"task": "hello"})
    assert resp.status_code == 200
    assert agent.invoke.called


def test_websocket_cross_origin_is_refused(gateway_client):
    from starlette.websockets import WebSocketDisconnect

    client, _ = gateway_client
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws", headers={"Origin": "https://evil.example"}):
            pass
    assert exc.value.code == 4003


def test_websocket_without_origin_connects(gateway_client):
    client, _ = gateway_client
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "req", "id": "1", "method": "ping", "params": {}})
        assert ws.receive_json()["ok"] is True


def test_gateway_key_comparison_is_constant_time():
    from clawagents.gateway import server, ws

    assert "compare_digest" in inspect.getsource(server._keys_equal)
    assert "compare_digest" in inspect.getsource(ws.attach_websocket)
    assert server._keys_equal("secret", "secret")
    assert not server._keys_equal("secreu", "secret")
    assert not server._keys_equal("ключ", "secret")  # non-ASCII must not raise


# ── Channels: sender allowlist ───────────────────────────────────────────


def _msg(sender: str, conversation: str = "chat-1", channel: str = "telegram", body: str = "hi"):
    return ChannelMessage(
        channel_id=channel,
        sender_id=sender,
        conversation_id=conversation,
        body=body,
        timestamp=0.0,
    )


def test_allowlist_unset_denies_everyone(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_SENDERS", raising=False)
    log: list[str] = []
    gate = make_sender_gate(load_sender_allowlists(["telegram"]), log=log.append)
    assert gate(_msg("111")) is False
    assert gate(_msg("111")) is False
    assert len(log) == 1, "each refused sender is reported once"
    assert "111" in log[0] and "TELEGRAM_ALLOWED_SENDERS" in log[0]


def test_allowlist_matches_sender_or_conversation(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_SENDERS", " 111 , group-9 ")
    gate = make_sender_gate(load_sender_allowlists(["telegram"]), log=lambda _: None)
    assert gate(_msg("111")) is True
    assert gate(_msg("222", conversation="group-9")) is True
    assert gate(_msg("222")) is False
    # A list for one channel does not open another.
    assert gate(_msg("111", channel="signal")) is False


def test_allowlist_star_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_SENDERS", "*")
    gate = make_sender_gate(load_sender_allowlists(["signal"]), log=lambda _: None)
    assert gate(_msg("+15550000", channel="signal")) is True


async def test_auto_router_installs_the_gate(monkeypatch):
    """The auto-started router must refuse an unlisted sender before any agent is built."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:abc")
    monkeypatch.delenv("TELEGRAM_ALLOWED_SENDERS", raising=False)
    monkeypatch.delenv("WHATSAPP_AUTH_DIR", raising=False)
    monkeypatch.delenv("WHATSAPP_API_URL", raising=False)
    monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)

    class FakeTelegram:
        id = "telegram"
        on_message = None

        async def start(self, config):
            return None

        async def stop(self):
            return None

        async def send(self, *a, **k):
            return None

    built: list[int] = []
    with patch("clawagents.channels.telegram.TelegramAdapter", FakeTelegram), patch(
        "clawagents.agent.create_claw_agent", side_effect=lambda **k: built.append(1)
    ):
        router = await start_channel_router(llm=object())
        assert router is not None
        assert router._on_inbound is not None
        await router._dispatch(_msg("999"))
    assert built == [], "no agent may be created for a refused sender"


async def test_debounce_does_not_merge_across_senders():
    """A combined batch carries one sender id, so batches must be per sender."""
    seen: list[tuple[str, str]] = []

    def gate(msg: ChannelMessage) -> bool:
        seen.append((msg.sender_id, msg.body))
        return False

    async def factory():
        raise AssertionError("gate refuses everything")

    router = ChannelRouter(factory, on_inbound=gate, debounce_ms=20)
    router._handle_inbound(_msg("attacker", body="rm -rf ~"))
    router._handle_inbound(_msg("owner", body="hello"))
    for _ in range(100):
        if len(seen) == 2:
            break
        import asyncio

        await asyncio.sleep(0.01)
    assert sorted(seen) == [("attacker", "rm -rf ~"), ("owner", "hello")]


# ── CodeAct: permission gate cannot be disabled from the snippet ─────────


def _codeact(code: str, **kw):
    kw.setdefault("run_async", lambda c: None)
    return run_code_action(code, ToolRegistry(), **kw)


def _deny_all():
    calls: list[str] = []

    def deny(name, args):
        from clawagents import HookResult

        calls.append(name)
        return HookResult(allowed=False, reason="read-only")

    return deny, calls


@pytest.mark.parametrize(
    "code",
    [
        "tools._before_tool = None\nprint(tools.execute(command='id'))",
        "tools._run_async(tools._registry.execute_tool('execute', {'command': 'id'}))",
        "print(tools._invoke('execute', {'command': 'id'}))",
        "r = tools._registry",
        "tools._tool_seconds[0] = 1e9",
    ],
)
def test_private_attribute_access_is_rejected(code):
    deny, calls = _deny_all()
    r = _codeact(code, before_tool=deny)
    assert r["error"], code
    assert "not allowed" in r["error"]
    assert r["tool_calls"] == []


def test_tools_proxy_holds_no_gate_state_and_is_read_only():
    from clawagents.graph.codeact import _ToolsProxy

    deny, _ = _deny_all()
    proxy = _ToolsProxy(ToolRegistry(), before_tool=deny, run_async=lambda c: None)
    for name in ("_registry", "_before_tool", "_run_async", "_run_context"):
        assert not hasattr(proxy, name)
    with pytest.raises(AttributeError):
        proxy._before_tool = None
    with pytest.raises(AttributeError):
        proxy.execute = lambda **k: "pwned"
    assert "[blocked]" in proxy.execute(command="id")


def test_format_string_cannot_reach_gate_state():
    """``str.format`` attribute lookups bypass the AST gate; there must be nothing to find."""
    r = _codeact("print('{0._before_tool}'.format(tools))")
    assert r["error"]


def test_runaway_snippet_is_stopped():
    started = time.monotonic()
    r = _codeact("while True:\n    pass", timeout_s=0.3)
    assert time.monotonic() - started < 5
    assert r["error"] and "CodeActTimeout" in r["error"]
    assert r["done"] is False


def test_timeout_cannot_be_swallowed_by_the_snippet():
    code = "while True:\n    try:\n        pass\n    except Exception:\n        pass"
    r = _codeact(code, timeout_s=0.3)
    assert r["error"] and "CodeActTimeout" in r["error"]
    assert _codeact("try:\n    pass\nexcept:\n    pass")["error"]
    assert _codeact("try:\n    pass\nexcept BaseException:\n    pass")["error"]


def test_tool_time_is_not_charged_to_the_snippet():
    class _Result:
        success = True
        output = "ok"
        error = None

    def slow_run_async(coro):
        coro.close()
        time.sleep(0.4)
        return _Result()

    r = _codeact(
        "print(tools.execute(command='sleep'))\ndone = True",
        run_async=slow_run_async,
        timeout_s=0.2,
    )
    assert r["error"] is None, r["error"]
    assert r["done"] is True


def test_trace_hook_is_restored():
    import sys

    before = sys.gettrace()
    _codeact("x = 1")
    _codeact("while True:\n    pass", timeout_s=0.05)
    assert sys.gettrace() is before


# ── Seatbelt: profile must not live in the sandbox-writable workspace ────


def _seatbelt_backend(tmp_path: Path):
    from clawagents.sandbox.local import LocalBackend
    from clawagents.sandbox.profiles import OSSandboxProfile, ProfileBackend

    return ProfileBackend(
        LocalBackend(root=str(tmp_path)),
        OSSandboxProfile(name="workspace", backend="seatbelt", network=False, require_binary=False),
    )


def test_seatbelt_profile_is_inline_and_writes_nothing(tmp_path):
    backend = _seatbelt_backend(tmp_path)
    with patch("clawagents.sandbox.profiles.shutil.which", return_value="/usr/bin/sandbox-exec"):
        wrapped = backend.wrap_command("echo 'it''s' $(id)", cwd=str(tmp_path))
    argv = shlex.split(wrapped)
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[1] == "-p"
    assert "(version 1)" in argv[2]
    assert argv[3:5] == ["/bin/sh", "-c"]
    assert argv[5] == "echo 'it''s' $(id)", "user command must survive quoting intact"
    assert "-f" not in argv[:3]
    assert list(tmp_path.iterdir()) == [], "no profile file may be written into the workspace"


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="macOS sandbox-exec not available")
def test_seatbelt_inline_profile_runs_and_confines_writes(tmp_path):
    backend = _seatbelt_backend(tmp_path)
    outside = Path.home() / f".clawagents_seatbelt_probe_{time.time_ns()}"
    wrapped = backend.wrap_command(
        f"echo inside > ok.txt; echo outside > {shlex.quote(str(outside))}; echo ran",
        cwd=str(tmp_path),
    )
    try:
        proc = subprocess.run(
            wrapped, shell=True, cwd=str(tmp_path), capture_output=True, text=True, timeout=30
        )
        if "Operation not permitted" in proc.stderr and "ran" not in proc.stdout:
            pytest.skip("sandbox-exec cannot nest inside an existing sandbox")
        assert "ran" in proc.stdout, proc.stderr
        assert (tmp_path / "ok.txt").exists()
        assert not outside.exists(), "write outside the workspace must be denied"
    finally:
        outside.unlink(missing_ok=True)


# ── Bash validator: bypass shapes ────────────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        "/bin/rm -rf ~",
        "/bin/rm -rf /etc",
        "/usr/bin/find / -delete",
        "/bin/dd if=/dev/zero of=/dev/sda",
        "{ rm -rf ~; }",
        "! rm -rf ~",
        "if true; then rm -rf /; fi",
        "if false; then :; else rm -rf /; fi",
        "while true; do rm -rf /; done",
        "sh <<< 'rm -rf /'",
        'bash <<< "rm -rf ~"',
        "x=rm; $x -rf /",
        "> /etc/passwd",
        "echo 'x ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers",
        "rm -rf /.",
        "rm -rf ../..",
        "rm -rf ../../..",
    ],
)
def test_destructive_bypass_shapes_are_blocked(command):
    assert validate_bash(command).decision == Decision.BLOCK, command


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "/bin/ls -la",
        "./scripts/run_tests.sh",
        "/usr/bin/python3 script.py",
        "echo hi 2>/dev/null",
        "if [ -f x ]; then echo y; fi",
        "while read line; do echo $line; done",
        "cat <<< 'hello'",
        "git status",
        "cd ../.. && ls",
        "echo done > out.txt",
    ],
)
def test_benign_commands_are_not_blocked_or_warned(command):
    assert validate_bash(command).decision == Decision.ALLOW, command


def test_variable_program_name_warns_rather_than_allows():
    assert validate_bash("$PYTHON -m pytest").decision == Decision.WARN


def test_absolute_path_keeps_non_blocking_classification():
    assert validate_bash("/bin/rm -rf build").decision == Decision.WARN
    assert validate_bash("/bin/chmod 644 file.txt").decision == Decision.WARN
