"""Tests for Context Observatory Gateway toggle setting."""

from unittest.mock import AsyncMock, patch
import pytest

from clawagents.gateway.server import create_app


@pytest.mark.asyncio
async def test_gateway_observatory_disabled_by_default(tmp_path, monkeypatch):
    """Verify that when enable_context_observatory is false/omitted, no observer hooks are attached."""
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))

    with patch("clawagents.gateway.server.create_claw_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_result = AsyncMock()
        mock_result.status = "done"
        mock_result.result = "ok"
        mock_result.iterations = 1
        mock_agent.invoke.return_value = mock_result
        mock_create.return_value = mock_agent

        app, _, _ = create_app()
        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.post("/chat/stream", json={"task": "hello", "enable_context_observatory": False})
        assert response.status_code == 200

        # Check mock_agent.invoke call args — hooks should not be passed
        assert mock_agent.invoke.called
        kwargs = mock_agent.invoke.call_args.kwargs
        assert "hooks" not in kwargs or kwargs["hooks"] is None

        # Verify no observatory dir created
        obs_dir = tmp_path / ".clawagents" / "context-observatory"
        assert not obs_dir.exists()


@pytest.mark.asyncio
async def test_gateway_observatory_enabled(tmp_path, monkeypatch):
    """Verify that when enable_context_observatory is True, observer hooks are attached and session saved."""
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))

    with patch("clawagents.gateway.server.create_claw_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_result = AsyncMock()
        mock_result.status = "done"
        mock_result.result = "ok"
        mock_result.iterations = 1
        async def fake_invoke(task, on_event=None, hooks=None):
            if hooks and hasattr(hooks, "store"):
                await hooks.on_llm_start(None, "test-model", [])
                await hooks.on_llm_end(None, "test-model", "ok", None)
            return mock_result

        mock_agent.invoke.side_effect = fake_invoke
        mock_create.return_value = mock_agent

        app, _, _ = create_app()
        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.post("/chat/stream", json={"task": "hello", "chat_id": "vscode_chat_001", "enable_context_observatory": True})
        assert response.status_code == 200
        assert "event: observatory" in response.text
        assert '"kind": "llm_call"' in response.text

        # Check mock_agent.invoke call args — hooks should be passed
        assert mock_agent.invoke.called
        kwargs = mock_agent.invoke.call_args.kwargs
        assert "hooks" in kwargs and kwargs["hooks"] is not None

        # Verify session dir created
        session_dir = tmp_path / ".clawagents" / "context-observatory" / "vscode_chat_001"
        assert session_dir.exists()
        assert (session_dir / "session.json").exists()
