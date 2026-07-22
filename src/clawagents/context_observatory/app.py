"""Context Observatory — Streamlit dashboard with SSE streaming chat.

This is the main Streamlit application that connects to the ClawAgents sidecar
via the ``POST /chat/stream`` SSE endpoint, fully simulating the VSCode plugin's
interactive experience:

- Streaming assistant deltas (typewriter effect)
- Real-time tool execution status
- Permission request / ask-user interactive prompts
- Token usage tracking and analytics
- Context inspector, event timeline, and session comparison

The architecture mirrors the VSCode extension:

    [User Input] → [chat_panel] → [SseClient.stream_chat()] → [sidecar /chat/stream]
                                         ↓
                                    SSE event stream
                                         ↓
                        [apply_sse_event()] → [chat_items in session_state]
                        [SseEventBridge]    → [EventStore for analytics]

Launch::

    streamlit run src/clawagents/context_observatory/app.py
    # or
    python -m clawagents.context_observatory
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


class _SseStreamSession:
    """Background SSE reader that keeps the sidecar connection open for HITL.

    Streamlit reruns close any in-script HTTP streams. Breaking out of
    ``stream_chat`` on ``permission_required`` therefore cancelled the agent
    run on the sidecar. This worker keeps reading on a daemon thread so
    ``POST /permissions/...`` can unblock the still-running turn.
    """

    def __init__(
        self,
        client: Any,
        task: str,
        *,
        chat_id: str | None,
        mode: str,
        model: str | None,
        reasoning_effort: str | None,
        interaction: str,
    ) -> None:
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.done = False
        self.waiting_prompt = False
        self.error: str | None = None
        self._client = client
        self._task = task
        self._chat_id = chat_id
        self._mode = mode
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._interaction = interaction
        self._thread = threading.Thread(target=self._run, name="obs-sse-stream", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._collect())
        except Exception as exc:
            logger.exception("Background SSE stream failed")
            self.error = str(exc)
            self.events.put({"type": "error", "message": f"Connection failed: {exc}"})
        finally:
            self.done = True
            self.waiting_prompt = False
            self.events.put(None)
            loop.close()

    async def _collect(self) -> None:
        async for event in self._client.stream_chat(
            self._task,
            chat_id=self._chat_id,
            mode=self._mode,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            interaction=self._interaction,
            enable_context_observatory=True,
        ):
            et = event.get("type")
            if et in ("permission_required", "ask_user_required"):
                self.waiting_prompt = True
            elif et in ("done", "error", "cancelled"):
                self.waiting_prompt = False
            self.events.put(event)


def _configure_page() -> None:
    """Set Streamlit page config — must be first Streamlit call."""
    st.set_page_config(
        page_title="Context Observatory — ClawAgents",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _inject_custom_css() -> None:
    """Inject custom CSS for a polished dark-mode-friendly look."""
    st.markdown("""
    <style>
    /* Global refinements */
    .stApp { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Progress bars */
    .stProgress > div > div {
        border-radius: 8px;
    }

    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }

    /* Tool cards */
    .tool-running {
        border-left: 3px solid #fd7e14;
        padding-left: 8px;
    }
    .tool-success {
        border-left: 3px solid #198754;
        padding-left: 8px;
    }
    .tool-failed {
        border-left: 3px solid #dc3545;
        padding-left: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    """Main entry point for the Streamlit dashboard."""
    _configure_page()
    _inject_custom_css()

    # Lazy imports to avoid loading heavy dependencies at module level
    from clawagents.context_observatory.components.chat_panel import (
        apply_sse_event,
        clear_chat_items,
        render_chat_panel,
    )
    from clawagents.context_observatory.components.compare_view import render_compare_view
    from clawagents.context_observatory.components.context_inspector import (
        render_context_inspector,
    )
    from clawagents.context_observatory.components.event_timeline import (
        render_event_timeline,
    )
    from clawagents.context_observatory.components.history_browser import (
        render_history_browser,
    )
    from clawagents.context_observatory.components.sidebar import render_sidebar
    from clawagents.context_observatory.components.token_chart import render_token_charts
    from clawagents.context_observatory.sse_client import SseClient
    from clawagents.context_observatory.sse_hooks_bridge import SseEventBridge
    from clawagents.context_observatory.store import EventStore

    # ── Initialize session state ──
    if "event_store" not in st.session_state:
        st.session_state["event_store"] = EventStore()
    if "compare_store" not in st.session_state:
        st.session_state["compare_store"] = None
    if "chat_busy" not in st.session_state:
        st.session_state["chat_busy"] = False

    store: EventStore = st.session_state["event_store"]

    # ── Sidebar ──
    config = render_sidebar(store)

    # Handle health check
    if st.session_state.get("check_health"):
        st.session_state["check_health"] = False
        _handle_health_check(config)

    # Handle chat list refresh
    if st.session_state.get("refresh_chats"):
        st.session_state["refresh_chats"] = False
        _handle_refresh_chats(config)

    # Handle chat selection
    if st.session_state.get("select_chat_id"):
        chat_id = st.session_state.pop("select_chat_id")
        st.session_state["active_chat_id"] = chat_id
        clear_chat_items()
        # TODO: restore chat history from sidecar
        st.rerun()

    # Handle clear history
    if config.get("clear_history"):
        store.clear()
        clear_chat_items()
        st.session_state["chat_usage"] = {}
        st.rerun()

    # Handle replay file upload
    if st.session_state.get("replay_file") is not None:
        try:
            uploaded = st.session_state["replay_file"]
            raw = json.loads(uploaded.read())
            replay_store = EventStore()
            replay_store._session_meta = raw.get("session_meta", {})
            for entry in raw.get("events", []):
                from clawagents.context_observatory.store import _deserialize_event
                event = _deserialize_event(entry)
                if event is not None:
                    replay_store._events.append(event)
            st.session_state["compare_store"] = replay_store
            st.session_state["replay_file"] = None
            st.toast("✅ Session loaded for comparison!", icon="📥")
        except Exception as e:
            st.error(f"Failed to load session: {e}")
            st.session_state["replay_file"] = None

    # Handle exports
    if st.session_state.get("export_json"):
        _handle_export_json(store)
        st.session_state["export_json"] = False

    if st.session_state.get("export_csv"):
        _handle_export_csv(store)
        st.session_state["export_csv"] = False

    # ── Title ──
    st.title("🔭 Context Observatory")
    st.caption(
        "Real-time context management observability for ClawAgents — "
        "streaming chat via SSE, inspect messages, track token usage, analyze events"
    )

    # ── Main layout ──
    tab_chat, tab_inspector, tab_charts, tab_events, tab_history, tab_compare = st.tabs([
        "💬 Chat",
        "🔍 Context Inspector",
        "📈 Token Analytics",
        "📋 Event Timeline",
        "📚 History",
        "🔄 Compare",
    ])

    with tab_chat:
        user_input = render_chat_panel()
        if user_input and not st.session_state.get("chat_busy"):
            _handle_user_input(user_input, config, store)

        # Handle pending permission actions (POST while SSE stays open)
        _handle_pending_permissions(config)

        # Handle pending ask-user actions
        _handle_pending_ask_user(config)

        # Drain background SSE (keeps HITL turns alive across Streamlit reruns)
        _drain_sse_stream(store, config)

    with tab_inspector:
        render_context_inspector(store)

    with tab_charts:
        render_token_charts(store)

    with tab_events:
        render_event_timeline(store)

    with tab_history:
        render_history_browser(store)

    with tab_compare:
        render_compare_view(
            store,
            st.session_state.get("compare_store"),
        )


# ── SSE streaming handler ───────────────────────────────────────────────


def _handle_user_input(
    user_input: str,
    config: dict[str, Any],
    store: EventStore,
) -> None:
    """Start a background SSE turn against the sidecar."""
    from clawagents.context_observatory.components.chat_panel import add_chat_item
    from clawagents.context_observatory.sse_client import SseClient
    from clawagents.context_observatory.sse_hooks_bridge import SseEventBridge

    if st.session_state.get("sse_stream") and not st.session_state["sse_stream"].done:
        return

    add_chat_item({"kind": "user", "text": user_input})
    st.session_state["chat_busy"] = True
    st.session_state["chat_streaming"] = True

    client = SseClient(
        host=config.get("sidecar_host", "127.0.0.1"),
        port=config.get("sidecar_port", 3001),
        token=config.get("sidecar_token", ""),
    )

    context_window = config.get("context_window", 128_000)
    model = config.get("model") or ""
    bridge = SseEventBridge(
        store,
        context_window=context_window,
        model=model,
        user_text=user_input,
    )
    st.session_state["sse_bridge"] = bridge
    st.session_state["sse_user_input"] = user_input
    st.session_state["sse_collected"] = []

    store.set_session_meta(
        model=model,
        context_window=context_window,
        started_at=time.time(),
    )

    if not st.session_state.get("active_chat_id"):
        import uuid
        st.session_state["active_chat_id"] = f"chat_{uuid.uuid4().hex[:8]}"

    session = _SseStreamSession(
        client,
        user_input,
        chat_id=st.session_state["active_chat_id"],
        mode=config.get("mode", "auto"),
        model=config.get("model"),
        reasoning_effort=config.get("reasoning_effort"),
        interaction=config.get("interaction", "interactive"),
    )
    st.session_state["sse_stream"] = session
    session.start()
    st.rerun()


def _drain_sse_stream(store: EventStore, config: dict[str, Any]) -> None:
    """Apply queued SSE events; keep rerunning until the background turn ends."""
    from clawagents.context_observatory.components.chat_panel import apply_sse_event

    session: _SseStreamSession | None = st.session_state.get("sse_stream")
    if session is None:
        return

    bridge = st.session_state.get("sse_bridge")
    collected: list[dict[str, Any]] = st.session_state.setdefault("sse_collected", [])
    saw_prompt = False
    finished = False

    while True:
        try:
            event = session.events.get_nowait()
        except queue.Empty:
            break
        if event is None:
            finished = True
            continue
        collected.append(event)
        apply_sse_event(event)
        if bridge is not None:
            try:
                bridge.ingest(event)
            except Exception:
                logger.debug("SSE bridge ingest failed", exc_info=True)
        if event.get("type") == "done" and event.get("chatId"):
            st.session_state["active_chat_id"] = event["chatId"]
        if event.get("type") in ("permission_required", "ask_user_required"):
            saw_prompt = True
        if event.get("type") in ("done", "error", "cancelled"):
            finished = True

    if session.error:
        st.session_state["sidecar_status"] = "error"
        st.session_state["sidecar_error"] = session.error
    elif collected:
        st.session_state["sidecar_status"] = "connected"

    # HITL prompt: stay in the turn (connection open) but do not spin forever
    # without a UI refresh — rerun so Allow/Deny buttons appear.
    if finished or session.done:
        user_input = st.session_state.get("sse_user_input") or ""
        if collected:
            _dump_events_to_file(collected, user_input)
        st.session_state["chat_busy"] = False
        st.session_state["chat_streaming"] = False
        st.session_state["sse_stream"] = None
        st.session_state.pop("sse_bridge", None)
        st.session_state.pop("sse_user_input", None)
        st.session_state.pop("sse_collected", None)
        st.rerun()
        return

    if saw_prompt:
        # New HITL prompt just arrived — refresh so Allow/Deny widgets render.
        # Keep chat_busy True so a second user message is not sent mid-turn.
        st.rerun()
        return

    if session.waiting_prompt:
        # Idle until the user clicks Allow/Deny (that click triggers a rerun).
        return

    # Still streaming agent output — short poll then refresh
    time.sleep(0.05)
    st.rerun()


def _dump_events_to_file(events: list[dict[str, Any]], user_input: str) -> None:
    """Save raw SSE events to a JSONL file in the session directory."""
    import datetime
    from clawagents.context_observatory.store import get_history_dir

    chat_id = st.session_state.get("active_chat_id") or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = get_history_dir() / chat_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dump_file = session_dir / "events.jsonl"

    try:
        with open(dump_file, "a", encoding="utf-8") as f:
            # Write turn header
            f.write(json.dumps({
                "__meta__": True,
                "timestamp": datetime.datetime.now().isoformat(),
                "user_input": user_input,
                "event_count": len(events),
            }) + "\n")
            # Write each event
            for event in events:
                f.write(json.dumps(event, default=str) + "\n")
        logger.info("Dumped %d events to %s", len(events), dump_file)
    except Exception:
        logger.debug("Failed to dump events", exc_info=True)


# ── Permission / ask-user resolution ────────────────────────────────────


def _handle_pending_permissions(config: dict[str, Any]) -> None:
    """Check for any permission actions the user has clicked and resolve them."""
    from clawagents.context_observatory.sse_client import SseClient

    items = st.session_state.get("chat_items", [])
    for item in items:
        if item.get("kind") != "permission" or not item.get("resolved"):
            continue

        request_id = item.get("requestId", "")
        action_key = f"perm_action_{request_id}"
        decision = st.session_state.get(action_key)

        if decision:
            client = SseClient(
                host=config.get("sidecar_host", "127.0.0.1"),
                port=config.get("sidecar_port", 3001),
                token=config.get("sidecar_token", ""),
            )
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(
                    client.resolve_permission(request_id, decision)
                )
                loop.close()
                # Resume draining the still-open SSE stream for post-approval events.
                st.session_state.pop(action_key, None)
                st.rerun()
                return
            except Exception as e:
                logger.warning("Failed to resolve permission: %s", e)

            # Clean up
            st.session_state.pop(action_key, None)


def _handle_pending_ask_user(config: dict[str, Any]) -> None:
    """Check for any ask-user actions the user has submitted and resolve them."""
    from clawagents.context_observatory.sse_client import SseClient

    items = st.session_state.get("chat_items", [])
    for item in items:
        if item.get("kind") != "ask" or not item.get("resolved"):
            continue

        request_id = item.get("requestId", "")
        action_key = f"ask_action_{request_id}"
        action = st.session_state.get(action_key)

        if action:
            client = SseClient(
                host=config.get("sidecar_host", "127.0.0.1"),
                port=config.get("sidecar_port", 3001),
                token=config.get("sidecar_token", ""),
            )
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(
                    client.resolve_ask_user(
                        request_id,
                        answer=action.get("answer"),
                        skip=action.get("skip", False),
                    )
                )
                loop.close()
                st.session_state.pop(action_key, None)
                st.rerun()
                return
            except Exception as e:
                logger.warning("Failed to resolve ask_user: %s", e)

            # Clean up
            st.session_state.pop(action_key, None)


# ── Health check ────────────────────────────────────────────────────────


def _handle_health_check(config: dict[str, Any]) -> None:
    """Check sidecar health and update status."""
    from clawagents.context_observatory.sse_client import SseClient

    client = SseClient(
        host=config.get("sidecar_host", "127.0.0.1"),
        port=config.get("sidecar_port", 3001),
        token=config.get("sidecar_token", ""),
    )

    try:
        loop = asyncio.new_event_loop()
        health = loop.run_until_complete(client.fetch_health())
        loop.close()

        if health:
            st.session_state["sidecar_status"] = "connected"
            st.session_state["sidecar_health"] = health
            st.toast(f"✅ Connected — {health.get('model', 'unknown')}", icon="🟢")
        else:
            st.session_state["sidecar_status"] = "error"
            st.session_state["sidecar_error"] = "No response"
            st.toast("❌ Cannot reach sidecar", icon="🔴")
    except Exception as e:
        st.session_state["sidecar_status"] = "error"
        st.session_state["sidecar_error"] = str(e)
        st.toast(f"❌ Connection failed: {e}", icon="🔴")

    st.rerun()


# ── Chat list refresh ───────────────────────────────────────────────────


def _handle_refresh_chats(config: dict[str, Any]) -> None:
    """Refresh the chat list from the sidecar."""
    from clawagents.context_observatory.sse_client import SseClient

    client = SseClient(
        host=config.get("sidecar_host", "127.0.0.1"),
        port=config.get("sidecar_port", 3001),
        token=config.get("sidecar_token", ""),
    )

    try:
        loop = asyncio.new_event_loop()
        chats = loop.run_until_complete(client.list_chats())
        loop.close()
        st.session_state["chat_list"] = chats
    except Exception as e:
        logger.warning("Failed to refresh chats: %s", e)

    st.rerun()


# ── Exports ─────────────────────────────────────────────────────────────


def _handle_export_json(store: "EventStore") -> None:
    """Export current session to JSON."""
    data = json.dumps(store.to_dict(), indent=2, default=str, ensure_ascii=False)
    st.download_button(
        label="📥 Download JSON",
        data=data,
        file_name=f"context_observatory_{int(time.time())}.json",
        mime="application/json",
    )


def _handle_export_csv(store: "EventStore") -> None:
    """Export token curve to CSV."""
    import csv
    import io

    curve = store.get_token_curve()
    if not curve:
        st.warning("No token data to export.")
        return

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(curve[0].keys()))
    writer.writeheader()
    writer.writerows(curve)

    st.download_button(
        label="📊 Download CSV",
        data=output.getvalue(),
        file_name=f"token_curve_{int(time.time())}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
