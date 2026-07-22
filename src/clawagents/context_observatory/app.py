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
import time
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


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

        # Handle pending permission actions
        _handle_pending_permissions(config)

        # Handle pending ask-user actions
        _handle_pending_ask_user(config)

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
    """Send user input to the sidecar via SSE and process the event stream."""
    from clawagents.context_observatory.components.chat_panel import (
        add_chat_item,
        apply_sse_event,
    )
    from clawagents.context_observatory.sse_client import SseClient
    from clawagents.context_observatory.sse_hooks_bridge import SseEventBridge

    # Add user message
    add_chat_item({"kind": "user", "text": user_input})
    st.session_state["chat_busy"] = True
    st.session_state["chat_streaming"] = False

    # Render user message immediately
    from clawagents.context_observatory.components.chat_panel import _render_user
    _render_user(len(st.session_state["chat_items"]) - 1, {"kind": "user", "text": user_input})

    # Create SSE client
    client = SseClient(
        host=config.get("sidecar_host", "127.0.0.1"),
        port=config.get("sidecar_port", 3001),
        token=config.get("sidecar_token", ""),
    )

    # Create event bridge for analytics
    context_window = config.get("context_window", 128_000)
    model = config.get("model") or ""
    bridge = SseEventBridge(
        store,
        context_window=context_window,
        model=model,
        user_text=user_input,
    )

    store.set_session_meta(
        model=model,
        context_window=context_window,
        started_at=time.time(),
    )

    # Generate a chat_id upfront if this is a new session
    if not st.session_state.get("active_chat_id"):
        import uuid
        st.session_state["active_chat_id"] = f"chat_{uuid.uuid4().hex[:8]}"

    # Run the SSE stream
    try:
        loop = asyncio.new_event_loop()
        thinking_container = st.chat_message("assistant", avatar="🤖")
        with thinking_container:
            with st.spinner("Agent is thinking... ⏳"):
                events = loop.run_until_complete(
                    _collect_sse_events(
                        client,
                        user_input,
                        chat_id=st.session_state["active_chat_id"],
                        mode=config.get("mode", "auto"),
                        model=config.get("model"),
                        reasoning_effort=config.get("reasoning_effort"),
                        interaction=config.get("interaction", "interactive"),
                    )
                )
        loop.close()

        # Update connection status
        st.session_state["sidecar_status"] = "connected"

        # Dump raw events to file for debugging
        _dump_events_to_file(events, user_input)

        # Apply all events to chat items and bridge
        for event in events:
            apply_sse_event(event)
            bridge.ingest(event)

            # Track chat_id from done events
            if event.get("type") == "done" and event.get("chatId"):
                st.session_state["active_chat_id"] = event["chatId"]

    except Exception as e:
        logger.exception("SSE stream failed")
        st.session_state["sidecar_status"] = "error"
        st.session_state["sidecar_error"] = str(e)
        apply_sse_event({
            "type": "error",
            "message": f"Connection failed: {e}",
        })
        st.session_state["chat_busy"] = False

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



async def _collect_sse_events(
    client: "SseClient",
    task: str,
    *,
    chat_id: str | None,
    mode: str,
    model: str | None,
    reasoning_effort: str | None,
    interaction: str,
) -> list[dict[str, Any]]:
    """Collect all SSE events from the stream into a list.

    Streamlit's rerun model prevents true incremental rendering during
    a single script execution. We collect all events, then apply them
    in bulk before rerunning the script.

    For permission_required / ask_user_required events, we stop collecting
    and return immediately so the UI can render the interactive prompt.
    """
    events: list[dict[str, Any]] = []

    async for event in client.stream_chat(
        task,
        chat_id=chat_id,
        mode=mode,
        model=model,
        reasoning_effort=reasoning_effort,
        interaction=interaction,
        enable_context_observatory=True,
    ):
        events.append(event)

        # If we hit a blocking event, stop and let the UI render it
        if event.get("type") in ("permission_required", "ask_user_required"):
            # Don't mark as done — we'll resume after the user responds
            break

    return events


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
