"""Sidebar component — sidecar connection, configuration, and session metrics.

Now includes sidecar connection configuration (host/port/token), agent mode
selection, chat session management, and the original session summary metrics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import streamlit as st

from clawagents.context_observatory.store import EventStore

logger = logging.getLogger(__name__)


def render_sidebar(store: EventStore) -> dict[str, Any]:
    """Render the sidebar with config controls and session summary.

    Returns a dict of current configuration values.
    """
    st.sidebar.title("⚙️ Context Observatory")
    st.sidebar.markdown("---")

    # ── Agent Configuration (model + context window) ──
    st.sidebar.subheader("Agent Configuration")

    model = st.sidebar.selectbox(
        "Model",
        [
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-5.6-sol",
            "gpt-5",
            "gpt-4o",
            "gpt-4o-mini",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        index=0,
        key="obs_model",
    )

    reasoning_effort = st.sidebar.selectbox(
        "Reasoning Effort",
        ["low", "medium", "high"],
        index=1,
        key="obs_reasoning_effort",
        help="Reasoning effort for reasoning models (e.g., o1, o3-mini)",
    )

    context_window = st.sidebar.number_input(
        "Context Window (tokens)",
        min_value=4_000,
        max_value=2_000_000,
        value=1_050_000,
        step=1_000,
        key="obs_context_window",
        help="Used for token analytics (actual context window is set on the sidecar)",
    )

    st.sidebar.markdown("---")

    # ── Gateway Connection ──
    st.sidebar.subheader("🔌 Gateway Connection")

    sidecar_host = st.sidebar.text_input(
        "Host",
        value=st.session_state.get("sidecar_host", "127.0.0.1"),
        key="sidecar_host_input",
        help="ClawAgents gateway hostname",
    )
    st.session_state["sidecar_host"] = sidecar_host

    sidecar_port = st.sidebar.number_input(
        "Port",
        min_value=1,
        max_value=65535,
        value=st.session_state.get("sidecar_port", 3000),
        step=1,
        key="sidecar_port_input",
    )
    st.session_state["sidecar_port"] = int(sidecar_port)

    sidecar_token = st.sidebar.text_input(
        "API Token",
        type="password",
        value=st.session_state.get("sidecar_token", ""),
        key="sidecar_token_input",
        help="GATEWAY_API_KEY (leave empty if auth is disabled)",
    )
    st.session_state["sidecar_token"] = sidecar_token

    # Connection status
    _render_connection_status()

    st.sidebar.markdown("---")



    # ── Chat Sessions ──
    st.sidebar.subheader("💬 Chat Sessions")

    # Active chat
    active_chat = st.session_state.get("active_chat_id")
    if active_chat:
        st.sidebar.caption(f"Active: `{active_chat[:16]}…`")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🆕 New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state["active_chat_id"] = None
            st.session_state["chat_items"] = []
            st.session_state["chat_usage"] = {}
            st.session_state["chat_busy"] = False
            st.session_state["chat_streaming"] = False
    with col2:
        if st.button("🔄 Refresh", key="refresh_chats_btn", use_container_width=True):
            st.session_state["refresh_chats"] = True

    # Chat list
    chats = st.session_state.get("chat_list", [])
    if chats:
        for chat in chats[:20]:
            chat_id = chat.get("id", "")
            title = chat.get("title", chat_id[:20])
            mode_badge = chat.get("mode", "")
            msg_count = chat.get("message_count", 0)
            label = f"{title} ({msg_count} msgs)" if msg_count else title
            is_active = chat_id == active_chat

            if is_active:
                st.sidebar.markdown(f"▶ **{label}**")
            else:
                if st.sidebar.button(
                    label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                ):
                    st.session_state["select_chat_id"] = chat_id

    st.sidebar.markdown("---")

    # ── Session Summary ──
    st.sidebar.subheader("📊 Session Summary")

    cumulative = store.get_cumulative_tokens()
    st.sidebar.metric("LLM Calls", cumulative["calls"])
    st.sidebar.metric("Total Input Tokens", f"{cumulative['total_input_tokens']:,}")
    st.sidebar.metric("Total Output Tokens", f"{cumulative['total_output_tokens']:,}")
    st.sidebar.metric("Cached Tokens", f"{cumulative['total_cached_tokens']:,}")

    # Usage from latest SSE stream
    usage = st.session_state.get("chat_usage", {})
    if usage:
        session_cost = usage.get("sessionCostUsd")
        if session_cost is not None:
            st.sidebar.metric("Session Cost", f"${session_cost:.4f}")

    compact_summary = store.get_compaction_summary()
    if compact_summary["compaction_count"] > 0:
        st.sidebar.markdown("**Compaction**")
        st.sidebar.metric("Compactions", compact_summary["compaction_count"])
        st.sidebar.metric("Tokens Saved", f"{compact_summary['total_tokens_saved']:,}")
        st.sidebar.metric(
            "Avg Savings",
            f"{compact_summary['avg_savings_pct']:.1f}%",
        )

    crush_summary = store.get_crush_summary()
    if crush_summary["crush_count"] > 0:
        st.sidebar.markdown("**Content Crush**")
        st.sidebar.metric("Crush Events", crush_summary["crush_count"])
        st.sidebar.metric("Chars Saved", f"{crush_summary['total_chars_saved']:,}")

    st.sidebar.markdown("---")

    # ── Session Actions ──
    st.sidebar.subheader("Session Actions")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📥 Export JSON", key="export_json", use_container_width=True):
            st.session_state["export_json"] = True
    with col2:
        if st.button("📊 Export CSV", key="export_csv", use_container_width=True):
            st.session_state["export_csv"] = True

    clear_history = st.sidebar.button("🧹 Clear History", key="clear_history", use_container_width=True)

    uploaded = st.sidebar.file_uploader(
        "Load session for replay",
        type=["json"],
        key="replay_upload",
    )
    if uploaded is not None:
        st.session_state["replay_file"] = uploaded

    return {
        "sidecar_host": sidecar_host,
        "sidecar_port": int(sidecar_port),
        "sidecar_token": sidecar_token,
        "mode": "auto",
        "interaction": "auto",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "context_window": int(context_window),
        "clear_history": clear_history,
    }


def _render_connection_status() -> None:
    """Show sidecar connection status."""
    status = st.session_state.get("sidecar_status")

    if status == "connected":
        health = st.session_state.get("sidecar_health", {})
        model = health.get("model", "unknown")
        provider = health.get("provider", "")
        st.sidebar.success(f"🟢 Connected — {model}")
    elif status == "error":
        detail = st.session_state.get("sidecar_error", "")
        st.sidebar.error(f"🔴 Disconnected{': ' + detail if detail else ''}")
    else:
        st.sidebar.info("⚪ Not connected — send a message to connect")

    if st.sidebar.button("🏥 Check Health", key="check_health_btn", use_container_width=True):
        st.session_state["check_health"] = True
