"""Event timeline — compaction, crush, and trim event log with details."""

from __future__ import annotations

import streamlit as st

from clawagents.context_observatory.events import (
    CompactionEvent,
    CrushEvent,
    TrimEvent,
)
from clawagents.context_observatory.store import EventStore


def render_event_timeline(store: EventStore) -> None:
    """Render a timeline of context management events.

    Shows compaction, crush, and trim events in chronological order
    with color-coded cards and expandable details.
    """
    st.subheader("📋 Event Timeline")

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        show_compaction = st.checkbox("Compaction", value=True, key="tl_compact")
    with col2:
        show_crush = st.checkbox("Crush", value=True, key="tl_crush")
    with col3:
        show_trim = st.checkbox("Trim", value=True, key="tl_trim")

    # Collect filtered events
    events = []
    if show_compaction:
        events.extend(store.get_compaction_events())
    if show_crush:
        events.extend(store.get_crush_events())
    if show_trim:
        events.extend(store.get_trim_events())

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)

    if not events:
        st.info("No context management events recorded yet.")
        return

    st.caption(f"{len(events)} events")

    for event in events:
        if isinstance(event, CompactionEvent):
            _render_compaction_event(event)
        elif isinstance(event, CrushEvent):
            _render_crush_event(event)
        elif isinstance(event, TrimEvent):
            _render_trim_event(event)


def _render_compaction_event(event: CompactionEvent) -> None:
    """Render a compaction event card."""
    if event.phase == "start":
        icon = "🔴"
        label = "Compaction Started"
        details = (
            f"**Tokens**: {event.tokens_before:,} | "
            f"**Messages**: {event.messages_before}"
        )
    else:
        icon = "🟢"
        label = "Compaction Complete"
        details = (
            f"**Before**: {event.tokens_before:,} tokens → "
            f"**After**: {event.tokens_after:,} tokens\n\n"
            f"**Messages**: {event.messages_before} → {event.messages_after} "
            f"(dropped {event.messages_dropped})\n\n"
            f"**Savings**: {event.savings_pct:.1f}%"
        )

    with st.expander(
        f"{icon} Turn {event.turn} — {label}",
        expanded=(event.phase == "end"),
    ):
        st.markdown(details)
        if event.summary_preview:
            st.markdown("**Summary Preview:**")
            st.text(event.summary_preview)

        if event.phase == "end" and event.tokens_before > 0:
            saved = event.tokens_before - event.tokens_after
            st.progress(
                min(event.savings_pct / 100, 1.0),
                text=f"Saved {saved:,} tokens ({event.savings_pct:.1f}%)",
            )


def _render_crush_event(event: CrushEvent) -> None:
    """Render a content crush event card."""
    ratio = (
        f"{event.crushed_chars / event.original_chars * 100:.0f}%"
        if event.original_chars > 0
        else "N/A"
    )

    with st.expander(
        f"🗜️ Turn {event.turn} — Crush: {event.tool_name} ({event.content_kind})",
        expanded=False,
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original", f"{event.original_chars:,} chars")
        with col2:
            st.metric("Crushed", f"{event.crushed_chars:,} chars")
        with col3:
            st.metric("Saved", f"{event.saved_chars:,} chars")

        st.markdown(
            f"**Content type**: `{event.content_kind}` | "
            f"**Compression**: {ratio} | "
            f"**Token saving**: {event.original_tokens - event.crushed_tokens:,} tokens"
        )

        if event.original_chars > 0:
            compression_ratio = event.crushed_chars / event.original_chars
            st.progress(
                1 - compression_ratio,
                text=f"Compression: {(1 - compression_ratio) * 100:.1f}% reduction",
            )


def _render_trim_event(event: TrimEvent) -> None:
    """Render an output trim event card."""
    with st.expander(
        f"✂️ Turn {event.turn} — Trim: {event.role}",
        expanded=False,
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original", f"{event.original_chars:,} chars")
        with col2:
            st.metric("Trimmed", f"{event.trimmed_chars:,} chars")
        with col3:
            st.metric("Saved", f"{event.saved_chars:,} chars")
