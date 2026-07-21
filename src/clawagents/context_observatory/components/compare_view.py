"""Compare view — side-by-side session comparison for strategy analysis."""

from __future__ import annotations

from typing import Any

import streamlit as st

from clawagents.context_observatory.store import EventStore


def render_compare_view(store_a: EventStore, store_b: EventStore | None = None) -> None:
    """Render a comparison view between two recorded sessions.

    If only one store is provided, shows single-session summary with
    a prompt to load a second session for comparison.
    """
    st.subheader("🔄 Session Comparison")

    if store_b is None:
        st.info(
            "Load a second session JSON file in the sidebar to compare strategies. "
            "Export your current session first, then load a different recording."
        )
        _render_single_session_summary(store_a, "Current Session")
        return

    col1, col2 = st.columns(2)

    with col1:
        meta_a = store_a.session_meta
        title_a = meta_a.get("label", meta_a.get("model", "Session A"))
        _render_single_session_summary(store_a, str(title_a))

    with col2:
        meta_b = store_b.session_meta
        title_b = meta_b.get("label", meta_b.get("model", "Session B"))
        _render_single_session_summary(store_b, str(title_b))

    st.markdown("---")

    # ── Token curve overlay ──
    st.markdown("#### Token Utilization Comparison")
    _render_comparison_curve(store_a, store_b)

    # ── Delta table ──
    st.markdown("#### Metrics Comparison")
    _render_delta_table(store_a, store_b)


def _render_single_session_summary(store: EventStore, title: str) -> None:
    """Render summary metrics for a single session."""
    st.markdown(f"### {title}")

    meta = store.session_meta
    if meta:
        cols = st.columns(2)
        with cols[0]:
            st.caption(f"Model: **{meta.get('model', 'N/A')}**")
        with cols[1]:
            st.caption(f"Context Window: **{meta.get('context_window', 'N/A'):,}**")

    cumulative = store.get_cumulative_tokens()
    compact_summary = store.get_compaction_summary()
    crush_summary = store.get_crush_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tokens", f"{cumulative['total_tokens']:,}")
        st.metric("LLM Calls", cumulative["calls"])
    with col2:
        st.metric("Compactions", compact_summary["compaction_count"])
        st.metric(
            "Avg Savings",
            f"{compact_summary['avg_savings_pct']:.1f}%",
        )
    with col3:
        st.metric("Crush Events", crush_summary["crush_count"])
        st.metric("Chars Saved", f"{crush_summary['total_chars_saved']:,}")


def _render_comparison_curve(store_a: EventStore, store_b: EventStore) -> None:
    """Overlay token utilization curves from two sessions."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for comparison charts.")
        return

    curve_a = store_a.get_token_curve()
    curve_b = store_b.get_token_curve()

    if not curve_a and not curve_b:
        st.info("No token data in either session.")
        return

    fig = go.Figure()

    if curve_a:
        fig.add_trace(go.Scatter(
            x=[c["turn"] for c in curve_a],
            y=[c["input_tokens"] for c in curve_a],
            name="Session A — Input",
            line=dict(color="rgb(55, 126, 184)", width=2),
        ))

    if curve_b:
        fig.add_trace(go.Scatter(
            x=[c["turn"] for c in curve_b],
            y=[c["input_tokens"] for c in curve_b],
            name="Session B — Input",
            line=dict(color="rgb(228, 26, 28)", width=2),
        ))

    # Context window lines
    for curve, name, color in [
        (curve_a, "A", "rgba(55, 126, 184, 0.3)"),
        (curve_b, "B", "rgba(228, 26, 28, 0.3)"),
    ]:
        if curve and curve[0]["context_window"] > 0:
            fig.add_hline(
                y=curve[0]["context_window"],
                line=dict(color=color, dash="dash"),
                annotation_text=f"CW-{name}: {curve[0]['context_window']:,}",
            )

    fig.update_layout(
        title="Input Token Usage Comparison",
        xaxis_title="Turn",
        yaxis_title="Input Tokens",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_delta_table(store_a: EventStore, store_b: EventStore) -> None:
    """Table comparing key metrics between two sessions."""
    cum_a = store_a.get_cumulative_tokens()
    cum_b = store_b.get_cumulative_tokens()
    comp_a = store_a.get_compaction_summary()
    comp_b = store_b.get_compaction_summary()
    crush_a = store_a.get_crush_summary()
    crush_b = store_b.get_crush_summary()

    rows: list[dict[str, Any]] = [
        {
            "Metric": "Total Input Tokens",
            "Session A": f"{cum_a['total_input_tokens']:,}",
            "Session B": f"{cum_b['total_input_tokens']:,}",
            "Delta": _delta(cum_a["total_input_tokens"], cum_b["total_input_tokens"]),
        },
        {
            "Metric": "Total Output Tokens",
            "Session A": f"{cum_a['total_output_tokens']:,}",
            "Session B": f"{cum_b['total_output_tokens']:,}",
            "Delta": _delta(cum_a["total_output_tokens"], cum_b["total_output_tokens"]),
        },
        {
            "Metric": "LLM Calls",
            "Session A": str(cum_a["calls"]),
            "Session B": str(cum_b["calls"]),
            "Delta": _delta(cum_a["calls"], cum_b["calls"]),
        },
        {
            "Metric": "Compactions",
            "Session A": str(comp_a["compaction_count"]),
            "Session B": str(comp_b["compaction_count"]),
            "Delta": _delta(comp_a["compaction_count"], comp_b["compaction_count"]),
        },
        {
            "Metric": "Tokens Saved (Compaction)",
            "Session A": f"{comp_a['total_tokens_saved']:,}",
            "Session B": f"{comp_b['total_tokens_saved']:,}",
            "Delta": _delta(
                comp_a["total_tokens_saved"], comp_b["total_tokens_saved"]
            ),
        },
        {
            "Metric": "Crush Events",
            "Session A": str(crush_a["crush_count"]),
            "Session B": str(crush_b["crush_count"]),
            "Delta": _delta(crush_a["crush_count"], crush_b["crush_count"]),
        },
    ]

    st.table(rows)


def _delta(a: int | float, b: int | float) -> str:
    """Format a delta value with sign and percentage."""
    diff = b - a
    if a == 0:
        return f"+{diff:,}" if diff > 0 else str(diff)
    pct = diff / a * 100
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:,} ({sign}{pct:.1f}%)"
