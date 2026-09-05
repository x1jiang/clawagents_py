"""Token charts — Plotly-based utilization curve and budget breakdown."""

from __future__ import annotations


import streamlit as st

from clawagents.context_observatory.store import EventStore


def render_token_charts(store: EventStore) -> None:
    """Render token utilization charts.

    1. Context Window Utilization Curve — line chart per turn
    2. Token Budget vs Actual — grouped bar chart
    """
    st.subheader("📈 Token Analytics")

    llm_calls = store.get_llm_calls()
    if not llm_calls:
        st.info("No data yet. Start a conversation to see token analytics.")
        return

    tab1, tab2, tab3 = st.tabs([
        "Utilization Curve",
        "Budget vs Actual",
        "System Prompt Composition",
    ])

    with tab1:
        _render_utilization_curve(store)
    with tab2:
        _render_budget_chart(store)
    with tab3:
        _render_system_prompt_trend(store)


def _render_utilization_curve(store: EventStore) -> None:
    """Line chart: context window utilization per turn."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for interactive charts: `pip install plotly`")
        _render_utilization_curve_fallback(store)
        return

    curve = store.get_token_curve()
    if not curve:
        return

    fig = go.Figure()

    turns = [c["turn"] for c in curve]
    input_tokens = [c["input_tokens"] for c in curve]
    output_tokens = [c["output_tokens"] for c in curve]
    context_window = curve[0]["context_window"] if curve else 0
    utilization = [c["utilization_pct"] for c in curve]
    cached = [c.get("cached_tokens", 0) for c in curve]

    # Input tokens area
    fig.add_trace(go.Scatter(
        x=turns, y=input_tokens,
        fill="tozeroy",
        name="Input Tokens",
        fillcolor="rgba(55, 126, 184, 0.3)",
        line=dict(color="rgb(55, 126, 184)", width=2),
        hovertemplate="Turn %{x}<br>Input: %{y:,} tokens<extra></extra>",
    ))

    # Output tokens stacked
    fig.add_trace(go.Scatter(
        x=turns, y=output_tokens,
        fill="tozeroy",
        name="Output Tokens",
        fillcolor="rgba(77, 175, 74, 0.3)",
        line=dict(color="rgb(77, 175, 74)", width=2),
        hovertemplate="Turn %{x}<br>Output: %{y:,} tokens<extra></extra>",
    ))

    # Cached tokens
    if any(c > 0 for c in cached):
        fig.add_trace(go.Scatter(
            x=turns, y=cached,
            name="Cached Tokens",
            line=dict(color="rgb(255, 127, 14)", width=2, dash="dot"),
            hovertemplate="Turn %{x}<br>Cached: %{y:,} tokens<extra></extra>",
        ))

    # Context window limit line
    if context_window > 0:
        fig.add_hline(
            y=context_window,
            line=dict(color="red", width=1, dash="dash"),
            annotation_text=f"Context Window: {context_window:,}",
            annotation_position="top right",
        )

    # Compaction events as vertical markers
    for evt in store.get_compaction_events():
        if evt.phase == "start":
            fig.add_vline(
                x=evt.turn,
                line=dict(color="red", width=1, dash="dot"),
                annotation_text="⚠️ Compact",
                annotation_position="top",
            )

    fig.update_layout(
        title="Context Window Utilization",
        xaxis_title="Turn",
        yaxis_title="Tokens",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Utilization percentage bar
    if utilization:
        latest = utilization[-1]
        st.progress(min(latest / 100, 1.0), text=f"Current utilization: {latest:.1f}%")


def _render_utilization_curve_fallback(store: EventStore) -> None:
    """Fallback chart using Streamlit's built-in charting (no plotly)."""
    curve = store.get_token_curve()
    if not curve:
        return

    chart_data: dict[str, list] = {
        "Turn": [],
        "Input Tokens": [],
        "Output Tokens": [],
    }
    for c in curve:
        chart_data["Turn"].append(c["turn"])
        chart_data["Input Tokens"].append(c["input_tokens"])
        chart_data["Output Tokens"].append(c["output_tokens"])

    st.line_chart(
        data={"Input": chart_data["Input Tokens"], "Output": chart_data["Output Tokens"]},
    )


def _render_budget_chart(store: EventStore) -> None:
    """Grouped bar chart: budget limits vs actual usage per role."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for budget charts.")
        return

    budgets = store.get_budget_snapshots()
    if not budgets:
        st.info("No budget data available.")
        return

    # Use latest snapshot
    latest = budgets[-1]
    roles = ["system", "tools", "user_assistant", "images"]
    role_labels = ["System", "Tools", "User/Assistant", "Images"]
    limits = [latest.budget_limits.get(r, 0) for r in roles]
    actual_values = [
        latest.system_tokens,
        latest.tool_tokens,
        latest.user_assistant_tokens,
        latest.image_tokens,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Budget Limit",
        x=role_labels,
        y=limits,
        marker_color="rgba(55, 126, 184, 0.5)",
        text=[f"{v:,}" for v in limits],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Actual Usage",
        x=role_labels,
        y=actual_values,
        marker_color=[
            "rgba(77, 175, 74, 0.8)" if a <= l else "rgba(228, 26, 28, 0.8)"
            for a, l in zip(actual_values, limits)
        ],
        text=[f"{v:,}" for v in actual_values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Token Budget vs Actual Usage",
        barmode="group",
        template="plotly_white",
        height=350,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_system_prompt_trend(store: EventStore) -> None:
    """Stacked area chart: system prompt components across turns."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for trend charts.")
        return

    llm_calls = store.get_llm_calls()
    if not llm_calls or not any(e.system_prompt_breakdown for e in llm_calls):
        st.info("No system prompt breakdown data available.")
        return

    # Collect all component names
    all_components: set[str] = set()
    for e in llm_calls:
        all_components.update(e.system_prompt_breakdown.keys())

    if not all_components:
        return

    colors = [
        "#4a90d9", "#e74c3c", "#f39c12", "#27ae60", "#8e44ad",
        "#2c3e50", "#16a085", "#d35400", "#c0392b", "#2980b9",
    ]

    fig = go.Figure()
    for i, comp in enumerate(sorted(all_components)):
        values = [e.system_prompt_breakdown.get(comp, 0) for e in llm_calls]
        turns = [e.turn for e in llm_calls]
        fig.add_trace(go.Scatter(
            x=turns,
            y=values,
            name=comp,
            stackgroup="one",
            line=dict(color=colors[i % len(colors)]),
            hovertemplate=f"{comp}: %{{y:,}} tokens<extra></extra>",
        ))

    fig.update_layout(
        title="System Prompt Composition Across Turns",
        xaxis_title="Turn",
        yaxis_title="Tokens",
        template="plotly_white",
        height=400,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
