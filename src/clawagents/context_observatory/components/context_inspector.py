"""Context Inspector — linear timeline of LLM calls with token usage + full context.

No turn selector — each LLM call is appended as a card in chronological order.
Each card shows:
1. Precise API-reported token usage
2. Messages sent to the LLM — chunked preview, click to expand full content
3. LLM response — expandable
"""

from __future__ import annotations

import streamlit as st

from clawagents.context_observatory.events import LLMCallEvent
from clawagents.context_observatory.store import EventStore


# ── Helpers ──────────────────────────────────────────────────────────────

_ROLE_EMOJI = {"system": "🔧", "user": "👤", "assistant": "🤖", "tool": "⚡"}
_CHUNK_PREVIEW_LEN = 150  # chars shown in the collapsed row


def _fmt(n: int | None) -> str:
    """Format a token count, returning '—' for None/0."""
    if not n:
        return "—"
    return f"{n:,}"


# ── Public entry point ───────────────────────────────────────────────────


def render_context_inspector(store: EventStore) -> None:
    """Render the context inspector as a linear timeline."""
    st.subheader("🔍 Context Inspector")

    llm_calls = store.get_llm_calls()
    if not llm_calls:
        st.info("No LLM calls recorded yet. Start a conversation to see context details.")
        return

    # Summary bar
    total_in = sum(e.total_input_tokens or 0 for e in llm_calls)
    total_out = sum(e.total_output_tokens or 0 for e in llm_calls)
    cols = st.columns(4)
    with cols[0]:
        st.metric("LLM Calls", len(llm_calls))
    with cols[1]:
        st.metric("Total Input", _fmt(total_in))
    with cols[2]:
        st.metric("Total Output", _fmt(total_out))
    with cols[3]:
        st.metric("Grand Total", _fmt(total_in + total_out))

    st.markdown("---")

    # ── Linear timeline: one card per LLM call ───────────────────────────
    for idx, event in enumerate(llm_calls):
        _render_llm_call_card(idx, event, len(llm_calls))


def _render_llm_call_card(idx: int, event: LLMCallEvent, total_calls: int) -> None:
    """Render a single LLM call as a card."""
    input_t = event.total_input_tokens or 0
    output_t = event.total_output_tokens or 0
    cached_t = event.cached_input_tokens or 0
    reasoning_t = event.reasoning_tokens or 0

    # Card header
    label = event.call_label if hasattr(event, 'call_label') else ""
    header = (
        f"**Turn {event.turn}{label}** — "
        f"📥 {_fmt(input_t)} in · 📤 {_fmt(output_t)} out"
    )
    if cached_t:
        header += f" · 💾 {_fmt(cached_t)} cached"
    if reasoning_t:
        header += f" · 🧠 {_fmt(reasoning_t)} reasoning"

    with st.expander(header, expanded=(idx == total_calls - 1)):
        # Token metrics row
        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric("Input", _fmt(input_t))
        with metric_cols[1]:
            st.metric("Output", _fmt(output_t))
        with metric_cols[2]:
            st.metric("Cached", _fmt(cached_t))
        with metric_cols[3]:
            st.metric("Reasoning", _fmt(reasoning_t))
        with metric_cols[4]:
            st.metric("Total", _fmt(input_t + output_t))

        # Context window utilisation
        ctx_win = event.context_window or 0
        if ctx_win > 0 and input_t:
            pct = input_t / ctx_win * 100.0
            _render_utilization_bar(input_t, ctx_win, pct)

        # ── Messages sent to LLM ────────────────────────────────────
        if event.messages:
            st.markdown("##### 📥 Messages Sent to LLM")
            st.caption(f"{len(event.messages)} messages in context")

            for i, msg in enumerate(event.messages):
                _render_message_row(idx, i, msg)

            # Raw JSON payload toggle
            with st.expander("Show Raw JSON Payload"):
                import json
                raw_payload = [
                    {
                        "role": m.role,
                        "content": m.full_content or m.content_preview,
                        **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})
                    }
                    for m in event.messages
                ]
                raw_json = json.dumps(raw_payload, indent=2, ensure_ascii=False)
                # For very large payloads, offer download instead
                if len(raw_json) > 50_000:
                    st.warning(
                        f"Raw payload is large ({len(raw_json):,} chars). "
                        "Use the download button below."
                    )
                    st.download_button(
                        "📥 Download Full Context JSON",
                        data=raw_json,
                        file_name=f"context_turn_{event.turn}.json",
                        mime="application/json",
                        key=f"dl_ctx_{idx}",
                    )
                else:
                    st.code(raw_json, language="json")
                    st.download_button(
                        "📥 Download as file",
                        data=raw_json,
                        file_name=f"context_turn_{event.turn}.json",
                        mime="application/json",
                        key=f"dl_ctx_{idx}",
                    )

            # Full content viewer (opened by button click)
            _render_message_drawer(idx, event)

        # ── LLM Response ────────────────────────────────────────────
        if event.response_text_preview:
            st.markdown("##### 🤖 Response")
            resp_text = event.response_text_preview
            if len(resp_text) > 500:
                st.text(resp_text[:500] + "…")
                with st.expander("Show full response"):
                    st.text(resp_text)
            else:
                st.text(resp_text)

        # ── Tool calls ──────────────────────────────────────────────
        if event.tool_calls_made:
            st.markdown("##### 🔧 Tool Calls")
            for tc in event.tool_calls_made:
                status = "✅" if tc.success else ("❌" if tc.success is False else "⏳")
                duration = f" · {tc.duration_ms}ms" if tc.duration_ms else ""
                with st.expander(f"{status} {tc.tool_name}{duration}", expanded=False):
                    if tc.args_preview:
                        st.code(tc.args_preview[:2000], language="json")
                    if tc.output_preview:
                        st.code(tc.output_preview[:2000], language="text")


def _render_message_row(turn_idx: int, msg_idx: int, msg) -> None:
    """Render a single message as a compact row with a view button."""
    emoji = _ROLE_EMOJI.get(msg.role, "💬")
    preview = (msg.content_preview or "").replace("\n", " ")
    if len(preview) > _CHUNK_PREVIEW_LEN:
        preview = preview[:_CHUNK_PREVIEW_LEN] + "…"

    chars = msg.content_length or 0

    row_cols = st.columns([0.8, 5, 1.2, 0.8])
    with row_cols[0]:
        st.markdown(f"**{emoji} {msg.role}**")
    with row_cols[1]:
        st.caption(preview if preview else "(empty)")
    with row_cols[2]:
        st.caption(f"{chars:,} chars")
    with row_cols[3]:
        if st.button("👁️", key=f"view_{turn_idx}_{msg_idx}", help="View full content"):
            st.session_state[f"drawer_{turn_idx}"] = msg_idx


def _render_message_drawer(turn_idx: int, event: LLMCallEvent) -> None:
    """Render the full content drawer for a selected message.

    Handles large content gracefully:
    - < 10K chars: inline display
    - 10K-50K chars: inline display + download button
    - > 50K chars: download button only (no inline)
    """
    drawer_key = f"drawer_{turn_idx}"
    active = st.session_state.get(drawer_key)
    if active is None or active < 0 or active >= len(event.messages):
        return

    msg = event.messages[active]
    emoji = _ROLE_EMOJI.get(msg.role, "💬")
    full = msg.full_content or msg.content_preview or "(empty)"
    content_len = msg.content_length or len(full)

    st.markdown("---")
    hcols = st.columns([8, 1])
    with hcols[0]:
        st.markdown(
            f"**{emoji} Full Content — {msg.role}** "
            f"({content_len:,} chars)"
        )
    with hcols[1]:
        if st.button("✖", key=f"close_{turn_idx}"):
            del st.session_state[drawer_key]
            st.rerun()

    if msg.tool_call_id:
        st.caption(f"📎 tool_call_id: `{msg.tool_call_id}`")

    # Graduated display based on content size
    if content_len > 50_000:
        # Very large — download only
        st.warning(
            f"Content is very large ({content_len:,} chars). "
            "Showing first 2,000 chars below. Use the download button for full content."
        )
        st.code(full[:2_000] + f"\n\n... [{content_len - 2_000:,} more chars]", language="text")
        st.download_button(
            "📥 Download Full Content (.txt)",
            data=full,
            file_name=f"msg_{msg.role}_{turn_idx}_{active}.txt",
            mime="text/plain",
            key=f"dl_msg_{turn_idx}_{active}",
        )
    elif content_len > 10_000:
        # Large — show inline with download option
        st.code(full, language="text")
        st.download_button(
            "📥 Download as .txt",
            data=full,
            file_name=f"msg_{msg.role}_{turn_idx}_{active}.txt",
            mime="text/plain",
            key=f"dl_msg_{turn_idx}_{active}",
        )
    else:
        # Normal — inline display
        st.code(full, language="text")


# ── Internal renderers ───────────────────────────────────────────────────


def _render_utilization_bar(used: int, total: int, pct: float) -> None:
    if pct < 60:
        color, icon = "#198754", "🟢"
    elif pct < 80:
        color, icon = "#fd7e14", "🟡"
    else:
        color, icon = "#dc3545", "🔴"

    st.markdown(
        f'<div style="margin:4px 0 12px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px;">'
        f'<span>{icon} Context Window: {used:,} / {total:,} tokens</span>'
        f'<span style="color:{color};font-weight:600;">{pct:.1f}%</span>'
        f'</div>'
        f'<div style="background:#2d2d2d;border-radius:6px;height:12px;overflow:hidden;">'
        f'<div style="width:{min(pct, 100):.1f}%;background:{color};height:100%;'
        f'border-radius:6px;transition:width 0.3s;"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
