"""Chat panel component — VSCode-style streaming chat interface.

Renders a chat UI that mirrors the VSCode extension's webview (App.tsx),
supporting all ``HostToWebview`` event types:

- **assistant_delta / assistant_message** — streaming text with typewriter effect
- **tool_started / tool_completed** — real-time tool execution status
- **permission_required** — interactive Allow / Deny buttons
- **ask_user_required** — question prompt with text input
- **file_changed** — file modification notifications
- **usage** — token usage stats
- **status / error / done / cancelled** — lifecycle indicators

The chat state is stored in ``st.session_state["chat_items"]`` as a list of
typed dicts (``ChatItem``), mirroring the ``ChatItem`` union type from the
VSCode webview's ``App.tsx``.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


# ── Role/kind styling ────────────────────────────────────────────────────

_ROLE_STYLES: dict[str, tuple[str, str]] = {
    "user": ("👤", "#0d6efd"),
    "assistant": ("🤖", "#198754"),
    "tool": ("⚡", "#fd7e14"),
    "status": ("ℹ️", "#6c757d"),
    "error": ("❌", "#dc3545"),
    "file": ("📁", "#17a2b8"),
    "permission": ("🔒", "#e83e8c"),
    "ask": ("❓", "#6f42c1"),
}


# ── Chat items management ───────────────────────────────────────────────


def _ensure_chat_items() -> list[dict[str, Any]]:
    if "chat_items" not in st.session_state:
        st.session_state["chat_items"] = []
    return st.session_state["chat_items"]


def get_chat_items() -> list[dict[str, Any]]:
    """Return the current chat items list."""
    return _ensure_chat_items()


def clear_chat_items() -> None:
    """Clear all chat items."""
    st.session_state["chat_items"] = []


def add_chat_item(item: dict[str, Any]) -> None:
    """Append a chat item to the history."""
    items = _ensure_chat_items()
    items.append(item)


def apply_sse_event(event: dict[str, Any]) -> None:
    """Apply an SSE event to the chat items list.

    This replicates the logic from App.tsx's message handler (L600-850)
    to maintain the same state transitions.
    """
    items = _ensure_chat_items()
    event_type = event.get("type", "")

    if event_type == "user_echo":
        items.append({"kind": "user", "text": event.get("text", "")})
        st.session_state["chat_streaming"] = False

    elif event_type == "status":
        # Replace the last status if consecutive
        if items and items[-1].get("kind") == "status":
            items[-1] = {"kind": "status", "text": event.get("message", "")}
        else:
            items.append({"kind": "status", "text": event.get("message", "")})

    elif event_type == "assistant_delta":
        delta = event.get("delta", "")
        streaming = st.session_state.get("chat_streaming", False)
        if items and items[-1].get("kind") == "assistant" and streaming:
            items[-1]["text"] = items[-1].get("text", "") + delta
        else:
            st.session_state["chat_streaming"] = True
            items.append({"kind": "assistant", "text": delta})

    elif event_type == "assistant_message":
        was_streaming = st.session_state.get("chat_streaming", False)
        st.session_state["chat_streaming"] = False
        text = event.get("text", "")
        if not text.strip():
            return
        if items and items[-1].get("kind") == "assistant":
            if was_streaming or items[-1].get("text") == text:
                items[-1] = {"kind": "assistant", "text": text}
                return
        items.append({"kind": "assistant", "text": text})

    elif event_type == "tool_started":
        st.session_state["chat_streaming"] = False
        items.append({
            "kind": "tool",
            "id": event.get("id", ""),
            "name": event.get("name", "tool"),
            "args": event.get("args"),
            "filePath": event.get("filePath"),
            "status": "running",
        })

    elif event_type == "tool_completed":
        # Find and update the matching tool item
        matched = False
        tool_id = event.get("id", "")
        tool_name = event.get("name", "tool")
        for i in range(len(items) - 1, -1, -1):
            it = items[i]
            if it.get("kind") == "tool" and (
                it.get("id") == tool_id
                or (not tool_id and it.get("name") == tool_name and it.get("status") == "running")
            ):
                items[i] = {
                    **it,
                    "status": "done",
                    "success": event.get("success", True),
                    "output": event.get("output", ""),
                    "filePath": event.get("filePath") or it.get("filePath"),
                }
                matched = True
                break
        if not matched:
            items.append({
                "kind": "tool",
                "id": tool_id or tool_name,
                "name": tool_name,
                "status": "done",
                "success": event.get("success", True),
                "output": event.get("output", ""),
                "filePath": event.get("filePath"),
            })

    elif event_type == "permission_required":
        items.append({
            "kind": "permission",
            "requestId": event.get("requestId", ""),
            "tool": event.get("tool", "tool"),
            "filePath": event.get("filePath"),
            "command": event.get("command"),
            "reason": event.get("reason"),
            "resolved": False,
        })

    elif event_type == "ask_user_required":
        items.append({
            "kind": "ask",
            "requestId": event.get("requestId", ""),
            "question": event.get("question", ""),
            "draft": "",
            "resolved": False,
        })

    elif event_type == "file_changed":
        path = event.get("path", "")
        if path:
            items.append({
                "kind": "file",
                "path": path,
                "snapshotId": event.get("snapshotId"),
                "snapshotRel": event.get("snapshotRel"),
            })

    elif event_type == "usage":
        st.session_state["chat_usage"] = {
            "promptTokens": event.get("promptTokens"),
            "completionTokens": event.get("completionTokens"),
            "totalTokens": event.get("totalTokens"),
            "lastInputTokens": event.get("lastInputTokens"),
            "runCostUsd": event.get("runCostUsd"),
            "sessionCostUsd": event.get("sessionCostUsd"),
        }

    elif event_type == "compact_progress":
        phase = event.get("phase", "")
        msg = event.get("message", "")
        if phase and phase not in ("end", "failed", "dropped"):
            # Show compaction in progress
            if items and items[-1].get("kind") == "status":
                items[-1] = {"kind": "status", "text": f"🗜️ Compacting… ({phase})"}
            else:
                items.append({"kind": "status", "text": f"🗜️ Compacting… ({phase})"})

    elif event_type == "done":
        st.session_state["chat_streaming"] = False
        st.session_state["chat_busy"] = False
        status = event.get("status", "done")
        iterations = event.get("iterations")
        run_cost = event.get("runCostUsd")

        # Fallback: if there is a final result text, make sure it is added as an assistant message
        result_text = event.get("result")
        if result_text and result_text.strip():
            # If the last item is assistant, update it (useful for streaming finishes);
            # otherwise append a new assistant item.
            if items and items[-1].get("kind") == "assistant":
                items[-1]["text"] = result_text
            else:
                items.append({"kind": "assistant", "text": result_text})

        # Build done status line (mirrors App.tsx L825-833)
        status_parts = [f"Done · {status}"]
        if iterations is not None:
            status_parts.append(f"{iterations} iters")
        if run_cost is not None:
            status_parts.append(f"run ~${run_cost:.4f}")
        status_text = " · ".join(status_parts)
        # Remove trailing status items and add done status
        filtered = [it for it in items if it.get("kind") != "status"]
        filtered.append({"kind": "status", "text": status_text})
        st.session_state["chat_items"] = filtered

    elif event_type == "error":
        st.session_state["chat_streaming"] = False
        st.session_state["chat_busy"] = False
        message = event.get("message", "Unknown error")
        if message == "cancelled":
            items.append({"kind": "status", "text": "Cancelled"})
        else:
            items.append({"kind": "error", "text": message})

    elif event_type == "cancelled":
        st.session_state["chat_streaming"] = False
        st.session_state["chat_busy"] = False
        items.append({"kind": "status", "text": "Cancelled"})


# ── Rendering ────────────────────────────────────────────────────────────


def render_chat_panel() -> str | None:
    """Render the full chat panel and return user input if submitted."""
    st.subheader("💬 Chat")

    items = _ensure_chat_items()

    # Display existing items
    for i, item in enumerate(items):
        kind = item.get("kind", "user")
        renderer = _RENDERERS.get(kind)
        if renderer:
            renderer(i, item)

    # Usage bar
    usage = st.session_state.get("chat_usage", {})
    if usage and any(usage.values()):
        _render_usage_bar(usage)

    # Input
    is_busy = st.session_state.get("chat_busy", False)
    user_input = st.chat_input(
        "Agent is running…" if is_busy else "Send a message to the agent...",
        key="chat_input",
        disabled=is_busy,
    )

    return user_input


# ── Item renderers ───────────────────────────────────────────────────────


def _render_user(idx: int, item: dict[str, Any]) -> None:
    with st.chat_message("user", avatar="👤"):
        st.markdown(item.get("text", ""))


def _render_assistant(idx: int, item: dict[str, Any]) -> None:
    with st.chat_message("assistant", avatar="🤖"):
        text = item.get("text", "")
        # Show streaming cursor if this is the last item and we're streaming
        is_last = idx == len(_ensure_chat_items()) - 1
        is_streaming = st.session_state.get("chat_streaming", False)
        if is_last and is_streaming:
            st.markdown(text + "▊")
        else:
            st.markdown(text)


def _render_tool(idx: int, item: dict[str, Any]) -> None:
    status = item.get("status", "running")
    name = item.get("name", "tool")
    success = item.get("success", True)
    file_path = item.get("filePath")

    if status == "running":
        icon = "⏳"
        status_text = "Running…"
        color = "#fd7e14"
    elif success:
        icon = "✅"
        status_text = "Done"
        color = "#198754"
    else:
        icon = "❌"
        status_text = "Failed"
        color = "#dc3545"

    # Tool header
    header_parts = [f"{icon} **{name}**"]
    if file_path:
        # Show just the filename
        short_path = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
        header_parts.append(f"`{short_path}`")
    header_parts.append(f"— {status_text}")

    with st.expander(" ".join(header_parts), expanded=(status == "running")):
        # Show args if present
        args = item.get("args")
        if args:
            if isinstance(args, dict):
                # Show key args compactly
                for key, val in args.items():
                    val_str = str(val)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "…"
                    st.caption(f"**{key}**: `{val_str}`")
            else:
                st.caption(str(args)[:500])

        # Show output
        output = item.get("output", "")
        if output:
            if len(output) > 500:
                st.code(output[:2000], language="text")
            else:
                st.code(output, language="text")

        if status == "running":
            st.spinner("Executing…")


def _render_status(idx: int, item: dict[str, Any]) -> None:
    text = item.get("text", "")
    st.caption(f"ℹ️ {text}")


def _render_error(idx: int, item: dict[str, Any]) -> None:
    st.error(f"❌ {item.get('text', 'Unknown error')}")


def _render_file(idx: int, item: dict[str, Any]) -> None:
    path = item.get("path", "")
    if path:
        short = path.rsplit("/", 1)[-1] if "/" in path else path
        st.info(f"📁 File changed: `{short}`\n\n`{path}`")


def _render_permission(idx: int, item: dict[str, Any]) -> None:
    if item.get("resolved"):
        st.caption(f"🔒 Permission for **{item.get('tool', 'tool')}** — resolved")
        return

    tool = item.get("tool", "tool")
    reason = item.get("reason", "")
    file_path = item.get("filePath", "")
    command = item.get("command", "")
    request_id = item.get("requestId", "")

    st.warning(f"🔒 **Permission Required**: {tool}")
    if reason:
        st.caption(f"Reason: {reason}")
    if file_path:
        st.caption(f"File: `{file_path}`")
    if command:
        st.caption(f"Command: `{command}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Allow Once", key=f"perm_once_{idx}_{request_id}"):
            st.session_state[f"perm_action_{request_id}"] = "allow_once"
            item["resolved"] = True
    with col2:
        if st.button("✅ Always Allow", key=f"perm_always_{idx}_{request_id}"):
            st.session_state[f"perm_action_{request_id}"] = "allow_always"
            item["resolved"] = True
    with col3:
        if st.button("🚫 Deny", key=f"perm_deny_{idx}_{request_id}"):
            st.session_state[f"perm_action_{request_id}"] = "deny"
            item["resolved"] = True


def _render_ask(idx: int, item: dict[str, Any]) -> None:
    if item.get("resolved"):
        answer = item.get("answer", "(skipped)")
        st.caption(f"❓ {item.get('question', '')} — **{answer}**")
        return

    request_id = item.get("requestId", "")
    question = item.get("question", "")

    st.info(f"❓ **Agent asks**: {question}")

    answer = st.text_input(
        "Your answer:",
        key=f"ask_input_{idx}_{request_id}",
        placeholder="Type your response…",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Submit", key=f"ask_submit_{idx}_{request_id}"):
            st.session_state[f"ask_action_{request_id}"] = {"answer": answer, "skip": False}
            item["resolved"] = True
            item["answer"] = answer
    with col2:
        if st.button("⏭️ Skip", key=f"ask_skip_{idx}_{request_id}"):
            st.session_state[f"ask_action_{request_id}"] = {"answer": None, "skip": True}
            item["resolved"] = True
            item["answer"] = "(skipped)"


def _render_usage_bar(usage: dict[str, Any]) -> None:
    """Render a compact usage summary bar."""
    prompt = usage.get("promptTokens") or 0
    completion = usage.get("completionTokens") or 0
    total = usage.get("totalTokens") or (prompt + completion)
    session_cost = usage.get("sessionCostUsd")

    parts = []
    if prompt:
        parts.append(f"📥 {prompt:,} in")
    if completion:
        parts.append(f"📤 {completion:,} out")
    if total:
        parts.append(f"📊 {total:,} total")
    if session_cost is not None:
        parts.append(f"💰 ${session_cost:.4f}")

    if parts:
        st.caption(" · ".join(parts))


# Registry of renderers by item kind
_RENDERERS = {
    "user": _render_user,
    "assistant": _render_assistant,
    "tool": _render_tool,
    "status": _render_status,
    "error": _render_error,
    "file": _render_file,
    "permission": _render_permission,
    "ask": _render_ask,
}


# ── Legacy API (backwards compatibility) ─────────────────────────────────


def add_chat_message(role: str, content: str, **extra: Any) -> None:
    """Legacy: append a message as a ChatItem.

    Maps old role-based messages to the new ChatItem format.
    """
    kind_map = {"user": "user", "assistant": "assistant", "system": "status", "tool": "tool"}
    kind = kind_map.get(role, "status")

    if kind == "tool":
        add_chat_item({
            "kind": "tool",
            "id": extra.get("tool_call_id", ""),
            "name": extra.get("tool_name", "tool"),
            "status": "done",
            "success": True,
            "output": content,
        })
    elif kind == "status":
        add_chat_item({"kind": "status", "text": content})
    else:
        add_chat_item({"kind": kind, "text": content})


def render_message_details(messages: list[dict[str, Any]]) -> None:
    """Render detailed message list with token counts (for context inspector)."""
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        emoji = {"system": "🔧", "user": "👤", "assistant": "🤖", "tool": "⚡"}.get(role, "💬")
        token_count = msg.get("token_count", 0)
        content_length = msg.get("content_length", 0)

        header = (
            f"{emoji} **{role.upper()}** — "
            f"{token_count:,} tokens · {content_length:,} chars"
        )

        with st.expander(header, expanded=(i == 0 and role == "system")):
            preview = msg.get("content_preview", msg.get("content", ""))
            if role == "system":
                st.code(preview[:3000], language="markdown")
            elif role == "tool":
                st.code(preview[:2000], language="text")
            else:
                st.markdown(preview[:2000])

            if msg.get("has_tool_calls"):
                st.caption("🔧 Contains tool calls")
            if msg.get("tool_call_id"):
                st.caption(f"📎 tool_call_id: `{msg['tool_call_id']}`")
