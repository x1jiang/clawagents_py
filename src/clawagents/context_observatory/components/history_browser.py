"""History Browser — browse, load, import, and manage saved observatory sessions.

Lists all sessions saved in ``.clawagents/observatory_history/``.
Each row shows timestamp, model, LLM call count, total tokens, cost, and file size.
Users can:
- Click to load a session into the main inspector view
- Import external ``.json`` files
- Import all sessions from a local folder
- Download or delete individual sessions
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import streamlit as st

from clawagents.context_observatory.store import EventStore, get_history_dir

logger = logging.getLogger(__name__)

_SIZE_UNITS = ["B", "KB", "MB", "GB"]


def _fmt_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    val = float(size_bytes)
    for unit in _SIZE_UNITS[:-1]:
        if abs(val) < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} {_SIZE_UNITS[-1]}"


def _fmt_ts(ts: float | None) -> str:
    """Format a unix timestamp for display."""
    if not ts:
        return "—"
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def render_history_browser(store: EventStore) -> None:
    """Render the history browser tab."""
    st.subheader("📚 Session History")

    # ── Import section ──────────────────────────────────────────────────
    with st.expander("📥 Import Sessions / Packages", expanded=False):
        imp_col1, imp_col2 = st.columns(2)

        with imp_col1:
            st.markdown("**Import Package or File**")
            uploaded = st.file_uploader(
                "Upload session (.zip package or .json file)",
                type=["json", "zip"],
                accept_multiple_files=True,
                key="history_import_files",
            )
            if uploaded:
                _handle_file_import(uploaded)

        with imp_col2:
            st.markdown("**Import Folder**")
            folder_path = st.text_input(
                "Local folder path (session directory or package folder)",
                key="history_import_folder",
                placeholder="/path/to/context-observatory/session_id/",
            )
            if st.button("📂 Import Folder", key="import_folder_btn"):
                if folder_path:
                    _handle_folder_import(folder_path)
                else:
                    st.warning("Please enter a folder path.")

    st.markdown("---")

    # ── Session list ────────────────────────────────────────────────────
    entries = EventStore.list_history()

    if not entries:
        st.info(
            "No saved sessions found.\n\n"
            "Sessions are automatically saved to `.clawagents/context-observatory/<session_id>/` when a conversation runs. "
            "You can also import session packages (.zip / .json) using the controls above."
        )
        return

    st.caption(f"{len(entries)} session(s) in history")

    # ── Column headers ──────────────────────────────────────────────────
    header_cols = st.columns([2.5, 1.5, 1, 1, 1, 1, 0.8, 0.8])
    with header_cols[0]:
        st.markdown("**Session / File**")
    with header_cols[1]:
        st.markdown("**Model**")
    with header_cols[2]:
        st.markdown("**LLM Calls**")
    with header_cols[3]:
        st.markdown("**Events**")
    with header_cols[4]:
        st.markdown("**Cost**")
    with header_cols[5]:
        st.markdown("**Size**")
    with header_cols[6]:
        st.markdown("**Load**")
    with header_cols[7]:
        st.markdown("**Delete**")

    # ── Session rows ────────────────────────────────────────────────────
    for i, entry in enumerate(entries):
        _render_session_row(i, entry, store)


def _render_session_row(
    idx: int,
    entry: dict[str, Any],
    store: EventStore,
) -> None:
    """Render a single session row."""
    filename = entry.get("filename", "unknown")
    model = entry.get("model", "—")
    llm_calls = entry.get("llm_calls", 0)
    event_count = entry.get("event_count", 0)
    cost = entry.get("session_cost_usd")
    size = entry.get("size_bytes", 0)
    path = entry.get("path", "")
    dir_path = entry.get("dir_path") or path

    cost_str = f"${cost:.4f}" if cost else "—"
    is_dir = entry.get("is_directory", False)
    display_name = f"📁 {filename}" if is_dir else f"📄 {filename}"

    row_cols = st.columns([2.5, 1.5, 1, 1, 1, 1, 0.8, 0.8])

    with row_cols[0]:
        st.caption(display_name)
    with row_cols[1]:
        st.caption(model or "—")
    with row_cols[2]:
        st.caption(str(llm_calls) if llm_calls else "—")
    with row_cols[3]:
        st.caption(str(event_count) if event_count else "—")
    with row_cols[4]:
        st.caption(cost_str)
    with row_cols[5]:
        st.caption(_fmt_size(size))
    with row_cols[6]:
        if st.button("📂", key=f"load_hist_{idx}", help="Load this session into Observatory"):
            _load_session(path, store)
    with row_cols[7]:
        if st.button("🗑️", key=f"del_hist_{idx}", help="Delete this session"):
            _delete_session(dir_path if is_dir else path)

    # ── Expanded detail panel ───────────────────────────────────────────
    _render_detail_panel(idx, entry)


def _load_session(path: str, store: EventStore) -> None:
    """Load a saved session into the main event store for viewing."""
    try:
        loaded = EventStore.load_from_json(path)
        # Replace the current store's events
        store.clear()
        store._session_meta = loaded._session_meta
        store._events = loaded._events

        st.toast(
            f"✅ Loaded session with {len(loaded)} events",
            icon="📂",
        )
        st.session_state["_history_loaded_path"] = path
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load session: {e}")


def _delete_session(path: str) -> None:
    """Delete a saved session file or directory."""
    try:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            os.remove(p)
        st.toast("🗑️ Session deleted", icon="✅")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete: {e}")


def _handle_file_import(uploaded_files: list) -> None:
    """Import uploaded JSON or ZIP files into the history directory."""
    import zipfile

    history_dir = get_history_dir()
    history_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    for uploaded in uploaded_files:
        try:
            filename = uploaded.name
            raw = uploaded.read()

            if filename.lower().endswith(".zip"):
                session_id = Path(filename).stem
                dest_dir = history_dir / session_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                    zf.extractall(dest_dir)
                imported += 1
            else:
                # Validate JSON
                json.loads(raw)
                dest = history_dir / filename
                if dest.exists():
                    stem = dest.stem
                    suffix = dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = history_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                dest.write_bytes(raw)
                imported += 1
        except Exception as e:
            st.warning(f"Skipped {uploaded.name}: {e}")

    if imported:
        st.toast(f"✅ Imported {imported} session package(s)", icon="📥")
        st.rerun()


def _handle_folder_import(folder_path: str) -> None:
    """Import a local folder or session directory into history."""
    src = Path(folder_path).expanduser().resolve()
    if not src.is_dir():
        st.error(f"Not a valid directory: {folder_path}")
        return

    history_dir = get_history_dir()
    history_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    # Check if src itself is a session folder (contains session.json)
    if (src / "session.json").is_file():
        dest = history_dir / src.name
        if dest.exists() and dest != src:
            dest = history_dir / f"{src.name}_{int(os.path.getmtime(src))}"
        if dest != src:
            shutil.copytree(src, dest, dirs_exist_ok=True)
        imported += 1
    else:
        # Import any subdirectories or json files inside src
        for sub in src.iterdir():
            if sub.is_dir() and (sub / "session.json").is_file():
                dest = history_dir / sub.name
                if dest != sub:
                    shutil.copytree(sub, dest, dirs_exist_ok=True)
                imported += 1
            elif sub.is_file() and sub.suffix.lower() == ".json":
                try:
                    json.loads(sub.read_text(encoding="utf-8"))
                    dest = history_dir / sub.name
                    if dest != sub:
                        shutil.copy2(sub, dest)
                    imported += 1
                except Exception:
                    pass

    if imported:
        st.toast(f"✅ Imported {imported} session(s) from folder", icon="📂")
        st.rerun()
    else:
        st.warning(f"No valid session directories or .json files found in {folder_path}")


def _render_detail_panel(idx: int, entry: dict[str, Any]) -> None:
    """Render details and download buttons for a session."""
    with st.expander(f"ℹ️ Details — {entry.get('filename')}", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"**Started:** {_fmt_ts(entry.get('started_at'))}")
        with cols[1]:
            st.markdown(f"**Completed:** {_fmt_ts(entry.get('completed_at'))}")
        with cols[2]:
            st.markdown(f"**Status:** {entry.get('status', '—')}")

        dl_cols = st.columns(2)
        path = entry.get("path", "")
        dir_path = entry.get("dir_path", "")

        # 1. Download Session ZIP Package
        with dl_cols[0]:
            try:
                loaded_store = EventStore.load_from_json(path)
                zip_bytes = loaded_store.export_package_zip()
                st.download_button(
                    label="📦 Download ZIP Package",
                    data=zip_bytes,
                    file_name=f"{entry.get('filename', 'session')}.zip",
                    mime="application/zip",
                    key=f"dl_zip_hist_{idx}",
                )
            except Exception as e:
                st.caption(f"ZIP package unavailable: {e}")

        # 2. Download Session JSON
        with dl_cols[1]:
            if path and Path(path).exists():
                data = Path(path).read_text(encoding="utf-8")
                st.download_button(
                    label="📄 Download JSON File",
                    data=data,
                    file_name=entry.get("filename", "session.json") if not entry.get("filename", "").endswith(".json") else entry.get("filename"),
                    mime="application/json",
                    key=f"dl_json_hist_{idx}",
                )

