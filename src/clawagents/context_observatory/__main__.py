"""CLI entry point for the Context Observatory.

Usage:
    # Launch the Streamlit dashboard against an existing ClawAgents gateway
    python -m clawagents.context_observatory

    # Replay a previously exported session (UI only, no server)
    python -m clawagents.context_observatory --replay session.json

    # Compare two sessions
    python -m clawagents.context_observatory --compare a.json b.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="context-observatory",
        description="Interactive context management observability for ClawAgents",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Path to a JSON session file to replay in the dashboard",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("SESSION_A", "SESSION_B"),
        default=None,
        help="Paths to two JSON session files to compare side-by-side",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit UI server (default: 8501)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Don't open the browser automatically",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Launch Streamlit UI ──────────────────────────────────────────
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--browser.gatherUsageStats", "false",
    ]

    if args.no_browser:
        cmd.extend(["--server.headless", "true"])

    # Pass args through environment variables
    env = os.environ.copy()
    if args.replay:
        replay_path = Path(args.replay).resolve()
        if not replay_path.exists():
            print(f"Error: replay file not found: {args.replay}", file=sys.stderr)
            sys.exit(1)
        env["OBSERVATORY_REPLAY"] = str(replay_path)

    if args.compare:
        paths = []
        for p in args.compare:
            resolved = Path(p).resolve()
            if not resolved.exists():
                print(f"Error: compare file not found: {p}", file=sys.stderr)
                sys.exit(1)
            paths.append(str(resolved))
        env["OBSERVATORY_COMPARE_A"] = paths[0]
        env["OBSERVATORY_COMPARE_B"] = paths[1]

    print(f"🔭 Starting Context Observatory UI on port {args.port}...")
    print(f"   App: {app_path}")
    if args.replay:
        print(f"   Replay: {args.replay}")
    if args.compare:
        print(f"   Compare: {args.compare[0]} vs {args.compare[1]}")

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Observatory stopped.")
    except FileNotFoundError:
        print(
            "\nError: streamlit is not installed. Install it with:\n"
            "  pip install streamlit plotly\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nStreamlit exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
