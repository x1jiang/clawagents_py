#!/usr/bin/env python3
"""Freeze, run and compare paired efficiency experiments. Live calls require --allow-live."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

# Never discover a parent .env during a benchmark. Live keys must be explicitly exported.
os.environ["CLAWAGENTS_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawagents.benchmarks.efficiency import compare, freeze, runtime_digest
from clawagents.benchmarks.runner import run


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source-hash", help="Print the actual imported runtime's source digest")
    make = commands.add_parser("freeze", help="Freeze cases, prices, runtime and serving configuration before evaluation")
    make.add_argument("--suite", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--model", default="scripted-fixture")
    make.add_argument("--provider", choices=["fixture", "openai", "mantle"], default="fixture")
    make.add_argument("--base-url", default="")
    make.add_argument("--repeats", type=int, default=2)
    make.add_argument("--full-sends", type=int, choices=[0, 1, 2], default=1)
    make.add_argument("--prices", type=Path, help="JSON: four token-category rates per million plus source")
    make.add_argument("--baseline-sha256")
    make.add_argument("--candidate-sha256")
    make.add_argument("--max-iterations", type=int, default=12)
    make.add_argument("--timeout-seconds", type=int, default=120)
    make.add_argument("--reasoning-effort", default="")
    make.add_argument("--max-tokens", type=int, default=2048)
    make.add_argument("--min-cost-reduction", type=float, default=0.05)
    make.add_argument("--max-latency-ratio", type=float, default=1.25)
    execute = commands.add_parser("run", help="Run each selected pair in a fresh in-memory workspace")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--suite", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--split", choices=["development", "heldout"], default="development")
    execute.add_argument("--arm", choices=["baseline", "candidate", "both"], default="both")
    execute.add_argument("--allow-live", action="store_true")
    report = commands.add_parser("compare", help="Reject invalid pairs and apply frozen qualification gates")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--rows", nargs="+", type=Path, required=True)
    report.add_argument("--split", choices=["development", "heldout"], default="heldout")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "source-hash":
            print(runtime_digest())
        elif args.command == "freeze":
            values = vars(args).copy()
            values.pop("command")
            output = values.pop("output")
            values["suite"] = load(values["suite"])
            values["prices"] = load(values["prices"]) if values["prices"] else None
            manifest = freeze(**values)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
            print(manifest["sha256"])
        elif args.command == "run":
            rows = asyncio.run(run(load(args.manifest), load(args.suite), split=args.split,
                                   output=args.output, arm=args.arm, allow_live=args.allow_live))
            print(json.dumps({"runs": len(rows), "successes": sum(r["passed"] for r in rows)}))
        else:
            rows = [json.loads(line) for path in args.rows for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            result = compare(load(args.manifest), rows, split=args.split)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            print(json.dumps({key: result[key] for key in ("mode", "capability_pass", "efficiency_pass", "qualified")}))
            return 0 if result["qualified"] else 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"benchmark rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
