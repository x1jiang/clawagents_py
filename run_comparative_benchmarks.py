import asyncio
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, List
from tabulate import tabulate
import os

TASKS = {
    "Simple Read": "Read the contents of pyproject.toml and tell me the version number.",
    "OS Command": "Use the execute tool to run `echo 'benchmark test'`, then use the execute tool to run `date`. Return both outputs.",
    "Multi-step Chain": "Create a file named 'temp_bench.txt' with the word 'Claw', then read it, then write a new file 'temp_bench2.txt' with the word 'Agents'.",
    "Code Generation": "Calculate the 10th fibonacci number by writing a small python script to a file 'fib.py', executing it using the execute tool, and returning the result."
}

PROVIDERS = ["openai", "gemini"]


def get_providers() -> list:
    """Use PROVIDER env to run a single provider, or both by default."""
    single = os.getenv("PROVIDER", "").lower()
    if single in ("openai", "gemini"):
        return [single]
    return PROVIDERS


def parse_output(output: str, engine: str, provider: str, elapsed: float) -> list:
    tool_calls = 0
    iterations = 0
    if "Tool calls:" in output:
        match = re.search(r"Tool calls:\s*(\d+)", output)
        if match:
            tool_calls = int(match.group(1))
    if "Iterations:" in output:
        match = re.search(r"Iterations:\s*(\d+)", output)
        if match:
            iterations = int(match.group(1))
    return [provider, engine, f"{elapsed:.2f}s", tool_calls, iterations]


async def run_benchmark():
    # This script lives in clawagents_py/; repo root is parent
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    py_cwd = script_dir  # clawagents_py
    ts_cwd = repo_root / "clawagents"

    results = []

    providers = get_providers()
    print("\nStarting Comprehensive Comparative Benchmarks (Streaming Mode)...\n")
    print(f"Providers: {' + '.join(p.upper() for p in providers)} | STREAMING=1\n")

    env = os.environ.copy()
    env["STREAMING"] = "1"

    for provider in providers:
        env["PROVIDER"] = provider
        print(f"\n--- Provider: {provider.upper()} ---")

        for task_name, prompt in TASKS.items():
            print(f"Task: {task_name}...")
            print(f"  Prompt: {prompt[:60]}...")

            # --- PY RUN ---
            print("  Running Python Engine...")
            p_start = time.time()
            py_proc = subprocess.run(
                ["python", "-m", "clawagents", "--task", prompt],
                cwd=str(py_cwd), capture_output=True, text=True, env=env
            )
            p_elapsed = time.time() - p_start
            py_metrics = parse_output(py_proc.stdout, "clawagents_py", provider, p_elapsed)
            py_metrics.insert(0, task_name)
            results.append(py_metrics)
            if py_proc.returncode != 0:
                print(f"   [Error in Py] {py_proc.stderr[:100]}")

            # --- TS RUN ---
            print("  Running TypeScript Engine...")
            t_start = time.time()
            ts_proc = subprocess.run(
                ["npx", "tsx", "src/index.ts", "--task", prompt],
                cwd=str(ts_cwd), capture_output=True, text=True, env=env
            )
            t_elapsed = time.time() - t_start
            ts_metrics = parse_output(ts_proc.stdout, "clawagents_ts", provider, t_elapsed)
            ts_metrics.insert(0, task_name)
            results.append(ts_metrics)
            if ts_proc.returncode != 0:
                print(f"   [Error in TS] {ts_proc.stderr[:100]}")

            # Cleanup
            subprocess.run(["rm", "-f", "temp_bench.txt", "temp_bench2.txt", "fib.py"], cwd=str(py_cwd))
            subprocess.run(["rm", "-f", "temp_bench.txt", "temp_bench2.txt", "fib.py"], cwd=str(ts_cwd))

            print("-" * 50)

    print("\n\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)

    headers = ["Task", "Provider", "Engine", "Total Time", "Tool Calls", "Iterations"]
    print(tabulate(results, headers=headers, tablefmt="github"))
    
    with open(py_cwd / "benchmark_report.md", "w") as f:
        f.write("# Comprehensive Benchmark Report\n\n")
        f.write(f"Providers: {' + '.join(p.upper() for p in providers)} | STREAMING=1\n\n")
        f.write(tabulate(results, headers=headers, tablefmt="github"))
        f.write("\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
