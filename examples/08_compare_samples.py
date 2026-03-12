"""Example: Multi-sample comparison (GRPO-inspired)

Runs the same task N times and picks the best result based on objective scoring.
Useful for high-stakes tasks where you want the best possible outcome.
"""
import asyncio
from clawagents import create_claw_agent


async def main():
    agent = create_claw_agent(
        "gpt-5-mini",
        learn=True,
        rethink=True,
    )

    result = await agent.compare(
        "Write a Python function to merge two sorted lists efficiently",
        n_samples=3,             # run 3 times, pick the best
    )

    print(f"Best score: {result['best_score']}")
    print(f"Best result:\n{result['best_result']}")
    print(f"\nAll scores: {result['all_scores']}")


if __name__ == "__main__":
    asyncio.run(main())
