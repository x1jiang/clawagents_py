"""Per-role context budget allocation (Headroom/OpenClaw-inspired)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContentBudgets:
    """Fractions of the effective context window reserved per role family."""

    system: float = 0.15
    tools: float = 0.45
    user_assistant: float = 0.35
    images: float = 0.05

    def validate(self) -> None:
        total = self.system + self.tools + self.user_assistant + self.images
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"ContentBudgets fractions should sum ~1.0, got {total}")


DEFAULT_CONTENT_BUDGETS = ContentBudgets()


def chars_budget_for_tools(max_input_tokens: int, budgets: ContentBudgets | None = None) -> int:
    """Approximate char budget for tool messages (4 chars ≈ 1 token heuristic)."""
    b = budgets or DEFAULT_CONTENT_BUDGETS
    return max(int(max_input_tokens * b.tools * 4), 4000)
