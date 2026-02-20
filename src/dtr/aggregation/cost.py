"""Inference cost computation for aggregation strategies."""
from __future__ import annotations


def compute_cons_cost(token_counts: list[int]) -> int:
    """Cons@n, Mean@n, Long@n, Short@n cost: sum of ALL token counts."""
    return sum(token_counts)


def compute_selective_cost(
    selected_token_counts: list[int],
    n_unselected: int,
    prefix_len: int,
) -> int:
    """Think@n / Self-Certainty@n cost:
    full generation for selected + prefix-only for unselected."""
    return sum(selected_token_counts) + prefix_len * n_unselected


def compute_cost_ratio(strategy_cost: int, cons_cost: int) -> float:
    """Cost ratio relative to Cons@n baseline."""
    if cons_cost == 0:
        return 0.0
    return strategy_cost / cons_cost


def summarize_costs(strategy_results: list[dict]) -> dict:
    """Summarize costs across multiple questions/trials.
    Returns mean cost, total cost, and cost per question."""
    costs = [r["cost"] for r in strategy_results]
    return {
        "mean_cost": sum(costs) / len(costs) if costs else 0.0,
        "total_cost": sum(costs),
        "n_questions": len(costs),
    }
