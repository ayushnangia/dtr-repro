"""Response aggregation strategies (Cons@n, Think@n, etc.)."""
from __future__ import annotations

from dtr.aggregation.cost import (
    compute_cons_cost,
    compute_cost_ratio,
    compute_selective_cost,
    summarize_costs,
)
from dtr.aggregation.strategies import (
    SampleResult,
    cons_at_n,
    long_at_n,
    majority_vote,
    mean_at_n,
    run_all_strategies,
    run_trials,
    self_certainty_at_n,
    short_at_n,
    think_at_n,
)

__all__ = [
    "SampleResult",
    "compute_cons_cost",
    "compute_cost_ratio",
    "compute_selective_cost",
    "cons_at_n",
    "long_at_n",
    "majority_vote",
    "mean_at_n",
    "run_all_strategies",
    "run_trials",
    "self_certainty_at_n",
    "short_at_n",
    "summarize_costs",
    "think_at_n",
]
