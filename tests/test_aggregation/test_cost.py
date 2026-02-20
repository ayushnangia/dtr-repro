"""Tests for inference cost computation."""
from __future__ import annotations

import pytest

from dtr.aggregation.cost import (
    compute_cons_cost,
    compute_cost_ratio,
    compute_selective_cost,
    summarize_costs,
)


# ---------------------------------------------------------------------------
# compute_cons_cost
# ---------------------------------------------------------------------------


class TestComputeConsCost:
    def test_sums_all_tokens(self) -> None:
        assert compute_cons_cost([100, 200, 300]) == 600

    def test_single_sample(self) -> None:
        assert compute_cons_cost([500]) == 500

    def test_empty(self) -> None:
        assert compute_cons_cost([]) == 0

    def test_large_values(self) -> None:
        counts = [10_000, 20_000, 30_000]
        assert compute_cons_cost(counts) == 60_000


# ---------------------------------------------------------------------------
# compute_selective_cost
# ---------------------------------------------------------------------------


class TestComputeSelectiveCost:
    def test_formula(self) -> None:
        # selected tokens + prefix * n_unselected
        result = compute_selective_cost([100, 200], n_unselected=3, prefix_len=50)
        assert result == 300 + 150  # 450

    def test_no_unselected(self) -> None:
        """All samples selected: cost = sum of selected only."""
        result = compute_selective_cost([100, 200, 300], n_unselected=0, prefix_len=50)
        assert result == 600

    def test_all_unselected(self) -> None:
        """No samples selected: cost = prefix * n_unselected."""
        result = compute_selective_cost([], n_unselected=5, prefix_len=50)
        assert result == 250

    def test_prefix_len_zero(self) -> None:
        """With prefix_len=0, unselected contribute nothing."""
        result = compute_selective_cost([100], n_unselected=10, prefix_len=0)
        assert result == 100

    def test_single_selected_single_unselected(self) -> None:
        result = compute_selective_cost([500], n_unselected=1, prefix_len=50)
        assert result == 550


# ---------------------------------------------------------------------------
# compute_cost_ratio
# ---------------------------------------------------------------------------


class TestComputeCostRatio:
    def test_same_cost(self) -> None:
        assert compute_cost_ratio(1000, 1000) == pytest.approx(1.0)

    def test_half_cost(self) -> None:
        assert compute_cost_ratio(500, 1000) == pytest.approx(0.5)

    def test_zero_cons_cost(self) -> None:
        """Avoid division by zero."""
        assert compute_cost_ratio(100, 0) == 0.0

    def test_higher_cost(self) -> None:
        """Strategy can theoretically cost more than cons."""
        assert compute_cost_ratio(1500, 1000) == pytest.approx(1.5)

    def test_zero_strategy_cost(self) -> None:
        assert compute_cost_ratio(0, 1000) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# summarize_costs
# ---------------------------------------------------------------------------


class TestSummarizeCosts:
    def test_basic(self) -> None:
        results = [{"cost": 100}, {"cost": 200}, {"cost": 300}]
        summary = summarize_costs(results)
        assert summary["mean_cost"] == pytest.approx(200.0)
        assert summary["total_cost"] == 600
        assert summary["n_questions"] == 3

    def test_single_result(self) -> None:
        summary = summarize_costs([{"cost": 500}])
        assert summary["mean_cost"] == pytest.approx(500.0)
        assert summary["total_cost"] == 500
        assert summary["n_questions"] == 1

    def test_empty(self) -> None:
        summary = summarize_costs([])
        assert summary["mean_cost"] == 0.0
        assert summary["total_cost"] == 0
        assert summary["n_questions"] == 0

    def test_uniform_costs(self) -> None:
        results = [{"cost": 100}] * 5
        summary = summarize_costs(results)
        assert summary["mean_cost"] == pytest.approx(100.0)
        assert summary["total_cost"] == 500
