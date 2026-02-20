"""Tests for dtr.analysis.correlation -- binned Pearson correlation."""
from __future__ import annotations

import numpy as np
import pytest

from dtr.analysis.correlation import (
    average_over_seeds,
    compute_binned_correlation,
    compute_correlation_table,
)


# ========================================================================
# Tests: compute_binned_correlation
# ========================================================================


class TestComputeBinnedCorrelation:
    """Tests for the core binned correlation function."""

    def test_perfect_positive_correlation(self) -> None:
        """Metric perfectly predicts accuracy -> r ~ 1."""
        rng = np.random.default_rng(42)
        n = 500
        # Higher metric = higher accuracy
        metric = np.linspace(0, 1, n)
        # Accuracy is a deterministic function of metric
        accuracies = (metric > 0.5).astype(float)

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        assert result["r"] > 0.9
        assert result["p_value"] < 0.1

    def test_perfect_negative_correlation(self) -> None:
        """Higher metric = lower accuracy -> r ~ -1."""
        n = 500
        metric = np.linspace(0, 1, n)
        accuracies = (metric < 0.5).astype(float)

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        assert result["r"] < -0.9

    def test_no_correlation(self) -> None:
        """Random metric should have |r| near 0 on average."""
        rng = np.random.default_rng(123)
        n = 10000
        metric = rng.standard_normal(n)
        accuracies = rng.integers(0, 2, size=n).astype(float)

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        # With random data, |r| should be small (though not guaranteed exactly 0)
        assert abs(result["r"]) < 0.8

    def test_bin_sizes_sum_to_n(self) -> None:
        """All samples should be assigned to exactly one bin."""
        rng = np.random.default_rng(7)
        n = 100
        metric = rng.standard_normal(n)
        accuracies = rng.integers(0, 2, size=n).astype(float)

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        assert result["bin_sizes"].sum() == n

    def test_n_bins_output_shapes(self) -> None:
        """Output arrays should have the correct shapes."""
        rng = np.random.default_rng(99)
        n = 200
        metric = rng.standard_normal(n)
        accuracies = rng.integers(0, 2, size=n).astype(float)

        for n_bins in [3, 5, 10]:
            result = compute_binned_correlation(metric, accuracies, n_bins=n_bins)
            assert len(result["bin_metric_means"]) == n_bins
            assert len(result["bin_accuracy_means"]) == n_bins
            assert len(result["bin_edges"]) == n_bins + 1
            assert len(result["bin_sizes"]) == n_bins

    def test_too_few_samples(self) -> None:
        """Fewer samples than bins should return NaN."""
        metric = np.array([1.0, 2.0])
        accuracies = np.array([0.0, 1.0])

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        assert np.isnan(result["r"])
        assert np.isnan(result["p_value"])

    def test_single_bin(self) -> None:
        """With n_bins=1, correlation is undefined (only one point)."""
        rng = np.random.default_rng(42)
        n = 50
        metric = rng.standard_normal(n)
        accuracies = rng.integers(0, 2, size=n).astype(float)

        result = compute_binned_correlation(metric, accuracies, n_bins=1)
        # Only 1 bin means only 1 data point for correlation -> NaN
        assert np.isnan(result["r"])

    def test_constant_metric(self) -> None:
        """Constant metric values -> zero variance -> NaN correlation."""
        n = 100
        metric = np.ones(n)
        accuracies = np.random.default_rng(0).integers(0, 2, size=n).astype(float)

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        assert np.isnan(result["r"])

    def test_constant_accuracy(self) -> None:
        """Constant accuracy -> zero variance in bin means -> NaN."""
        rng = np.random.default_rng(42)
        n = 100
        metric = rng.standard_normal(n)
        accuracies = np.ones(n)

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        assert np.isnan(result["r"])

    def test_bin_metric_means_monotonic(self) -> None:
        """Bin means of the metric should be approximately monotonic since
        bins are sorted by metric value."""
        rng = np.random.default_rng(42)
        n = 500
        metric = rng.standard_normal(n)
        accuracies = rng.integers(0, 2, size=n).astype(float)

        result = compute_binned_correlation(metric, accuracies, n_bins=5)
        means = result["bin_metric_means"]
        # Each subsequent bin mean should be >= the previous
        for i in range(1, len(means)):
            assert means[i] >= means[i - 1] - 1e-10


# ========================================================================
# Tests: compute_correlation_table
# ========================================================================


class TestComputeCorrelationTable:
    """Tests for the multi-metric correlation table."""

    def test_basic_table(self) -> None:
        """Compute correlation for two metrics; both should appear in output."""
        rng = np.random.default_rng(42)
        n = 200
        samples = []
        for i in range(n):
            correct = rng.random() > 0.5
            samples.append({
                "metric_a": float(i) + rng.standard_normal() * 10,
                "metric_b": rng.standard_normal(),
                "correct": correct,
            })

        result = compute_correlation_table(samples, ["metric_a", "metric_b"], n_bins=5)
        assert "metric_a" in result
        assert "metric_b" in result
        assert "r" in result["metric_a"]
        assert "p_value" in result["metric_a"]

    def test_empty_metric_list(self) -> None:
        """No metrics requested -> empty dict."""
        samples = [{"correct": True}]
        result = compute_correlation_table(samples, [], n_bins=5)
        assert result == {}

    def test_predictive_metric(self) -> None:
        """A metric that perfectly predicts correctness should have high |r|."""
        samples = []
        for i in range(300):
            correct = i >= 150
            samples.append({
                "score": float(i),
                "correct": correct,
            })

        result = compute_correlation_table(samples, ["score"], n_bins=5)
        assert result["score"]["r"] > 0.9


# ========================================================================
# Tests: average_over_seeds
# ========================================================================


class TestAverageOverSeeds:
    """Tests for seed averaging."""

    def test_single_seed(self) -> None:
        """With one seed, mean_r == the single r value and std_r == 0."""
        per_seed = [{"dtr": {"r": 0.85, "p_value": 0.01}}]
        avg = average_over_seeds(per_seed)
        assert avg["dtr"]["mean_r"] == pytest.approx(0.85)
        assert avg["dtr"]["std_r"] == pytest.approx(0.0)
        assert avg["dtr"]["mean_p_value"] == pytest.approx(0.01)

    def test_multiple_seeds(self) -> None:
        """Average over 3 seeds."""
        per_seed = [
            {"dtr": {"r": 0.8, "p_value": 0.01}},
            {"dtr": {"r": 0.9, "p_value": 0.02}},
            {"dtr": {"r": 0.85, "p_value": 0.015}},
        ]
        avg = average_over_seeds(per_seed)
        assert avg["dtr"]["mean_r"] == pytest.approx(0.85)
        assert avg["dtr"]["mean_p_value"] == pytest.approx(0.015)
        assert len(avg["dtr"]["all_r"]) == 3

    def test_nan_seeds_filtered(self) -> None:
        """NaN results should be excluded from the average."""
        per_seed = [
            {"dtr": {"r": 0.8, "p_value": 0.01}},
            {"dtr": {"r": float("nan"), "p_value": float("nan")}},
            {"dtr": {"r": 0.9, "p_value": 0.02}},
        ]
        avg = average_over_seeds(per_seed)
        assert avg["dtr"]["mean_r"] == pytest.approx(0.85)
        assert len(avg["dtr"]["all_r"]) == 2

    def test_all_nan(self) -> None:
        """If all seeds are NaN, result should be NaN."""
        per_seed = [
            {"dtr": {"r": float("nan"), "p_value": float("nan")}},
            {"dtr": {"r": float("nan"), "p_value": float("nan")}},
        ]
        avg = average_over_seeds(per_seed)
        assert np.isnan(avg["dtr"]["mean_r"])

    def test_empty_input(self) -> None:
        """Empty input returns empty dict."""
        assert average_over_seeds([]) == {}

    def test_multiple_metrics(self) -> None:
        """Average over seeds with multiple metrics."""
        per_seed = [
            {"dtr": {"r": 0.9, "p_value": 0.01}, "entropy": {"r": -0.5, "p_value": 0.1}},
            {"dtr": {"r": 0.8, "p_value": 0.02}, "entropy": {"r": -0.6, "p_value": 0.05}},
        ]
        avg = average_over_seeds(per_seed)
        assert "dtr" in avg
        assert "entropy" in avg
        assert avg["dtr"]["mean_r"] == pytest.approx(0.85)
        assert avg["entropy"]["mean_r"] == pytest.approx(-0.55)
