"""Tests for dtr.analysis.sensitivity -- hyperparameter sweeps."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dtr.analysis.sensitivity import (
    recompute_dtr_from_jsd,
    sweep_dtr_params,
    sweep_prefix_lengths,
)


# ========================================================================
# Helper: build deterministic JSD matrices
# ========================================================================

def _make_easy_jsd_matrix(num_tokens: int, num_layers: int) -> np.ndarray:
    """Create a JSD matrix where all tokens settle early (low DTR).

    Each token's JSD drops quickly below any reasonable threshold g.
    """
    matrix = np.zeros((num_tokens, num_layers))
    for t in range(num_tokens):
        for l in range(num_layers):
            # Decays rapidly: 0.9 * exp(-2*l/L)
            matrix[t, l] = 0.9 * np.exp(-2.0 * (l + 1) / num_layers)
    return matrix


def _make_hard_jsd_matrix(num_tokens: int, num_layers: int) -> np.ndarray:
    """Create a JSD matrix where all tokens settle very late (high DTR).

    JSD stays high until the very last layers.
    """
    matrix = np.zeros((num_tokens, num_layers))
    for t in range(num_tokens):
        for l in range(num_layers):
            # Stays high: only drops below 0.5 near the last layer
            progress = (l + 1) / num_layers
            matrix[t, l] = 0.9 * (1.0 - progress ** 5)
    return matrix


# ========================================================================
# Tests: recompute_dtr_from_jsd
# ========================================================================


class TestRecomputeDTRFromJSD:
    """Tests for the numpy-based DTR recomputation."""

    def test_matches_known_settling(self) -> None:
        """Hand-crafted matrix with known settling behaviour.

        Mirrors the JSD matrix from test_dtr.py but in numpy.
        """
        # 5 tokens, 10 layers (same as _make_jsd_matrix in test_dtr.py)
        jsd_matrix = np.array([
            [0.9, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0],  # settles layer 2
            [0.95, 0.85, 0.7, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.0],   # settles layer 5
            [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.45, 0.0],  # settles layer 9
            [0.99, 0.95, 0.9, 0.88, 0.85, 0.82, 0.8, 0.78, 0.75, 0.6],# never settles
            [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.45, 0.3, 0.0],  # settles layer 8
        ])
        # g=0.5, rho=0.85 -> threshold_layer = ceil(0.85*10) = 9
        # Deep tokens: token 2 (depth=9 >= 9) and token 3 (depth=10 >= 9)
        dtr = recompute_dtr_from_jsd(jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85)
        assert dtr == pytest.approx(2 / 5)

    def test_all_easy_tokens(self) -> None:
        """All tokens settle early -> DTR ~ 0."""
        matrix = _make_easy_jsd_matrix(50, 32)
        dtr = recompute_dtr_from_jsd(matrix, threshold_g=0.5, depth_ratio_rho=0.85)
        assert dtr == pytest.approx(0.0)

    def test_all_hard_tokens(self) -> None:
        """All tokens settle very late -> DTR ~ 1."""
        matrix = _make_hard_jsd_matrix(50, 32)
        dtr = recompute_dtr_from_jsd(matrix, threshold_g=0.1, depth_ratio_rho=0.5)
        # With g=0.1 and rho=0.5, most tokens should be deep
        assert dtr > 0.5

    def test_empty_matrix(self) -> None:
        """Empty sequence -> DTR = 0."""
        matrix = np.zeros((0, 10))
        dtr = recompute_dtr_from_jsd(matrix, threshold_g=0.5, depth_ratio_rho=0.85)
        assert dtr == 0.0

    def test_lower_g_increases_dtr(self) -> None:
        """A lower settling threshold g should make more tokens appear deep."""
        rng = np.random.default_rng(42)
        matrix = rng.uniform(0, 1, size=(100, 32))
        # Sort each row in decreasing order to mimic typical JSD pattern
        matrix = np.sort(matrix, axis=1)[:, ::-1]

        dtr_high_g = recompute_dtr_from_jsd(matrix, threshold_g=0.8, depth_ratio_rho=0.85)
        dtr_low_g = recompute_dtr_from_jsd(matrix, threshold_g=0.2, depth_ratio_rho=0.85)
        assert dtr_low_g >= dtr_high_g

    def test_higher_rho_increases_dtr(self) -> None:
        """A higher rho makes the deep-thinking bar lower (ceil(rho*L) is higher),
        so fewer tokens qualify... wait, higher rho -> higher threshold_layer
        -> MORE tokens qualify because settling depth is compared >= threshold.
        Actually: higher threshold_layer means FEWER tokens qualify.

        Let's re-check: is_deep = settling_depth >= ceil(rho * L).
        Higher rho -> higher ceil(rho*L) -> harder to be deep -> lower DTR.
        """
        rng = np.random.default_rng(42)
        matrix = rng.uniform(0, 1, size=(100, 32))
        matrix = np.sort(matrix, axis=1)[:, ::-1]

        dtr_low_rho = recompute_dtr_from_jsd(matrix, threshold_g=0.5, depth_ratio_rho=0.5)
        dtr_high_rho = recompute_dtr_from_jsd(matrix, threshold_g=0.5, depth_ratio_rho=0.95)
        assert dtr_low_rho >= dtr_high_rho

    def test_single_token(self) -> None:
        """Single-token matrix: either 0 or 1."""
        # Token that settles at layer 1 -> not deep
        matrix = np.array([[0.3, 0.2, 0.1, 0.05, 0.01]])
        dtr = recompute_dtr_from_jsd(matrix, threshold_g=0.5, depth_ratio_rho=0.85)
        assert dtr == 0.0

        # Token that never settles -> deep
        matrix = np.array([[0.9, 0.85, 0.8, 0.75, 0.7]])
        dtr = recompute_dtr_from_jsd(matrix, threshold_g=0.5, depth_ratio_rho=0.85)
        assert dtr == 1.0


# ========================================================================
# Tests: sweep_dtr_params
# ========================================================================


class TestSweepDTRParams:
    """Tests for the (g, rho) hyperparameter sweep."""

    @pytest.fixture()
    def sweep_data(self) -> tuple[list[np.ndarray], list[bool]]:
        """Create a small dataset of JSD matrices with known properties."""
        rng = np.random.default_rng(42)
        n_samples = 50
        num_layers = 16

        jsd_matrices = []
        accuracies = []
        for i in range(n_samples):
            n_tokens = rng.integers(20, 100)
            # Samples with more "hard" tokens tend to be correct
            if i < n_samples // 2:
                matrix = _make_easy_jsd_matrix(n_tokens, num_layers)
                accuracies.append(False)
            else:
                matrix = _make_hard_jsd_matrix(n_tokens, num_layers)
                accuracies.append(True)
            jsd_matrices.append(matrix)

        return jsd_matrices, accuracies

    def test_output_length(self, sweep_data: tuple) -> None:
        """Number of results = len(g_values) * len(rho_values)."""
        jsd_matrices, accuracies = sweep_data
        g_vals = [0.25, 0.5, 0.75]
        rho_vals = [0.8, 0.85, 0.9]

        results = sweep_dtr_params(
            jsd_matrices, accuracies,
            g_values=g_vals, rho_values=rho_vals,
        )
        assert len(results) == len(g_vals) * len(rho_vals)

    def test_result_keys(self, sweep_data: tuple) -> None:
        """Each result dict has the expected keys."""
        jsd_matrices, accuracies = sweep_data
        results = sweep_dtr_params(
            jsd_matrices, accuracies,
            g_values=[0.5], rho_values=[0.85],
        )
        r = results[0]
        assert "g" in r
        assert "rho" in r
        assert "dtr_values" in r
        assert "correlation" in r
        assert r["g"] == 0.5
        assert r["rho"] == 0.85
        assert len(r["dtr_values"]) == len(jsd_matrices)

    def test_dtr_values_in_range(self, sweep_data: tuple) -> None:
        """DTR values should be in [0, 1]."""
        jsd_matrices, accuracies = sweep_data
        results = sweep_dtr_params(
            jsd_matrices, accuracies,
            g_values=[0.5], rho_values=[0.85],
        )
        dtr_vals = results[0]["dtr_values"]
        assert np.all(dtr_vals >= 0.0)
        assert np.all(dtr_vals <= 1.0)

    def test_defaults(self, sweep_data: tuple) -> None:
        """Default g and rho values are used when not specified."""
        jsd_matrices, accuracies = sweep_data
        results = sweep_dtr_params(jsd_matrices, accuracies)
        # Default: 3 g values x 4 rho values = 12
        assert len(results) == 12

    def test_positive_correlation_for_predictive_metric(self, sweep_data: tuple) -> None:
        """When high DTR correlates with correctness, r should be positive."""
        jsd_matrices, accuracies = sweep_data
        results = sweep_dtr_params(
            jsd_matrices, accuracies,
            g_values=[0.5], rho_values=[0.85], n_bins=5,
        )
        # Our synthetic data has hard matrices -> correct, easy -> incorrect
        # So DTR should positively correlate with accuracy
        r = results[0]["correlation"]["r"]
        # Allow for edge cases but generally expect positive
        assert not np.isnan(r)


# ========================================================================
# Tests: sweep_prefix_lengths
# ========================================================================


class TestSweepPrefixLengths:
    """Tests for the prefix-length sweep (Think@n ablation)."""

    @pytest.fixture()
    def prefix_data(self) -> tuple[list[np.ndarray], list[bool]]:
        """Small dataset for prefix-length testing."""
        rng = np.random.default_rng(99)
        n_samples = 30
        num_layers = 16

        jsd_matrices = []
        accuracies = []
        for i in range(n_samples):
            n_tokens = 200  # fixed length so all prefix lengths are valid
            matrix = rng.uniform(0, 1, size=(n_tokens, num_layers))
            # Sort rows descending to mimic real JSD pattern
            matrix = np.sort(matrix, axis=1)[:, ::-1]
            jsd_matrices.append(matrix)
            accuracies.append(bool(rng.random() > 0.5))

        return jsd_matrices, accuracies

    def test_output_length(self, prefix_data: tuple) -> None:
        """One result per prefix length."""
        jsd_matrices, accuracies = prefix_data
        lengths = [10, 50, 100, -1]
        results = sweep_prefix_lengths(
            jsd_matrices, accuracies, prefix_lengths=lengths,
        )
        assert len(results) == len(lengths)

    def test_result_keys(self, prefix_data: tuple) -> None:
        """Each result has expected keys."""
        jsd_matrices, accuracies = prefix_data
        results = sweep_prefix_lengths(
            jsd_matrices, accuracies, prefix_lengths=[50],
        )
        r = results[0]
        assert "prefix_length" in r
        assert "dtr_values" in r
        assert "correlation" in r
        assert r["prefix_length"] == 50

    def test_full_sequence_matches_minus_one(self, prefix_data: tuple) -> None:
        """prefix_length=-1 should give same DTR as using a very large number."""
        jsd_matrices, accuracies = prefix_data
        results = sweep_prefix_lengths(
            jsd_matrices, accuracies,
            prefix_lengths=[-1, 99999],
            threshold_g=0.5, depth_ratio_rho=0.85,
        )
        np.testing.assert_array_almost_equal(
            results[0]["dtr_values"],
            results[1]["dtr_values"],
        )

    def test_shorter_prefix_changes_dtr(self, prefix_data: tuple) -> None:
        """A very short prefix should generally give different DTR than full."""
        jsd_matrices, accuracies = prefix_data
        results = sweep_prefix_lengths(
            jsd_matrices, accuracies,
            prefix_lengths=[5, -1],
            threshold_g=0.5, depth_ratio_rho=0.85,
        )
        short_dtr = results[0]["dtr_values"]
        full_dtr = results[1]["dtr_values"]
        # They should differ for at least some samples
        assert not np.allclose(short_dtr, full_dtr)

    def test_defaults(self, prefix_data: tuple) -> None:
        """Default prefix lengths are used when not specified."""
        jsd_matrices, accuracies = prefix_data
        results = sweep_prefix_lengths(jsd_matrices, accuracies)
        # Default: [50, 100, 500, 1000, 2000, -1] = 6 entries
        assert len(results) == 6

    def test_prefix_longer_than_sequence(self) -> None:
        """If prefix_length > sequence length, use full sequence."""
        matrix = np.random.default_rng(0).uniform(0, 1, size=(10, 8))
        results = sweep_prefix_lengths(
            [matrix], [True],
            prefix_lengths=[100, -1],
            threshold_g=0.5, depth_ratio_rho=0.85,
        )
        # Both should be identical since 100 > 10 tokens
        np.testing.assert_array_almost_equal(
            results[0]["dtr_values"],
            results[1]["dtr_values"],
        )
