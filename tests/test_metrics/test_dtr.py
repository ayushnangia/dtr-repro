"""Tests for dtr.metrics.dtr -- settling depth, deep-thinking detection, DTR, and DTRAccumulator."""

from __future__ import annotations

import math

import pytest
import torch

from dtr.metrics.dtr import (
    DTRAccumulator,
    compute_dtr,
    compute_jsd_per_layer,
    compute_settling_depth,
    is_deep_thinking_token,
)


# ========================================================================
# Hand-crafted JSD matrix for deterministic tests
# ========================================================================

def _make_jsd_matrix() -> torch.Tensor:
    """Create a (5 tokens, 10 layers) JSD matrix with known settling behaviour.

    Token 0: settles at layer 2  (functional word -- JSD drops immediately)
    Token 1: settles at layer 5  (moderate)
    Token 2: settles at layer 9  (deep thinking with rho=0.85, threshold=9)
    Token 3: never settles       (deep thinking -- all JSD > 0.5)
    Token 4: settles at layer 8  (NOT deep thinking: ceil(0.85*10)=9)

    Settling threshold g = 0.5.
    """
    # Start with high JSD and decrease towards the settled layer.
    rows = []

    # Token 0 -- settles at layer index 1 (layer 2, 1-indexed)
    # cummin drops to <= 0.5 at index 1
    rows.append(torch.tensor([0.9, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0]))

    # Token 1 -- settles at layer index 4 (layer 5, 1-indexed)
    rows.append(torch.tensor([0.95, 0.85, 0.7, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.0]))

    # Token 2 -- settles at layer index 8 (layer 9, 1-indexed)
    rows.append(torch.tensor([0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.45, 0.0]))

    # Token 3 -- never settles (all > 0.5 up through last; note last = 0.0 from
    # the definition but for "never settles" we need cummin > g everywhere).
    # Actually JSD with final layer at the last layer is always 0. So to get
    # "never settles" we need the cummin to stay > 0.5 up to and including
    # the last layer.  We'll use a pattern that stays > 0.5 and then the
    # last layer is 0.0 -- but cummin will hit 0 at last layer.
    # Instead: keep ALL values > 0.5 including the last.
    rows.append(torch.tensor([0.99, 0.95, 0.9, 0.88, 0.85, 0.82, 0.8, 0.78, 0.75, 0.6]))

    # Token 4 -- settles at layer index 7 (layer 8, 1-indexed)
    rows.append(torch.tensor([0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.45, 0.3, 0.0]))

    return torch.stack(rows)  # (5, 10)


# ========================================================================
# Tests: compute_settling_depth
# ========================================================================

class TestSettlingDepth:
    """Tests for ``compute_settling_depth``."""

    def test_monotonically_decreasing(self) -> None:
        """Monotonically decreasing JSD settles at first value <= g."""
        jsd_vals = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        # cummin: 0.9, 0.7, 0.5, 0.3, 0.1 -- first <= 0.5 at index 2 => layer 3
        assert compute_settling_depth(jsd_vals, threshold_g=0.5) == 3

    def test_all_above_threshold(self) -> None:
        """If JSD never drops below g, return num_layers."""
        jsd_vals = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.55])
        assert compute_settling_depth(jsd_vals, threshold_g=0.5) == 5

    def test_non_monotonic_cummin(self) -> None:
        """Non-monotonic JSD: cummin still tracks the running minimum."""
        # JSD dips below 0.5 at index 2, then bounces up, but cummin stays low
        jsd_vals = torch.tensor([0.9, 0.8, 0.4, 0.7, 0.3])
        # cummin: 0.9, 0.8, 0.4, 0.4, 0.3 -> first <= 0.5 at index 2 => layer 3
        assert compute_settling_depth(jsd_vals, threshold_g=0.5) == 3

    def test_immediate_settle(self) -> None:
        """First layer already below threshold."""
        jsd_vals = torch.tensor([0.2, 0.1, 0.05])
        assert compute_settling_depth(jsd_vals, threshold_g=0.5) == 1

    def test_settles_at_exact_threshold(self) -> None:
        """Value exactly equal to g counts as settled."""
        jsd_vals = torch.tensor([0.9, 0.5, 0.3])
        assert compute_settling_depth(jsd_vals, threshold_g=0.5) == 2


# ========================================================================
# Tests: is_deep_thinking_token
# ========================================================================

class TestIsDeepThinkingToken:
    """Tests for ``is_deep_thinking_token``."""

    def test_deep_at_boundary(self) -> None:
        """ceil(0.85 * 10) = 9.  Settling depth 9 IS deep thinking."""
        assert is_deep_thinking_token(settling_depth=9, num_layers=10, depth_ratio_rho=0.85)

    def test_not_deep_below_boundary(self) -> None:
        """Settling depth 8 is NOT deep thinking (8 < 9)."""
        assert not is_deep_thinking_token(settling_depth=8, num_layers=10, depth_ratio_rho=0.85)

    def test_deep_when_never_settles(self) -> None:
        """Settling depth == num_layers means it never settled -- deep thinking."""
        assert is_deep_thinking_token(settling_depth=10, num_layers=10, depth_ratio_rho=0.85)

    def test_not_deep_early_settle(self) -> None:
        """Early settling is not deep thinking."""
        assert not is_deep_thinking_token(settling_depth=2, num_layers=10, depth_ratio_rho=0.85)


# ========================================================================
# Tests: compute_dtr (batch)
# ========================================================================

class TestComputeDTR:
    """End-to-end DTR tests using the hand-crafted JSD matrix."""

    def test_dtr_value(self) -> None:
        """With g=0.5, rho=0.85, only tokens 2 and 3 are deep -> DTR = 2/5."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr(jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85)

        assert result["total_tokens"] == 5
        assert result["num_deep"] == 2
        assert result["dtr"] == pytest.approx(0.4)

    def test_settling_depths(self) -> None:
        """Verify per-token settling depths."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr(jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85)

        # Token 0: layer 2, Token 1: layer 5, Token 2: layer 9,
        # Token 3: 10 (never), Token 4: layer 8
        assert result["settling_depths"] == [2, 5, 9, 10, 8]

    def test_deep_thinking_mask(self) -> None:
        """Tokens 2 and 3 are deep thinking."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr(jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85)

        assert result["deep_thinking_mask"] == [False, False, True, True, False]


# ========================================================================
# Tests: compute_jsd_per_layer
# ========================================================================

class TestComputeJSDPerLayer:
    """Tests for the logit-lens JSD computation."""

    def test_final_layer_jsd_zero(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """JSD of the final layer vs. itself should be ~0."""
        jsd_vals = compute_jsd_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm
        )
        assert jsd_vals[-1].item() == pytest.approx(0.0, abs=1e-5)

    def test_output_shape(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Output shape matches num_layers."""
        jsd_vals = compute_jsd_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm
        )
        assert jsd_vals.shape == (mock_hidden_states.size(0),)

    def test_all_non_negative(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """All JSD values should be >= 0."""
        jsd_vals = compute_jsd_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm
        )
        assert (jsd_vals >= -1e-7).all()

    def test_early_layers_higher(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """On average, earlier layers should diverge more from the final layer."""
        jsd_vals = compute_jsd_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm
        )
        first_half_mean = jsd_vals[: len(jsd_vals) // 2].mean()
        second_half_mean = jsd_vals[len(jsd_vals) // 2 :].mean()
        assert first_half_mean > second_half_mean


# ========================================================================
# Tests: DTRAccumulator
# ========================================================================

class TestDTRAccumulator:
    """Streaming accumulator should match batch compute_dtr."""

    def test_accumulator_matches_batch(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Feed the same token data through the accumulator and through
        compute_dtr; results must agree."""
        num_tokens = 3
        num_layers = mock_hidden_states.size(0)

        # Build varied token states by perturbing the mock hidden states.
        torch.manual_seed(7)
        all_hidden = []
        for _ in range(num_tokens):
            perturbed = mock_hidden_states + 0.3 * torch.randn_like(mock_hidden_states)
            all_hidden.append(perturbed)

        # -- batch path --
        jsd_rows = []
        for hs in all_hidden:
            jsd_row = compute_jsd_per_layer(hs, mock_lm_head, mock_layer_norm)
            jsd_rows.append(jsd_row)
        jsd_matrix = torch.stack(jsd_rows)
        batch_result = compute_dtr(jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85)

        # -- streaming path --
        acc = DTRAccumulator(
            num_layers=num_layers,
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
            threshold_g=0.5,
            depth_ratio_rho=0.85,
        )
        for hs in all_hidden:
            acc.add_token(hs)
        stream_result = acc.get_results()

        # Compare.
        assert stream_result["dtr"] == pytest.approx(batch_result["dtr"], abs=1e-6)
        assert stream_result["settling_depths"] == batch_result["settling_depths"]
        assert stream_result["deep_thinking_mask"] == batch_result["deep_thinking_mask"]
        assert stream_result["num_deep"] == batch_result["num_deep"]
        assert stream_result["total_tokens"] == batch_result["total_tokens"]

    def test_accumulator_add_token_returns_info(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """add_token should return a dict with per-token info."""
        acc = DTRAccumulator(
            num_layers=mock_hidden_states.size(0),
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
        )
        info = acc.add_token(mock_hidden_states)
        assert "jsd_values" in info
        assert "settling_depth" in info
        assert "is_deep" in info
        assert isinstance(info["settling_depth"], int)
        assert isinstance(info["is_deep"], bool)
