"""Tests for new DTR variants: soft DTR, continuous DTR, generic distance-per-layer."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dtr.metrics.dtr import (
    DTRAccumulator,
    _sigmoid,
    compute_distance_per_layer,
    compute_dtr,
    compute_dtr_continuous,
    compute_dtr_soft,
    compute_jsd_per_layer,
    recompute_dtr_continuous,
    recompute_dtr_soft,
)


# ========================================================================
# Reusable JSD matrix from test_dtr.py
# ========================================================================

def _make_jsd_matrix() -> torch.Tensor:
    """Same hand-crafted (5 tokens, 10 layers) JSD matrix as test_dtr.py.

    Token 0: settles at layer 2  -> NOT deep (2 < 9)
    Token 1: settles at layer 5  -> NOT deep (5 < 9)
    Token 2: settles at layer 9  -> DEEP     (9 >= 9)
    Token 3: never settles (10)  -> DEEP     (10 >= 9)
    Token 4: settles at layer 8  -> NOT deep (8 < 9)

    threshold_layer = ceil(0.85 * 10) = 9
    """
    rows = [
        torch.tensor([0.9, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0]),
        torch.tensor([0.95, 0.85, 0.7, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.0]),
        torch.tensor([0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.45, 0.0]),
        torch.tensor([0.99, 0.95, 0.9, 0.88, 0.85, 0.82, 0.8, 0.78, 0.75, 0.6]),
        torch.tensor([0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.45, 0.3, 0.0]),
    ]
    return torch.stack(rows)


# ========================================================================
# _sigmoid helper
# ========================================================================

class TestSigmoid:
    """Tests for the numerically stable sigmoid."""

    def test_zero_is_half(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_near_one(self) -> None:
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative_near_zero(self) -> None:
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self) -> None:
        """sigmoid(x) + sigmoid(-x) == 1."""
        for x in [0.5, 1.0, 3.0, 10.0]:
            assert _sigmoid(x) + _sigmoid(-x) == pytest.approx(1.0, abs=1e-10)

    def test_monotonically_increasing(self) -> None:
        vals = [_sigmoid(x) for x in [-5, -1, 0, 1, 5]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]


# ========================================================================
# compute_dtr_soft (Rao-Blackwellized DTR)
# ========================================================================

class TestComputeDTRSoft:
    """Rao-Blackwellized (soft) DTR."""

    def test_includes_hard_dtr_keys(self) -> None:
        """Soft DTR result should contain all keys from hard DTR plus extras."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_soft(jsd_matrix)
        # Hard DTR keys
        assert "dtr" in result
        assert "settling_depths" in result
        assert "deep_thinking_mask" in result
        assert "num_deep" in result
        assert "total_tokens" in result
        # New soft keys
        assert "dtr_soft" in result
        assert "soft_deep_values" in result

    def test_hard_dtr_unchanged(self) -> None:
        """The hard DTR values should be identical to compute_dtr."""
        jsd_matrix = _make_jsd_matrix()
        hard = compute_dtr(jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85)
        soft = compute_dtr_soft(jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85)

        assert soft["dtr"] == hard["dtr"]
        assert soft["settling_depths"] == hard["settling_depths"]
        assert soft["deep_thinking_mask"] == hard["deep_thinking_mask"]

    def test_soft_dtr_in_unit_interval(self) -> None:
        """Soft DTR should be in [0, 1]."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_soft(jsd_matrix)
        assert 0.0 <= result["dtr_soft"] <= 1.0

    def test_soft_values_in_unit_interval(self) -> None:
        """Each soft_deep value should be in [0, 1]."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_soft(jsd_matrix)
        for val in result["soft_deep_values"]:
            assert 0.0 <= val <= 1.0

    def test_deep_tokens_have_high_soft_values(self) -> None:
        """Tokens classified as deep (settling_depth >= 9) should have
        soft_deep values > 0.5 (since sigmoid center is at rho)."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_soft(jsd_matrix, sharpness=20.0)
        for i, is_deep in enumerate(result["deep_thinking_mask"]):
            if is_deep:
                assert result["soft_deep_values"][i] > 0.5

    def test_shallow_tokens_have_low_soft_values(self) -> None:
        """Tokens that settle early (depth << rho*L) should have low soft_deep."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_soft(jsd_matrix, sharpness=20.0)
        # Token 0 settles at layer 2 -> normalized = 0.2, far below rho=0.85
        assert result["soft_deep_values"][0] < 0.1

    def test_high_sharpness_approaches_hard(self) -> None:
        """With very high sharpness, soft DTR should approach hard DTR."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_soft(jsd_matrix, sharpness=1000.0)
        assert result["dtr_soft"] == pytest.approx(result["dtr"], abs=0.01)

    def test_low_sharpness_is_smoother(self) -> None:
        """With low sharpness, soft DTR should differ from hard DTR."""
        jsd_matrix = _make_jsd_matrix()
        high = compute_dtr_soft(jsd_matrix, sharpness=100.0)
        low = compute_dtr_soft(jsd_matrix, sharpness=1.0)
        # Low sharpness should differ more from the hard value
        diff_high = abs(high["dtr_soft"] - high["dtr"])
        diff_low = abs(low["dtr_soft"] - low["dtr"])
        assert diff_low > diff_high

    def test_empty_matrix(self) -> None:
        """Empty JSD matrix should return 0.0 for soft DTR."""
        jsd_matrix = torch.zeros(0, 10)
        result = compute_dtr_soft(jsd_matrix)
        assert result["dtr_soft"] == 0.0


# ========================================================================
# compute_dtr_continuous
# ========================================================================

class TestComputeDTRContinuous:
    """Continuous DTR (mean normalized settling depth)."""

    def test_result_keys(self) -> None:
        """Should have the expected keys."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_continuous(jsd_matrix)
        assert "dtr_continuous" in result
        assert "settling_depths" in result
        assert "normalized_depths" in result

    def test_settling_depths_match_hard_dtr(self) -> None:
        """Settling depths should match compute_dtr."""
        jsd_matrix = _make_jsd_matrix()
        hard = compute_dtr(jsd_matrix, threshold_g=0.5)
        cont = compute_dtr_continuous(jsd_matrix, threshold_g=0.5)
        assert cont["settling_depths"] == hard["settling_depths"]

    def test_normalized_depths_correct(self) -> None:
        """Normalized depths = settling_depth / num_layers."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_continuous(jsd_matrix)
        num_layers = jsd_matrix.size(1)
        for sd, nd in zip(result["settling_depths"], result["normalized_depths"]):
            assert nd == pytest.approx(sd / num_layers)

    def test_continuous_dtr_value(self) -> None:
        """DTR_continuous = mean of normalized depths."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_continuous(jsd_matrix)
        # Settling depths: [2, 5, 9, 10, 8], num_layers = 10
        # Normalized: [0.2, 0.5, 0.9, 1.0, 0.8]
        # Mean: (0.2 + 0.5 + 0.9 + 1.0 + 0.8) / 5 = 3.4 / 5 = 0.68
        assert result["dtr_continuous"] == pytest.approx(0.68)

    def test_in_unit_interval(self) -> None:
        """DTR_continuous should be in [0, 1]."""
        jsd_matrix = _make_jsd_matrix()
        result = compute_dtr_continuous(jsd_matrix)
        assert 0.0 <= result["dtr_continuous"] <= 1.0

    def test_empty_matrix(self) -> None:
        """Empty JSD matrix -> 0.0."""
        result = compute_dtr_continuous(torch.zeros(0, 10))
        assert result["dtr_continuous"] == 0.0


# ========================================================================
# compute_distance_per_layer (generic multi-method)
# ========================================================================

class TestComputeDistancePerLayer:
    """Generic per-layer distance computation."""

    def test_jsd_matches_original(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """method='jsd' should produce the same result as compute_jsd_per_layer."""
        original = compute_jsd_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm
        )
        generic = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="jsd"
        )
        assert torch.allclose(original, generic, atol=1e-6)

    def test_kld_output_shape(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """KLD method returns correct shape."""
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="kld"
        )
        assert result.shape == (mock_hidden_states.size(0),)

    def test_kld_final_layer_zero(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """KL(p_final || p_final) should be ~0."""
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="kld"
        )
        assert result[-1].item() == pytest.approx(0.0, abs=1e-5)

    def test_reverse_kld_output_shape(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Reverse KLD method returns correct shape."""
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="reverse_kld"
        )
        assert result.shape == (mock_hidden_states.size(0),)

    def test_reverse_kld_final_layer_zero(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """KL(p_final || p_final) should be ~0 for reverse KL too."""
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="reverse_kld"
        )
        assert result[-1].item() == pytest.approx(0.0, abs=1e-5)

    def test_cosine_output_shape(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Cosine method returns correct shape."""
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="cosine"
        )
        assert result.shape == (mock_hidden_states.size(0),)

    def test_cosine_final_layer_zero(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Cosine distance of final layer with itself should be ~0."""
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="cosine"
        )
        assert result[-1].item() == pytest.approx(0.0, abs=1e-5)

    def test_cosine_early_layers_higher(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Earlier layers should have higher cosine distance from final."""
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm, method="cosine"
        )
        first_half = result[: len(result) // 2].mean()
        second_half = result[len(result) // 2 :].mean()
        assert first_half > second_half

    def test_wasserstein_output_shape(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Wasserstein method returns correct shape."""
        torch.manual_seed(42)
        # Embeddings for the 100-token vocabulary used in conftest
        embeddings = torch.randn(100, 64)
        result = compute_distance_per_layer(
            mock_hidden_states, mock_lm_head, mock_layer_norm,
            method="wasserstein", embeddings=embeddings, wasserstein_k=10,
        )
        assert result.shape == (mock_hidden_states.size(0),)

    def test_wasserstein_raises_without_embeddings(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Wasserstein without embeddings should raise ValueError."""
        with pytest.raises(ValueError, match="embeddings"):
            compute_distance_per_layer(
                mock_hidden_states, mock_lm_head, mock_layer_norm,
                method="wasserstein",
            )

    def test_unknown_method_raises(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown distance method"):
            compute_distance_per_layer(
                mock_hidden_states, mock_lm_head, mock_layer_norm,
                method="magic",
            )

    def test_all_methods_non_negative(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """All distance methods should produce non-negative values."""
        for method in ["jsd", "kld", "reverse_kld", "cosine"]:
            result = compute_distance_per_layer(
                mock_hidden_states, mock_lm_head, mock_layer_norm,
                method=method,
            )
            assert (result >= -1e-6).all(), f"method={method} produced negative values"


# ========================================================================
# Numpy recomputation helpers
# ========================================================================

class TestRecomputeDTRSoft:
    """Numpy-based soft DTR recomputation."""

    def test_matches_torch_version(self) -> None:
        """Numpy version should produce the same result as torch version."""
        jsd_matrix = _make_jsd_matrix()
        torch_result = compute_dtr_soft(
            jsd_matrix, threshold_g=0.5, depth_ratio_rho=0.85, sharpness=20.0
        )
        numpy_result = recompute_dtr_soft(
            jsd_matrix.numpy(), threshold_g=0.5, depth_ratio_rho=0.85, sharpness=20.0
        )
        assert numpy_result == pytest.approx(torch_result["dtr_soft"], abs=1e-6)

    def test_empty_matrix(self) -> None:
        """Empty matrix returns 0.0."""
        assert recompute_dtr_soft(np.zeros((0, 10)), 0.5, 0.85) == 0.0

    def test_in_unit_interval(self) -> None:
        """Should be in [0, 1]."""
        jsd_matrix = _make_jsd_matrix().numpy()
        result = recompute_dtr_soft(jsd_matrix, 0.5, 0.85)
        assert 0.0 <= result <= 1.0


class TestRecomputeDTRContinuous:
    """Numpy-based continuous DTR recomputation."""

    def test_matches_torch_version(self) -> None:
        """Numpy version should produce the same result as torch version."""
        jsd_matrix = _make_jsd_matrix()
        torch_result = compute_dtr_continuous(jsd_matrix, threshold_g=0.5)
        numpy_result = recompute_dtr_continuous(jsd_matrix.numpy(), threshold_g=0.5)
        assert numpy_result == pytest.approx(
            torch_result["dtr_continuous"], abs=1e-6
        )

    def test_known_value(self) -> None:
        """Known settling depths [2, 5, 9, 10, 8] -> mean normalized = 0.68."""
        jsd_matrix = _make_jsd_matrix().numpy()
        result = recompute_dtr_continuous(jsd_matrix, threshold_g=0.5)
        assert result == pytest.approx(0.68)

    def test_empty_matrix(self) -> None:
        """Empty matrix returns 0.0."""
        assert recompute_dtr_continuous(np.zeros((0, 10)), 0.5) == 0.0


# ========================================================================
# DTRAccumulator extensions
# ========================================================================

class TestDTRAccumulatorExtensions:
    """Extended DTRAccumulator with method and soft DTR support."""

    def test_default_backward_compatible(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """Default parameters should produce the same results as before."""
        acc = DTRAccumulator(
            num_layers=mock_hidden_states.size(0),
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
        )
        info = acc.add_token(mock_hidden_states)
        # Should still have jsd_values for backward compat
        assert "jsd_values" in info
        assert "distance_values" in info
        assert "settling_depth" in info
        assert "is_deep" in info

    def test_soft_dtr_included_when_requested(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """compute_soft=True should include dtr_soft in results."""
        acc = DTRAccumulator(
            num_layers=mock_hidden_states.size(0),
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
            compute_soft=True,
        )
        acc.add_token(mock_hidden_states)
        result = acc.get_results()
        assert "dtr_soft" in result
        assert "soft_deep_values" in result
        assert 0.0 <= result["dtr_soft"] <= 1.0

    def test_soft_dtr_not_included_by_default(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """compute_soft=False (default) should not include dtr_soft."""
        acc = DTRAccumulator(
            num_layers=mock_hidden_states.size(0),
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
        )
        acc.add_token(mock_hidden_states)
        result = acc.get_results()
        assert "dtr_soft" not in result

    def test_cosine_method(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """method='cosine' should work through the accumulator."""
        acc = DTRAccumulator(
            num_layers=mock_hidden_states.size(0),
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
            method="cosine",
        )
        info = acc.add_token(mock_hidden_states)
        assert "distance_values" in info
        # cosine method should NOT include jsd_values key
        assert "jsd_values" not in info

    def test_kld_method(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """method='kld' should work."""
        acc = DTRAccumulator(
            num_layers=mock_hidden_states.size(0),
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
            method="kld",
        )
        info = acc.add_token(mock_hidden_states)
        result = acc.get_results()
        assert result["total_tokens"] == 1
        assert "dtr" in result

    def test_reverse_kld_method(
        self,
        mock_hidden_states: torch.Tensor,
        mock_lm_head: torch.Tensor,
        mock_layer_norm,
    ) -> None:
        """method='reverse_kld' should work."""
        acc = DTRAccumulator(
            num_layers=mock_hidden_states.size(0),
            lm_head_weight=mock_lm_head,
            layer_norm=mock_layer_norm,
            method="reverse_kld",
        )
        info = acc.add_token(mock_hidden_states)
        result = acc.get_results()
        assert result["total_tokens"] == 1
