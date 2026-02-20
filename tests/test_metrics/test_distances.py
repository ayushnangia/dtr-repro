"""Tests for dtr.metrics.distances -- JSD, KLD, cosine_distance, batch_jsd."""

from __future__ import annotations

import math

import pytest
import torch

from dtr.metrics.distances import batch_jsd, cosine_distance, jsd, kld


# ========================================================================
# JSD
# ========================================================================

class TestJSD:
    """Properties of Jensen-Shannon Divergence."""

    def test_same_distribution_is_zero(self, uniform_dist: torch.Tensor) -> None:
        """JSD(p, p) == 0."""
        result = jsd(uniform_dist, uniform_dist)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """JSD(p, q) == JSD(q, p)."""
        forward = jsd(uniform_dist, peaked_dist)
        backward = jsd(peaked_dist, uniform_dist)
        assert forward.item() == pytest.approx(backward.item(), abs=1e-6)

    def test_bounded_base_2(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """With base=2, JSD is in [0, 1]."""
        result = jsd(uniform_dist, peaked_dist, base=2.0)
        assert 0.0 <= result.item() <= 1.0 + 1e-6

    def test_bounded_base_e(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """With base=e, JSD is in [0, ln(2)]."""
        result = jsd(uniform_dist, peaked_dist, base=math.e)
        assert 0.0 <= result.item() <= math.log(2) + 1e-6

    def test_uniform_vs_peaked_positive(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """JSD of distinct distributions should be > 0."""
        result = jsd(uniform_dist, peaked_dist)
        assert result.item() > 0.0

    def test_non_negative(self) -> None:
        """JSD is always non-negative even for random distributions."""
        torch.manual_seed(99)
        p = torch.softmax(torch.randn(50), dim=0)
        q = torch.softmax(torch.randn(50), dim=0)
        assert jsd(p, q).item() >= -1e-7


# ========================================================================
# KLD
# ========================================================================

class TestKLD:
    """Properties of KL Divergence."""

    def test_same_distribution_is_zero(self, uniform_dist: torch.Tensor) -> None:
        """KL(p, p) == 0."""
        result = kld(uniform_dist, uniform_dist)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self) -> None:
        """KL divergence is non-negative (Gibbs' inequality)."""
        torch.manual_seed(7)
        p = torch.softmax(torch.randn(30), dim=0)
        q = torch.softmax(torch.randn(30), dim=0)
        assert kld(p, q).item() >= -1e-7

    def test_not_symmetric(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """KL(p, q) != KL(q, p) in general."""
        forward = kld(uniform_dist, peaked_dist)
        backward = kld(peaked_dist, uniform_dist)
        assert forward.item() != pytest.approx(backward.item(), abs=1e-3)


# ========================================================================
# Cosine distance
# ========================================================================

class TestCosineDistance:
    """Properties of cosine distance."""

    def test_same_vector_is_zero(self) -> None:
        """cos_dist(x, x) == 0."""
        x = torch.randn(128)
        result = cosine_distance(x, x)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_bounded(self) -> None:
        """Cosine distance in [0, 2]."""
        torch.manual_seed(1)
        x = torch.randn(64)
        y = torch.randn(64)
        result = cosine_distance(x, y)
        assert -1e-6 <= result.item() <= 2.0 + 1e-6

    def test_opposite_vectors(self) -> None:
        """Opposite vectors have cosine distance = 2."""
        x = torch.ones(10)
        y = -torch.ones(10)
        result = cosine_distance(x, y)
        assert result.item() == pytest.approx(2.0, abs=1e-6)


# ========================================================================
# batch_jsd
# ========================================================================

class TestBatchJSD:
    """Batch JSD should produce the correct shape and match element-wise JSD."""

    def test_output_shape(self, uniform_dist: torch.Tensor) -> None:
        """Returns (num_layers,) tensor."""
        torch.manual_seed(0)
        num_layers = 8
        V = uniform_dist.size(0)
        p_batch = torch.softmax(torch.randn(num_layers, V), dim=-1)
        result = batch_jsd(p_batch, uniform_dist)
        assert result.shape == (num_layers,)

    def test_matches_elementwise(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """Batch result matches calling jsd per-row."""
        p_batch = torch.stack([uniform_dist, peaked_dist, uniform_dist])
        q = peaked_dist
        batch_result = batch_jsd(p_batch, q)

        for i in range(p_batch.size(0)):
            expected = jsd(p_batch[i], q)
            assert batch_result[i].item() == pytest.approx(
                expected.item(), abs=1e-6
            )

    def test_identical_rows_zero(self, peaked_dist: torch.Tensor) -> None:
        """If every row equals q, all JSDs should be zero."""
        p_batch = peaked_dist.unsqueeze(0).expand(5, -1)
        result = batch_jsd(p_batch, peaked_dist)
        assert torch.allclose(result, torch.zeros(5), atol=1e-6)
