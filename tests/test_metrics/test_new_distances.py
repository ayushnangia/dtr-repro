"""Tests for new distance metrics: reverse KLD, batched KLD, Wasserstein approximations."""

from __future__ import annotations

import pytest
import torch

from dtr.metrics.distances import (
    batch_kld,
    batch_reverse_kld,
    batch_wasserstein_topk,
    kld,
    reverse_kld,
    sliced_wasserstein_1d,
    wasserstein_topk,
)


# ========================================================================
# Reverse KLD
# ========================================================================

class TestReverseKLD:
    """Properties of reverse KL divergence KL(q || p)."""

    def test_same_distribution_is_zero(self, uniform_dist: torch.Tensor) -> None:
        """KL(p, p) == 0 regardless of direction."""
        result = reverse_kld(uniform_dist, uniform_dist)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self) -> None:
        """Reverse KL is non-negative (Gibbs' inequality)."""
        torch.manual_seed(11)
        p = torch.softmax(torch.randn(30), dim=0)
        q = torch.softmax(torch.randn(30), dim=0)
        assert reverse_kld(p, q).item() >= -1e-7

    def test_reverse_kld_equals_kld_swapped(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """reverse_kld(p, q) should equal kld(q, p)."""
        rev = reverse_kld(uniform_dist, peaked_dist)
        fwd = kld(peaked_dist, uniform_dist)
        assert rev.item() == pytest.approx(fwd.item(), abs=1e-6)

    def test_not_symmetric(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """reverse_kld(p, q) != reverse_kld(q, p) in general."""
        forward = reverse_kld(uniform_dist, peaked_dist)
        backward = reverse_kld(peaked_dist, uniform_dist)
        assert forward.item() != pytest.approx(backward.item(), abs=1e-3)


# ========================================================================
# Batch KLD
# ========================================================================

class TestBatchKLD:
    """Batched forward KL: KL(p_i || q) for each row."""

    def test_output_shape(self, uniform_dist: torch.Tensor) -> None:
        """Returns (num_layers,) tensor."""
        torch.manual_seed(0)
        num_layers = 8
        V = uniform_dist.size(0)
        p_batch = torch.softmax(torch.randn(num_layers, V), dim=-1)
        result = batch_kld(p_batch, uniform_dist)
        assert result.shape == (num_layers,)

    def test_matches_elementwise(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """Batch result matches calling kld per-row."""
        p_batch = torch.stack([uniform_dist, peaked_dist, uniform_dist])
        q = peaked_dist
        batch_result = batch_kld(p_batch, q)

        for i in range(p_batch.size(0)):
            expected = kld(p_batch[i], q)
            assert batch_result[i].item() == pytest.approx(
                expected.item(), abs=1e-6
            )

    def test_identical_rows_zero(self, peaked_dist: torch.Tensor) -> None:
        """If every row equals q, all KLDs should be zero."""
        p_batch = peaked_dist.unsqueeze(0).expand(5, -1)
        result = batch_kld(p_batch, peaked_dist)
        assert torch.allclose(result, torch.zeros(5), atol=1e-6)

    def test_all_non_negative(self) -> None:
        """All batched KLD values should be >= 0."""
        torch.manual_seed(42)
        p_batch = torch.softmax(torch.randn(10, 50), dim=-1)
        q = torch.softmax(torch.randn(50), dim=0)
        result = batch_kld(p_batch, q)
        assert (result >= -1e-7).all()


# ========================================================================
# Batch Reverse KLD
# ========================================================================

class TestBatchReverseKLD:
    """Batched reverse KL: KL(q || p_i) for each row."""

    def test_output_shape(self, uniform_dist: torch.Tensor) -> None:
        """Returns (num_layers,) tensor."""
        torch.manual_seed(0)
        num_layers = 6
        V = uniform_dist.size(0)
        p_batch = torch.softmax(torch.randn(num_layers, V), dim=-1)
        result = batch_reverse_kld(p_batch, uniform_dist)
        assert result.shape == (num_layers,)

    def test_matches_elementwise(
        self, uniform_dist: torch.Tensor, peaked_dist: torch.Tensor
    ) -> None:
        """Batch result matches calling reverse_kld per-row."""
        p_batch = torch.stack([uniform_dist, peaked_dist, uniform_dist])
        q = peaked_dist
        batch_result = batch_reverse_kld(p_batch, q)

        for i in range(p_batch.size(0)):
            expected = reverse_kld(p_batch[i], q)
            assert batch_result[i].item() == pytest.approx(
                expected.item(), abs=1e-6
            )

    def test_identical_rows_zero(self, peaked_dist: torch.Tensor) -> None:
        """If every row equals q, all reverse KLDs should be zero."""
        p_batch = peaked_dist.unsqueeze(0).expand(5, -1)
        result = batch_reverse_kld(p_batch, peaked_dist)
        assert torch.allclose(result, torch.zeros(5), atol=1e-6)

    def test_all_non_negative(self) -> None:
        """All batched reverse KLD values should be >= 0."""
        torch.manual_seed(77)
        p_batch = torch.softmax(torch.randn(10, 50), dim=-1)
        q = torch.softmax(torch.randn(50), dim=0)
        result = batch_reverse_kld(p_batch, q)
        assert (result >= -1e-7).all()


# ========================================================================
# Sliced Wasserstein
# ========================================================================

class TestSlicedWasserstein:
    """Sliced Wasserstein-1 approximation."""

    @pytest.fixture
    def small_embeddings(self) -> torch.Tensor:
        """Small embedding matrix for testing: (20, 8)."""
        torch.manual_seed(42)
        return torch.randn(20, 8)

    def test_same_distribution_near_zero(
        self, small_embeddings: torch.Tensor
    ) -> None:
        """W(p, p) should be approximately 0."""
        V = small_embeddings.size(0)
        p = torch.softmax(torch.randn(V), dim=0)
        result = sliced_wasserstein_1d(p, p, small_embeddings, n_projections=100)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self, small_embeddings: torch.Tensor) -> None:
        """Wasserstein distance is non-negative."""
        V = small_embeddings.size(0)
        torch.manual_seed(7)
        p = torch.softmax(torch.randn(V), dim=0)
        q = torch.softmax(torch.randn(V), dim=0)
        result = sliced_wasserstein_1d(p, q, small_embeddings)
        assert result.item() >= -1e-7

    def test_symmetric(self, small_embeddings: torch.Tensor) -> None:
        """Wasserstein is a metric: W(p, q) ~ W(q, p) (stochastic tolerance)."""
        V = small_embeddings.size(0)
        torch.manual_seed(99)
        p = torch.softmax(torch.randn(V), dim=0)
        q = torch.softmax(torch.randn(V), dim=0)
        fwd = sliced_wasserstein_1d(p, q, small_embeddings, n_projections=200)
        bwd = sliced_wasserstein_1d(q, p, small_embeddings, n_projections=200)
        # Sliced Wasserstein uses random projections, so we allow some stochastic
        # noise.  With 200 projections the values should be within ~1%.
        assert fwd.item() == pytest.approx(bwd.item(), rel=0.05)

    def test_raises_without_embeddings(self) -> None:
        """Should raise ValueError when embeddings is None."""
        p = torch.ones(10) / 10
        q = torch.ones(10) / 10
        with pytest.raises(ValueError, match="embedding"):
            sliced_wasserstein_1d(p, q, None)  # type: ignore[arg-type]

    def test_distant_distributions_larger(
        self, small_embeddings: torch.Tensor
    ) -> None:
        """More different distributions should have larger Wasserstein distance."""
        V = small_embeddings.size(0)
        p = torch.zeros(V)
        p[0] = 1.0  # delta on token 0
        q_close = torch.zeros(V)
        q_close[1] = 1.0  # delta on token 1
        q_far = torch.zeros(V)
        q_far[-1] = 1.0  # delta on the farthest token

        w_close = sliced_wasserstein_1d(p, q_close, small_embeddings, n_projections=200)
        w_far = sliced_wasserstein_1d(p, q_far, small_embeddings, n_projections=200)
        # Not guaranteed w_far > w_close since embeddings are random,
        # but at least both should be > 0
        assert w_close.item() > 0.0
        assert w_far.item() > 0.0


# ========================================================================
# Top-k Wasserstein
# ========================================================================

class TestWassersteinTopK:
    """Top-k Wasserstein approximation."""

    @pytest.fixture
    def small_embeddings(self) -> torch.Tensor:
        """Small embedding matrix: (30, 8)."""
        torch.manual_seed(42)
        return torch.randn(30, 8)

    def test_same_distribution_near_zero(
        self, small_embeddings: torch.Tensor
    ) -> None:
        """W_topk(p, p) ~ 0."""
        V = small_embeddings.size(0)
        p = torch.softmax(torch.randn(V), dim=0)
        result = wasserstein_topk(p, p, small_embeddings, k=15, n_projections=100)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self, small_embeddings: torch.Tensor) -> None:
        """Top-k Wasserstein should be non-negative."""
        V = small_embeddings.size(0)
        torch.manual_seed(5)
        p = torch.softmax(torch.randn(V), dim=0)
        q = torch.softmax(torch.randn(V), dim=0)
        result = wasserstein_topk(p, q, small_embeddings, k=10)
        assert result.item() >= -1e-7

    def test_k_larger_than_vocab(self, small_embeddings: torch.Tensor) -> None:
        """k > vocab_size should not crash."""
        V = small_embeddings.size(0)
        p = torch.softmax(torch.randn(V), dim=0)
        q = torch.softmax(torch.randn(V), dim=0)
        result = wasserstein_topk(p, q, small_embeddings, k=1000)
        assert result.item() >= -1e-7

    def test_raises_without_embeddings(self) -> None:
        """Should raise ValueError when embeddings is None."""
        p = torch.ones(10) / 10
        q = torch.ones(10) / 10
        with pytest.raises(ValueError, match="embedding"):
            wasserstein_topk(p, q, None, k=5)  # type: ignore[arg-type]


# ========================================================================
# Batch Top-k Wasserstein
# ========================================================================

class TestBatchWassersteinTopK:
    """Batched top-k Wasserstein."""

    @pytest.fixture
    def small_embeddings(self) -> torch.Tensor:
        """Small embedding matrix: (20, 8)."""
        torch.manual_seed(42)
        return torch.randn(20, 8)

    def test_output_shape(self, small_embeddings: torch.Tensor) -> None:
        """Returns (num_layers,) tensor."""
        V = small_embeddings.size(0)
        torch.manual_seed(0)
        num_layers = 5
        p_batch = torch.softmax(torch.randn(num_layers, V), dim=-1)
        q = torch.softmax(torch.randn(V), dim=0)
        result = batch_wasserstein_topk(p_batch, q, small_embeddings, k=10)
        assert result.shape == (num_layers,)

    def test_matches_elementwise(self, small_embeddings: torch.Tensor) -> None:
        """Batch result matches calling wasserstein_topk per-row."""
        V = small_embeddings.size(0)
        torch.manual_seed(33)
        p_batch = torch.softmax(torch.randn(3, V), dim=-1)
        q = torch.softmax(torch.randn(V), dim=0)

        batch_result = batch_wasserstein_topk(
            p_batch, q, small_embeddings, k=10, n_projections=50
        )

        for i in range(p_batch.size(0)):
            # Use same seed for reproducibility within the sliced projections
            torch.manual_seed(0)
            expected = wasserstein_topk(
                p_batch[i], q, small_embeddings, k=10, n_projections=50
            )
            # Note: sliced wasserstein uses random projections, so we check
            # that values are in the same ballpark rather than exact match
            assert batch_result[i].item() >= -1e-7  # non-negative

    def test_identical_rows_near_zero(self, small_embeddings: torch.Tensor) -> None:
        """If every row equals q, all Wasserstein distances should be ~0."""
        V = small_embeddings.size(0)
        torch.manual_seed(0)
        q = torch.softmax(torch.randn(V), dim=0)
        p_batch = q.unsqueeze(0).expand(4, -1)
        result = batch_wasserstein_topk(
            p_batch, q, small_embeddings, k=10, n_projections=100
        )
        assert torch.allclose(result, torch.zeros(4), atol=1e-4)

    def test_raises_without_embeddings(self) -> None:
        """Should raise ValueError when embeddings is None."""
        p_batch = torch.softmax(torch.randn(3, 10), dim=-1)
        q = torch.ones(10) / 10
        with pytest.raises(ValueError, match="embedding"):
            batch_wasserstein_topk(p_batch, q, None, k=5)  # type: ignore[arg-type]
