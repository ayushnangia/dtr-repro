"""Tests for dtr.metrics.baselines -- all six baseline metrics."""

from __future__ import annotations

import math

import pytest
import torch

from dtr.metrics.baselines import (
    compute_all_baselines,
    mean_log_probability,
    negative_entropy,
    negative_perplexity,
    reverse_token_count,
    self_certainty,
    token_count,
)


# ========================================================================
# Token count metrics
# ========================================================================

class TestTokenCount:
    """Tests for token_count and reverse_token_count."""

    def test_token_count(self) -> None:
        ids = torch.tensor([10, 20, 30, 40, 50])
        assert token_count(ids) == 5

    def test_reverse_token_count(self) -> None:
        ids = torch.tensor([10, 20, 30, 40, 50])
        assert reverse_token_count(ids) == -5

    def test_single_token(self) -> None:
        ids = torch.tensor([42])
        assert token_count(ids) == 1
        assert reverse_token_count(ids) == -1


# ========================================================================
# Uniform distribution tests
# ========================================================================

class TestUniformDistribution:
    """With uniform logits / distributions, verify expected metric values."""

    @pytest.fixture()
    def uniform_logits_and_ids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform logits over V=100 vocab, T=20 tokens."""
        V = 100
        T = 20
        # All-zero logits -> softmax = uniform = 1/V
        logits = torch.zeros(T, V)
        # Random token ids
        torch.manual_seed(0)
        token_ids = torch.randint(0, V, (T,))
        return logits, token_ids

    def test_entropy_equals_log_v(
        self, uniform_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """For a uniform distribution, H(p) = log(V)."""
        logits, _ = uniform_logits_and_ids
        V = logits.size(-1)
        # negative_entropy = -mean(H) and H = log(V) for uniform
        result = negative_entropy(logits)
        assert result == pytest.approx(-math.log(V), abs=1e-4)

    def test_self_certainty_zero(
        self, uniform_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """KL(uniform || uniform) = 0, so self_certainty should be ~0."""
        logits, _ = uniform_logits_and_ids
        result = self_certainty(logits)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_negative_entropy_sign(
        self, uniform_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """negative_entropy should be negative (since entropy > 0)."""
        logits, _ = uniform_logits_and_ids
        assert negative_entropy(logits) < 0.0

    def test_negative_perplexity_sign(
        self, uniform_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """negative_perplexity should be negative."""
        logits, token_ids = uniform_logits_and_ids
        assert negative_perplexity(logits, token_ids) < 0.0


# ========================================================================
# Peaked distribution tests
# ========================================================================

class TestPeakedDistribution:
    """With a peaked distribution (most mass on a single token)."""

    @pytest.fixture()
    def peaked_logits_and_ids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits that give ~99% probability to token 0, T=20 tokens."""
        V = 100
        T = 20
        logits = torch.full((T, V), -10.0)
        logits[:, 0] = 10.0  # token 0 dominates
        token_ids = torch.zeros(T, dtype=torch.long)  # always predicting token 0
        return logits, token_ids

    def test_low_entropy(
        self, peaked_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """A peaked distribution should have low entropy (near 0)."""
        logits, _ = peaked_logits_and_ids
        # negative_entropy = -H, and H is small => negative_entropy is close to 0 (but negative)
        result = negative_entropy(logits)
        assert result > -0.5  # entropy is very small

    def test_high_self_certainty(
        self, peaked_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Peaked distribution should have high self_certainty."""
        logits, _ = peaked_logits_and_ids
        result = self_certainty(logits)
        assert result > 1.0  # significantly above zero

    def test_mean_log_prob_near_zero(
        self, peaked_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """When always predicting the correct token with high confidence,
        mean_log_prob should be close to 0 (log(1)=0)."""
        logits, token_ids = peaked_logits_and_ids
        result = mean_log_probability(logits, token_ids)
        assert result > -0.1  # very close to 0

    def test_negative_perplexity_near_minus_one(
        self, peaked_logits_and_ids: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perplexity close to 1 => negative_perplexity close to -1."""
        logits, token_ids = peaked_logits_and_ids
        result = negative_perplexity(logits, token_ids)
        assert result > -1.1
        assert result < -0.9


# ========================================================================
# compute_all_baselines
# ========================================================================

class TestComputeAllBaselines:
    """Integration test for ``compute_all_baselines``."""

    def test_returns_all_keys(self) -> None:
        V, T = 50, 10
        logits = torch.randn(T, V)
        token_ids = torch.randint(0, V, (T,))
        result = compute_all_baselines(logits, token_ids)

        expected_keys = {
            "token_count",
            "reverse_token_count",
            "mean_log_probability",
            "negative_perplexity",
            "negative_entropy",
            "self_certainty",
        }
        assert set(result.keys()) == expected_keys

    def test_token_count_values(self) -> None:
        V, T = 50, 10
        logits = torch.randn(T, V)
        token_ids = torch.randint(0, V, (T,))
        result = compute_all_baselines(logits, token_ids)
        assert result["token_count"] == float(T)
        assert result["reverse_token_count"] == float(-T)
