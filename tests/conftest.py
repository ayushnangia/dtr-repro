"""Shared test fixtures for DTR tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def rng():
    """Seeded numpy random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def uniform_dist():
    """Uniform distribution over 100 tokens."""
    V = 100
    return torch.ones(V) / V


@pytest.fixture
def peaked_dist():
    """Peaked distribution (90% on first token)."""
    V = 100
    p = torch.ones(V) * (0.1 / (V - 1))
    p[0] = 0.9
    return p


@pytest.fixture
def mock_hidden_states():
    """Mock hidden states: (num_layers, hidden_dim).

    Simulates a 10-layer model with hidden_dim=64.
    Earlier layers have random states, later layers converge to final.
    """
    torch.manual_seed(42)
    num_layers = 10
    hidden_dim = 64
    final_state = torch.randn(hidden_dim)

    states = []
    for layer_idx in range(num_layers):
        # Interpolate: early layers are random, later layers approach final
        alpha = layer_idx / (num_layers - 1)
        random_state = torch.randn(hidden_dim)
        state = (1 - alpha) * random_state + alpha * final_state
        states.append(state)

    return torch.stack(states)  # (10, 64)


@pytest.fixture
def mock_lm_head():
    """Mock lm_head weight matrix: (vocab_size, hidden_dim)."""
    torch.manual_seed(123)
    vocab_size = 100
    hidden_dim = 64
    return torch.randn(vocab_size, hidden_dim)


class MockLayerNorm:
    """Simple mock for RMSNorm that just normalizes."""

    def __init__(self, hidden_dim: int = 64):
        self.weight = torch.ones(hidden_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2) + 1e-6)
        return x / rms * self.weight


@pytest.fixture
def mock_layer_norm():
    """Mock final layer norm."""
    return MockLayerNorm(hidden_dim=64)
