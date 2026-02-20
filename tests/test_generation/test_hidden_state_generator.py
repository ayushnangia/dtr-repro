"""Tests for the generation pipeline (sampling, GenerationResult, PostHocAnalyzer).

All tests run WITHOUT loading real models -- mocks are used throughout.
"""
from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest
import torch

from dtr.generation.hidden_state_generator import GenerationResult, PostHocAnalyzer
from dtr.generation.model_loader import LoadedModel
from dtr.generation.sampling import create_generator, sample_next_token


# =========================================================================
# Sampling tests
# =========================================================================


class TestSampleNextToken:
    """Tests for :func:`sample_next_token`."""

    def test_greedy_returns_argmax(self):
        """Temperature=0 should return the argmax token."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        token = sample_next_token(logits, temperature=0)
        assert token == 1  # index of 3.0

    def test_greedy_with_negative_logits(self):
        """Greedy should work with all-negative logits."""
        logits = torch.tensor([-5.0, -1.0, -3.0, -10.0])
        token = sample_next_token(logits, temperature=0)
        assert token == 1  # index of -1.0 (largest)

    def test_top_p_one_does_not_filter(self):
        """top_p=1.0 should not filter any tokens (all remain candidates)."""
        logits = torch.randn(100)
        gen = create_generator(42)
        # Should not raise and should return a valid token index
        token = sample_next_token(logits, temperature=1.0, top_p=1.0, generator=gen)
        assert 0 <= token < 100

    def test_very_small_top_p_returns_argmax(self):
        """A very small top_p should effectively return the argmax token.

        When top_p is tiny, only the single highest-probability token survives
        the nucleus filter.
        """
        logits = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0])
        # top_p so small only the peak token survives
        token = sample_next_token(logits, temperature=1.0, top_p=1e-9)
        assert token == 1

    def test_seeded_sampling_is_deterministic(self):
        """Two runs with the same seed should produce the same token."""
        logits = torch.randn(1000)

        gen1 = create_generator(12345)
        token1 = sample_next_token(logits, temperature=0.8, top_p=0.9, generator=gen1)

        gen2 = create_generator(12345)
        token2 = sample_next_token(logits, temperature=0.8, top_p=0.9, generator=gen2)

        assert token1 == token2

    def test_different_seeds_can_differ(self):
        """Different seeds should (with high probability) produce different tokens.

        We use a flat distribution so every token is roughly equally likely.
        With 10_000 tokens and two different seeds, the chance of collision
        is ~1/10_000.
        """
        logits = torch.zeros(10_000)  # uniform distribution after softmax

        gen1 = create_generator(1)
        token1 = sample_next_token(logits, temperature=1.0, top_p=1.0, generator=gen1)

        gen2 = create_generator(999)
        token2 = sample_next_token(logits, temperature=1.0, top_p=1.0, generator=gen2)

        # This could *theoretically* fail, but with 10k tokens the chance is negligible
        assert token1 != token2

    def test_rejects_non_1d_logits(self):
        """Should raise ValueError for 2-D logits."""
        logits = torch.randn(2, 10)
        with pytest.raises(ValueError, match="1-D"):
            sample_next_token(logits)

    def test_temperature_affects_distribution(self):
        """Lower temperature should make sampling more deterministic.

        We sample many times with low vs high temperature and check that
        the low-temperature run concentrates more mass on the peak token.
        """
        logits = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0])

        low_temp_count = 0
        high_temp_count = 0
        n_trials = 200

        for i in range(n_trials):
            gen_low = create_generator(i)
            if sample_next_token(logits, temperature=0.1, top_p=1.0, generator=gen_low) == 1:
                low_temp_count += 1

            gen_high = create_generator(i + n_trials)
            if sample_next_token(logits, temperature=2.0, top_p=1.0, generator=gen_high) == 1:
                high_temp_count += 1

        # Low temperature should select the peak far more often
        assert low_temp_count > high_temp_count


class TestCreateGenerator:
    """Tests for :func:`create_generator`."""

    def test_returns_generator(self):
        gen = create_generator(42)
        assert isinstance(gen, torch.Generator)

    def test_same_seed_same_sequence(self):
        gen1 = create_generator(0)
        gen2 = create_generator(0)
        # Draw from uniform -- should match
        t1 = torch.rand(10, generator=gen1)
        t2 = torch.rand(10, generator=gen2)
        assert torch.allclose(t1, t2)


# =========================================================================
# GenerationResult tests
# =========================================================================


class TestGenerationResult:
    """Tests for the :class:`GenerationResult` dataclass."""

    def test_default_construction(self):
        result = GenerationResult()
        assert result.token_ids == []
        assert result.text == ""
        assert result.metrics == {}
        assert result.jsd_matrix is None

    def test_construction_with_values(self):
        jsd = torch.randn(5, 10)
        result = GenerationResult(
            token_ids=[1, 2, 3],
            text="hello",
            metrics={"dtr": 0.42, "settling_depths": [5, 6, 7]},
            jsd_matrix=jsd,
        )
        assert result.token_ids == [1, 2, 3]
        assert result.text == "hello"
        assert result.metrics["dtr"] == 0.42
        assert torch.equal(result.jsd_matrix, jsd)

    def test_has_expected_fields(self):
        field_names = {f.name for f in fields(GenerationResult)}
        assert field_names == {"token_ids", "text", "metrics", "jsd_matrix"}


# =========================================================================
# PostHocAnalyzer tests (with mocked model)
# =========================================================================


def _make_mock_loaded_model(num_layers: int = 4, hidden_dim: int = 32, vocab_size: int = 50):
    """Create a :class:`LoadedModel` backed by mocks.

    The mock model's forward pass returns controlled hidden states so we can
    verify that :class:`PostHocAnalyzer` correctly extracts and processes them.
    """
    config = {
        "hf_id": "mock/model",
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "is_moe": False,
        "gpus_needed": 1,
    }

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0

    # Mock lm_head weight and layer norm
    torch.manual_seed(99)
    lm_head_weight = torch.randn(vocab_size, hidden_dim)

    class _MockNorm:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
            return x / rms

    # Mock model
    model = MagicMock()
    model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
    model.eval = MagicMock(return_value=model)

    def _mock_forward(input_ids, output_hidden_states=False, use_cache=False, **kwargs):
        """Return hidden states that mimic a real transformer.

        For each layer, the hidden state at every position is a deterministic
        function of (layer_idx, position) so tests can verify extraction.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Build hidden states: embedding + num_layers transformer layers
        hs_tuple = []
        for layer_idx in range(num_layers + 1):  # 0 = embedding
            # Simple deterministic pattern: layer_idx * 0.1 + position * 0.01
            hs = torch.zeros(batch, seq_len, hidden_dim, device=device)
            for pos in range(seq_len):
                hs[:, pos, :] = layer_idx * 0.1 + pos * 0.01
            hs_tuple.append(hs)

        output = MagicMock()
        output.hidden_states = tuple(hs_tuple)
        output.logits = torch.randn(batch, seq_len, vocab_size, device=device)
        output.past_key_values = None
        return output

    model.side_effect = _mock_forward
    model.__call__ = _mock_forward

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        lm_head_weight=lm_head_weight,
        final_layer_norm=_MockNorm(),
        config=config,
    )


class TestPostHocAnalyzer:
    """Tests for :class:`PostHocAnalyzer` with a mocked model."""

    def test_analyze_returns_expected_keys(self):
        """The result dict should contain dtr, settling_depths, deep_thinking_mask, jsd_matrix."""
        loaded = _make_mock_loaded_model(num_layers=4, hidden_dim=32)
        analyzer = PostHocAnalyzer(loaded, chunk_size=1024)

        prompt_ids = torch.tensor([1, 2, 3])
        gen_ids = torch.tensor([4, 5, 6, 7])

        with patch("dtr.generation.hidden_state_generator.DTRAccumulator") as MockAccum:
            # Configure the mock accumulator
            mock_instance = MockAccum.return_value
            mock_instance.add_token.return_value = {
                "jsd_vector": torch.zeros(4),
                "settling_depth": 2,
            }
            mock_instance.get_results.return_value = {
                "dtr": 0.25,
                "settling_depths": [2, 3, 2, 1],
                "deep_thinking_mask": [True, False, True, False],
            }

            result = analyzer.analyze(prompt_ids, gen_ids)

        assert "dtr" in result
        assert "settling_depths" in result
        assert "deep_thinking_mask" in result
        assert "jsd_matrix" in result

    def test_analyze_calls_accumulator_per_token(self):
        """DTRAccumulator.add_token should be called once per generated token."""
        loaded = _make_mock_loaded_model(num_layers=4, hidden_dim=32)
        analyzer = PostHocAnalyzer(loaded, chunk_size=1024)

        prompt_ids = torch.tensor([10, 20])
        gen_ids = torch.tensor([30, 40, 50])

        with patch("dtr.generation.hidden_state_generator.DTRAccumulator") as MockAccum:
            mock_instance = MockAccum.return_value
            mock_instance.add_token.return_value = {
                "jsd_vector": torch.zeros(4),
            }
            mock_instance.get_results.return_value = {
                "dtr": 0.5,
                "settling_depths": [1, 2, 3],
                "deep_thinking_mask": [False, True, True],
            }

            analyzer.analyze(prompt_ids, gen_ids)

        assert mock_instance.add_token.call_count == 3  # one per generated token

    def test_analyze_accumulator_receives_correct_shape(self):
        """Each call to add_token should receive a (num_layers, hidden_dim) tensor."""
        num_layers = 4
        hidden_dim = 32
        loaded = _make_mock_loaded_model(num_layers=num_layers, hidden_dim=hidden_dim)
        analyzer = PostHocAnalyzer(loaded, chunk_size=1024)

        prompt_ids = torch.tensor([1])
        gen_ids = torch.tensor([2, 3])

        with patch("dtr.generation.hidden_state_generator.DTRAccumulator") as MockAccum:
            mock_instance = MockAccum.return_value
            mock_instance.add_token.return_value = {
                "jsd_vector": torch.zeros(num_layers),
            }
            mock_instance.get_results.return_value = {
                "dtr": 0.0,
                "settling_depths": [0, 0],
                "deep_thinking_mask": [False, False],
            }

            analyzer.analyze(prompt_ids, gen_ids)

        for call_args in mock_instance.add_token.call_args_list:
            tensor_arg = call_args[0][0]
            assert tensor_arg.shape == (num_layers, hidden_dim)

    def test_analyze_jsd_matrix_shape(self):
        """jsd_matrix should have shape (gen_len, num_layers)."""
        num_layers = 4
        gen_len = 5
        loaded = _make_mock_loaded_model(num_layers=num_layers, hidden_dim=32)
        analyzer = PostHocAnalyzer(loaded, chunk_size=1024)

        prompt_ids = torch.tensor([1, 2])
        gen_ids = torch.arange(10, 10 + gen_len)

        with patch("dtr.generation.hidden_state_generator.DTRAccumulator") as MockAccum:
            mock_instance = MockAccum.return_value
            mock_instance.add_token.return_value = {
                "jsd_vector": torch.ones(num_layers) * 0.1,
            }
            mock_instance.get_results.return_value = {
                "dtr": 0.6,
                "settling_depths": [3] * gen_len,
                "deep_thinking_mask": [True] * gen_len,
            }

            result = analyzer.analyze(prompt_ids, gen_ids)

        assert result["jsd_matrix"] is not None
        assert result["jsd_matrix"].shape == (gen_len, num_layers)

    def test_analyze_handles_2d_input(self):
        """analyze() should accept both 1-D and 2-D (batched) input tensors."""
        loaded = _make_mock_loaded_model(num_layers=4, hidden_dim=32)
        analyzer = PostHocAnalyzer(loaded, chunk_size=1024)

        # 2-D inputs (batch dim = 1)
        prompt_ids = torch.tensor([[1, 2, 3]])
        gen_ids = torch.tensor([[4, 5]])

        with patch("dtr.generation.hidden_state_generator.DTRAccumulator") as MockAccum:
            mock_instance = MockAccum.return_value
            mock_instance.add_token.return_value = {"jsd_vector": torch.zeros(4)}
            mock_instance.get_results.return_value = {
                "dtr": 0.0,
                "settling_depths": [0, 0],
                "deep_thinking_mask": [False, False],
            }

            result = analyzer.analyze(prompt_ids, gen_ids)

        assert "dtr" in result

    def test_analyze_dtr_value_propagated(self):
        """The DTR value from the accumulator should appear in the result."""
        loaded = _make_mock_loaded_model(num_layers=4, hidden_dim=32)
        analyzer = PostHocAnalyzer(loaded, chunk_size=1024)

        prompt_ids = torch.tensor([1])
        gen_ids = torch.tensor([2])

        expected_dtr = 0.73

        with patch("dtr.generation.hidden_state_generator.DTRAccumulator") as MockAccum:
            mock_instance = MockAccum.return_value
            mock_instance.add_token.return_value = {"jsd_vector": torch.zeros(4)}
            mock_instance.get_results.return_value = {
                "dtr": expected_dtr,
                "settling_depths": [3],
                "deep_thinking_mask": [True],
            }

            result = analyzer.analyze(prompt_ids, gen_ids)

        assert result["dtr"] == expected_dtr
