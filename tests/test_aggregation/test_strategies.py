"""Comprehensive tests for response aggregation strategies."""
from __future__ import annotations

import pytest

from dtr.aggregation.strategies import (
    SampleResult,
    cons_at_n,
    long_at_n,
    majority_vote,
    mean_at_n,
    run_all_strategies,
    run_trials,
    self_certainty_at_n,
    short_at_n,
    think_at_n,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def samples_10() -> list[SampleResult]:
    """10 SampleResults: 5 correct (high DTR, varied lengths), 5 incorrect (low DTR).

    Correct samples have answer="A", high DTR (0.7-0.9), high self-certainty (0.8-0.95),
    and varied token counts (100-500).

    Incorrect samples have answer="B", low DTR (0.1-0.3), low self-certainty (0.1-0.4),
    and varied token counts (80-450).
    """
    correct = [
        SampleResult(
            answer="A", correct=True, token_count=500, dtr=0.9, self_certainty=0.95, log_prob=-0.1
        ),
        SampleResult(
            answer="A", correct=True, token_count=400, dtr=0.85, self_certainty=0.9, log_prob=-0.2
        ),
        SampleResult(
            answer="A", correct=True, token_count=300, dtr=0.8, self_certainty=0.85, log_prob=-0.3
        ),
        SampleResult(
            answer="A", correct=True, token_count=200, dtr=0.75, self_certainty=0.82, log_prob=-0.4
        ),
        SampleResult(
            answer="A", correct=True, token_count=100, dtr=0.7, self_certainty=0.8, log_prob=-0.5
        ),
    ]
    incorrect = [
        SampleResult(
            answer="B", correct=False, token_count=450, dtr=0.3, self_certainty=0.4, log_prob=-1.0
        ),
        SampleResult(
            answer="B", correct=False, token_count=350, dtr=0.25, self_certainty=0.3, log_prob=-1.1
        ),
        SampleResult(
            answer="B", correct=False, token_count=250, dtr=0.2, self_certainty=0.25, log_prob=-1.2
        ),
        SampleResult(
            answer="B", correct=False, token_count=150, dtr=0.15, self_certainty=0.15, log_prob=-1.3
        ),
        SampleResult(
            answer="B", correct=False, token_count=80, dtr=0.1, self_certainty=0.1, log_prob=-1.5
        ),
    ]
    return correct + incorrect


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_basic(self) -> None:
        assert majority_vote(["A", "A", "B"]) == "A"

    def test_all_same(self) -> None:
        assert majority_vote(["X", "X", "X"]) == "X"

    def test_none_filtered(self) -> None:
        assert majority_vote([None, "A", None, "A", "B"]) == "A"

    def test_all_none(self) -> None:
        assert majority_vote([None, None]) is None

    def test_empty(self) -> None:
        assert majority_vote([]) is None

    def test_single(self) -> None:
        assert majority_vote(["Z"]) == "Z"


# ---------------------------------------------------------------------------
# cons@n
# ---------------------------------------------------------------------------


class TestConsAtN:
    def test_majority_wins(self, samples_10: list[SampleResult]) -> None:
        result = cons_at_n(samples_10)
        assert result["strategy"] == "cons@n"
        # 5 "A" vs 5 "B" -- Counter.most_common picks one; both are valid
        assert result["predicted"] in ("A", "B")

    def test_correct_majority(self) -> None:
        """When correct answers dominate, accuracy should be True."""
        samples = [
            SampleResult("A", True, 100, 0.5, 0.5, -0.5),
            SampleResult("A", True, 100, 0.5, 0.5, -0.5),
            SampleResult("B", False, 100, 0.5, 0.5, -0.5),
        ]
        result = cons_at_n(samples)
        assert result["predicted"] == "A"
        assert result["correct"] is True

    def test_cost_sums_all(self, samples_10: list[SampleResult]) -> None:
        result = cons_at_n(samples_10)
        expected_cost = sum(s.token_count for s in samples_10)
        assert result["cost"] == expected_cost


# ---------------------------------------------------------------------------
# mean@n
# ---------------------------------------------------------------------------


class TestMeanAtN:
    def test_average_accuracy(self, samples_10: list[SampleResult]) -> None:
        result = mean_at_n(samples_10)
        assert result["strategy"] == "mean@n"
        assert result["correct"] == pytest.approx(0.5)

    def test_all_correct(self) -> None:
        samples = [SampleResult("A", True, 100, 0.5, 0.5, -0.5) for _ in range(4)]
        result = mean_at_n(samples)
        assert result["correct"] == pytest.approx(1.0)

    def test_none_correct(self) -> None:
        samples = [SampleResult("B", False, 100, 0.5, 0.5, -0.5) for _ in range(3)]
        result = mean_at_n(samples)
        assert result["correct"] == pytest.approx(0.0)

    def test_cost_sums_all(self, samples_10: list[SampleResult]) -> None:
        result = mean_at_n(samples_10)
        expected_cost = sum(s.token_count for s in samples_10)
        assert result["cost"] == expected_cost

    def test_predicted_is_none(self, samples_10: list[SampleResult]) -> None:
        """mean@n does not aggregate, so predicted is always None."""
        result = mean_at_n(samples_10)
        assert result["predicted"] is None


# ---------------------------------------------------------------------------
# long@n
# ---------------------------------------------------------------------------


class TestLongAtN:
    def test_selects_longest(self, samples_10: list[SampleResult]) -> None:
        result = long_at_n(samples_10, eta=0.5)
        assert result["strategy"] == "long@n"
        # Top 5 by length: 500(A), 450(B), 400(A), 350(B), 300(A) => 3A,2B => A wins
        assert result["predicted"] == "A"
        assert result["correct"] is True

    def test_cost_includes_all(self, samples_10: list[SampleResult]) -> None:
        """Long@n must generate all samples to determine which are longest."""
        result = long_at_n(samples_10, eta=0.5)
        expected_cost = sum(s.token_count for s in samples_10)
        assert result["cost"] == expected_cost

    def test_eta_one(self, samples_10: list[SampleResult]) -> None:
        """With eta=1.0, all samples are selected (same as cons@n vote pool)."""
        result = long_at_n(samples_10, eta=1.0)
        cons_result = cons_at_n(samples_10)
        assert result["predicted"] == cons_result["predicted"]


# ---------------------------------------------------------------------------
# short@n
# ---------------------------------------------------------------------------


class TestShortAtN:
    def test_selects_shortest(self, samples_10: list[SampleResult]) -> None:
        result = short_at_n(samples_10, eta=0.5)
        assert result["strategy"] == "short@n"
        # Shortest 5: 80(B), 100(A), 150(B), 200(A), 250(B) => 2A,3B => B wins
        assert result["predicted"] == "B"
        assert result["correct"] is False

    def test_cost_includes_all(self, samples_10: list[SampleResult]) -> None:
        result = short_at_n(samples_10, eta=0.5)
        expected_cost = sum(s.token_count for s in samples_10)
        assert result["cost"] == expected_cost


# ---------------------------------------------------------------------------
# think@n
# ---------------------------------------------------------------------------


class TestThinkAtN:
    def test_selects_highest_dtr(self, samples_10: list[SampleResult]) -> None:
        result = think_at_n(samples_10, eta=0.5)
        assert result["strategy"] == "think@n"
        # Top 5 by DTR: 0.9(A), 0.85(A), 0.8(A), 0.75(A), 0.7(A) => all correct
        assert result["predicted"] == "A"
        assert result["correct"] is True

    def test_higher_accuracy_than_random(self, samples_10: list[SampleResult]) -> None:
        """think@n with high-DTR selection should achieve higher accuracy than mean@n."""
        think_result = think_at_n(samples_10, eta=0.5)
        mean_result = mean_at_n(samples_10)
        # think@n selects all correct => True (1.0), mean@n => 0.5
        assert think_result["correct"] is True
        assert mean_result["correct"] < 1.0

    def test_cost_saves_on_unselected(self, samples_10: list[SampleResult]) -> None:
        """think@n only pays full cost for selected; prefix for rest."""
        prefix_len = 50
        result = think_at_n(samples_10, eta=0.5, prefix_len=prefix_len)
        # Top 5 by DTR are the 5 correct ones: tokens 500+400+300+200+100 = 1500
        # Unselected 5: prefix_len * 5 = 250
        expected_cost = 1500 + 250
        assert result["cost"] == expected_cost

    def test_cost_less_than_cons(self, samples_10: list[SampleResult]) -> None:
        """think@n should cost less than cons@n (which pays full for all)."""
        think_result = think_at_n(samples_10, eta=0.5, prefix_len=50)
        cons_result = cons_at_n(samples_10)
        assert think_result["cost"] < cons_result["cost"]


# ---------------------------------------------------------------------------
# self_certainty@n
# ---------------------------------------------------------------------------


class TestSelfCertaintyAtN:
    def test_selects_highest_certainty(self, samples_10: list[SampleResult]) -> None:
        result = self_certainty_at_n(samples_10, eta=0.5)
        assert result["strategy"] == "self_certainty@n"
        # Top 5 by self_certainty: 0.95, 0.9, 0.85, 0.82, 0.8 => all "A" => correct
        assert result["predicted"] == "A"
        assert result["correct"] is True

    def test_cost_saves_on_unselected(self, samples_10: list[SampleResult]) -> None:
        prefix_len = 50
        result = self_certainty_at_n(samples_10, eta=0.5, prefix_len=prefix_len)
        # Selected top-5 by certainty = correct samples: 500+400+300+200+100 = 1500
        # Unselected 5: 50 * 5 = 250
        expected_cost = 1500 + 250
        assert result["cost"] == expected_cost

    def test_cost_less_than_cons(self, samples_10: list[SampleResult]) -> None:
        sc_result = self_certainty_at_n(samples_10, eta=0.5, prefix_len=50)
        cons_result = cons_at_n(samples_10)
        assert sc_result["cost"] < cons_result["cost"]


# ---------------------------------------------------------------------------
# run_all_strategies
# ---------------------------------------------------------------------------


class TestRunAllStrategies:
    def test_returns_six_results(self, samples_10: list[SampleResult]) -> None:
        results = run_all_strategies(samples_10)
        assert len(results) == 6

    def test_strategy_names(self, samples_10: list[SampleResult]) -> None:
        results = run_all_strategies(samples_10)
        names = {r["strategy"] for r in results}
        expected = {"cons@n", "mean@n", "long@n", "short@n", "self_certainty@n", "think@n"}
        assert names == expected

    def test_all_have_required_keys(self, samples_10: list[SampleResult]) -> None:
        results = run_all_strategies(samples_10)
        for r in results:
            assert "strategy" in r
            assert "predicted" in r
            assert "correct" in r
            assert "cost" in r


# ---------------------------------------------------------------------------
# run_trials
# ---------------------------------------------------------------------------


class TestRunTrials:
    def test_correct_structure(self, samples_10: list[SampleResult]) -> None:
        results = run_trials(samples_10, n=5, n_trials=3)
        assert len(results) == 6  # 6 strategies
        for name, trial_list in results.items():
            assert len(trial_list) == 3  # 3 trials each

    def test_deterministic_with_seed(self, samples_10: list[SampleResult]) -> None:
        import numpy as np

        r1 = run_trials(samples_10, n=5, n_trials=3, rng=np.random.default_rng(123))
        r2 = run_trials(samples_10, n=5, n_trials=3, rng=np.random.default_rng(123))
        for name in r1:
            for t1, t2 in zip(r1[name], r2[name]):
                assert t1["cost"] == t2["cost"]
                assert t1["predicted"] == t2["predicted"]

    def test_n_larger_than_pool(self, samples_10: list[SampleResult]) -> None:
        """When n > len(all_samples), should use all samples."""
        results = run_trials(samples_10, n=100, n_trials=1)
        for name, trial_list in results.items():
            assert len(trial_list) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_samples(self) -> None:
        """All strategies should handle empty input gracefully."""
        empty: list[SampleResult] = []
        # cons@n with empty
        result = cons_at_n(empty)
        assert result["predicted"] is None
        assert result["cost"] == 0

        # mean@n with empty
        result = mean_at_n(empty)
        assert result["correct"] == 0.0

    def test_single_sample(self) -> None:
        s = SampleResult("A", True, 100, 0.8, 0.9, -0.2)
        samples = [s]

        result = cons_at_n(samples)
        assert result["predicted"] == "A"
        assert result["correct"] is True
        assert result["cost"] == 100

        result = think_at_n(samples, eta=0.5)
        assert result["predicted"] == "A"
        assert result["cost"] == 100  # Only 1 sample, selected

    def test_all_same_answer(self) -> None:
        samples = [
            SampleResult("X", True, 100, 0.5, 0.5, -0.5),
            SampleResult("X", True, 200, 0.6, 0.6, -0.4),
            SampleResult("X", True, 300, 0.7, 0.7, -0.3),
        ]
        result = cons_at_n(samples)
        assert result["predicted"] == "X"
        assert result["correct"] is True

    def test_all_none_answers(self) -> None:
        samples = [
            SampleResult(None, False, 100, 0.5, 0.5, -0.5),
            SampleResult(None, False, 200, 0.6, 0.6, -0.4),
        ]
        result = cons_at_n(samples)
        assert result["predicted"] is None
        assert result["correct"] is False

    def test_n_equals_one(self) -> None:
        """With n=1, all selection strategies should pick the single sample."""
        s = SampleResult("A", True, 150, 0.8, 0.9, -0.2)
        samples = [s]

        for strategy_fn in [cons_at_n, long_at_n, short_at_n]:
            result = strategy_fn(samples)
            assert result["predicted"] == "A"

        for strategy_fn in [think_at_n, self_certainty_at_n]:
            result = strategy_fn(samples)
            assert result["predicted"] == "A"
            # With 1 sample, 0 unselected, cost = token_count
            assert result["cost"] == 150

    def test_eta_edge_values(self) -> None:
        """eta close to 0 should select at least 1; eta=1 selects all."""
        samples = [
            SampleResult("A", True, 100, 0.9, 0.9, -0.1),
            SampleResult("B", False, 200, 0.1, 0.1, -1.0),
        ]
        # eta very small -> k = max(1, int(2*0.01)) = max(1,0) = 1
        result = think_at_n(samples, eta=0.01)
        assert result["predicted"] == "A"  # Highest DTR selected

        # eta = 1.0 -> k = int(2*1.0) = 2 -> all selected
        result = think_at_n(samples, eta=1.0)
        assert result["cost"] == 300  # Both fully generated, 0 unselected
