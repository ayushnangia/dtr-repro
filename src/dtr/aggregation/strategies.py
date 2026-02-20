"""Response aggregation strategies: Cons@n, Mean@n, Long@n, Short@n, Self-Certainty@n, Think@n."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class SampleResult:
    """A single generation sample for one question."""

    answer: str | None  # Extracted answer
    correct: bool  # Whether answer matches gold
    token_count: int  # Number of generated tokens
    dtr: float  # Deep-thinking ratio
    self_certainty: float  # Self-certainty metric
    log_prob: float  # Mean log probability


def majority_vote(answers: list[str | None]) -> str | None:
    """Return most common non-None answer. Ties broken arbitrarily."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


def cons_at_n(samples: list[SampleResult]) -> dict:
    """Consensus@n: majority vote over ALL n samples.
    Cost = sum of all token counts."""
    answers = [s.answer for s in samples]
    predicted = majority_vote(answers)
    cost = sum(s.token_count for s in samples)
    accuracy = any(s.correct for s in samples if s.answer == predicted) if predicted else False
    return {"strategy": "cons@n", "predicted": predicted, "correct": accuracy, "cost": cost}


def mean_at_n(samples: list[SampleResult]) -> dict:
    """Mean@n: average accuracy (no aggregation). Expected accuracy = correct/total.
    Cost = sum of all token counts."""
    n = len(samples)
    correct_count = sum(1 for s in samples if s.correct)
    cost = sum(s.token_count for s in samples)
    return {
        "strategy": "mean@n",
        "predicted": None,
        "correct": correct_count / n if n > 0 else 0.0,
        "cost": cost,
    }


def long_at_n(samples: list[SampleResult], eta: float = 0.5) -> dict:
    """Long@n: majority vote over longest eta fraction of samples (by token count).
    Cost = sum of all token counts (must generate all to know which are longest)."""
    k = max(1, int(len(samples) * eta))
    sorted_samples = sorted(samples, key=lambda s: s.token_count, reverse=True)
    selected = sorted_samples[:k]
    answers = [s.answer for s in selected]
    predicted = majority_vote(answers)
    cost = sum(s.token_count for s in samples)  # Must generate all
    accuracy = any(s.correct for s in selected if s.answer == predicted) if predicted else False
    return {"strategy": "long@n", "predicted": predicted, "correct": accuracy, "cost": cost}


def short_at_n(samples: list[SampleResult], eta: float = 0.5) -> dict:
    """Short@n: majority vote over shortest eta fraction."""
    k = max(1, int(len(samples) * eta))
    sorted_samples = sorted(samples, key=lambda s: s.token_count)
    selected = sorted_samples[:k]
    answers = [s.answer for s in selected]
    predicted = majority_vote(answers)
    cost = sum(s.token_count for s in samples)
    accuracy = any(s.correct for s in selected if s.answer == predicted) if predicted else False
    return {"strategy": "short@n", "predicted": predicted, "correct": accuracy, "cost": cost}


def self_certainty_at_n(
    samples: list[SampleResult], eta: float = 0.5, prefix_len: int = 50
) -> dict:
    """Self-Certainty@n: majority vote over top eta fraction by self-certainty (from prefix).
    Cost = sum of selected token counts + prefix overhead for unselected.

    Note: prefix_len parameter indicates that self-certainty is computed from only
    the first prefix_len tokens. The cost computation accounts for this.
    """
    k = max(1, int(len(samples) * eta))
    sorted_samples = sorted(samples, key=lambda s: s.self_certainty, reverse=True)
    selected = sorted_samples[:k]
    unselected = sorted_samples[k:]
    answers = [s.answer for s in selected]
    predicted = majority_vote(answers)
    # Cost: full tokens for selected + prefix for unselected
    cost = sum(s.token_count for s in selected) + prefix_len * len(unselected)
    accuracy = any(s.correct for s in selected if s.answer == predicted) if predicted else False
    return {
        "strategy": "self_certainty@n",
        "predicted": predicted,
        "correct": accuracy,
        "cost": cost,
    }


def think_at_n(samples: list[SampleResult], eta: float = 0.5, prefix_len: int = 50) -> dict:
    """Think@n: majority vote over top eta fraction by DTR (from prefix).
    Cost = sum of selected token counts + prefix overhead for unselected.

    KEY INSIGHT: DTR computed from just the first prefix_len tokens strongly
    predicts overall response quality. Only need to generate prefix to rank.
    """
    k = max(1, int(len(samples) * eta))
    sorted_samples = sorted(samples, key=lambda s: s.dtr, reverse=True)
    selected = sorted_samples[:k]
    unselected = sorted_samples[k:]
    answers = [s.answer for s in selected]
    predicted = majority_vote(answers)
    # Cost: full tokens for selected + prefix for unselected
    cost = sum(s.token_count for s in selected) + prefix_len * len(unselected)
    accuracy = any(s.correct for s in selected if s.answer == predicted) if predicted else False
    return {"strategy": "think@n", "predicted": predicted, "correct": accuracy, "cost": cost}


def run_all_strategies(
    samples: list[SampleResult],
    eta: float = 0.5,
    prefix_len: int = 50,
) -> list[dict]:
    """Run all 6 strategies on the same set of samples."""
    return [
        cons_at_n(samples),
        mean_at_n(samples),
        long_at_n(samples, eta=eta),
        short_at_n(samples, eta=eta),
        self_certainty_at_n(samples, eta=eta, prefix_len=prefix_len),
        think_at_n(samples, eta=eta, prefix_len=prefix_len),
    ]


def run_trials(
    all_samples: list[SampleResult],
    n: int,
    n_trials: int,
    eta: float = 0.5,
    prefix_len: int = 50,
    rng: np.random.Generator | None = None,
) -> dict[str, list[dict]]:
    """Run multiple random trials, each subsampling n responses.
    Returns {strategy_name: [trial_results]}."""
    if rng is None:
        rng = np.random.default_rng(42)

    results: dict[str, list[dict]] = {}
    for _ in range(n_trials):
        indices = rng.choice(len(all_samples), size=min(n, len(all_samples)), replace=False)
        trial_samples = [all_samples[i] for i in indices]
        trial_results = run_all_strategies(trial_samples, eta=eta, prefix_len=prefix_len)
        for r in trial_results:
            name = r["strategy"]
            if name not in results:
                results[name] = []
            results[name].append(r)
    return results
