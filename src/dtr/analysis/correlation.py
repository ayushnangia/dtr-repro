"""Binned Pearson correlation analysis for DTR paper Table 1.

Implements the correlation methodology described in the paper: samples are
sorted by metric value, divided into equal-frequency quantile bins, and the
Pearson correlation is computed between bin-level mean metric and bin-level
mean accuracy.  This smooths out per-sample noise and provides a robust
signal of monotonic association.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def compute_binned_correlation(
    metric_values: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Compute binned Pearson correlation between a metric and accuracy.

    Paper method:
    1. Sort samples by metric value
    2. Divide into n_bins equal quantile bins
    3. Compute mean metric and mean accuracy per bin
    4. Pearson r between bin means

    Parameters
    ----------
    metric_values : np.ndarray
        Shape ``(N,)`` of metric values per sample.
    accuracies : np.ndarray
        Shape ``(N,)`` of 0/1 correctness per sample.
    n_bins : int
        Number of quantile bins (default 5, matching the paper).

    Returns
    -------
    dict
        Keys:

        - ``r``: Pearson correlation coefficient
        - ``p_value``: two-sided p-value
        - ``bin_metric_means``: ``(n_bins,)`` mean metric per bin
        - ``bin_accuracy_means``: ``(n_bins,)`` mean accuracy per bin
        - ``bin_edges``: ``(n_bins + 1,)`` bin edge values
        - ``bin_sizes``: ``(n_bins,)`` number of samples per bin
    """
    metric_values = np.asarray(metric_values, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    n_samples = len(metric_values)

    # Edge case: too few samples to form the requested bins
    if n_samples < n_bins:
        nan = float("nan")
        return {
            "r": nan,
            "p_value": nan,
            "bin_metric_means": np.array([]),
            "bin_accuracy_means": np.array([]),
            "bin_edges": np.array([]),
            "bin_sizes": np.array([], dtype=int),
        }

    # Compute quantile bin edges (0th, 20th, 40th, 60th, 80th, 100th for 5 bins)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(metric_values, quantiles)

    # Make the rightmost edge slightly larger so np.digitize assigns the max
    # value to the last bin rather than creating an overflow bin.
    bin_edges[-1] = bin_edges[-1] + 1e-10

    # Assign each sample to a bin (1-indexed from np.digitize, with right=False)
    bin_indices = np.digitize(metric_values, bin_edges[1:], right=False)
    # bin_indices is in [0, n_bins-1]: bin 0 means value < edge[1], etc.
    # Clamp to valid range
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute per-bin statistics
    bin_metric_means = np.zeros(n_bins, dtype=np.float64)
    bin_accuracy_means = np.zeros(n_bins, dtype=np.float64)
    bin_sizes = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        count = mask.sum()
        bin_sizes[b] = count
        if count > 0:
            bin_metric_means[b] = metric_values[mask].mean()
            bin_accuracy_means[b] = accuracies[mask].mean()
        else:
            bin_metric_means[b] = float("nan")
            bin_accuracy_means[b] = float("nan")

    # Filter out empty bins for correlation computation
    valid = bin_sizes > 0
    valid_metric = bin_metric_means[valid]
    valid_accuracy = bin_accuracy_means[valid]

    if len(valid_metric) < 2:
        # Cannot compute correlation with fewer than 2 data points
        return {
            "r": float("nan"),
            "p_value": float("nan"),
            "bin_metric_means": bin_metric_means,
            "bin_accuracy_means": bin_accuracy_means,
            "bin_edges": bin_edges,
            "bin_sizes": bin_sizes,
        }

    # Check for zero variance (constant values in either array)
    if np.std(valid_metric) < 1e-15 or np.std(valid_accuracy) < 1e-15:
        return {
            "r": float("nan"),
            "p_value": float("nan"),
            "bin_metric_means": bin_metric_means,
            "bin_accuracy_means": bin_accuracy_means,
            "bin_edges": bin_edges,
            "bin_sizes": bin_sizes,
        }

    r, p_value = stats.pearsonr(valid_metric, valid_accuracy)

    return {
        "r": float(r),
        "p_value": float(p_value),
        "bin_metric_means": bin_metric_means,
        "bin_accuracy_means": bin_accuracy_means,
        "bin_edges": bin_edges,
        "bin_sizes": bin_sizes,
    }


def compute_correlation_table(
    samples: list[dict],
    metric_names: list[str],
    n_bins: int = 5,
) -> dict[str, dict]:
    """Compute binned correlation for multiple metrics.

    Parameters
    ----------
    samples : list[dict]
        Each dict must contain keys for every name in *metric_names* plus a
        ``"correct"`` key with a 0/1 (or bool) value.
    metric_names : list[str]
        Names of metrics to correlate with accuracy.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    dict[str, dict]
        Mapping from metric name to the correlation result dict returned by
        :func:`compute_binned_correlation`.
    """
    accuracies = np.array([float(s["correct"]) for s in samples], dtype=np.float64)

    results: dict[str, dict] = {}
    for name in metric_names:
        values = np.array([float(s[name]) for s in samples], dtype=np.float64)
        results[name] = compute_binned_correlation(values, accuracies, n_bins=n_bins)

    return results


def average_over_seeds(
    per_seed_results: list[dict[str, dict]],
) -> dict[str, dict]:
    """Average correlation results over multiple random seeds.

    Parameters
    ----------
    per_seed_results : list[dict[str, dict]]
        Each element is a ``{metric_name: {r, p_value, ...}}`` dict as returned
        by :func:`compute_correlation_table`.

    Returns
    -------
    dict[str, dict]
        ``{metric_name: {mean_r, std_r, mean_p_value, all_r}}`` where
        ``all_r`` is the list of per-seed r values (useful for error bars).
    """
    if not per_seed_results:
        return {}

    metric_names = list(per_seed_results[0].keys())
    averaged: dict[str, dict] = {}

    for name in metric_names:
        r_values = []
        p_values = []
        for seed_result in per_seed_results:
            if name not in seed_result:
                continue
            r_val = seed_result[name]["r"]
            p_val = seed_result[name]["p_value"]
            # Skip NaN results from degenerate seeds
            if not (np.isnan(r_val) or np.isnan(p_val)):
                r_values.append(r_val)
                p_values.append(p_val)

        if r_values:
            averaged[name] = {
                "mean_r": float(np.mean(r_values)),
                "std_r": float(np.std(r_values, ddof=1)) if len(r_values) > 1 else 0.0,
                "mean_p_value": float(np.mean(p_values)),
                "all_r": r_values,
            }
        else:
            averaged[name] = {
                "mean_r": float("nan"),
                "std_r": float("nan"),
                "mean_p_value": float("nan"),
                "all_r": [],
            }

    return averaged
