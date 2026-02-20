"""Hyperparameter sensitivity analysis for DTR (Figure 4, Table 3).

Provides efficient parameter sweeps that reuse pre-computed JSD matrices
so that no additional model inference is required.  The two main sweeps are:

1. **g / rho sweep** (Figure 4): vary the settling threshold *g* and depth
   ratio *rho* and measure how the binned correlation with accuracy changes.
2. **Prefix-length sweep** (Table 3, Think@n ablation): truncate the JSD
   matrices to varying numbers of initial tokens and measure correlation.
"""
from __future__ import annotations

import math

import numpy as np

from dtr.analysis.correlation import compute_binned_correlation


# ---------------------------------------------------------------------------
# Core: recompute DTR from a numpy JSD matrix
# ---------------------------------------------------------------------------

def recompute_dtr_from_jsd(
    jsd_matrix: np.ndarray,
    threshold_g: float,
    depth_ratio_rho: float,
) -> float:
    """Recompute DTR from a pre-computed JSD matrix with new hyperparameters.

    Applies Algorithm 1 from the paper using cumulative minimum and thresholds.

    Parameters
    ----------
    jsd_matrix : np.ndarray
        Shape ``(T, L)`` where *T* is the number of tokens and *L* is the
        number of layers.  Each entry ``[t, l]`` is the JSD between layer
        *l*'s distribution and the final layer's distribution for token *t*.
    threshold_g : float
        Settling threshold -- a token's JSD cummin must drop to or below this
        value for the token to be considered "settled".
    depth_ratio_rho : float
        Depth ratio threshold -- a token is "deep thinking" if its settling
        depth is >= ``ceil(rho * num_layers)``.

    Returns
    -------
    float
        The Deep-Thinking Ratio (fraction of deep-thinking tokens).
    """
    jsd_matrix = np.asarray(jsd_matrix, dtype=np.float64)
    num_tokens, num_layers = jsd_matrix.shape

    if num_tokens == 0:
        return 0.0

    threshold_layer = math.ceil(depth_ratio_rho * num_layers)
    num_deep = 0

    for t in range(num_tokens):
        row = jsd_matrix[t]
        # Cumulative minimum across layers
        cummin = np.minimum.accumulate(row)
        # Find first layer where cummin <= g
        settled_mask = cummin <= threshold_g
        if settled_mask.any():
            settling_depth = int(np.argmax(settled_mask)) + 1  # 1-indexed
        else:
            settling_depth = num_layers

        if settling_depth >= threshold_layer:
            num_deep += 1

    return num_deep / num_tokens


# ---------------------------------------------------------------------------
# Sweep over g and rho (Figure 4)
# ---------------------------------------------------------------------------

def sweep_dtr_params(
    jsd_matrices: list[np.ndarray],
    accuracies: list[bool],
    g_values: list[float] | None = None,
    rho_values: list[float] | None = None,
    n_bins: int = 5,
) -> list[dict]:
    """Sweep over DTR hyperparameters *g* and *rho*.

    For each ``(g, rho)`` pair:

    1. Recompute DTR for all samples using the stored JSD matrices.
    2. Compute binned correlation with accuracy.

    This is efficient because no new model inference is needed -- we simply
    recompute DTR from saved JSD matrices.

    Parameters
    ----------
    jsd_matrices : list[np.ndarray]
        Per-sample JSD matrices, each of shape ``(T_i, L)`` where *T_i* may
        differ across samples but *L* (number of layers) is constant.
    accuracies : list[bool]
        Correctness labels per sample (``True`` / ``False`` or 1 / 0).
    g_values : list[float] or None
        Settling thresholds to sweep.  Defaults to ``[0.25, 0.5, 0.75]``.
    rho_values : list[float] or None
        Depth ratio thresholds to sweep.  Defaults to ``[0.8, 0.85, 0.9, 0.95]``.
    n_bins : int
        Number of quantile bins for the correlation computation.

    Returns
    -------
    list[dict]
        One dict per ``(g, rho)`` pair with keys:

        - ``g``: the settling threshold
        - ``rho``: the depth ratio threshold
        - ``dtr_values``: ``np.ndarray`` of DTR per sample
        - ``correlation``: result dict from :func:`compute_binned_correlation`
    """
    if g_values is None:
        g_values = [0.25, 0.5, 0.75]
    if rho_values is None:
        rho_values = [0.8, 0.85, 0.9, 0.95]

    acc_array = np.array([float(a) for a in accuracies], dtype=np.float64)
    results: list[dict] = []

    for g in g_values:
        for rho in rho_values:
            dtr_values = np.array(
                [recompute_dtr_from_jsd(m, threshold_g=g, depth_ratio_rho=rho)
                 for m in jsd_matrices],
                dtype=np.float64,
            )
            correlation = compute_binned_correlation(dtr_values, acc_array, n_bins=n_bins)
            results.append({
                "g": g,
                "rho": rho,
                "dtr_values": dtr_values,
                "correlation": correlation,
            })

    return results


# ---------------------------------------------------------------------------
# Sweep over prefix lengths (Table 3 -- Think@n ablation)
# ---------------------------------------------------------------------------

def sweep_prefix_lengths(
    jsd_matrices: list[np.ndarray],
    accuracies: list[bool],
    prefix_lengths: list[int] | None = None,
    threshold_g: float = 0.5,
    depth_ratio_rho: float = 0.85,
    n_bins: int = 5,
) -> list[dict]:
    """Sweep over prefix lengths for the Think@n ablation (Table 3).

    For each *prefix_length*:

    1. Truncate each sample's JSD matrix to the first *prefix_length* tokens
       (``-1`` means use the full sequence).
    2. Recompute DTR from the truncated matrices.
    3. Compute correlation with the **full-sequence** accuracy labels.

    Parameters
    ----------
    jsd_matrices : list[np.ndarray]
        Per-sample JSD matrices, each ``(T_i, L)``.
    accuracies : list[bool]
        Correctness labels per sample.
    prefix_lengths : list[int] or None
        Token counts to truncate to.  ``-1`` means no truncation (full
        sequence).  Defaults to ``[50, 100, 500, 1000, 2000, -1]``.
    threshold_g : float
        Settling threshold.
    depth_ratio_rho : float
        Depth ratio threshold.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    list[dict]
        One dict per prefix length with keys:

        - ``prefix_length``: the prefix length (``-1`` for full)
        - ``dtr_values``: ``np.ndarray`` of DTR per sample
        - ``correlation``: result dict from :func:`compute_binned_correlation`
    """
    if prefix_lengths is None:
        prefix_lengths = [50, 100, 500, 1000, 2000, -1]

    acc_array = np.array([float(a) for a in accuracies], dtype=np.float64)
    results: list[dict] = []

    for plen in prefix_lengths:
        dtr_values: list[float] = []
        for m in jsd_matrices:
            m = np.asarray(m, dtype=np.float64)
            if plen == -1 or plen >= m.shape[0]:
                truncated = m
            else:
                truncated = m[:plen, :]

            dtr_val = recompute_dtr_from_jsd(
                truncated,
                threshold_g=threshold_g,
                depth_ratio_rho=depth_ratio_rho,
            )
            dtr_values.append(dtr_val)

        dtr_arr = np.array(dtr_values, dtype=np.float64)
        correlation = compute_binned_correlation(dtr_arr, acc_array, n_bins=n_bins)
        results.append({
            "prefix_length": plen,
            "dtr_values": dtr_arr,
            "correlation": correlation,
        })

    return results
