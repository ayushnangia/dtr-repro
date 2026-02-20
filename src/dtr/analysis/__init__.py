"""Correlation analysis and sensitivity sweeps for the DTR paper."""
from __future__ import annotations

from dtr.analysis.correlation import (
    average_over_seeds,
    compute_binned_correlation,
    compute_correlation_table,
)
from dtr.analysis.sensitivity import (
    recompute_dtr_from_jsd,
    sweep_dtr_params,
    sweep_prefix_lengths,
)

__all__ = [
    # correlation.py
    "compute_binned_correlation",
    "compute_correlation_table",
    "average_over_seeds",
    # sensitivity.py
    "recompute_dtr_from_jsd",
    "sweep_dtr_params",
    "sweep_prefix_lengths",
]
