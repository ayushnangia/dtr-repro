"""DTR (Deep-Thinking Ratio) computation -- Algorithm 1 from the paper.

Provides per-layer JSD computation via the "logit lens" technique, settling
depth detection, deep-thinking token classification, and both batch and
streaming (online) DTR computation.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from dtr.metrics.distances import batch_jsd


# ---------------------------------------------------------------------------
# Core per-layer computation
# ---------------------------------------------------------------------------

def compute_jsd_per_layer(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layer_norm: Any,
    base: float = 2.0,
) -> torch.Tensor:
    """For a single token position, compute JSD between each layer's
    distribution and the final layer's distribution.

    Steps
    -----
    1. Apply *layer_norm* to **every** layer's hidden state (the "logit lens"
       technique -- without normalisation, scale mismatch dominates the JSD).
    2. Project through the language-model head:
       ``logits = normed_hidden @ lm_head_weight.T``
    3. Apply softmax to obtain per-layer probability distributions.
    4. Compute ``JSD(p_final, p_layer)`` for each layer.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``(num_layers, hidden_dim)``.
    lm_head_weight : torch.Tensor
        Shape ``(vocab_size, hidden_dim)``.
    layer_norm : callable
        The model's final RMSNorm (or equivalent) module.
    base : float, optional
        Logarithm base for JSD (default 2.0 gives range [0, 1]).

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of JSD values.
    """
    num_layers = hidden_states.size(0)

    # 1. Normalise all layers using the final layer norm.
    normed = torch.stack([layer_norm(hidden_states[i]) for i in range(num_layers)])

    # 2. Project to vocabulary space.
    logits = normed @ lm_head_weight.t()  # (num_layers, vocab_size)

    # 3. Softmax -> probability distributions.
    probs = torch.softmax(logits, dim=-1)  # (num_layers, vocab_size)

    # 4. JSD of each layer against the final layer.
    p_final = probs[-1]  # (vocab_size,)
    jsd_values = batch_jsd(probs, p_final, base=base)  # (num_layers,)

    return jsd_values


# ---------------------------------------------------------------------------
# Settling depth & deep-thinking classification
# ---------------------------------------------------------------------------

def compute_settling_depth(
    jsd_values: torch.Tensor,
    threshold_g: float = 0.5,
) -> int:
    """Find the first layer where the cumulative minimum of JSD drops
    below threshold *g*.

    .. math::
        \\bar{D}_l = \\operatorname{cummin}(\\text{JSD}_1, \\dots, \\text{JSD}_l)

        c = \\min\\{l : \\bar{D}_l \\le g\\}

    If the JSD never settles, returns ``num_layers``.

    Parameters
    ----------
    jsd_values : torch.Tensor
        Shape ``(num_layers,)`` of JSD values.
    threshold_g : float
        Settling threshold (default 0.5).

    Returns
    -------
    int
        Settling depth (1-indexed layer number, or ``num_layers`` if unsettled).
    """
    cummin_vals, _ = torch.cummin(jsd_values, dim=0)
    mask = cummin_vals <= threshold_g
    if mask.any():
        return int(mask.nonzero(as_tuple=False)[0].item()) + 1  # 1-indexed
    return int(jsd_values.size(0))


def is_deep_thinking_token(
    settling_depth: int,
    num_layers: int,
    depth_ratio_rho: float = 0.85,
) -> bool:
    """A token is *deep thinking* if its settling depth >= ceil(rho * num_layers).

    Parameters
    ----------
    settling_depth : int
        Settling depth as returned by :func:`compute_settling_depth`.
    num_layers : int
        Total number of layers in the model.
    depth_ratio_rho : float
        Depth ratio threshold (default 0.85).

    Returns
    -------
    bool
    """
    threshold_layer = math.ceil(depth_ratio_rho * num_layers)
    return settling_depth >= threshold_layer


# ---------------------------------------------------------------------------
# Batch DTR
# ---------------------------------------------------------------------------

def compute_dtr(
    jsd_matrix: torch.Tensor,
    threshold_g: float = 0.5,
    depth_ratio_rho: float = 0.85,
) -> dict:
    """Compute the Deep-Thinking Ratio for a full sequence.

    Parameters
    ----------
    jsd_matrix : torch.Tensor
        Shape ``(num_tokens, num_layers)`` -- each row holds the per-layer
        JSD values for one token position.
    threshold_g : float
        Settling threshold (default 0.5).
    depth_ratio_rho : float
        Depth ratio threshold (default 0.85).

    Returns
    -------
    dict
        Keys: ``dtr``, ``settling_depths``, ``deep_thinking_mask``,
        ``num_deep``, ``total_tokens``.
    """
    num_tokens, num_layers = jsd_matrix.shape
    settling_depths: list[int] = []
    deep_thinking_mask: list[bool] = []

    for t in range(num_tokens):
        sd = compute_settling_depth(jsd_matrix[t], threshold_g)
        settling_depths.append(sd)
        deep_thinking_mask.append(
            is_deep_thinking_token(sd, num_layers, depth_ratio_rho)
        )

    num_deep = sum(deep_thinking_mask)
    return {
        "dtr": num_deep / num_tokens if num_tokens > 0 else 0.0,
        "settling_depths": settling_depths,
        "deep_thinking_mask": deep_thinking_mask,
        "num_deep": num_deep,
        "total_tokens": num_tokens,
    }


# ---------------------------------------------------------------------------
# Streaming / online DTR
# ---------------------------------------------------------------------------

class DTRAccumulator:
    """Accumulates DTR statistics token-by-token during generation.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers.
    lm_head_weight : torch.Tensor
        Shape ``(vocab_size, hidden_dim)``.
    layer_norm : callable
        Final RMSNorm module (or equivalent).
    threshold_g : float
        Settling threshold.
    depth_ratio_rho : float
        Depth ratio threshold.
    """

    def __init__(
        self,
        num_layers: int,
        lm_head_weight: torch.Tensor,
        layer_norm: Any,
        threshold_g: float = 0.5,
        depth_ratio_rho: float = 0.85,
    ) -> None:
        self.num_layers = num_layers
        self.lm_head_weight = lm_head_weight
        self.layer_norm = layer_norm
        self.threshold_g = threshold_g
        self.depth_ratio_rho = depth_ratio_rho

        # Accumulated results.
        self._jsd_rows: list[torch.Tensor] = []
        self._settling_depths: list[int] = []
        self._deep_thinking_mask: list[bool] = []

    # -----------------------------------------------------------------
    def add_token(self, hidden_states: torch.Tensor) -> dict:
        """Process one token's hidden states.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(num_layers, hidden_dim)``.

        Returns
        -------
        dict
            Per-token info: ``jsd_values``, ``settling_depth``, ``is_deep``.
        """
        jsd_values = compute_jsd_per_layer(
            hidden_states, self.lm_head_weight, self.layer_norm
        )
        sd = compute_settling_depth(jsd_values, self.threshold_g)
        is_deep = is_deep_thinking_token(sd, self.num_layers, self.depth_ratio_rho)

        self._jsd_rows.append(jsd_values)
        self._settling_depths.append(sd)
        self._deep_thinking_mask.append(is_deep)

        return {
            "jsd_values": jsd_values,
            "settling_depth": sd,
            "is_deep": is_deep,
        }

    # -----------------------------------------------------------------
    def get_results(self) -> dict:
        """Return final DTR and all accumulated statistics.

        Returns
        -------
        dict
            Same schema as :func:`compute_dtr`.
        """
        num_tokens = len(self._settling_depths)
        num_deep = sum(self._deep_thinking_mask)
        return {
            "dtr": num_deep / num_tokens if num_tokens > 0 else 0.0,
            "settling_depths": list(self._settling_depths),
            "deep_thinking_mask": list(self._deep_thinking_mask),
            "num_deep": num_deep,
            "total_tokens": num_tokens,
        }


# ---------------------------------------------------------------------------
# Convenience streaming function
# ---------------------------------------------------------------------------

def compute_dtr_online(
    hidden_states_per_token: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layer_norm: Any,
    threshold_g: float = 0.5,
    depth_ratio_rho: float = 0.85,
) -> None:
    """Streaming placeholder -- call once per token during generation.

    For full streaming use, prefer :class:`DTRAccumulator` directly.  This
    function is provided for API compatibility and is intentionally a no-op;
    instantiate a ``DTRAccumulator`` and call :meth:`add_token` instead.
    """
    # Intentionally a no-op; use DTRAccumulator.add_token directly.
    pass
