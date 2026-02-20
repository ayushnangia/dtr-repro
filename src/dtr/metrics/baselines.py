"""Baseline metrics from the DTR paper (Section 4 / Equations 7-10).

All six metrics used for comparison with DTR:
  1. token_count
  2. reverse_token_count
  3. mean_log_probability  (Eq. 7)
  4. negative_perplexity   (Eq. 8)
  5. negative_entropy       (Eq. 9)
  6. self_certainty         (Eq. 10)

Every function operates exclusively on ``torch.Tensor`` inputs.  Logits are
always assumed to be **raw** (pre-softmax).
"""

from __future__ import annotations

import torch

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Simple token-level metrics
# ---------------------------------------------------------------------------

def token_count(token_ids: torch.Tensor) -> int:
    """Total number of tokens.

    Parameters
    ----------
    token_ids : torch.Tensor
        1-D tensor of token IDs.
    """
    return int(token_ids.numel())


def reverse_token_count(token_ids: torch.Tensor) -> int:
    """Negative token count: ``-len(tokens)``.

    Parameters
    ----------
    token_ids : torch.Tensor
        1-D tensor of token IDs.
    """
    return -int(token_ids.numel())


# ---------------------------------------------------------------------------
# Probability-based metrics
# ---------------------------------------------------------------------------

def mean_log_probability(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    r"""Mean log-probability of the generated tokens under the final-layer
    distribution.

    .. math::
        \frac{1}{T}\sum_{t=1}^{T}\log p_{t,L}(y_t)

    Paper Eq. 7.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(T, vocab_size)`` -- raw (pre-softmax) logits.
    token_ids : torch.Tensor
        Shape ``(T,)`` -- ground-truth / generated token IDs.

    Returns
    -------
    float
    """
    log_probs = torch.log_softmax(logits, dim=-1)  # (T, V)
    # Gather log-prob of each actual token.
    selected = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return float(selected.mean())


def negative_perplexity(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    r"""Negative perplexity.

    .. math::
        -\exp\!\bigl(-\text{mean\_log\_prob}\bigr)

    Paper Eq. 8.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(T, vocab_size)``.
    token_ids : torch.Tensor
        Shape ``(T,)``.

    Returns
    -------
    float
    """
    mlp = mean_log_probability(logits, token_ids)
    return float(-torch.exp(torch.tensor(-mlp)))


def negative_entropy(logits: torch.Tensor) -> float:
    r"""Negative mean Shannon entropy of the final-layer distributions.

    .. math::
        -\frac{1}{T}\sum_{t=1}^{T} H\!\bigl(p_{t,L}\bigr)

    Paper Eq. 9.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(T, vocab_size)``.

    Returns
    -------
    float
    """
    probs = torch.softmax(logits, dim=-1)            # (T, V)
    log_probs = torch.log(probs.clamp(min=_EPS))     # (T, V)
    entropies = -(probs * log_probs).sum(dim=-1)      # (T,)
    return float(-entropies.mean())


def self_certainty(logits: torch.Tensor) -> float:
    r"""Mean KL divergence from a uniform distribution to the final-layer
    distribution.

    .. math::
        \frac{1}{T}\sum_{t=1}^{T} \mathrm{KL}\!\bigl(\text{uniform}\;\|\;\,p_{t,L}\bigr)

    Paper Eq. 10.  Higher values indicate greater model certainty.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(T, vocab_size)``.

    Returns
    -------
    float
    """
    T, V = logits.shape
    probs = torch.softmax(logits, dim=-1)             # (T, V)
    log_probs = torch.log(probs.clamp(min=_EPS))      # (T, V)

    # KL(uniform || p) = sum_v (1/V) * [log(1/V) - log p_v]
    #                   = log(1/V) - (1/V)*sum_v log p_v
    log_uniform = -torch.log(torch.tensor(float(V), dtype=logits.dtype, device=logits.device))
    kl_values = log_uniform - (1.0 / V) * log_probs.sum(dim=-1)  # (T,)
    return float(kl_values.mean())


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def compute_all_baselines(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> dict[str, float]:
    """Compute all six baseline metrics at once.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(T, vocab_size)``.
    token_ids : torch.Tensor
        Shape ``(T,)``.

    Returns
    -------
    dict[str, float]
    """
    return {
        "token_count": float(token_count(token_ids)),
        "reverse_token_count": float(reverse_token_count(token_ids)),
        "mean_log_probability": mean_log_probability(logits, token_ids),
        "negative_perplexity": negative_perplexity(logits, token_ids),
        "negative_entropy": negative_entropy(logits),
        "self_certainty": self_certainty(logits),
    }
