"""Distance metrics between probability distributions.

Provides JSD, KLD, cosine distance, and batched variants used throughout
the DTR computation pipeline. All functions operate on torch tensors and
are compatible with both CPU and CUDA devices.
"""

from __future__ import annotations

import torch

# Small constant added before log to avoid log(0).
_EPS = 1e-10


def jsd(p: torch.Tensor, q: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """Jensen-Shannon Divergence between distributions *p* and *q*.

    JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)  where  m = 0.5*(p+q)

    Bounded in [0, log(2)] for base=e, or [0, 1] for base=2.
    Handles zeros by adding a small epsilon before taking logarithms.

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability distributions (must sum to ~1 and be non-negative).
    base : float, optional
        Logarithm base.  Use 2.0 (default) to get JSD in [0, 1].

    Returns
    -------
    torch.Tensor
        Scalar JSD value.
    """
    p = p.clamp(min=_EPS)
    q = q.clamp(min=_EPS)
    m = 0.5 * (p + q)

    log_base = torch.tensor(base, dtype=p.dtype, device=p.device).log()

    kl_pm = (p * (p.log() - m.log())).sum() / log_base
    kl_qm = (q * (q.log() - m.log())).sum() / log_base

    return 0.5 * kl_pm + 0.5 * kl_qm


def kld(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Kullback-Leibler Divergence KL(p || q).

    Used for Appendix A analysis.

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability distributions.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence (natural log, base e).
    """
    p = p.clamp(min=_EPS)
    q = q.clamp(min=_EPS)
    return (p * (p.log() - q.log())).sum()


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cosine distance: 1 - cosine_similarity.

    Operates on raw hidden-state vectors.

    Parameters
    ----------
    x, y : torch.Tensor
        1-D vectors of the same length.

    Returns
    -------
    torch.Tensor
        Scalar cosine distance in [0, 2].
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        x.unsqueeze(0), y.unsqueeze(0)
    )
    return 1.0 - cos_sim.squeeze(0)


def batch_jsd(p_batch: torch.Tensor, q: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """Compute JSD between each row of *p_batch* and *q*.

    Parameters
    ----------
    p_batch : torch.Tensor
        2-D tensor of shape ``(num_layers, vocab_size)`` where each row is a
        probability distribution.
    q : torch.Tensor
        1-D reference distribution of shape ``(vocab_size,)``.
    base : float, optional
        Logarithm base (default 2.0).

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of JSD values.
    """
    p_batch = p_batch.clamp(min=_EPS)
    q = q.clamp(min=_EPS)

    # Expand q to match batch: (1, vocab_size)
    q_expanded = q.unsqueeze(0).expand_as(p_batch)

    m = 0.5 * (p_batch + q_expanded)

    log_base = torch.tensor(base, dtype=p_batch.dtype, device=p_batch.device).log()

    kl_pm = (p_batch * (p_batch.log() - m.log())).sum(dim=-1) / log_base
    kl_qm = (q_expanded * (q_expanded.log() - m.log())).sum(dim=-1) / log_base

    return 0.5 * kl_pm + 0.5 * kl_qm
