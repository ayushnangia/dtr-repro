"""Distance metrics between probability distributions.

Provides JSD, KLD, cosine distance, Wasserstein approximations, and batched
variants used throughout the DTR computation pipeline. All functions operate
on torch tensors and are compatible with both CPU and CUDA devices.
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


# ---------------------------------------------------------------------------
# KLD variants (reverse and batched)
# ---------------------------------------------------------------------------


def reverse_kld(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Reverse Kullback-Leibler Divergence KL(q || p).

    Given a layer distribution *p* and a reference (final-layer) distribution
    *q*, this computes the divergence in the *reverse* direction compared to
    :func:`kld`.  Useful when the reference should be the "proposal" inside the
    log ratio.

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability distributions.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence KL(q || p) (natural log, base e).
    """
    p = p.clamp(min=_EPS)
    q = q.clamp(min=_EPS)
    return (q * (q.log() - p.log())).sum()


def batch_kld(
    p_batch: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Batched KL(p_i || q) for each row of *p_batch*.

    Parameters
    ----------
    p_batch : torch.Tensor
        2-D tensor of shape ``(num_layers, vocab_size)`` where each row is a
        probability distribution.
    q : torch.Tensor
        1-D reference distribution of shape ``(vocab_size,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of KL divergence values.
    """
    p_batch = p_batch.clamp(min=_EPS)
    q = q.clamp(min=_EPS)

    q_expanded = q.unsqueeze(0).expand_as(p_batch)
    return (p_batch * (p_batch.log() - q_expanded.log())).sum(dim=-1)


def batch_reverse_kld(
    p_batch: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Batched KL(q || p_i) for each row of *p_batch*.

    Parameters
    ----------
    p_batch : torch.Tensor
        2-D tensor of shape ``(num_layers, vocab_size)`` where each row is a
        probability distribution.
    q : torch.Tensor
        1-D reference distribution of shape ``(vocab_size,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of reverse KL divergence values.
    """
    p_batch = p_batch.clamp(min=_EPS)
    q = q.clamp(min=_EPS)

    q_expanded = q.unsqueeze(0).expand_as(p_batch)
    return (q_expanded * (q_expanded.log() - p_batch.log())).sum(dim=-1)


# ---------------------------------------------------------------------------
# Wasserstein / Earth-mover approximations
# ---------------------------------------------------------------------------


def _sliced_wasserstein_core(
    p: torch.Tensor,
    q: torch.Tensor,
    locs: torch.Tensor,
    n_projections: int = 50,
) -> torch.Tensor:
    """Shared implementation of sliced Wasserstein-1 distance.

    Projects the two discrete distributions onto random 1-D directions, then
    computes the exact 1-D Wasserstein (via CDF differences) on each projection
    and averages the results.

    Complexity: O(K * log K * n_projections) where K = len(p).

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability mass vectors over K support points (must sum to ~1).
    locs : torch.Tensor
        2-D tensor of shape ``(K, hidden_dim)`` giving the embedding location
        for each of the K support points.
    n_projections : int, optional
        Number of random projection directions.

    Returns
    -------
    torch.Tensor
        Scalar approximate Wasserstein-1 distance.
    """
    K, d = locs.shape

    # Random unit directions: (n_projections, d)
    directions = torch.randn(n_projections, d, dtype=locs.dtype, device=locs.device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=_EPS)

    # Project support locations onto each direction: (n_projections, K)
    projected = directions @ locs.T  # (n_projections, K)

    # For each projection, sort and compute 1-D Wasserstein via CDF diff.
    total = torch.tensor(0.0, dtype=locs.dtype, device=locs.device)

    for i in range(n_projections):
        proj_i = projected[i]  # (K,)
        sorted_indices = proj_i.argsort()

        sorted_proj = proj_i[sorted_indices]  # sorted locations
        p_sorted = p[sorted_indices]
        q_sorted = q[sorted_indices]

        # CDF differences
        cdf_diff = (p_sorted - q_sorted).cumsum(dim=0)

        # Distances between adjacent sorted projected locations
        gaps = sorted_proj[1:] - sorted_proj[:-1]

        # W1 = sum of |CDF_diff[j]| * gap[j]
        total = total + (cdf_diff[:-1].abs() * gaps).sum()

    return total / n_projections


def sliced_wasserstein_1d(
    p: torch.Tensor,
    q: torch.Tensor,
    embeddings: torch.Tensor,
    n_projections: int = 50,
) -> torch.Tensor:
    """Sliced Wasserstein-1 approximation between distributions *p* and *q*.

    Projects both distributions onto random 1-D directions in embedding space,
    computes the exact 1-D Wasserstein on each projection (sort + cumulative sum
    of |CDF differences|), and averages over projections.

    Efficient: O(V * log V * n_projections) instead of O(V^3) for exact OT.

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability distributions of shape ``(vocab_size,)``.
    embeddings : torch.Tensor
        Embedding matrix of shape ``(vocab_size, hidden_dim)``.
    n_projections : int, optional
        Number of random projection directions (default 50).

    Returns
    -------
    torch.Tensor
        Scalar approximate Wasserstein-1 distance.

    Raises
    ------
    ValueError
        If *embeddings* is ``None``.
    """
    if embeddings is None:
        raise ValueError(
            "sliced_wasserstein_1d requires an embedding matrix but received None."
        )
    return _sliced_wasserstein_core(p, q, embeddings, n_projections=n_projections)


def wasserstein_topk(
    p: torch.Tensor,
    q: torch.Tensor,
    embeddings: torch.Tensor,
    k: int = 100,
    n_projections: int = 50,
) -> torch.Tensor:
    """Top-k Wasserstein-1 approximation using embedding distances.

    1. Find the union of the top-k token indices from *p* and *q*.
    2. Restrict both distributions to those indices and renormalize.
    3. Use sliced Wasserstein on the restricted token embeddings.

    This dramatically reduces the support size (from full vocab to at most 2k
    tokens), making the distance tractable even for very large vocabularies.

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability distributions of shape ``(vocab_size,)``.
    embeddings : torch.Tensor
        Embedding matrix of shape ``(vocab_size, hidden_dim)``.
    k : int, optional
        Number of top tokens to consider from each distribution (default 100).
    n_projections : int, optional
        Number of random projection directions for the sliced approximation.

    Returns
    -------
    torch.Tensor
        Scalar approximate Wasserstein-1 distance.

    Raises
    ------
    ValueError
        If *embeddings* is ``None``.
    """
    if embeddings is None:
        raise ValueError(
            "wasserstein_topk requires an embedding matrix but received None."
        )

    vocab_size = p.shape[0]
    k = min(k, vocab_size)

    # Union of top-k indices
    topk_p = p.topk(k).indices
    topk_q = q.topk(k).indices
    union_indices = torch.unique(torch.cat([topk_p, topk_q]))

    # Restrict and renormalize
    p_restricted = p[union_indices]
    q_restricted = q[union_indices]

    p_sum = p_restricted.sum().clamp(min=_EPS)
    q_sum = q_restricted.sum().clamp(min=_EPS)

    p_restricted = p_restricted / p_sum
    q_restricted = q_restricted / q_sum

    # Restricted embeddings
    emb_restricted = embeddings[union_indices]

    return _sliced_wasserstein_core(
        p_restricted, q_restricted, emb_restricted, n_projections=n_projections
    )


def batch_wasserstein_topk(
    p_batch: torch.Tensor,
    q: torch.Tensor,
    embeddings: torch.Tensor,
    k: int = 100,
    n_projections: int = 50,
) -> torch.Tensor:
    """Batched top-k Wasserstein-1 for per-layer computation.

    Computes :func:`wasserstein_topk` between each row of *p_batch* and *q*.

    Parameters
    ----------
    p_batch : torch.Tensor
        2-D tensor of shape ``(num_layers, vocab_size)`` where each row is a
        probability distribution.
    q : torch.Tensor
        1-D reference distribution of shape ``(vocab_size,)``.
    embeddings : torch.Tensor
        Embedding matrix of shape ``(vocab_size, hidden_dim)``.
    k : int, optional
        Number of top tokens per distribution (default 100).
    n_projections : int, optional
        Number of random projection directions.

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of Wasserstein-1 distance values.

    Raises
    ------
    ValueError
        If *embeddings* is ``None``.
    """
    if embeddings is None:
        raise ValueError(
            "batch_wasserstein_topk requires an embedding matrix but received None."
        )

    num_layers = p_batch.shape[0]
    results = torch.empty(num_layers, dtype=p_batch.dtype, device=p_batch.device)

    for i in range(num_layers):
        results[i] = wasserstein_topk(
            p_batch[i], q, embeddings, k=k, n_projections=n_projections
        )

    return results


# ---------------------------------------------------------------------------
# Lightweight hidden-space metrics (no vocab projection)
# ---------------------------------------------------------------------------


def norm_weighted_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Norm-weighted cosine distance: captures both direction and scale.

    ``d(x, y) = (1 - cos(x, y)) * (||x|| + ||y||) / 2``

    More informative than plain cosine because it is sensitive to norm
    changes across layers (which affect softmax temperature / distribution
    sharpness).

    Parameters
    ----------
    x, y : torch.Tensor
        1-D vectors of the same length.

    Returns
    -------
    torch.Tensor
        Scalar distance (>= 0).
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        x.unsqueeze(0), y.unsqueeze(0)
    ).squeeze(0)
    avg_norm = 0.5 * (x.norm() + y.norm())
    return (1.0 - cos_sim) * avg_norm


def batch_cosine_distance(
    h_batch: torch.Tensor, h_ref: torch.Tensor,
) -> torch.Tensor:
    """Cosine distance between each row of *h_batch* and *h_ref*.

    Parameters
    ----------
    h_batch : torch.Tensor
        Shape ``(num_layers, hidden_dim)``.
    h_ref : torch.Tensor
        Shape ``(hidden_dim,)`` -- typically the final layer hidden state.

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of cosine distances in [0, 2].
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        h_batch, h_ref.unsqueeze(0), dim=-1,
    )
    return 1.0 - cos_sim


def batch_norm_weighted_cosine(
    h_batch: torch.Tensor, h_ref: torch.Tensor,
) -> torch.Tensor:
    """Norm-weighted cosine distance between each row and *h_ref*.

    Parameters
    ----------
    h_batch : torch.Tensor
        Shape ``(num_layers, hidden_dim)``.
    h_ref : torch.Tensor
        Shape ``(hidden_dim,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of distances (>= 0).
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        h_batch, h_ref.unsqueeze(0), dim=-1,
    )
    norms = h_batch.norm(dim=-1)
    ref_norm = h_ref.norm()
    avg_norms = 0.5 * (norms + ref_norm)
    return (1.0 - cos_sim) * avg_norms


# ---------------------------------------------------------------------------
# SVD-compressed unembedding JSD
# ---------------------------------------------------------------------------


class SVDCompressedUnembedding:
    """Low-rank approximation of the unembedding matrix for fast JSD.

    Pre-computes the top-*k* SVD of the unembedding (lm_head) weight matrix.
    Hidden states are projected to *k* dimensions instead of the full
    vocabulary, then JSD is computed in this compressed space.

    This gives ~300-600x speedup over full-vocab JSD while capturing the
    dominant modes of distributional variation.

    Parameters
    ----------
    lm_head_weight : torch.Tensor
        Shape ``(vocab_size, hidden_dim)`` -- the unembedding matrix.
    k : int
        Number of singular components to keep (default 256).
    """

    def __init__(self, lm_head_weight: torch.Tensor, k: int = 256) -> None:
        # W = U @ diag(S) @ V^T  where W is (V, d)
        # We want to project hidden states h (d,) -> compressed logits (k,)
        # compressed_logits = h @ V_k @ diag(S_k) = h @ projection_matrix
        # where projection_matrix is (d, k)
        U, S, Vh = torch.linalg.svd(lm_head_weight.float(), full_matrices=False)
        # U: (V, min(V,d)), S: (min(V,d),), Vh: (min(V,d), d)
        # Keep top-k components
        self.k = min(k, S.shape[0])
        # projection = Vh[:k].T @ diag(S[:k]) -> shape (d, k)
        self.projection = (Vh[:self.k].T * S[:self.k].unsqueeze(0))
        self._device_cached = False

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to compressed logit space.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(num_layers, hidden_dim)`` or ``(hidden_dim,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers, k)`` or ``(k,)``.
        """
        if not self._device_cached:
            self.projection = self.projection.to(hidden_states.device)
            self._device_cached = True
        return hidden_states @ self.projection

    def batch_jsd(
        self,
        hidden_states: torch.Tensor,
        layer_norm: object | None = None,
        base: float = 2.0,
    ) -> torch.Tensor:
        """Compute per-layer JSD in compressed space.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(num_layers, hidden_dim)``.
        layer_norm : callable or None
            If provided, apply final layer norm before projection (logit
            lens style).  Pass ``None`` to skip (paper's literal formula).
        base : float
            Logarithm base for JSD.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers,)`` of JSD values.
        """
        if layer_norm is not None:
            norm_device = (
                next(layer_norm.parameters()).device
                if hasattr(layer_norm, "parameters")
                else hidden_states.device
            )
            device = hidden_states.device
            normed = torch.stack([
                layer_norm(hidden_states[i].to(norm_device)).to(device)
                for i in range(hidden_states.size(0))
            ])
        else:
            normed = hidden_states

        # Project to compressed space: (num_layers, k)
        compressed_logits = self.project(normed)

        # Softmax over compressed dims
        probs = torch.softmax(compressed_logits, dim=-1)

        # JSD of each layer against final layer
        p_final = probs[-1]
        return batch_jsd(probs, p_final, base=base)
