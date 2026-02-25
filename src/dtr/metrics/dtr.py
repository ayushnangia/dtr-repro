"""DTR (Deep-Thinking Ratio) computation -- Algorithm 1 from the paper.

Provides per-layer JSD computation via the "logit lens" technique, settling
depth detection, deep-thinking token classification, and both batch and
streaming (online) DTR computation.

Extended variants:
- **Soft (Rao-Blackwellized) DTR** via sigmoid relaxation of the binary
  deep-thinking classification.
- **Continuous DTR** based on normalised settling depth (no threshold).
- **Generic distance-per-layer** supporting JSD, KLD, reverse KLD, cosine
  distance, and Wasserstein distance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from dtr.metrics.distances import (
    batch_jsd,
    batch_cosine_distance,
    batch_norm_weighted_cosine,
    cosine_distance,
    jsd,
    kld,
    SVDCompressedUnembedding,
)


# ---------------------------------------------------------------------------
# Core per-layer computation
# ---------------------------------------------------------------------------

def compute_jsd_per_layer(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layer_norm: Any,
    base: float = 2.0,
    *,
    _buffers: dict[str, torch.Tensor] | None = None,
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
    _buffers : dict or None, optional
        Pre-allocated working buffers to avoid CPU memory fragmentation.
        Expected keys: ``"logits"``, ``"probs"``, ``"m"`` -- each of shape
        ``(num_layers, vocab_size)`` in float32.  When ``None``, buffers
        are allocated on the fly (original behaviour).

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of JSD values.
    """
    num_layers = hidden_states.size(0)
    device = hidden_states.device

    # Apply norm on whatever device it lives on (don't move it -- it's
    # shared with the model and device_map="auto" owns its placement).
    norm_device = next(layer_norm.parameters()).device if hasattr(layer_norm, "parameters") else device

    # 1. Normalise all layers using the final layer norm.
    normed = torch.stack([
        layer_norm(hidden_states[i].to(norm_device)).to(device)
        for i in range(num_layers)
    ])

    if _buffers is not None:
        # --- In-place path: reuse pre-allocated buffers ---
        logits_buf = _buffers["logits"]
        probs_buf = _buffers["probs"]
        m_buf = _buffers["m"]

        # 2. Project to vocabulary space (in-place into logits_buf).
        torch.mm(normed, lm_head_weight.t(), out=logits_buf)
        del normed

        # 3. Softmax in-place into probs_buf.
        torch.softmax(logits_buf, dim=-1, out=probs_buf)

        # 4. JSD of each layer against the final layer (in-place).
        jsd_values = _batch_jsd_inplace(probs_buf, m_buf, base=base)
        return jsd_values
    else:
        # --- Original allocation path (for non-streaming callers) ---
        logits = normed @ lm_head_weight.t()
        del normed

        probs = torch.softmax(logits, dim=-1)
        del logits

        p_final = probs[-1]
        jsd_values = batch_jsd(probs, p_final, base=base)
        del probs, p_final

        return jsd_values


def _batch_jsd_inplace(
    probs: torch.Tensor,
    m_buf: torch.Tensor,
    base: float = 2.0,
) -> torch.Tensor:
    """Compute batch JSD using a pre-allocated buffer for the mixture ``m``.

    Operates on *probs* in-place where possible to minimise allocations.
    """
    _EPS = 1e-10
    probs.clamp_(min=_EPS)
    p_final = probs[-1]  # view, no alloc

    # m = 0.5 * (probs + p_final)
    m_buf.copy_(probs)
    m_buf.add_(p_final.unsqueeze(0))
    m_buf.mul_(0.5)

    log_base = math.log(base)

    # KL(p || m) per layer
    # Use probs.log() - m_buf.log() in a memory-friendly way:
    # Compute log in-place on m_buf, store result, then restore m_buf later.
    log_m = m_buf.log()  # (num_layers, vocab_size) -- reuses m_buf storage
    log_p = probs.log()
    kl_pm = (probs * (log_p - log_m)).sum(dim=-1) / log_base

    # KL(q || m) where q = p_final
    q_log = p_final.log()
    kl_qm = (p_final.unsqueeze(0) * (q_log.unsqueeze(0) - log_m)).sum(dim=-1) / log_base

    jsd_values = 0.5 * kl_pm + 0.5 * kl_qm
    return jsd_values


# ---------------------------------------------------------------------------
# Generic per-layer distance computation
# ---------------------------------------------------------------------------

_SUPPORTED_DISTANCE_METHODS = (
    "jsd", "kld", "reverse_kld", "cosine", "wasserstein",
    "norm_weighted_cosine", "svd_jsd",
)


def compute_distance_per_layer(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    layer_norm: Any,
    method: str = "jsd",
    embeddings: torch.Tensor | None = None,
    base: float = 2.0,
    wasserstein_k: int = 100,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute a per-layer distance between each layer and the final layer.

    This is a generalisation of :func:`compute_jsd_per_layer` that supports
    multiple distance functions.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``(num_layers, hidden_dim)``.
    lm_head_weight : torch.Tensor
        Shape ``(vocab_size, hidden_dim)``.
    layer_norm : callable
        The model's final RMSNorm (or equivalent) module.
    method : str
        Distance method.  One of ``"jsd"``, ``"kld"``, ``"reverse_kld"``,
        ``"cosine"``, ``"wasserstein"``.
    embeddings : torch.Tensor or None
        Embedding matrix needed for the Wasserstein distance ground metric.
        Shape ``(vocab_size, hidden_dim)``.
    base : float, optional
        Logarithm base for JSD (only used when *method* is ``"jsd"``).
    wasserstein_k : int, optional
        Number of top-*k* tokens to use for the Wasserstein approximation.

    Returns
    -------
    torch.Tensor
        Shape ``(num_layers,)`` of distance values.

    Raises
    ------
    ValueError
        If *method* is not recognised, or if ``"wasserstein"`` is requested
        without providing *embeddings*.
    """
    if method not in _SUPPORTED_DISTANCE_METHODS:
        raise ValueError(
            f"Unknown distance method {method!r}.  "
            f"Supported: {_SUPPORTED_DISTANCE_METHODS}"
        )

    num_layers = hidden_states.size(0)

    # --- Hidden-space methods (no vocab projection) ---------------------
    if method == "cosine":
        return batch_cosine_distance(hidden_states, hidden_states[-1])

    if method == "norm_weighted_cosine":
        return batch_norm_weighted_cosine(hidden_states, hidden_states[-1])

    # --- SVD-compressed JSD (needs svd_projector kwarg) ----------------
    if method == "svd_jsd":
        # Caller must pass svd_projector via the DTRAccumulator; for the
        # generic API we fall back to building one on the fly.
        svd_proj = kwargs.get("svd_projector")
        if svd_proj is None:
            svd_proj = SVDCompressedUnembedding(lm_head_weight, k=256)
        return svd_proj.batch_jsd(hidden_states, layer_norm=layer_norm, base=base)

    # --- All other methods require vocab-space projection ----------------
    # 1. Layer-norm all hidden states.
    normed = torch.stack([layer_norm(hidden_states[i]) for i in range(num_layers)])

    # 2. Project to vocabulary space.
    logits = normed @ lm_head_weight.t()  # (num_layers, vocab_size)

    # 3. Softmax -> probability distributions.
    probs = torch.softmax(logits, dim=-1)  # (num_layers, vocab_size)
    p_final = probs[-1]  # (vocab_size,)

    if method == "jsd":
        return batch_jsd(probs, p_final, base=base)

    if method == "kld":
        # KL(p_final || p_layer) for each layer
        return torch.stack([kld(p_final, probs[i]) for i in range(num_layers)])

    if method == "reverse_kld":
        # KL(p_layer || p_final) for each layer
        return torch.stack([kld(probs[i], p_final) for i in range(num_layers)])

    if method == "wasserstein":
        if embeddings is None:
            raise ValueError(
                "The 'wasserstein' method requires the 'embeddings' parameter."
            )
        # Lazy import to avoid circular / missing dependency issues when
        # wasserstein is not needed.
        from dtr.metrics.distances import batch_wasserstein_topk  # noqa: WPS433

        return batch_wasserstein_topk(
            probs, p_final, embeddings, k=wasserstein_k
        )

    # Should never reach here due to the check at the top, but just in case.
    raise ValueError(f"Unhandled method {method!r}")  # pragma: no cover


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
# Soft (Rao-Blackwellized) DTR
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable scalar sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_dtr_soft(
    jsd_matrix: torch.Tensor,
    threshold_g: float = 0.5,
    depth_ratio_rho: float = 0.85,
    sharpness: float = 20.0,
) -> dict:
    """Rao-Blackwellized (soft) DTR using a sigmoid relaxation.

    Instead of hard binary classification of deep-thinking tokens, this
    variant passes each token's normalised settling depth through a smooth
    sigmoid centred at *depth_ratio_rho*:

    .. math::
        \\text{soft\\_deep}_t = \\sigma\\bigl(
            \\text{sharpness} \\cdot (c_t / L - \\rho)
        \\bigr)

    where *c_t* is the settling depth, *L* is the number of layers, and
    *sigma* is the logistic sigmoid.  ``DTR_soft`` is the mean of the
    soft deep values across all tokens.

    Parameters
    ----------
    jsd_matrix : torch.Tensor
        Shape ``(num_tokens, num_layers)``.
    threshold_g : float
        Settling threshold (default 0.5).
    depth_ratio_rho : float
        Centre of the sigmoid (default 0.85).
    sharpness : float
        Sharpness / temperature of the sigmoid (default 20.0).  Higher
        values make the sigmoid approach a hard step function.

    Returns
    -------
    dict
        Same keys as :func:`compute_dtr` plus ``dtr_soft`` and
        ``soft_deep_values``.
    """
    # Start from the hard DTR computation so we keep the same base results.
    result = compute_dtr(jsd_matrix, threshold_g, depth_ratio_rho)

    num_tokens, num_layers = jsd_matrix.shape
    soft_deep_values: list[float] = []
    for sd in result["settling_depths"]:
        normalised = sd / num_layers
        soft_deep_values.append(
            _sigmoid(sharpness * (normalised - depth_ratio_rho))
        )

    result["dtr_soft"] = (
        sum(soft_deep_values) / num_tokens if num_tokens > 0 else 0.0
    )
    result["soft_deep_values"] = soft_deep_values
    return result


# ---------------------------------------------------------------------------
# Continuous DTR
# ---------------------------------------------------------------------------

def compute_dtr_continuous(
    jsd_matrix: torch.Tensor,
    threshold_g: float = 0.5,
) -> dict:
    """Continuous DTR -- mean normalised settling depth (no binary threshold).

    For each token, compute the settling depth and normalise it by the number
    of layers.  ``DTR_continuous`` is the mean of these normalised depths
    across all tokens.

    Parameters
    ----------
    jsd_matrix : torch.Tensor
        Shape ``(num_tokens, num_layers)``.
    threshold_g : float
        Settling threshold used to determine settling depth (default 0.5).

    Returns
    -------
    dict
        Keys: ``dtr_continuous``, ``settling_depths``, ``normalized_depths``.
    """
    num_tokens, num_layers = jsd_matrix.shape
    settling_depths: list[int] = []
    normalized_depths: list[float] = []

    for t in range(num_tokens):
        sd = compute_settling_depth(jsd_matrix[t], threshold_g)
        settling_depths.append(sd)
        normalized_depths.append(sd / num_layers)

    dtr_continuous = (
        sum(normalized_depths) / num_tokens if num_tokens > 0 else 0.0
    )
    return {
        "dtr_continuous": dtr_continuous,
        "settling_depths": settling_depths,
        "normalized_depths": normalized_depths,
    }


# ---------------------------------------------------------------------------
# Numpy post-hoc recomputation helpers
# ---------------------------------------------------------------------------

def recompute_dtr_soft(
    jsd_matrix: np.ndarray,
    threshold_g: float,
    depth_ratio_rho: float,
    sharpness: float = 20.0,
) -> float:
    """Recompute soft DTR from a pre-computed numpy JSD matrix.

    This is the numpy analogue of :func:`compute_dtr_soft`, intended for
    post-hoc recomputation during parameter sweeps without requiring torch.

    Parameters
    ----------
    jsd_matrix : np.ndarray
        Shape ``(T, L)``.
    threshold_g : float
        Settling threshold.
    depth_ratio_rho : float
        Centre of the sigmoid.
    sharpness : float
        Sigmoid sharpness (default 20.0).

    Returns
    -------
    float
        Soft DTR value.
    """
    jsd_matrix = np.asarray(jsd_matrix, dtype=np.float64)
    num_tokens, num_layers = jsd_matrix.shape

    if num_tokens == 0:
        return 0.0

    total = 0.0
    for t in range(num_tokens):
        row = jsd_matrix[t]
        cummin = np.minimum.accumulate(row)
        settled_mask = cummin <= threshold_g
        if settled_mask.any():
            settling_depth = int(np.argmax(settled_mask)) + 1  # 1-indexed
        else:
            settling_depth = num_layers

        normalised = settling_depth / num_layers
        total += _sigmoid(sharpness * (normalised - depth_ratio_rho))

    return total / num_tokens


def recompute_dtr_continuous(
    jsd_matrix: np.ndarray,
    threshold_g: float,
) -> float:
    """Recompute continuous DTR from a pre-computed numpy JSD matrix.

    This is the numpy analogue of :func:`compute_dtr_continuous`, intended for
    post-hoc recomputation during parameter sweeps without requiring torch.

    Parameters
    ----------
    jsd_matrix : np.ndarray
        Shape ``(T, L)``.
    threshold_g : float
        Settling threshold.

    Returns
    -------
    float
        Continuous DTR value (mean normalised settling depth).
    """
    jsd_matrix = np.asarray(jsd_matrix, dtype=np.float64)
    num_tokens, num_layers = jsd_matrix.shape

    if num_tokens == 0:
        return 0.0

    total = 0.0
    for t in range(num_tokens):
        row = jsd_matrix[t]
        cummin = np.minimum.accumulate(row)
        settled_mask = cummin <= threshold_g
        if settled_mask.any():
            settling_depth = int(np.argmax(settled_mask)) + 1  # 1-indexed
        else:
            settling_depth = num_layers

        total += settling_depth / num_layers

    return total / num_tokens


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
    method : str
        Distance method passed to :func:`compute_distance_per_layer`.
        Defaults to ``"jsd"`` which uses the original
        :func:`compute_jsd_per_layer` for backward compatibility.
    compute_soft : bool
        If ``True``, :meth:`get_results` will additionally include
        ``dtr_soft`` and ``soft_deep_values`` computed via sigmoid
        relaxation.
    sharpness : float
        Sigmoid sharpness for the soft DTR computation (default 20.0).
        Only used when *compute_soft* is ``True``.
    embeddings : torch.Tensor or None
        Embedding matrix required when *method* is ``"wasserstein"``.
    wasserstein_k : int
        Top-*k* for Wasserstein approximation (default 100).
    """

    def __init__(
        self,
        num_layers: int,
        lm_head_weight: torch.Tensor,
        layer_norm: Any,
        threshold_g: float = 0.5,
        depth_ratio_rho: float = 0.85,
        method: str = "jsd",
        compute_soft: bool = False,
        sharpness: float = 20.0,
        embeddings: torch.Tensor | None = None,
        wasserstein_k: int = 100,
    ) -> None:
        self.num_layers = num_layers
        self.lm_head_weight = lm_head_weight
        self.layer_norm = layer_norm
        self.threshold_g = threshold_g
        self.depth_ratio_rho = depth_ratio_rho
        self.method = method
        self.compute_soft = compute_soft
        self.sharpness = sharpness
        self.embeddings = embeddings
        self.wasserstein_k = wasserstein_k

        # Cache lm_head_weight and layer_norm on the device that will be
        # used for hidden states (typically CUDA).  This avoids a 1.5GB+
        # CPU->GPU copy on every single token.
        self._device_cached = False

        # Pre-allocated working buffers for compute_jsd_per_layer.
        # Allocated lazily on first add_token() once we know vocab_size
        # and device.  Reusing these buffers across 32K+ calls eliminates
        # ~3 TB of CPU allocation churn that causes memory fragmentation.
        self._jsd_buffers: dict[str, torch.Tensor] | None = None

        # SVD projector for svd_jsd method (created once, reused).
        self._svd_projector: SVDCompressedUnembedding | None = None

        # Accumulated results.
        self._distance_rows: list[torch.Tensor] = []
        self._settling_depths: list[int] = []
        self._deep_thinking_mask: list[bool] = []

    # -----------------------------------------------------------------
    def _ensure_buffers(self, device: torch.device) -> None:
        """Lazily allocate reusable working buffers on first call."""
        if self._jsd_buffers is not None:
            return
        vocab_size = self.lm_head_weight.size(0)
        self._jsd_buffers = {
            "logits": torch.empty(self.num_layers, vocab_size, dtype=torch.float32, device=device),
            "probs": torch.empty(self.num_layers, vocab_size, dtype=torch.float32, device=device),
            "m": torch.empty(self.num_layers, vocab_size, dtype=torch.float32, device=device),
        }

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
            Per-token info: ``distance_values``, ``settling_depth``,
            ``is_deep``.  When *method* is ``"jsd"`` the key
            ``jsd_values`` is also included for backward compatibility.
        """
        # On first call, move lm_head_weight to hidden states' device once.
        # Do NOT move layer_norm -- it's a reference to the model's own norm
        # module and moving it breaks device_map="auto" dispatch.
        if not self._device_cached:
            device = hidden_states.device
            self.lm_head_weight = self.lm_head_weight.to(device)
            self._device_cached = True

        if self.method == "jsd":
            self._ensure_buffers(hidden_states.device)
            distance_values = compute_jsd_per_layer(
                hidden_states, self.lm_head_weight, self.layer_norm,
                _buffers=self._jsd_buffers,
            )
        elif self.method == "svd_jsd":
            if self._svd_projector is None:
                self._svd_projector = SVDCompressedUnembedding(
                    self.lm_head_weight, k=256,
                )
            distance_values = self._svd_projector.batch_jsd(
                hidden_states, layer_norm=self.layer_norm,
            )
        else:
            distance_values = compute_distance_per_layer(
                hidden_states,
                self.lm_head_weight,
                self.layer_norm,
                method=self.method,
                embeddings=self.embeddings,
                wasserstein_k=self.wasserstein_k,
            )

        sd = compute_settling_depth(distance_values, self.threshold_g)
        is_deep = is_deep_thinking_token(sd, self.num_layers, self.depth_ratio_rho)

        self._distance_rows.append(distance_values)
        self._settling_depths.append(sd)
        self._deep_thinking_mask.append(is_deep)

        result: dict[str, Any] = {
            "distance_values": distance_values,
            "settling_depth": sd,
            "is_deep": is_deep,
        }
        # Backward compatibility: also expose as ``jsd_values`` when using JSD.
        if self.method == "jsd":
            result["jsd_values"] = distance_values
        return result

    # -----------------------------------------------------------------
    def get_results(self) -> dict:
        """Return final DTR and all accumulated statistics.

        Returns
        -------
        dict
            Same schema as :func:`compute_dtr`.  When *compute_soft* was
            set to ``True`` at construction, additional keys ``dtr_soft``
            and ``soft_deep_values`` are included.
        """
        num_tokens = len(self._settling_depths)
        num_deep = sum(self._deep_thinking_mask)
        result: dict[str, Any] = {
            "dtr": num_deep / num_tokens if num_tokens > 0 else 0.0,
            "settling_depths": list(self._settling_depths),
            "deep_thinking_mask": list(self._deep_thinking_mask),
            "num_deep": num_deep,
            "total_tokens": num_tokens,
        }

        if self.compute_soft:
            soft_deep_values: list[float] = []
            for sd in self._settling_depths:
                normalised = sd / self.num_layers
                soft_deep_values.append(
                    _sigmoid(self.sharpness * (normalised - self.depth_ratio_rho))
                )
            result["dtr_soft"] = (
                sum(soft_deep_values) / num_tokens if num_tokens > 0 else 0.0
            )
            result["soft_deep_values"] = soft_deep_values

        return result


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
