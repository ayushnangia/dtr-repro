"""Sampling utilities with temperature and top-p (nucleus) sampling."""
from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
) -> int:
    """Sample next token with temperature and nucleus (top-p) sampling.

    Parameters
    ----------
    logits:
        Raw logits for a single position, shape ``(vocab_size,)``.
    temperature:
        Sampling temperature.  Values < 1 sharpen, > 1 flatten.
        If 0, greedy (argmax) decoding is used.
    top_p:
        Nucleus sampling threshold.  Only the smallest set of tokens whose
        cumulative probability exceeds *top_p* are kept.  Set to 1.0 to
        disable nucleus filtering.
    generator:
        Optional :class:`torch.Generator` for reproducible sampling.

    Returns
    -------
    int
        The sampled token id.
    """
    if logits.dim() != 1:
        raise ValueError(f"Expected 1-D logits, got shape {logits.shape}")

    # Greedy decoding
    if temperature == 0:
        return int(logits.argmax().item())

    # Apply temperature
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = torch.softmax(scaled_logits, dim=-1)

    # Nucleus (top-p) filtering
    if top_p < 1.0:
        probs = _top_p_filter(probs, top_p)

    # Sample
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(idx.item())


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out tokens outside the top-p nucleus and renormalize.

    Parameters
    ----------
    probs:
        Probability distribution, shape ``(vocab_size,)``.  Must sum to 1.
    top_p:
        Cumulative probability threshold.

    Returns
    -------
    torch.Tensor
        Filtered and renormalized probability distribution.
    """
    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative sum
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask: remove tokens with cumulative probability above threshold.
    # We shift cumulative_probs right by 1 so the token that crosses the
    # threshold is *included* (its shifted cumsum is still below top_p).
    sorted_mask = torch.zeros_like(cumulative_probs, dtype=torch.bool)
    sorted_mask[1:] = cumulative_probs[:-1] >= top_p

    # Scatter mask back to original order
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(0, sorted_indices, sorted_mask)

    # Zero out masked tokens
    probs = probs.masked_fill(mask, 0.0)

    # Renormalize
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        # Fallback: if everything was masked (shouldn't happen), put all mass
        # on the highest-probability token
        probs[sorted_indices[0]] = 1.0

    return probs


def create_generator(seed: int, device: torch.device = torch.device("cpu")) -> torch.Generator:
    """Create a seeded :class:`torch.Generator`.

    Parameters
    ----------
    seed:
        Random seed.
    device:
        Device for the generator.  Note that CUDA generators are device-specific.

    Returns
    -------
    torch.Generator
        A seeded generator instance.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
