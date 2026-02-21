"""Unified model loading for dense and MoE architectures."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry with architecture configs
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "deepseek_r1_70b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "local_path": "/model-weights/DeepSeek-R1-Distill-Llama-70B",
        "num_layers": 80,
        "hidden_dim": 8192,
        "is_moe": False,
        "gpus_needed": 4,
    },
    "qwen3_30b": {
        "hf_id": "Qwen/Qwen3-30B-A3B",
        "local_path": "/model-weights/Qwen3-30B-A3B",
        "num_layers": 48,
        "hidden_dim": 2048,
        "is_moe": True,
        "gpus_needed": 2,
    },
    "qwen3_4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "local_path": "/model-weights/Qwen3-4B",
        "num_layers": 36,
        "hidden_dim": 2560,
        "is_moe": False,
        "gpus_needed": 1,
    },
}

# Common attribute paths for lm_head weight and final layer norm across
# different HuggingFace model architectures.
_LM_HEAD_PATHS = [
    "lm_head",           # most architectures (Llama, Qwen2, Mistral, ...)
    "output",            # some older architectures
]

_FINAL_NORM_PATHS = [
    "model.norm",        # LlamaForCausalLM, Qwen2ForCausalLM, Qwen3MoeForCausalLM
    "transformer.ln_f",  # GPT-2 style
    "model.final_layernorm",  # some Falcon variants
    "model.layer_norm",  # generic fallback
]


class LoadedModel:
    """Container for loaded model components needed for DTR computation.

    Attributes
    ----------
    model:
        The HuggingFace causal-LM model (in eval mode).
    tokenizer:
        The corresponding tokenizer.
    lm_head_weight:
        The language-model head weight matrix, shape ``(vocab_size, hidden_dim)``.
        Stored as a detached CPU tensor for DTR metric computation.
    final_layer_norm:
        A callable that applies the model's final layer normalization.  When
        called with a tensor of shape ``(..., hidden_dim)`` it returns a tensor
        of the same shape.
    config:
        The registry dict for this model (from :data:`MODEL_REGISTRY`).
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        lm_head_weight: torch.Tensor,
        final_layer_norm: Any,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lm_head_weight = lm_head_weight  # (vocab_size, hidden_dim)
        self.final_layer_norm = final_layer_norm  # callable
        self.config = config

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self.config["num_layers"]

    @property
    def device(self) -> torch.device:
        """Device of the underlying model parameters."""
        return next(self.model.parameters()).device


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_attr(obj: Any, dotted_path: str) -> Any:
    """Resolve a dot-separated attribute path on *obj*.

    Raises :class:`AttributeError` if any component is missing.
    """
    current = obj
    for part in dotted_path.split("."):
        current = getattr(current, part)
    return current


def _extract_lm_head_weight(model: AutoModelForCausalLM) -> torch.Tensor:
    """Extract the lm_head weight matrix from *model*.

    Tries several known attribute paths.  Falls back to checking whether the
    model ties word embeddings (in which case `lm_head.weight` is shared with
    the input embedding).

    Returns
    -------
    torch.Tensor
        Weight matrix of shape ``(vocab_size, hidden_dim)``, detached and on
        CPU in float32.
    """
    for path in _LM_HEAD_PATHS:
        try:
            module = _resolve_attr(model, path)
            weight = module.weight
            logger.info("Found lm_head weight at '%s' with shape %s", path, weight.shape)
            return weight.detach().float().cpu()
        except AttributeError:
            continue

    # Fallback: try the input embedding (weight tying)
    try:
        embed = _resolve_attr(model, "model.embed_tokens")
        weight = embed.weight
        logger.warning(
            "Using tied embedding weight as lm_head (shape %s). "
            "Verify that this model uses weight tying.",
            weight.shape,
        )
        return weight.detach().float().cpu()
    except AttributeError:
        pass

    raise RuntimeError(
        "Could not locate lm_head weight.  Tried paths: "
        f"{_LM_HEAD_PATHS} and embed_tokens fallback."
    )


def _extract_final_layer_norm(model: AutoModelForCausalLM) -> Any:
    """Extract the final layer normalization module from *model*.

    The returned object is callable: ``norm(x) -> x_normalized``.

    Returns
    -------
    callable
        The layer-norm module (e.g. ``RMSNorm``).
    """
    for path in _FINAL_NORM_PATHS:
        try:
            norm = _resolve_attr(model, path)
            # Quick sanity check: it should be callable (nn.Module)
            if callable(norm):
                logger.info("Found final layer norm at '%s'", path)
                return norm
        except AttributeError:
            continue

    raise RuntimeError(
        "Could not locate final layer norm.  Tried paths: "
        f"{_FINAL_NORM_PATHS}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> LoadedModel:
    """Load a model, tokenizer, and extract components for DTR computation.

    Handles both dense and MoE architectures.  Uses ``accelerate``
    ``device_map="auto"`` for multi-GPU distribution.

    Parameters
    ----------
    model_name:
        Key into :data:`MODEL_REGISTRY` (e.g. ``"deepseek_r1_70b"``).
    dtype:
        Torch dtype for model weights.  Default ``bfloat16``.
    device_map:
        Accelerate device-map strategy.  ``"auto"`` distributes across
        available GPUs.  Use ``"cpu"`` for testing without a GPU.

    Returns
    -------
    LoadedModel
        Container with model, tokenizer, lm_head weight, layer norm, and
        config metadata.

    Raises
    ------
    ValueError
        If *model_name* is not in :data:`MODEL_REGISTRY`.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {model_name!r}.  "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_name]
    hf_id = config["hf_id"]

    # Use local path if available (e.g. /model-weights/ on cluster), else HF hub
    local_path = config.get("local_path")
    model_path = local_path if local_path and Path(local_path).exists() else hf_id
    logger.info(
        "Loading tokenizer for %s (path=%s, source=%s)...",
        model_name, model_path, "local" if model_path == local_path else "hub",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # Ensure pad token is set (many models lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(
        "Loading model %s (%s) with dtype=%s, device_map=%s...",
        model_name, model_path, dtype, device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="flash_attention_2" if dtype in (torch.float16, torch.bfloat16) else None,
    )
    model.eval()

    # Extract DTR-critical components
    lm_head_weight = _extract_lm_head_weight(model)
    final_layer_norm = _extract_final_layer_norm(model)

    logger.info(
        "Model loaded successfully.  num_layers=%d, hidden_dim=%d, "
        "lm_head shape=%s, is_moe=%s",
        config["num_layers"],
        config["hidden_dim"],
        lm_head_weight.shape,
        config["is_moe"],
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        lm_head_weight=lm_head_weight,
        final_layer_norm=final_layer_norm,
        config=config,
    )
