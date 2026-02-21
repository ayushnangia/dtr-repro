"""Pre-download model weights to local cache.

Usage:
    # Download all models
    python scripts/download_models.py --models deepseek_r1_70b,qwen3_30b,qwen3_4b

    # Download a single model
    python scripts/download_models.py --models qwen3_4b

    # Download to a specific cache directory
    python scripts/download_models.py --models qwen3_4b --cache_dir /scratch/models

    # List available models
    python scripts/download_models.py --list

This script uses transformers snapshot_download to pre-fetch model weights
before running experiments. This is especially useful on SLURM clusters where
compute nodes may not have internet access.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mirror of MODEL_REGISTRY from dtr.generation.model_loader
MODEL_INFO = {
    "deepseek_r1_70b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "description": "DeepSeek-R1-Distill-Llama-70B (4x H100-80GB)",
    },
    "qwen3_30b": {
        "hf_id": "Qwen/Qwen3-30B-A3B",
        "description": "Qwen3-30B-A3B MoE (2x H100-80GB)",
    },
    "qwen3_4b": {
        "hf_id": "Qwen/Qwen3-4B-Thinking-2507",
        "description": "Qwen3-4B-Thinking-2507 dense (1x L40S-48GB or H100)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download model weights to local cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Available models:
  deepseek_r1_70b   DeepSeek-R1-Distill-Llama-70B  (4x H100-80GB)
  qwen3_30b         Qwen3-30B-A3B MoE              (2x H100-80GB)
  qwen3_4b          Qwen3-4B dense                  (1x L40S-48GB or H100)

Examples:
  python scripts/download_models.py --models deepseek_r1_70b,qwen3_30b,qwen3_4b
  python scripts/download_models.py --models qwen3_4b --cache_dir /scratch/models
  python scripts/download_models.py --list
""",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to download (default: all)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom cache directory for model weights (default: HuggingFace default)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (for gated models)",
    )
    return parser.parse_args()


def list_models() -> None:
    """Print available models and their details."""
    print("\nAvailable models for DTR experiments:")
    print("-" * 60)
    for name, info in MODEL_INFO.items():
        print(f"  {name:<20s} {info['description']}")
        print(f"  {'':20s} HuggingFace ID: {info['hf_id']}")
        print()


def download_model(
    model_name: str,
    cache_dir: str | None = None,
    token: str | None = None,
) -> None:
    """Download a single model's weights using snapshot_download."""
    from huggingface_hub import snapshot_download

    if model_name not in MODEL_INFO:
        available = ", ".join(sorted(MODEL_INFO.keys()))
        logger.error("Unknown model %r. Available: %s", model_name, available)
        return

    info = MODEL_INFO[model_name]
    hf_id = info["hf_id"]

    logger.info("Downloading %s (%s)...", model_name, hf_id)

    kwargs = {
        "repo_id": hf_id,
        "repo_type": "model",
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if token is not None:
        kwargs["token"] = token

    try:
        local_path = snapshot_download(**kwargs)
        logger.info("Downloaded %s to %s", model_name, local_path)
    except Exception as e:
        logger.error("Failed to download %s: %s", model_name, e)
        raise

    # Also download the tokenizer (usually included, but be explicit)
    from transformers import AutoTokenizer

    logger.info("Verifying tokenizer for %s...", model_name)
    try:
        tokenizer_kwargs = {"trust_remote_code": True}
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir
        if token is not None:
            tokenizer_kwargs["token"] = token
        tokenizer = AutoTokenizer.from_pretrained(hf_id, **tokenizer_kwargs)
        logger.info("Tokenizer loaded successfully (vocab_size=%d)", tokenizer.vocab_size)
    except Exception as e:
        logger.warning("Tokenizer verification failed for %s: %s", model_name, e)


def main() -> None:
    args = parse_args()

    if args.list:
        list_models()
        return

    if args.models is not None:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(MODEL_INFO.keys())

    logger.info("Will download %d model(s): %s", len(model_names), model_names)

    for model_name in model_names:
        download_model(model_name, cache_dir=args.cache_dir, token=args.token)

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
