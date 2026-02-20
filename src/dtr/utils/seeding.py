"""Deterministic seeding utilities."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sample_seed(base_seed: int, question_id: int, sample_id: int) -> int:
    """Create a deterministic seed from (base_seed, question_id, sample_id).

    Uses hashing to ensure uniform distribution of seeds.
    """
    key = f"{base_seed}:{question_id}:{sample_id}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h, 16) % (2**32)
