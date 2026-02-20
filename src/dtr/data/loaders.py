"""Unified benchmark data loaders.

Loads normalized JSON files produced by scripts/download_data.py.
Each record has keys: id (int), question (str), answer (str), source (str).
"""

from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_NAMES: list[str] = [
    "aime_2024",
    "aime_2025",
    "hmmt_2025",
    "gpqa_diamond",
]


def get_benchmark_names() -> list[str]:
    """Return the list of supported benchmark names."""
    return list(BENCHMARK_NAMES)


def load_benchmark(name: str, data_dir: str = "data_cache") -> list[dict]:
    """Load a benchmark dataset from its normalized JSON file.

    Parameters
    ----------
    name:
        Benchmark name, one of :func:`get_benchmark_names`.
    data_dir:
        Directory containing the ``{name}.json`` files produced by
        ``scripts/download_data.py``.

    Returns
    -------
    list[dict]
        List of records, each with keys ``id``, ``question``, ``answer``,
        ``source``.  GPQA records also include ``choices``.

    Raises
    ------
    ValueError
        If *name* is not a recognized benchmark.
    FileNotFoundError
        If the JSON file does not exist (run download_data.py first).
    """
    if name not in BENCHMARK_NAMES:
        raise ValueError(
            f"Unknown benchmark {name!r}. Choose from: {BENCHMARK_NAMES}"
        )

    path = Path(data_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. "
            f"Run `python scripts/download_data.py --benchmarks {name}` first."
        )

    with open(path) as f:
        records: list[dict] = json.load(f)

    # Validate and coerce types
    for record in records:
        record["id"] = int(record["id"])
        record["question"] = str(record["question"])
        record["answer"] = str(record["answer"])
        record["source"] = str(record["source"])

    return records
