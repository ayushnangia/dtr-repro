"""I/O utilities for saving/loading generation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Save records as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from JSON Lines file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)
