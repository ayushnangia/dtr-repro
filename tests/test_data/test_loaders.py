"""Tests for dtr.data.loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dtr.data.loaders import get_benchmark_names, load_benchmark


# ---------------------------------------------------------------------------
# get_benchmark_names
# ---------------------------------------------------------------------------


def test_get_benchmark_names_returns_list():
    names = get_benchmark_names()
    assert isinstance(names, list)
    assert len(names) == 4


def test_get_benchmark_names_contents():
    names = get_benchmark_names()
    assert "aime_2024" in names
    assert "aime_2025" in names
    assert "hmmt_2025" in names
    assert "gpqa_diamond" in names


def test_get_benchmark_names_returns_copy():
    """Modifying the returned list should not affect the module constant."""
    names = get_benchmark_names()
    names.append("bogus")
    assert "bogus" not in get_benchmark_names()


# ---------------------------------------------------------------------------
# load_benchmark -- invalid inputs
# ---------------------------------------------------------------------------


def test_load_benchmark_unknown_name():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark("not_a_real_benchmark")


def test_load_benchmark_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
        load_benchmark("aime_2024", data_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# load_benchmark -- with mock data
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with mock benchmark files."""
    # AIME 2024
    aime_2024 = [
        {"id": 0, "question": "What is 2+2?", "answer": "4", "source": "aime_2024"},
        {"id": 1, "question": "What is 3+3?", "answer": "6", "source": "aime_2024"},
    ]
    (tmp_path / "aime_2024.json").write_text(json.dumps(aime_2024))

    # AIME 2025
    aime_2025 = [
        {"id": 0, "question": "Solve x^2=9", "answer": "3", "source": "aime_2025"},
    ]
    (tmp_path / "aime_2025.json").write_text(json.dumps(aime_2025))

    # HMMT 2025
    hmmt_2025 = [
        {"id": 0, "question": "Compute 10!", "answer": "3628800", "source": "hmmt_2025"},
    ]
    (tmp_path / "hmmt_2025.json").write_text(json.dumps(hmmt_2025))

    # GPQA Diamond
    gpqa = [
        {
            "id": 0,
            "question": "What is the capital of France?\n\n(A) Berlin\n(B) London\n(C) Paris\n(D) Rome",
            "answer": "C",
            "source": "gpqa_diamond",
            "choices": ["Berlin", "London", "Paris", "Rome"],
        },
    ]
    (tmp_path / "gpqa_diamond.json").write_text(json.dumps(gpqa))

    return tmp_path


def test_load_aime_2024(mock_data_dir: Path):
    records = load_benchmark("aime_2024", data_dir=str(mock_data_dir))
    assert len(records) == 2
    assert records[0]["id"] == 0
    assert records[0]["question"] == "What is 2+2?"
    assert records[0]["answer"] == "4"
    assert records[0]["source"] == "aime_2024"


def test_load_aime_2025(mock_data_dir: Path):
    records = load_benchmark("aime_2025", data_dir=str(mock_data_dir))
    assert len(records) == 1
    assert records[0]["answer"] == "3"


def test_load_hmmt_2025(mock_data_dir: Path):
    records = load_benchmark("hmmt_2025", data_dir=str(mock_data_dir))
    assert len(records) == 1
    assert records[0]["answer"] == "3628800"


def test_load_gpqa_diamond(mock_data_dir: Path):
    records = load_benchmark("gpqa_diamond", data_dir=str(mock_data_dir))
    assert len(records) == 1
    assert records[0]["answer"] == "C"
    assert records[0]["source"] == "gpqa_diamond"
    assert "choices" in records[0]


def test_record_types_are_coerced(tmp_path: Path):
    """Ensure id is coerced to int even if stored as string in JSON."""
    data = [
        {"id": "5", "question": "Q?", "answer": "42", "source": "aime_2024"},
    ]
    (tmp_path / "aime_2024.json").write_text(json.dumps(data))

    records = load_benchmark("aime_2024", data_dir=str(tmp_path))
    assert records[0]["id"] == 5
    assert isinstance(records[0]["id"], int)
