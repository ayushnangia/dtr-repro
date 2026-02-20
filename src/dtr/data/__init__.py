"""Dataset loaders, prompt templates, and answer extraction."""

from __future__ import annotations

from dtr.data.answer_extraction import (
    check_correct,
    extract_answer,
    extract_boxed_answer,
    extract_choice_answer,
)
from dtr.data.loaders import get_benchmark_names, load_benchmark
from dtr.data.prompts import format_prompt

__all__ = [
    "check_correct",
    "extract_answer",
    "extract_boxed_answer",
    "extract_choice_answer",
    "format_prompt",
    "get_benchmark_names",
    "load_benchmark",
]
