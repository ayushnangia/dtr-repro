"""Download and normalize benchmark datasets for DTR reproduction.

Saves each benchmark to data_cache/{benchmark_name}.json as a list of dicts
with keys: id, question, answer, source.

Supported benchmarks:
  - aime_2024: HuggingFaceH4/aime_2024 (30 problems)
  - aime_2025: opencompass/AIME2025 (30 problems)
  - hmmt_2025: MathArena/hmmt_feb_2025 (~30 problems)
  - gpqa_diamond: Idavidrein/gpqa, diamond subset (198 questions)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALL_BENCHMARKS = ["aime_2024", "aime_2025", "hmmt_2025", "gpqa_diamond"]

CHOICE_LABELS = ["A", "B", "C", "D"]


def _normalize_aime_2024(output_dir: Path) -> None:
    """Download and normalize AIME 2024 from HuggingFaceH4/aime_2024."""
    logger.info("Downloading AIME 2024 from HuggingFaceH4/aime_2024 ...")
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

    records = []
    for idx, row in enumerate(ds):
        # Field names: 'problem' and 'answer' (integer)
        question = row.get("problem") or row.get("question", "")
        answer = str(row.get("answer", ""))
        records.append({
            "id": idx,
            "question": question.strip(),
            "answer": answer.strip(),
            "source": "aime_2024",
        })

    _save(records, output_dir / "aime_2024.json")
    logger.info("AIME 2024: saved %d problems", len(records))


def _normalize_aime_2025(output_dir: Path) -> None:
    """Download and normalize AIME 2025 from opencompass/AIME2025."""
    logger.info("Downloading AIME 2025 from opencompass/AIME2025 ...")
    ds = load_dataset("opencompass/AIME2025", split="train")

    records = []
    for idx, row in enumerate(ds):
        question = row.get("problem") or row.get("question", "")
        answer = str(row.get("answer", ""))
        records.append({
            "id": idx,
            "question": question.strip(),
            "answer": answer.strip(),
            "source": "aime_2025",
        })

    _save(records, output_dir / "aime_2025.json")
    logger.info("AIME 2025: saved %d problems", len(records))


def _normalize_hmmt_2025(output_dir: Path) -> None:
    """Download and normalize HMMT Feb 2025 from MathArena/hmmt_feb_2025."""
    logger.info("Downloading HMMT 2025 from MathArena/hmmt_feb_2025 ...")
    ds = load_dataset("MathArena/hmmt_feb_2025", split="train")

    records = []
    for idx, row in enumerate(ds):
        question = row.get("problem") or row.get("question", "")
        answer = str(row.get("answer") if row.get("answer") is not None else row.get("solution", ""))
        records.append({
            "id": idx,
            "question": question.strip(),
            "answer": answer.strip(),
            "source": "hmmt_2025",
        })

    _save(records, output_dir / "hmmt_2025.json")
    logger.info("HMMT 2025: saved %d problems", len(records))


def _normalize_gpqa_diamond(output_dir: Path) -> None:
    """Download and normalize GPQA Diamond from Idavidrein/gpqa."""
    logger.info("Downloading GPQA Diamond from Idavidrein/gpqa ...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    records = []
    for idx, row in enumerate(ds):
        question = row.get("Question", "") or row.get("question", "")

        # Build the choices list from the dataset columns
        correct = row.get("Correct Answer", "") or ""
        incorrect = []
        for key in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            val = row.get(key, "")
            if val:
                incorrect.append(val)

        # The dataset stores the full text of each choice.  We need to figure
        # out which letter (A/B/C/D) maps to the correct answer.  The choices
        # are typically shuffled per-row; reconstruct a stable ordering.
        all_choices = [correct] + incorrect
        # Maintain a deterministic order: place correct answer at its original
        # position.  The dataset does not guarantee a fixed ordering, so we
        # just store the choices and record which letter is correct.
        # Use a simple approach: choices in order, correct answer first, then
        # incorrect.  But we need to record the gold letter.
        # Actually, let's just keep the ordering and mark (A) as the correct one,
        # but that would be trivially learnable.  Instead, store all choices in
        # a canonical alphabetical order and find which letter is correct.
        all_choices_sorted = sorted(all_choices)
        correct_idx = all_choices_sorted.index(correct)
        correct_letter = CHOICE_LABELS[correct_idx]

        # Store choices in the question text for self-containment
        choices_text = "\n".join(
            f"({CHOICE_LABELS[i]}) {choice}" for i, choice in enumerate(all_choices_sorted)
        )
        full_question = f"{question.strip()}\n\n{choices_text}"

        records.append({
            "id": idx,
            "question": full_question,
            "answer": correct_letter,
            "source": "gpqa_diamond",
            "choices": all_choices_sorted,
        })

    _save(records, output_dir / "gpqa_diamond.json")
    logger.info("GPQA Diamond: saved %d questions", len(records))


def _save(records: list[dict], path: Path) -> None:
    """Save records as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


DOWNLOADERS = {
    "aime_2024": _normalize_aime_2024,
    "aime_2025": _normalize_aime_2025,
    "hmmt_2025": _normalize_hmmt_2025,
    "gpqa_diamond": _normalize_gpqa_diamond,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize benchmark datasets for DTR reproduction."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_cache",
        help="Directory to save normalized JSON files (default: data_cache)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=ALL_BENCHMARKS,
        choices=ALL_BENCHMARKS,
        help="Which benchmarks to download (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in args.benchmarks:
        try:
            DOWNLOADERS[benchmark](output_dir)
        except Exception:
            logger.exception("Failed to download %s", benchmark)
            raise

    logger.info("All done. Files saved to %s/", output_dir)


if __name__ == "__main__":
    main()
