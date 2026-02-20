"""Recompute DTR with different hyperparameters from saved JSD matrices.

Usage:
    python scripts/compute_metrics.py \
        --input_dir results/raw/deepseek_r1_70b/gpqa_diamond \
        --output_dir results/metrics/sensitivity \
        --g_values 0.25,0.5,0.75 \
        --rho_values 0.8,0.85,0.9,0.95

This script loads pre-saved .pt JSD matrices (produced by generate_responses.py
with --store_jsd_matrix) and recomputes DTR for each combination of threshold
g and depth ratio rho.  This avoids re-running expensive generation for
hyperparameter sensitivity analysis (Figure 4 in the paper).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dtr.metrics.dtr import compute_dtr
from dtr.utils.io import save_json, save_jsonl, load_jsonl
from dtr.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute DTR with different hyperparameters from saved JSD matrices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Sweep g and rho on DeepSeek-R1-70B / GPQA Diamond results
  python scripts/compute_metrics.py \\
      --input_dir results/raw/deepseek_r1_70b/gpqa_diamond \\
      --output_dir results/metrics/sensitivity/deepseek_r1_70b/gpqa_diamond \\
      --g_values 0.25,0.5,0.75 \\
      --rho_values 0.8,0.85,0.9,0.95

  # Single hyperparameter set
  python scripts/compute_metrics.py \\
      --input_dir results/raw/qwen3_30b/aime_2025 \\
      --output_dir results/metrics/recomputed \\
      --g_values 0.5 --rho_values 0.85
""",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .pt JSD matrix files (from generate_responses.py --store_jsd_matrix)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write recomputed DTR results",
    )
    parser.add_argument(
        "--g_values",
        type=str,
        default="0.25,0.5,0.75",
        help="Comma-separated JSD threshold values to sweep (default: 0.25,0.5,0.75)",
    )
    parser.add_argument(
        "--rho_values",
        type=str,
        default="0.8,0.85,0.9,0.95",
        help="Comma-separated depth ratio values to sweep (default: 0.8,0.85,0.9,0.95)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Optional directory containing .jsonl result files (to attach correctness labels)",
    )
    return parser.parse_args()


def find_jsd_files(input_dir: Path) -> list[Path]:
    """Find all .pt JSD matrix files in the input directory."""
    files = sorted(input_dir.glob("jsd_q*_s*.pt"))
    return files


def parse_jsd_filename(path: Path) -> tuple[int, int]:
    """Parse question_id and sample_id from a JSD filename like jsd_q5_s3.pt."""
    stem = path.stem  # e.g., "jsd_q5_s3"
    parts = stem.split("_")
    question_id = int(parts[1][1:])  # "q5" -> 5
    sample_id = int(parts[2][1:])    # "s3" -> 3
    return question_id, sample_id


def sweep_single_matrix(
    jsd_matrix: torch.Tensor,
    g_values: list[float],
    rho_values: list[float],
) -> list[dict]:
    """Compute DTR for all (g, rho) combinations on a single JSD matrix."""
    results = []
    for g in g_values:
        for rho in rho_values:
            dtr_result = compute_dtr(jsd_matrix, threshold_g=g, depth_ratio_rho=rho)
            results.append({
                "threshold_g": g,
                "depth_ratio_rho": rho,
                "dtr": dtr_result["dtr"],
                "num_deep": dtr_result["num_deep"],
                "total_tokens": dtr_result["total_tokens"],
            })
    return results


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g_values = [float(x) for x in args.g_values.split(",")]
    rho_values = [float(x) for x in args.rho_values.split(",")]

    logger.info("Sweeping g=%s, rho=%s", g_values, rho_values)

    # Find all JSD matrix files
    jsd_files = find_jsd_files(input_dir)
    if not jsd_files:
        logger.error("No .pt JSD matrix files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d JSD matrix files in %s", len(jsd_files), input_dir)

    # Load optional results for correctness labels
    results_lookup: dict[tuple[int, int], dict] = {}
    results_dir = Path(args.results_dir) if args.results_dir else input_dir
    for jsonl_path in results_dir.glob("question_*.jsonl"):
        try:
            records = load_jsonl(jsonl_path)
            for record in records:
                key = (record["question_id"], record["sample_id"])
                results_lookup[key] = record
        except Exception:
            pass

    # Process each JSD matrix
    all_sweep_results = []
    for jsd_path in jsd_files:
        question_id, sample_id = parse_jsd_filename(jsd_path)
        logger.info("Processing q%d_s%d: %s", question_id, sample_id, jsd_path.name)

        jsd_matrix = torch.load(jsd_path, map_location="cpu", weights_only=True)
        sweep_results = sweep_single_matrix(jsd_matrix, g_values, rho_values)

        # Attach metadata
        original = results_lookup.get((question_id, sample_id), {})
        for r in sweep_results:
            r["question_id"] = question_id
            r["sample_id"] = sample_id
            r["correct"] = original.get("correct")
            r["token_count"] = original.get("token_count")

        all_sweep_results.extend(sweep_results)

    # Save results
    out_path = output_dir / "sensitivity_sweep.jsonl"
    save_jsonl(all_sweep_results, out_path)
    logger.info("Saved %d sweep results to %s", len(all_sweep_results), out_path)

    # Also save a summary grouped by (g, rho)
    summary = {}
    for r in all_sweep_results:
        key = (r["threshold_g"], r["depth_ratio_rho"])
        if key not in summary:
            summary[key] = {"threshold_g": key[0], "depth_ratio_rho": key[1], "dtr_values": [], "correct_values": []}
        summary[key]["dtr_values"].append(r["dtr"])
        if r["correct"] is not None:
            summary[key]["correct_values"].append(r["correct"])

    summary_records = []
    for key, data in sorted(summary.items()):
        dtr_vals = data["dtr_values"]
        correct_vals = data["correct_values"]
        summary_records.append({
            "threshold_g": data["threshold_g"],
            "depth_ratio_rho": data["depth_ratio_rho"],
            "mean_dtr": sum(dtr_vals) / len(dtr_vals) if dtr_vals else 0.0,
            "n_samples": len(dtr_vals),
            "accuracy": sum(correct_vals) / len(correct_vals) if correct_vals else None,
        })

    summary_path = output_dir / "sensitivity_summary.json"
    save_json(summary_records, summary_path)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
