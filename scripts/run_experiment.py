"""Run a complete experiment from a config file.

Usage:
    python scripts/run_experiment.py --config configs/experiments/table1.yaml
    python scripts/run_experiment.py --config configs/experiments/table2.yaml
    python scripts/run_experiment.py --config configs/experiments/figure4.yaml
    python scripts/run_experiment.py --config configs/experiments/appendix_a.yaml
    python scripts/run_experiment.py --config configs/experiments/appendix_c.yaml

    # Override specific settings
    python scripts/run_experiment.py --config configs/experiments/table1.yaml \
        --models deepseek_r1_70b --benchmarks aime_2025 --dry_run

This orchestrator reads experiment YAML configs and dispatches to the
appropriate generation, metric computation, and aggregation modules.
It handles:
    - Table 1: Correlation analysis (DTR vs accuracy across quantile bins)
    - Table 2: Think@n comparison against baseline selection strategies
    - Table 3: Prefix length ablation for Think@n
    - Figure 4: Hyperparameter sensitivity sweep
    - Appendix A: Distance metric comparison
    - Appendix C: Think@n analysis (varying n and eta)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dtr.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a complete DTR experiment from a config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Supported experiments:
  table1         Correlation analysis (DTR vs accuracy, 2 models x 4 benchmarks)
  table2         Think@n comparison (Qwen3-4B, 6 strategies x 4 benchmarks)
  table3         Prefix length ablation (Qwen3-4B on AIME 2025)
  figure4        Hyperparameter sensitivity (g, rho sweep on GPQA-Diamond)
  appendix_a     Distance metric comparison (JSD vs KLD vs cosine)
  appendix_c     Think@n analysis (varying n and eta)

Examples:
  python scripts/run_experiment.py --config configs/experiments/table1.yaml
  python scripts/run_experiment.py --config configs/experiments/table2.yaml --dry_run
""",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Override: comma-separated model names (overrides config)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Override: comma-separated benchmark names (overrides config)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override: comma-separated base seeds (default: range from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands that would be run without executing them",
    )
    parser.add_argument(
        "--store_jsd_matrix",
        action="store_true",
        help="Pass --store_jsd_matrix to generate_responses.py (needed for Figure 4 / Appendix A)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and merge experiment config with base config."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load base config if referenced
    base_path = config_path.parent.parent / "base.yaml"
    if base_path.exists():
        with open(base_path) as f:
            base = yaml.safe_load(f)
        # Merge: experiment config overrides base
        merged = _deep_merge(base, config)
        return merged

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def detect_experiment_type(config: dict) -> str:
    """Detect experiment type from config content."""
    name = config.get("experiment", {}).get("name", "")
    if "table1" in name or "correlation" in name:
        return "table1"
    elif "table2" in name or "think_at_n" in name:
        if "prefix" in name or "ablation" in name:
            return "table3"
        return "table2"
    elif "table3" in name or "prefix_ablation" in name:
        return "table3"
    elif "figure4" in name or "sensitivity" in name:
        return "figure4"
    elif "appendix_a" in name or "distance" in name:
        return "appendix_a"
    elif "appendix_c" in name or "think_at_n_analysis" in name:
        return "appendix_c"
    return "unknown"


def run_command(cmd: list[str], dry_run: bool, logger: logging.Logger) -> int:
    """Run a command, or print it if dry_run is True."""
    cmd_str = " ".join(cmd)
    if dry_run:
        logger.info("[DRY RUN] %s", cmd_str)
        return 0
    logger.info("Running: %s", cmd_str)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Command failed with return code %d", result.returncode)
    return result.returncode


def run_generation(
    models: list[str],
    benchmarks: list[str],
    config: dict,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Run generate_responses.py for each model x benchmark combination."""
    gen_config = config.get("generation", {})
    exp_config = config.get("experiment", {})
    dtr_config = config.get("dtr", {})
    paths_config = config.get("paths", {})

    n_samples = exp_config.get("n_samples", 25)
    base_seed = exp_config.get("base_seed", 42)
    max_new_tokens = gen_config.get("max_new_tokens", 32768)
    temperature = gen_config.get("temperature", 0.6)
    top_p = gen_config.get("top_p", 0.95)
    threshold_g = dtr_config.get("threshold_g", 0.5)
    depth_ratio_rho = dtr_config.get("depth_ratio_rho", 0.85)
    output_dir = args.output_dir or paths_config.get("raw_dir", "results/raw")

    for model in models:
        for benchmark in benchmarks:
            cmd = [
                sys.executable, "scripts/generate_responses.py",
                "--model", model,
                "--benchmark", benchmark,
                "--n_samples", str(n_samples),
                "--seed", str(base_seed),
                "--max_new_tokens", str(max_new_tokens),
                "--temperature", str(temperature),
                "--top_p", str(top_p),
                "--threshold_g", str(threshold_g),
                "--depth_ratio_rho", str(depth_ratio_rho),
                "--output_dir", output_dir,
            ]
            if args.store_jsd_matrix:
                cmd.append("--store_jsd_matrix")

            run_command(cmd, args.dry_run, logger)


def run_sensitivity_sweep(
    config: dict,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Run compute_metrics.py for Figure 4 sensitivity sweep."""
    sweep_config = config.get("sweep", {})
    paths_config = config.get("paths", {})
    models = config.get("models", [])
    benchmarks = config.get("source", {}).get("benchmarks", [])

    g_values = sweep_config.get("threshold_g", [0.25, 0.5, 0.75])
    rho_values = sweep_config.get("depth_ratio_rho", [0.8, 0.85, 0.9, 0.95])
    raw_dir = paths_config.get("raw_dir", "results/raw")
    output_base = args.output_dir or paths_config.get("metrics_dir", "results/metrics")

    for model in models:
        for benchmark in benchmarks:
            input_dir = f"{raw_dir}/{model}/{benchmark}"
            output_dir = f"{output_base}/sensitivity/{model}/{benchmark}"

            cmd = [
                sys.executable, "scripts/compute_metrics.py",
                "--input_dir", input_dir,
                "--output_dir", output_dir,
                "--g_values", ",".join(str(g) for g in g_values),
                "--rho_values", ",".join(str(r) for r in rho_values),
            ]

            run_command(cmd, args.dry_run, logger)


def run_aggregation(
    config: dict,
    args: argparse.Namespace,
    logger: logging.Logger,
    experiment_type: str,
) -> None:
    """Run aggregation strategies for Table 2, Table 3, and Appendix C.

    This loads results from the raw output directory and applies the aggregation
    strategies directly via Python (no subprocess needed).
    """
    from dtr.aggregation.strategies import SampleResult, run_all_strategies, run_trials
    from dtr.utils.io import load_jsonl, save_json

    paths_config = config.get("paths", {})
    exp_config = config.get("experiment", {})
    think_config = config.get("think_at_n", {})

    raw_dir = args.output_dir or paths_config.get("raw_dir", "results/raw")
    tables_dir = paths_config.get("tables_dir", "results/tables")

    model_name = config.get("model", {}).get("name", "qwen3_4b")
    benchmarks = args.benchmarks.split(",") if args.benchmarks else config.get("benchmarks", [])
    eta = think_config.get("eta", 0.5)
    prefix_len = think_config.get("prefix_tokens", 50)
    n_trials = exp_config.get("n_trials", 10)
    n_samples = exp_config.get("n_samples", 48)

    if args.dry_run:
        logger.info("[DRY RUN] Would run %s aggregation for %s on %s", experiment_type, model_name, benchmarks)
        return

    import numpy as np

    all_results = {}
    for benchmark in benchmarks:
        result_dir = Path(raw_dir) / model_name / benchmark
        if not result_dir.exists():
            logger.warning("Results not found: %s", result_dir)
            continue

        # Load all samples
        samples = []
        for jsonl_file in sorted(result_dir.glob("question_*.jsonl")):
            records = load_jsonl(jsonl_file)
            for record in records:
                metrics = record.get("metrics", {})
                samples.append(SampleResult(
                    answer=record.get("answer_predicted"),
                    correct=record.get("correct", False),
                    token_count=record.get("token_count", 0),
                    dtr=metrics.get("dtr", 0.0),
                    self_certainty=metrics.get("self_certainty", 0.0),
                    log_prob=metrics.get("mean_log_prob", 0.0),
                ))

        if not samples:
            logger.warning("No samples found in %s", result_dir)
            continue

        logger.info("Loaded %d samples for %s/%s", len(samples), model_name, benchmark)

        if experiment_type == "table3":
            # Prefix length ablation
            prefix_lengths = config.get("think_at_n", {}).get("prefix_lengths", [50, 100, 500, 1000, 2000, -1])
            benchmark_results = {}
            for pl in prefix_lengths:
                effective_pl = pl if pl > 0 else max(s.token_count for s in samples)
                trial_results = run_trials(
                    samples, n=n_samples, n_trials=n_trials,
                    eta=eta, prefix_len=effective_pl,
                )
                benchmark_results[str(pl)] = trial_results
            all_results[benchmark] = benchmark_results
        elif experiment_type == "appendix_c":
            # Varying n and eta
            n_values = config.get("sweep", {}).get("n_values", [16, 32, 48])
            eta_values = config.get("sweep", {}).get("eta_values", [0.25, 0.5, 0.75])
            benchmark_results = {}
            for n in n_values:
                for e in eta_values:
                    key = f"n{n}_eta{e}"
                    trial_results = run_trials(
                        samples, n=n, n_trials=n_trials,
                        eta=e, prefix_len=prefix_len,
                    )
                    benchmark_results[key] = trial_results
            all_results[benchmark] = benchmark_results
        else:
            # Table 2: standard strategy comparison
            trial_results = run_trials(
                samples, n=n_samples, n_trials=n_trials,
                eta=eta, prefix_len=prefix_len,
            )
            all_results[benchmark] = trial_results

    # Save
    output_path = Path(tables_dir) / f"{experiment_type}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(all_results, output_path)
    logger.info("Saved aggregation results to %s", output_path)


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    config = load_config(args.config)
    experiment_type = detect_experiment_type(config)
    logger.info("Detected experiment type: %s", experiment_type)
    logger.info("Experiment: %s", config.get("experiment", {}).get("name", "unknown"))
    logger.info("Description: %s", config.get("experiment", {}).get("description", ""))

    # Resolve models and benchmarks (with CLI overrides)
    models = args.models.split(",") if args.models else config.get("models", [])
    benchmarks = args.benchmarks.split(",") if args.benchmarks else config.get("benchmarks", [])

    # Handle model config that may be a single model dict instead of a list
    if not models:
        model_config = config.get("model", {})
        if isinstance(model_config, dict) and "name" in model_config:
            models = [model_config["name"]]

    logger.info("Models: %s", models)
    logger.info("Benchmarks: %s", benchmarks)

    # Dispatch based on experiment type
    if experiment_type in ("table1",):
        # Table 1: Generation + correlation analysis
        logger.info("Step 1: Generating responses...")
        run_generation(models, benchmarks, config, args, logger)
        logger.info("Step 2: Correlation analysis would be run as a post-processing step.")
        logger.info("  Use notebooks or run the analysis modules directly.")

    elif experiment_type in ("table2", "table3", "appendix_c"):
        # Table 2/3/Appendix C: Generation + aggregation
        logger.info("Step 1: Generating responses...")
        run_generation(models, benchmarks, config, args, logger)
        logger.info("Step 2: Running aggregation strategies...")
        run_aggregation(config, args, logger, experiment_type)

    elif experiment_type == "figure4":
        # Figure 4: Assumes generation already done (reuses Table 1 data)
        needs_jsd = config.get("source", {}).get("reuse_from") == "table1"
        if needs_jsd:
            logger.info("Figure 4 reuses Table 1 generation data with --store_jsd_matrix.")
            logger.info("Ensure Table 1 was run with --store_jsd_matrix first.")
        logger.info("Running hyperparameter sensitivity sweep...")
        run_sensitivity_sweep(config, args, logger)

    elif experiment_type == "appendix_a":
        # Appendix A: Reuses Table 1 data, computes alternative distance metrics
        logger.info("Appendix A reuses Table 1 generation data.")
        logger.info("Distance metric comparison is a post-processing step.")
        logger.info("Ensure Table 1 was run with --store_jsd_matrix first.")
        # The actual distance metric comparison would use the analysis modules

    else:
        logger.error("Unknown experiment type: %s", experiment_type)
        logger.error("Config: %s", config.get("experiment", {}).get("name", ""))
        sys.exit(1)

    logger.info("Experiment %s complete.", experiment_type)


if __name__ == "__main__":
    main()
