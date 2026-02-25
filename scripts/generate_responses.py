"""Generate model responses with hidden state capture and metric computation.

Usage:
    python scripts/generate_responses.py --model deepseek_r1_70b --benchmark aime_2025 \
        --n_samples 25 --seed 42 --question_ids 0,1,2

    Or via Hydra:
    python scripts/generate_responses.py model=deepseek_r1_70b benchmark=aime_2025
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dtr.data.loaders import load_benchmark
from dtr.data.prompts import format_prompt
from dtr.data.answer_extraction import extract_answer, check_correct
from dtr.generation.model_loader import load_model
from dtr.generation.hidden_state_generator import HiddenStateGenerator
from dtr.utils.seeding import make_sample_seed
from dtr.utils.io import save_json, save_jsonl
from dtr.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses with DTR metrics")
    parser.add_argument("--model", required=True, choices=["deepseek_r1_70b", "qwen3_30b", "qwen3_4b"])
    parser.add_argument("--benchmark", required=True, choices=["aime_2024", "aime_2025", "hmmt_2025", "gpqa_diamond"])
    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question_ids", type=str, default=None, help="Comma-separated question IDs or SLURM_ARRAY_TASK_ID")
    parser.add_argument("--data_dir", default="data_cache")
    parser.add_argument("--output_dir", default="results/raw")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--store_jsd_matrix", action="store_true", help="Save full JSD matrix for sensitivity analysis")
    parser.add_argument("--threshold_g", type=float, default=0.5)
    parser.add_argument("--depth_ratio_rho", type=float, default=0.85)
    parser.add_argument("--method", default="jsd",
                        choices=["jsd", "svd_jsd", "cosine", "norm_weighted_cosine",
                                 "kld", "reverse_kld", "wasserstein"],
                        help="Distance metric for DTR computation")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    # Resolve question IDs (support SLURM_ARRAY_TASK_ID)
    if args.question_ids is not None:
        question_ids = [int(x) for x in args.question_ids.split(",")]
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        question_ids = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    else:
        question_ids = None  # Process all

    # Load data
    questions = load_benchmark(args.benchmark, args.data_dir)
    if question_ids is not None:
        questions = [q for q in questions if q["id"] in question_ids]

    logger.info(f"Processing {len(questions)} questions, {args.n_samples} samples each")

    # Load model
    loaded_model = load_model(args.model)
    tokenizer = loaded_model.tokenizer

    # Generate
    for question in questions:
        question_results = []
        prompt_messages = format_prompt(question["question"], args.benchmark,
                                        question.get("choices"))

        # Tokenize prompt (apply chat template)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(loaded_model.device)

        for sample_idx in range(args.n_samples):
            seed = make_sample_seed(args.seed, question["id"], sample_idx)

            generator = HiddenStateGenerator(
                loaded_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
                threshold_g=args.threshold_g,
                depth_ratio_rho=args.depth_ratio_rho,
                store_jsd_matrix=args.store_jsd_matrix,
                method=args.method,
            )

            result = generator.generate(input_ids)

            # Extract and check answer
            predicted = extract_answer(result.text, args.benchmark)
            correct = check_correct(predicted, question["answer"], args.benchmark)

            record = {
                "question_id": question["id"],
                "sample_id": sample_idx,
                "seed": seed,
                "method": args.method,
                "text": result.text,
                "answer_predicted": predicted,
                "answer_gold": question["answer"],
                "correct": correct,
                "metrics": result.metrics,
                "token_count": len(result.token_ids),
            }

            # Save JSD matrix separately if requested (large file)
            if args.store_jsd_matrix and result.jsd_matrix is not None:
                jsd_path = Path(args.output_dir) / args.model / args.benchmark / f"jsd_q{question['id']}_s{sample_idx}.pt"
                jsd_path.parent.mkdir(parents=True, exist_ok=True)
                import torch
                torch.save(result.jsd_matrix, jsd_path)

            question_results.append(record)

        # Save per-question results (include method in path to avoid overwrites)
        method_suffix = f"_{args.method}" if args.method != "jsd" else ""
        out_path = Path(args.output_dir) / args.model / args.benchmark / f"question_{question['id']}{method_suffix}.jsonl"
        save_jsonl(question_results, out_path)
        logger.info(f"Saved {len(question_results)} results to {out_path}")


if __name__ == "__main__":
    main()
