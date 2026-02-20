#!/bin/bash
#SBATCH --job-name=dtr-qwen4b
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
#SBATCH --array=0-29
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#
# Qwen3-4B generation job.
# Requires 1x A100 GPU (or any GPU with >= 16GB VRAM).
#
# Required environment variables:
#   BENCHMARK   - Benchmark name (aime_2024, aime_2025, hmmt_2025, gpqa_diamond)
#
# Optional environment variables:
#   N_SAMPLES   - Number of samples per question (default: 25)
#   SEED        - Base random seed (default: 42)
#   OUTPUT_DIR  - Output directory (default: results/raw)
#   DATA_DIR    - Data cache directory (default: data_cache)
#   STORE_JSD   - Set to 1 to save JSD matrices (default: 0)
#
# Usage:
#   BENCHMARK=aime_2025 sbatch slurm/generate_qwen3_4b.sbatch
#   BENCHMARK=gpqa_diamond N_SAMPLES=48 sbatch slurm/generate_qwen3_4b.sbatch

set -euo pipefail

MODEL=qwen3_4b

# Activate environment (adjust for your cluster)
source ~/.bashrc
conda activate dtr 2>/dev/null || true
# module load cuda/12.1  # Uncomment if using modules

echo "=== DTR Generation: Qwen3-4B ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Benchmark:    ${BENCHMARK}"
echo "N Samples:    ${N_SAMPLES:-25}"
echo "Seed:         ${SEED:-42}"
echo "================================="

nvidia-smi || echo "WARNING: nvidia-smi not available"

CMD="python scripts/generate_responses.py \
    --model ${MODEL} \
    --benchmark ${BENCHMARK} \
    --n_samples ${N_SAMPLES:-25} \
    --seed ${SEED:-42} \
    --question_ids ${SLURM_ARRAY_TASK_ID} \
    --output_dir ${OUTPUT_DIR:-results/raw} \
    --data_dir ${DATA_DIR:-data_cache}"

if [[ "${STORE_JSD:-0}" == "1" ]]; then
    CMD="${CMD} --store_jsd_matrix"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=== Job complete ==="
