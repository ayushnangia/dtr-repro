#!/bin/bash
#SBATCH --job-name=dtr-qwen4b
#SBATCH --account=def-CHANGEME
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
#SBATCH --array=0-29
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#
# Qwen3-4B generation job for Killarney.
# Requires 1x H100 GPU (4B model fits easily on a single GPU).
#
# Required environment variables:
#   BENCHMARK   - Benchmark name (aime_2024, aime_2025, hmmt_2025, gpqa_diamond)
#
# Optional environment variables:
#   N_SAMPLES   - Number of samples per question (default: 25)
#   SEED        - Base random seed (default: 42)
#   OUTPUT_DIR  - Output directory (default: $SCRATCH/dtr-results/raw)
#   DATA_DIR    - Data cache directory (default: $PROJECT/dtr-data)
#   STORE_JSD   - Set to 1 to save JSD matrices (default: 0)
#
# Usage:
#   BENCHMARK=aime_2025 sbatch slurm/generate_qwen3_4b.sbatch
#   BENCHMARK=gpqa_diamond N_SAMPLES=48 sbatch slurm/generate_qwen3_4b.sbatch

set -euo pipefail

MODEL=qwen3_4b

# Alliance Canada environment setup
module load python/3.11
source ~/dtr-env/bin/activate

echo "=== DTR Generation: Qwen3-4B ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Benchmark:    ${BENCHMARK}"
echo "N Samples:    ${N_SAMPLES:-25}"
echo "Seed:         ${SEED:-42}"
echo "================================="

nvidia-smi

# Stage data to fast local storage
DATA_SRC="${DATA_DIR:-$PROJECT/dtr-data}"
if [[ -d "${DATA_SRC}" ]]; then
    echo "Staging data to \$SLURM_TMPDIR..."
    cp -r "${DATA_SRC}" "${SLURM_TMPDIR}/data"
    LOCAL_DATA="${SLURM_TMPDIR}/data"
else
    echo "WARNING: Data dir ${DATA_SRC} not found, using as-is"
    LOCAL_DATA="${DATA_SRC}"
fi

CMD="python scripts/generate_responses.py \
    --model ${MODEL} \
    --benchmark ${BENCHMARK} \
    --n_samples ${N_SAMPLES:-25} \
    --seed ${SEED:-42} \
    --question_ids ${SLURM_ARRAY_TASK_ID} \
    --output_dir ${OUTPUT_DIR:-${SCRATCH}/dtr-results/raw} \
    --data_dir ${LOCAL_DATA}"

if [[ "${STORE_JSD:-0}" == "1" ]]; then
    CMD="${CMD} --store_jsd_matrix"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=== Job complete ==="
