#!/bin/bash
#SBATCH --job-name=dtr-qwen30b
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
#SBATCH --array=0-29
#SBATCH --gpus-per-node=h100:2
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#
# Qwen3-30B-A3B (MoE) generation job for Fir.
# Requires 2x H100-80GB GPUs for the MoE model (~3B active parameters).
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
#   MODE        - Generation mode: streaming or posthoc (default: streaming)
#
# Usage:
#   BENCHMARK=aime_2025 sbatch slurm/generate_qwen3_30b.sbatch
#   BENCHMARK=gpqa_diamond STORE_JSD=1 sbatch slurm/generate_qwen3_30b.sbatch

set -euo pipefail

MODEL=qwen3_30b

# Alliance Canada environment setup
if ! command -v module &>/dev/null; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh 2>/dev/null || true
fi
module load python/3.11 gcc arrow
source ~/dtr-env/bin/activate
export HF_HOME="${SCRATCH:-$HOME}/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== DTR Generation: Qwen3-30B-A3B (MoE) ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Benchmark:    ${BENCHMARK}"
echo "N Samples:    ${N_SAMPLES:-25}"
echo "Seed:         ${SEED:-42}"
echo "Mode:         ${MODE:-streaming}"
echo "=============================================="

nvidia-smi

# Stage data to fast local storage
DATA_SRC="${DATA_DIR:-${SCRATCH}/dtr-data}"
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
    --output_dir ${OUTPUT_DIR:-${SCRATCH:-$HOME}/dtr-results/raw} \
    --data_dir ${LOCAL_DATA} \
    --mode ${MODE:-streaming}"

if [[ "${STORE_JSD:-0}" == "1" ]]; then
    CMD="${CMD} --store_jsd_matrix"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=== Job complete ==="
