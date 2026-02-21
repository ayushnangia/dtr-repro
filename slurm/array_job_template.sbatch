#!/bin/bash
#SBATCH --job-name=dtr-gen
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#
# Generic SLURM array job template for DTR response generation on Trillium.
#
# Trillium uses whole-node scheduling. Only set GPU count per-model:
#   --gpus-per-node=h100:N  (1 for 4B, 2 for 30B, 4 for 70B)
#   --nodes=1
#
# No --mem or --cpus-per-task needed (you get the full node: 96 cores, 749 GB RAM).
# Or use the model-specific .sbatch files instead.
#
# Required environment variables:
#   MODEL       - Model name (deepseek_r1_70b, qwen3_30b, qwen3_4b)
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
#   MODEL=qwen3_4b BENCHMARK=aime_2025 sbatch --gpus-per-node=h100:1 --nodes=1 slurm/array_job_template.sbatch

set -euo pipefail

# Alliance Canada environment setup
if ! command -v module &>/dev/null; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh 2>/dev/null || true
fi
module load python/3.11 gcc arrow
source ~/dtr-env/bin/activate
export HF_HOME="${SCRATCH:-$HOME}/hf-cache"

echo "=== DTR Generation Job ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model:        ${MODEL}"
echo "Benchmark:    ${BENCHMARK}"
echo "N Samples:    ${N_SAMPLES:-25}"
echo "Seed:         ${SEED:-42}"
echo "=========================="

nvidia-smi

# Stage data to fast local storage
DATA_SRC="${DATA_DIR:-$PROJECT/dtr-data}"
if [[ -d "${DATA_SRC}" ]]; then
    echo "Staging data to \$SLURM_TMPDIR..."
    cp -r "${DATA_SRC}" "${SLURM_TMPDIR}/data"
    LOCAL_DATA="${SLURM_TMPDIR}/data"
else
    LOCAL_DATA="${DATA_SRC}"
fi

CMD="python scripts/generate_responses.py \
    --model ${MODEL} \
    --benchmark ${BENCHMARK} \
    --n_samples ${N_SAMPLES:-25} \
    --seed ${SEED:-42} \
    --question_ids ${SLURM_ARRAY_TASK_ID} \
    --output_dir ${OUTPUT_DIR:-${SCRATCH:-$HOME}/dtr-results/raw} \
    --data_dir ${LOCAL_DATA}"

if [[ "${STORE_JSD:-0}" == "1" ]]; then
    CMD="${CMD} --store_jsd_matrix"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=== Job complete ==="
