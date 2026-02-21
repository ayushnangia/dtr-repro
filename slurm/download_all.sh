#!/bin/bash
# Download all data and models on an Alliance Canada login node.
#
# IMPORTANT: Run this from a login node (compute nodes may not have internet).
#
# Usage:
#   bash slurm/download_all.sh                           # Download everything
#   bash slurm/download_all.sh --data-only               # Just benchmarks
#   bash slurm/download_all.sh --models qwen3_4b         # Just one model
#
# Data is saved to $PROJECT/dtr-data/ (persistent, backed up).
# Model weights go to the default HuggingFace cache (~/.cache/huggingface/).

set -euo pipefail

DATA_ONLY=0
MODELS_ONLY=0
SELECTED_MODELS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-only)   DATA_ONLY=1; shift ;;
        --models-only) MODELS_ONLY=1; shift ;;
        --models)      SELECTED_MODELS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--data-only] [--models-only] [--models name1,name2]"
            echo ""
            echo "Downloads benchmark data to \$PROJECT/dtr-data/"
            echo "Downloads model weights to HuggingFace cache"
            echo ""
            echo "Run from a login node (compute nodes have no internet)."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Activate environment
if ! command -v module &>/dev/null; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh 2>/dev/null || true
fi
module load python/3.11 git-lfs/3.4.0
source ~/dtr-env/bin/activate

# Load HF token for gated model access (e.g. from ~/.env)
if [[ -f ~/.env ]]; then
    source ~/.env 2>/dev/null || true
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    echo "HF_TOKEN loaded for gated model access."
fi

# Disable xet transport (avoids hf_xet dependency issues on clusters)
export HF_HUB_DISABLE_XET=1

# Store model weights outside $HOME (50GB quota).
# Prefer $PROJECT > $SCRATCH > $HOME/hf-cache
export HF_HOME="${PROJECT:-${SCRATCH:-$HOME}}/hf-cache"
mkdir -p "${HF_HOME}"

echo "=== DTR: Downloading data and models ==="
echo "Hostname: $(hostname)"
echo "PROJECT:  ${PROJECT:-not set}"
echo "HF_HOME:  ${HF_HOME}"
echo ""

# Download benchmark data
if [[ ${MODELS_ONLY} -eq 0 ]]; then
    DATA_DIR="${PROJECT}/dtr-data"
    echo "--- Downloading benchmarks to ${DATA_DIR} ---"
    python scripts/download_data.py --output_dir "${DATA_DIR}"
    echo ""
fi

# Download model weights
if [[ ${DATA_ONLY} -eq 0 ]]; then
    echo "--- Downloading model weights ---"
    if [[ -n "${SELECTED_MODELS}" ]]; then
        python scripts/download_models.py --models "${SELECTED_MODELS}" --cache_dir "${HF_HOME}"
    else
        python scripts/download_models.py --cache_dir "${HF_HOME}"
    fi
    echo ""
fi

echo "=== Downloads complete ==="
echo ""
echo "Next steps:"
echo "  1. Replace 'your-account' in slurm/*.sbatch with your account"
echo "  2. Submit jobs: ./slurm/submit_all.sh --experiment table1 --dry_run"
