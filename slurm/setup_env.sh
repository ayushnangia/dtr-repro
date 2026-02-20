#!/bin/bash
# Set up the DTR virtualenv on Alliance Canada clusters (Killarney, etc.)
#
# Run ONCE from a login node:
#   bash slurm/setup_env.sh
#
# This creates ~/dtr-env with all required packages using Alliance pre-built wheels.

set -euo pipefail

echo "=== Setting up DTR environment on $(hostname) ==="

module load python/3.11

# Create virtualenv in $HOME (persistent, accessible from all nodes)
if [[ -d ~/dtr-env ]]; then
    echo "WARNING: ~/dtr-env already exists. Remove it first to recreate."
    echo "  rm -rf ~/dtr-env"
    exit 1
fi

virtualenv --no-download ~/dtr-env
source ~/dtr-env/bin/activate

pip install --no-index --upgrade pip

# Core ML packages (Alliance pre-built wheels, optimized for cluster CUDA)
pip install --no-index torch torchvision torchaudio
pip install --no-index transformers datasets tokenizers accelerate
pip install --no-index numpy scipy pandas scikit-learn

# Packages that may not have Alliance wheels -- fall back to PyPI
pip install matplotlib seaborn tqdm rich hydra-core omegaconf

echo ""
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"

# Install the project in editable mode
pip install -e "$(dirname "$0")/.."

echo ""
echo "=== Environment ready at ~/dtr-env ==="
echo "In job scripts, activate with:"
echo "  module load python/3.11"
echo "  source ~/dtr-env/bin/activate"
