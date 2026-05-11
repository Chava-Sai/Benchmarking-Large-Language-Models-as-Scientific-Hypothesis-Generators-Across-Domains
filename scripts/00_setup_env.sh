#!/bin/bash
# One-time setup. Run from VS Code terminal or SSH terminal on SCC.
# Usage: bash scripts/00_setup_env.sh

set -e

PROJECT_DIR="$(pwd)"
CONDA_ENV="icml2026"

echo "=== Setting up environment in $PROJECT_DIR ==="

# Load modules available on HPC cluster
module load miniconda/23.11.0
module load cuda/12.5

# Create conda environment with Python 3.12
conda create -y -n "$CONDA_ENV" python=3.12
conda activate "$CONDA_ENV"

# Install all dependencies
pip install --upgrade pip wheel setuptools
pip install -r "$PROJECT_DIR/requirements.txt"

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

echo ""
echo "=== Done! Activate with: conda activate $CONDA_ENV ==="
