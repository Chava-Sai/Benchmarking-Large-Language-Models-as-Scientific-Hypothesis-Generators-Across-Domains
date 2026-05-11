#!/bin/bash
#SBATCH --job-name=icml_data
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com


set -e
cd $(pwd)

module load miniconda/23.11.0
module load cuda/12.5
conda activate icml2026

mkdir -p logs data/raw data/processed data/finetune data/embeddings

echo "=== Downloading datasets ==="
python -m data.download_datasets

echo "=== Preprocessing ==="
python -m data.preprocess

echo "=== Done ==="
