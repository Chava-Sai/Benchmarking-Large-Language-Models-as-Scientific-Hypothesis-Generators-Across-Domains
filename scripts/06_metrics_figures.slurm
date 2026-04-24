#!/bin/bash
#SBATCH --job-name=icml_metrics
#SBATCH --output=logs/metrics_%j.out
#SBATCH --error=logs/metrics_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saichava@bu.edu
#SBATCH -P cs505am

# Compute novelty (FAISS) + plausibility (LLM judge) + aggregate + figures.
# Run after all hypothesis generation is complete.

set -e
cd /projectnb/cs505am/students/saichava/icml2026

module load miniconda/23.11.0
module load cuda/12.5
conda activate icml2026

mkdir -p logs results/metrics results/figures

echo "=== Computing Metrics ==="
echo "Start: $(date)"

# Compute novelty + plausibility + leaderboard
python run_pipeline.py --stage metrics

# Cross-domain analysis table
python -m analysis.compute_results cross_domain

# Generate all paper figures
python run_pipeline.py --stage figures

echo "=== All done at $(date) ==="
echo ""
echo "Results:"
echo "  Leaderboard CSV: results/metrics/leaderboard.csv"
echo "  Figures:         results/figures/"
echo "  Full JSON:       results/metrics/all_results.json"
