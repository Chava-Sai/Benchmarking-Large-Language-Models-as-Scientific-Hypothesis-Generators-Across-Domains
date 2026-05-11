#!/bin/bash
#SBATCH --job-name=icml_factual
#SBATCH --output=logs/factual_%j.out
#SBATCH --error=logs/factual_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com


set -e
cd $(pwd)

module load miniconda/23.11.0
module load cuda/12.5
conda activate icml2026
set -a; source .env; set +a

mkdir -p logs results/hypotheses

echo "=== Factual Evaluation ==="
echo "Start: $(date)"

# API models
for MODEL in claude_sonnet gpt4o_mini; do
    echo "Model: $MODEL at $(date)"
    python run_pipeline.py --stage factual --models "$MODEL" --batch_size 1
done

# Large local
for MODEL in llama3_70b qwen2_72b; do
    echo "Model: $MODEL at $(date)"
    python run_pipeline.py --stage factual --models "$MODEL" --batch_size 16
done

# Small local
for MODEL in llama3_8b mistral_7b qwen2_7b; do
    echo "Model: $MODEL at $(date)"
    python run_pipeline.py --stage factual --models "$MODEL" --batch_size 64
done

echo "=== Factual eval complete at $(date) ==="
