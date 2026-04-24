#!/bin/bash
#SBATCH --job-name=icml_gen
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saichava@bu.edu
#SBATCH -P cs505am

# Hypothesis generation: all 7 models x 3 strategies.
# API models run first (fast, no GPU), then local models sequentially.
# Estimated total: ~8-9 hours.

set -e
cd /projectnb/cs505am/students/saichava/icml2026

module load miniconda/23.11.0
module load cuda/12.5
conda activate icml2026

# Load API keys from .env
set -a; source .env; set +a

mkdir -p logs results/hypotheses

echo "=== Hypothesis Generation ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

STRATEGIES="zero_shot,few_shot,cot"

# --- API models first (fast, ~10-15 min each) ---
echo "--- API Models ---"
for MODEL in claude_sonnet gpt4o_mini; do
    echo "Model: $MODEL at $(date)"
    python run_pipeline.py --stage generate --models "$MODEL" --strategies "$STRATEGIES" --batch_size 1
done

# --- Large local models (70B, AWQ, ~2-3 hrs each) ---
echo "--- Large Local Models ---"
for MODEL in llama3_70b qwen2_72b; do
    echo "Model: $MODEL at $(date)"
    python run_pipeline.py --stage generate --models "$MODEL" --strategies "$STRATEGIES" --batch_size 16
done

# --- Small local models (7-8B, ~30-45 min each) ---
echo "--- Small Local Models ---"
for MODEL in llama3_8b mistral_7b qwen2_7b; do
    echo "Model: $MODEL at $(date)"
    python run_pipeline.py --stage generate --models "$MODEL" --strategies "$STRATEGIES" --batch_size 32
done

echo "=== All generation complete at $(date) ==="
