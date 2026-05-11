#!/bin/bash
#SBATCH --job-name=icml_ft
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com


# LoRA fine-tuning of LLaMA-3.1-8B on (abstract, hypothesis) pairs.
# Runs in bf16 with Flash Attention 2.
# Estimated time: ~4-6 hours for 3 epochs on 5000 samples.

set -e
cd $(pwd)

module load miniconda/23.11.0
module load cuda/12.5
conda activate icml2026

mkdir -p logs checkpoints

echo "=== LoRA Fine-tuning ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

# W&B login (set WANDB_API_KEY env var or run wandb login first)
export WANDB_PROJECT="icml2026-hypothesis"

python run_pipeline.py --stage finetune

echo "=== Fine-tuning complete at $(date) ==="

# After fine-tuning, enable the model in config and run eval
echo "=== Enabling fine-tuned model in config ==="
python -c "
import yaml
with open('config/models.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['models']['llama3_8b_finetuned']['enabled'] = True
with open('config/models.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Fine-tuned model enabled.')
"

echo "=== Now run: sbatch scripts/05_eval_finetuned.slurm ==="
