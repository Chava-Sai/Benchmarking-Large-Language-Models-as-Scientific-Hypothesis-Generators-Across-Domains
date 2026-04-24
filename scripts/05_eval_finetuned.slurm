#!/bin/bash
#SBATCH --job-name=icml_eval_ft
#SBATCH --output=logs/eval_ft_%j.out
#SBATCH --error=logs/eval_ft_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saichava@bu.edu
#SBATCH -P cs505am

# Evaluate the fine-tuned model (hypothesis gen + factual).
# Runs after 04_finetune.slurm completes.

set -e
cd /projectnb/cs505am/students/saichava/icml2026

module load miniconda/23.11.0
module load cuda/12.5
conda activate icml2026

echo "=== Evaluating Fine-tuned Model ==="
echo "Start: $(date)"

# Generate hypotheses with fine-tuned model
python run_pipeline.py --stage generate \
    --models llama3_8b_finetuned \
    --strategies "zero_shot,few_shot,cot" \
    --batch_size 32

# Factual eval
python run_pipeline.py --stage factual \
    --models llama3_8b_finetuned \
    --batch_size 64

echo "=== Fine-tuned model eval complete at $(date) ==="
echo "=== Run sbatch scripts/06_metrics_figures.slurm next ==="
