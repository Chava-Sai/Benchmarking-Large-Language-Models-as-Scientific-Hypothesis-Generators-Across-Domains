# Can LLMs Generate Scientific Hypotheses? A Benchmark for Autonomous Discovery

**ICML 2026 AI for Science Workshop** | Seoul, South Korea | Jul 10 2026

## Setup

```bash
# 1. Clone / copy to SCC
bash deploy_to_scc.sh YOUR_BU_USERNAME

# 2. On SCC — one-time environment setup
bash scripts/00_setup_env.sh

# 3. Copy .env.example to .env and fill in tokens
cp .env.example .env
```

## Running the Full Pipeline

```bash
# Step 1: Download & preprocess datasets (~2 hrs)
sbatch scripts/01_download_data.slurm

# Step 2: Hypothesis generation — all 5 models, 3 strategies (~6-7 hrs)
sbatch scripts/02_generate_hypotheses.slurm

# Step 3: Factual QA evaluation — all models (~3 hrs)
sbatch scripts/03_factual_eval.slurm

# Step 4: LoRA fine-tuning of LLaMA-3.1-8B (~5 hrs)
sbatch scripts/04_finetune.slurm

# Step 5: Evaluate fine-tuned model (~2 hrs)
sbatch scripts/05_eval_finetuned.slurm

# Step 6: Compute metrics + generate figures (~1 hr)
sbatch scripts/06_metrics_figures.slurm
```

## Running Locally (quick test)

```bash
python run_pipeline.py --stage generate --models llama3_8b --strategies few_shot
python run_pipeline.py --stage metrics --skip_plausibility
python run_pipeline.py --stage figures
```

## Project Structure

```
├── config/          # YAML configs for models, datasets, eval
├── data/            # Download + preprocessing scripts
├── src/
│   ├── models/      # vLLM inference engine + prompt templates
│   ├── eval/        # Factual, novelty, plausibility, aggregate metrics
│   ├── finetune/    # LoRA fine-tuning (dataset + trainer)
│   └── utils/       # I/O helpers
├── analysis/        # Results aggregation + all paper figures
├── scripts/         # SLURM job scripts (BU SCC H200)
├── results/         # Generated hypotheses, metrics, figures
└── run_pipeline.py  # Master runner
```

## Models Benchmarked

| Model | Params | Family |
|---|---|---|
| LLaMA-3.1-8B-Instruct | 8B | Meta |
| LLaMA-3.1-70B-Instruct | 70B | Meta |
| Mistral-7B-Instruct-v0.3 | 7B | Mistral AI |
| Qwen2.5-7B-Instruct | 7B | Alibaba |
| Phi-3.5-mini-instruct | 3.8B | Microsoft |
| **LLaMA-3.1-8B-FT** (ours) | 8B | Fine-tuned |

## Evaluation Metrics

- **Factual Grounding** (Token F1, ROUGE-L) — on SciQ
- **Novelty Score** (FAISS cosine similarity vs. arXiv corpus)
- **Plausibility** (LLM-as-judge via LLaMA-3.1-70B, 4 aspects × 1-5 scale)
- **Composite Score** (weighted average)
