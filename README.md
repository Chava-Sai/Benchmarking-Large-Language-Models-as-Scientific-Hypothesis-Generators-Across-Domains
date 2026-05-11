# SciHypo-Bench: Benchmarking LLMs as Scientific Hypothesis Generators

**ICML 2026 AI for Science Workshop** | Seoul, South Korea

> Can large language models generate novel, testable scientific hypotheses?  
> We benchmark 7 models across biology, chemistry, physics, and CS — and reveal a fundamental **novelty–accuracy trade-off** that scaling alone cannot resolve.

---

## Installation

```bash
git clone https://github.com/Chava-Sai/Benchmarking-Large-Language-Models-as-Scientific-Hypothesis-Generators-Across-Domains.git
cd Benchmarking-Large-Language-Models-as-Scientific-Hypothesis-Generators-Across-Domains

pip install -r requirements.txt
cp .env.example .env   # add your HuggingFace + OpenAI API keys
```

**Requirements:** Python 3.10+, PyTorch 2.1+, ~16GB VRAM for 8B models, ~40GB for 70B (AWQ 4-bit).

---

## Quick Start

```bash
# Run hypothesis generation for one model, one strategy
python run_pipeline.py --stage generate --models llama3_8b --strategies few_shot

# Compute metrics and generate figures
python run_pipeline.py --stage metrics --skip_plausibility
python run_pipeline.py --stage figures
```

---

## Full Pipeline

```bash
# 1. Download and preprocess datasets (SciQ, PubMedQA, arXiv abstracts)
python data/download_datasets.py
python data/preprocess.py

# 2. Generate hypotheses — all models × 3 strategies
python run_pipeline.py --stage generate

# 3. Factual evaluation
python run_pipeline.py --stage factual

# 4. LoRA fine-tuning of LLaMA-3.1-8B
python -m src.finetune.train

# 5. Evaluate fine-tuned model
python run_pipeline.py --stage generate --models llama3_8b_finetuned
python run_pipeline.py --stage factual --models llama3_8b_finetuned

# 6. Aggregate metrics + produce all paper figures
python -m analysis.compute_results
python -m analysis.visualize
```

---

## Project Structure

```
├── config/          # YAML configs (models, datasets, evaluation)
├── data/            # Dataset download and preprocessing scripts
├── src/
│   ├── models/      # Inference engine + prompt templates
│   ├── eval/        # Factual, novelty, plausibility, aggregate metrics
│   ├── finetune/    # LoRA fine-tuning (dataset builder + trainer)
│   └── utils/       # I/O helpers
├── analysis/        # Results aggregation and figure generation
├── paper/           # LaTeX source and compiled PDF
├── results/
│   ├── hypotheses/  # Model outputs (all models × strategies)
│   ├── metrics/     # Aggregated metrics per model
│   └── figures/     # All paper figures
└── run_pipeline.py  # Master runner script
```

---

## Configuration

Edit `config/models.yaml` to enable/disable models or change generation settings.  
Edit `config/eval.yaml` to configure embedding model, FAISS index, and novelty settings.

API keys go in `.env`:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
HF_TOKEN=...
```

---

## Datasets

| Dataset | Domain | Use | N |
|---|---|---|---|
| [SciQ](https://allenai.org/data/sciq) | General science | Factual evaluation | 1,000 |
| [PubMedQA](https://pubmedqa.github.io/) | Biomedical | Hypothesis generation | 500 |
| arXiv abstracts | Multi-domain | Hypothesis generation + fine-tuning | 6,500 |

---

## Citation

```bibtex
@inproceedings{scihypobench2026,
  title     = {SciHypo-Bench: Benchmarking Large Language Models as Scientific Hypothesis Generators Across Domains},
  author    = {Anonymous},
  booktitle = {ICML 2026 Workshop on AI for Science},
  year      = {2026}
}
```
