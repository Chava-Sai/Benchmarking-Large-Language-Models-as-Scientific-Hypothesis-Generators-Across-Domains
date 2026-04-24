"""
Master pipeline runner.

Stages:
  1  download    — download all datasets
  2  preprocess  — preprocess into unified format
  3  generate    — run hypothesis generation for all models
  4  factual     — run factual QA evaluation for all models
  5  novelty     — compute novelty scores (builds FAISS index)
  6  plausibility — LLM-as-judge evaluation
  7  metrics     — aggregate all metrics + leaderboard
  8  figures     — generate paper figures
  9  finetune    — LoRA fine-tuning of LLaMA-3.1-8B
  all            — run stages 1–8 (no finetune)

Usage:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage generate --models llama3_8b,mistral_7b
  python run_pipeline.py --stage finetune
"""

import json
import sys
from pathlib import Path

import fire
import yaml
from rich.console import Console

console = Console()

STRATEGIES = ["zero_shot", "few_shot", "cot"]


def load_configs():
    with open("config/models.yaml") as f:
        mcfg = yaml.safe_load(f)
    with open("config/datasets.yaml") as f:
        dcfg = yaml.safe_load(f)
    with open("config/eval.yaml") as f:
        ecfg = yaml.safe_load(f)
    return mcfg, dcfg, ecfg


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------
def stage_download():
    console.print("\n[bold cyan]STAGE 1: Downloading datasets[/bold cyan]")
    from data.download_datasets import main as download_main
    download_main()


# ---------------------------------------------------------------------------
# Stage 2: Preprocess
# ---------------------------------------------------------------------------
def stage_preprocess():
    console.print("\n[bold cyan]STAGE 2: Preprocessing[/bold cyan]")
    from data.preprocess import main as preprocess_main
    preprocess_main()


# ---------------------------------------------------------------------------
# Stage 3: Generate hypotheses
# ---------------------------------------------------------------------------
def stage_generate(models: str = "all", strategies: str = "few_shot", batch_size: int = 32):
    console.print("\n[bold cyan]STAGE 3: Hypothesis generation[/bold cyan]")
    from src.models.loader import run_hypothesis_generation

    mcfg, dcfg, _ = load_configs()

    if models in ("all", ("all",)):
        model_keys = [k for k, v in mcfg["models"].items() if v.get("enabled", False)]
    elif isinstance(models, (list, tuple)):
        model_keys = [m.strip() for m in models]
    else:
        model_keys = [m.strip() for m in models.split(",")]

    if isinstance(strategies, (list, tuple)):
        strat_list = [s.strip() for s in strategies]
    else:
        strat_list = [s.strip() for s in strategies.split(",")]

    hyp_eval_path = Path(dcfg["paths"]["processed"]) / "hypothesis_eval_set.jsonl"
    if not hyp_eval_path.exists():
        console.print("[red]Run preprocess first.[/red]")
        sys.exit(1)

    samples = load_jsonl(hyp_eval_path)
    out_dir = Path("results/hypotheses")

    for model_key in model_keys:
        console.print(f"\n[bold]Model: {model_key}[/bold]")
        try:
            run_hypothesis_generation(
                model_key=model_key,
                samples=samples,
                strategies=strat_list,
                output_dir=out_dir,
                batch_size=batch_size,
            )
        except Exception as e:
            console.print(f"[red]Error generating for {model_key}: {e}[/red]")
            raise

        # Force GPU memory release between models
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 4: Factual evaluation
# ---------------------------------------------------------------------------
def stage_factual(models: str = "all", batch_size: int = 64):
    console.print("\n[bold cyan]STAGE 4: Factual QA evaluation[/bold cyan]")
    from src.models.loader import run_factual_eval

    mcfg, dcfg, _ = load_configs()

    if models in ("all", ("all",)):
        model_keys = [k for k, v in mcfg["models"].items() if v.get("enabled", False)]
    elif isinstance(models, (list, tuple)):
        model_keys = [m.strip() for m in models]
    else:
        model_keys = [m.strip() for m in models.split(",")]

    # Use SciQ for factual eval (fast, known ground truth)
    sciq_path = Path(dcfg["paths"]["processed"]) / "sciq.jsonl"
    if not sciq_path.exists():
        console.print("[red]Run preprocess first.[/red]")
        sys.exit(1)

    samples = load_jsonl(sciq_path)
    out_dir = Path("results/hypotheses")

    for model_key in model_keys:
        console.print(f"\n[bold]Model: {model_key}[/bold]")
        try:
            run_factual_eval(
                model_key=model_key,
                samples=samples,
                output_dir=out_dir,
                batch_size=batch_size,
            )
        except Exception as e:
            console.print(f"[red]Error in factual eval for {model_key}: {e}[/red]")
            raise

        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 5–7: Metrics (novelty + plausibility + aggregate)
# ---------------------------------------------------------------------------
def stage_metrics(strategy: str = "few_shot", skip_plausibility: bool = False):
    console.print("\n[bold cyan]STAGE 5-7: Computing all metrics[/bold cyan]")
    from analysis.compute_results import compute_all
    compute_all(strategy=strategy, skip_plausibility=skip_plausibility)


# ---------------------------------------------------------------------------
# Stage 8: Figures
# ---------------------------------------------------------------------------
def stage_figures():
    console.print("\n[bold cyan]STAGE 8: Generating figures[/bold cyan]")
    from analysis.visualize import generate_all
    generate_all()
    from analysis.compute_results import cross_domain_analysis
    cross_domain_analysis()


# ---------------------------------------------------------------------------
# Stage 9: Fine-tune
# ---------------------------------------------------------------------------
def stage_finetune(**kwargs):
    console.print("\n[bold cyan]STAGE 9: LoRA Fine-tuning[/bold cyan]")
    from src.finetune.train import train, FinetuneConfig
    cfg = FinetuneConfig(**kwargs)
    train(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    stage: str = "all",
    models: str = "all",
    strategies: str = "few_shot",
    batch_size: int = 32,
    skip_plausibility: bool = False,
):
    """
    Args:
        stage: download|preprocess|generate|factual|metrics|figures|finetune|all
        models: comma-separated model keys or 'all'
        strategies: comma-separated prompt strategies
        batch_size: inference batch size
        skip_plausibility: skip expensive LLM judge step
    """
    if stage in ("all", "download"):
        stage_download()
    if stage in ("all", "preprocess"):
        stage_preprocess()
    if stage in ("all", "generate"):
        stage_generate(models=models, strategies=strategies, batch_size=batch_size)
    if stage in ("all", "factual"):
        stage_factual(models=models, batch_size=batch_size)
    if stage in ("all", "metrics"):
        stage_metrics(skip_plausibility=skip_plausibility)
    if stage in ("all", "figures"):
        stage_figures()
    if stage == "finetune":
        stage_finetune()

    console.print("\n[bold green]Pipeline complete![/bold green]")


if __name__ == "__main__":
    fire.Fire(run)
