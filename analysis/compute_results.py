"""
Compute all metrics from raw model outputs and build the results table.

Usage:
  python -m analysis.compute_results --results_dir results/hypotheses
"""

import json
import sys
from pathlib import Path

# Patch transformers.activations before any import chain triggers awq/peft
try:
    import transformers.activations as _ta
    if not hasattr(_ta, "PytorchGELUTanh"):
        import torch.nn as nn
        import torch.nn.functional as F
        class PytorchGELUTanh(nn.Module):
            def forward(self, x):
                return F.gelu(x, approximate="tanh")
        _ta.PytorchGELUTanh = PytorchGELUTanh
except Exception:
    pass

import numpy as np
import pandas as pd
import yaml
import fire
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.factual import evaluate_factual, evaluate_factual_by_domain
from src.eval.novelty import NoveltyEvaluator
from src.eval.plausibility import PlausibilityJudge
from src.eval.metrics import MetricsAggregator

console = Console()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_all(
    results_dir: str = "results/hypotheses",
    metrics_dir: str = "results/metrics",
    processed_dir: str = "data/processed",
    strategy: str = "few_shot",
    skip_plausibility: bool = False,
    skip_novelty: bool = False,
):
    results_dir = Path(results_dir)
    metrics_dir = Path(metrics_dir)
    processed_dir = Path(processed_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open("config/models.yaml") as f:
        model_cfg = yaml.safe_load(f)

    enabled_models = [
        k for k, v in model_cfg["models"].items() if v.get("enabled", False)
    ]
    console.print(f"[bold]Processing models: {enabled_models}[/bold]")

    # Build novelty index once (expensive, cache it)
    novelty_eval = None
    if not skip_novelty:
        corpus_path = processed_dir / "arxiv_abstracts.jsonl"
        if corpus_path.exists():
            novelty_eval = NoveltyEvaluator()
            corpus = load_jsonl(corpus_path)
            corpus_texts = [s["abstract"] for s in corpus]
            novelty_eval.build_corpus_index(
                corpus_texts,
                index_path=Path("data/embeddings/arxiv_faiss.index"),
            )

    plaus_judge = None
    if not skip_plausibility:
        plaus_judge = PlausibilityJudge()

    aggregator = MetricsAggregator()
    all_model_results = []

    for model_key in enabled_models:
        short_name = model_cfg["models"][model_key]["short_name"]
        console.print(f"\n[bold magenta]=== {short_name} ===[/bold magenta]")

        # Load factual eval results
        factual_path = results_dir / f"{model_key}_factual.jsonl"
        factual_results = load_jsonl(factual_path) if factual_path.exists() else None
        if factual_results is None:
            console.print(f"[yellow]No factual results for {model_key}[/yellow]")

        # Load hypothesis generation results
        hyp_path = results_dir / f"{model_key}_{strategy}.jsonl"
        if not hyp_path.exists():
            console.print(f"[red]No hypothesis results for {model_key} ({strategy})[/red]")
            continue

        hyp_results = load_jsonl(hyp_path)
        console.print(f"  Loaded {len(hyp_results)} hypotheses")

        # Add novelty scores
        if novelty_eval and not all("novelty_score" in r for r in hyp_results):
            console.print("  Computing novelty scores...")
            hyp_results = novelty_eval.evaluate_batch(hyp_results)
            # Update cache
            with open(hyp_path, "w") as f:
                for r in hyp_results:
                    f.write(json.dumps(r) + "\n")

        # Add plausibility scores
        plaus_cache = metrics_dir / f"{model_key}_plausibility.jsonl"
        if plaus_judge and not plaus_cache.exists():
            console.print("  Running LLM-as-judge...")
            hyp_results = plaus_judge.evaluate(hyp_results, cache_path=plaus_cache)
        elif plaus_cache.exists():
            hyp_results = load_jsonl(plaus_cache)

        # Aggregate
        model_result = aggregator.aggregate_model(
            model_key=model_key,
            factual_results=factual_results,
            hypothesis_results=hyp_results,
        )
        model_result["short_name"] = short_name
        all_model_results.append(model_result)

        console.print(f"  Factual F1: {model_result['factual_token_f1']:.3f}")
        console.print(f"  Novelty:    {model_result['novelty_mean']:.3f}")
        console.print(f"  Plausibility: {model_result['plausibility_mean']:.2f}/5")
        console.print(f"  Composite:  {model_result['composite_score']:.3f}")

    # Build leaderboard
    df = aggregator.save_all(all_model_results, metrics_dir)
    console.print("\n[bold green]Results computation complete![/bold green]")
    return df


def cross_domain_analysis(
    metrics_dir: str = "results/metrics",
    output_path: str = "results/figures/cross_domain.csv",
):
    """Build cross-domain analysis table from saved metrics."""
    metrics_dir = Path(metrics_dir)
    all_results = []
    for path in metrics_dir.glob("*_metrics.json"):
        with open(path) as f:
            all_results.append(json.load(f))

    rows = []
    for r in all_results:
        model = r.get("short_name", r["model"])
        for domain, scores in r.get("by_domain", {}).items():
            rows.append({
                "model": model,
                "domain": domain,
                "novelty_mean": scores.get("novelty_mean", 0.0),
                "plausibility_mean": scores.get("plausibility_mean", 0.0),
            })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Cross-domain table → {output_path}[/green]")
    return df


if __name__ == "__main__":
    fire.Fire({"compute": compute_all, "cross_domain": cross_domain_analysis})
