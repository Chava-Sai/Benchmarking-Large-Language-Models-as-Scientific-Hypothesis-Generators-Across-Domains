"""
Generate all paper figures from computed metrics.

Produces:
  - Figure 1: Leaderboard bar chart (composite + per-metric)
  - Figure 2: Radar chart per model
  - Figure 3: Cross-domain novelty heatmap
  - Figure 4: Novelty vs. Plausibility scatter (Pareto front)
  - Figure 5: Fine-tuned vs. base model delta
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

PALETTE = {
    "LLaMA-3.1-8B":    "#4C72B0",
    "LLaMA-3.1-70B":   "#DD8452",
    "Mistral-7B":      "#55A868",
    "Qwen2.5-7B":      "#C44E52",
    "Phi-3.5-mini":    "#8172B3",
    "LLaMA-3.1-8B-FT": "#937860",
}

FIGURE_DIR = Path("results/figures")


def load_results(metrics_dir: str = "results/metrics") -> list[dict]:
    metrics_dir = Path(metrics_dir)
    results = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def fig1_leaderboard(results: list[dict]):
    """Grouped bar chart: Factual F1 | Novelty | Plausibility | Composite."""
    models = [r.get("short_name", r["model"]) for r in results]
    metrics = {
        "Factual F1": [r["factual_token_f1"] for r in results],
        "Novelty":    [r["novelty_mean"] for r in results],
        "Plausibility\n(norm.)": [(r["plausibility_mean"] - 1) / 4 for r in results],
        "Composite":  [r["composite_score"] for r in results],
    }

    x = np.arange(len(models))
    width = 0.2
    n_metrics = len(metrics)
    offsets = np.linspace(-(n_metrics - 1) / 2 * width, (n_metrics - 1) / 2 * width, n_metrics)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    for (label, vals), offset, color in zip(metrics.items(), offsets, colors):
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score (0–1)", fontsize=12)
    ax.set_title("LLM Hypothesis Generation Benchmark — All Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "fig1_leaderboard.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Figure 1 → {path}[/green]")


def fig2_radar(results: list[dict]):
    """Radar chart with 4 axes per model."""
    categories = ["Factual F1", "Novelty", "Plausibility\n(norm.)", "Composite"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for r in results:
        model = r.get("short_name", r["model"])
        values = [
            r["factual_token_f1"],
            r["novelty_mean"],
            (r["plausibility_mean"] - 1) / 4,
            r["composite_score"],
        ]
        values += values[:1]
        color = PALETTE.get(model, "#888888")
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model Capability Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    path = FIGURE_DIR / "fig2_radar.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Figure 2 → {path}[/green]")


def fig3_cross_domain_heatmap(csv_path: str = "results/figures/cross_domain.csv"):
    """Heatmap: rows=models, cols=domains, values=novelty_mean."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        console.print(f"[yellow]Cross-domain CSV not found: {csv_path}[/yellow]")
        return

    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index="model", columns="domain", values="novelty_mean", aggfunc="mean")
    pivot = pivot.fillna(0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.7)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Novelty Score"},
        vmin=0, vmax=1,
    )
    ax.set_title("Cross-Domain Novelty Scores", fontsize=13, fontweight="bold")
    ax.set_xlabel("Domain", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    path = FIGURE_DIR / "fig3_cross_domain_heatmap.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Figure 3 → {path}[/green]")


def fig4_novelty_vs_plausibility(results: list[dict]):
    """Scatter: novelty vs plausibility with Pareto front."""
    fig, ax = plt.subplots(figsize=(7, 6))

    xs = [r["novelty_mean"] for r in results]
    ys = [r["plausibility_mean"] for r in results]
    labels = [r.get("short_name", r["model"]) for r in results]

    for x, y, label in zip(xs, ys, labels):
        color = PALETTE.get(label, "#888888")
        marker = "*" if "FT" in label else "o"
        size = 200 if "FT" in label else 120
        ax.scatter(x, y, c=color, s=size, marker=marker, zorder=3, edgecolors="white", linewidth=1.5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Pareto front
    pareto_xs, pareto_ys = _pareto_front(xs, ys)
    if pareto_xs:
        ax.plot(pareto_xs, pareto_ys, "k--", alpha=0.4, lw=1.5, label="Pareto front")

    ax.set_xlabel("Novelty Score", fontsize=12)
    ax.set_ylabel("Plausibility Score (1–5)", fontsize=12)
    ax.set_title("Novelty vs. Plausibility Trade-off", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = FIGURE_DIR / "fig4_novelty_vs_plausibility.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Figure 4 → {path}[/green]")


def fig5_finetune_delta(results: list[dict]):
    """Bar chart showing improvement from fine-tuning."""
    base = next((r for r in results if r.get("model") == "llama3_8b"), None)
    ft = next((r for r in results if r.get("model") == "llama3_8b_finetuned"), None)
    if not base or not ft:
        console.print("[yellow]Fine-tuned model results not found, skipping Fig 5[/yellow]")
        return

    metrics = {
        "Factual F1": (base["factual_token_f1"], ft["factual_token_f1"]),
        "Novelty":    (base["novelty_mean"], ft["novelty_mean"]),
        "Plausibility\n(norm.)": (
            (base["plausibility_mean"] - 1) / 4,
            (ft["plausibility_mean"] - 1) / 4,
        ),
        "Composite":  (base["composite_score"], ft["composite_score"]),
    }

    labels = list(metrics.keys())
    base_vals = [v[0] for v in metrics.values()]
    ft_vals   = [v[1] for v in metrics.values()]
    deltas    = [ft - b for b, ft in zip(base_vals, ft_vals)]

    x = np.arange(len(labels))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Side-by-side comparison
    ax1.bar(x - width / 2, base_vals, width, label="LLaMA-3.1-8B (base)", color="#4C72B0", alpha=0.85)
    ax1.bar(x + width / 2, ft_vals,   width, label="LLaMA-3.1-8B-FT",     color="#937860", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Score (0–1)", fontsize=11)
    ax1.set_title("Base vs. Fine-tuned", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Delta plot
    colors = ["#55A868" if d > 0 else "#C44E52" for d in deltas]
    ax2.bar(x, deltas, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Δ Score (FT − Base)", fontsize=11)
    ax2.set_title("Fine-tuning Improvement", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    path = FIGURE_DIR / "fig5_finetune_delta.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Figure 5 → {path}[/green]")


def _pareto_front(xs, ys):
    """Compute Pareto front for maximization of both axes."""
    points = sorted(zip(xs, ys), key=lambda p: p[0])
    pareto_x, pareto_y = [], []
    max_y = -float("inf")
    for x, y in reversed(points):
        if y >= max_y:
            pareto_x.append(x)
            pareto_y.append(y)
            max_y = y
    pareto_x.reverse()
    pareto_y.reverse()
    return pareto_x, pareto_y


def generate_all(metrics_dir: str = "results/metrics"):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(metrics_dir)
    if not results:
        console.print("[red]No metric files found. Run compute_results.py first.[/red]")
        return

    fig1_leaderboard(results)
    fig2_radar(results)
    fig3_cross_domain_heatmap()
    fig4_novelty_vs_plausibility(results)
    fig5_finetune_delta(results)

    console.print("\n[bold green]All figures generated![/bold green]")


if __name__ == "__main__":
    import fire
    fire.Fire(generate_all)
