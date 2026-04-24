"""Aggregate all metrics into final per-model scores."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def load_eval_config(path: str = "config/eval.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["evaluation"]


def compute_composite_score(
    factual: float,
    novelty: float,
    plausibility: float,
    weights: dict,
) -> float:
    """Weighted composite of the three core metrics (all normalized 0-1)."""
    w_f = weights.get("factual", 0.35)
    w_n = weights.get("novelty", 0.35)
    w_p = weights.get("plausibility", 0.30)
    # Normalize plausibility from 1-5 scale to 0-1
    plausibility_norm = (plausibility - 1) / 4
    return w_f * factual + w_n * novelty + w_p * plausibility_norm


class MetricsAggregator:
    def __init__(self, config_path: str = "config/eval.yaml"):
        self.cfg = load_eval_config(config_path)
        self.weights = self.cfg["aggregate_weights"]

    def aggregate_model(
        self,
        model_key: str,
        factual_results: Optional[list[dict]],
        hypothesis_results: list[dict],  # already has novelty_score + judge_* fields
    ) -> dict:
        from src.eval.factual import evaluate_factual, evaluate_factual_by_domain
        from src.eval.novelty import NoveltyEvaluator

        result = {"model": model_key}

        # Factual
        if factual_results:
            fact = evaluate_factual(factual_results)
            result["factual_exact_match"] = fact["exact_match"]
            result["factual_token_f1"] = fact["token_f1"]
            result["factual_rouge_l"] = fact["rouge_l"]
            result["factual_n"] = fact["n"]
            result["factual_by_domain"] = evaluate_factual_by_domain(factual_results)
        else:
            result["factual_exact_match"] = 0.0
            result["factual_token_f1"] = 0.0
            result["factual_rouge_l"] = 0.0

        # Novelty
        novelty_scores = [r["novelty_score"] for r in hypothesis_results if "novelty_score" in r]
        result["novelty_mean"] = float(np.mean(novelty_scores)) if novelty_scores else 0.0
        result["novelty_std"] = float(np.std(novelty_scores)) if novelty_scores else 0.0
        result["novelty_n"] = len(novelty_scores)

        # Plausibility
        plaus_scores = [r.get("plausibility_mean", 0.0) for r in hypothesis_results]
        result["plausibility_mean"] = float(np.mean(plaus_scores)) if plaus_scores else 0.0
        result["plausibility_std"] = float(np.std(plaus_scores)) if plaus_scores else 0.0

        # Per-aspect
        for aspect in self.cfg["plausibility"]["aspects"]:
            key = f"judge_{aspect}"
            vals = [r[key] for r in hypothesis_results if key in r]
            result[f"plausibility_{aspect}"] = float(np.mean(vals)) if vals else 0.0

        # Composite
        result["composite_score"] = compute_composite_score(
            factual=result["factual_token_f1"],
            novelty=result["novelty_mean"],
            plausibility=result["plausibility_mean"],
            weights=self.weights,
        )

        # Domain breakdown
        domains = {}
        for r in hypothesis_results:
            d = r.get("domain", "unknown")
            domains.setdefault(d, {"novelty": [], "plausibility": []})
            if "novelty_score" in r:
                domains[d]["novelty"].append(r["novelty_score"])
            if "plausibility_mean" in r:
                domains[d]["plausibility"].append(r["plausibility_mean"])

        result["by_domain"] = {
            d: {
                "novelty_mean": float(np.mean(v["novelty"])) if v["novelty"] else 0.0,
                "plausibility_mean": float(np.mean(v["plausibility"])) if v["plausibility"] else 0.0,
            }
            for d, v in domains.items()
        }

        return result

    def build_leaderboard(self, all_model_results: list[dict]) -> pd.DataFrame:
        rows = []
        for r in all_model_results:
            rows.append({
                "Model": r.get("short_name", r["model"]),
                "Factual F1": round(r["factual_token_f1"], 3),
                "ROUGE-L": round(r["factual_rouge_l"], 3),
                "Novelty": round(r["novelty_mean"], 3),
                "Plausibility": round(r["plausibility_mean"], 2),
                "Composite": round(r["composite_score"], 3),
            })
        df = pd.DataFrame(rows).sort_values("Composite", ascending=False)
        return df

    def print_leaderboard(self, df: pd.DataFrame):
        table = Table(title="Model Leaderboard", show_header=True, header_style="bold magenta")
        for col in df.columns:
            table.add_column(col, justify="right" if col != "Model" else "left")
        for _, row in df.iterrows():
            table.add_row(*[str(v) for v in row.values])
        console.print(table)

    def save_all(self, all_model_results: list[dict], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-model JSON
        for r in all_model_results:
            path = output_dir / f"{r['model']}_metrics.json"
            with open(path, "w") as f:
                json.dump(r, f, indent=2)

        # Leaderboard CSV
        df = self.build_leaderboard(all_model_results)
        df.to_csv(output_dir / "leaderboard.csv", index=False)
        self.print_leaderboard(df)

        # Full combined JSON
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_model_results, f, indent=2)

        console.print(f"[green]All metrics saved to {output_dir}[/green]")
        return df
