"""LLM-as-judge plausibility evaluation for generated hypotheses."""

import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from rich.console import Console
from tqdm import tqdm

console = Console()


def load_eval_config(path: str = "config/eval.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["evaluation"]["plausibility"]


class PlausibilityJudge:
    """
    Uses LLaMA-3.1-70B (or GPT-4o-mini as fallback) to rate hypotheses on:
    - scientific_validity (1-5)
    - testability (1-5)
    - specificity (1-5)
    - novelty_claimed (1-5)
    """

    def __init__(self, config_path: str = "config/eval.yaml"):
        cfg = load_eval_config(config_path)
        self.cfg = cfg
        self.judge_model = cfg["judge_model"]
        self.batch_size = cfg.get("batch_size", 32)
        self.aspects = cfg["aspects"]
        self.scale = cfg.get("scale", 5)
        self.engine = None

    def _load_vllm_judge(self):
        from vllm import LLM, SamplingParams
        console.print(f"[cyan]Loading judge: {self.judge_model}[/cyan]")
        self.engine = LLM(
            model=self.judge_model,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    def _load_openai_judge(self):
        import openai
        self.openai_client = openai.OpenAI()
        console.print("[cyan]Using GPT-4o-mini as judge.[/cyan]")

    def evaluate(self, results: list[dict], cache_path: Optional[Path] = None) -> list[dict]:
        """Add plausibility scores to result dicts."""
        if cache_path and cache_path.exists():
            console.print(f"[yellow]Loading cached plausibility from {cache_path}[/yellow]")
            with open(cache_path) as f:
                return [json.loads(l) for l in f if l.strip()]

        from src.models.prompts import build_judge_prompt

        # Load judge model
        try:
            self._load_vllm_judge()
            use_vllm = True
        except Exception as e:
            console.print(f"[yellow]vLLM judge failed ({e}), falling back to OpenAI.[/yellow]")
            self._load_openai_judge()
            use_vllm = False

        enriched = []
        for i in tqdm(range(0, len(results), self.batch_size), desc="Judging"):
            batch = results[i : i + self.batch_size]
            prompts = [
                build_judge_prompt(r["abstract"], r["hypothesis"])
                for r in batch
            ]

            if use_vllm:
                tokenizer = self.engine.get_tokenizer()
                formatted = [
                    tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                    for p in prompts
                ]
                outputs = self.engine.generate(formatted, self.sampling)
                raw_outputs = [o.outputs[0].text.strip() for o in outputs]
            else:
                raw_outputs = []
                for p in prompts:
                    resp = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=p,
                        temperature=0.0,
                        max_tokens=256,
                    )
                    raw_outputs.append(resp.choices[0].message.content.strip())
                    time.sleep(0.05)

            for result, raw in zip(batch, raw_outputs):
                scores = self._parse_judge_output(raw)
                enriched.append({
                    **result,
                    **{f"judge_{k}": v for k, v in scores.items()},
                    "judge_raw": raw,
                    "plausibility_mean": float(np.mean([
                        scores.get(a, 3.0) for a in self.aspects
                    ])),
                })

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                for r in enriched:
                    f.write(json.dumps(r) + "\n")

        return enriched

    def _parse_judge_output(self, text: str) -> dict:
        try:
            # Find JSON block
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                scores = {}
                for aspect in self.aspects:
                    val = parsed.get(aspect, 3)
                    scores[aspect] = float(max(1, min(self.scale, val)))
                scores["reasoning"] = parsed.get("reasoning", "")
                return scores
        except Exception:
            pass

        # Regex fallback: look for "aspect": N patterns
        scores = {}
        for aspect in self.aspects:
            pattern = rf'"{aspect}"\s*:\s*(\d+)'
            m = re.search(pattern, text)
            scores[aspect] = float(m.group(1)) if m else 3.0
        return scores

    def aggregate(self, results: list[dict]) -> dict:
        overall = {}
        for aspect in self.aspects:
            key = f"judge_{aspect}"
            vals = [r[key] for r in results if key in r]
            if vals:
                overall[aspect] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
        means = [r["plausibility_mean"] for r in results if "plausibility_mean" in r]
        overall["plausibility_mean"] = {
            "mean": float(np.mean(means)) if means else 0.0,
            "std": float(np.std(means)) if means else 0.0,
        }
        overall["n"] = len(results)
        return overall

    def aggregate_by_domain(self, results: list[dict]) -> dict[str, dict]:
        domains = {}
        for r in results:
            d = r.get("domain", "unknown")
            domains.setdefault(d, []).append(r)
        return {d: self.aggregate(items) for d, items in domains.items()}
