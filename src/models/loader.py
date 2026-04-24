"""
Inference engines for both local (vLLM) and API (OpenAI, Anthropic) models.
Routes automatically based on model config type field.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from tqdm import tqdm

console = Console()


def _patch_transformers_for_autoawq():
    """Patch transformers.activations to restore PytorchGELUTanh removed in 4.52+."""
    try:
        import transformers.activations as _ta
        if not hasattr(_ta, "PytorchGELUTanh"):
            import torch.nn as nn
            class PytorchGELUTanh(nn.Module):
                def forward(self, x):
                    import torch.nn.functional as F
                    return F.gelu(x, approximate="tanh")
            _ta.PytorchGELUTanh = PytorchGELUTanh
    except Exception:
        pass

_patch_transformers_for_autoawq()


def load_model_config(config_path: str = "config/models.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Local HuggingFace engine (replaces vLLM — works with torch 2.7 + Blackwell)
# ---------------------------------------------------------------------------
class HFInferenceEngine:
    """Batched inference via HuggingFace transformers. Works on any GPU."""

    def __init__(self, model_key: str, config_path: str = "config/models.yaml"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = load_model_config(config_path)
        model_cfg = cfg["models"][model_key]
        gen_cfg = cfg["generation"]

        self.model_key = model_key
        self.short_name = model_cfg["short_name"]
        self.model_name = model_cfg["name"]
        self.gen_cfg = gen_cfg

        console.print(f"[bold cyan]Loading {self.short_name} via HuggingFace...[/bold cyan]")

        dtype_str = model_cfg.get("dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        quant = model_cfg.get("quantization")

        load_kwargs = dict(
            pretrained_model_name_or_path=self.model_name,
            device_map="auto",
            trust_remote_code=True,
        )

        if quant == "awq":
            # AWQ models: transformers loads them natively via autoawq backend
            load_kwargs["torch_dtype"] = torch.float16
        elif quant == "bnb4":
            # bitsandbytes 4-bit — works on Blackwell without Triton AWQ kernels
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self.model.eval()
        console.print(f"[green]{self.short_name} loaded.[/green]")

    def generate(
        self,
        prompts: list[list[dict]],
        greedy: bool = False,
        batch_size: int = 8,
    ) -> list[str]:
        import torch

        max_new_tokens = (
            self.gen_cfg["greedy"]["max_new_tokens"] if greedy
            else self.gen_cfg["max_new_tokens"]
        )

        outputs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"[{self.short_name}]"):
            batch = prompts[i : i + batch_size]
            texts = []
            for msgs in batch:
                try:
                    text = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    text = "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in msgs
                    )
                    text += "\nASSISTANT:"
                texts.append(text)

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3072,
            ).to("cuda")

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if not greedy:
                gen_kwargs["temperature"] = self.gen_cfg["temperature"]
                gen_kwargs["top_p"] = self.gen_cfg["top_p"]

            with torch.no_grad():
                generated = self.model.generate(**gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            for gen in generated:
                new_tokens = gen[input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append(text.strip())

        return outputs

    def generate_single(self, messages: list[dict], greedy: bool = False) -> str:
        return self.generate([messages], greedy=greedy)[0]

    def __del__(self):
        try:
            import torch, gc
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


# Keep alias for backward compatibility
VLLMInferenceEngine = HFInferenceEngine


# ---------------------------------------------------------------------------
# OpenAI API engine
# ---------------------------------------------------------------------------
class OpenAIInferenceEngine:
    """Inference via OpenAI API (gpt-4o-mini)."""

    def __init__(self, model_key: str, config_path: str = "config/models.yaml"):
        import openai

        cfg = load_model_config(config_path)
        model_cfg = cfg["models"][model_key]
        self.model_key = model_key
        self.short_name = model_cfg["short_name"]
        self.model_name = model_cfg["name"]
        self.max_tokens = model_cfg.get("max_tokens", 512)
        self.gen_cfg = cfg["generation"]

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        self.client = openai.OpenAI(api_key=api_key)
        console.print(f"[bold cyan]{self.short_name} API client ready.[/bold cyan]")

    def generate(
        self,
        prompts: list[list[dict]],
        greedy: bool = False,
        batch_size: int = 32,  # ignored for API, kept for uniform interface
    ) -> list[str]:
        temperature = 0.0 if greedy else self.gen_cfg["temperature"]
        outputs = []
        for msgs in tqdm(prompts, desc=f"[{self.short_name}]"):
            for attempt in range(5):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=msgs,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                    )
                    outputs.append(resp.choices[0].message.content.strip())
                    time.sleep(0.05)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    console.print(f"[yellow]OpenAI error ({e}), retrying in {wait}s...[/yellow]")
                    time.sleep(wait)
            else:
                outputs.append("")
        return outputs

    def generate_single(self, messages: list[dict], greedy: bool = False) -> str:
        return self.generate([messages], greedy=greedy)[0]


# ---------------------------------------------------------------------------
# Anthropic API engine
# ---------------------------------------------------------------------------
class AnthropicInferenceEngine:
    """Inference via Anthropic API (claude-sonnet-4-6)."""

    def __init__(self, model_key: str, config_path: str = "config/models.yaml"):
        import anthropic

        cfg = load_model_config(config_path)
        model_cfg = cfg["models"][model_key]
        self.model_key = model_key
        self.short_name = model_cfg["short_name"]
        self.model_name = model_cfg["name"]
        self.max_tokens = model_cfg.get("max_tokens", 512)
        self.gen_cfg = cfg["generation"]

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")
        self.client = anthropic.Anthropic(api_key=api_key)
        console.print(f"[bold cyan]{self.short_name} API client ready.[/bold cyan]")

    def generate(
        self,
        prompts: list[list[dict]],
        greedy: bool = False,
        batch_size: int = 32,
    ) -> list[str]:
        temperature = 0.0 if greedy else self.gen_cfg["temperature"]
        outputs = []
        for msgs in tqdm(prompts, desc=f"[{self.short_name}]"):
            # Anthropic separates system prompt from messages
            system_msg = ""
            user_msgs = []
            for m in msgs:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    user_msgs.append(m)

            for attempt in range(5):
                try:
                    kwargs = dict(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=temperature,
                        messages=user_msgs,
                    )
                    if system_msg:
                        kwargs["system"] = system_msg

                    resp = self.client.messages.create(**kwargs)
                    outputs.append(resp.content[0].text.strip())
                    time.sleep(0.1)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    console.print(f"[yellow]Anthropic error ({e}), retrying in {wait}s...[/yellow]")
                    time.sleep(wait)
            else:
                outputs.append("")
        return outputs

    def generate_single(self, messages: list[dict], greedy: bool = False) -> str:
        return self.generate([messages], greedy=greedy)[0]


# ---------------------------------------------------------------------------
# Factory: pick the right engine from config
# ---------------------------------------------------------------------------
def get_engine(model_key: str, config_path: str = "config/models.yaml"):
    cfg = load_model_config(config_path)
    model_cfg = cfg["models"][model_key]
    model_type = model_cfg.get("type", "local")
    provider = model_cfg.get("api_provider", "")

    if model_type == "api":
        if provider == "openai":
            return OpenAIInferenceEngine(model_key, config_path)
        elif provider == "anthropic":
            return AnthropicInferenceEngine(model_key, config_path)
        else:
            raise ValueError(f"Unknown API provider: {provider}")
    else:
        return HFInferenceEngine(model_key, config_path)


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------
def run_hypothesis_generation(
    model_key: str,
    samples: list[dict],
    strategies: list[str],
    output_dir: Path,
    config_path: str = "config/models.yaml",
    batch_size: int = 32,
) -> dict[str, list[dict]]:
    from src.models.prompts import build_hypothesis_prompt, extract_hypothesis_from_cot

    engine = get_engine(model_key, config_path)
    results = {}

    for strategy in strategies:
        out_path = output_dir / f"{model_key}_{strategy}.jsonl"
        if out_path.exists():
            console.print(f"[yellow]Skipping {out_path} (exists)[/yellow]")
            with open(out_path) as f:
                results[strategy] = [json.loads(l) for l in f if l.strip()]
            continue

        console.print(f"[bold]  {engine.short_name} / {strategy}[/bold]")
        prompts = [build_hypothesis_prompt(s["abstract"], strategy) for s in samples]
        outputs = engine.generate(prompts, batch_size=batch_size)

        strategy_results = []
        for sample, output in zip(samples, outputs):
            hyp = extract_hypothesis_from_cot(output) if strategy == "cot" else output
            strategy_results.append({
                "id": sample["id"],
                "model": model_key,
                "short_name": engine.short_name,
                "strategy": strategy,
                "abstract": sample["abstract"],
                "domain": sample.get("domain", "unknown"),
                "hypothesis": hyp,
                "raw_output": output,
                "dataset": sample.get("dataset", "unknown"),
            })

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in strategy_results:
                f.write(json.dumps(r) + "\n")

        results[strategy] = strategy_results
        console.print(f"[green]  Saved {len(strategy_results)} → {out_path}[/green]")

    # Release local model memory
    if hasattr(engine, "llm"):
        del engine
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    return results


def run_factual_eval(
    model_key: str,
    samples: list[dict],
    output_dir: Path,
    config_path: str = "config/models.yaml",
    batch_size: int = 32,
) -> list[dict]:
    from src.models.prompts import build_factual_prompt

    out_path = output_dir / f"{model_key}_factual.jsonl"
    if out_path.exists():
        console.print(f"[yellow]Skipping {out_path} (exists)[/yellow]")
        with open(out_path) as f:
            return [json.loads(l) for l in f if l.strip()]

    engine = get_engine(model_key, config_path)
    prompts = [build_factual_prompt(s["question"], s.get("context", "")) for s in samples]
    outputs = engine.generate(prompts, greedy=True, batch_size=batch_size)

    results = []
    for sample, output in zip(samples, outputs):
        results.append({
            "id": sample["id"],
            "model": model_key,
            "short_name": engine.short_name,
            "question": sample["question"],
            "reference_answer": sample.get("answer", ""),
            "predicted_answer": output.strip(),
            "domain": sample.get("domain", "unknown"),
            "dataset": sample.get("dataset", "unknown"),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    console.print(f"[green]Factual eval → {out_path}[/green]")
    return results
