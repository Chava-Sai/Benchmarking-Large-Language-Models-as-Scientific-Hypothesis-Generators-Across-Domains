"""Download and cache all datasets from HuggingFace."""

import os
import json
import random
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console
from rich.progress import track

console = Console()


def load_config(path: str = "config/datasets.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_sciq(cfg: dict, out_dir: Path):
    console.print("[bold cyan]Downloading SciQ...[/bold cyan]")
    ds = load_dataset(cfg["hf_name"], split=cfg["split"])
    samples = []
    for item in track(ds, description="Processing SciQ"):
        samples.append({
            "id": f"sciq_{len(samples)}",
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "support": item.get("support", ""),
            "distractor1": item.get("distractor1", ""),
            "distractor2": item.get("distractor2", ""),
            "distractor3": item.get("distractor3", ""),
            "domain": "general_science",
            "dataset": "sciq",
        })
    samples = samples[:cfg["n_samples"]]
    _save(samples, out_dir / "sciq.jsonl")
    console.print(f"[green]SciQ: {len(samples)} samples saved.[/green]")
    return samples


def download_pubmedqa(cfg: dict, out_dir: Path):
    console.print("[bold cyan]Downloading PubMedQA...[/bold cyan]")
    ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["split"])
    samples = []
    for item in track(ds, description="Processing PubMedQA"):
        context = item.get("context", {})
        # context is dict with 'contexts' list
        if isinstance(context, dict):
            abstract_text = " ".join(context.get("contexts", []))
        else:
            abstract_text = str(context)
        if len(abstract_text) < 50:
            continue
        samples.append({
            "id": f"pubmedqa_{len(samples)}",
            "abstract": abstract_text,
            "question": item["question"],
            "long_answer": item.get("long_answer", ""),
            "final_decision": item.get("final_decision", ""),
            "domain": "biomedical",
            "dataset": "pubmedqa",
        })
        if len(samples) >= cfg["n_samples"]:
            break
    _save(samples, out_dir / "pubmedqa.jsonl")
    console.print(f"[green]PubMedQA: {len(samples)} samples saved.[/green]")
    return samples


def download_arxiv(cfg: dict, out_dir: Path):
    console.print("[bold cyan]Downloading arXiv abstracts...[/bold cyan]")
    ds = load_dataset(cfg["hf_name"], split=cfg["split"], streaming=True)
    categories = set(cfg.get("categories", []))
    samples = []
    for item in track(ds, description="Streaming arXiv", total=cfg["n_samples"] * 10):
        cats = item.get("categories", "") or ""
        if isinstance(cats, list):
            cats = " ".join(cats)
        if categories and not any(c in cats for c in categories):
            continue
        abstract = item.get("abstract", "") or ""
        if len(abstract.strip()) < 100:
            continue
        samples.append({
            "id": f"arxiv_{len(samples)}",
            "title": item.get("title", ""),
            "abstract": abstract.strip().replace("\n", " "),
            "categories": cats,
            "domain": _map_arxiv_domain(cats),
            "dataset": "arxiv",
        })
        if len(samples) >= cfg["n_samples"]:
            break
    _save(samples, out_dir / "arxiv_abstracts.jsonl")
    console.print(f"[green]arXiv: {len(samples)} samples saved.[/green]")
    return samples


def download_scibench(cfg: dict, out_dir: Path):
    console.print("[bold cyan]Downloading SciBench...[/bold cyan]")
    try:
        ds = load_dataset(cfg["hf_name"], split="train")
    except Exception:
        ds = load_dataset(cfg["hf_name"], "default", split="train")
    samples = []
    for item in track(ds, description="Processing SciBench"):
        samples.append({
            "id": f"scibench_{len(samples)}",
            "question": item.get("problem_text", item.get("question", "")),
            "answer": str(item.get("answer_number", item.get("answer", ""))),
            "unit": item.get("unit", ""),
            "domain": item.get("source", "stem"),
            "dataset": "scibench",
        })
        if len(samples) >= cfg["n_samples"]:
            break
    _save(samples, out_dir / "scibench.jsonl")
    console.print(f"[green]SciBench: {len(samples)} samples saved.[/green]")
    return samples


def build_finetune_data(arxiv_samples: list, cfg: dict, out_dir: Path):
    """Extract (abstract, hypothesis) pairs from arXiv for fine-tuning."""
    console.print("[bold cyan]Building fine-tune dataset from arXiv...[/bold cyan]")

    keywords = cfg["finetune_data"]["hypothesis_keywords"]
    min_abs = cfg["finetune_data"]["min_abstract_len"]
    min_hyp = cfg["finetune_data"]["min_hypothesis_len"]
    max_hyp = cfg["finetune_data"]["max_hypothesis_len"]
    n_train = cfg["finetune_data"]["n_train"]
    n_val = cfg["finetune_data"]["n_val"]

    # Load ALL arXiv abstracts (need larger pool for filtering)
    console.print("Loading large arXiv pool for hypothesis extraction...")
    ds = load_dataset(cfg["datasets"]["arxiv_abstracts"]["hf_name"],
                      split="train", streaming=True)

    pairs = []
    for item in tqdm(ds, desc="Extracting hypothesis pairs", total=(n_train + n_val) * 8):
        abstract = (item.get("abstract", "") or "").strip().replace("\n", " ")
        if len(abstract) < min_abs:
            continue

        # Search for hypothesis sentence
        hypothesis = _extract_hypothesis_sentence(abstract, keywords)
        if hypothesis is None:
            continue
        if not (min_hyp <= len(hypothesis) <= max_hyp):
            continue

        # Use abstract-minus-hypothesis as context
        context = abstract.replace(hypothesis, "").strip()
        if len(context) < min_abs // 2:
            continue

        pairs.append({
            "id": f"ft_{len(pairs)}",
            "title": (item.get("title", "") or "").strip(),
            "context": context,
            "hypothesis": hypothesis,
            "full_abstract": abstract,
            "categories": item.get("categories", ""),
            "domain": _map_arxiv_domain(item.get("categories", "")),
        })

        if len(pairs) >= (n_train + n_val):
            break

    random.shuffle(pairs)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]

    _save(train_pairs, out_dir / "finetune_train.jsonl")
    _save(val_pairs, out_dir / "finetune_val.jsonl")
    console.print(f"[green]Fine-tune: {len(train_pairs)} train, {len(val_pairs)} val pairs.[/green]")
    return train_pairs, val_pairs


def _extract_hypothesis_sentence(text: str, keywords: list) -> str | None:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in keywords):
            return sent.strip()
    return None


def _map_arxiv_domain(categories) -> str:
    if isinstance(categories, list):
        categories = " ".join(categories)
    cats = (categories or "").lower()
    if "q-bio" in cats or "bio" in cats:
        return "biology"
    if "chem" in cats:
        return "chemistry"
    if "phys" in cats or "cond-mat" in cats or "quant-ph" in cats:
        return "physics"
    if "cs." in cats:
        return "computer_science"
    if "math" in cats:
        return "mathematics"
    return "general"


def _save(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw"])
    ft_dir = Path(cfg["paths"]["finetune"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    ft_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Starting dataset downloads...[/bold]")

    download_sciq(cfg["datasets"]["sciq"], raw_dir)
    download_pubmedqa(cfg["datasets"]["pubmedqa"], raw_dir)
    download_scibench(cfg["datasets"]["scibench"], raw_dir)
    arxiv_samples = download_arxiv(cfg["datasets"]["arxiv_abstracts"], raw_dir)
    build_finetune_data(arxiv_samples, cfg, ft_dir)

    console.print("\n[bold green]All datasets downloaded successfully![/bold green]")


if __name__ == "__main__":
    main()
