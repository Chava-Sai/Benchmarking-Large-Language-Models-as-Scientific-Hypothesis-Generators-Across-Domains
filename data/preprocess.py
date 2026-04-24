"""Preprocess raw datasets into unified format for evaluation."""

import json
import re
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def preprocess_sciq(raw_dir: Path, proc_dir: Path):
    samples = load_jsonl(raw_dir / "sciq.jsonl")
    processed = []
    for s in samples:
        processed.append({
            "id": s["id"],
            "dataset": "sciq",
            "task": "factual_qa",
            "domain": s["domain"],
            "question": s["question"],
            "context": s.get("support", ""),
            "answer": s["correct_answer"],
            "answer_normalized": normalize_answer(s["correct_answer"]),
            "choices": [
                s["correct_answer"],
                s.get("distractor1", ""),
                s.get("distractor2", ""),
                s.get("distractor3", ""),
            ],
        })
    save_jsonl(processed, proc_dir / "sciq.jsonl")
    console.print(f"[green]SciQ preprocessed: {len(processed)} samples[/green]")


def preprocess_pubmedqa(raw_dir: Path, proc_dir: Path):
    samples = load_jsonl(raw_dir / "pubmedqa.jsonl")
    processed = []
    for s in samples:
        processed.append({
            "id": s["id"],
            "dataset": "pubmedqa",
            "task": "hypothesis_generation",
            "domain": "biomedical",
            "abstract": s["abstract"],
            "question": s["question"],
            "reference_answer": s.get("long_answer", ""),
            "decision": s.get("final_decision", ""),
        })
    save_jsonl(processed, proc_dir / "pubmedqa.jsonl")
    console.print(f"[green]PubMedQA preprocessed: {len(processed)} samples[/green]")


def preprocess_arxiv(raw_dir: Path, proc_dir: Path):
    samples = load_jsonl(raw_dir / "arxiv_abstracts.jsonl")
    processed = []
    for s in samples:
        processed.append({
            "id": s["id"],
            "dataset": "arxiv",
            "task": "hypothesis_generation",
            "domain": s["domain"],
            "title": s["title"],
            "abstract": s["abstract"],
            "categories": s.get("categories", ""),
        })
    save_jsonl(processed, proc_dir / "arxiv_abstracts.jsonl")
    console.print(f"[green]arXiv preprocessed: {len(processed)} samples[/green]")


def preprocess_scibench(raw_dir: Path, proc_dir: Path):
    samples = load_jsonl(raw_dir / "scibench.jsonl")
    processed = []
    for s in samples:
        processed.append({
            "id": s["id"],
            "dataset": "scibench",
            "task": "factual_qa",
            "domain": s.get("domain", "stem"),
            "question": s["question"],
            "answer": s["answer"],
            "answer_normalized": normalize_answer(str(s["answer"])),
            "unit": s.get("unit", ""),
        })
    save_jsonl(processed, proc_dir / "scibench.jsonl")
    console.print(f"[green]SciBench preprocessed: {len(processed)} samples[/green]")


def build_hypothesis_eval_set(proc_dir: Path):
    """Combine pubmedqa + arxiv into unified hypothesis eval set."""
    pubmed = load_jsonl(proc_dir / "pubmedqa.jsonl")
    arxiv = load_jsonl(proc_dir / "arxiv_abstracts.jsonl")

    combined = []
    for s in pubmed[:300]:
        combined.append({**s, "source_dataset": "pubmedqa"})
    for s in arxiv[:700]:
        combined.append({**s, "source_dataset": "arxiv"})

    save_jsonl(combined, proc_dir / "hypothesis_eval_set.jsonl")
    console.print(f"[green]Hypothesis eval set: {len(combined)} samples[/green]")


def main():
    with open("config/datasets.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw"])
    proc_dir = Path(cfg["paths"]["processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "sciq.jsonl").exists():
        preprocess_sciq(raw_dir, proc_dir)
    if (raw_dir / "pubmedqa.jsonl").exists():
        preprocess_pubmedqa(raw_dir, proc_dir)
    if (raw_dir / "arxiv_abstracts.jsonl").exists():
        preprocess_arxiv(raw_dir, proc_dir)
    if (raw_dir / "scibench.jsonl").exists():
        preprocess_scibench(raw_dir, proc_dir)

    build_hypothesis_eval_set(proc_dir)
    console.print("[bold green]Preprocessing complete![/bold green]")


if __name__ == "__main__":
    main()
