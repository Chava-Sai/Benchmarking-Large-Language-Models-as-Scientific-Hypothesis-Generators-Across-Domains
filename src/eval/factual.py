"""Factual grounding evaluation: exact match, F1, ROUGE-L."""

import re
import string
from collections import Counter
from pathlib import Path
from typing import Optional

import json
import numpy as np
from rouge_score import rouge_scorer


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores["rougeL"].fmeasure


def evaluate_factual(results: list[dict]) -> dict:
    """Compute factual metrics over a list of prediction dicts."""
    em_scores, f1_scores, rouge_scores = [], [], []

    for r in results:
        ref = r.get("reference_answer", "")
        pred = r.get("predicted_answer", "")
        if not ref or not pred:
            continue
        em_scores.append(exact_match(pred, ref))
        f1_scores.append(token_f1(pred, ref))
        rouge_scores.append(rouge_l_score(pred, ref))

    if not em_scores:
        return {"exact_match": 0.0, "token_f1": 0.0, "rouge_l": 0.0, "n": 0}

    return {
        "exact_match": float(np.mean(em_scores)),
        "token_f1": float(np.mean(f1_scores)),
        "rouge_l": float(np.mean(rouge_scores)),
        "n": len(em_scores),
    }


def evaluate_factual_by_domain(results: list[dict]) -> dict[str, dict]:
    domains = {}
    for r in results:
        d = r.get("domain", "unknown")
        domains.setdefault(d, []).append(r)
    return {d: evaluate_factual(items) for d, items in domains.items()}


def load_and_evaluate(results_path: Path) -> dict:
    with open(results_path) as f:
        results = [json.loads(l) for l in f if l.strip()]
    overall = evaluate_factual(results)
    by_domain = evaluate_factual_by_domain(results)
    return {
        "overall": overall,
        "by_domain": by_domain,
        "model": results[0]["model"] if results else "unknown",
        "n_samples": len(results),
    }
