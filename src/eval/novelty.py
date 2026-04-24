"""Novelty score: embedding-based similarity against reference corpus."""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from rich.console import Console
from tqdm import tqdm

console = Console()


def load_eval_config(path: str = "config/eval.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["evaluation"]["novelty"]


class NoveltyEvaluator:
    """
    Novelty = 1 - max_cosine_similarity(hypothesis, corpus)

    High novelty means the hypothesis is dissimilar to all known abstracts.
    We use the source abstract + arXiv reference corpus as the comparison set.
    """

    def __init__(self, config_path: str = "config/eval.yaml"):
        import faiss
        from sentence_transformers import SentenceTransformer

        import torch
        cfg = load_eval_config(config_path)
        self.cfg = cfg
        self.batch_size = cfg.get("batch_size", 512)
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = cfg.get("device", default_device)

        console.print(f"[cyan]Loading embedding model: {cfg['embedding_model']}[/cyan]")
        try:
            self.embedder = SentenceTransformer(cfg["embedding_model"], device=self.device)
        except Exception:
            console.print(f"[yellow]Falling back to {cfg['embedding_model_fallback']}[/yellow]")
            self.embedder = SentenceTransformer(cfg["embedding_model_fallback"], device=self.device)

        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.corpus_texts = []

    def build_corpus_index(self, corpus_texts: list[str], index_path: Optional[Path] = None):
        """Build FAISS index from reference corpus."""
        import faiss

        if index_path and index_path.exists():
            console.print(f"[yellow]Loading existing FAISS index from {index_path}[/yellow]")
            self.index = faiss.read_index(str(index_path))
            corpus_meta_path = index_path.with_suffix(".texts.pkl")
            if corpus_meta_path.exists():
                with open(corpus_meta_path, "rb") as f:
                    self.corpus_texts = pickle.load(f)
            return

        console.print(f"[cyan]Building FAISS index over {len(corpus_texts)} texts...[/cyan]")
        embeddings = self._embed(corpus_texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, self.dim,
            min(100, len(corpus_texts) // 10),
            faiss.METRIC_INNER_PRODUCT,
        )
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = self.cfg.get("faiss_nprobe", 32)
        self.corpus_texts = corpus_texts

        if index_path:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            with open(index_path.with_suffix(".texts.pkl"), "wb") as f:
                pickle.dump(corpus_texts, f)
            console.print(f"[green]FAISS index saved to {index_path}[/green]")

    def compute_novelty(
        self,
        hypotheses: list[str],
        source_abstracts: Optional[list[str]] = None,
    ) -> list[float]:
        """
        Returns novelty score (0-1) for each hypothesis.
        novelty = 1 - max(similarity to corpus + source abstract).
        """
        import faiss

        if self.index is None:
            raise RuntimeError("Must call build_corpus_index first.")

        hyp_embeddings = self._embed(hypotheses)
        faiss.normalize_L2(hyp_embeddings)

        # k=1: only need the most similar corpus item
        distances, _ = self.index.search(hyp_embeddings, k=1)
        corpus_sim = distances[:, 0].clip(0, 1)  # cosine sim to nearest corpus item

        novelty_scores = []
        for i, (hyp, sim) in enumerate(zip(hypotheses, corpus_sim)):
            max_sim = float(sim)
            # Also compare against source abstract if provided
            if source_abstracts:
                abs_emb = self._embed([source_abstracts[i]])
                faiss.normalize_L2(abs_emb)
                hyp_emb = hyp_embeddings[i : i + 1]
                abs_sim = float(np.dot(hyp_emb, abs_emb.T)[0, 0])
                max_sim = max(max_sim, abs_sim)
            novelty_scores.append(max(0.0, 1.0 - max_sim))

        return novelty_scores

    def evaluate_batch(self, results: list[dict]) -> list[dict]:
        """Add novelty scores to hypothesis result dicts."""
        hypotheses = [r["hypothesis"] for r in results]
        abstracts = [r.get("abstract", "") for r in results]

        scores = self.compute_novelty(hypotheses, source_abstracts=abstracts)

        enriched = []
        for r, score in zip(results, scores):
            enriched.append({**r, "novelty_score": round(score, 4)})
        return enriched

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def aggregate_novelty(self, results: list[dict]) -> dict:
        scores = [r["novelty_score"] for r in results if "novelty_score" in r]
        if not scores:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "n": 0}
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "n": len(scores),
        }

    def aggregate_by_domain(self, results: list[dict]) -> dict[str, dict]:
        domains = {}
        for r in results:
            d = r.get("domain", "unknown")
            domains.setdefault(d, []).append(r)
        return {d: self.aggregate_novelty(items) for d, items in domains.items()}
