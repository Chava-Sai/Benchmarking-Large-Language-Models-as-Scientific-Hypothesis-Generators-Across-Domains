"""Build SFT dataset for LoRA fine-tuning on hypothesis generation."""

import json
from pathlib import Path
from typing import Optional

import yaml
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.models.prompts import HYPOTHESIS_SYSTEM, HYPOTHESIS_FEW_SHOT


FINETUNE_SYSTEM = HYPOTHESIS_SYSTEM

FINETUNE_USER_TEMPLATE = """Given the following research abstract, generate ONE novel scientific hypothesis that extends or builds upon the described work. The hypothesis should be specific, testable, and go beyond what is already stated.

Abstract:
{abstract}

Generate a single hypothesis sentence starting with "We hypothesize that" or "We propose that":"""


class HypothesisDataset(Dataset):
    """SFT dataset: (abstract, hypothesis) pairs formatted as chat turns."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.samples = self._load(data_path)

    def _load(self, path: Path) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        abstract = sample["context"] if "context" in sample else sample.get("full_abstract", sample["abstract"])
        hypothesis = sample["hypothesis"]

        messages = [
            {"role": "system", "content": FINETUNE_SYSTEM},
            {"role": "user", "content": FINETUNE_USER_TEMPLATE.format(abstract=abstract)},
            {"role": "assistant", "content": hypothesis},
        ]

        # Apply chat template with labels
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Mask everything before the assistant's turn in labels
        input_ids = encodings["input_ids"]
        labels = self._mask_prompt_labels(input_ids, messages, abstract, hypothesis)

        return {
            "input_ids": input_ids,
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    def _mask_prompt_labels(
        self, input_ids: list[int], messages: list[dict], abstract: str, hypothesis: str
    ) -> list[int]:
        """Set labels to -100 for prompt tokens; only train on hypothesis."""
        labels = list(input_ids)

        # Build prompt-only text (without assistant response)
        prompt_messages = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, truncation=False)["input_ids"]
        prompt_len = len(prompt_ids)

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return labels


def load_finetune_data(config_path: str = "config/datasets.yaml") -> tuple[Path, Path]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ft_dir = Path(cfg["paths"]["finetune"])
    return ft_dir / "finetune_train.jsonl", ft_dir / "finetune_val.jsonl"
