"""LoRA fine-tuning of LLaMA-3.1-8B for hypothesis generation."""

# Patch transformers.activations before peft/awq import chain
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

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

from src.finetune.dataset import HypothesisDataset, load_finetune_data

console = Console()


@dataclass
class FinetuneConfig:
    # Model
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir: str = "checkpoints/llama3-8b-hypothesis"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    max_seq_length: int = 1024

    # Eval & saving
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    save_total_limit: int = 3

    # Hardware
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

    # W&B
    wandb_project: str = "icml2026-hypothesis"
    wandb_run_name: str = "llama3-8b-lora-v1"
    report_to: str = "wandb"


def load_model_and_tokenizer(cfg: FinetuneConfig):
    console.print(f"[cyan]Loading base model: {cfg.base_model}[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, tokenizer


def apply_lora(model, cfg: FinetuneConfig):
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(cfg: Optional[FinetuneConfig] = None):
    if cfg is None:
        cfg = FinetuneConfig()

    # Setup W&B
    if cfg.report_to == "wandb":
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg),
        )

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # Datasets
    train_path, val_path = load_finetune_data()
    train_dataset = HypothesisDataset(train_path, tokenizer, cfg.max_seq_length, "train")
    val_dataset = HypothesisDataset(val_path, tokenizer, cfg.max_seq_length, "val")
    console.print(f"[green]Train: {len(train_dataset)}, Val: {len(val_dataset)}[/green]")

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        save_total_limit=cfg.save_total_limit,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=cfg.report_to,
        run_name=cfg.wandb_run_name,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    console.print("[bold green]Starting training...[/bold green]")
    trainer.train()

    # Save final adapter
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save config
    with open(output_path / "finetune_config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    console.print(f"[bold green]Model saved to {output_path}[/bold green]")

    if cfg.report_to == "wandb":
        wandb.finish()

    return str(output_path)


if __name__ == "__main__":
    import fire
    fire.Fire(train)
