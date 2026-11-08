"""Training module."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from rubert_classifier.config import Config
from rubert_classifier.data import get_training_datasets
from rubert_classifier.model import create_model


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_compute_metrics(cfg: Config):
    """Build the compute_metrics callable."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        f1_macro = f1_score(labels, predictions, average="macro")
        f1_weighted = f1_score(labels, predictions, average="weighted")
        accuracy = float(np.mean(predictions == labels))

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    return compute_metrics


def _resolve_fp16(value) -> bool:
    """Resolve 'auto' | true | false into a bool."""
    if isinstance(value, str) and value.lower() == "auto":
        return torch.cuda.is_available()
    return bool(value)


def train_model(cfg: Optional[Config] = None) -> Trainer:
    """Full training pipeline."""
    if cfg is None:
        cfg = Config()

    _seed_everything(cfg.seed)

    tokenized_train, tokenized_val, tokenized_test, tokenizer = get_training_datasets(cfg)

    model = create_model(cfg)

    t = cfg.training
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        num_train_epochs=t.num_train_epochs,
        weight_decay=t.weight_decay,
        warmup_steps=t.warmup_ratio,           
        eval_strategy=t.strategy["eval"],
        save_strategy=t.strategy["save"],
        load_best_model_at_end=t.strategy["load_best_model_at_end"],
        metric_for_best_model=t.strategy["metric_for_best_model"],
        greater_is_better=t.strategy["greater_is_better"],
        optim=t.optimizer,
        lr_scheduler_type=t.scheduler,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        save_total_limit=t.save_total_limit,
        logging_steps=t.logging_steps,
        logging_strategy="steps",
        fp16=_resolve_fp16(t.fp16),
        dataloader_num_workers=t.dataloader_num_workers,
        report_to="none",
        seed=cfg.seed,
        label_smoothing_factor=t.label_smoothing_factor,
        max_grad_norm=t.max_grad_norm,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=cfg.preprocessing.max_length,
        pad_to_multiple_of=cfg.preprocessing.tokenization["pad_to_multiple_of"],
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=t.early_stopping["patience"],
        early_stopping_threshold=t.early_stopping["threshold"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(cfg),
        callbacks=[early_stopping, LoggingCallback()],
    )

    print("Starting training...")
    train_results = trainer.train()
    print(f"Training done in {train_results.metrics['train_runtime']:.2f}s")
    print(f"Final training loss: {train_results.training_loss:.4f}")

    Path(cfg.best_model_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.best_model_dir)
    tokenizer.save_pretrained(cfg.best_model_dir)
    print(f"Best model saved to {cfg.best_model_dir}")

    return trainer