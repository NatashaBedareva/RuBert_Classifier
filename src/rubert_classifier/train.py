"""Training module."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import f1_score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from rubert_classifier.config import Config
from rubert_classifier.data import load_sib200_dataset
from rubert_classifier.model import create_model
from rubert_classifier.preprocessing import prepare_datasets


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


def build_compute_metrics(id2label: dict):
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


def train_model(
    cfg: Optional[Config] = None,
    dataset: Optional[DatasetDict] = None,
) -> Trainer:
    """Full training pipeline."""
    if cfg is None:
        cfg = Config()

    _seed_everything(cfg.seed)

    if dataset is None:
        dataset = load_sib200_dataset(cfg)
    elif not cfg.list_of_categories:
        categories = sorted(set(dataset["train"]["category"]))
        cfg.list_of_categories = categories
        cfg.num_labels = len(categories)
        cfg.id2label = {i: c for i, c in enumerate(categories)}
        cfg.label2id = {c: i for i, c in enumerate(categories)}

    tokenized_train, tokenized_val, tokenized_test, tokenizer = prepare_datasets(dataset, cfg)

    model = create_model(cfg)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_restarts",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        save_total_limit=cfg.save_total_limit,
        logging_dir=cfg.logs_dir,
        logging_steps=cfg.logging_steps,
        logging_strategy="steps",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        seed=cfg.seed,
        label_smoothing_factor=cfg.label_smoothing_factor,
        max_grad_norm=cfg.max_grad_norm,
        eval_delay=1,
        save_only_model=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, max_length=cfg.max_length, pad_to_multiple_of=8
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_threshold=cfg.early_stopping_threshold,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(cfg.id2label),
        callbacks=[early_stopping, LoggingCallback()],
    )

    print("Starting training...")
    train_results = trainer.train()
    print(f"Training done in {train_results.metrics['train_runtime']:.2f}s")
    print(f"Final training loss: {train_results.training_loss:.4f}")

    # Save best model + config
    Path(cfg.best_model_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.best_model_dir)
    tokenizer.save_pretrained(cfg.best_model_dir)
    cfg.save(str(Path(cfg.best_model_dir) / "config.json"))
    print(f"Best model saved to {cfg.best_model_dir}")

    return trainer
