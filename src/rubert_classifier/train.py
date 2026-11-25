"""Training module with MLflow experiment tracking."""

import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib
import mlflow
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class LoggingCallback(TrainerCallback):
    """Print Trainer log events to the console."""

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
        return {
            "accuracy": float(np.mean(predictions == labels)),
            "f1_macro": float(f1_score(labels, predictions, average="macro")),
            "f1_weighted": float(f1_score(labels, predictions, average="weighted")),
        }

    return compute_metrics


def _resolve_fp16(value) -> bool:
    """Resolve 'auto' | true | false into a bool."""
    if isinstance(value, str) and value.lower() == "auto":
        return torch.cuda.is_available()
    return bool(value)


def _git_commit_id() -> str:
    """Get the checked-out Git commit, if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _log_config(cfg: Config) -> None:
    """Log all configuration fields as an artifact and scalar fields as params."""
    config_dict = asdict(cfg)
    mlflow.log_text(
        json.dumps(config_dict, indent=2, ensure_ascii=False, default=str),
        "configuration/config.json",
    )

    def flatten(value, prefix=""):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                yield from flatten(item, name)
            elif isinstance(item, (str, int, float, bool)) or item is None:
                yield name, str(item)

    # MLflow parameters are immutable within a run. Log once, before Trainer starts.
    mlflow.log_params(dict(flatten(config_dict)))


def _save_training_plots(log_history, plots_dir: Path) -> None:
    """Save metric history and three plots locally and as MLflow artifacts."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_file = plots_dir / "training_metrics.json"
    history_file.write_text(
        json.dumps(log_history, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(history_file), artifact_path="plots")

    plots = (
        ("loss", "train_loss.png", "Training loss"),
        ("eval_loss", "eval_loss.png", "Validation loss"),
        ("eval_f1_macro", "eval_f1_macro.png", "Validation macro-F1"),
    )
    for metric, filename, title in plots:
        points = [
            (entry["step"], entry[metric])
            for entry in log_history
            if metric in entry and "step" in entry
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        if points:
            steps, values = zip(*points)
            ax.plot(steps, values, marker="o", markersize=3)
        else:
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="Training step", ylabel=metric)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(path), artifact_path="plots")


def train_model(cfg: Optional[Config] = None) -> Trainer:
    """Full training pipeline with optional MLflow tracking."""
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
        report_to="none",  # Explicit logging through our callback below
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

    def run_training():
        print("Starting training...")
        results = trainer.train()
        print(f"Training done in {results.metrics['train_runtime']:.2f}s")
        print(f"Final training loss: {results.training_loss:.4f}")
        Path(cfg.best_model_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(cfg.best_model_dir)
        tokenizer.save_pretrained(cfg.best_model_dir)
        print(f"Best model saved to {cfg.best_model_dir}")
        return trainer

    if not cfg.mlflow.enabled:
        return run_training()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.set_tag("git_commit_id", _git_commit_id())
        _log_config(cfg)

        class MLflowMetricsCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                numeric_metrics = {
                    key: float(value)
                    for key, value in logs.items()
                    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
                }
                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics, step=state.global_step)

        trainer.add_callback(MLflowMetricsCallback())
        try:
            return run_training()
        finally:
            _save_training_plots(trainer.state.log_history, Path(cfg.mlflow.plots_dir))
